"""Version-owned dataset preparation, reversible file edits and real cache jobs.

This is an independent implementation. Operations keep durable file journals and use
existing version leases; a cancelled preparation never publishes half a transformed set.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageOps

from ypuddin.config import TrainConfig
from ypuddin.data.image_metadata import alpha_channel, transparency_source
from ypuddin.data.index import IMAGE_EXTS, content_hash, iter_images, mask_for

from .db import new_id, now
from .errors import ApiError, NotFound

TERMINAL = {"completed", "failed", "cancelled"}
# A head detection keeps its results for review in its operation folder.
PROPOSALS = "proposals.json"


class PipelineCancelled(Exception):
    pass


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _stat(path: Path) -> list[int] | None:
    try:
        st = path.stat()
        return [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino]
    except OSError:
        return None


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def merged_caption(previous: str, tags: str, existing: str, trigger: str = "") -> str:
    """The caption an automatic tagging pass writes: new tags replace, follow or lead the old ones,
    each tag once (case-insensitively, first spelling kept), with the trigger word first."""

    def split(text: str) -> list[str]:
        return [token.strip() for token in text.split(",") if token.strip()]

    old, new = split(previous), split(tags)
    seen = {token.casefold() for token in old}
    fresh = [token for token in new if token.casefold() not in seen]
    tokens = (
        new if existing in {"overwrite", "skip"} else old + fresh if existing == "append" else fresh + old
    )
    if trigger.strip():
        tokens = [
            trigger.strip(),
            *(token for token in tokens if token.casefold() != trigger.strip().casefold()),
        ]
    result, kept = [], set()
    for token in tokens:
        if token.casefold() not in kept:
            kept.add(token.casefold())
            result.append(token)
    return ", ".join(result)


# Every tag group of a structured caption, in the order training reads them.
_CAPTION_GROUPS = ("quality", "count", "character", "series", "artist", "appearance", "tags", "environment")


def _same_caption(path: Path, content: str) -> bool:
    """Whether writing ``content`` would leave the caption's meaning unchanged."""
    try:
        current = path.read_text(encoding="utf-8-sig")
        if path.suffix.lower() == ".json":
            return json.loads(current) == json.loads(content)
    except (OSError, ValueError):
        return False
    return current.strip() == content.strip()


class DatasetPipeline:
    def __init__(self, context: Any):
        self.c = context
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dataset-pipeline")
        self.stopping = threading.Event()
        self.cancel_events: dict[str, threading.Event] = {}
        self.vision = None  # VisionModels, attached by the app
        self.credentials = None  # ModelCredentials, attached by the app
        self.c.db.execute("""CREATE TABLE IF NOT EXISTS dataset_pipeline_operations (
            id TEXT PRIMARY KEY, project_id TEXT NOT NULL, version_id TEXT NOT NULL,
            action TEXT NOT NULL, status TEXT NOT NULL, phase TEXT NOT NULL,
            done INTEGER NOT NULL DEFAULT 0, total INTEGER NOT NULL DEFAULT 0,
            request_json TEXT NOT NULL, result_json TEXT NOT NULL DEFAULT '{}',
            logs_json TEXT NOT NULL DEFAULT '[]', error TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0,
            job_id TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL, finished_at REAL
        )""")
        self._recover()

    def root(self, pid: str, vid: str) -> Path:
        path = self.c.version_dir(pid, vid) / "pipeline"
        if path.is_symlink():
            raise ApiError("pipeline storage cannot be a symbolic link", code="pipeline.path")
        return path

    def _recover(self) -> None:
        for row in self.c.db.fetchall(
            "SELECT * FROM dataset_pipeline_operations WHERE status NOT IN ('completed','failed','cancelled')"
        ):
            if row["job_id"]:
                continue
            try:
                journal = self.root(row["project_id"], row["version_id"]) / row["id"] / "journal.json"
                if journal.exists():
                    changes = json.loads(journal.read_text())["changes"]
                    self._rollback(changes)
                    from .routes_work import _index_dataset

                    for dataset in self.c.db.fetchall(
                        "SELECT id FROM datasets WHERE version_id=?", (row["version_id"],)
                    ):
                        _index_dataset(self.c, dataset["id"])
                message = "Service stopped during preparation. Uncommitted file changes were rolled back; retry the operation."
            except Exception as exc:
                message = f"Recovery requires attention: {exc}. Original backups were preserved."
            self.c.db.update(
                "dataset_pipeline_operations",
                row["id"],
                {
                    "status": "failed",
                    "phase": "failed",
                    "error": message,
                    "finished_at": now(),
                    "updated_at": now(),
                },
            )

    def close(self) -> None:
        self.stopping.set()
        with self.c.db.lock:
            for event in self.cancel_events.values():
                event.set()
        self.executor.shutdown(wait=True)

    def _sources(self, pid: str, vid: str) -> list[dict]:
        from .routes_work import get_project_config

        config = get_project_config(pid, self.c, vid)
        registered = self.c.db.fetchall(
            "SELECT * FROM datasets WHERE version_id=? ORDER BY created_at", (vid,)
        )
        registry = {str(Path(row["path"]).expanduser().resolve()): row for row in registered}
        sources: dict[str, dict] = {}
        for role, items in (
            ("train", config.get("dataset", {}).get("sources", [])),
            ("validation", config.get("validation", {}).get("sources", [])),
        ):
            for item in items:
                path = str(Path(item["path"]).expanduser().resolve())
                if not self.c.is_allowed(Path(path)):
                    raise ApiError(
                        "dataset source is outside the permitted roots", code="pipeline.path", status=403
                    )
                if path not in sources:
                    row = registry.get(path)
                    sources[path] = {
                        **item,
                        "path": path,
                        "dataset_id": row["id"] if row else None,
                        "roles": [],
                    }
                sources[path]["roles"].append(role)
        for path, row in registry.items():
            if path not in sources:
                sources[path] = {**row, "dataset_id": row["id"], "roles": ["registered"]}
        return list(sources.values())

    def signature(self, pid: str, vid: str, *, recipe: bool = False) -> str:
        from .routes_work import get_project_config

        config = get_project_config(pid, self.c, vid)
        # Inspection advice depends on the model and the effective caption
        # transforms, even when image/caption bytes have not changed.
        payload: list[Any] = [{
            "caption_inspection_version": 3,
            "transparency_inspection_version": 4,
            "model_family": config.get("model", {}).get("family"),
            "caption": config.get("dataset", {}).get("caption"),
        }]
        for source in self._sources(pid, vid):
            root = Path(source["path"])
            payload.append(
                {key: source.get(key) for key in (
                    "path", "caption_ext", "class_prompt", "roles", "repeats", "caption", "is_reg"
                )}
            )
            if not root.is_dir():
                payload.append([str(root), None])
                continue
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    payload.append([str(path), _stat(path)])
        if recipe:
            payload.append({key: config.get(key) for key in ("model", "dataset", "validation")})
            for key in ("dit_path", "vae_path", "text_encoder_path", "text_encoder_2_path", "tokenizer_path"):
                value = config.get("model", {}).get(key)
                if value:
                    path = Path(value).expanduser()
                    assets = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
                    payload.extend([str(asset), _stat(asset)] for asset in assets)
        return hashlib.sha256(_dump(payload).encode()).hexdigest()

    def _get(self, oid: str) -> dict:
        row = self.c.db.fetchone("SELECT * FROM dataset_pipeline_operations WHERE id=?", (oid,))
        if not row:
            raise NotFound("dataset operation not found", code="pipeline.not_found")
        self.c.resolve_version(row["project_id"], row["version_id"])
        return row

    def operation(self, oid: str) -> dict:
        row = self._get(oid)
        if row["job_id"]:
            job = self.c.db.fetchone(
                "SELECT * FROM jobs WHERE id=? AND version_id=?", (row["job_id"], row["version_id"])
            )
            if job:
                status = {
                    "completed": "completed",
                    "failed": "failed",
                    "cancelled": "cancelled",
                    "paused": "cancelled",
                    "cancelling": "cancelling",
                }.get(job["status"], "running")
                progress = json.loads(job.get("progress_json") or "{}")
                fields = {
                    "status": status,
                    "phase": "cache" if status not in TERMINAL else status,
                    "done": int(progress.get("cache_done") or 0),
                    "total": int(progress.get("cache_total") or 0),
                    "error": job.get("error"),
                    "finished_at": job.get("finished_at"),
                }
                if any(row.get(key) != value for key, value in fields.items()):
                    self.c.db.update("dataset_pipeline_operations", oid, fields | {"updated_at": now()})
                    row.update(fields)
            elif row["status"] not in TERMINAL:
                self.c.db.update(
                    "dataset_pipeline_operations",
                    oid,
                    {
                        "status": "failed",
                        "error": "The associated cache job no longer exists",
                        "phase": "failed",
                        "finished_at": now(),
                    },
                )
                row = self._get(oid)
        result = json.loads(row["result_json"])
        return {
            **{
                key: value
                for key, value in row.items()
                if key not in {"request_json", "result_json", "logs_json", "cancel_requested"}
            },
            "request": json.loads(row["request_json"]),
            "result": result,
            "logs": json.loads(row["logs_json"]),
            "can_undo": row["status"] == "completed"
            and bool(result.get("changes"))
            and not result.get("undone_by"),
            "can_cancel": row["status"] not in TERMINAL and row["phase"] != "applying",
        }

    def _apply_membership(self, pid: str, vid: str, inspection: dict) -> dict:
        from .routes_dataset_management import included, source_states
        states = source_states(self.c, {"project_id": pid, "version_id": vid})
        for image in inspection["images"]:
            image["training_enabled"] = included(image["path"], states)
        inspection["training_errors"] = sum(
            issue["severity"] == "error"
            for image in inspection["images"]
            if image["training_enabled"] or "validation" in image["roles"]
            for issue in image["issues"]
        ) + sum(issue["severity"] == "error" for issue in inspection["source_issues"])
        return inspection

    def snapshot(self, pid: str, vid: str) -> dict:
        version = self.c.resolve_version(pid, vid)
        operations = [
            self.operation(row["id"])
            for row in self.c.db.fetchall(
                "SELECT id FROM dataset_pipeline_operations WHERE version_id=? ORDER BY created_at DESC LIMIT 60",
                (vid,),
            )
        ]
        signature = self.signature(pid, vid) if version["status"] == "ready" else ""
        recipe = self.signature(pid, vid, recipe=True) if version["status"] == "ready" else ""
        inspection = next(
            (
                op["result"]["inspection"]
                for op in operations
                if op["result"].get("inspection", {}).get("signature") == signature
            ),
            None,
        )
        if inspection:
            self._apply_membership(pid, vid, inspection)
        plan = next(
            (
                op["result"]["plan"]
                for op in operations
                if op["result"].get("recipe_signature") == recipe and op["result"].get("plan")
            ),
            None,
        )
        prepared = next(
            (
                op
                for op in operations
                if op["action"] == "prepare"
                and op["status"] == "completed"
                and op["result"].get("recipe_signature") == recipe
            ),
            None,
        )
        datasets = self.c.db.fetchall(
            "SELECT id,index_status,stats_json FROM datasets WHERE version_id=?", (vid,)
        )
        return {
            "project_id": pid,
            "version_id": vid,
            "signature": signature,
            "inspection": inspection,
            "plan": plan,
            # Poll one current report, not up to sixty complete historical image lists.
            # The operation detail endpoint retains its full request and file journal.
            "operations": [
                {
                    **op,
                    "request": {
                        key: (
                            {k: v for k, v in value.items() if k != "selections"}
                            if key == "automask" and isinstance(value, dict)
                            else value
                        )
                        for key, value in op["request"].items()
                        if key != "images"
                    },
                    "result": {
                        key: value
                        for key, value in op["result"].items()
                        if key not in {"inspection", "plan", "changes"}
                    },
                }
                for op in operations
            ],
            "busy": bool(version["busy"]),
            "archived": bool(version["archived"]),
            "ready_to_train": bool(
                prepared and inspection and not inspection["training_errors"] and plan and plan["ok"]
            ),
            "prepared_job_id": prepared["job_id"] if prepared else None,
            "datasets": [
                {"id": row["id"], "index_status": row["index_status"], "stats": json.loads(row["stats_json"])}
                for row in datasets
            ],
            "stale": bool(not inspection and any(op["result"].get("inspection") for op in operations)),
        }

    def _progress(self, oid: str, phase: str, done: int, total: int, message: str | None = None) -> None:
        fields: dict[str, Any] = {"phase": phase, "done": done, "total": total, "updated_at": now()}
        if message:
            logs = json.loads(self._get(oid)["logs_json"])
            logs.append({"time": now(), "message": message})
            fields["logs_json"] = _dump(logs[-100:])
        self.c.db.update("dataset_pipeline_operations", oid, fields)
        row = self._get(oid)
        self.c.bus.publish(
            "dataset.pipeline",
            {
                "operation_id": oid,
                "project_id": row["project_id"],
                "version_id": row["version_id"],
                "phase": phase,
                "done": done,
                "total": total,
            },
        )

    def _cancelled(self, oid: str) -> None:
        if self.stopping.is_set() or self._get(oid)["cancel_requested"]:
            raise PipelineCancelled("Operation cancelled; original files were preserved")

    def start(self, pid: str, vid: str, request: dict) -> dict:
        # Historic automatic-tagging operations remain readable/undoable, but
        # retry must not resurrect a removed inference capability.
        if request["action"] == "paint":
            raise ApiError(
                "reopen the image editor to retry painting", code="pipeline.paint_retry", status=409
            )
        if request["action"] == "tag":
            raise ApiError(
                "automatic tagging is no longer available; existing captions can be viewed and edited",
                code="pipeline.tagging_removed",
                status=410,
            )
        with self.c.db.lock:
            return self._start_locked(pid, vid, request)

    def _start_locked(self, pid: str, vid: str, request: dict) -> dict:
        lease = self.c.versions.mutation(pid, vid)
        lease.__enter__()
        oid = new_id("dp")
        try:
            if request["action"] == "restore":
                original = self.operation(request.get("restore_operation_id") or "")
                if original["project_id"] != pid or original["version_id"] != vid or not original["can_undo"]:
                    raise ApiError(
                        "this operation cannot be restored in this version",
                        code="pipeline.restore_invalid",
                        status=409,
                    )
            if request["action"] == "automask" and (request.get("automask") or {}).get("proposal_id"):
                self._open_detection(request["automask"]["proposal_id"], pid, vid)
            self.c.db.insert(
                "dataset_pipeline_operations",
                {
                    "id": oid,
                    "project_id": pid,
                    "version_id": vid,
                    "action": request["action"],
                    "status": "queued",
                    "phase": "queued",
                    "request_json": _dump(request),
                    "created_at": now(),
                    "updated_at": now(),
                },
            )
            with self.c.db.lock:
                self.cancel_events[oid] = threading.Event()
            self.executor.submit(self._run, oid, lease)
        except BaseException as exc:
            with self.c.db.lock:
                self.cancel_events.pop(oid, None)
            self.c.db.execute(
                "UPDATE dataset_pipeline_operations SET status='failed',phase='failed',error=?,finished_at=? WHERE id=?",
                (str(exc), now(), oid),
            )
            lease.__exit__(None, None, None)
            raise
        return self.operation(oid)

    def cancel(self, oid: str) -> dict:
        with self.c.db.lock:
            self.operation(oid)  # Synchronize a cache job that may have just completed.
            row = self._get(oid)
            if row["status"] in TERMINAL:
                return self.operation(oid)
            if row["phase"] == "applying":
                raise ApiError(
                    "file changes are being committed; wait and use Undo",
                    code="pipeline.committing",
                    status=409,
                )
            if row["job_id"]:
                self.c.supervisor.request(row["job_id"], "cancel")
            self.c.db.update(
                "dataset_pipeline_operations", oid, {"cancel_requested": 1, "status": "cancelling"}
            )
            if oid in self.cancel_events:
                self.cancel_events[oid].set()
        return self.operation(oid)

    def retry(self, oid: str) -> dict:
        row = self.operation(oid)
        if row["status"] not in {"failed", "cancelled"}:
            raise ApiError(
                "only failed or cancelled operations can be retried",
                code="pipeline.retry_invalid",
                status=409,
            )
        return self.start(row["project_id"], row["version_id"], row["request"])

    def _dataset_refs(self, pid: str, vid: str, dataset_ids: list[str]) -> list[dict]:
        """Every image of these datasets that training uses, as pipeline image references."""
        from .routes_dataset_management import included, source_states
        from .routes_work import _get_dataset, _records

        refs = []
        for did in dict.fromkeys(dataset_ids):
            row = _get_dataset(self.c, did)
            if row["project_id"] != pid or row["version_id"] != vid:
                raise ApiError(
                    "dataset belongs to another project version", code="pipeline.version_mismatch", status=409
                )
            root = Path(row["path"]).expanduser().resolve()
            states = [state for state in source_states(self.c, row) if state[0] == root]
            for record in _records(self.c, did):
                path = Path(record["path"])
                if not states or included(record["path"], states):
                    refs.append({"dataset_id": did, "rel_path": path.resolve().relative_to(root).as_posix()})
        return refs

    def _resolve_images(self, pid: str, vid: str, refs: list[dict]) -> list[dict]:
        from ypuddin.data.index import caption_target

        from .routes_work import _get_dataset, _validate_caption_extension

        resolved, seen = [], set()
        caption_directories = {}
        managed_roots = (self.c.dataset_dir(pid, vid), self.c.reg_dir(pid, vid))
        sources = {source["dataset_id"]: source for source in self._sources(pid, vid)}
        for ref in refs:
            row = _get_dataset(self.c, ref["dataset_id"])
            managed = next(
                (
                    root
                    for root in managed_roots
                    if Path(row["path"]).resolve().is_relative_to(root.resolve())
                ),
                managed_roots[0],
            )
            if row["project_id"] != pid or row["version_id"] != vid:
                raise ApiError(
                    "image belongs to another project version", code="pipeline.version_mismatch", status=409
                )
            relative = PurePosixPath(ref["rel_path"])
            if (
                relative.is_absolute()
                or any(part in {".", ".."} for part in relative.parts)
                or "\\" in ref["rel_path"]
            ):
                raise ApiError("invalid image relative path", code="pipeline.path")
            source = Path(row["path"])
            path = source / Path(*relative.parts)
            if (
                path.is_symlink()
                or source.is_symlink()
                or any(parent.is_symlink() for parent in path.parents if parent.is_relative_to(managed))
                or not path.resolve().is_relative_to(source.resolve())
                or not source.resolve().is_relative_to(managed.resolve())
            ):
                raise ApiError(
                    "copy this legacy/external source into a version before batch processing",
                    code="pipeline.unmanaged",
                    status=409,
                )
            if (
                not path.is_file()
                or path.suffix.lower() not in IMAGE_EXTS
                or path.name.lower().endswith(".mask.png")
            ):
                raise ApiError(f"image is unavailable: {relative}", code="pipeline.image_missing")
            if str(path) in seen:
                continue
            seen.add(str(path))
            caption_ext = _validate_caption_extension(sources.get(row["id"], row).get("caption_ext", "auto"))
            caption = caption_target(path, caption_ext, directory_cache=caption_directories)
            mask = mask_for(path)
            for sidecar in (caption, Path(mask) if mask else None):
                if sidecar and sidecar.is_symlink():
                    raise ApiError("sidecar links cannot be changed", code="pipeline.path")
            resolved.append(
                {
                    "path": path,
                    "caption": caption,
                    "mask": Path(mask) if mask else None,
                    "dataset_id": row["id"],
                    "rel_path": relative.as_posix(),
                    "row": row,
                    "caption_ext": caption_ext,
                }
            )
        if not resolved:
            raise ApiError("select at least one image", code="pipeline.empty_selection")
        return resolved

    def _inspect(self, oid: str, pid: str, vid: str) -> dict:
        from ypuddin.data.anima_caption_inspection import inspect_anima_caption
        from ypuddin.data.caption_formats import (
            effective_caption_extension,
            family_caption_formats,
            require_caption_format,
        )
        from ypuddin.data.caption_json import StructuredCaption
        from ypuddin.data.captions import read_training_caption
        from ypuddin.data.index import caption_target

        from .routes_work import get_project_config

        signature = self.signature(pid, vid)
        config = get_project_config(pid, self.c, vid)
        family_name = config.get("model", {}).get("family", "anima")
        caption_formats = family_caption_formats(family_name)
        caption_profile = "anima" if config.get("model", {}).get("family") == "anima" else None
        caption_directories = {}
        files, global_issues = [], []
        for source in self._sources(pid, vid):
            root = Path(source["path"])
            if not root.is_dir():
                global_issues.append(
                    {
                        "severity": "error",
                        "code": "missing_source",
                        "path": str(root),
                        "message": "Source folder is missing",
                    }
                )
                continue
            files.extend((source, path) for path in iter_images(root))
        records, groups = [], {}
        for index, (source, path) in enumerate(files):
            self._cancelled(oid)
            relative = path.relative_to(source["path"]).as_posix()
            caption = caption_target(
                path, effective_caption_extension(source.get("caption_ext", "auto"), caption_formats),
                directory_cache=caption_directories,
            )
            mask = mask_for(path)
            record = {
                "dataset_id": source["dataset_id"],
                "rel_path": relative,
                "path": str(path),
                "roles": source["roles"],
                "hash": None,
                "width": None,
                "height": None,
                "caption": "",
                "has_mask": bool(mask),
                "has_alpha": None,
                "has_alpha_channel": None,
                "transparency_source": None,
                "image_mode": None,
                "has_transparency": None,
                "issues": [],
                "editable": bool(
                    source["dataset_id"]
                    and any(
                        path.resolve().is_relative_to(root.resolve())
                        for root in (self.c.dataset_dir(pid, vid), self.c.reg_dir(pid, vid))
                    )
                ),
            }
            try:
                digest = _digest(path)
                with Image.open(path) as image:
                    image.load()
                    record["width"], record["height"] = ImageOps.exif_transpose(image).size
                    alpha = alpha_channel(image)
                    record["image_mode"] = image.mode
                    record["transparency_source"] = transparency_source(image)
                    record["has_alpha_channel"] = record["transparency_source"] == "alpha"
                    histogram = alpha.histogram() if alpha is not None else None
                    transparent_pixels = sum(histogram[:255]) if histogram else 0
                    record["has_alpha"] = alpha is not None
                    record["has_transparency"] = transparent_pixels > 0
                    record["transparent_pixels"] = transparent_pixels
                    record["min_alpha"] = alpha.getextrema()[0] if alpha is not None else 255
                record["hash"] = content_hash(path)
                groups.setdefault(digest, []).append(len(records))
                if record["has_transparency"]:
                    record["issues"].append(
                        {
                            "severity": "warning",
                            "code": "transparent_image",
                            "message": "Transparent or semi-transparent pixels are present",
                        }
                    )
                if min(record["width"], record["height"]) < 256:
                    record["issues"].append(
                        {
                            "severity": "warning",
                            "code": "small_image",
                            "message": "Short side is below 256 px",
                        }
                    )
            except Exception as exc:
                record["issues"].append(
                    {"severity": "error", "code": "unreadable_image", "message": str(exc)}
                )
            try:
                require_caption_format(caption, caption_formats, family_name)
                if caption.is_file():
                    raw_caption = read_training_caption(str(caption), None, require_known_format=True)
                    record["caption"] = (
                        raw_caption.text() if isinstance(raw_caption, StructuredCaption) else raw_caption
                    )
                    if caption_profile:
                        # Match expand_items: an explicit source override wins;
                        # regularization otherwise has no stochastic transforms.
                        cap_cfg = source.get("caption")
                        if cap_cfg is None:
                            cap_cfg = {} if source.get("is_reg") else config.get("dataset", {}).get("caption", {})
                        advice = inspect_anima_caption(caption, record["caption"], cap_cfg)
                        record["caption_format"] = advice
                        record["issues"].extend(advice["issues"])
                if not record["caption"] and not source.get("class_prompt"):
                    record["issues"].append(
                        {
                            "severity": "warning",
                            "code": "missing_caption",
                            "message": "Caption is empty or missing",
                        }
                    )
            except (UnicodeError, OSError, ValueError) as exc:
                record["issues"].append(
                    {
                        "severity": "error",
                        "code": getattr(exc, "code", "caption_encoding"),
                        "path": caption.relative_to(source["path"]).as_posix(),
                        "message": str(exc),
                    }
                )
            if mask:
                try:
                    with Image.open(mask) as image:
                        image.load()
                        if ImageOps.exif_transpose(image).size != (record["width"], record["height"]):
                            record["issues"].append(
                                {
                                    "severity": "warning",
                                    "code": "mask_size",
                                    "message": "Mask size differs from the oriented image",
                                }
                            )
                except Exception as exc:
                    record["issues"].append(
                        {"severity": "error", "code": "unreadable_mask", "message": str(exc)}
                    )
            records.append(record)
            self._progress(
                oid,
                "inspecting",
                index + 1,
                len(files),
                relative if index % 25 == 0 or index + 1 == len(files) else None,
            )
        # Keep the copy with the most useful sidecars, then a stable path. Alphabetical
        # discovery alone could retain an uncaptioned download and exclude curated data.
        duplicates = [
            sorted(
                indices,
                key=lambda i: (
                    -int(records[i]["has_mask"]),
                    -int(bool(records[i]["caption"])),
                    records[i]["path"],
                ),
            )
            for indices in groups.values()
            if len(indices) > 1
        ]
        if not records:
            global_issues.append(
                {
                    "severity": "error",
                    "code": "empty_dataset",
                    "message": "Import at least one training image",
                }
            )
        for indices in duplicates:
            for index in indices:
                records[index]["issues"].append(
                    {
                        "severity": "warning",
                        "code": "duplicate",
                        "message": "Identical image bytes occur more than once",
                    }
                )
        issues = global_issues + [issue for record in records for issue in record["issues"]]
        if self.signature(pid, vid) != signature:
            raise ApiError(
                "dataset changed during inspection; retry with stable source files",
                code="pipeline.source_changed",
                status=409,
            )
        return {
            "signature": signature,
            "caption_profile": caption_profile,
            "images": records,
            "duplicate_groups": duplicates,
            "errors": sum(issue["severity"] == "error" for issue in issues),
            "warnings": sum(issue["severity"] == "warning" for issue in issues),
            "captioned": sum(bool(record["caption"]) for record in records),
            "masks": sum(record["has_mask"] for record in records),
            "alpha_images": sum(record["has_alpha"] is True for record in records),
            "alpha_channel_images": sum(record["has_alpha_channel"] is True for record in records),
            "transparency_metadata_images": sum(
                record["has_alpha"] is True and record["has_alpha_channel"] is False for record in records
            ),
            "transparent_images": sum(record["has_transparency"] is True for record in records),
            "source_issues": global_issues,
        }

    def _change(self, work: Path, target: Path, staged: Path | None) -> dict:
        index = len(list((work / "backups").glob("*"))) if (work / "backups").exists() else 0
        backup = work / "backups" / str(index)
        backup.parent.mkdir(parents=True, exist_ok=True)
        exists = target.exists()
        if exists:
            shutil.copy2(target, backup)
            if _digest(target) != _digest(backup):
                raise ApiError(
                    "source changed while creating its backup; retry",
                    code="pipeline.source_changed",
                    status=409,
                )
        return {
            "target": str(target),
            "backup": str(backup) if exists else None,
            "staged": str(staged) if staged else None,
            "before": _digest(target) if exists else None,
            "after": _digest(staged) if staged else None,
        }

    def _rollback(self, changes: list[dict]) -> None:
        for change in reversed(changes):
            target = Path(change["target"])
            actual = _digest(target) if target.exists() else None
            if actual == change["before"]:
                continue
            if actual != change["after"]:
                raise ApiError(
                    f"file changed outside the operation; backup retained: {target}",
                    code="pipeline.restore_conflict",
                    status=409,
                )
            if change["backup"]:
                if _digest(Path(change["backup"])) != change["before"]:
                    raise ApiError(
                        "original backup was modified; restore was stopped",
                        code="pipeline.restore_conflict",
                        status=409,
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(change["backup"], target)
            else:
                target.unlink(missing_ok=True)

    def _commit(self, oid: str, work: Path, changes: list[dict]) -> None:
        self._cancelled(oid)
        with self.c.db.lock:
            self._cancelled(oid)
            self._progress(oid, "applying", 0, len(changes), "Applying reversible changes")
        with (work / "journal.json").open("w", encoding="utf-8") as stream:
            stream.write(_dump({"changes": changes}))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            for index, change in enumerate(changes):
                target = Path(change["target"])
                if (_digest(target) if target.exists() else None) != change["before"]:
                    raise ApiError(
                        f"source changed while preparing: {target}",
                        code="pipeline.source_changed",
                        status=409,
                    )
                if change["staged"]:
                    os.replace(change["staged"], target)
                else:
                    target.unlink()
                self._progress(oid, "applying", index + 1, len(changes))
        except BaseException:
            self._rollback(changes)
            raise

    def _preprocess(self, oid: str, work: Path, images: list[dict], options: dict) -> list[dict]:
        changes = []
        width, height = options["width"], options["height"]
        for index, record in enumerate(images):
            self._cancelled(oid)
            siblings = [
                p
                for p in record["path"].parent.iterdir()
                if p.stem == record["path"].stem and p.suffix.lower() in IMAGE_EXTS and p != record["path"]
            ]
            if siblings and record["mask"]:
                raise ApiError(
                    "images with the same stem share a mask; rename them before preprocessing",
                    code="pipeline.shared_mask",
                )
            with Image.open(record["path"]) as source:
                fmt = source.format
                image = ImageOps.exif_transpose(source).copy()
            box = (0, 0, image.width, image.height)
            if options["mode"] == "crop_rect":
                rect = options["crop"]
                if rect["x"] + rect["width"] > image.width or rect["y"] + rect["height"] > image.height:
                    raise ApiError(
                        f"crop rectangle exceeds source image dimensions ({image.width} × {image.height})",
                        code="pipeline.crop_bounds",
                    )
                box = (rect["x"], rect["y"], rect["x"] + rect["width"], rect["y"] + rect["height"])
            elif options["mode"] == "center_crop":
                ratio = width / height
                crop_w = min(image.width, round(image.height * ratio))
                crop_h = min(image.height, round(image.width / ratio))
                left, top = (image.width - crop_w) // 2, (image.height - crop_h) // 2
                box = (left, top, left + crop_w, top + crop_h)
            cropped = image.crop(box)
            if options["mode"] == "crop_rect":
                size = cropped.size
            elif options["mode"] == "resize":
                scale = min(width / cropped.width, height / cropped.height)
                if not options.get("allow_upscale"):
                    scale = min(scale, 1)
                size = (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale)))
            else:
                scale = (
                    1
                    if options.get("allow_upscale")
                    else min(1, cropped.width / width, cropped.height / height)
                )
                size = (max(1, round(width * scale)), max(1, round(height * scale)))
            staged = work / "staged" / f"image-{index}"
            staged.parent.mkdir(parents=True, exist_ok=True)
            cropped.resize(size, Image.Resampling.LANCZOS).save(staged, format=fmt)
            changes.append(self._change(work, record["path"], staged))
            if record["mask"]:
                with Image.open(record["mask"]) as source:
                    mask = (
                        ImageOps.exif_transpose(source)
                        .convert("L")
                        .resize(image.size, Image.Resampling.NEAREST)
                    )
                staged_mask = staged.with_name(f"mask-{index}")
                mask.crop(box).resize(size, Image.Resampling.NEAREST).save(staged_mask, format="PNG")
                changes.append(self._change(work, record["mask"], staged_mask))
            self._progress(oid, "preprocessing", index + 1, len(images), record["rel_path"])
        return changes

    def _captions(self, oid: str, work: Path, images: list[dict], options: dict) -> list[dict]:
        from ypuddin.data.captions import caption_content, read_editable_caption

        changes = []
        seen = set()
        for index, record in enumerate(images):
            self._cancelled(oid)
            path = record["caption"]
            self._progress(oid, "captions", index + 1, len(images), record["rel_path"])
            if path in seen:
                continue
            seen.add(path)
            previous = read_editable_caption(str(path), None) if path.exists() else ""
            text = (
                options["text"]
                .replace("{filename}", record["path"].stem)
                .replace("{folder}", record["path"].parent.name)
            )
            mode = options["mode"]
            if mode == "fill_missing" and previous:
                continue
            if mode in {"append", "remove"}:
                tags = [tag.strip() for tag in previous.split(",") if tag.strip()]
                wanted = [tag.strip() for tag in text.split(",") if tag.strip()]
                if mode == "append":
                    tags += [tag for tag in wanted if tag not in tags]
                else:
                    tags = [tag for tag in tags if tag not in wanted]
                text = ", ".join(tags)
            if text == previous and path.exists():
                continue
            staged = work / "staged" / f"caption-{index}"
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_text(caption_content(path, text), encoding="utf-8")
            changes.append(self._change(work, path, staged))
            self._progress(oid, "captions", index + 1, len(images), record["rel_path"])
        return changes

    def _vision_provider(self, device: str) -> tuple[str, int]:
        """The CUDA card with the most free memory when ONNX Runtime can use one, else the CPU."""
        from .hardware import gpu_info
        from .vision_models import runtime_status

        if device == "cpu" or "cuda" not in runtime_status()["providers"]:
            return "cpu", 0
        cards = [card for card in gpu_info() if str(card.get("device", "")).startswith("cuda:")]
        if not cards:
            return "cpu", 0
        best = max(cards, key=lambda card: card.get("mem_free_mb") or 0)
        return "cuda", int(str(best["device"]).split(":", 1)[1])

    def _vision_call(self, oid: str, call: Any) -> Any:
        """Run a tagger or detector; its child process stops with RuntimeError when cancelled."""
        try:
            return call()
        except RuntimeError:
            self._cancelled(oid)
            raise

    def _anima(self, pid: str, vid: str) -> bool:
        """Anima writes artist tags with a leading @."""
        from .routes_work import get_project_config

        return get_project_config(pid, self.c, vid).get("model", {}).get("family") == "anima"

    def _tagger_run(self, oid: str, paths: list[str], options: dict, exclude, *, anima: bool):
        """Each image's tags as (tag, category) in caption order, from the chosen tagger."""
        from .model_catalog import LABEL_FILES, VISION_MODELS
        from .vision_models import DEFAULT_CATEGORIES, tag_images

        if self.vision is None:
            raise ApiError("tagging models are unavailable", code="vision.unavailable", status=503)
        entry = VISION_MODELS[options["model"]]
        files = self.vision.files(options["model"])
        provider, device_index = self._vision_provider(options["device"])
        return self._vision_call(
            oid,
            lambda: tag_images(
                paths,
                model_path=files["model.onnx"],
                tags_path=files[LABEL_FILES[entry["labels"]]],
                spec=entry,
                general=options["general_threshold"],
                character=options["character_threshold"],
                categories=tuple(options.get("categories") or DEFAULT_CATEGORIES),
                exclude=tuple(exclude),
                replace_underscore=options.get("replace_underscore", True),
                escape_parentheses=options.get("escape_parentheses", False),
                anima=anima,
                provider=provider,
                device_index=device_index,
                progress=lambda done, total, name: self._progress(oid, "tagging", done, total),
                cancel=self.cancel_events.get(oid),
            ),
        )

    def _autotag(
        self, oid: str, work: Path, images: list[dict], options: dict, *, anima: bool = False
    ) -> list[dict]:
        from ypuddin.data.caption_json import category_tokens, with_trigger
        from ypuddin.data.captions import caption_content, read_editable_caption

        if self.vision is None:
            raise ApiError("tagging models are unavailable", code="vision.unavailable", status=503)
        self.vision.files(options["model"])
        pending, skipped = [], 0
        for record in images:
            previous = (
                read_editable_caption(str(record["caption"]), None) if record["caption"].exists() else ""
            )
            if options["existing"] == "skip" and previous.strip():
                skipped += 1
                continue
            pending.append((record, previous))
        self._progress(
            oid, "tagging", 0, len(pending), f"Tagging {len(pending)} images; {skipped} already captioned"
        )
        if not pending:
            return []
        results = self._tagger_run(
            oid, [str(record["path"]) for record, _ in pending], options, options["exclude_tags"], anima=anima
        )
        captions = [", ".join(tag for tag, _ in tags) for tags in results]
        trigger = (options.get("trigger_word") or "").strip()
        changes = []
        for index, ((record, previous), tags) in enumerate(zip(pending, captions, strict=True)):
            self._cancelled(oid)
            path = record["caption"]
            structured = path.suffix.lower() == ".json"
            # JSON keeps its trigger in meta.trigger, outside the shuffled tag groups; the file's
            # own trigger stays unless a new one replaces it.
            first = (category_tokens(path)["trigger"] if path.exists() else "") if structured else trigger
            text = merged_caption(previous, tags, options["existing"], first)
            content = caption_content(path, text)
            if structured and trigger:
                content = with_trigger(content, trigger, filename=path.name)
            if path.exists() and _same_caption(path, content):
                continue
            staged = work / "staged" / f"caption-{index}"
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_text(content, encoding="utf-8")
            changes.append(self._change(work, path, staged))
        return changes

    def _detect(self, oid: str, images: list[dict], options: dict) -> list[dict]:
        """Head regions of each image, in order; an unreadable image comes back with ``error``."""
        from .vision_models import detect_heads

        if self.vision is None:
            raise ApiError("head detection models are unavailable", code="vision.unavailable", status=503)
        files = self.vision.files(options["model"])
        provider, device_index = self._vision_provider(options["device"])
        self._progress(oid, "detecting", 0, len(images), f"Detecting heads in {len(images)} images")
        return self._vision_call(
            oid,
            lambda: detect_heads(
                [str(record["path"]) for record in images],
                model_path=files["model.onnx"],
                confidence=options["confidence"],
                padding=options["padding"],
                feather=options["feather"],
                provider=provider,
                device_index=device_index,
                progress=lambda done, total, name: self._progress(oid, "detecting", done, total),
                cancel=self.cancel_events.get(oid),
            ),
        )

    def _detectheads(self, oid: str, work: Path, images: list[dict], options: dict) -> dict:
        """Detect heads for review; no data file changes until chosen heads are written."""
        from .routes_work import _records

        results = self._detect(oid, images, options)
        hashes = {}
        for did in {record["dataset_id"] for record in images}:
            for row in _records(self.c, did):
                hashes[str(Path(row["path"]).resolve())] = row.get("content_hash")
        entries = []
        for record, found in zip(images, results, strict=True):
            stat = record["path"].stat()
            width, height = found["size"] or (None, None)
            entries.append(
                {
                    "dataset_id": record["dataset_id"],
                    "rel_path": record["rel_path"],
                    "hash": hashes.get(str(record["path"].resolve())),
                    "width": width,
                    "height": height,
                    "regions": found["regions"],
                    # Writing skips an image whose file changed after it was detected.
                    "signature": [stat.st_mtime_ns, stat.st_size],
                    **({"error": found["error"]} if found.get("error") else {}),
                }
            )
        proposals = {
            "parameters": {key: options[key] for key in ("model", "confidence", "padding", "feather")},
            "images": entries,
        }
        staged = work / "proposals.json.tmp"
        staged.write_text(_dump(proposals), encoding="utf-8")
        os.replace(staged, work / PROPOSALS)
        return {
            "images": len(entries),
            "with_heads": sum(1 for entry in entries if entry["regions"]),
            "heads": sum(len(entry["regions"]) for entry in entries),
            "unreadable": sum(1 for entry in entries if entry.get("error")),
        }

    def _chosen_heads(self, oid: str, images: list[dict], options: dict) -> tuple[list[dict], int]:
        """The reviewed heads of each image as detections; images changed since then are left out."""
        proposal = self._read_proposals(self._get(options["proposal_id"]))
        entries = {(entry["dataset_id"], entry["rel_path"]): entry for entry in proposal["images"]}
        chosen = {
            (pick["dataset_id"], pick["rel_path"]): set(pick["regions"]) for pick in options["selections"]
        }
        results, stale = [], 0
        for record in images:
            key = (record["dataset_id"], record["rel_path"])
            entry = entries.get(key)
            if entry is None or entry.get("error"):
                raise ApiError(
                    f"{record['rel_path']} has no detected heads to write",
                    code="automask.selection",
                    status=422,
                )
            stat = record["path"].stat()
            if [stat.st_mtime_ns, stat.st_size] != entry["signature"]:
                stale += 1
                self._progress(
                    oid, "masking", 0, len(images), f"{record['rel_path']}: changed after detection"
                )
                results.append({"regions": [], "size": None})
                continue
            regions = [region for region in entry["regions"] if region["index"] in chosen[key]]
            if len(regions) != len(chosen[key]):
                raise ApiError(
                    f"{record['rel_path']}: unknown head in the selection",
                    code="automask.selection",
                    status=422,
                )
            results.append({"regions": regions, "size": [entry["width"], entry["height"]]})
        return results, stale

    def _automask(self, oid: str, work: Path, images: list[dict], options: dict) -> tuple[list[dict], dict]:
        import numpy as np

        from .routes_dataset_masks import _load
        from .vision_models import rasterize_regions

        report: dict[str, Any] = {}
        if options.get("proposal_id"):
            results, stale = self._chosen_heads(oid, images, options)
            report = {"proposal_id": options["proposal_id"], "stale_images": stale}
        else:
            results = self._detect(oid, images, options)
        changes = []
        for index, (record, found) in enumerate(zip(images, results, strict=True)):
            self._cancelled(oid)
            if not found["regions"]:
                continue
            target = record["path"].with_name(record["path"].stem + ".mask.png")
            try:
                current, _ = _load(record["path"], target)
            except ApiError as error:
                self._progress(oid, "masking", index + 1, len(images), f"{record['rel_path']}: {error}")
                continue
            if current.size != tuple(found["size"]):
                continue
            base = np.asarray(current, dtype=np.float32)
            # Heads only ever leave training: a region the user already excluded stays excluded.
            combined = np.minimum(base, rasterize_regions(found["regions"], current.size))
            if np.array_equal(combined, base):
                continue
            staged = work / "staged" / f"mask-{index}"
            staged.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.round(combined).astype(np.uint8), "L").save(staged, format="PNG")
            changes.append(self._change(work, target, staged))
        return changes, report

    def _read_proposals(self, row: dict) -> dict:
        path = self.root(row["project_id"], row["version_id"]) / row["id"] / PROPOSALS
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ApiError(
                "the detection results are missing", code="automask.proposals", status=410
            ) from exc

    def _open_detection(self, oid: str, pid: str | None = None, vid: str | None = None) -> dict:
        """A finished detection whose heads have been neither written nor dismissed."""
        row = self._get(oid)
        result = json.loads(row["result_json"])
        if row["action"] != "detectheads" or row["status"] != "completed":
            raise ApiError("that is not a finished head detection", code="automask.proposals", status=409)
        if pid is not None and (row["project_id"] != pid or row["version_id"] != vid):
            raise ApiError(
                "the detection belongs to another project version",
                code="pipeline.version_mismatch",
                status=409,
            )
        if result.get("applied_by") or result.get("dismissed"):
            raise ApiError(
                "these heads were already written or dismissed", code="automask.proposals_closed", status=409
            )
        return row

    def proposals(self, oid: str) -> dict:
        row = self._get(oid)
        if row["action"] != "detectheads" or row["status"] != "completed":
            raise ApiError("that is not a finished head detection", code="automask.proposals", status=409)
        result = json.loads(row["result_json"])
        proposal = self._read_proposals(row)
        return {
            "operation_id": oid,
            "parameters": proposal["parameters"],
            "images": [{k: v for k, v in entry.items() if k != "signature"} for entry in proposal["images"]],
            "applied_by": result.get("applied_by"),
            "dismissed": bool(result.get("dismissed")),
        }

    def dismiss(self, oid: str) -> dict:
        with self.c.db.lock:
            row = self._open_detection(oid)
            result = json.loads(row["result_json"]) | {"dismissed": True}
            self.c.db.update(
                "dataset_pipeline_operations", oid, {"result_json": _dump(result), "updated_at": now()}
            )
        self.c.bus.publish(
            "dataset.pipeline",
            {"operation_id": oid, "project_id": row["project_id"], "version_id": row["version_id"]},
        )
        return self.operation(oid)

    def _vlmtag(
        self,
        oid: str,
        work: Path,
        images: list[dict],
        options: dict,
        tagging: dict | None = None,
        *,
        anima: bool = False,
    ) -> tuple[list[dict], dict]:
        """Caption images with a vision model; stopping keeps every image already answered.

        With ``tagging`` (assisted tagging) each request carries reference tags: the image's own
        caption when refining, otherwise tagger output.
        """
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        from ypuddin.data.caption_json import category_tokens, unique
        from ypuddin.data.captions import read_editable_caption

        from . import vlm
        from .network import ProxyPolicy

        provider = options["provider"]
        url = vlm.base_url(provider, options.get("base_url"))
        key = self.credentials.vlm_key(provider) if self.credentials is not None else None
        if not vlm.PROVIDERS[provider]["editable"] and not key:
            raise ApiError("save an API key for this service first", code="vlm.key_missing", status=409)
        output, existing = options["output"], options["existing"]
        structured = output in {"categories", "sort"}
        trigger = (options.get("trigger_word") or "").strip()
        jobs, skipped, txt_only = [], 0, set()
        failures: list[dict] = []
        for record in images:
            source = record["caption"] if record["caption"].exists() else None
            target = record["caption"]
            if structured and target.suffix.lower() != ".json":
                if record["caption_ext"] != "auto":
                    txt_only.add(Path(record["row"]["path"]).name)
                    continue
                # Auto captions prefer JSON, so the new file takes over and the TXT stays as it was.
                target = record["path"].with_suffix(".json")
            as_json = target.suffix.lower() == ".json"
            try:
                fields = category_tokens(source) if source and source.suffix == ".json" else None
                text = read_editable_caption(source) if source else ""
            except ValueError as error:
                failures.append({"rel_path": record["rel_path"], "error": str(error)})
                continue
            if fields is not None:
                flat = list(unique(tag for group in _CAPTION_GROUPS for tag in fields[group]))
                grouped = list(
                    unique((*fields["count"], *fields["appearance"], *fields["tags"], *fields["environment"]))
                )
                done = bool(fields["nl"]) if output == "description" else bool(flat)
            else:
                flat = grouped = [tag.strip() for tag in text.split(",") if tag.strip()]
                done = bool(text.strip())
            if existing == "skip" and done:
                skipped += 1
                continue
            reference = (
                (grouped if structured else flat) if tagging is not None and existing == "refine" else []
            )
            # The trigger word is written by the app, not judged by the model.
            reference = [tag for tag in reference if tag.casefold() != trigger.casefold()]
            jobs.append(
                {
                    "record": record,
                    "target": target,
                    "json": as_json,
                    "fields": fields,
                    "reference": reference,
                    "tagged": tagging is not None and not reference,
                    "characters": [],
                }
            )
        if txt_only:
            raise ApiError(
                f"grouped captions need JSON; set the caption format of {', '.join(sorted(txt_only))} "
                "to Auto or JSON",
                code="vlm.json_required",
                status=409,
            )
        tagged = [job for job in jobs if job["tagged"]]
        if tagged:
            if self.vision is None:
                raise ApiError("tagging models are unavailable", code="vision.unavailable", status=503)
            self.vision.files(tagging["model"])
            self._progress(oid, "tagging", 0, len(tagged), f"Tagging {len(tagged)} images for reference")
            results = self._tagger_run(
                oid, [str(job["record"]["path"]) for job in tagged], tagging, options["exclude_tags"], anima=anima
            )
            for job, pairs in zip(tagged, results, strict=True):
                tags = [tag for tag, _ in pairs]
                characters = [tag for tag, category in pairs if category == "character"]
                # Grouped outputs keep character names in their own field; the model sorts the rest.
                job["characters"] = characters if structured else []
                job["reference"] = [tag for tag in tags if tag not in characters] if structured else tags
        total = len(jobs)
        self._progress(
            oid, "requesting", 0, total, f"Sending {total} images to {options['model']}; {skipped} skipped"
        )
        report = {"skipped": skipped, "stopped": False}
        if not jobs:
            return [], report | {"failed_files": len(failures), "failures": failures[:100]}
        policy = ProxyPolicy.from_context(self.c)
        pacer = vlm.Pacer(options["interval"])
        cancel = self.cancel_events.get(oid) or threading.Event()
        halt = threading.Event()

        def prompt_for(job: dict) -> str:
            text = options["prompt"]
            if tagging is not None and "{tags}" not in text:
                text = text.rstrip() + "\n\nExisting tags: {tags}"
            if trigger:
                text = text.replace("{trigger}", trigger)
            else:
                # Without a trigger word, drop the instruction that would name one.
                text = "\n".join(line for line in text.splitlines() if "{trigger}" not in line)
            return text.replace("{tags}", ", ".join(job["reference"]))

        def request(job: dict) -> str:
            return vlm.caption(
                job["record"]["path"],
                prompt_for(job),
                provider=provider,
                url=url,
                key=key,
                model=options["model"],
                policy=policy,
                temperature=options["temperature"],
                max_tokens=options.get("max_tokens"),
                image_size=options["image_size"],
                detail=options["image_detail"],
                timeout=options["timeout"],
                retries=options["retries"],
                pacer=pacer,
                cancel=halt,
            )

        replies: dict[int, str] = {}
        fatal: ApiError | None = None
        executor = ThreadPoolExecutor(max_workers=options["concurrency"], thread_name_prefix="vlm")
        try:
            futures = {executor.submit(request, job): index for index, job in enumerate(jobs)}
            remaining, answered = set(futures), 0
            while remaining:
                finished, remaining = wait(remaining, timeout=0.5, return_when=FIRST_COMPLETED)
                if (cancel.is_set() or self.stopping.is_set()) and not halt.is_set():
                    halt.set()
                    report["stopped"] = True
                    self._progress(oid, "requesting", answered, total, "Stopping; finished images are kept")
                for future in finished:
                    job = jobs[futures[future]]
                    answered += 1
                    try:
                        replies[futures[future]] = future.result()
                        message = None
                    except vlm.VlmStopped:
                        continue
                    except vlm.VlmFatal as error:
                        fatal = fatal or error
                        halt.set()
                        continue
                    except (vlm.VlmError, OSError, ValueError) as error:
                        failures.append({"rel_path": job["record"]["rel_path"], "error": str(error)})
                        message = f"{job['record']['rel_path']}: {error}" if len(failures) <= 20 else None
                    self._progress(oid, "requesting", answered, total, message)
                    if not replies and len(failures) >= min(3, total) and not halt.is_set():
                        # The first answers all failed: the service or model is not usable as set up.
                        fatal = ApiError(
                            f"the first {len(failures)} requests failed: {failures[0]['error']}",
                            code="vlm.failed",
                            status=502,
                        )
                        halt.set()
                if halt.is_set():
                    for future in remaining:
                        future.cancel()
                    remaining = {future for future in remaining if not future.cancelled()}
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        if self.stopping.is_set():
            raise PipelineCancelled("Service is stopping; original files were preserved")
        if fatal is not None and not replies:
            raise fatal
        if not replies and failures and not report["stopped"]:
            raise ApiError(f"no image was captioned: {failures[0]['error']}", code="vlm.failed", status=502)
        if report["stopped"]:
            # Stopping ends the requests; what already came back is still written.
            self.c.db.update("dataset_pipeline_operations", oid, {"cancel_requested": 0, "status": "running"})
            cancel.clear()
        if fatal is not None:
            report["stopped_reason"] = str(fatal)
        changes = []
        for index, job in enumerate(jobs):
            if index not in replies:
                continue
            try:
                content = self._vlm_content(job, replies[index], options, trigger)
            except (vlm.VlmError, ValueError) as error:
                failures.append({"rel_path": job["record"]["rel_path"], "error": str(error)})
                continue
            if job["target"].exists() and _same_caption(job["target"], content):
                continue
            staged = work / "staged" / f"vlm-{index}"
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_text(content, encoding="utf-8")
            changes.append(self._change(work, job["target"], staged))
        return changes, report | {"failed_files": len(failures), "failures": failures[:100]}

    @staticmethod
    def _vlm_content(job: dict, reply: str, options: dict, trigger: str) -> str:
        """The caption file a reply becomes, in the target's own format."""
        from ypuddin.data.caption_json import categorized_content, unique, with_trigger
        from ypuddin.data.captions import caption_content

        from . import vlm

        vlm.reject_refusal(reply)
        blocked = {vlm.readable(tag).casefold() for tag in options["exclude_tags"] if tag.strip()}

        def allowed(tags: list[str]) -> list[str]:
            return [
                tag
                for tag in unique(tags)
                if tag.casefold() not in blocked and tag.casefold() != trigger.casefold()
            ]

        target, as_json, output = job["target"], job["json"], options["output"]
        if output == "tags":
            tags = allowed(vlm.tag_list(reply))
            if not tags:
                raise vlm.VlmError("the reply held no tags")
            if not as_json:
                return caption_content(target, ", ".join(([trigger] if trigger else []) + tags))
            # The file's trigger, character, series, artist and quality stay whatever the model returned.
            fields = job["fields"] or {}
            kept = [fields["trigger"]] if fields.get("trigger") else []
            kept += [
                tag for group in ("quality", "character", "series", "artist") for tag in fields.get(group, ())
            ]
            content = caption_content(target, ", ".join(unique((*kept, *tags))))
            return with_trigger(content, trigger, filename=target.name) if trigger else content
        if output == "description":
            prose = vlm.description(reply)
            if not prose:
                raise vlm.VlmError("the reply held no description")
            if as_json:
                content = caption_content(target, None, description=prose)
                return with_trigger(content, trigger, filename=target.name) if trigger else content
            if trigger and not prose.casefold().startswith(trigger.casefold()):
                prose = f"{trigger}, {prose}"
            return caption_content(target, prose)
        groups = vlm.markers(reply)
        tag_groups = {key: allowed(values) for key, values in groups.items() if key in vlm.MARKERS}
        if not tag_groups and "nl" not in groups:
            raise vlm.VlmError("the reply did not use the COUNT / APPEARANCE / TAGS / ENVIRONMENT / NL lines")
        if output == "sort":
            # Only the grouping comes from the model: every reference tag stays, nothing new is added.
            placed: dict[str, str] = {}
            for key in vlm.MARKERS:
                for tag in tag_groups.get(key, ()):
                    placed.setdefault(tag.casefold(), key)
            reference = allowed(job["reference"])
            tag_groups = {
                key: [tag for tag in reference if placed.get(tag.casefold(), "tags") == key]
                for key in vlm.MARKERS
            }
        return categorized_content(
            target, tag_groups, nl=groups.get("nl"), trigger=trigger, character=job["characters"] or None
        )

    def _run(self, oid: str, lease: Any) -> None:
        from ypuddin.train.plan import plan

        from .hardware import gpu_info
        from .routes_work import JobBody, _index_dataset, create_job, get_project_config

        row = self._get(oid)
        pid, vid = row["project_id"], row["version_id"]
        request, result = json.loads(row["request_json"]), {}
        released = False
        changes, dataset_ids = [], []
        self.c.db.update("dataset_pipeline_operations", oid, {"status": "running"})
        try:
            work = self.root(pid, vid) / oid
            work.mkdir(parents=True, exist_ok=False)
            action = request["action"]
            if action in {"inspect", "prepare"}:
                result["inspection"] = self._inspect(oid, pid, vid)
            elif action == "detectheads":
                refs = request["images"] or self._dataset_refs(pid, vid, request.get("dataset_ids") or [])
                images = self._resolve_images(pid, vid, refs)
                result.update(self._detectheads(oid, work, images, request["automask"]))
                self._cancelled(oid)
            else:
                if action == "restore":
                    original = self.operation(request["restore_operation_id"])
                    if original["action"] == "paint":
                        from .routes_dataset_paint import _restore_guards

                        _restore_guards(self, original)
                    changes = []
                    for index, change in enumerate(original["result"]["changes"]):
                        self._cancelled(oid)
                        target = Path(change["target"])
                        if (_digest(target) if target.exists() else None) != change["after"]:
                            raise ApiError(
                                f"file was edited after that operation: {target}",
                                code="pipeline.restore_conflict",
                                status=409,
                            )
                        staged = None
                        if change["backup"]:
                            if _digest(Path(change["backup"])) != change["before"]:
                                raise ApiError(
                                    "original backup was modified; restore was stopped",
                                    code="pipeline.restore_conflict",
                                    status=409,
                                )
                            staged = work / "staged" / str(index)
                            staged.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(change["backup"], staged)
                            if _digest(staged) != change["before"]:
                                raise ApiError(
                                    "original backup changed while restoring; no files were replaced",
                                    code="pipeline.restore_conflict",
                                    status=409,
                                )
                        changes.append(self._change(work, target, staged))
                    dataset_ids = original["result"].get("dataset_ids", [])
                else:
                    refs = request["images"] or self._dataset_refs(pid, vid, request.get("dataset_ids") or [])
                    if action == "automask" and request["automask"].get("proposal_id"):
                        refs = [
                            {"dataset_id": pick["dataset_id"], "rel_path": pick["rel_path"]}
                            for pick in request["automask"]["selections"]
                        ]
                    images = self._resolve_images(pid, vid, refs)
                    dataset_ids = sorted({record["dataset_id"] for record in images})
                    if action == "preprocess":
                        changes = self._preprocess(oid, work, images, request["preprocess"])
                    elif action == "captions":
                        changes = self._captions(oid, work, images, request["captions"])
                    elif action == "autotag":
                        changes = self._autotag(
                            oid, work, images, request["tagging"], anima=self._anima(pid, vid)
                        )
                    elif action == "automask":
                        changes, report = self._automask(oid, work, images, request["automask"])
                        result.update(report)
                    elif action in {"vlmtag", "assisttag"}:
                        changes, report = self._vlmtag(
                            oid,
                            work,
                            images,
                            request["vlm"],
                            request["tagging"] if action == "assisttag" else None,
                            anima=self._anima(pid, vid),
                        )
                        result.update(report)
                    else:
                        changes = []
                        seen = set()
                        selected = {record["path"] for record in images}
                        for index, record in enumerate(images):
                            self._cancelled(oid)
                            shared = any(
                                p.stem == record["path"].stem
                                and p.suffix.lower() in IMAGE_EXTS
                                and p not in selected
                                for p in record["path"].parent.iterdir()
                            )
                            for target in (record["path"], record["caption"], record["mask"]):
                                if shared and target != record["path"]:
                                    continue
                                if target and target.exists() and str(target) not in seen:
                                    changes.append(self._change(work, target, None))
                                    seen.add(str(target))
                            self._progress(oid, "excluding", index + 1, len(images), record["rel_path"])
                self._commit(oid, work, changes)
                result.update(
                    changes=changes if action != "restore" else [],
                    dataset_ids=dataset_ids,
                    changed_files=len(changes),
                )
                for did in dataset_ids:
                    _index_dataset(self.c, did)
                result["inspection"] = self._inspect(oid, pid, vid)
                self._cancelled(oid)
            if request["action"] == "prepare":
                self._apply_membership(pid, vid, result["inspection"])
                if result["inspection"]["training_errors"]:
                    raise ApiError(
                        "resolve the reported data errors before caching", code="pipeline.quality_errors"
                    )
                cfg = TrainConfig.model_validate(get_project_config(pid, self.c, vid))
                devices = gpu_info()
                self._progress(
                    oid, "planning", 0, 1, "Checking actual bucket layout and training requirements"
                )
                result["plan"] = plan(cfg, device=devices[0]["device"] if devices else "cpu",
                                      index_db_path=self.c.service_cache_dir("index") / "index.sqlite")
                result["recipe_signature"] = self.signature(pid, vid, recipe=True)
                if not result["plan"]["ok"]:
                    raise ApiError(
                        "training preparation check failed; review the reported requirements",
                        code="pipeline.plan_failed",
                    )
                self._cancelled(oid)
                with self.c.db.lock:
                    lease.__exit__(None, None, None)
                    released = True
                    job = create_job(
                        JobBody(
                            type="cache",
                            name=f"Prepare data · {self.c.resolve_version(pid, vid)['name']}",
                            project_id=pid,
                            version_id=vid,
                            config=cfg.to_dict(),
                        ),
                        self.c,
                    )
                    self.c.db.update(
                        "dataset_pipeline_operations",
                        oid,
                        {
                            "job_id": job["id"],
                            "phase": "cache",
                            "result_json": _dump(result),
                            "done": 0,
                            "total": 0,
                        },
                    )
                    if self._get(oid)["cancel_requested"] or self.stopping.is_set():
                        self.c.supervisor.request(job["id"], "cancel")
                return
            with self.c.db.lock:
                # Commit Undo's marker with its completion. If the service stops
                # first, recovery rolls files back and the original stays undoable.
                self.c.db.execute("BEGIN IMMEDIATE")
                try:
                    if action == "restore":
                        self.c.db.update(
                            "dataset_pipeline_operations",
                            original["id"],
                            {"result_json": _dump(original["result"] | {"undone_by": oid})},
                        )
                    if action == "automask" and request["automask"].get("proposal_id"):
                        # The reviewed heads are written; that detection is closed.
                        detection = self._get(request["automask"]["proposal_id"])
                        self.c.db.update(
                            "dataset_pipeline_operations",
                            detection["id"],
                            {
                                "result_json": _dump(
                                    json.loads(detection["result_json"]) | {"applied_by": oid}
                                )
                            },
                        )
                    self.c.db.update(
                        "dataset_pipeline_operations",
                        oid,
                        {
                            "status": "completed",
                            "phase": "completed",
                            "result_json": _dump(result),
                            "finished_at": now(),
                            "updated_at": now(),
                        },
                    )
                    self.c.db.execute("COMMIT")
                except BaseException:
                    self.c.db.execute("ROLLBACK")
                    raise
        except Exception as exc:
            status = "cancelled" if isinstance(exc, PipelineCancelled) else "failed"
            error = str(exc)
            if changes:
                try:
                    self._rollback(changes)
                    for did in dataset_ids:
                        _index_dataset(self.c, did)
                    result.pop("inspection", None)
                    result.update(changes=[], rolled_back=True)
                except Exception as rollback_error:
                    status = "failed"
                    error += (
                        f". Recovery requires attention: {rollback_error}. Original backups were preserved."
                    )
                    result.update(changes=changes, recovery_needed=True)
            self.c.db.update(
                "dataset_pipeline_operations",
                oid,
                {
                    "status": status,
                    "phase": status,
                    "error": error,
                    "result_json": _dump(result),
                    "finished_at": now(),
                },
            )
        finally:
            if not released:
                lease.__exit__(None, None, None)
            with self.c.db.lock:
                self.cancel_events.pop(oid, None)
            self.c.bus.publish(
                "dataset.pipeline", {"operation_id": oid, "project_id": pid, "version_id": vid}
            )
            self.c.bus.publish("dataset.changed", {"project_id": pid, "version_id": vid})
