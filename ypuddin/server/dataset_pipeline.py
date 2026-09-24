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


class DatasetPipeline:
    def __init__(self, context: Any):
        self.c = context
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dataset-pipeline")
        self.stopping = threading.Event()
        self.cancel_events: dict[str, threading.Event] = {}
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
                    "request": {key: value for key, value in op["request"].items() if key != "images"},
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
                prepared and inspection and not inspection["errors"] and plan and plan["ok"]
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
            caption = caption_target(
                path,
                _validate_caption_extension(sources.get(row["id"], row).get("caption_ext", "auto")),
                directory_cache=caption_directories,
            )
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
                    images = self._resolve_images(pid, vid, request["images"])
                    dataset_ids = sorted({record["dataset_id"] for record in images})
                    if action == "preprocess":
                        changes = self._preprocess(oid, work, images, request["preprocess"])
                    elif action == "captions":
                        changes = self._captions(oid, work, images, request["captions"])
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
                if result["inspection"]["errors"]:
                    raise ApiError(
                        "resolve the reported data errors before caching", code="pipeline.quality_errors"
                    )
                cfg = TrainConfig.model_validate(get_project_config(pid, self.c, vid))
                devices = gpu_info()
                self._progress(
                    oid, "planning", 0, 1, "Checking actual bucket layout and training requirements"
                )
                result["plan"] = plan(cfg, device=devices[0]["device"] if devices else "cpu")
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
