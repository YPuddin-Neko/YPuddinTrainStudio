"""Project version snapshots and managed data copies, without shared mutable files."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .db import new_id, now
from .errors import ApiError, NotFound
from .family_config import change_config_family, initial_family_config, version_family

ACTIVE_JOBS = "('queued','scheduled','running','pausing','cancelling')"


def dataset_directory_name(dataset_id: str, source: Path) -> str:
    label = re.sub(r"^(?:d_[0-9a-f]+-)+", "", source.name).strip()[:60] or "dataset"
    return f"{dataset_id}-{label}"


def assert_version_writable(c: Any, pid: str, vid: str | None, *, data: bool = False) -> dict:
    row = c.resolve_version(pid, vid)
    if row["status"] != "ready" or row["busy"] or row["archived"]:
        raise ApiError(
            "version is archived or busy; wait for its operation to finish", code="version.busy", status=409
        )
    if data and c.db.fetchone(
        f"SELECT id FROM jobs WHERE version_id=? AND status IN {ACTIVE_JOBS}", (row["id"],)
    ):
        raise ApiError("version data is used by queued or running jobs", code="version.jobs_busy", status=409)
    if data and c.db.fetchone(
        "SELECT id FROM datasets WHERE version_id=? AND index_status='indexing'", (row["id"],)
    ):
        raise ApiError(
            "wait for dataset indexing before editing version data", code="version.indexing", status=409
        )
    return row


def version_row(c: Any, row: dict) -> dict:
    pid, vid = row["project_id"], row["id"]
    datasets = c.db.fetchall(
        "SELECT id,stats_json FROM datasets WHERE version_id=? ORDER BY created_at", (vid,)
    )
    data_root = c.project_dir(pid) if row["legacy_layout"] else c.version_dir(pid, vid)
    return {
        **{k: v for k, v in row.items() if k not in {"progress_json", "legacy_layout", "busy"}},
        "archived": bool(row["archived"]),
        "busy": bool(row["busy"]),
        "family": version_family(c, row),
        "progress": json.loads(row["progress_json"] or "{}"),
        "dataset_ids": [r["id"] for r in datasets],
        "stats": {
            "datasets": len(datasets),
            "images": sum(json.loads(r["stats_json"] or "{}").get("images", 0) for r in datasets),
            "jobs": c.db.fetchone("SELECT count(*) n FROM jobs WHERE version_id=?", (vid,))["n"],
            "artifacts": c.db.fetchone("SELECT count(*) n FROM artifacts WHERE version_id=?", (vid,))["n"],
        },
        "paths": {
            "root": str(data_root),
            "config": str(c.config_path(pid, vid)),
            "datasets": str(data_root / "datasets") if row["legacy_layout"] else str(c.dataset_dir(pid, vid)),
            "runs": str(c.runs_dir(pid, vid)),
            "cache": str(c.cache_dir(pid, vid)),
            "traindata": str(data_root / "datasets")
            if row["legacy_layout"]
            else str(c.dataset_dir(pid, vid)),
            "reg": str(c.reg_dir(pid, vid)),
            "samples": str(c.samples_dir(pid, vid)),
            "output": str(c.runs_dir(pid, vid)),
        },
    }


def file_manifest(source: Path) -> list[tuple[Path, int, int]]:
    """Reject links/special files and record enough identity to detect ordinary concurrent edits."""
    if not source.is_dir() or source.is_symlink():
        raise ApiError(f"dataset directory unavailable or symlinked: {source}", code="version.source_invalid")
    result = []
    for root, directories, files in os.walk(source, followlinks=False):
        for name in directories + files:
            path = Path(root) / name
            if path.is_symlink():
                raise ApiError(
                    f"snapshot source contains a symbolic link: {path}", code="version.source_link"
                )
        for name in sorted(files):
            path = Path(root) / name
            if not path.is_file():
                raise ApiError(
                    f"snapshot source contains a special file: {path}", code="version.source_invalid"
                )
            st = path.stat()
            result.append((path.relative_to(source), st.st_size, st.st_mtime_ns))
    return sorted(result)


def copy_source(source: Path, target: Path, manifest: list, progress: Any = None) -> None:
    target.mkdir(parents=True, exist_ok=False)
    for relative, size, mtime in manifest:
        original = source / relative
        before = original.stat()
        if before.st_size != size or before.st_mtime_ns != mtime or original.is_symlink():
            raise ApiError(f"source changed during snapshot: {original}", code="version.source_changed")
        dest = target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, dest)  # Deliberately never use hard links.
        after = original.stat()
        if after.st_size != size or after.st_mtime_ns != mtime:
            raise ApiError(f"source changed during snapshot: {original}", code="version.source_changed")
        if progress:
            progress(size)
    if file_manifest(source) != manifest:
        raise ApiError(f"source changed during snapshot: {source}", code="version.source_changed")


class VersionManager:
    def __init__(self, c: Any):
        self.c = c
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="version-copy")
        # A prior process cannot continue its operation. Keep failure visible and old data intact.
        for row in c.db.fetchall("SELECT id,progress_json FROM project_versions WHERE status='copying'"):
            progress = json.loads(row["progress_json"] or "{}") | {"phase": "failed"}
            c.db.update(
                "project_versions",
                row["id"],
                {
                    "status": "failed",
                    "busy": None,
                    "progress_json": json.dumps(progress),
                    "error": "Service stopped during version copy; create a new version to retry",
                },
            )
        c.db.execute("UPDATE project_versions SET busy=NULL WHERE busy IS NOT NULL")
        for row in c.db.fetchall("SELECT id,stats_json FROM datasets WHERE index_status='indexing'"):
            stats = json.loads(row["stats_json"] or "{}") | {
                "error": "Service stopped while indexing; rescan this dataset"
            }
            c.db.update("datasets", row["id"], {"index_status": "failed", "stats_json": json.dumps(stats)})

    def close(self) -> None:
        self.executor.shutdown(wait=True)

    @contextmanager
    def mutation(self, pid: str, vid: str | None, *, data: bool = True) -> Iterator[dict]:
        c = self.c
        token = new_id("op")
        with c.db.lock:
            row = assert_version_writable(c, pid, vid, data=data)
            c.db.update("project_versions", row["id"], {"busy": token})
        try:
            yield row
        finally:
            c.db.execute("UPDATE project_versions SET busy=NULL WHERE id=? AND busy=?", (row["id"], token))

    def create(
        self,
        pid: str,
        name: str,
        note: str,
        source_id: str | None,
        data_mode: str,
        copy_config: bool = True,
        *,
        family: str | None = None,
    ) -> dict:
        from .routes_work import get_project_config

        c = self.c
        with c.db.lock:
            source = assert_version_writable(c, pid, source_id)
            if c.db.fetchone("SELECT id FROM project_versions WHERE project_id=? AND name=?", (pid, name)):
                raise ApiError(
                    "a version with this name already exists", code="version.duplicate", status=409
                )
            config = get_project_config(pid, c, source["id"])
            if not copy_config:
                from ypuddin.config import TrainConfig

                from .environment import environment_attention_default

                config = TrainConfig().to_dict()
                config["model"]["attention"] = environment_attention_default(c)
                if data_mode != "empty":
                    raise ApiError("copy_config=false requires data_mode=empty", code="version.invalid")
            if family is not None:
                config = (
                    change_config_family(c, config, family)
                    if copy_config
                    else initial_family_config(c, family)
                )
            datasets = c.db.fetchall("SELECT * FROM datasets WHERE version_id=?", (source["id"],))
            if any(row["index_status"] == "indexing" for row in datasets):
                raise ApiError(
                    "wait for dataset indexing before creating a snapshot", code="version.busy", status=409
                )
            vid = new_id("v")
            number = c.db.fetchone(
                "SELECT coalesce(max(number),0)+1 n FROM project_versions WHERE project_id=?", (pid,)
            )["n"]
            t = now()
            c.db.insert(
                "project_versions",
                {
                    "id": vid,
                    "number": number,
                    "project_id": pid,
                    "name": name,
                    "note": note,
                    "parent_version_id": source["id"],
                    "created_at": t,
                    "updated_at": t,
                    "status": "copying",
                    "progress_json": json.dumps(
                        {
                            "phase": "planning",
                            "files_done": 0,
                            "files_total": 0,
                            "bytes_done": 0,
                            "bytes_total": 0,
                        }
                    ),
                },
            )
            c.db.update("project_versions", source["id"], {"busy": vid})
            self.executor.submit(self._copy, pid, vid, source["id"], config, datasets, data_mode)
            return version_row(c, c.resolve_version(pid, vid))

    def _copy(self, pid: str, vid: str, source_id: str, config: dict, datasets: list, mode: str) -> None:
        from .routes_work import _index_dataset, _records_path

        c = self.c
        staging = None
        final = None
        registered = []
        promoted = False
        progress = {"phase": "planning", "files_done": 0, "files_total": 0, "bytes_done": 0, "bytes_total": 0}

        def update() -> None:
            c.db.update("project_versions", vid, {"progress_json": json.dumps(progress), "updated_at": now()})
            c.bus.publish(
                "version.changed", {"project_id": pid, "version_id": vid, "progress": dict(progress)}
            )

        def copied(size: int) -> None:
            progress["files_done"] += 1
            progress["bytes_done"] += size
            update()

        try:
            final = c.version_dir(pid, vid)  # Validate inside the failure/cleanup boundary.
            final.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=f".copy-{vid}-", dir=final.parent))
            sources = config.setdefault("dataset", {}).setdefault("sources", [])
            validation = config.setdefault("validation", {}).setdefault("sources", [])
            entries: dict[str, dict] = {}
            origins = {
                str(Path(row["path"]).expanduser().resolve()): row.get("origin_path") for row in datasets
            }
            if mode == "copy":
                for item in [*sources, *validation, *datasets]:
                    path = Path(item["path"]).expanduser().resolve()
                    if final.resolve().is_relative_to(path):
                        raise ApiError(
                            "dataset source cannot contain its snapshot destination",
                            code="version.source_recursive",
                        )
                    if not c.is_allowed(path):
                        raise ApiError(
                            f"source is outside allowed roots: {path}", code="fs.forbidden", status=403
                        )
                    key = str(path)
                    if key not in entries:
                        did = new_id("d")
                        entries[key] = {
                            "id": did,
                            "path": path,
                            "source": item,
                            "origin_path": origins.get(key) or key,
                            "manifest": file_manifest(path),
                            "relative": c.dataset_dir(pid, vid, is_reg=bool(item.get("is_reg"))).relative_to(
                                final
                            )
                            / dataset_directory_name(did, path),
                        }
                progress["files_total"] = sum(len(e["manifest"]) for e in entries.values())
                progress["bytes_total"] = sum(size for e in entries.values() for _, size, _ in e["manifest"])
                progress["phase"] = "copying"
                update()
                for e in entries.values():
                    (staging / e["relative"].parent).mkdir(parents=True, exist_ok=True)
                    copy_source(e["path"], staging / e["relative"], e["manifest"], copied)
                for item in [*sources, *validation]:
                    item["path"] = str(
                        final / entries[str(Path(item["path"]).expanduser().resolve())]["relative"]
                    )
            else:
                config["dataset"]["sources"] = []
                config["validation"]["sources"] = []
                config["validation"]["enabled"] = False
            checkpoint = config.setdefault("checkpoint", {})
            if c.inherits_output_dir(pid, source_id, checkpoint.get("output_dir")):
                checkpoint["output_dir"] = str(c.default_runs_dir(pid, vid))
            checkpoint["resume"] = None
            config.setdefault("sampling", {})["output_dir"] = None
            config.setdefault("adapter", {})["resume_weights"] = None
            config.setdefault("dataset", {})["cache_dir"] = str(c.cache_dir(pid, vid))
            config.setdefault("logging", {})["events_path"] = None
            for directory in (
                c.dataset_dir(pid, vid),
                c.reg_dir(pid, vid),
                c.samples_dir(pid, vid),
                c.default_runs_dir(pid, vid),
                final / "cache",
            ):
                (staging / directory.relative_to(final)).mkdir(parents=True, exist_ok=True)
            (staging / "config.json").write_text(
                json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            staging.rename(final)
            promoted = True
            progress["phase"] = "indexing"
            update()
            for entry in entries.values():
                item = entry["source"]
                did = entry["id"]
                c.db.insert(
                    "datasets",
                    {
                        "id": did,
                        "project_id": pid,
                        "version_id": vid,
                        "path": str(final / entry["relative"]),
                        "origin_path": entry["origin_path"],
                        "repeats": item.get("repeats", 1),
                        "caption_ext": item.get("caption_ext", "auto"),
                        "is_reg": int(item.get("is_reg", False)),
                        "prior_weight": item.get("prior_weight", 1.0),
                        "class_prompt": item.get("class_prompt"),
                        "created_at": now(),
                        "index_status": "indexing",
                        "stats_json": "{}",
                    },
                )
                registered.append(did)
                _index_dataset(c, did)
                if (
                    c.db.fetchone("SELECT index_status FROM datasets WHERE id=?", (did,))["index_status"]
                    != "ready"
                ):
                    raise ApiError("copied dataset could not be indexed", code="version.index_failed")
            progress["phase"] = "ready"
            update()
            c.db.update("project_versions", vid, {"status": "ready", "updated_at": now()})
        except Exception as exc:
            for did in registered:
                c.db.delete("datasets", did)
                _records_path(c, did).unlink(missing_ok=True)
            if promoted:
                shutil.rmtree(final, ignore_errors=True)
            progress["phase"] = "failed"
            c.db.update(
                "project_versions",
                vid,
                {
                    "status": "failed",
                    "error": str(exc),
                    "progress_json": json.dumps(progress),
                    "updated_at": now(),
                },
            )
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)
            c.db.execute("UPDATE project_versions SET busy=NULL WHERE id=? AND busy=?", (source_id, vid))
            c.bus.publish("version.changed", {"project_id": pid, "version_id": vid})

    def import_directory(self, pid: str, vid: str, body: Any) -> str:
        from .routes_work import _register_dataset, get_project_config

        c = self.c
        source = Path(body.path).expanduser().resolve()
        if not source.is_dir():
            raise NotFound(f"directory not found: {source}", code="fs.not_found")
        if not c.is_allowed(source):
            raise ApiError("source is outside allowed roots", code="fs.forbidden", status=403)
        with self.mutation(pid, vid):
            if c.db.fetchone(
                "SELECT id FROM datasets WHERE version_id=? AND (path=? OR origin_path=?)",
                (vid, str(source), str(source)),
            ):
                raise ApiError(
                    "this dataset directory is already imported", code="dataset.duplicate", status=409
                )
            did = new_id("d")
            is_reg = body.is_reg
            config = get_project_config(pid, c, vid)
            match = next(
                (
                    item
                    for item in [
                        *config.get("dataset", {}).get("sources", []),
                        *config.get("validation", {}).get("sources", []),
                    ]
                    if Path(item["path"]).expanduser().resolve() == source
                ),
                None,
            )
            if match and "is_reg" not in body.model_fields_set:
                is_reg = bool(match.get("is_reg"))
            # Match _register_dataset: explicitly supplied options override the existing
            # source; omitted fields inherit it. Preflight must inspect that same format.
            caption_ext = body.caption_ext
            if match and "caption_ext" not in body.model_fields_set:
                caption_ext = match.get("caption_ext", body.caption_ext)
            root = c.dataset_dir(pid, vid, is_reg=is_reg)
            if root.resolve().is_relative_to(source):
                raise ApiError(
                    "dataset source cannot contain the managed dataset destination",
                    code="version.source_recursive",
                )
            root.mkdir(parents=True, exist_ok=True)
            if root.is_symlink():
                raise ApiError("managed dataset root cannot be a symbolic link", code="version.source_link")
            target = root / dataset_directory_name(did, source)
            staging = Path(tempfile.mkdtemp(prefix=".import-", dir=root))
            promoted = False
            try:
                copy_source(source, staging / "data", file_manifest(source))
                # Validate the selected structured sidecars before promoting a copied
                # folder or recording it in this version. JSON bytes are never captions.
                from ypuddin.data.captions import read_training_caption
                from ypuddin.data.index import caption_for, iter_images

                try:
                    caption_directories = {}
                    for image in iter_images(staging / "data"):
                        caption = caption_for(image, caption_ext, directory_cache=caption_directories)
                        if caption and Path(caption).suffix.lower() == ".json":
                            read_training_caption(caption)
                except ValueError as error:
                    raise ApiError(str(error), code="dataset.caption_invalid", status=400) from error
                (staging / "data").rename(target)
                promoted = True
                _register_dataset(
                    c,
                    pid,
                    body.model_copy(update={"path": str(target)}),
                    did=did,
                    version_id=vid,
                    internal=True,
                    origin_path=str(source),
                )
                return did
            except BaseException as exc:
                if promoted:
                    shutil.rmtree(target, ignore_errors=True)
                if isinstance(exc, OSError):
                    raise ApiError(f"could not import dataset: {exc}", code="version.import_failed") from exc
                raise
            finally:
                shutil.rmtree(staging, ignore_errors=True)
