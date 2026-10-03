"""Project version snapshots and managed data copies, without shared mutable files."""

from __future__ import annotations

import json
import os
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


def dataset_directory_name(source: Path) -> str:
    return source.name or "dataset"


def assert_version_writable(c: Any, pid: str, vid: str | None, *, data: bool = False) -> dict:
    row = c.resolve_version(pid, vid)
    project = c.db.fetchone("SELECT archived FROM projects WHERE id=?", (pid,))
    if project is None or project["archived"]:
        raise ApiError("项目已归档，请先恢复项目再修改配置或创建任务。", code="project.archived", status=409)
    if row["status"] != "ready" or row["busy"] or row["archived"]:
        raise ApiError(
            "版本已归档或正在处理，请恢复版本或等待当前操作完成。", code="version.busy", status=409
        )
    if data and c.db.fetchone(
        f"SELECT id FROM jobs WHERE version_id=? AND status IN {ACTIVE_JOBS}", (row["id"],)
    ):
        raise ApiError(
            "版本数据正被排队或运行中的任务使用，请等任务结束或取消后再修改。", code="version.jobs_busy", status=409
        )
    if data and c.db.fetchone(
        "SELECT id FROM datasets WHERE version_id=? AND index_status='indexing'", (row["id"],)
    ):
        raise ApiError(
            "数据集正在建立索引，请等索引完成后再修改版本数据。", code="version.indexing", status=409
        )
    return row


def version_row(c: Any, row: dict) -> dict:
    from .artifact_inventory import artifact_count
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
            "jobs": c.db.fetchone(
                "SELECT count(*) n FROM jobs WHERE version_id=? AND archived_at IS NULL", (vid,)
            )["n"],
            "artifacts": artifact_count(c.db, version_id=vid),
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
            "jobs": str(c.records_root(pid, vid)),
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


def copy_source(
    source: Path, target: Path, manifest: list, progress: Any = None, *, import_progress: Any = None
) -> None:
    target.mkdir(parents=True, exist_ok=False)
    for relative, size, mtime in manifest:
        original = source / relative
        before = original.stat()
        if before.st_size != size or before.st_mtime_ns != mtime or original.is_symlink():
            raise ApiError(f"source changed during snapshot: {original}", code="version.source_changed")
        dest = target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if import_progress:
            with original.open("rb") as stream, dest.open("xb") as output:
                while chunk := stream.read(1024**2):
                    output.write(chunk)
                    import_progress.advance(bytes_done=len(chunk))
            shutil.copystat(original, dest)
        else:
            shutil.copy2(original, dest)  # Deliberately never use hard links.
        after = original.stat()
        if after.st_size != size or after.st_mtime_ns != mtime:
            raise ApiError(f"source changed during snapshot: {original}", code="version.source_changed")
        if progress:
            progress(size)
        if import_progress:
            import_progress.advance(files_done=1)
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
            stats = json.loads(row["stats_json"] or "{}") | {"error": "服务停止时索引还没有完成。"}
            # Without a signature the next read finds the folder changed and indexes it again.
            stats.pop("_source_signature", None)
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
                config["adapter"]["dora_compute_mode"] = "comfyui"
                config["model"]["attention"] = environment_attention_default(c)
                if data_mode != "empty":
                    raise ApiError("copy_config=false requires data_mode=empty", code="version.invalid")
            if family is not None:
                config = (
                    change_config_family(c, config, family)
                    if copy_config
                    else initial_family_config(c, family)
                )
            model = config.get("model", {})
            if model.get("family") == "flux" or (
                model.get("family") == "flux2" and model.get("flux2_variant") == "dev"
            ):
                raise ApiError(
                    "此配置使用已停用的 FLUX 模型；请新建 Klein 基础版参数，不要继承旧模型配置。",
                    code="version.family_retired",
                    status=422,
                )
            datasets = c.db.fetchall("SELECT * FROM datasets WHERE version_id=?", (source["id"],))
            from .source_roles import managed_source_role

            # Registry-only sources also follow the source version's real directory
            # ownership. Keep historical rows untouched while planning this copy.
            for dataset in datasets:
                role = managed_source_role(c, pid, source["id"], dataset["path"])
                if role is not None:
                    dataset["is_reg"] = role[0]
            if any(row["index_status"] == "indexing" for row in datasets):
                raise ApiError(
                    "数据集正在建立索引，请等索引完成后再创建版本。", code="version.busy", status=409
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
                items = [*sources, *validation, *datasets]
                relative_paths = {}
                reserved = set()
                for item in items:
                    path = Path(item["path"]).expanduser().resolve()
                    is_reg = bool(item.get("is_reg"))
                    old_root = c.dataset_dir(pid, source_id, is_reg=is_reg).resolve()
                    if path.is_relative_to(old_root):
                        relative = c.dataset_dir(pid, vid, is_reg=is_reg).relative_to(
                            final
                        ) / path.relative_to(old_root)
                        relative_paths[str(path)] = relative
                        reserved.add(relative.as_posix().casefold())
                for item in items:
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
                        relative = relative_paths.get(key)
                        if relative is None:
                            parent = c.dataset_dir(pid, vid, is_reg=bool(item.get("is_reg"))).relative_to(
                                final
                            )
                            name, number = dataset_directory_name(path), 2
                            relative = parent / name
                            while relative.as_posix().casefold() in reserved:
                                relative, number = parent / f"{name}-{number}", number + 1
                            reserved.add(relative.as_posix().casefold())
                        if any(
                            relative.is_relative_to(entry["relative"])
                            or entry["relative"].is_relative_to(relative)
                            for entry in entries.values()
                        ):
                            raise ApiError(
                                "dataset sources overlap; keep either the parent source or its children",
                                code="version.source_overlap",
                            )
                        entries[key] = {
                            "id": did,
                            "path": path,
                            "source": item,
                            "origin_path": origins.get(key) or key,
                            "manifest": file_manifest(path),
                            "relative": relative,
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
            config.setdefault("adapter", {})["resume_weights"] = None
            requested_cache = config.setdefault("dataset", {}).get("cache_dir")
            if not requested_cache or c.training_cache_dir(pid, source_id, requested_cache) == c.cache_dir(
                pid, source_id
            ):
                config["dataset"]["cache_dir"] = str(c.version_dir(pid, vid) / "cache")
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

    def import_directory(self, pid: str, vid: str, body: Any, *, progress: Any = None) -> str:
        from .dataset_uploads import merge_dataset_files
        from .routes_work import _register_dataset, get_project_config

        c = self.c
        source = Path(body.path).expanduser().resolve()
        if not source.is_dir():
            raise NotFound(f"directory not found: {source}", code="fs.not_found")
        if not c.is_allowed(source):
            raise ApiError("source is outside allowed roots", code="fs.forbidden", status=403)
        with self.mutation(pid, vid):
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
            from .source_roles import managed_source_role

            role = managed_source_role(c, pid, vid, str(source))
            if role is not None:
                is_reg = role[0]
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
            rows = c.db.fetchall("SELECT * FROM datasets WHERE version_id=?", (vid,))
            target = root / dataset_directory_name(source)
            # A previous import may use an older managed name. Synchronize that
            # same source without moving existing user data or creating a duplicate.
            origin = next(
                (
                    row
                    for row in rows
                    if row.get("origin_path") == str(source)
                    and Path(row["path"]).resolve().is_relative_to(root.resolve())
                    and Path(row["path"]).resolve() != root.resolve()
                ),
                None,
            )
            if origin:
                target = Path(origin["path"])
            ancestors = [row for row in rows if target.resolve().is_relative_to(Path(row["path"]).resolve())]
            if len(ancestors) > 1 or (
                ancestors and not Path(ancestors[0]["path"]).resolve().is_relative_to(root.resolve())
            ):
                raise ApiError(
                    "dataset sources overlap across managed directories", code="dataset.overlap", status=409
                )
            existing = ancestors[0] if ancestors else None
            if not existing and any(
                Path(row["path"]).resolve().is_relative_to(target.resolve()) for row in rows
            ):
                raise ApiError(
                    "a child directory is already a dataset; import that child instead",
                    code="dataset.overlap",
                    status=409,
                )
            registration = body.model_copy(update={"path": str(target), "is_reg": is_reg})
            if existing:
                caption_ext = existing["caption_ext"]
            else:
                configured = [
                    *config.get("dataset", {}).get("sources", []),
                    *config.get("validation", {}).get("sources", []),
                ]
                parents = {
                    Path(item["path"]).expanduser().resolve()
                    for item in configured
                    if item.get("path")
                    and target.resolve().is_relative_to(Path(item["path"]).expanduser().resolve())
                }
                if len(parents) > 1 or any(not parent.is_relative_to(root.resolve()) for parent in parents):
                    raise ApiError(
                        "configured sources overlap across managed directories",
                        code="dataset.overlap",
                        status=409,
                    )
                if parents:
                    from .routes_work import DatasetBody

                    parent = parents.pop()
                    if any(Path(row["path"]).resolve().is_relative_to(parent) for row in rows):
                        raise ApiError(
                            "the configured parent overlaps an indexed child dataset",
                            code="dataset.overlap",
                            status=409,
                        )
                    registration = DatasetBody(path=str(parent), is_reg=is_reg)
                    caption_ext = next(
                        item.get("caption_ext", "auto")
                        for item in configured
                        if item.get("path") and Path(item["path"]).expanduser().resolve() == parent
                    )
                elif any(
                    item.get("path")
                    and Path(item["path"]).expanduser().resolve().is_relative_to(target.resolve())
                    for item in configured
                ):
                    raise ApiError(
                        "a child directory is already configured; import that child instead",
                        code="dataset.overlap",
                        status=409,
                    )
            staging = Path(tempfile.mkdtemp(prefix=".import-", dir=root))
            try:
                snapshot = staging / target.relative_to(root)
                manifest = file_manifest(source)
                if progress:
                    progress.set_phase(
                        "copying", bytes_total=sum(item[1] for item in manifest), files_total=len(manifest)
                    )
                copy_source(source, snapshot, manifest, import_progress=progress)
                # Validate the selected structured sidecars before promoting a copied
                # folder or recording it in this version. JSON bytes are never captions.
                from ypuddin.data.captions import read_training_caption
                from ypuddin.data.index import caption_for, iter_images

                try:
                    caption_directories = {}
                    images = list(iter_images(snapshot))
                    if progress:
                        progress.set_phase("validating", files_total=len(images))
                    for image in images:
                        caption = caption_for(image, caption_ext, directory_cache=caption_directories)
                        if caption and Path(caption).suffix.lower() == ".json":
                            read_training_caption(caption)
                        if progress:
                            progress.advance(files_done=1)
                except ValueError as error:
                    raise ApiError(str(error), code="dataset.caption_invalid", status=400) from error
                with merge_dataset_files(
                    root, staging, [snapshot / item[0] for item in manifest], progress=progress
                ):
                    if progress:
                        progress.set_phase("registering")
                    if existing:
                        did = existing["id"]
                        c.db.update("datasets", did, {"index_status": "indexing"})
                    else:
                        _register_dataset(
                            c,
                            pid,
                            registration,
                            did=did,
                            version_id=vid,
                            internal=True,
                            origin_path=str(source),
                        )
                return did
            except BaseException as exc:
                if isinstance(exc, OSError):
                    raise ApiError(f"could not import dataset: {exc}", code="version.import_failed") from exc
                raise
            finally:
                shutil.rmtree(staging, ignore_errors=True)
