"""Independent speech version copies with owned configuration and audio data."""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any

from ypuddin.tts.source_scan import file_identity, identity_changes
from ypuddin.tts.version_config import TtsConfigEnvelope, default_config, write_envelope

from . import tts_source_copy as data_copy
from .db import new_id, now
from .errors import ApiError


def _progress(phase: str = "planning") -> dict:
    return {"phase": phase, "files_done": 0, "files_total": 0, "bytes_done": 0, "bytes_total": 0}


def _release_source(c: Any, source_id: str, vid: str) -> None:
    c.db.execute("UPDATE project_versions SET busy=NULL WHERE id=? AND busy=?", (source_id, vid))


def _failed(c: Any, vid: str, source_id: str, progress: dict, exc: Exception) -> None:
    progress["phase"] = "failed"
    with c.db.transaction():
        c.db.update("project_versions", vid, {"status": "failed", "error": str(exc),
                    "progress_json": json.dumps(progress), "updated_at": now()})
        _release_source(c, source_id, vid)


def _admit(c: Any, pid: str, source_id: str | None) -> dict:
    from .project_deletion import deleting
    from .versions import assert_version_writable

    c.require_project_type(pid, "tts")
    if deleting(c, pid):
        raise ApiError("项目正在删除。", code="project.deleting", status=409)
    source = assert_version_writable(c, pid, source_id)
    if c.db.fetchone("SELECT id FROM tts_sources WHERE version_id=? AND state='checking'", (source["id"],)):
        raise ApiError("数据来源正在检查，请等待检查完成。", code="version.busy", status=409)
    return source


def create_version(manager: Any, pid: str, name: str, note: str, source_id: str | None,
                   data_mode: str, copy_config: bool, *, family: str | None = None) -> dict:
    from .tts_projects import get_config, version_engine
    from .tts_references import reject_deleting_references
    from .versions import version_row

    c = manager.c
    with c.db.lock:
        source = _admit(c, pid, source_id)
        if family is not None:
            raise ApiError("语音版本不能设置图像模型类型。", code="project.type_mismatch", status=409)
        if data_mode not in {"copy", "empty"} or not copy_config and data_mode != "empty":
            raise ApiError("copy_config=false requires data_mode=empty", code="version.invalid")
        source_id = source["id"]
        source_path = c.version_dir(pid, source_id)
    source_guard = data_copy.DirectoryGuard.capture(source_path, c.is_allowed)
    plan = data_copy.prepare(c, pid, source_id) if data_mode == "copy" else None
    with c.db.lock:
        source = _admit(c, pid, source_id)
        data_copy.assert_identity(c, pid, source_id, plan)
        source_guard.check(c.is_allowed)
        if c.version_dir(pid, source_id) != source_path:
            raise ApiError("来源版本目录已改变。", code="version.path", status=409)
        reject_deleting_references(c, [source_path, *data_copy.dependency_paths(plan)])
        if c.db.fetchone("SELECT id FROM project_versions WHERE project_id=? AND name=?", (pid, name)):
            raise ApiError("a version with this name already exists", code="version.duplicate", status=409)
        config = get_config(c, pid, source_id).config if copy_config else default_config(version_engine(c, pid, source_id) or "voxcpm1.5")
        if not copy_config and config.engine == "gpt-sovits-v5":
            config.variant = source.get("tts_variant") or "v5dev"
        envelope = TtsConfigEnvelope(revision=1, config=config)
        vid = new_id("v")
        number = c.db.fetchone("SELECT coalesce(max(number),0)+1 n FROM project_versions WHERE project_id=?", (pid,))["n"]
        progress, timestamp = _progress(), now()
        with c.db.transaction():
            c.db.insert("project_versions", {"id": vid, "project_id": pid, "number": number, "name": name,
                        "note": note, "parent_version_id": source_id, "data_revision": 1, "tts_engine": config.engine, "tts_variant": getattr(config, "variant", None),
                        "created_at": timestamp, "updated_at": timestamp, "status": "copying",
                        "progress_json": json.dumps(progress)})
            c.db.update("project_versions", source_id, {"busy": vid})
            response = version_row(c, c.resolve_version(pid, vid))
        try:
            future = manager.executor.submit(_copy, c, pid, vid, source_id, envelope, plan, source_guard)

            def cancelled(completed):
                if completed.cancelled():
                    _failed(c, vid, source_id, progress, RuntimeError("版本复制已取消。"))
                    c.bus.publish("version.changed", {"project_id": pid, "version_id": vid})

            future.add_done_callback(cancelled)
        except Exception as exc:
            _failed(c, vid, source_id, progress, exc)
            c.bus.publish("version.changed", {"project_id": pid, "version_id": vid})
            response = version_row(c, c.resolve_version(pid, vid))
        return response


def _directory_identity(path: Path) -> tuple[int, int]:
    return data_copy._directory_key(path)


def _remove_owned(path: Path | None, identity: tuple[int, int] | None, *, parent: data_copy.DirectoryGuard | None = None) -> None:
    if path is None or identity is None:
        return
    try:
        if parent is not None and shutil.rmtree.avoids_symlink_attacks:
            with parent.opened(lambda _: True) as parent_fd:
                if parent_fd is not None:
                    fd = os.open(path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
                    try:
                        info = os.fstat(fd)
                        if (info.st_dev, info.st_ino) != identity:
                            return
                        for name in os.listdir(fd):
                            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                            if stat.S_ISDIR(info.st_mode):
                                shutil.rmtree(name, dir_fd=fd)
                            else:
                                os.unlink(name, dir_fd=fd)
                        info = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
                        if (info.st_dev, info.st_ino) == identity:
                            os.rmdir(path.name, dir_fd=parent_fd)
                    finally:
                        os.close(fd)
                    return
        if parent is not None:
            parent.check(lambda _: True)
        if _directory_identity(path) == identity:
            shutil.rmtree(path)
    except (OSError, ApiError):
        return


def _assert_copy_state(c: Any, pid: str, vid: str, source_id: str, plan: dict | None) -> None:
    from .project_deletion import deleting
    from .tts_references import reject_deleting_references

    project = c.require_project_type(pid, "tts")
    source, target = c.resolve_version(pid, source_id), c.resolve_version(pid, vid)
    if deleting(c, pid):
        raise ApiError("项目正在删除。", code="project.deleting", status=409)
    if project["archived"] or source["archived"] or target["archived"]:
        raise ApiError("项目或版本已归档。", code="version.busy", status=409)
    if source["status"] != "ready" or source["busy"] != vid or target["status"] != "copying" or target["parent_version_id"] != source_id:
        raise ApiError("版本复制已取消或版本状态已改变。", code="version.busy", status=409)
    data_copy.assert_identity(c, pid, source_id, plan)
    reject_deleting_references(c, data_copy.dependency_paths(plan))


def _promote(c: Any, staging: data_copy.DirectoryGuard, final: Path,
             parent: data_copy.DirectoryGuard, owned: dict) -> data_copy.DirectoryGuard:
    """Reserve the destination exclusively; never replace a pre-existing directory."""
    try:
        target = parent.child(final.name, c.is_allowed)
    except FileExistsError as exc:
        raise ApiError("目标版本目录已改变。", code="version.path", status=409) from exc
    owned["final"] = target.chain[-1][1]
    with staging.opened(c.is_allowed) as source_fd, target.opened(c.is_allowed) as target_fd:
        for path in sorted(staging.path.iterdir()):
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ApiError("版本临时文件已改变。", code="version.path", status=409)
            os.rename(path.name if source_fd is not None else path,
                      path.name if target_fd is not None else final / path.name,
                      src_dir_fd=source_fd, dst_dir_fd=target_fd)
    return target


def _copy(c: Any, pid: str, vid: str, source_id: str, envelope: TtsConfigEnvelope,
          plan: dict | None = None, source_guard: data_copy.DirectoryGuard | None = None) -> None:
    staging = final = parent = None
    owned: dict[str, tuple[int, int]] = {}
    progress = _progress()
    committed = False
    try:
        with c.db.lock:
            _assert_copy_state(c, pid, vid, source_id, plan)
            final = c.version_dir(pid, vid)
        parent = data_copy.DirectoryGuard.capture(final.parent, c.is_allowed)
        if source_guard:
            source_guard.check(c.is_allowed)
        if final.exists() or final.is_symlink():
            raise ApiError("目标版本目录已存在。", code="version.path", status=409)
        data_copy.recheck(c, pid, source_id, plan)
        staging = data_copy.new_staging(parent, vid, c.is_allowed)
        owned["staging"] = staging.chain[-1][1]
        target = staging.path / "tts-config.json"
        write_envelope(target, envelope)
        config_identity = file_identity(target, c.is_allowed)
        size = config_identity["size"]
        progress.update(phase="copying", files_total=1, files_done=1, bytes_total=size, bytes_done=size)

        def advance(**increments):
            for key, value in increments.items():
                progress[key] += value
            with c.db.lock:
                _assert_copy_state(c, pid, vid, source_id, plan)
                c.db.update("project_versions", vid, {"progress_json": json.dumps(progress), "updated_at": now()})
            c.bus.publish("version.changed", {"project_id": pid, "version_id": vid, "progress": dict(progress)})

        advance()
        entries = data_copy.copy_data(c, plan, staging, final, advance)
        data_copy.recheck(c, pid, source_id, plan)
        if source_guard:
            source_guard.check(c.is_allowed)
        parent.check(c.is_allowed)
        with c.db.lock:
            _assert_copy_state(c, pid, vid, source_id, plan)
            if c.version_dir(pid, vid) != final:
                raise ApiError("目标版本目录已改变。", code="version.path", status=409)
        destination = _promote(c, staging, final, parent, owned)
        entries = data_copy.scan_copies(c, entries, engine=envelope.config.engine)
        data_copy.recheck(c, pid, source_id, plan)
        if any(identity_changes(entry["scan"].identities, allowed=c.is_allowed) for entry in entries):
            raise ApiError("复制后的录音或清单已变化。", code="tts.source_stale", status=409)
        current_config = file_identity(final / "tts-config.json", c.is_allowed)
        if any(current_config[key] != config_identity[key] for key in ("sha256", "size")):
            raise ApiError("复制后的语音配置已变化。", code="version.path", status=409)
        destination.check(c.is_allowed)
        if source_guard:
            source_guard.check(c.is_allowed)
        with c.db.transaction():
            _assert_copy_state(c, pid, vid, source_id, plan)
            destination.check(c.is_allowed)
            if c.version_dir(pid, vid) != final:
                raise ApiError("目标版本目录已改变。", code="version.path", status=409)
            data_copy.publish(c, pid, vid, entries)
            progress["phase"] = "ready"
            c.db.update("project_versions", vid, {"status": "ready", "progress_json": json.dumps(progress), "updated_at": now()})
            _release_source(c, source_id, vid)
        committed = True
        for entry in entries:
            c.tts_sources._event(c.db.fetchone("SELECT * FROM tts_sources WHERE id=?", (entry["id"],)), "check_completed")
    except Exception as exc:
        if not committed:
            _remove_owned(final, owned.get("final"), parent=parent)
            _failed(c, vid, source_id, progress, exc)
    finally:
        _remove_owned(staging.path if staging else None, owned.get("staging"), parent=parent)
        _release_source(c, source_id, vid)
        c.bus.publish("version.changed", {"project_id": pid, "version_id": vid})
