"""Prevent speech registrations from acquiring files while their owner is deleting them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import ApiError


def reject_job_file_references(c: Any, targets: list[Path]) -> None:
    """Repeat under the admission lock immediately before marking job files for deletion."""
    manager = getattr(c, "tts_sources", None)
    if manager is None or not targets:
        return
    from .tts_source_copy import active_references

    references = manager.references_to(targets)
    references += active_references(c, targets)
    if references:
        raise ApiError(
            "语音数据来源或正在复制的版本仍引用这些清单或录音，请移除引用或等待复制完成后再删除。",
            code="job.files_in_use", status=409, details={"sources": references},
        )


def _deletion_targets(c: Any, row: dict, *, version_id: str | None = None) -> list[Path]:
    from .project_deletion import plan

    if row.get("plan_json"):
        try:
            return [Path(item["path"]).resolve() for item in json.loads(row["plan_json"])["locations"]]
        except (KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
            raise ApiError("无法读取删除位置，请完成删除后再引用文件。",
                           code="version.references_unresolved" if version_id else "project.references_unresolved", status=409) from exc
    found = plan(c, row["project_id"], version_id=version_id, check_tts_refs=False)
    return [location.path.resolve() for location in found.locations]


def _overlaps(path: Path, targets: list[Path]) -> bool:
    directory = path.is_dir()
    return any(path == target or path.is_relative_to(target) or directory and target.is_relative_to(path)
               for target in targets)


def reject_deleting_references(c: Any, paths: list[Path]) -> None:
    """Caller holds the database admission lock; only existing deletion plans matter."""
    from .job_paths import deleting_jobs, job_directories

    for jid in tuple(deleting_jobs):
        job = c.db.fetchone("SELECT * FROM jobs WHERE id=?", (jid,))
        if job is None:
            continue
        targets = [folder.resolve() for folder in job_directories(job) if folder.name == jid]
        for path in paths:
            resolved = Path(path).expanduser().resolve()
            if _overlaps(resolved, targets):
                raise ApiError(
                    "引用文件所在的任务正在删除。", code="job.deleting", status=409,
                    details={"blocker": {"job_id": jid, "path": str(path)}},
                )

    projects = c.db.fetchall("SELECT * FROM project_deletions WHERE state='deleting'") if c.db.fetchone(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='project_deletions'"
    ) else []
    for row in projects:
        # Include skipped locations conservatively while the owning project is being removed.
        targets = _deletion_targets(c, row)
        for path in paths:
            resolved = Path(path).expanduser().resolve()
            if _overlaps(resolved, targets):
                raise ApiError(
                    "引用文件所在的项目正在删除。",
                    code="project.deleting",
                    status=409,
                    details={"blocker": {"project_id": row["project_id"], "path": str(path)}},
                )

    versions = c.db.fetchall("SELECT * FROM version_deletions WHERE state IN ('deleting','failed')") if c.db.fetchone(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='version_deletions'"
    ) else []
    for row in versions:
        targets = _deletion_targets(c, row, version_id=row["version_id"])
        for path in paths:
            resolved = Path(path).expanduser().resolve()
            if _overlaps(resolved, targets):
                raise ApiError(
                    "引用文件所在的版本尚未完成删除。", code="version.deleting", status=409,
                    details={"blocker": {"project_id": row["project_id"], "version_id": row["version_id"], "path": str(path)}},
                )
