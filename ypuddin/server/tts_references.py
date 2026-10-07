"""Prevent speech registrations from acquiring files while their owner is deleting them."""

from __future__ import annotations

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


def reject_deleting_references(c: Any, paths: list[Path]) -> None:
    """Caller holds the database admission lock; only existing deletion plans matter."""
    from .job_paths import deleting_jobs, job_directories
    from .project_deletion import plan

    for jid in tuple(deleting_jobs):
        job = c.db.fetchone("SELECT * FROM jobs WHERE id=?", (jid,))
        if job is None:
            continue
        targets = [folder.resolve() for folder in job_directories(job) if folder.name == jid]
        for path in paths:
            resolved = Path(path).expanduser().resolve()
            if any(resolved == target or resolved.is_relative_to(target) for target in targets):
                raise ApiError(
                    "引用文件所在的任务正在删除。", code="job.deleting", status=409,
                    details={"blocker": {"job_id": jid, "path": str(path)}},
                )

    if not c.db.fetchone("SELECT name FROM sqlite_master WHERE type='table' AND name='project_deletions'"):
        return
    for row in c.db.fetchall("SELECT project_id,skipped_json FROM project_deletions WHERE state='deleting'"):
        # Include skipped locations conservatively while the owning project is being removed.
        found = plan(c, row["project_id"], check_tts_refs=False)
        targets = [location.path.resolve() for location in found.locations]
        for path in paths:
            resolved = Path(path).expanduser().resolve()
            if any(resolved == target or resolved.is_relative_to(target) for target in targets):
                raise ApiError(
                    "录音或清单所在的项目正在删除。",
                    code="project.deleting",
                    status=409,
                    details={"blocker": {"project_id": row["project_id"], "path": str(path)}},
                )
