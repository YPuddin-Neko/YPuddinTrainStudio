"""Shared speech-job action projection and execution admission."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, get_args

from ypuddin.tts.issues import TtsIssue

from .errors import ApiError
from .job_paths import deleting_jobs

JobAction = Literal[
    "cancel", "retry", "force", "pause", "resume", "save", "archive", "unarchive", "delete",
    "change_gpu", "change_priority",
]
JOB_ACTIONS: tuple[JobAction, ...] = get_args(JobAction)
TTS_JOBS = frozenset({"tts_train", "tts_sample"})
TERMINAL = frozenset({"completed", "failed", "cancelled"})
WAITING = frozenset({"queued", "scheduled"})
ACTIVE = frozenset({"queued", "scheduled", "running", "cancelling", "pausing", "paused"})


def _issue(code: str, message: str, *, loc: list[str | int] | None = None, **details: Any) -> TtsIssue:
    return TtsIssue(code=code, loc=loc or ["job"], message=message, details=details)


def _payload(job: dict) -> dict:
    try:
        value = json.loads(job.get("config_json") or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _id(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _source_id(job: dict, payload: dict | None = None) -> str | None:
    payload = _payload(job) if payload is None else payload
    sample = payload.get("tts_sample")
    summary = payload.get("source_summary")
    return (
        _id(job.get("source_job_id")) or _id(payload.get("source_job_id"))
        or (_id(sample.get("source_job_id")) if isinstance(sample, dict) else None)
        or (_id(summary.get("job_id")) if isinstance(summary, dict) else None)
    )


def _running(c: Any, job: dict) -> bool:
    supervisor = getattr(c, "supervisor", None)
    return bool(supervisor is not None and supervisor.is_running(job["id"]))


def _owner_deletion_issue(c: Any, job: dict) -> TtsIssue | None:
    from .project_deletion import deleting

    pid = job.get("project_id")
    if pid and deleting(c, pid):
        return _issue("project.deleting", "任务所属项目正在删除。", loc=["project_id"], project_id=pid)
    return None


def owner_issue(c: Any, job: dict) -> TtsIssue | None:
    """Owner admission for new execution; existing jobs do not make a ready version busy."""
    pid, vid = job.get("project_id"), job.get("version_id")
    if pid is None and vid is None:
        return None
    project = c.db.fetchone("SELECT * FROM projects WHERE id=?", (pid,)) if pid else None
    if project is None:
        return _issue("project.not_found", "任务所属项目不存在。", loc=["project_id"], project_id=pid)
    if project.get("project_type") != "tts":
        return _issue("project.type_mismatch", "语音任务的项目类型不匹配。", loc=["project_id"], project_id=pid)
    version = c.db.fetchone("SELECT * FROM project_versions WHERE id=?", (vid,)) if vid else None
    if version is None:
        return _issue("version.not_found", "任务所属版本不存在。", loc=["version_id"], version_id=vid)
    if version.get("project_id") != pid:
        return _issue("project.type_mismatch", "任务所属版本与项目不匹配。", loc=["version_id"], project_id=pid, version_id=vid)
    if problem := _owner_deletion_issue(c, job):
        return problem
    if project.get("archived"):
        return _issue("project.archived", "请先恢复项目，再创建或调整语音任务。", loc=["project_id"], project_id=pid)
    if version.get("archived") or version.get("busy") or version.get("status") != "ready":
        return _issue("version.busy", "任务所属版本已归档或尚未就绪。", loc=["version_id"], version_id=vid,
                      status=version.get("status"), archived=bool(version.get("archived")), busy=version.get("busy"))
    return None


def active_children(c: Any, parentid: str) -> list[dict]:
    """Include terminal rows whose process has not yet been reaped."""
    rows = c.db.fetchall("SELECT * FROM jobs WHERE type='tts_sample' ORDER BY created_at,id")
    return [row for row in rows if _source_id(row) == parentid and (row.get("status") in ACTIVE or _running(c, row))]


def _readable_file(c: Any, value: Any, *, label: str) -> Path:
    from ypuddin.tts.source_scan import absolute_path, open_source

    if not isinstance(value, str) or not value.strip() or not Path(value).expanduser().is_absolute():
        raise ValueError(f"{label}缺少有效的绝对路径。")
    path = absolute_path(value)
    with open_source(path, c.is_allowed) as stream:
        if os.fstat(stream.fileno()).st_size <= 0:
            raise ValueError(f"{label}不存在或不是有效文件。")
    return path


def dependency_issue(c: Any, job: dict) -> TtsIssue | None:
    """Check frozen references using metadata; creation verifies their full content identities."""
    payload = _payload(job)
    if not isinstance(payload.get("tts"), dict):
        return _issue("tts.source_unavailable", "任务的语音配置快照不可用。", loc=["config"])
    paths = []
    if job.get("type") == "tts_sample":
        source_id = _source_id(job, payload)
        if not source_id:
            return _issue("tts.source_unavailable", "试听任务缺少来源训练身份。", loc=["source_job_id"])
        source = c.db.fetchone("SELECT * FROM jobs WHERE id=?", (source_id,))
        if source is not None:
            if source.get("type") != "tts_train":
                return _issue("tts.job_type_mismatch", "试听来源不是语音训练任务。", loc=["source_job_id"], source_job_id=source_id)
            if source_id in deleting_jobs:
                return _issue("job.deleting", "来源训练任务正在删除。", loc=["source_job_id"], source_job_id=source_id)
            if source.get("archived_at") is not None:
                return _issue("job.archived", "请先恢复来源训练任务，再重试试听。", loc=["source_job_id"], source_job_id=source_id)
            if problem := owner_issue(c, source):
                return problem
        sample = payload.get("tts_sample")
        if not isinstance(sample, dict):
            return _issue("tts.source_unavailable", "试听任务缺少冻结的检查点信息。", loc=["tts_sample"])
        from .tts_results import frozen_checkpoint_issue

        if problem := frozen_checkpoint_issue(c, sample, verify=False):
            return problem
        if sample.get("reference_audio"):
            try:
                paths.append(_readable_file(c, sample["reference_audio"], label="参考音频"))
            except (OSError, ValueError, RuntimeError) as exc:
                return _issue("tts.source_unavailable", str(exc), loc=["tts_sample", "reference_audio"])
    else:
        for name in ("train_manifest", "val_manifest"):
            value = payload["tts"].get(name)
            if not value and name == "val_manifest":
                continue
            try:
                paths.append(_readable_file(c, value, label="冻结数据清单"))
            except (OSError, ValueError, RuntimeError) as exc:
                return _issue("tts.source_unavailable", str(exc), loc=["tts", name])
    try:
        from .tts_references import reject_deleting_references

        reject_deleting_references(c, paths)
    except ApiError as exc:
        return _issue(exc.code, exc.message, **exc.details)
    return None


def _decisions(c: Any, job: dict, requested: tuple[str, ...] = JOB_ACTIONS) -> dict[str, TtsIssue | None]:
    status = job.get("status")
    archived = job.get("archived_at") is not None
    running = _running(c, job)
    owner = owner_issue(c, job)
    owner_deleting = _owner_deletion_issue(c, job)
    dependencies: list[TtsIssue | None] = []
    outcomes = {}
    for action in requested:
        if action in {"pause", "resume", "save"}:
            problem = _issue("job.bad_state", "语音任务不支持暂停、恢复或手动保存。", action=action)
        elif job["id"] in deleting_jobs:
            problem = _issue("job.deleting", "这个任务正在删除。")
        elif owner_deleting:
            problem = owner_deleting
        elif archived and action not in {"unarchive", "delete"}:
            problem = _issue("job.archived", "请先恢复已归档的任务。")
        elif action in {"retry", "force", "change_gpu", "change_priority"} and owner:
            problem = owner
        elif action == "cancel":
            problem = None if status in WAITING | {"running"} else _issue("job.bad_state", "当前任务状态不能取消。", status=status)
        elif action == "unarchive":
            problem = None if archived else _issue("job.bad_state", "任务尚未归档。", status=status)
        elif running:
            problem = _issue("job.bad_state", "任务进程尚未退出，请等待退出后再操作。", status=status)
        elif action == "archive":
            problem = None if status in TERMINAL else _issue("job.bad_state", "只能归档已结束的任务。", status=status)
        elif action == "delete":
            if not archived:
                problem = _issue("job.archive_required", "请先归档任务，再永久删除。")
            elif status not in TERMINAL:
                problem = _issue("job.bad_state", "只能删除已结束的任务。", status=status)
            elif job.get("type") == "tts_train" and (children := active_children(c, job["id"])):
                problem = _issue("job.files_in_use", "来源训练仍被活动试听任务使用，请等试听结束后再删除。", jobs=[child["id"] for child in children])
            else:
                problem = None
        elif action == "change_gpu":
            problem = None if status in WAITING | {"failed", "cancelled"} else _issue("job.bad_state", "当前任务状态不能更换显卡。", status=status)
        elif action == "change_priority":
            problem = None if status in WAITING else _issue("job.bad_state", "只能调整排队任务的优先级。", status=status)
        elif action == "force":
            problem = None if status in WAITING and job.get("forced_at") is None else _issue("job.bad_state", "只有尚未强制开始的排队任务可以强制开始。", status=status)
        else:
            problem = None if status in TERMINAL else _issue("job.bad_state", "只能重试已结束的任务。", status=status)
        if problem is None and action in {"retry", "force", "change_gpu"}:
            if not dependencies:
                dependencies.append(dependency_issue(c, job))
            problem = dependencies[0]
        outcomes[action] = problem
    return outcomes


def present(c: Any, job: dict) -> dict:
    if job.get("type") not in TTS_JOBS:
        return {}
    payload = _payload(job)
    outcomes = _decisions(c, job)
    return {
        "source_job_id": _source_id(job, payload),
        "retry_of_job_id": _id(job.get("retry_of_job_id")) or _id(payload.get("retry_of_job_id")),
        "allowed_actions": [action for action, issue in outcomes.items() if issue is None],
        "action_reasons": {action: issue.model_dump() for action, issue in outcomes.items() if issue is not None},
    }


def assert_action(c: Any, job: dict, action: str) -> None:
    if job.get("type") not in TTS_JOBS:
        return
    if action not in JOB_ACTIONS:
        raise ApiError("未知的语音任务操作。", code="job.bad_command", status=404)
    issue = _decisions(c, job, (action,))[action]
    if issue is not None:
        raise ApiError(issue.message, code=issue.code, status=409,
                       details={**issue.details, "issue": issue.model_dump(), "issues": [issue.model_dump()]})
