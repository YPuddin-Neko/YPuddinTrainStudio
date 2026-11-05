"""Deleting a project with its files, in the background.

The confirm dialog first lists every folder the deletion removes (``storage``). Deleting marks the
project under the database lock, removes the folders outside the lock with progress in the task
center, and removes the records last, so the rest of the studio keeps working meanwhile. Only the
folders the studio made for the project and its jobs go, never a custom root itself or anything
else in it. A failure keeps the project archived to retry; a restart continues the deletion.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .db import now
from .errors import ApiError, NotFound
from .job_paths import deleting_jobs, deletion_roots, folder_problem, job_directories, studio_root
from .versions import ACTIVE_JOBS

log = logging.getLogger(__name__)

TABLE = """CREATE TABLE IF NOT EXISTS project_deletions (
    project_id TEXT PRIMARY KEY, state TEXT NOT NULL, skipped_json TEXT NOT NULL DEFAULT '[]',
    task_id TEXT, error TEXT, started_at REAL NOT NULL, updated_at REAL NOT NULL
)"""
VERSION_TABLE = """CREATE TABLE IF NOT EXISTS version_deletions (
    version_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, state TEXT NOT NULL,
    skipped_json TEXT NOT NULL DEFAULT '[]', plan_json TEXT,
    task_id TEXT, error TEXT, started_at REAL NOT NULL, updated_at REAL NOT NULL
)"""
# Rows of other services that only describe the project's own versions.
_VERSION_TABLES = ("dataset_pipeline_operations", "regularization_operations", "site_downloads")


def deletion_state(c: Any, pid: str, version_id: str | None = None) -> dict | None:
    try:
        if version_id is not None:
            return c.db.fetchone(
                "SELECT * FROM version_deletions WHERE project_id=? AND version_id=?", (pid, version_id)
            )
        return c.db.fetchone("SELECT * FROM project_deletions WHERE project_id=?", (pid,))
    except sqlite3.OperationalError:  # a context whose deletion service never started
        return None


def deleting(c: Any, pid: str, version_id: str | None = None) -> bool:
    state = deletion_state(c, pid, version_id)
    return bool(state and state["state"] == "deleting")


def public_state(c: Any, pid: str, version_id: str | None = None) -> dict | None:
    state = deletion_state(c, pid, version_id)
    return {key: state[key] for key in ("state", "error", "task_id")} if state else None


@dataclass
class Location:
    path: Path
    kinds: list[str]
    custom: bool
    problem: str | None = None
    # Folders under a custom root that only existed for this project, removed once empty.
    parents: list[Path] = field(default_factory=list)
    root: Path | None = None
    identity: list[dict] | None = None


@dataclass
class Plan:
    project: dict
    locations: list[Location]
    records: list[Path]
    jobs: list[dict]
    blocked: ApiError | None = None
    version: dict | None = None


def _job_roles(job: dict) -> list[tuple[str, Path | None]]:
    from .job_paths import event_file, output_directory, state_directory

    return [
        ("products", output_directory(job)),
        ("records", Path(job["run_dir"])),
        ("resume", state_directory(job)),
        ("logs", event_file(job).parent),
        ("samples", Path(job["samples_dir"]) if job.get("samples_dir") else None),
    ]


def _resolved(path: Path) -> Path | None:
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return None


def plan(c: Any, pid: str, *, version_id: str | None = None, check_tts_refs: bool = True) -> Plan:
    """What deleting the project removes and what stops it; reads the disk but holds no lock."""
    from .routes_work import _get_project, _jobs_using

    project = _get_project(c, pid)
    data_root = c.data_root.resolve()
    project_root = c.project_dir(pid)
    version = c.resolve_version(pid, version_id) if version_id is not None else None
    clause, args = ("version_id=?", (version_id,)) if version else ("project_id=?", (pid,))
    jobs = c.db.fetchall(f"SELECT * FROM jobs WHERE {clause} ORDER BY created_at", args)
    versions = [version] if version else c.db.fetchall("SELECT id FROM project_versions WHERE project_id=?", (pid,))
    datasets = c.db.fetchall(f"SELECT id,path FROM datasets WHERE {clause}", args)
    # Keep the lexical location so a redirected root can be listed and explicitly kept.
    owned_root = (project_root / f"v{version['number']}" if c.project_layout(pid) >= 2 else
                  project_root / "versions" / version_id) if version else project_root
    candidates: dict[Path, list[str]] = {owned_root: ["version" if version else "project"]}
    if version and version["legacy_layout"] and project["project_type"] == "image":
        candidates[c.config_path(pid, version_id)] = ["config"]
        legacy_data = project_root / "datasets"
        for dataset in datasets:
            path = Path(dataset["path"]).expanduser().absolute()
            if path.is_relative_to(legacy_data.absolute()):
                candidates.setdefault(path, ["datasets"])
    extra_roots: list[Path] = []
    for job in jobs:
        roles = _job_roles(job)
        # Named after the job, including a linked one, which is then reported instead of followed.
        for folder in dict.fromkeys(path for path in job_directories(job) if path.name == job["id"]):
            if root := studio_root(c, job, folder):
                extra_roots.append(root)
            kinds = [role for role, path in roles if path is not None and (path == folder or folder in path.parents)]
            candidates.setdefault(folder, [])
            candidates[folder] += [kind for kind in kinds or ["records"] if kind not in candidates[folder]]
    # Training caches: the version folder holds them by default, a custom cache root a <project>/<version>
    # folder each, also for jobs made while an earlier cache root was set.
    cache_root = Path(c.settings()["paths"]["cache_dir"])
    caches = []
    if cache_root.resolve() != (c.data_root / "cache").resolve():
        caches += [cache_root / pid / version["id"] for version in versions]
    for job in jobs:
        value = (json.loads(job.get("config_json") or "{}").get("dataset") or {}).get("cache_dir")
        if isinstance(value, str) and value:
            folder = Path(value)
            if folder.name == job.get("version_id") and folder.parent.name == pid:
                caches.append(folder)
                extra_roots.append(folder.parent.parent)
    for folder in caches:
        candidates.setdefault(folder, ["cache"])
    roots = deletion_roots(c, extra_roots)
    locations: list[Location] = []
    covered: list[Path] = []
    for folder, kinds in sorted(candidates.items(), key=lambda item: (item[0] != owned_root, len(item[0].parts), str(item[0]))):
        resolved = _resolved(folder)
        # A job folder inside the project folder goes with it.
        if resolved is not None and any(resolved.is_relative_to(parent) for parent in covered):
            continue
        problem = folder_problem(folder, roots) if resolved is not None else "unavailable"
        if version and folder.is_relative_to(project_root) and any(
            path.is_symlink() for path in (project_root.parent, project_root, owned_root.parent, owned_root)
        ):
            problem = "linked"
        if problem in ("outside", "linked") and not os.path.lexists(folder) and not (version and folder == owned_root):
            continue  # nothing left there to remove
        location = Location(folder, kinds, custom=resolved is None or not resolved.is_relative_to(data_root), problem=problem)
        if problem is None and resolved is not None:
            covered.append(resolved)
            root = max((r for r in roots if resolved.is_relative_to(r)), key=lambda r: len(r.parts))
            location.root = root
            if root != data_root:
                location.parents = [parent for parent in folder.parents[: len(resolved.relative_to(root).parts) - 1]]
        locations.append(location)
    records = [c.data_root / "datasets" / f"{row['id']}.json" for row in datasets]
    result = Plan(project, locations, records, jobs, version=version)
    targets = [location.path for location in locations if location.problem is None]
    result.blocked = (
        _version_references(c, pid, version_id, targets) if version else
        _blocking(c, pid, targets, covered) or _in_use(c, pid, targets, _jobs_using)
    )
    result.blocked = result.blocked or _reference_lease_blocker(c, targets, version_id)
    if check_tts_refs and result.blocked is None:
        result.blocked = _tts_reference_blocker(c, pid, targets, version_id=version_id)
    return result


def _reference_lease_blocker(c: Any, targets: list[Path], version_id: str | None) -> ApiError | None:
    from .versions import active_references

    with c.db.lock:
        if active_references(c, targets):
            return ApiError("这些文件正在被复制或检查，请等待操作完成。",
                            code="version.files_in_use" if version_id is not None else "project.files_in_use", status=409)
    return None


def _tts_reference_blocker(c: Any, pid: str, targets: list[Path], *, version_id: str | None = None) -> ApiError | None:
    manager = getattr(c, "tts_sources", None)
    if manager is None:
        return None
    try:
        exclusion = {"excluding_version_id": version_id} if version_id is not None else {"excluding_project_id": pid}
        references = manager.references_to(targets, **exclusion)
        from .tts_source_copy import active_references

        references += active_references(c, targets, **exclusion)
    except ApiError as exc:
        return exc
    if references:
        return ApiError(
            "其他版本仍引用这些清单或录音，移除引用后才能删除。",
            code="version.files_in_use" if version_id is not None else "project.files_in_use",
            status=409, details={"sources": references},
        )
    return None


def _blocking(c: Any, pid: str, targets: list[Path], resolved: list[Path]) -> ApiError | None:
    """Another job's recorded folder, or another project's dataset, inside the folders to remove."""
    if not resolved:
        return None
    for row in c.db.fetchall(
        "SELECT run_dir, samples_dir, config_json FROM jobs WHERE project_id IS NULL OR project_id!=?", (pid,)
    ):
        for path in job_directories(row):
            if (path := _resolved(path)) and any(path.is_relative_to(target) for target in resolved):
                return ApiError("目录中包含其他任务的文件，无法删除。", code="project.files_in_use", status=409)
    for row in c.db.fetchall(
        "SELECT d.path, p.name FROM datasets d LEFT JOIN projects p ON p.id=d.project_id"
        " WHERE d.project_id IS NULL OR d.project_id!=?",
        (pid,),
    ):
        path = _resolved(Path(row["path"]))
        if path and any(path.is_relative_to(target) for target in resolved):
            owner = f"项目“{row['name']}”的" if row["name"] else "另一个"
            return ApiError(
                f"{owner}数据集在这个项目的文件夹里，删除会一并删掉它，因此不能删除。",
                code="project.files_in_use",
                status=409,
            )
    return None


def _in_use(c: Any, pid: str, targets: list[Path], jobs_using: Any) -> ApiError | None:
    if users := jobs_using(c, pid, targets, owner_type="project"):
        return ApiError(
            f"任务“{users[0]['name']}”还要用到这个项目的文件，请等它结束或取消后再删除。",
            code="project.files_in_use",
            status=409,
            details={"jobs": [row["id"] for row in users]},
        )
    return None


def _version_references(c: Any, pid: str, vid: str, targets: list[Path]) -> ApiError | None:
    """Version deletion keeps references from sibling versions as well as other projects."""
    from .xyz import file_reference_config

    resolved = [path.resolve() for path in targets]

    def refers(value: Any, *, contains: bool = False) -> bool:
        if isinstance(value, dict):
            return any(refers(item) for item in value.values())
        if isinstance(value, list):
            return any(refers(item) for item in value)
        if not isinstance(value, str) or not value:
            return False
        path = Path(value).expanduser()
        if not path.is_absolute() or (path := _resolved(path)) is None:
            return False
        return any(path.is_relative_to(target) or contains and target.is_relative_to(path) for target in resolved)

    for row in c.db.fetchall("SELECT * FROM jobs WHERE version_id IS NULL OR version_id!=?", (vid,)):
        try:
            used = any(refers(str(path)) for path in job_directories(row))
            historical_consumer = row["type"] in {"tts_sample", "xyz"} and row["status"] in {
                "completed", "failed", "cancelled"
            } and not c.supervisor.is_running(row["id"])
            if not historical_consumer:
                config = file_reference_config(json.loads(row.get("config_json") or "{}"))
                used = used or refers(row.get("resume_from")) or refers(config)
        except (ValueError, TypeError):
            return ApiError("其他任务的路径记录无法读取，请修正后再删除版本。", code="version.references_unresolved", status=409)
        if used:
            return ApiError("其他任务仍引用这个版本的文件，不能删除。", code="version.files_in_use", status=409,
                            details={"jobs": [row["id"]]})
    for row in c.db.fetchall("SELECT path FROM datasets WHERE version_id IS NULL OR version_id!=?", (vid,)):
        if refers(row["path"], contains=True):
            return ApiError("其他版本的数据集仍使用这些文件，不能删除。", code="version.files_in_use", status=409)
    for row in c.db.fetchall("SELECT path FROM models"):
        if refers(row["path"], contains=True):
            return ApiError("本地模型仍引用这个版本的文件，请先移除模型登记。", code="version.files_in_use", status=409)
    for row in c.db.fetchall(
        "SELECT v.id,v.project_id,p.project_type FROM project_versions v JOIN projects p ON p.id=v.project_id WHERE v.id!=?",
        (vid,),
    ):
        try:
            path = (c.tts_config_path(row["project_id"], row["id"]) if row["project_type"] == "tts" else
                    c.config_path(row["project_id"], row["id"]))
            if path.is_file() and refers(json.loads(path.read_text(encoding="utf-8"))):
                return ApiError("其他版本的参数仍引用这些文件，请先修改引用。", code="version.files_in_use", status=409,
                                details={"version_id": row["id"]})
        except (OSError, ValueError, TypeError, RuntimeError, ApiError) as exc:
            if isinstance(exc, ApiError):
                return exc
            return ApiError("其他版本的参数无法读取，请修正后再删除版本。", code="version.references_unresolved", status=409)
    return None


def _version_fallback(c: Any, pid: str, vid: str) -> dict | None:
    project = c.db.fetchone("SELECT archived FROM projects WHERE id=?", (pid,))
    return c.db.fetchone(
        "SELECT v.* FROM project_versions v WHERE v.project_id=? AND v.id!=? AND v.status='ready' AND v.busy IS NULL"
        + (" AND v.archived=0" if project and not project["archived"] else "")
        + " AND NOT EXISTS (SELECT 1 FROM version_deletions d WHERE d.version_id=v.id)"
        " ORDER BY v.archived ASC,v.created_at DESC,v.id LIMIT 1", (pid, vid),
    )


def _version_busy(c: Any, pid: str, version: dict, *, continuing: bool = False) -> ApiError | None:
    vid = version["id"]
    project = c.db.fetchone("SELECT * FROM projects WHERE id=?", (pid,))
    if not project:
        return NotFound("项目不存在。", code="project.not_found")
    if deletion_state(c, pid) is not None:
        return ApiError("项目正在删除或等待重试，请先完成项目删除。", code="project.deleting", status=409)
    if c.db.fetchone("SELECT version_id FROM version_deletions WHERE project_id=? AND state='deleting' AND version_id!=?", (pid, vid)):
        return ApiError("这个项目中有版本正在删除，请等待删除完成。", code="version.deleting", status=409)
    if not continuing and deleting(c, pid, vid):
        return ApiError("这个版本正在删除。", code="version.deleting", status=409)
    if not version["archived"] and not project["archived"]:
        return ApiError("请先归档版本，再永久删除。", code="version.archive_required", status=409)
    if version["status"] == "copying" or version["busy"]:
        return ApiError("版本的数据正在复制或处理，请等操作完成。", code="version.busy", status=409)
    if c.db.fetchone("SELECT id FROM datasets WHERE version_id=? AND index_status='indexing'", (vid,)):
        return ApiError("版本的数据集正在建立索引，请等待索引完成。", code="version.busy", status=409)
    refresh = getattr(c, "dataset_refresh", None)
    if refresh is not None and refresh.active_datasets(pid, version_id=vid):
        return ApiError("版本的数据集正在刷新，请等待刷新完成。", code="version.busy", status=409)
    if c.db.fetchone("SELECT id FROM tts_sources WHERE version_id=? AND state='checking'", (vid,)):
        return ApiError("版本的数据来源正在检查，请等待检查完成。", code="version.busy", status=409)
    from .tts_validation import validation_busy

    if blocker := validation_busy(c, version_id=vid):
        return ApiError("版本正在检查训练参数，请等待检查完成。", code="version.busy", status=409, details={"blocker": blocker})
    jobs = c.db.fetchall("SELECT * FROM jobs WHERE version_id=?", (vid,))
    if any(row["status"] in {"queued", "scheduled", "running", "pausing", "cancelling", "paused", "interrupted"}
           or c.supervisor.is_running(row["id"]) for row in jobs):
        return ApiError("版本还有未结束的任务，请先取消或等待任务结束。", code="version.jobs_busy", status=409)
    if any(row["id"] in deleting_jobs for row in jobs):
        return ApiError("版本中有任务的文件正在删除，请等待删除完成。", code="version.busy", status=409)
    from .tts_job_actions import active_children
    from .xyz import dependent_tests

    for row in jobs:
        if row["type"] == "tts_train" and active_children(c, row["id"]):
            return ApiError("版本的训练仍被活动试听使用，请等待试听结束。", code="version.files_in_use", status=409)
        if row["type"] == "train":
            try:
                dependent_tests(c, row)
            except ApiError as exc:
                return exc
    if _version_fallback(c, pid, vid) is None:
        return ApiError("请通过删除整个项目移除最后一个可用版本。", code="version.last", status=409)
    return None


def _project_busy(c: Any, pid: str, project: dict) -> ApiError | None:
    """Conditions in the database that refuse any deletion of the project; caller holds the lock."""
    refresh = getattr(c, "dataset_refresh", None)
    if refresh is not None and refresh.active_datasets(pid):
        return ApiError("项目的数据集正在刷新，请等待刷新完成。", code="project.busy", status=409)
    if c.db.fetchone("SELECT id FROM tts_sources WHERE project_id=? AND state='checking'", (pid,)):
        return ApiError("项目的数据来源正在检查，请等待检查完成。", code="project.busy", status=409)
    from .tts_validation import validation_busy

    if blocker := validation_busy(c, project_id=pid):
        return ApiError("项目正在检查训练参数，请等待检查完成。", code="project.busy", status=409, details={"blocker": blocker})
    if c.db.fetchone(f"SELECT id FROM jobs WHERE project_id=? AND status IN {ACTIVE_JOBS[:-1]},'interrupted')", (pid,)):
        return ApiError("项目还有排队或运行中的任务，请等它们结束或取消后再删除。", code="project.busy", status=409)
    if c.db.fetchone(
        "SELECT id FROM project_versions WHERE project_id=? AND (status='copying' OR busy IS NOT NULL)", (pid,)
    ):
        return ApiError("项目的数据正在复制或处理，请等操作完成后再删除。", code="project.busy", status=409)
    jobs = [row["id"] for row in c.db.fetchall("SELECT id FROM jobs WHERE project_id=?", (pid,))]
    if any(c.supervisor.is_running(jid) for jid in jobs):
        return ApiError("项目的任务进程还没有退出，请稍后再删除。", code="project.busy", status=409)
    if any(jid in deleting_jobs for jid in jobs):
        return ApiError("项目中有任务的文件正在删除，请等删除完成后再删除项目。", code="project.busy", status=409)
    from .tts_job_actions import active_children

    for source in c.db.fetchall("SELECT id FROM jobs WHERE project_id=? AND type='tts_train'", (pid,)):
        if children := active_children(c, source["id"]):
            return ApiError("项目的训练仍被活动试听使用，请先等待试听结束。", code="job.files_in_use", status=409,
                            details={"jobs": [child["id"] for child in children]})
    if not project["archived"]:
        return ApiError("请先归档项目，再永久删除。", code="project.archive_required", status=409)
    if deleting(c, pid):
        return ApiError("这个项目正在删除。", code="project.deleting", status=409)
    if c.db.fetchone("SELECT version_id FROM version_deletions WHERE project_id=? AND state='deleting'", (pid,)):
        return ApiError("这个项目中有版本正在删除，请等待删除完成。", code="version.deleting", status=409)
    return None


def _kept_entry(path: Path, skipped: tuple[Path, ...]) -> bool:
    return any(path.is_relative_to(keep) or keep.is_relative_to(path) for keep in skipped)


def _count_files(path: Path, skipped: tuple[Path, ...] = ()) -> int:
    total = 0
    for directory, folders, files in os.walk(os.path.abspath(path)):
        folders[:] = [name for name in folders if not any((Path(directory) / name).is_relative_to(keep) for keep in skipped)]
        total += sum(not _kept_entry(Path(directory) / name, skipped) for name in files)
    return total


def _failure(error: Exception) -> str:
    where = f"“{error.filename}”" if isinstance(error, OSError) and error.filename else ""
    if isinstance(error, PermissionError):
        return f"没有权限删除{where or '项目文件'}，请检查文件权限后重试。"
    if isinstance(error, OSError) and where:
        return f"删除{where}时出错：{error.strerror or error}"
    return f"删除文件时出错：{error}"


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except PermissionError:
        # Read-only files on Windows keep their delete permission once writable.
        if os.lstat(path).st_nlink > 1:
            raise  # Changing permissions would also change another hard link's file.
        os.chmod(path, stat.S_IWRITE)
        os.unlink(path)


def _path_identity(path: Path) -> dict | None:
    try:
        value = path.lstat()
    except FileNotFoundError:
        return None
    return {"device": value.st_dev, "inode": value.st_ino, "type": stat.S_IFMT(value.st_mode)}


def _freeze_plan(found: Plan) -> str:
    locations = []
    for location in found.locations:
        identity = None
        if location.problem is None and location.root is not None:
            path = Path(os.path.abspath(location.path))
            identity = []
            while True:
                identity.append({"path": str(path), "stat": _path_identity(path)})
                if path.resolve() == location.root:
                    break
                if path.parent == path:
                    raise ApiError("删除位置已变化，请重新检查。", code="project.path", status=409)
                path = path.parent
            identity.reverse()
        location.identity = identity
        locations.append({
            "path": str(location.path), "kinds": location.kinds, "custom": location.custom,
            "problem": location.problem, "parents": [str(path) for path in location.parents],
            "root": str(location.root) if location.root is not None else None, "identity": identity,
        })
    return json.dumps({"version": 1, "locations": locations, "records": [str(path) for path in found.records]})


def _validate_location(location: Location) -> None:
    if location.problem is not None or location.identity is None:
        return
    if not location.root or not location.root.is_dir():
        raise ApiError("删除位置暂时无法访问，请检查磁盘后重试。", code="project.location_unavailable", status=409)
    if folder_problem(location.path, [location.root]) is not None:
        raise ApiError("删除位置已变化或经过符号链接，请检查后重试。", code="project.path", status=409)
    # Missing children can have been removed by an earlier attempt; a replaced root cannot.
    chain = location.identity if os.path.lexists(location.path) else location.identity[:1]
    for item in chain:
        if _path_identity(Path(item["path"])) != item["stat"]:
            raise ApiError("删除位置已被替换，请检查后重试。", code="project.path", status=409)


def _saved_plan(c: Any, pid: str, vid: str | None, payload: str) -> Plan:
    from .routes_work import _get_project

    data = json.loads(payload)
    if data.get("version") != 1:
        raise ApiError("删除记录无法读取，请检查后重试。", code="project.path", status=409)
    locations = [Location(
        Path(item["path"]), item["kinds"], item["custom"], item["problem"],
        [Path(path) for path in item["parents"]], Path(item["root"]) if item["root"] else None, item["identity"],
    ) for item in data["locations"]]
    clause, args = ("version_id=?", (vid,)) if vid is not None else ("project_id=?", (pid,))
    return Plan(
        _get_project(c, pid), locations, [Path(path) for path in data["records"]],
        c.db.fetchall(f"SELECT * FROM jobs WHERE {clause} ORDER BY created_at", args),
        version=c.resolve_version(pid, vid) if vid is not None else None,
    )


def _recheck(c: Any, pid: str, found: Plan, skipped: set[str]) -> None:
    from .routes_work import _jobs_using

    targets = []
    for location in found.locations:
        if str(location.path) in skipped:
            continue
        if location.problem:
            raise ApiError("有位置无法删除，请重新确认需要保留的位置。", code="project.path", status=409)
        _validate_location(location)
        targets.append(location.path)
    vid = found.version["id"] if found.version else None
    if vid is not None:
        blocked = _version_references(c, pid, vid, targets)
    else:
        blocked = _blocking(c, pid, targets, [path.resolve() for path in targets]) or _in_use(c, pid, targets, _jobs_using)
    blocked = blocked or _reference_lease_blocker(c, targets, vid) or _tts_reference_blocker(c, pid, targets, version_id=vid)
    if blocked:
        raise blocked


class ProjectDeletions:
    def __init__(self, context: Any):
        self.c = context
        self.stopping = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="project-delete")
        self._running: dict[tuple[str, str | None], Any] = {}
        self._lock = threading.Lock()
        self.c.db.execute(TABLE)
        self.c.db.execute(VERSION_TABLE)
        if "plan_json" not in {row["name"] for row in self.c.db.fetchall("PRAGMA table_info(project_deletions)")}:
            self.c.db.execute("ALTER TABLE project_deletions ADD COLUMN plan_json TEXT")
        for row in self.c.db.fetchall("SELECT project_id FROM project_deletions WHERE state='deleting'"):
            # The service stopped partway: continue with the locations the user confirmed.
            self._submit(row["project_id"], resumed=True)
        for row in self.c.db.fetchall("SELECT project_id,version_id FROM version_deletions WHERE state='deleting'"):
            self._submit(row["project_id"], version_id=row["version_id"], resumed=True)

    def close(self) -> None:
        self.stopping.set()
        self.executor.shutdown(wait=True)

    def wait(self, timeout: float = 30.0) -> bool:
        """Wait for deletions that are running; for tests and shutdown."""
        with self._lock:
            futures = list(self._running.values())
        for future in futures:
            try:
                future.result(timeout=timeout)
            except Exception:  # noqa: BLE001 - the deletion records its own failure
                pass
        return all(future.done() for future in futures)

    def storage(self, pid: str, *, version_id: str | None = None) -> dict:
        """Everything deleting the project removes from disk, for the confirm dialog."""
        from .routes_work import _folder_size

        state = deletion_state(self.c, pid, version_id)
        found = (_saved_plan(self.c, pid, version_id, state["plan_json"]) if state and state.get("plan_json") else
                 plan(self.c, pid, version_id=version_id))
        locations = []
        path_error = None
        for location in found.locations:
            problem = location.problem
            try:
                _validate_location(location)
            except ApiError as exc:
                problem, path_error = "unavailable", exc
            exists = problem != "unavailable" and location.path.exists()
            size, files = (0, 0)
            if exists and problem is None:
                size, files = (_folder_size(location.path) if location.path.is_dir() else (location.path.stat().st_size, 1))
            locations.append({
                "path": str(location.path), "kinds": location.kinds, "custom": location.custom,
                "exists": exists, "bytes": size, "files": files, "problem": problem,
            })
        with self.c.db.lock:
            blocked = (_version_busy(self.c, pid, found.version) if found.version else
                       _project_busy(self.c, pid, found.project)) or path_error or found.blocked
            if blocked is None and state and state.get("plan_json"):
                try:
                    _recheck(self.c, pid, found, set(json.loads(state["skipped_json"])))
                except ApiError as exc:
                    blocked = exc
        column, owner = ("version_id", version_id) if version_id is not None else ("project_id", pid)
        artifacts = self.c.db.fetchone(f"SELECT count(*) n FROM artifacts WHERE {column}=?", (owner,))["n"]
        return {
            "locations": locations,
            "total_bytes": sum(item["bytes"] for item in locations),
            "jobs": len(found.jobs),
            "artifacts": artifacts,
            "blocked": {
                "code": blocked.code, "message": blocked.message,
                "jobs": (blocked.details or {}).get("jobs", []),
            } if blocked else None,
            "deletion": public_state(self.c, pid, version_id),
        }

    def start(self, pid: str, skip: list[str], *, version_id: str | None = None) -> str | None:
        """Mark the project as being deleted and remove its files in the background."""
        previous = deletion_state(self.c, pid, version_id)
        found = (_saved_plan(self.c, pid, version_id, previous["plan_json"]) if previous and previous.get("plan_json") else
                 plan(self.c, pid, version_id=version_id))
        if version_id is not None and set(skip) - {str(location.path) for location in found.locations}:
            raise ApiError("只能保留删除预览中列出的位置，请刷新预览后重试。", code="version.skip_invalid", status=422)
        skip = sorted(set(skip) | set(json.loads(previous["skipped_json"]) if previous else []))
        problems = [
            location for location in found.locations
            if location.problem and str(location.path) not in skip
        ]
        with self.c.db.lock:
            project = self.c.db.fetchone("SELECT * FROM projects WHERE id=?", (pid,))
            if project is None:
                raise NotFound(f"project {pid} not found", code="project.not_found")
            version = self.c.resolve_version(pid, version_id) if version_id is not None else None
            if error := (_version_busy(self.c, pid, version) if version else _project_busy(self.c, pid, project)):
                raise error
            if found.blocked:
                raise found.blocked
            if problems:
                unavailable = all(location.problem == "unavailable" for location in problems)
                raise ApiError(
                    "有位置无法访问，请确认后选择跳过或取消。" if unavailable
                    else "有位置不在数据文件夹或已设置的保存位置内，或经过符号链接，不能自动删除。",
                    code="project.location_unavailable" if unavailable else "project.path",
                    status=409 if unavailable else 403,
                    details={"locations": [
                        {"path": str(location.path), "problem": location.problem} for location in problems
                    ]},
                )
            _recheck(self.c, pid, found, set(skip))
            frozen = previous["plan_json"] if previous and previous.get("plan_json") else _freeze_plan(found)
            tasks = self.c.background_tasks
            task = (
                tasks.start(
                    "version_delete" if version else "project_delete",
                    f"{project['name']} · {version['name']}" if version else project["name"], unit="files",
                    link=f"/archive?tab=versions&project={pid}" if version else "/projects?archived=1",
                )
                if tasks is not None else None
            )
            table = "version_deletions" if version else "project_deletions"
            columns = "project_id,state,skipped_json,plan_json,task_id,error,started_at,updated_at"
            values = (pid, "deleting", json.dumps(skip), frozen, task, None, now(), now())
            if version:
                columns, values = "version_id," + columns, (version_id, *values)
            self.c.db.execute(f"INSERT OR REPLACE INTO {table} ({columns}) VALUES ({','.join('?' * len(values))})", values)
        self._changed(pid, version_id)
        self._submit(pid, found=found, task=task, version_id=version_id)
        return task

    def _changed(self, pid: str, vid: str | None) -> None:
        payload = {"project_id": pid, "version_id": vid}
        self.c.bus.publish("projects.changed", payload)
        self.c.bus.publish("versions.changed", payload)

    def _submit(self, pid: str, *, found: Plan | None = None, task: str | None = None, resumed: bool = False,
                version_id: str | None = None) -> None:
        with self._lock:
            self._running[(pid, version_id)] = self.executor.submit(self._run, pid, found, task, resumed, version_id)

    def _run(self, pid: str, found: Plan | None, task: str | None, resumed: bool, version_id: str | None = None) -> None:
        tasks = self.c.background_tasks
        table, key, owner = ("version_deletions", "version_id", version_id) if version_id is not None else (
            "project_deletions", "project_id", pid
        )
        try:
            state = deletion_state(self.c, pid, version_id)
            if state is None or state["state"] != "deleting":
                return
            skipped = set(json.loads(state["skipped_json"] or "[]"))
            # Keep both a link's name and its destination, including paths through /tmp-style roots.
            kept_paths = tuple({
                path for value in skipped
                for path in (Path(os.path.abspath(Path(value).expanduser())), Path(value).expanduser().resolve())
            })
            if found is None:
                found = (_saved_plan(self.c, pid, version_id, state["plan_json"]) if state.get("plan_json") else
                         plan(self.c, pid, version_id=version_id))
                if found.blocked:
                    raise found.blocked
                if any(location.problem and str(location.path) not in skipped for location in found.locations):
                    raise ApiError("服务重启后，有位置无法删除，请重新确认。", code="project.path", status=409)
            with self.c.db.lock:
                if found.version and (error := _version_busy(self.c, pid, found.version, continuing=True)):
                    raise error
                _recheck(self.c, pid, found, skipped)
            if task is None and tasks is not None:
                task = tasks.start(
                    "version_delete" if found.version else "project_delete",
                    f"{found.project['name']} · {found.version['name']}" if found.version else found.project["name"],
                    unit="files", link=f"/archive?tab=versions&project={pid}" if found.version else "/projects?archived=1",
                )
                with self.c.db.lock:
                    self.c.db.execute(
                        f"UPDATE {table} SET task_id=?, updated_at=? WHERE {key}=?", (task, now(), owner)
                    )
            targets = [
                location for location in found.locations
                if location.problem is None and location.path.exists()
                and not any(location.path.resolve().is_relative_to(keep) for keep in kept_paths)
            ]
            total = sum(_count_files(location.path, kept_paths) if location.path.is_dir() else 1 for location in targets)
            done = 0
            if task is not None:
                tasks.update(task, done=0, total=total, force=True)

            def removed() -> None:
                nonlocal done
                done += 1
                if self.stopping.is_set():
                    raise InterruptedError
                if task is not None and done % 50 == 0:
                    tasks.update(task, done=done)

            for location in targets:
                _validate_location(location)
                self._remove(location.path, removed, kept_paths)
                for parent in location.parents:
                    try:
                        parent.rmdir()  # only an emptied folder made for this project
                    except OSError:
                        break
            for record in found.records:
                record.unlink(missing_ok=True)
            self._remove_rows(pid, version_id=version_id)
            if task is not None:
                tasks.finish(task)
            self.c.bus.publish("queue.changed", {})
            self._changed(pid, version_id)
        except InterruptedError:
            # Stopping the service leaves the deletion marked; the next start continues it.
            return
        except Exception as exc:  # noqa: BLE001 - the project stays archived for a retry
            message = exc.message if isinstance(exc, ApiError) else _failure(exc)
            if not isinstance(exc, ApiError):
                log.exception("project deletion failed")
            with self.c.db.lock:
                self.c.db.execute(
                    f"UPDATE {table} SET state='failed', error=?, updated_at=? WHERE {key}=?",
                    (message, now(), owner),
                )
            if task is not None:
                tasks.finish(task, error=message)
            self._changed(pid, version_id)
        finally:
            with self._lock:
                self._running.pop((pid, version_id), None)

    @staticmethod
    def _remove(folder: Path, removed: Any, skipped: tuple[Path, ...] = ()) -> None:
        """Remove files without following links; keep skipped folders and the parents holding them."""
        folder = Path(os.path.abspath(folder))
        if folder.is_symlink():
            raise ApiError("删除位置已变成符号链接，请检查后重试。", code="project.path", status=409)
        if folder.is_file():
            if not _kept_entry(folder, skipped):
                _unlink(str(folder))
                removed()
            return
        for directory, folders, files in os.walk(folder, topdown=False):
            current = Path(directory)
            if any(current.is_relative_to(keep) for keep in skipped):
                continue
            for name in files:
                # An offline directory link is listed as a file by os.walk.
                if _kept_entry(current / name, skipped):
                    continue
                _unlink(os.path.join(directory, name))
                removed()
            for name in folders:
                path = os.path.join(directory, name)
                if any(keep.is_relative_to(path) for keep in skipped):
                    continue
                if os.path.islink(path):
                    os.unlink(path)
                else:
                    os.rmdir(path)
        if not any(keep.is_relative_to(folder) for keep in skipped):
            os.rmdir(folder)

    def _remove_rows(self, pid: str, *, version_id: str | None = None) -> None:
        c = self.c
        if version_id is not None:
            with c.db.transaction():
                fallback = _version_fallback(c, pid, version_id)
                if fallback is None:
                    raise ApiError("请通过删除整个项目移除最后一个可用版本。", code="version.last", status=409)
                from .routes_work import _preserve_tts_source
                from .xyz import dependent_tests, preserve_source

                jobs = c.db.fetchall("SELECT * FROM jobs WHERE version_id=?", (version_id,))
                for job in jobs:
                    _preserve_tts_source(c, job)
                    if job["type"] == "train":
                        preserve_source(c, job, [row for row in dependent_tests(c, job) if row.get("version_id") != version_id])
                for table, column in (("tts_checkpoints", "source_job_id"), ("tts_sample_audio", "sample_job_id")):
                    if c.db.fetchone("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)):
                        c.db.execute(f"DELETE FROM {table} WHERE {column} IN (SELECT id FROM jobs WHERE version_id=?)", (version_id,))
                c.db.execute("DELETE FROM artifacts WHERE version_id=? OR job_id IN (SELECT id FROM jobs WHERE version_id=?)",
                             (version_id, version_id))
                c.db.execute("DELETE FROM jobs WHERE version_id=?", (version_id,))
                for table in (*_VERSION_TABLES, "site_download_posts"):
                    if c.db.fetchone("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)):
                        c.db.execute(f"DELETE FROM {table} WHERE version_id=?", (version_id,))
                for row in c.db.fetchall("SELECT id FROM datasets WHERE version_id=?", (version_id,)):
                    c.db.execute("DELETE FROM kv WHERE key=?", ("dataset.edit_baseline." + row["id"],))
                c.db.execute("DELETE FROM datasets WHERE version_id=?", (version_id,))
                c.db.execute("UPDATE project_versions SET parent_version_id=NULL WHERE parent_version_id=?", (version_id,))
                c.db.execute("UPDATE projects SET active_version_id=?,updated_at=? WHERE id=? AND active_version_id=?",
                             (fallback["id"], now(), pid, version_id))
                c.db.execute("DELETE FROM version_deletions WHERE project_id=? AND version_id=?", (pid, version_id))
                c.db.delete("project_versions", version_id)
            return
        with c.db.lock:
            versions = [row["id"] for row in c.db.fetchall("SELECT id FROM project_versions WHERE project_id=?", (pid,))]
            c.db.execute("BEGIN IMMEDIATE")
            try:
                from .routes_work import _preserve_tts_source

                for source in c.db.fetchall("SELECT * FROM jobs WHERE project_id=? AND type='tts_train'", (pid,)):
                    _preserve_tts_source(c, source)
                for table, column in (("tts_checkpoints", "source_job_id"), ("tts_sample_audio", "sample_job_id")):
                    if c.db.fetchone("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)):
                        c.db.execute(f"DELETE FROM {table} WHERE {column} IN (SELECT id FROM jobs WHERE project_id=?)", (pid,))
                c.db.execute("DELETE FROM jobs WHERE project_id=?", (pid,))
                c.db.execute("DELETE FROM artifacts WHERE project_id=?", (pid,))
                for table in _VERSION_TABLES:
                    try:
                        c.db.execute(f"DELETE FROM {table} WHERE project_id=?", (pid,))
                    except sqlite3.OperationalError:
                        continue
                if versions:
                    marks = ",".join("?" * len(versions))
                    try:
                        c.db.execute(f"DELETE FROM site_download_posts WHERE version_id IN ({marks})", tuple(versions))
                    except sqlite3.OperationalError:
                        pass
                c.db.execute("DELETE FROM project_deletions WHERE project_id=?", (pid,))
                c.db.execute("DELETE FROM version_deletions WHERE project_id=?", (pid,))
                c.db.delete("projects", pid)
                c.db.execute("COMMIT")
            except BaseException:
                c.db.execute("ROLLBACK")
                raise


def deletions(c: Any) -> ProjectDeletions:
    service = getattr(c, "project_deletions", None)
    if service is None:
        service = c.project_deletions = ProjectDeletions(c)
    return service
