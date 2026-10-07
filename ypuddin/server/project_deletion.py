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
# Rows of other services that only describe the project's own versions.
_VERSION_TABLES = ("dataset_pipeline_operations", "regularization_operations", "site_downloads")


def deletion_state(c: Any, pid: str) -> dict | None:
    try:
        return c.db.fetchone("SELECT * FROM project_deletions WHERE project_id=?", (pid,))
    except sqlite3.OperationalError:  # a context whose deletion service never started
        return None


def deleting(c: Any, pid: str) -> bool:
    state = deletion_state(c, pid)
    return bool(state and state["state"] == "deleting")


def public_state(c: Any, pid: str) -> dict | None:
    state = deletion_state(c, pid)
    return {key: state[key] for key in ("state", "error", "task_id")} if state else None


@dataclass
class Location:
    path: Path
    kinds: list[str]
    custom: bool
    problem: str | None = None
    # Folders under a custom root that only existed for this project, removed once empty.
    parents: list[Path] = field(default_factory=list)


@dataclass
class Plan:
    project: dict
    locations: list[Location]
    records: list[Path]
    jobs: list[dict]
    blocked: ApiError | None = None


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


def plan(c: Any, pid: str, *, check_tts_refs: bool = True) -> Plan:
    """What deleting the project removes and what stops it; reads the disk but holds no lock."""
    from .routes_work import _get_project, _jobs_using

    project = _get_project(c, pid)
    data_root = c.data_root.resolve()
    project_root = c.project_dir(pid)
    jobs = c.db.fetchall("SELECT * FROM jobs WHERE project_id=? ORDER BY created_at", (pid,))
    versions = c.db.fetchall("SELECT id FROM project_versions WHERE project_id=?", (pid,))
    datasets = c.db.fetchall("SELECT id FROM datasets WHERE project_id=?", (pid,))
    candidates: dict[Path, list[str]] = {project_root: ["project"]}
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
    for folder, kinds in sorted(candidates.items(), key=lambda item: (item[0] != project_root, str(item[0]))):
        resolved = _resolved(folder)
        # A job folder inside the project folder goes with it.
        if resolved is not None and any(resolved.is_relative_to(parent) for parent in covered):
            continue
        problem = folder_problem(folder, roots) if resolved is not None else "unavailable"
        if problem in ("outside", "linked") and not os.path.lexists(folder):
            continue  # nothing left there to remove
        location = Location(folder, kinds, custom=resolved is None or not resolved.is_relative_to(data_root), problem=problem)
        if problem is None and resolved is not None:
            covered.append(resolved)
            root = max((r for r in roots if resolved.is_relative_to(r)), key=lambda r: len(r.parts))
            if root != data_root:
                location.parents = [parent for parent in folder.parents[: len(resolved.relative_to(root).parts) - 1]]
        locations.append(location)
    records = [c.data_root / "datasets" / f"{row['id']}.json" for row in datasets]
    result = Plan(project, locations, records, jobs)
    targets = [location.path for location in locations if location.problem is None]
    result.blocked = _blocking(c, pid, targets, covered) or _in_use(c, pid, targets, _jobs_using)
    if check_tts_refs and result.blocked is None:
        result.blocked = _tts_reference_blocker(c, pid, targets)
    return result


def _tts_reference_blocker(c: Any, pid: str, targets: list[Path]) -> ApiError | None:
    manager = getattr(c, "tts_sources", None)
    if manager is None:
        return None
    try:
        references = manager.references_to(targets, excluding_project_id=pid)
        from .tts_source_copy import active_references

        references += active_references(c, targets, excluding_project_id=pid)
    except ApiError as exc:
        return exc
    if references:
        return ApiError(
            "其他语音项目仍引用这些清单或录音，移除引用后才能删除。",
            code="project.files_in_use", status=409, details={"sources": references},
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


def _project_busy(c: Any, pid: str, project: dict) -> ApiError | None:
    """Conditions in the database that refuse any deletion of the project; caller holds the lock."""
    if c.db.fetchone("SELECT id FROM tts_sources WHERE project_id=? AND state='checking'", (pid,)):
        return ApiError("项目的数据来源正在检查，请等待检查完成。", code="project.busy", status=409)
    if c.db.fetchone(f"SELECT id FROM jobs WHERE project_id=? AND status IN {ACTIVE_JOBS}", (pid,)):
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
        os.chmod(path, stat.S_IWRITE)
        os.unlink(path)


class ProjectDeletions:
    def __init__(self, context: Any):
        self.c = context
        self.stopping = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="project-delete")
        self._running: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.c.db.execute(TABLE)
        for row in self.c.db.fetchall("SELECT project_id FROM project_deletions WHERE state='deleting'"):
            # The service stopped partway: continue with the locations the user confirmed.
            self._submit(row["project_id"], resumed=True)

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

    def storage(self, pid: str) -> dict:
        """Everything deleting the project removes from disk, for the confirm dialog."""
        from .routes_work import _folder_size

        found = plan(self.c, pid)
        locations = []
        for location in found.locations:
            exists = location.problem != "unavailable" and location.path.is_dir()
            size, files = _folder_size(location.path) if exists and location.problem is None else (0, 0)
            locations.append({
                "path": str(location.path), "kinds": location.kinds, "custom": location.custom,
                "exists": exists, "bytes": size, "files": files, "problem": location.problem,
            })
        with self.c.db.lock:
            blocked = _project_busy(self.c, pid, found.project) or found.blocked
        artifacts = self.c.db.fetchone("SELECT count(*) n FROM artifacts WHERE project_id=?", (pid,))["n"]
        return {
            "locations": locations,
            "total_bytes": sum(item["bytes"] for item in locations),
            "jobs": len(found.jobs),
            "artifacts": artifacts,
            "blocked": {
                "code": blocked.code, "message": blocked.message,
                "jobs": (blocked.details or {}).get("jobs", []),
            } if blocked else None,
            "deletion": public_state(self.c, pid),
        }

    def start(self, pid: str, skip: list[str]) -> str | None:
        """Mark the project as being deleted and remove its files in the background."""
        found = plan(self.c, pid)
        problems = [
            location for location in found.locations
            if location.problem and str(location.path) not in skip
        ]
        with self.c.db.lock:
            project = self.c.db.fetchone("SELECT * FROM projects WHERE id=?", (pid,))
            if project is None:
                raise NotFound(f"project {pid} not found", code="project.not_found")
            if error := _project_busy(self.c, pid, project):
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
            from .routes_work import _jobs_using

            targets = [location.path for location in found.locations if location.problem is None]
            if error := _tts_reference_blocker(self.c, pid, targets):
                raise error
            if error := _in_use(self.c, pid, targets, _jobs_using):
                raise error
            tasks = self.c.background_tasks
            task = (
                tasks.start("project_delete", project["name"], unit="files", link="/projects?archived=1")
                if tasks is not None else None
            )
            self.c.db.execute(
                "INSERT OR REPLACE INTO project_deletions"
                " (project_id,state,skipped_json,task_id,error,started_at,updated_at) VALUES (?,?,?,?,?,?,?)",
                (pid, "deleting", json.dumps(sorted(skip)), task, None, now(), now()),
            )
        self._submit(pid, found=found, task=task)
        return task

    def _submit(self, pid: str, *, found: Plan | None = None, task: str | None = None, resumed: bool = False) -> None:
        with self._lock:
            self._running[pid] = self.executor.submit(self._run, pid, found, task, resumed)

    def _run(self, pid: str, found: Plan | None, task: str | None, resumed: bool) -> None:
        tasks = self.c.background_tasks
        try:
            state = deletion_state(self.c, pid)
            if state is None or state["state"] != "deleting":
                return
            skipped = set(json.loads(state["skipped_json"] or "[]"))
            # Keep both a link's name and its destination, including paths through /tmp-style roots.
            kept_paths = tuple({
                path for value in skipped
                for path in (Path(os.path.abspath(Path(value).expanduser())), Path(value).expanduser().resolve())
            })
            if found is None:
                found = plan(self.c, pid)
                if found.blocked:
                    raise found.blocked
                if any(location.problem and str(location.path) not in skipped for location in found.locations):
                    raise ApiError("服务重启后，有位置无法删除，请重新确认。", code="project.path", status=409)
            if task is None and tasks is not None:
                task = tasks.start("project_delete", found.project["name"], unit="files", link="/projects?archived=1")
                with self.c.db.lock:
                    self.c.db.execute(
                        "UPDATE project_deletions SET task_id=?, updated_at=? WHERE project_id=?", (task, now(), pid)
                    )
            targets = [
                location for location in found.locations
                if location.problem is None and location.path.exists()
                and not any(location.path.resolve().is_relative_to(keep) for keep in kept_paths)
            ]
            total = sum(_count_files(location.path, kept_paths) for location in targets)
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
                self._remove(location.path, removed, kept_paths)
                for parent in location.parents:
                    try:
                        parent.rmdir()  # only an emptied folder made for this project
                    except OSError:
                        break
            for record in found.records:
                record.unlink(missing_ok=True)
            self._remove_rows(pid)
            if task is not None:
                tasks.finish(task)
            self.c.bus.publish("queue.changed", {})
        except InterruptedError:
            # Stopping the service leaves the deletion marked; the next start continues it.
            return
        except Exception as exc:  # noqa: BLE001 - the project stays archived for a retry
            message = exc.message if isinstance(exc, ApiError) else _failure(exc)
            if not isinstance(exc, ApiError):
                log.exception("project deletion failed")
            with self.c.db.lock:
                self.c.db.execute(
                    "UPDATE project_deletions SET state='failed', error=?, updated_at=? WHERE project_id=?",
                    (message, now(), pid),
                )
            if task is not None:
                tasks.finish(task, error=message)
        finally:
            with self._lock:
                self._running.pop(pid, None)

    @staticmethod
    def _remove(folder: Path, removed: Any, skipped: tuple[Path, ...] = ()) -> None:
        """Remove files without following links; keep skipped folders and the parents holding them."""
        folder = Path(os.path.abspath(folder))
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

    def _remove_rows(self, pid: str) -> None:
        c = self.c
        with c.db.lock:
            versions = [row["id"] for row in c.db.fetchall("SELECT id FROM project_versions WHERE project_id=?", (pid,))]
            c.db.execute("BEGIN IMMEDIATE")
            try:
                from .routes_work import _preserve_tts_source

                for source in c.db.fetchall("SELECT * FROM jobs WHERE project_id=? AND type='tts_train'", (pid,)):
                    _preserve_tts_source(c, source)
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
