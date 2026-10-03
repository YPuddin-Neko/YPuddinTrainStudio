"""User-requested source updates, handed to the owning launcher after preparation.

The service downloads an update and builds its frontend, then hands it to its launcher. The
launcher replaces the files in a separate process that keeps going when the launcher window
closes, installs the dependencies and starts the new version. The next launcher start finishes
or undoes an update that a closed window or a power loss interrupted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ypuddin.runtime_profiles import profile_root

from .errors import ApiError
from .lifecycle import update_blocked_message
from .network import ProxyPolicy
from .torch_environments import atomic_json
from .trainer_updates import local_version

UPDATE_ID_ENV = "YPUDDIN_TRAINER_UPDATE_ID"
BUSY = {"preparing", "downloading", "building", "applying", "installing", "restarting"}
HELPER_TIMEOUT = 1800
LOG_LINES = 80
# Finished updates whose backups and logs are kept for diagnosis; older ones are removed.
KEEP_UPDATES = 3
NODE_CHECK_SECONDS = 60
# A finished update stays in the task center this long; a failure of this run stays until dismissed.
TASK_KEEP_SECONDS = 600
_UNFINISHED = {"applying", "rolling_back"}
_UNREAD = object()
_STATE_TEXT = {
    "preparing": "正在准备更新", "downloading": "正在下载源码", "building": "正在构建前端",
    "applying": "正在替换源码", "installing": "正在安装依赖", "restarting": "正在重启训练器",
    "succeeded": "更新完成", "failed": "更新失败",
}


class TrainerInstallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    request_id: UUID


class TrainerInstallOperation(BaseModel):
    id: str
    target_commit: str
    previous_commit: str
    before_instance_id: str
    result_instance_id: str | None = None
    state: Literal["preparing", "downloading", "building", "applying", "installing", "restarting", "succeeded", "failed"]
    message: str
    error: str | None = None
    started_at: float
    updated_at: float
    log: list[str] = Field(default_factory=list)
    rolled_back: bool = False


class TrainerInstallStatus(BaseModel):
    can_apply: bool
    reason: str | None
    operation: TrainerInstallOperation | None = None
    running_commit: str | None
    instance_id: str


class UpdateRecoveryFailed(RuntimeError):
    """An interrupted update could be neither finished nor undone; the service must not start."""


def update_folder(data_root: Path) -> Path:
    return profile_root(data_root) / "service" / "trainer-updates"


def log_path(folder: Path, operation_id: str) -> Path:
    return folder / f"{operation_id}.log"


def _log_tail(folder: Path, operation_id: str) -> list[str] | None:
    """The last lines of an operation's log; a line starting with a carriage return replaces the one before."""
    try:
        with log_path(folder, operation_id).open("rb") as stream:
            size = stream.seek(0, os.SEEK_END)
            start = max(0, size - 256 * 1024)
            stream.seek(start)
            data = stream.read()
    except OSError:
        return None
    rows = data.decode("utf-8", errors="replace").split("\n")
    if start:
        rows = rows[1:]  # the first row may begin inside a line
    lines: list[str] = []
    for row in rows:
        if row.startswith("\r"):
            if lines:
                lines.pop()
            row = row[1:]
        if row:
            lines.append(row)
    return lines[-LOG_LINES:]


def _append(path: Path, text: str, mode: str = "a") -> None:
    # A virus scanner on Windows can hold a freshly written file for a moment.
    for attempt in range(20):
        try:
            with path.open(mode, encoding="utf-8", newline="\n") as stream:
                stream.write(text)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 19:
                raise
            time.sleep(0.05)


def append_log(folder: Path, operation_id: str, lines, *, replace_last: bool = False) -> None:
    """Add lines to an operation's log. The file is only ever appended to: the page and the
    launcher read it while it grows, which on Windows prevents replacing it."""
    text = "".join(
        ("\r" if replace_last and index == 0 else "") + " ".join(str(line).splitlines()) + "\n"
        for index, line in enumerate(lines)
    )
    if text:
        _append(log_path(folder, operation_id), text)


def _read_record(folder: Path) -> TrainerInstallOperation | None:
    try:
        return TrainerInstallOperation.model_validate_json((folder / "latest.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None


def read_operation(folder: Path) -> TrainerInstallOperation | None:
    operation = _read_record(folder)
    if operation is not None and (lines := _log_tail(folder, operation.id)) is not None:
        operation.log = lines
    return operation


def write_operation(folder: Path, **fields) -> TrainerInstallOperation:
    """Save an operation's state. Its log lines live in an append-only file: ``log`` starts the
    log of a new operation and is ignored for the current one."""
    previous = _read_record(folder)
    values = previous.model_dump() if previous else {}
    lines = fields.pop("log", None)
    values.update(fields, updated_at=time.time(), log=[])
    operation = TrainerInstallOperation.model_validate(values)
    if lines is not None and (previous is None or previous.id != operation.id):
        _append(log_path(folder, operation.id), "".join(" ".join(str(line).splitlines()) + "\n" for line in lines), "w")
    atomic_json(folder / "latest.json", operation.model_dump())
    tail = _log_tail(folder, operation.id)
    return operation.model_copy(update={"log": tail}) if tail is not None else operation


def record_update_progress(folder: Path, message: str, *, replace_last: bool = False, **fields) -> None:
    operation = _read_record(folder)
    assert operation is not None
    append_log(folder, operation.id, [f"[studio] {message}"], replace_last=replace_last)
    if fields or not replace_last:
        write_operation(folder, message=message, **fields)


def source_progress_callback(folder: Path):
    last_download = None
    last_written_at = 0.0

    def progress(event):
        nonlocal last_download, last_written_at
        phase = event["phase"]
        archive = event.get("archive")
        label = "当前版本源码" if archive == "baseline.zip" else "新版本源码"
        if phase in {"downloading", "downloaded"}:
            now = time.monotonic()
            same_download = last_download is not None and last_download == archive
            if phase == "downloading" and same_download and now - last_written_at < 1:
                return
            completed = int(event.get("completed", 0))
            message = f"{label}下载完成：{completed / 1024 / 1024:.1f} MiB" if phase == "downloaded" else (
                f"正在下载{label}：{completed / 1024 / 1024:.1f} MiB" if completed else f"正在下载{label}"
            )
            record_update_progress(folder, message, replace_last=same_download,
                                   **({} if same_download else {"state": "downloading"}))
            last_download, last_written_at = archive, now
            return
        messages = {
            "checking": "正在检查本地源码",
            "extracting": f"正在解压{label}",
            "verifying": "正在校验源码文件",
            "fetching": "正在同步 Git 版本记录",
            "prepared": "源码准备完成",
            "ready": "源码与前端构建校验完成",
        }
        if phase in messages:
            record_update_progress(folder, messages[phase], **({"state": "preparing"} if phase == "checking" else {}))
            last_download = None

    return progress


def plan_phase(work: Path) -> str | None:
    try:
        phase = json.loads((work / "plan.json").read_text("utf-8")).get("phase")
    except (OSError, ValueError, AttributeError):
        return None
    return phase if isinstance(phase, str) else None


def discard_preparation(folder: Path, operation_id: str) -> None:
    """Remove the downloads and staged copies of an update that is finished either way."""
    work = folder / str(UUID(operation_id))
    if plan_phase(work) in _UNFINISHED:
        return  # finishing or undoing it still needs them
    for name in ("baseline", "staged"):
        path = work / name
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
    for name in ("baseline.zip", "target.zip"):
        try:
            (work / name).unlink(missing_ok=True)
        except OSError:
            pass


def prune_updates(folder: Path, keep: int = KEEP_UPDATES) -> None:
    """Keep the backups and logs of the newest few updates; remove older ones."""
    current = _read_record(folder)
    updates: dict[str, float] = {}
    try:
        entries = list(folder.iterdir())
    except OSError:
        return
    for entry in entries:
        name = entry.name.removesuffix(".log") if entry.is_file() else entry.name
        try:
            identifier = str(UUID(name))
        except ValueError:
            continue
        if identifier != name or entry.is_symlink():
            continue
        try:
            updates[identifier] = max(updates.get(identifier, 0.0), entry.stat().st_mtime)
        except OSError:
            continue
    removable = sorted(updates, key=updates.get, reverse=True)[keep:]
    for identifier in removable:
        work = folder / identifier
        if current is not None and current.id == identifier or plan_phase(work) in _UNFINISHED or update_running(work):
            continue
        if work.is_dir():
            shutil.rmtree(work, ignore_errors=True)
        try:
            log_path(folder, identifier).unlink(missing_ok=True)
        except OSError:
            pass


def update_running(work: Path) -> bool:
    """Whether an apply or recovery of this update is running in some process now."""
    lock = work / "apply.lock"
    if not lock.is_file():
        return False
    try:
        with lock.open("a+b") as stream:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    except OSError:
        return True
    return False


def helper_command(helper: Path, stage: str, root: Path, python: str, work: Path) -> list[str]:
    return [python, str(helper), stage, "--root", str(root), "--settings-file", str(work / "settings.json"),
            "--work-dir", str(work), "--parent-pid", str(os.getpid())]


def redacted_log_lines(stream, policy: ProxyPolicy):
    dropping_line = False
    for line in iter(lambda: stream.readline(8192), ""):
        oversized = len(line) == 8192 and not line.endswith("\n")
        if oversized or dropping_line:
            already_dropping = dropping_line
            dropping_line = oversized
            if already_dropping:
                continue
            line = "[studio] 过长的日志行已省略。"
        line = policy.redact(line.strip())[-2000:]
        if line:
            yield line


class WorkerStartupLog:
    """Drain the worker pipe for its lifetime; retain only bounded startup output."""

    def __init__(self, child, policy: ProxyPolicy):
        self.lines: deque[str] = deque(maxlen=80)
        self.lock = threading.Lock()
        self.capturing = True
        self.reader = threading.Thread(target=self._read, args=(child.stdout, policy), daemon=True)
        self.reader.start()

    def _read(self, stream, policy: ProxyPolicy) -> None:
        try:
            for line in redacted_log_lines(stream, policy):
                with self.lock:
                    if self.capturing:
                        self.lines.append(line)
                print(line, flush=True)
        finally:
            stream.close()

    def finish(self) -> list[str]:
        # A descendant may still own the pipe after the HTTP worker exits.
        self.reader.join(timeout=1)
        with self.lock:
            self.capturing = False
            return list(self.lines)

    def release(self) -> None:
        with self.lock:
            self.capturing = False
            self.lines.clear()


def run_helper(folder: Path, stage: str, root: Path, python: str, policy: ProxyPolicy, cancel=None,
               *, log_name: str | None = None, record_log: bool = True) -> None:
    operation = _read_record(folder)
    assert operation is not None
    work = folder / operation.id
    env = policy.subprocess_env()
    env.pop(UPDATE_ID_ENV, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    output = work / (log_name or f"{stage}.log")
    # Installer output stays bounded on disk and is redacted before reaching the UI.
    with output.open("w", encoding="utf-8") as stream:
        # The helper leads its own process group so a stop ends npm and pip too; it also ends
        # itself if this process goes away first.
        child = subprocess.Popen(helper_command(work / "helper" / "scripts" / "update_prepare.py", stage, root, python, work),
                                 cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace", start_new_session=os.name != "nt")
        deadline = time.monotonic() + HELPER_TIMEOUT
        timed_out = threading.Event()
        finished = threading.Event()
        descendants = {}
        def stop():
            timed_out.set()
            if os.name == "nt":
                import psutil
                if child.poll() is None:
                    try:
                        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=10)
                    except (OSError, subprocess.TimeoutExpired):
                        child.kill()
                for descendant in list(descendants.values()):
                    try:
                        if descendant.is_running():
                            descendant.kill()
                    except psutil.Error:
                        pass
            else:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        def guard():
            while not finished.is_set():
                if os.name == "nt" and child.poll() is None:
                    import psutil
                    try:
                        for descendant in psutil.Process(child.pid).children(recursive=True):
                            descendants[(descendant.pid, descendant.create_time())] = descendant
                    except psutil.Error:
                        pass
                if (cancel is not None and cancel.is_set()) or time.monotonic() > deadline:
                    stop()
                    return
                time.sleep(0.2)
        watcher = threading.Thread(target=guard, daemon=True)
        watcher.start()
        completed = False
        tail: deque[str] = deque(maxlen=8)
        try:
            assert child.stdout is not None
            for line in redacted_log_lines(child.stdout, policy):
                tail.append(line)
                print(line, flush=True)
                if stream.tell() < 2 * 1024 * 1024:
                    stream.write(line + "\n")
                if record_log:
                    append_log(folder, operation.id, [line])
                if time.monotonic() > deadline:
                    stop()
            code = child.wait()
            if code or timed_out.is_set():
                label = "前端构建失败" if stage == "build" else "训练器依赖安装失败"
                stopped = cancel is not None and cancel.is_set()
                detail = "已停止" if stopped else "等待超时" if timed_out.is_set() else f"退出码 {code}"
                output_tail = "\n".join(tail)[-4000:]
                raise RuntimeError(f"{label}（{detail}）。" + ("\n" + output_tail if output_tail else ""))
            completed = True
        finally:
            finished.set()
            if not completed:
                stop()
            if child.poll() is None:
                child.wait()


def detached_options() -> list[dict]:
    """Ways to start a step that must finish after this console closes, most independent first."""
    if os.name != "nt":
        return [{"start_new_session": True}]
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    # Breaking away from the console's job fails where the job forbids it.
    return [{"creationflags": flags | subprocess.CREATE_BREAKAWAY_FROM_JOB}, {"creationflags": flags}]


def update_helper(work: Path) -> Path:
    """The helper that applies or recovers this update: the one saved with it, or for an update
    prepared by a release without recovery, the verified copy in its staged files."""
    saved = work / "helper" / "scripts" / "update_prepare.py"
    try:
        if '"recover"' in saved.read_text(encoding="utf-8"):
            return saved
        plan = json.loads((work / "plan.json").read_text("utf-8"))
        staged = Path(plan["staged_root"])
        for name in ("scripts/update_prepare.py", "ypuddin/server/source_update.py"):
            digest = hashlib.sha256((staged / name).read_bytes()).hexdigest()
            if digest != plan["after"].get(name):
                return saved
        return staged / "scripts" / "update_prepare.py"
    except (OSError, ValueError, KeyError, TypeError):
        return saved


def run_detached(folder: Path, operation_id: str, action: str, python: str, *,
                 record_log: bool = True) -> tuple[int, list[str]]:
    """Run ``apply`` or ``recover`` of the helper saved with the update in a process of its own,
    which keeps going when this console closes, and show its output meanwhile."""
    work = folder / operation_id
    output_path = work / f"{action}.log"
    command = [python, str(update_helper(work)), action, "--work-dir", str(work)]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
    env.pop(UPDATE_ID_ENV, None)
    with output_path.open("ab") as output:
        start = output.tell()
        options = detached_options()
        for index, extra in enumerate(options):
            try:
                child = subprocess.Popen(command, cwd=work, env=env, stdin=subprocess.DEVNULL, stdout=output,
                                         stderr=subprocess.STDOUT, **extra)
                break
            except OSError:
                if index + 1 == len(options):
                    raise
    lines: list[str] = []
    pending = b""

    def show(raw: bytes) -> None:
        line = raw.decode("utf-8", errors="replace").strip()
        if line:
            lines.append(line)
            print(line, flush=True)
            if record_log:
                append_log(folder, operation_id, [line])

    try:
        with output_path.open("rb") as stream:
            stream.seek(start)
            while True:
                chunk = stream.read(64 * 1024)
                if chunk:
                    *complete, pending = (pending + chunk).split(b"\n")
                    for raw in complete:
                        show(raw)
                    continue
                if child.poll() is not None:
                    *complete, pending = (pending + stream.read()).split(b"\n")
                    for raw in [*complete, pending]:
                        show(raw)
                    return child.returncode, lines
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("[studio] 源码替换会在后台完成，下次启动训练器时运行新版本。", flush=True)
        raise


class TrainerInstaller:
    def __init__(self, context, updates, lifecycle):
        self.context, self.updates, self.lifecycle = context, updates, lifecycle
        self.root = updates.root
        self.folder = update_folder(context.data_root)
        self.lock = threading.RLock()
        self.worker: threading.Thread | None = None
        self.cancel = threading.Event()
        self.starting_id = os.environ.pop(UPDATE_ID_ENV, None)
        self.previous_maintenance: dict = {}
        self._failed_operation: TrainerInstallOperation | None = None
        self._unfinished = False
        self._node: tuple[float, str | None] | None = None
        self._record_cache: tuple[tuple[int, int, int], TrainerInstallOperation | None] | None = None
        self._started_at = time.time()
        operation = read_operation(self.folder)
        if operation and plan_phase(self.folder / operation.id) in _UNFINISHED:
            # The launcher finishes or undoes it on its next start; keep the record for it.
            self._unfinished = True
            self._failed_operation = operation.model_copy(update={
                "state": "failed", "message": "上次更新尚未完成",
                "error": "源码替换中断后尚未完成。请关闭训练器，再用启动脚本重新启动，以完成或撤销这次更新。",
            })
        elif operation and operation.state in BUSY:
            if self.starting_id == operation.id:
                self.context.db.set_kv("environment.maintenance", {"blocked": True, "restarting": True, "trainer_update": operation.id})
            else:
                self._failed_operation = operation.model_copy(update={
                    "state": "failed", "message": "更新被中断", "updated_at": time.time(),
                    "error": operation.error or "更新被中断，请查看启动终端后重试。",
                })
                try:
                    write_operation(self.folder, **self._failed_operation.model_dump())
                    discard_preparation(self.folder, operation.id)
                except (OSError, ValueError) as exc:
                    print("[studio] 无法保存中断的更新记录：" + ProxyPolicy().redact(exc), flush=True)
        tasks = getattr(context, "background_tasks", None)
        if tasks is not None:
            tasks.add_source(self.background_tasks)

    def _operation(self) -> TrainerInstallOperation | None:
        return self._failed_operation or read_operation(self.folder)

    def node_problem(self) -> str | None:
        """Why this machine cannot build the interface of an update: node_missing, node_unsupported or None."""
        if self._node and time.monotonic() - self._node[0] < NODE_CHECK_SECONDS:
            return self._node[1]
        node = shutil.which("node")
        npm = shutil.which("npm") or shutil.which("npm.cmd")
        problem = None
        if not node or not npm:
            problem = "node_missing"
        else:
            try:
                version = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=10).stdout
            except (OSError, subprocess.SubprocessError):
                version = ""
            if not node_supported(version.strip()):
                problem = "node_unsupported"
        self._node = (time.monotonic(), problem)
        return problem

    def _reason(self, disk=None, background=_UNREAD, node=_UNREAD) -> str | None:
        """Why an update cannot start, or None. Under the database lock, pass the Git check,
        ``lifecycle.background_reason()`` and ``node_problem()`` read before taking it."""
        if self._unfinished:
            return "update_unfinished"
        operation = self._operation()
        if operation and operation.state in BUSY:
            return "update_in_progress"
        reason = self.lifecycle._blocked() if background is _UNREAD else self.lifecycle._blocked(background)
        if reason:
            return reason
        if getattr(self.context, "_update_requests", 0):
            return "data_operation_running"
        current = self.updates.current
        if not current.commit:
            return "current_version_unknown"
        disk = disk or local_version(self.root)
        if disk.commit != current.commit:
            return "version_changed"
        if disk.dirty is True or (disk.source == "git" and disk.dirty is None):
            return "local_changes"
        # Every update builds the new interface before anything is replaced.
        return self.node_problem() if node is _UNREAD else node

    def _status(self, reason: str | None) -> TrainerInstallStatus:
        return TrainerInstallStatus(can_apply=reason is None, reason=reason,
                                    operation=self._operation(), running_commit=self.updates.current.commit,
                                    instance_id=self.lifecycle.instance_id)

    def status(self) -> TrainerInstallStatus:
        with self.lock:
            return self._status(self._reason())

    def _record(self) -> TrainerInstallOperation | None:
        """The saved operation without its log; the small record file is read again only after it changed."""
        if self._failed_operation is not None:
            return self._failed_operation
        try:
            info = (self.folder / "latest.json").stat()
        except OSError:
            return None
        stamp = (info.st_ino, info.st_mtime_ns, info.st_size)
        cached = self._record_cache
        if cached is None or cached[0] != stamp:
            cached = self._record_cache = (stamp, _read_record(self.folder))
        return cached[1]

    def background_tasks(self) -> list[dict]:
        """The update as a task-center entry while it runs and for a while after it ends.
        Read every second by the task center, so it does not read the update log."""
        operation = self._record()
        if operation is None:
            return []
        running = operation.state in BUSY and not self._unfinished
        failed_now = operation.state == "failed" and operation.updated_at >= self._started_at
        if not running and not failed_now and time.time() - operation.updated_at > TASK_KEEP_SECONDS:
            return []
        return [{
            "id": f"trainer-update-{operation.id}", "kind": "trainer_update", "subject": operation.target_commit[:8],
            "state": "running" if running else "completed" if operation.state == "succeeded" else "failed",
            "done": None, "total": None, "unit": None, "detail": _STATE_TEXT.get(operation.state, operation.message),
            "link": "/settings/updates", "cancellable": False, "started_at": operation.started_at,
            "finished_at": None if running else operation.updated_at,
            "error": operation.error if operation.state == "failed" else None,
        }]

    def start(self, request: TrainerInstallRequest) -> TrainerInstallStatus:
        # Settings, Git, Node.js and other services' work are read before the locks below:
        # holding the database lock while waiting for them stalls or deadlocks the service.
        checked = self.updates.status()
        disk = local_version(self.root)
        background = self.lifecycle.background_reason()
        node = self.node_problem()
        try:
            policy, policy_error = ProxyPolicy.from_context(self.context), None
        except Exception as exc:  # recorded as the reason the update did not start
            policy, policy_error = ProxyPolicy(), exc
        with self.lock, self.lifecycle.lock, self.lifecycle.environment.lock, self.lifecycle.torch.lock, self.context.db.lock:
            existing = self._operation()
            if existing and existing.id == str(request.request_id):
                if existing.target_commit != request.target_commit:
                    raise ApiError("同一次更新不能更换目标版本。", status=409)
                return self._status(self._reason(disk, background, node))
            reason = self._reason(disk, background, node)
            if reason:
                raise ApiError("当前无法更新训练器，请先处理页面提示。", code="updates.blocked", status=409, details={"reason": reason})
            if checked.state != "available" or not checked.latest or checked.latest.commit != request.target_commit:
                raise ApiError("请先检查更新，再选择检测到的版本。", code="updates.check_required", status=409)
            self.previous_maintenance = self.context.db.get_kv("environment.maintenance", {})
            fields = dict(id=str(request.request_id), target_commit=request.target_commit,
                          previous_commit=self.updates.current.commit, before_instance_id=self.lifecycle.instance_id,
                          result_instance_id=None, state="preparing", message="正在准备更新", error=None,
                          started_at=time.time(), log=["[studio] 正在准备更新"], rolled_back=False)
            maintenance_set = False
            try:
                if policy_error is not None:
                    raise policy_error
                self.folder.mkdir(parents=True, exist_ok=True)
                write_operation(self.folder, **fields)
                self._failed_operation = None
                self.context.db.set_kv("environment.maintenance", {"blocked": True, "restarting": True, "trainer_update": str(request.request_id)})
                maintenance_set = True
                self.lifecycle.updating = True
                self.worker = threading.Thread(target=self._prepare, name="trainer-update", daemon=False)
                self.cancel.clear()
                self.worker.start()
            except Exception as exc:
                self.lifecycle.updating = False
                self.worker = None
                reason = policy.redact(exc)
                if maintenance_set:
                    try:
                        self.context.db.set_kv("environment.maintenance", self.previous_maintenance)
                    except Exception as restore_error:
                        reason += "；恢复维护状态失败：" + policy.redact(restore_error)
                fields.update(state="failed", message="更新未能开始", error=reason, updated_at=time.time())
                self._failed_operation = TrainerInstallOperation.model_validate(fields)
                try:
                    write_operation(self.folder, **fields)
                except Exception as save_error:
                    print("[studio] 无法保存更新失败记录：" + policy.redact(save_error), flush=True)
                raise ApiError("训练器更新未能开始：" + reason, code="updates.start_failed", status=500,
                               details={"reason": reason}) from exc
            return self._status("update_in_progress")

    def _prepare(self):
        policy = ProxyPolicy()
        operation = self._operation()
        try:
            policy = ProxyPolicy.from_context(self.context)
            from scripts.update_prepare import snapshot_helpers

            from .source_update import prepare_update, seal_update
            assert operation is not None
            work = self.folder / operation.id
            work.mkdir(parents=True, exist_ok=True)
            progress = source_progress_callback(self.folder)
            write_operation(self.folder, state="preparing", message="正在准备更新")
            plan = prepare_update(self.root, work, operation.previous_commit, operation.target_commit,
                                  self.updates.current.source, policy, progress, cancel=self.cancel)
            staged = Path(plan["staged_root"])
            if self.cancel.is_set():
                raise RuntimeError("更新已取消。")
            settings = self.context.settings()
            atomic_json(work / "settings.json", {"downloads": settings.get("downloads", {}), "paths": settings.get("paths", {})})
            snapshot_helpers(work)
            record_update_progress(self.folder, "正在构建前端", state="building")
            run_helper(self.folder, "build", staged, sys.executable, policy, self.cancel)
            seal_update(work / "plan.json", progress)
            background = self.lifecycle.background_reason()
            with self.lock, self.lifecycle.lock, self.context.db.lock:
                # Work may have started while the update was prepared; it keeps the service running.
                if reason := self.lifecycle._blocked(background, updating=True):
                    raise RuntimeError(update_blocked_message(reason))
                record_update_progress(self.folder, "正在安装并重启训练器", state="restarting")
                self.lifecycle.request_update(operation.id, background)
        except Exception as exc:
            with self.lock, self.lifecycle.lock, self.context.db.lock:
                reason = "训练器正在关闭，已停止本次更新。" if self.cancel.is_set() else policy.redact(exc)
                self.lifecycle.updating = False
                try:
                    self.context.db.set_kv("environment.maintenance", self.previous_maintenance)
                except Exception as restore_error:
                    reason += "；恢复维护状态失败：" + policy.redact(restore_error)
                operation = read_operation(self.folder) or operation
                if operation:
                    self._failed_operation = operation.model_copy(update={
                        "state": "failed", "message": "更新准备失败", "error": reason, "updated_at": time.time(),
                    })
                    try:
                        write_operation(self.folder, **self._failed_operation.model_dump())
                    except Exception as save_error:
                        print("[studio] 无法保存更新失败记录：" + policy.redact(save_error), flush=True)
            if operation:
                discard_preparation(self.folder, operation.id)
                prune_updates(self.folder)

    def ready(self) -> None:
        """Called only after Uvicorn has successfully bound its listening sockets."""
        operation = read_operation(self.folder)
        if not operation or operation.id != self.starting_id:
            return
        if operation.state != "restarting" or self.updates.current.commit != operation.target_commit:
            return
        if self.lifecycle.instance_id == operation.before_instance_id:
            return
        with self.context.db.lock:
            record_update_progress(self.folder, "更新完成", state="succeeded", result_instance_id=self.lifecycle.instance_id)
            self.context.db.set_kv("environment.maintenance", {"blocked": False})

        def tidy():
            discard_preparation(self.folder, operation.id)
            prune_updates(self.folder)

        threading.Thread(target=tidy, daemon=True).start()

    def close(self):
        if self.worker and self.worker.is_alive():
            if not self.lifecycle.restarting:
                self.cancel.set()  # downloads, Git and the frontend build stop within moments
            self.worker.join()


def node_supported(version: str) -> bool:
    """Node.js releases that build the frontend; the same floor as scripts/bootstrap.py."""
    match = re.match(r"v?(\d+)\.(\d+)", version)
    if not match:
        return False
    major, minor = map(int, match.groups())
    return (major == 20 and minor >= 19) or (major == 22 and minor >= 12) or major >= 23


def launcher_policy(data_root: Path) -> ProxyPolicy:
    try:
        settings = json.loads((data_root / "settings.json").read_text("utf-8"))
    except FileNotFoundError:
        settings = {}
    settings["network"] = {"proxy_mode": "system", "proxy_url": "", "proxy_username": "", **settings.get("network", {})}
    context = SimpleNamespace(data_root=data_root, _settings_lock=threading.RLock(), settings=lambda: settings)
    return ProxyPolicy.from_context(context)


def _repair_dependencies(folder: Path, work: Path, python: str, data_root: Path) -> None:
    """Reinstall the restored version's package metadata and requirements."""
    plan = json.loads((work / "plan.json").read_text("utf-8"))
    run_helper(folder, "deps", Path(plan["root"]), python, launcher_policy(data_root),
               log_name="rollback-deps.log", record_log=False)
    (work / "repair-dependencies").unlink(missing_ok=True)


def rollback_from_launcher(data_root: Path, python: str, reason: str, *, repair_dependencies: bool = True,
                           startup_log: list[str] | None = None) -> bool:
    from .source_update import rollback_update
    folder = update_folder(data_root)
    operation = read_operation(folder)
    assert operation is not None
    work = folder / operation.id
    policy = ProxyPolicy()
    restored = False
    dependencies_ready = not repair_dependencies
    if startup_log:
        reason += "\n" + "\n".join(startup_log[-8:])[-4000:]
    try:
        append_log(folder, operation.id, startup_log or [])
        write_operation(folder, state="applying", message="更新失败，正在恢复原版源码", error=reason)
    except (OSError, ValueError) as exc:
        print("[studio] 无法保存更新失败记录：" + policy.redact(exc), flush=True)
    if startup_log:
        try:
            (work / "startup.log").write_text("\n".join(startup_log) + "\n", encoding="utf-8")
        except OSError as exc:
            reason += "\n启动日志文件保存失败：" + policy.redact(exc)
    try:
        if repair_dependencies:
            # A start after an interruption still repairs them once the files are restored.
            (work / "repair-dependencies").touch()
        rollback_update(work / "plan.json", python)
        restored = True
        if repair_dependencies:
            policy = launcher_policy(data_root)
            _repair_dependencies(folder, work, python, data_root)
            dependencies_ready = True
    except Exception as exc:
        reason += " 恢复检查失败：" + policy.redact(exc)
    try:
        write_operation(folder, state="failed", message="更新失败，已恢复原版源码" if restored else "更新失败",
                        error=reason, rolled_back=restored)
    except (OSError, ValueError) as exc:
        print("[studio] " + policy.redact(reason), flush=True)
        print("[studio] 无法保存更新失败记录：" + policy.redact(exc), flush=True)
    discard_preparation(folder, operation.id)
    return restored and dependencies_ready


def _install_dependencies(data_root: Path, folder: Path, operation_id: str, python: str) -> None:
    plan = json.loads((folder / operation_id / "plan.json").read_text("utf-8"))
    record_update_progress(folder, "正在检查并安装依赖", state="installing")
    run_helper(folder, "deps", Path(plan["root"]), python, launcher_policy(data_root))
    record_update_progress(folder, "正在启动新版本", state="restarting")


def apply_from_launcher(data_root: Path, update_id: str, python: str) -> bool | None:
    """Replace the files of a prepared update and install its dependencies.

    True: start the new version; False: not installed, the previous version runs; None: the
    previous files could not be restored and the service must not start.
    """
    folder = update_folder(data_root)
    operation = read_operation(folder)
    if not operation or operation.id != update_id or operation.state != "restarting":
        return False
    work = folder / operation.id
    policy = ProxyPolicy()
    try:
        policy = launcher_policy(data_root)
        record_update_progress(folder, "正在替换源码", state="applying")
        _code, lines = run_detached(folder, operation.id, "apply", python)
        phase = plan_phase(work)
        if phase == "rolled_back":
            # The helper restored the previous files; its last line names the cause.
            write_operation(folder, state="failed", message="更新未能安装，已恢复原版源码", rolled_back=True,
                            error=(lines[-1].removeprefix("[studio] ") if lines else "源码替换失败。"))
            discard_preparation(folder, operation.id)
            return False
        if phase != "applied":
            raise RuntimeError(lines[-1].removeprefix("[studio] ") if lines else "源码替换没有完成。")
        _install_dependencies(data_root, folder, operation.id, python)
        return True
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        phase = plan_phase(work)
        if phase is None:
            write_operation(folder, state="failed", message="无法读取更新记录", error=policy.redact(exc))
            return None
        if phase in {"ready", "prepared"}:
            write_operation(folder, state="failed", message="更新未应用", error=policy.redact(exc))
            discard_preparation(folder, operation.id)
            return False
        if phase in _UNFINISHED:
            # The apply process stopped part-way: finish or undo it from the plan first.
            run_detached(folder, operation.id, "recover", python)
            if plan_phase(work) in _UNFINISHED:
                write_operation(folder, state="failed", message="更新失败", error=policy.redact(exc))
                return None
        operation = read_operation(folder)
        restored = rollback_from_launcher(data_root, python, policy.redact(exc),
                                          repair_dependencies=bool(operation and operation.state == "installing"))
        return False if restored else None


def resume_from_launcher(data_root: Path, python: str) -> str | None:
    """Continue an update that a closed window or a power loss interrupted, before the service starts.

    Returns the id of an update the next service start completes. Raises UpdateRecoveryFailed
    when the installed files can be neither finished nor restored.
    """
    folder = update_folder(data_root)
    operation = read_operation(folder)
    if operation is None:
        return None
    work = folder / operation.id
    phase = plan_phase(work)
    if phase in _UNFINISHED:
        print("[studio] 检测到上次中断的训练器更新，正在完成或撤销源码替换", flush=True)
        append_log(folder, operation.id, ["[studio] 训练器重新启动，正在完成或撤销中断的源码替换"])
        run_detached(folder, operation.id, "recover", python)
        phase = plan_phase(work)
        if phase in _UNFINISHED:
            raise UpdateRecoveryFailed(f"无法完成或撤销上次中断的训练器更新，已停止启动。备份与更新日志保存在 {work}。")
    if phase == "rolled_back" and (work / "repair-dependencies").exists():
        try:
            _repair_dependencies(folder, work, python, data_root)
        except Exception as exc:
            write_operation(folder, state="failed", message="更新失败", rolled_back=True,
                            error="已恢复原版源码，但依赖修复失败：" + ProxyPolicy().redact(exc))
            raise UpdateRecoveryFailed("已恢复原版源码，但依赖修复失败，已停止启动。请查看上方日志后重新运行启动脚本。") from exc
    if operation.state not in BUSY:
        if phase == "rolled_back" and not operation.rolled_back:
            write_operation(folder, message="更新失败，已恢复原版源码", rolled_back=True)
        discard_preparation(folder, operation.id)
        prune_updates(folder)
        return None
    if operation.state in {"preparing", "downloading", "building"} or phase in {None, "prepared"}:
        write_operation(folder, state="failed", message="更新被中断", error="训练器在准备更新时停止，未安装更新。")
        discard_preparation(folder, operation.id)
        return None
    if phase == "rolled_back":
        write_operation(folder, state="failed", message="更新被中断，已恢复原版源码", rolled_back=True,
                        error=operation.error or "源码替换被中断，已恢复原版源码。")
        discard_preparation(folder, operation.id)
        return None
    if phase == "ready":
        # Handed over, but nothing was replaced yet: install it now.
        write_operation(folder, state="restarting")
        outcome = apply_from_launcher(data_root, operation.id, python)
        if outcome is None:
            raise UpdateRecoveryFailed(f"更新恢复检查失败，已停止启动。备份与更新日志保存在 {work}。")
        return operation.id if outcome else None
    # The files are new; the dependencies may still need installing.
    try:
        if operation.state in {"applying", "installing"}:
            _install_dependencies(data_root, folder, operation.id, python)
    except Exception as exc:
        restored = rollback_from_launcher(data_root, python, ProxyPolicy().redact(exc))
        if not restored:
            raise UpdateRecoveryFailed(f"更新恢复检查失败，已停止启动。备份与更新日志保存在 {work}。") from exc
        return None
    return operation.id


def wait_for_updated_worker(child, data_root: Path, update_id: str, timeout: float = 120) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if child.poll() is not None:
            return False
        operation = read_operation(update_folder(data_root))
        if operation and operation.id == update_id and operation.state == "succeeded":
            return bool(operation.result_instance_id and operation.result_instance_id != operation.before_instance_id)
        time.sleep(0.2)
    return False
