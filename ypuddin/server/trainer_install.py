"""User-requested source updates, handed to the owning launcher after preparation."""

from __future__ import annotations

import json
import os
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
from .network import ProxyPolicy
from .torch_environments import atomic_json
from .trainer_updates import local_version

UPDATE_ID_ENV = "YPUDDIN_TRAINER_UPDATE_ID"
BUSY = {"preparing", "downloading", "building", "applying", "installing", "restarting"}
HELPER_TIMEOUT = 1800


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


def update_folder(data_root: Path) -> Path:
    return profile_root(data_root) / "service" / "trainer-updates"


def read_operation(folder: Path) -> TrainerInstallOperation | None:
    try:
        return TrainerInstallOperation.model_validate_json((folder / "latest.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None


def write_operation(folder: Path, **fields) -> TrainerInstallOperation:
    previous = read_operation(folder)
    values = previous.model_dump() if previous else {}
    values.update(fields, updated_at=time.time())
    operation = TrainerInstallOperation.model_validate(values)
    atomic_json(folder / "latest.json", operation.model_dump())
    return operation


def discard_preparation(folder: Path, operation_id: str) -> None:
    work = folder / str(UUID(operation_id))
    for name in ("baseline", "staged"):
        path = work / name
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
    for name in ("baseline.zip", "target.zip"):
        try:
            (work / name).unlink(missing_ok=True)
        except OSError:
            pass


def helper_command(helper: Path, stage: str, root: Path, python: str, work: Path) -> list[str]:
    return [python, str(helper), stage, "--root", str(root), "--settings-file", str(work / "settings.json"), "--work-dir", str(work)]


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
    operation = read_operation(folder)
    assert operation is not None
    work = folder / operation.id
    env = policy.subprocess_env()
    env.pop(UPDATE_ID_ENV, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    output = work / (log_name or f"{stage}.log")
    # Installer output stays bounded on disk and is redacted before reaching the UI.
    with output.open("w", encoding="utf-8") as stream:
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
                    op = read_operation(folder)
                    write_operation(folder, log=[*(op.log if op else []), line][-80:])
                if time.monotonic() > deadline:
                    stop()
            code = child.wait()
            if code or timed_out.is_set():
                label = "前端构建失败" if stage == "build" else "训练器依赖安装失败"
                detail = "等待超时或已停止" if timed_out.is_set() else f"退出码 {code}"
                output_tail = "\n".join(tail)[-4000:]
                raise RuntimeError(f"{label}（{detail}）。" + ("\n" + output_tail if output_tail else ""))
            completed = True
        finally:
            finished.set()
            if not completed:
                stop()
            if child.poll() is None:
                child.wait()


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
        operation = read_operation(self.folder)
        if operation and operation.state in BUSY:
            if self.starting_id == operation.id:
                self.context.db.set_kv("environment.maintenance", {"blocked": True, "restarting": True, "trainer_update": operation.id})
            else:
                self._failed_operation = operation.model_copy(update={
                    "state": "failed", "message": "更新被中断", "updated_at": time.time(),
                    "error": operation.error or "更新被中断，请查看启动终端后重试。",
                })
                try:
                    write_operation(self.folder, **self._failed_operation.model_dump())
                except (OSError, ValueError) as exc:
                    print("[studio] 无法保存中断的更新记录：" + ProxyPolicy().redact(exc), flush=True)

    def _operation(self) -> TrainerInstallOperation | None:
        return self._failed_operation or read_operation(self.folder)

    def _reason(self) -> str | None:
        operation = self._operation()
        if operation and operation.state in BUSY:
            return "update_in_progress"
        reason = self.lifecycle._blocked()
        if reason:
            return reason
        if getattr(self.context, "_update_requests", 0):
            return "data_operation_running"
        current = self.updates.current
        if not current.commit:
            return "current_version_unknown"
        disk = local_version(self.root)
        if disk.commit != current.commit:
            return "version_changed"
        if disk.dirty is True or (disk.source == "git" and disk.dirty is None):
            return "local_changes"
        return None

    def status(self) -> TrainerInstallStatus:
        with self.lock:
            reason = self._reason()
            return TrainerInstallStatus(can_apply=reason is None, reason=reason,
                                        operation=self._operation(), running_commit=self.updates.current.commit,
                                        instance_id=self.lifecycle.instance_id)

    def start(self, request: TrainerInstallRequest) -> TrainerInstallStatus:
        checked = self.updates.status()
        with self.lock, self.lifecycle.lock, self.lifecycle.environment.lock, self.lifecycle.torch.lock, self.context.db.lock:
            existing = self._operation()
            if existing and existing.id == str(request.request_id):
                if existing.target_commit != request.target_commit:
                    raise ApiError("同一次更新不能更换目标版本。", status=409)
                return self.status()
            reason = self._reason()
            if reason:
                raise ApiError("当前无法更新训练器，请先处理页面提示。", code="updates.blocked", status=409, details={"reason": reason})
            if checked.state != "available" or not checked.latest or checked.latest.commit != request.target_commit:
                raise ApiError("请先检查更新，再选择检测到的版本。", code="updates.check_required", status=409)
            self.previous_maintenance = self.context.db.get_kv("environment.maintenance", {})
            fields = dict(id=str(request.request_id), target_commit=request.target_commit,
                          previous_commit=self.updates.current.commit, before_instance_id=self.lifecycle.instance_id,
                          result_instance_id=None, state="preparing", message="正在准备更新", error=None,
                          started_at=time.time(), log=[], rolled_back=False)
            policy = ProxyPolicy()
            maintenance_set = False
            try:
                policy = ProxyPolicy.from_context(self.context)
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
            return self.status()

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
            write_operation(self.folder, state="downloading", message="正在下载更新")
            plan = prepare_update(self.root, work, operation.previous_commit, operation.target_commit,
                                  self.updates.current.source, policy, lambda *args: None)
            staged = Path(plan["staged_root"])
            if self.cancel.is_set():
                raise RuntimeError("更新准备已停止。")
            settings = self.context.settings()
            atomic_json(work / "settings.json", {"downloads": settings.get("downloads", {}), "paths": settings.get("paths", {})})
            snapshot_helpers(work)
            write_operation(self.folder, state="building", message="正在构建前端")
            run_helper(self.folder, "build", staged, sys.executable, policy, self.cancel)
            seal_update(work / "plan.json")
            with self.lock, self.lifecycle.lock, self.context.db.lock:
                write_operation(self.folder, state="restarting", message="正在安装并重启训练器")
                self.lifecycle.request_update(operation.id)
        except Exception as exc:
            with self.lock, self.lifecycle.lock, self.context.db.lock:
                reason = policy.redact(exc)
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
            write_operation(self.folder, state="succeeded", message="更新完成", result_instance_id=self.lifecycle.instance_id)
            self.context.db.set_kv("environment.maintenance", {"blocked": False})
        threading.Thread(target=discard_preparation, args=(self.folder, operation.id), daemon=True).start()

    def close(self):
        if self.worker and self.worker.is_alive():
            if not self.lifecycle.restarting:
                self.cancel.set()
            self.worker.join()


def launcher_policy(data_root: Path) -> ProxyPolicy:
    try:
        settings = json.loads((data_root / "settings.json").read_text("utf-8"))
    except FileNotFoundError:
        settings = {}
    settings["network"] = {"proxy_mode": "system", "proxy_url": "", "proxy_username": "", **settings.get("network", {})}
    context = SimpleNamespace(data_root=data_root, _settings_lock=threading.RLock(), settings=lambda: settings)
    return ProxyPolicy.from_context(context)


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
        write_operation(folder, state="applying", message="更新失败，正在恢复原版源码", error=reason,
                        log=[*operation.log, *(startup_log or [])][-80:])
    except (OSError, ValueError) as exc:
        print("[studio] 无法保存更新失败记录：" + policy.redact(exc), flush=True)
    if startup_log:
        try:
            (work / "startup.log").write_text("\n".join(startup_log) + "\n", encoding="utf-8")
        except OSError as exc:
            reason += "\n启动日志文件保存失败：" + policy.redact(exc)
    try:
        plan = json.loads((work / "plan.json").read_text("utf-8"))
        rollback_update(work / "plan.json", python)
        restored = True
        # Reinstall the previous package metadata and satisfy its requirements too.
        if repair_dependencies:
            policy = launcher_policy(data_root)
            run_helper(folder, "deps", Path(plan["root"]), python, policy,
                       log_name="rollback-deps.log", record_log=False)
            dependencies_ready = True
    except Exception as exc:
        reason += " 恢复检查失败：" + policy.redact(exc)
    try:
        write_operation(folder, state="failed", message="更新失败，已恢复原版源码" if restored else "更新失败",
                        error=reason, rolled_back=restored)
    except (OSError, ValueError) as exc:
        print("[studio] " + policy.redact(reason), flush=True)
        print("[studio] 无法保存更新失败记录：" + policy.redact(exc), flush=True)
    return restored and dependencies_ready


def apply_from_launcher(data_root: Path, update_id: str, python: str) -> bool | None:
    from .source_update import apply_update
    folder = update_folder(data_root)
    operation = read_operation(folder)
    if not operation or operation.id != update_id or operation.state != "restarting":
        return False
    work = folder / operation.id
    policy = ProxyPolicy()
    try:
        policy = launcher_policy(data_root)
        plan = json.loads((work / "plan.json").read_text("utf-8"))
        write_operation(folder, state="applying", message="正在替换源码")
        apply_update(work / "plan.json", python)
        write_operation(folder, state="installing", message="正在检查并安装依赖")
        run_helper(folder, "deps", Path(plan["root"]), python, policy)
        write_operation(folder, state="restarting", message="正在启动新版本")
        return True
    except Exception as exc:
        try:
            phase = json.loads((work / "plan.json").read_text("utf-8")).get("phase")
        except (OSError, ValueError):
            write_operation(folder, state="failed", message="无法读取更新记录", error=policy.redact(exc))
            return None
        if phase in {"ready", "prepared"}:
            write_operation(folder, state="failed", message="更新未应用", error=policy.redact(exc))
            return False
        operation = read_operation(folder)
        restored = rollback_from_launcher(data_root, python, policy.redact(exc),
                                          repair_dependencies=bool(operation and operation.state == "installing"))
        return False if restored else None


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
