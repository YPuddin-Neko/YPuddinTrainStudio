"""Owned service restart: an HTTP worker requests its launcher to start the next worker.

No PID discovery, process-name termination, or shell commands. The launcher retains the
original interpreter and effective address for recovery; changing a saved address is explicit.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ypuddin.runtime_profiles import current_profile, profile_root, selected_key

from .environment import EnvironmentError
from .network import ProxyPolicy
from .torch_environments import ACTIVE, atomic_json, same_interpreter

RESTART_TOKEN_ENV = "YPUDDIN_SERVICE_RESTART_TOKEN"
DOWNLOADING = ("queued", "downloading", "verifying")
# Task-center work a restart would cut off, by kind.
BACKGROUND_REASONS = {
    "model_download": "model_download_running",
    "dataset_upload": "data_operation_running",
    "dataset_refresh": "data_operation_running",
    "project_delete": "data_operation_running",
    "environment": "extension_operation_running",
}
# Why a prepared trainer update did not restart the service.
UPDATE_BLOCKED = {
    "restart_in_progress": "服务正在重启",
    "data_operation_running": "数据处理正在进行",
    "training_or_data_worker_running": "训练或数据任务正在运行",
    "extension_operation_running": "扩展安装正在进行",
    "torch_operation_running": "PyTorch 环境正在安装",
    "model_download_running": "模型正在下载",
    "version_operation_running": "版本操作正在进行",
}
_UNREAD = object()


def update_blocked_message(reason: str) -> str:
    return f"{UPDATE_BLOCKED.get(reason, '有任务正在运行')}，未安装更新。请等待完成后重新更新。"


def restart_blocked_message(reason: str) -> str:
    if reason == "start_with_studio_launcher":
        return "训练器不是由项目启动脚本启动的，无法在页面上重启。请用项目启动脚本启动训练器。"
    if reason == "restart_in_progress":
        return "训练器正在重启，请稍候。"
    if reason == "update_in_progress":
        return "训练器更新正在进行，暂时无法重启。"
    return f"{UPDATE_BLOCKED.get(reason, '有任务正在运行')}，暂时无法重启。请等它结束后再试。"


class RestartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    apply_saved_address: bool = False
    environment_id: str | None = None
    restore_original_environment: bool = False


class ServiceRuntime(BaseModel):
    environment_profile: str = "legacy"
    worker_id: int
    instance_id: str | None = None
    managed: bool
    can_restart: bool
    reason: str | None
    current_host: str | None
    current_port: int | None
    saved_host: str
    saved_port: int
    restart_required: bool
    restarting: bool
    current_python: str
    original_python: str | None
    can_restore_original: bool
    selected_environment: str | None


class RestartResult(BaseModel):
    status: Literal["restarting"] = "restarting"
    host: str
    port: int
    reconnect_url: str
    address_changed: bool
    environment_id: str | None = None


class ServiceLifecycle:
    def __init__(self, context, environment, torch_environments):
        self.context, self.environment, self.torch = context, environment, torch_environments
        self.profile = getattr(environment, "profile", current_profile())
        self.selected_key = selected_key(self.profile)
        # A PID can be reused after restart; this identifies this HTTP service instance.
        self.instance_id = uuid.uuid4().hex
        self.lock = threading.RLock()
        self.control_file: Path | None = None
        self.shutdown = None
        self.host: str | None = None
        self.port: int | None = None
        self.restarting = False
        self.updating = False
        self.original_python: str | None = None
        self.restart_token: str | None = None
        self.model_downloads = None
        self.vision_models = None
        self.dataset_pipeline = None

    def configure(
        self,
        *,
        control_file: Path,
        host: str,
        port: int,
        shutdown,
        original_python: str,
        restart_token: str | None = None,
    ):
        self.control_file, self.host, self.port, self.shutdown = control_file, host, port, shutdown
        self.original_python = original_python
        self.restart_token = restart_token
        selected = self.torch.current_environment()
        self.context.db.set_kv(self.selected_key, {"id": selected})

    def background_reason(self) -> str | None:
        """Work tracked outside the database that a restart would cut off.

        Read it before taking the database lock: these services write to the database while
        holding their own locks.
        """
        if self.model_downloads and any(d.get("status") in DOWNLOADING for d in self.model_downloads.list()):
            return "model_download_running"
        if self.vision_models:
            with self.vision_models.lock:
                if any(task.get("status") in DOWNLOADING for task in self.vision_models.tasks.values()):
                    return "model_download_running"
        tasks = getattr(self.context, "background_tasks", None)
        for task in tasks.list() if tasks is not None else []:
            if task.state == "running" and task.kind in BACKGROUND_REASONS:
                return BACKGROUND_REASONS[task.kind]
        return None

    def _blocked(self, background=_UNREAD, *, updating: bool = False) -> str | None:
        """Why a restart has to wait, or None.

        ``background`` is ``background_reason()`` read before the caller took the database lock;
        without it this must not be called while holding that lock. ``updating`` checks the
        handoff of the update this service is preparing itself.
        """
        if not self.shutdown or not self.restart_token:
            return "start_with_studio_launcher"
        if self.restarting:
            return "restart_in_progress"
        if self.updating and not updating:
            return "update_in_progress"
        if self.context._active_tts_downloads:
            return "model_download_running"
        if self.context._active_imports or self.context.db.fetchone(
            "SELECT id FROM datasets WHERE index_status='indexing' LIMIT 1"
        ):
            return "data_operation_running"
        if self.environment._running():
            return "training_or_data_worker_running"
        if any(op.status in ("planning", "installing", "verifying") for op in self.environment.list()):
            return "extension_operation_running"
        if any(op.status in ACTIVE for op in self.torch.list()):
            return "torch_operation_running"
        if background is _UNREAD:
            background = self.background_reason()
        if background:
            return background
        if self.context.db.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dataset_pipeline_operations'"
        ):
            if self.context.db.fetchone(
                "SELECT id FROM dataset_pipeline_operations WHERE status IN ('queued','running','cancelling') LIMIT 1"
            ):
                return "data_operation_running"
        if self.context.db.fetchone(
            "SELECT id FROM project_versions WHERE busy IS NOT NULL OR status='copying' LIMIT 1"
        ):
            return "version_operation_running"
        return None

    def status(self):
        settings = self.context.settings()["server"]
        reason = self._blocked()
        return ServiceRuntime(
            environment_profile=self.profile,
            worker_id=os.getpid(),
            instance_id=self.instance_id,
            managed=self.shutdown is not None and bool(self.restart_token),
            can_restart=reason is None,
            reason=reason,
            current_host=self.host,
            current_port=self.port,
            saved_host=settings["host"],
            saved_port=settings["port"],
            restart_required=self.context.db.get_kv("environment.maintenance", {}).get(
                "restart_required", False
            ),
            restarting=self.restarting,
            current_python=sys.executable,
            original_python=self.original_python,
            can_restore_original=bool(
                self.original_python and not same_interpreter(self.original_python, sys.executable)
            ),
            selected_environment=self.torch.current_environment(),
        )

    def restart(self, request: RestartRequest):
        # Settings and other services' work are read first: their locks are never taken while
        # holding the database lock.
        background = self.background_reason()
        settings = self.context.settings()["server"]
        with self.lock, self.environment.lock, self.torch.lock, self.context.db.lock:
            reason = self._blocked(background)
            if reason:
                raise EnvironmentError(409, restart_blocked_message(reason))
            if request.environment_id and request.restore_original_environment:
                raise EnvironmentError(422, "切换到准备好的环境和恢复原环境只能选择一项。")
            python = self.torch.resolve(request.environment_id) if request.environment_id else sys.executable
            environment_id = request.environment_id or self.torch.current_environment()
            if request.restore_original_environment:
                python, environment_id = self.original_python, None
            host = settings["host"] if request.apply_saved_address else self.host
            port = settings["port"] if request.apply_saved_address else self.port
            assert host is not None and port is not None and self.control_file and python
            result = RestartResult(
                host=host,
                port=port,
                address_changed=(host, port) != (self.host, self.port),
                environment_id=environment_id,
                reconnect_url=f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/",
            )
            self.context.db.set_kv("environment.maintenance", {"blocked": True, "restarting": True})
            atomic_json(
                self.control_file,
                {
                    "action": "restart",
                    "python": python,
                    "host": host,
                    "port": port,
                    "environment_id": environment_id,
                    "worker_pid": os.getpid(),
                    "restart_token": self.restart_token,
                },
            )
            self.restarting = True
            # Return the response before uvicorn stops accepting connections.
            timer = threading.Timer(0.4, self.shutdown)
            timer.daemon = True
            timer.start()
            return result

    def request_update(self, update_id: str, background=_UNREAD):
        """Hand a prepared update to the launcher. Call while holding the database lock, with
        ``background_reason()`` read before taking it: work that started while the update was
        being prepared keeps the service running."""
        if not self.updating or not self.control_file or not self.restart_token or not self.shutdown:
            raise EnvironmentError(409, "训练器不是由项目启动脚本启动的，无法安装更新。")
        if reason := self._blocked(background, updating=True):
            raise EnvironmentError(409, update_blocked_message(reason))
        atomic_json(self.control_file, {
            "action": "update", "update_id": update_id, "python": sys.executable,
            "host": self.host, "port": self.port, "environment_id": self.torch.current_environment(),
            "worker_pid": os.getpid(), "restart_token": self.restart_token,
        })
        self.restarting = True
        timer = threading.Timer(0.4, self.shutdown)
        timer.daemon = True
        timer.start()


def saved_address(root: Path, host: str | None, port: int | None) -> tuple[str, int]:
    settings = {}
    try:
        settings = json.loads((root / "settings.json").read_text(encoding="utf-8")).get("server", {})
    except (OSError, ValueError):
        pass
    return host or settings.get("host", "127.0.0.1"), port or settings.get("port", 8123)


def pending_data_root(root: Path) -> Path | None:
    try:
        value = json.loads((root / "settings.json").read_text(encoding="utf-8")).get("_pending_data_root")
    except (OSError, ValueError, TypeError):
        return None
    return Path(value).expanduser().resolve() if isinstance(value, str) and value.strip() else None


def launch_service(data_root: str, host: str | None, port: int | None) -> int:
    from .trainer_install import (
        UPDATE_ID_ENV,
        UpdateRecoveryFailed,
        WorkerStartupLog,
        apply_from_launcher,
        launcher_policy,
        resume_from_launcher,
        rollback_from_launcher,
        wait_for_updated_worker,
    )
    root = Path(data_root).expanduser().resolve()
    environment_root = profile_root(root)
    folder = environment_root / "service"
    folder.mkdir(parents=True, exist_ok=True)
    control = folder / f"restart-{uuid.uuid4().hex}.json"
    selected = folder / "selected.json"
    original_python = sys.executable
    python = original_python
    environment_id = None
    if selected.exists():
        try:
            record = json.loads(selected.read_text(encoding="utf-8"))
            candidate = Path(record["python"])
            expected_root = environment_root / "runtimes" / record["id"]
            if (
                candidate.is_file()
                and candidate.parent.parent == expected_root
                and not expected_root.is_symlink()
            ):
                python, environment_id = str(candidate), record["id"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
    host, port = saved_address(root, host, port)
    original_address = host, port
    fallback = False
    child = None
    pending_update = None
    terminated_signal = None

    def stop_owned_child(signum, _frame):
        nonlocal terminated_signal
        terminated_signal = signum
        if child is not None and child.poll() is None:
            child.terminate()

    previous_sigterm = None
    if threading.current_thread() is threading.main_thread():
        previous_sigterm = signal.signal(signal.SIGTERM, stop_owned_child)
    try:
        # An update a closed window or a power loss interrupted is finished or undone first.
        try:
            pending_update = resume_from_launcher(root, python)
        except UpdateRecoveryFailed as error:
            print(f"[studio] {error}", flush=True)
            return 1
        except KeyboardInterrupt:
            return 130
        while True:
            if terminated_signal is not None:
                return 128 + terminated_signal
            control.unlink(missing_ok=True)
            command = [
                python,
                "-m",
                "ypuddin.cli",
                "serve",
                "--host",
                host,
                "--port",
                str(port),
                "--data-root",
                str(root),
                "--service-worker",
                "--control-file",
                str(control),
                "--original-python",
                original_python,
                # The worker stops itself once this launcher is gone.
                "--launcher-pid",
                str(os.getpid()),
            ]
            # Windows venv python.exe may redirect to a second Python process: its
            # Popen PID is then different from the HTTP worker's PID. Authenticate
            # this spawn using an inherited, one-use capability, never PID equality alone.
            spawn_token = secrets.token_hex(32)
            child_env = dict(os.environ)
            child_env[RESTART_TOKEN_ENV] = spawn_token
            child_env.pop(UPDATE_ID_ENV, None)
            if pending_update:
                child_env[UPDATE_ID_ENV] = pending_update
                child_env["PYTHONUNBUFFERED"] = "1"
                child_env["PYTHONIOENCODING"] = "utf-8"
            startup_log = None
            policy = ProxyPolicy()
            try:
                if pending_update:
                    policy = launcher_policy(root)
                    child = subprocess.Popen(command, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             text=True, encoding="utf-8", errors="replace", bufsize=1)
                    startup_log = WorkerStartupLog(child, policy)
                else:
                    child = subprocess.Popen(command, env=child_env)
            except (OSError, ValueError, RuntimeError) as exc:
                reason = policy.redact(exc)
                print(f"[studio] Cannot start the selected interpreter: {reason}", flush=True)
                if pending_update:
                    if child is not None and child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=8)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=8)
                    if not rollback_from_launcher(root, python, "无法启动更新后的 Python 服务：" + reason):
                        return 1
                    pending_update = None
                    continue
                if python == original_python and (host, port) == original_address:
                    return 1
                python, environment_id = original_python, None
                host, port = original_address
                selected.unlink(missing_ok=True)
                fallback = True
                continue
            if pending_update:
                if not wait_for_updated_worker(child, root, pending_update):
                    exit_code = child.poll()
                    if exit_code is None:
                        child.terminate()
                        try:
                            child.wait(timeout=8)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=8)
                    if terminated_signal is not None:
                        reason = "新版本启动被中断，已停止本次更新。"
                    elif exit_code is None:
                        reason = "新版本启动超时，已停止本次更新。"
                    else:
                        reason = f"新版本未能正常启动（退出码 {exit_code}），已停止本次更新。"
                    restored = rollback_from_launcher(root, python, reason,
                                                      repair_dependencies=terminated_signal is None,
                                                      startup_log=startup_log.finish())
                    pending_update = None
                    if terminated_signal is not None:
                        return 128 + terminated_signal
                    if not restored:
                        print("[studio] 更新恢复检查失败，已停止启动。请查看更新日志并重新运行启动脚本修复环境。", flush=True)
                        return 1
                    continue
                startup_log.release()
                pending_update = None
            try:
                code = child.wait()
            except KeyboardInterrupt:
                # Same terminal delivers SIGINT to the owned child too. No unrelated PID is touched.
                try:
                    return child.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    return child.wait(timeout=8)
            if terminated_signal is not None:
                return 128 + terminated_signal
            request = None
            if control.exists():
                try:
                    request = json.loads(control.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
            if (
                isinstance(request, dict)
                and request.get("action") in {"restart", "update"}
                and type(request.get("worker_pid")) is int
                and request["worker_pid"] > 0
                and isinstance(request.get("restart_token"), str)
                and request["restart_token"].isascii()
                and secrets.compare_digest(request["restart_token"], spawn_token)
            ):
                python, host, port = request["python"], request["host"], request["port"]
                environment_id = request.get("environment_id")
                fallback = False
                if request["action"] == "update":
                    pending_update = request.get("update_id")
                    try:
                        outcome = apply_from_launcher(root, pending_update, python)
                    except KeyboardInterrupt:
                        return 130
                    if outcome is None:
                        print("[studio] 更新恢复检查失败，已停止启动。请查看更新日志并重新运行启动脚本修复环境。", flush=True)
                        return 1
                    if not outcome:
                        pending_update = None
                    continue
                if environment_id:
                    atomic_json(selected, {"id": environment_id, "python": python})
                else:
                    selected.unlink(missing_ok=True)
                if (next_root := pending_data_root(root)) is not None and next_root != root:
                    root = next_root
                    environment_root = profile_root(root)
                    folder = environment_root / "service"
                    folder.mkdir(parents=True, exist_ok=True)
                    control = folder / f"restart-{uuid.uuid4().hex}.json"
                    selected = folder / "selected.json"
                continue
            if request is not None:
                print(
                    "[studio] Ignored a restart request not authenticated for this worker spawn.", flush=True
                )
            if code and (python != original_python or (host, port) != original_address) and not fallback:
                print(
                    "[studio] Restart failed; returning to the original environment and address.", flush=True
                )
                python = original_python
                host, port = original_address
                environment_id = None
                fallback = True
                selected.unlink(missing_ok=True)
                continue
            return code
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=8)
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        control.unlink(missing_ok=True)
