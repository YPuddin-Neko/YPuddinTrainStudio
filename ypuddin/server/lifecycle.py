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
from .torch_environments import ACTIVE, atomic_json, same_interpreter

RESTART_TOKEN_ENV = "YPUDDIN_SERVICE_RESTART_TOKEN"


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
        self.original_python: str | None = None
        self.restart_token: str | None = None
        self.model_downloads = None
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

    def _blocked(self) -> str | None:
        if not self.shutdown or not self.restart_token:
            return "start_with_studio_launcher"
        if self.restarting:
            return "restart_in_progress"
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
        if self.model_downloads and any(
            d.get("status") in ("queued", "downloading", "verifying") for d in self.model_downloads.list()
        ):
            return "model_download_running"
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
        return ServiceRuntime(
            environment_profile=self.profile,
            worker_id=os.getpid(),
            instance_id=self.instance_id,
            managed=self.shutdown is not None and bool(self.restart_token),
            can_restart=self._blocked() is None,
            reason=self._blocked(),
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
        with self.lock, self.environment.lock, self.torch.lock, self.context.db.lock:
            reason = self._blocked()
            if reason:
                raise EnvironmentError(409, reason)
            if request.environment_id and request.restore_original_environment:
                raise EnvironmentError(
                    422, "Choose a prepared environment or restore the original one, not both"
                )
            python = self.torch.resolve(request.environment_id) if request.environment_id else sys.executable
            environment_id = request.environment_id or self.torch.current_environment()
            if request.restore_original_environment:
                python, environment_id = self.original_python, None
            settings = self.context.settings()["server"]
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
        while True:
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
            ]
            # Windows venv python.exe may redirect to a second Python process: its
            # Popen PID is then different from the HTTP worker's PID. Authenticate
            # this spawn using an inherited, one-use capability, never PID equality alone.
            spawn_token = secrets.token_hex(32)
            child_env = dict(os.environ)
            child_env[RESTART_TOKEN_ENV] = spawn_token
            try:
                child = subprocess.Popen(command, env=child_env)
            except OSError as exc:
                print(f"[studio] Cannot start the selected interpreter: {exc}", flush=True)
                if python == original_python and (host, port) == original_address:
                    return 1
                python, environment_id = original_python, None
                host, port = original_address
                selected.unlink(missing_ok=True)
                fallback = True
                continue
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
                and request.get("action") == "restart"
                and type(request.get("worker_pid")) is int
                and request["worker_pid"] > 0
                and isinstance(request.get("restart_token"), str)
                and request["restart_token"].isascii()
                and secrets.compare_digest(request["restart_token"], spawn_token)
            ):
                python, host, port = request["python"], request["host"], request["port"]
                environment_id = request.get("environment_id")
                fallback = False
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
