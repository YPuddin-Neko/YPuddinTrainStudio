"""Owned restart and isolated framework preparation, without changing installed packages."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import BackgroundTasks, FastAPI
from fastapi.testclient import TestClient

from ypuddin.runtime_profiles import current_profile, profile_root
from ypuddin.server.bus import EventBus
from ypuddin.server.context import ServiceContext
from ypuddin.server.db import Database, now
from ypuddin.server.environment import EnvironmentError
from ypuddin.server.errors import ApiError, install
from ypuddin.server.lifecycle import RESTART_TOKEN_ENV, RestartRequest, ServiceLifecycle, launch_service
from ypuddin.server.routes_environment import router
from ypuddin.server.supervisor import JobSupervisor
from ypuddin.server.torch_environments import (
    TorchEnvironments,
    TorchOperation,
    TorchRequest,
    atomic_json,
    build_catalog,
    python_in,
)


class StagedInstaller:
    def __init__(self):
        self.calls = []
        self.fail = None

    def run(self, args, log, cancel, **kwargs):
        self.calls.append(args)
        log("staged installer output")
        if cancel.is_set():
            raise InterruptedError()
        if self.fail and self.fail in args:
            raise RuntimeError("simulated download/verification failure")
        if args[1:3] == ["-m", "venv"]:
            root = Path(args[3])
            python_in(root).parent.mkdir(parents=True, exist_ok=True)
            python_in(root).write_text("fake interpreter", encoding="utf-8")
            site = root / (
                "Lib/site-packages"
                if os.name == "nt"
                else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
            )
            site.mkdir(parents=True, exist_ok=True)
        if "-c" in args:
            Path(args[-1]).write_text(
                json.dumps({"torch": args[-3], "device": "mps", "forward_backward": True}), encoding="utf-8"
            )


@pytest.fixture
def lifecycle(tmp_path):
    db = Database(tmp_path / "studio.db")
    bus = EventBus()
    context = ServiceContext(tmp_path, db, bus, JobSupervisor(db, bus, tmp_path))
    environment = SimpleNamespace(
        root=tmp_path / "environment",
        lock=threading.RLock(),
        _idle=Mock(),
        _running=Mock(return_value=False),
        list=Mock(return_value=[]),
        runtime=lambda: {
            "platform": "Darwin",
            "machine": "arm64",
            "python": "3.12.1",
            "gpu_capability": None,
        },
        versions=lambda: {
            "torch": "2.14.0",
            "torchvision": "0.29.0",
            "numpy": "2.3.0",
            "xformers": "0.0.32",
            "pydantic": "2.12.0",
        },
    )
    installer = StagedInstaller()
    torch = TorchEnvironments(context, environment, installer=installer, driver=lambda: None)
    service = ServiceLifecycle(context, environment, torch)
    app = FastAPI()
    app.state.lifecycle, app.state.torch_environments = service, torch
    app.state.environment = environment
    app.include_router(router, prefix="/api")
    install(app)
    client = TestClient(app)
    yield SimpleNamespace(
        context=context, env=environment, torch=torch, service=service, installer=installer, client=client
    )
    torch.close()
    context.versions.close()
    db.close()


def configure(fixture, monkeypatch):
    monkeypatch.setattr("ypuddin.server.lifecycle.threading.Timer", Mock(return_value=Mock()))
    fixture.service.configure(
        control_file=fixture.context.data_root / "restart.json",
        host="127.0.0.1",
        port=8877,
        shutdown=Mock(),
        original_python=sys.executable,
        restart_token="fixture-worker-capability",
    )


def finish(manager, id_):
    end = time.monotonic() + 3
    while time.monotonic() < end:
        op = manager.get(id_)
        if op.status not in ("planning", "installing", "verifying"):
            return op
        time.sleep(0.01)
    pytest.fail("staged operation did not finish")


def test_runtime_is_readable_without_managed_launcher(lifecycle):
    response = lifecycle.client.get("/api/service/runtime")
    assert response.status_code == 200
    assert response.json()["worker_id"] == os.getpid()
    assert not response.json()["can_restart"]
    assert lifecycle.client.post("/api/service/restart", json={}).status_code == 409


def test_runtime_instance_survives_status_reads_but_changes_even_when_pid_is_reused(lifecycle):
    first = lifecycle.client.get("/api/service/runtime").json()
    assert first["instance_id"]
    assert lifecycle.client.get("/api/service/runtime").json()["instance_id"] == first["instance_id"]
    lifecycle.context.db.set_kv(lifecycle.service.selected_key, {"id": "historical_selection"})
    unchanged = lifecycle.client.get("/api/service/runtime").json()
    assert unchanged["instance_id"] == first["instance_id"]
    assert unchanged["selected_environment"] is None
    assert lifecycle.client.get("/api/environment/torch").json()["selected_environment"] is None

    replacement = ServiceLifecycle(lifecycle.context, lifecycle.env, lifecycle.torch).status()
    assert replacement.worker_id == first["worker_id"]
    assert replacement.instance_id != first["instance_id"]
    assert "restart_token" not in first


@pytest.mark.parametrize(
    "recorded,current,selected",
    [
        (
            r"J:\Studio\runtimes\torch_new\Scripts\python.exe",
            "j:/studio/runtimes/torch_new/Scripts/python.exe",
            True,
        ),
        (
            r"J:\Studio\runtimes\torch_new\Scripts\python.exe",
            r"\\?\J:\Studio\runtimes\torch_new\Scripts\python.exe",
            True,
        ),
        (r"\\host\share\Studio\python.exe", r"\\?\UNC\HOST\share\studio\python.exe", True),
        (
            r"J:\Studio\runtimes\torch_old\Scripts\python.exe",
            r"J:\Studio\runtimes\torch_new\Scripts\python.exe",
            False,
        ),
    ],
)
def test_windows_current_environment_uses_interpreter_location_not_stale_selection(
    lifecycle, monkeypatch, recorded, current, selected
):
    op = TorchOperation(
        id="torch_new",
        build_id="2.11.0-cu128",
        status="completed",
        phase="ready_to_restart",
        environment_id="torch_new",
        created_at=now(),
        updated_at=now(),
    )
    lifecycle.context.db.set_kv("torch.operation." + op.id, op.model_dump())
    lifecycle.context.db.set_kv("torch.environment." + op.id, {"python": recorded})
    lifecycle.context.db.set_kv(lifecycle.service.selected_key, {"id": "stale_other_environment"})
    # Exercise Windows lexical rules without changing this test host's global OS state.
    monkeypatch.setattr(
        "ypuddin.server.torch_environments.sys", SimpleNamespace(platform="win32", executable=current)
    )
    monkeypatch.setattr("ypuddin.server.lifecycle.sys", SimpleNamespace(executable=current))
    expected = op.id if selected else None
    assert lifecycle.client.get("/api/environment/torch").json()["selected_environment"] == expected
    assert lifecycle.client.get("/api/service/runtime").json()["selected_environment"] == expected
    lifecycle.service.configure(
        control_file=lifecycle.context.data_root / "restart.json",
        host="127.0.0.1",
        port=8877,
        shutdown=Mock(),
        original_python=recorded,
        restart_token="fixture-worker-capability",
    )
    assert lifecycle.context.db.get_kv(lifecycle.service.selected_key) == {"id": expected}
    assert lifecycle.service.status().can_restore_original is not selected
    assert lifecycle.torch.get(op.id).phase == "ready_to_restart"


def test_unfinished_or_other_profile_environment_is_not_reported_active(lifecycle, monkeypatch):
    op = TorchOperation(
        id="torch_unverified",
        build_id="2.13.0-mps",
        status="verifying",
        environment_id="torch_unverified",
        created_at=now(),
        updated_at=now(),
    )
    lifecycle.context.db.set_kv("torch.operation." + op.id, op.model_dump())
    lifecycle.context.db.set_kv("torch.environment." + op.id, {"python": sys.executable})
    assert lifecycle.torch.current_environment() is None
    lifecycle.torch._update(op.id, status="completed")
    lifecycle.context.db.set_kv(
        "torch.environment." + op.id, {"python": sys.executable, "environment_profile": "windows-cuda"}
    )
    assert lifecycle.torch.current_environment() is None


def test_restart_keeps_effective_address_unless_explicit(lifecycle, monkeypatch):
    configure(lifecycle, monkeypatch)
    lifecycle.context.save_settings({"server": {"host": "0.0.0.0", "port": 8999}})
    result = lifecycle.client.post("/api/service/restart", json={})
    assert result.status_code == 202
    assert result.json()["port"] == 8877
    assert not result.json()["address_changed"]
    request = json.loads(lifecycle.service.control_file.read_text())
    assert request["worker_pid"] == os.getpid()
    assert request["restart_token"] == "fixture-worker-capability"
    assert request["python"] == sys.executable
    assert lifecycle.context.db.get_kv("environment.maintenance")["blocked"]
    assert lifecycle.client.post("/api/service/restart", json={}).status_code == 409


def test_restart_can_explicitly_apply_saved_address(lifecycle, monkeypatch):
    configure(lifecycle, monkeypatch)
    lifecycle.context.save_settings({"server": {"host": "0.0.0.0", "port": 8999}})
    response = lifecycle.client.post("/api/service/restart", json={"apply_saved_address": True})
    assert response.status_code == 202
    assert response.json()["address_changed"]
    assert response.json()["reconnect_url"] == "http://127.0.0.1:8999/"


@pytest.mark.parametrize(
    "busy,reason",
    [
        ("training", "training_or_data_worker_running"),
        ("extension", "extension_operation_running"),
        ("torch", "torch_operation_running"),
        ("download", "model_download_running"),
    ],
)
def test_restart_rejects_workers_and_installers(lifecycle, monkeypatch, busy, reason):
    configure(lifecycle, monkeypatch)
    if busy == "training":
        lifecycle.env._running.return_value = True
    elif busy == "extension":
        lifecycle.env.list.return_value = [SimpleNamespace(status="installing")]
    elif busy == "download":
        lifecycle.service.model_downloads = SimpleNamespace(list=lambda: [{"status": "downloading"}])
    else:
        op = TorchOperation(
            id="torch_busy", build_id="2.13.0-mps", status="installing", created_at=now(), updated_at=now()
        )
        lifecycle.context.db.set_kv("torch.operation." + op.id, op.model_dump())
    assert lifecycle.client.get("/api/service/runtime").json()["reason"] == reason
    assert lifecycle.client.post("/api/service/restart", json={}).status_code == 409
    assert not lifecycle.service.control_file.exists()


def test_restart_rejects_unverified_or_conflicting_environment(lifecycle, monkeypatch):
    configure(lifecycle, monkeypatch)
    assert (
        lifecycle.client.post("/api/service/restart", json={"environment_id": "../../venv"}).status_code
        == 422
    )
    assert (
        lifecycle.client.post(
            "/api/service/restart", json={"environment_id": "unknown", "restore_original_environment": True}
        ).status_code
        == 422
    )


@pytest.mark.parametrize("progress_id", [None, "receiving_restart_guard"])
@pytest.mark.parametrize("cancel", [False, True])
def test_upload_receiving_blocks_restart_and_releases_after_failure(
    lifecycle, monkeypatch, progress_id, cancel
):
    from ypuddin.server import routes_work

    configure(lifecycle, monkeypatch)
    monkeypatch.setattr(routes_work, "assert_version_writable", lambda *a, **kw: {"id": "v1"})
    monkeypatch.setattr(routes_work, "_register_upload", Mock(side_effect=RuntimeError("invalid upload")))

    async def scenario():
        receiving, release = asyncio.Event(), asyncio.Event()

        @asynccontextmanager
        async def read_upload(_request, progress):
            assert (progress is None) == (progress_id is None)
            receiving.set()
            await release.wait()
            yield object()

        monkeypatch.setattr(routes_work, "read_upload", read_upload)
        pending = asyncio.create_task(
            routes_work.upload_dataset(
                "p1",
                object(),
                BackgroundTasks(),
                lifecycle.context,
                version_id="v1",
                progress_id=progress_id,
            )
        )
        try:
            await asyncio.wait_for(receiving.wait(), timeout=3)
            assert lifecycle.service.status().reason == "data_operation_running"
            with pytest.raises(EnvironmentError, match="data_operation_running"):
                lifecycle.service.restart(RestartRequest())
            assert not lifecycle.service.control_file.exists()
            if cancel:
                pending.cancel()
                expected = asyncio.CancelledError
            else:
                release.set()
                expected = RuntimeError
            with pytest.raises(expected):
                await pending
            assert lifecycle.context._active_imports == 0
            assert lifecycle.service.status().can_restart
        finally:
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

    asyncio.run(scenario())


def test_restart_rejects_new_upload_before_reading_body(lifecycle, monkeypatch):
    from ypuddin.server import routes_work

    configure(lifecycle, monkeypatch)
    reader = Mock()
    monkeypatch.setattr(routes_work, "read_upload", reader)
    lifecycle.service.restart(RestartRequest())
    with pytest.raises(ApiError) as error:
        asyncio.run(routes_work.upload_dataset("p1", object(), BackgroundTasks(), lifecycle.context))
    assert error.value.status == 409 and error.value.code == "service.restarting"
    reader.assert_not_called()
    assert lifecycle.context._active_imports == 0


def test_import_waiting_for_admission_cannot_pass_a_restart_claim(lifecycle, monkeypatch):
    configure(lifecycle, monkeypatch)
    waiting = threading.Event()
    admitted = Mock()
    rejected = []

    def upload():
        waiting.set()
        try:
            with lifecycle.context.import_admission():
                admitted()
        except ApiError as error:
            rejected.append(error.code)

    with lifecycle.context.db.lock:
        worker = threading.Thread(target=upload)
        worker.start()
        assert waiting.wait(timeout=3)
        lifecycle.service.restart(RestartRequest())
    worker.join(timeout=3)
    assert not worker.is_alive()
    admitted.assert_not_called()
    assert rejected == ["service.restarting"]


def test_registered_upload_blocks_restart_until_indexing_finishes(lifecycle, monkeypatch):
    configure(lifecycle, monkeypatch)
    with lifecycle.context.import_admission():
        lifecycle.context.db.insert(
            "datasets",
            {
                "id": "d1",
                "path": str(lifecycle.context.data_root / "images"),
                "created_at": now(),
                "index_status": "indexing",
            },
        )
    assert lifecycle.context._active_imports == 0
    assert lifecycle.service.status().reason == "data_operation_running"
    assert lifecycle.client.post("/api/service/restart", json={}).status_code == 409
    lifecycle.context.db.update("datasets", "d1", {"index_status": "ready"})
    assert lifecycle.service.status().can_restart


def test_torch_failure_retains_current_environment_and_maintenance(lifecycle):
    canary = lifecycle.context.data_root / "original-venv.txt"
    canary.write_text("preserve me")
    prior = {"blocked": True, "restart_required": True, "operation_id": "previous_extension"}
    lifecycle.context.db.set_kv("environment.maintenance", prior)
    lifecycle.installer.fail = "install"
    op = lifecycle.torch.start(TorchRequest(build_id="2.13.0-mps"))
    lifecycle.torch.apply(op.id)
    result = finish(lifecycle.torch, op.id)
    assert result.status == "failed" and "simulated" in result.error
    assert canary.read_text() == "preserve me"
    assert lifecycle.context.db.get_kv("environment.maintenance") == prior
    assert lifecycle.context.db.get_kv("torch.selected") is None
    assert not lifecycle.context.db.get_kv("torch.environment." + op.id)
    assert all("uninstall" not in call for call in lifecycle.installer.calls)
    with pytest.raises(EnvironmentError):
        lifecycle.torch.resolve(op.id)
    assert lifecycle.torch.start(TorchRequest(build_id="2.13.0-mps")).status == "ready"


def test_torch_success_prepares_then_activates_only_on_restart(lifecycle, monkeypatch):
    configure(lifecycle, monkeypatch)
    op = lifecycle.torch.start(TorchRequest(build_id="2.13.0-mps"))
    lifecycle.torch.apply(op.id)
    result = finish(lifecycle.torch, op.id)
    assert result.status == "completed", result.error
    assert lifecycle.context.db.get_kv("torch.selected")["id"] is None
    constraints = (lifecycle.torch.root / op.id / "constraints.txt").read_text()
    assert "pydantic===2.12.0" in constraints
    assert "xformers" not in constraints
    assert "torch==2.13.0" in constraints
    response = lifecycle.client.post("/api/service/restart", json={"environment_id": op.id})
    assert response.status_code == 202
    request = json.loads(lifecycle.service.control_file.read_text())
    assert request["python"] == str(python_in(lifecycle.torch.root / op.id))
    assert request["environment_id"] == op.id


def test_failed_notification_dismiss_does_not_change_status(lifecycle):
    op = lifecycle.torch.start(TorchRequest(build_id="2.13.0-mps"))
    assert lifecycle.client.post(f"/api/environment/torch/operations/{op.id}/dismiss").status_code == 409
    lifecycle.torch._update(op.id, status="failed", error="keep me", logs=["keep log"])
    assert lifecycle.client.post(f"/api/environment/torch/operations/{op.id}/dismiss").status_code == 200
    assert lifecycle.torch.get(op.id).status == "failed"
    assert lifecycle.torch.get(op.id).error == "keep me"
    assert lifecycle.torch.get(op.id).dismissed_at


def test_platform_catalog_does_not_offer_macos_cuda_or_old_blackwell_build():
    mac = build_catalog({"platform": "Darwin", "machine": "arm64", "python": "3.12.1"}, None)
    assert all(b.backend == "mps" for b in mac if b.supported)
    win = build_catalog(
        {"platform": "Windows", "machine": "AMD64", "python": "3.12.1", "gpu_capability": [12, 0]}, 575
    )
    assert not any(b.supported for b in win if b.backend in ("mps", "cu126", "cu130"))
    assert next(b for b in win if b.id == "2.11.0-cu128").recommended


def test_dtk_profile_cannot_switch_to_official_cuda_or_cpu_torch():
    builds = build_catalog({"platform": "Linux", "machine": "x86_64", "python": "3.11.0"}, 580, "linux-dtk")
    assert builds and all(not build.supported for build in builds)
    assert all(build.reason == "different_deployment_profile" for build in builds)


@pytest.mark.parametrize("suffix", ["venv", "runtimes/torch_test"])
def test_dtk_profile_paths_are_inferred_and_separated_from_cuda(tmp_path, monkeypatch, suffix):
    monkeypatch.delenv("YPUDDIN_ENV_PROFILE", raising=False)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "environment/profiles/linux-dtk" / suffix))
    assert current_profile() == "linux-dtk"
    assert profile_root(tmp_path) == tmp_path / "environment/profiles/linux-dtk"
    assert profile_root(tmp_path) != profile_root(tmp_path, "linux-cuda")


def test_launcher_popen_failure_restores_original_interpreter(tmp_path, monkeypatch):
    folder = tmp_path / "environment/service"
    selected = folder / "selected.json"
    replacement = python_in(tmp_path / "environment/runtimes/torch_selected")
    replacement.parent.mkdir(parents=True)
    replacement.touch()
    atomic_json(selected, {"id": "torch_selected", "python": str(replacement)})
    child = Mock(pid=55, wait=Mock(return_value=0), poll=Mock(return_value=0))
    popen = Mock(side_effect=[OSError("missing shared library"), child])
    monkeypatch.setattr("ypuddin.server.lifecycle.subprocess.Popen", popen)
    assert launch_service(str(tmp_path), "127.0.0.1", 8877) == 0
    assert popen.call_args_list[0].args[0][0] == str(replacement)
    assert popen.call_args_list[1].args[0][0] == sys.executable
    assert not selected.exists()


def test_launcher_sigterm_terminates_only_owned_child(tmp_path, monkeypatch):
    installed = {}

    def set_signal(sig, handler):
        old = installed.get(sig, signal.SIG_DFL)
        installed[sig] = handler
        return old

    child = Mock(pid=55)
    child.poll.return_value = None

    def wait(*args, **kwargs):
        installed[signal.SIGTERM](signal.SIGTERM, None)
        child.poll.return_value = 0
        return 0

    child.wait.side_effect = wait
    monkeypatch.setattr("ypuddin.server.lifecycle.signal.signal", set_signal)
    monkeypatch.setattr("ypuddin.server.lifecycle.subprocess.Popen", Mock(return_value=child))
    assert launch_service(str(tmp_path), "127.0.0.1", 8877) == 143
    child.terminate.assert_called_once()
    child.kill.assert_not_called()
    assert installed[signal.SIGTERM] == signal.SIG_DFL


def test_launcher_accepts_authenticated_windows_venv_redirector_and_rotates_token(tmp_path, monkeypatch):
    tokens = []

    def spawn(command, *, env):
        index = len(tokens)
        tokens.append(env[RESTART_TOKEN_ENV])
        control = Path(command[command.index("--control-file") + 1])

        def wait():
            # The Windows venv redirector's PID is not os.getpid() in its Python child.
            atomic_json(
                control,
                {
                    "action": "restart",
                    "worker_pid": 32960,
                    "restart_token": tokens[0],
                    "python": sys.executable,
                    "host": "127.0.0.1",
                    "port": 8877,
                    "environment_id": None,
                },
            )
            return 0

        # A valid first request restarts. Replaying its token in the next spawn must fail.
        return Mock(pid=17012 + index, wait=Mock(side_effect=wait), poll=Mock(return_value=0))

    popen = Mock(side_effect=spawn)
    monkeypatch.setattr("ypuddin.server.lifecycle.subprocess.Popen", popen)
    assert launch_service(str(tmp_path), "127.0.0.1", 8877) == 0
    assert popen.call_count == 2
    assert len(tokens[0]) == 64 and tokens[0] != tokens[1]
    assert RESTART_TOKEN_ENV not in os.environ


@pytest.mark.parametrize("token", [None, "wrong", "非ASCII"])
def test_launcher_rejects_unauthenticated_request_even_when_pid_matches(tmp_path, monkeypatch, token):
    def spawn(command, *, env):
        control = Path(command[command.index("--control-file") + 1])

        def wait():
            atomic_json(control, {"action": "restart", "worker_pid": 55, "restart_token": token})
            return 0

        return Mock(pid=55, wait=Mock(side_effect=wait), poll=Mock(return_value=0))

    popen = Mock(side_effect=spawn)
    monkeypatch.setattr("ypuddin.server.lifecycle.subprocess.Popen", popen)
    assert launch_service(str(tmp_path), "127.0.0.1", 8877) == 0
    popen.assert_called_once()


def test_worker_without_launcher_capability_cannot_request_restart(lifecycle):
    lifecycle.service.configure(
        control_file=lifecycle.context.data_root / "restart.json",
        host="127.0.0.1",
        port=8877,
        shutdown=Mock(),
        original_python=sys.executable,
    )
    assert not lifecycle.service.status().managed
    assert lifecycle.client.post("/api/service/restart", json={}).status_code == 409
    assert not lifecycle.service.control_file.exists()


def test_torch_plans_and_prepared_interpreters_belong_to_one_profile(lifecycle):
    managers = []
    operations = []
    try:
        for profile, build in [("macos-cpu", "2.13.0-cpu"), ("macos-mps", "2.13.0-mps")]:
            environment = SimpleNamespace(
                **{
                    **vars(lifecycle.env),
                    "profile": profile,
                    "root": profile_root(lifecycle.context.data_root, profile),
                }
            )
            manager = TorchEnvironments(
                lifecycle.context, environment, installer=StagedInstaller(), driver=lambda: None
            )
            managers.append(manager)
            op = manager.start(TorchRequest(build_id=build))
            manager.apply(op.id)
            assert finish(manager, op.id).status == "completed"
            operations.append(op.id)
        for own, other in [(0, 1), (1, 0)]:
            manager = managers[own]
            assert [op.id for op in manager.list()] == [operations[own]]
            assert Path(manager.resolve(operations[own])).is_relative_to(manager.root)
            with pytest.raises(EnvironmentError):
                manager.apply(operations[other])
            with pytest.raises(EnvironmentError):
                manager.resolve(operations[other])
        assert managers[0].root != managers[1].root
        with pytest.raises(EnvironmentError, match="different_deployment_profile"):
            managers[0].start(TorchRequest(build_id="2.13.0-mps"))
    finally:
        for manager in managers:
            manager.close()


def test_profile_launcher_ignores_legacy_selected_interpreter(tmp_path, monkeypatch):
    monkeypatch.setenv("YPUDDIN_ENV_PROFILE", "macos-cpu")
    selected = tmp_path / "environment/service/selected.json"
    foreign = python_in(tmp_path / "environment/runtimes/torch_legacy")
    foreign.parent.mkdir(parents=True)
    foreign.touch()
    atomic_json(selected, {"id": "torch_legacy", "python": str(foreign)})
    original = selected.read_bytes()
    child = Mock(pid=55, wait=Mock(return_value=0), poll=Mock(return_value=0))
    popen = Mock(return_value=child)
    monkeypatch.setattr("ypuddin.server.lifecycle.subprocess.Popen", popen)
    assert launch_service(str(tmp_path), "127.0.0.1", 8877) == 0
    command = popen.call_args.args[0]
    assert command[0] == sys.executable
    assert "environment/profiles/macos-cpu/service" in command[command.index("--control-file") + 1]
    assert selected.read_bytes() == original


def test_cpu_profile_automatic_training_and_queue_ignore_visible_accelerator(lifecycle, monkeypatch):
    from ypuddin.train.trainer import Trainer

    monkeypatch.setenv("YPUDDIN_ENV_PROFILE", "macos-cpu")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    monkeypatch.setattr(
        "ypuddin.server.supervisor.gpu_info", Mock(side_effect=AssertionError("no GPU scheduling"))
    )
    assert Trainer._pick_device().type == "cpu"
    assert lifecycle.context.supervisor._choose_device({}) == "cpu"


@pytest.mark.parametrize("suffix", ["venv", "runtimes/torch_selected"])
def test_profile_is_recovered_when_interpreter_is_started_directly(tmp_path, monkeypatch, suffix):
    monkeypatch.delenv("YPUDDIN_ENV_PROFILE", raising=False)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "environment/profiles/windows-cpu" / suffix))
    assert current_profile() == "windows-cpu"
    monkeypatch.setenv("YPUDDIN_ENV_PROFILE", "windows-cuda")
    with pytest.raises(ValueError, match="Interpreter belongs"):
        current_profile()
