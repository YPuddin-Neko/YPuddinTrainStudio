"""Dependency management contracts; all installer and GPU calls are simulated."""

import hashlib
import importlib
import json
import sys
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from packaging.tags import sys_tags

from ypuddin.server.bus import EventBus
from ypuddin.server.context import ServiceContext
from ypuddin.server.db import Database, now
from ypuddin.server.environment import (
    CATALOG,
    PROBE,
    EnvironmentError,
    EnvironmentManager,
    EnvironmentRequest,
    environment_attention_default,
    maintenance_blocked,
    probe_packages,
)
from ypuddin.server.errors import install as install_errors
from ypuddin.server.routes_environment import router
from ypuddin.server.supervisor import JobSupervisor

WHEEL = b"simulated reviewed wheel bytes"


class FakeInstaller:
    def __init__(self, versions):
        self.versions = versions
        self.calls = []
        self.rows = None
        self.corrupt = False
        self.fail_install = False

    def command(self, log, cancel):
        return ["simulated-pip"]

    def run(self, args, log, cancel, **kwargs):
        self.calls.append(args)
        if cancel.is_set():
            raise InterruptedError()
        log("simulated installer output")
        if "--dry-run" in args:
            package = args[-1].split("==")[0]
            selected = args[-1].split("==")[-1] if "==" in args[-1] else "1.0"
            self.current = self.rows if self.rows is not None else [self.row(package, selected)]
            Path(args[args.index("--report") + 1]).write_text(json.dumps({"install": self.current}))
        elif "download" in args:
            dest = Path(args[args.index("--dest") + 1])
            for item in self.current:
                (dest / item["download_info"]["url"].rsplit("/", 1)[1]).write_bytes(
                    b"corrupt" if self.corrupt else WHEEL
                )
        elif "uninstall" in args:
            self.versions.pop(args[-1], None)
        elif "install" in args:
            if self.fail_install:
                raise RuntimeError("simulated install failure: locked DLL")
            for item in self.current:
                self.versions[item["metadata"]["name"]] = item["metadata"]["version"]

    @staticmethod
    def row(package, version):
        return {
            "metadata": {"name": package, "version": version},
            "download_info": {
                "url": f"https://files.pythonhosted.org/packages/{package}-{version}-py3-none-any.whl",
                "archive_info": {"hashes": {"sha256": hashlib.sha256(WHEEL).hexdigest()}},
            },
        }


@pytest.fixture
def env(tmp_path):
    db = Database(tmp_path / "studio.db")
    bus = EventBus()
    supervisor = JobSupervisor(db, bus, tmp_path)
    context = ServiceContext(tmp_path, db, bus, supervisor)
    versions = {"torch": "2.5.1", "numpy": "2.1.0"}
    runtime = {
        "python": "3.12.1",
        "python_executable": "C:/studio/venv/Scripts/python.exe",
        "platform": "Windows",
        "machine": "AMD64",
        "torch": "2.5.1+cu128",
        "cuda_runtime": "12.8",
        "cuda_available": True,
        "mps_available": False,
        "gpu_capability": [8, 9],
        "gpus": [],
        "virtual_environment": True,
    }
    installer = FakeInstaller(versions)
    probe = Mock(
        side_effect=lambda: {
            name: {"importable": name in versions, "kernel_tested": True, "error": None} for name in CATALOG
        }
    )
    manager = EnvironmentManager(
        context,
        installer=installer,
        versions=lambda: versions.copy(),
        runtime=lambda: runtime.copy(),
        probe=probe,
    )
    app = FastAPI()
    app.state.environment = manager
    app.include_router(router, prefix="/api")
    install_errors(app)
    client = TestClient(app)
    yield SimpleNamespace(
        client=client,
        manager=manager,
        context=context,
        versions=versions,
        runtime=runtime,
        installer=installer,
        probe=probe,
    )
    manager.close()
    client.close()
    db.close()


def wait_status(manager, id_, expected=("ready", "completed", "failed", "cancelled")):
    until = time.monotonic() + 3
    while time.monotonic() < until:
        op = manager.get(id_)
        if op.status in expected:
            return op
        time.sleep(0.01)
    raise AssertionError(manager.get(id_))


def test_extension_plans_cannot_be_applied_or_uninstalled_from_another_profile(env, monkeypatch):
    env.versions["tensorboard"] = "1.0"
    op = start(env, action="uninstall")
    assert op.status == "ready"
    calls = list(env.installer.calls)
    monkeypatch.setenv("YPUDDIN_ENV_PROFILE", "windows-cpu")
    other = EnvironmentManager(
        env.context,
        installer=env.installer,
        versions=lambda: env.versions.copy(),
        runtime=lambda: env.runtime.copy(),
        probe=env.probe,
    )
    try:
        assert other.list() == []
        with pytest.raises(EnvironmentError):
            other.apply(op.id)
        assert other.root != env.manager.root
        assert env.installer.calls == calls
        assert env.versions["tensorboard"] == "1.0"
        assert env.manager.get(op.id).status == "ready"
    finally:
        other.close()


def test_extension_plan_cannot_apply_after_same_profile_interpreter_switch(env, monkeypatch):
    op = start(env)
    calls = list(env.installer.calls)
    monkeypatch.setattr(sys, "executable", "/different/profile-runtime/bin/python")
    with pytest.raises(EnvironmentError, match="another interpreter"):
        env.manager.apply(op.id)
    assert env.installer.calls == calls


def start(env, **fields):
    response = env.client.post("/api/environment/operations", json={"package": "tensorboard", **fields})
    assert response.status_code == 202, response.text
    return wait_status(env.manager, response.json()["id"])


def job(env, status="running"):
    env.context.db.insert(
        "jobs", {"id": "job", "name": "job", "type": "train", "status": status, "created_at": now()}
    )


def test_reviewed_install_does_not_mutate_until_apply_then_requires_restart(env):
    op = start(env)
    assert op.status == "ready"
    assert "tensorboard" not in env.versions
    assert op.plan[0]["version"] == "1.0"
    assert not maintenance_blocked(env.context.db)
    assert env.client.post(f"/api/environment/operations/{op.id}/apply").status_code == 202
    done = wait_status(env.manager, op.id, ("completed", "failed"))
    assert done.status == "completed", done.error
    assert env.versions == {"torch": "2.5.1", "numpy": "2.1.0", "tensorboard": "1.0"}
    assert done.restart_required and maintenance_blocked(env.context.db)
    install_args = next(args for args in env.installer.calls if "--no-index" in args)
    assert "--no-deps" in install_args

    download_args = next(args for args in env.installer.calls if "download" in args)
    assert "--require-hashes" in download_args
    # Same database, fresh server process releases maintenance.
    env.manager.close()
    manager = EnvironmentManager(
        env.context,
        installer=env.installer,
        versions=lambda: env.versions.copy(),
        runtime=lambda: env.runtime.copy(),
        probe=env.probe,
    )
    assert not maintenance_blocked(env.context.db)
    manager.close()


@pytest.mark.parametrize("pipeline_status", ["queued", "running", "cancelling"])
def test_environment_apply_and_probe_wait_for_local_tagger_under_shared_lock(env, pipeline_status):
    op = start(env)
    env.context.db.execute(
        "CREATE TABLE dataset_pipeline_operations (id TEXT PRIMARY KEY, action TEXT, status TEXT, job_id TEXT)"
    )
    entered, finished = threading.Event(), threading.Event()
    errors = []

    def apply():
        entered.set()
        try:
            env.manager.apply(op.id)
        except Exception as error:
            errors.append(error)
        finally:
            finished.set()

    with env.context.db.lock:
        env.context.db.insert(
            "dataset_pipeline_operations",
            {"id": "tag", "action": "tag", "status": pipeline_status, "job_id": None},
        )
        worker = threading.Thread(target=apply)
        worker.start()
        assert entered.wait(1) and not finished.wait(0.05)
    worker.join(3)
    assert finished.is_set() and errors and errors[0].status == 409
    assert "worker" in str(errors[0])
    assert env.manager.get(op.id).status == "ready" and "tensorboard" not in env.versions
    state = env.client.get("/api/environment?refresh=true").json()
    assert state["running_jobs"] and state["probe_deferred"]
    env.probe.assert_not_called()
    assert env.client.post("/api/environment/operations", json={"package": "tensorboard"}).status_code == 409
    env.context.db.update("dataset_pipeline_operations", "tag", {"status": "completed"})
    assert env.client.post(f"/api/environment/operations/{op.id}/apply").status_code == 202
    assert wait_status(env.manager, op.id, ("completed", "failed")).status == "completed"


def test_runtime_probes_cleanup_import_side_effects_in_disposable_cwd(monkeypatch):
    import ypuddin.server.environment as module

    directories = []

    def run(args, **kwargs):
        directory = Path(kwargs["cwd"])
        assert directory != Path.cwd()
        directories.append(directory)
        (directory / ":memory:.ses").write_text("simulated native import side effect")
        return SimpleNamespace(
            stdout='YPUDDIN_ENV={"xformers":{"importable":true}}\n', stderr="", returncode=0
        )

    monkeypatch.setattr(module.subprocess, "run", run)
    assert probe_packages()["xformers"]["importable"]
    assert directories and not directories[0].exists()


def run_xformers_probe(
    monkeypatch, capsys, error, *, capability=(12, 0), import_error=False, sdpa_error=None
):
    tensor = Mock()
    attention = Mock(side_effect=error, return_value=tensor)
    sdpa = Mock(side_effect=sdpa_error, return_value=tensor)
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda: capability,
            synchronize=Mock(),
        ),
        randn=Mock(return_value=tensor),
        float16="float16",
        nn=SimpleNamespace(functional=SimpleNamespace(scaled_dot_product_attention=sdpa)),
    )

    def load(name):
        if name == "xformers.ops":
            if import_error:
                raise error
            return SimpleNamespace(memory_efficient_attention=attention)
        raise ModuleNotFoundError(name)

    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "torch", torch)
        patch.setattr(importlib, "import_module", load)
        exec(PROBE, {})
    output = capsys.readouterr().out.split("YPUDDIN_ENV=", 1)[1]
    return json.loads(output)["xformers"], sdpa, tensor


@pytest.mark.parametrize(
    "error,capability",
    [
        (NotImplementedError("No operator found for `memory_efficient_attention_forward`"), (12, 0)),
        (
            RuntimeError(
                "requires device with capability <= (9, 0) but your GPU has capability (12, 0) (too new)"
            ),
            (12, 0),
        ),
        (RuntimeError("CUDA error: no kernel image is available for execution on the device"), (8, 9)),
        (NotImplementedError("No operator found for `memory_efficient_attention_forward`"), (7, 5)),
    ],
)
def test_xformers_unavailable_kernel_reports_installed_package_and_tested_sdpa(
    monkeypatch, capsys, error, capability
):
    result, sdpa, tensor = run_xformers_probe(monkeypatch, capsys, error, capability=capability)
    assert result["importable"] and not result["kernel_tested"] and result["kernel_unavailable"]
    assert "已安装并可导入" in result["error"]
    assert "匹配当前 Python、PyTorch、CUDA 且支持此 GPU 的 FlashAttention 2 wheel" in result["error"]
    assert f"SM{capability[0]}{capability[1]}" in result["error"]
    if capability != (12, 0):
        assert "SM120" not in result["error"]
    assert "SDPA 正反向检测通过" in result["error"]
    assert str(error) in result["error"]
    sdpa.assert_called_once()
    tensor.float.return_value.sum.return_value.backward.assert_called_once()


@pytest.mark.parametrize(
    "error,import_error",
    [
        (RuntimeError("CUDA out of memory"), False),
        (RuntimeError("CUDA driver version is insufficient for CUDA runtime version"), False),
        (RuntimeError("CUDA error: an illegal memory access was encountered"), False),
        (RuntimeError("No operator found for an_unrelated_operation"), False),
        (RuntimeError("requires device with capability >= (8, 0), device (7, 5) is too old"), False),
        (ImportError("DLL load failed while importing _C"), True),
        (ImportError("No operator found for `memory_efficient_attention_forward` during import"), True),
    ],
)
def test_xformers_other_failures_keep_original_diagnosis(monkeypatch, capsys, error, import_error):
    result, sdpa, _ = run_xformers_probe(monkeypatch, capsys, error, import_error=import_error)
    assert result["importable"] is not import_error
    assert not result["kernel_tested"]
    assert not result.get("kernel_unavailable")
    assert result["error"] == str(error)
    sdpa.assert_not_called()


def test_xformers_kernel_failure_does_not_claim_sdpa_available_when_fallback_probe_fails(monkeypatch, capsys):
    result, sdpa, _ = run_xformers_probe(
        monkeypatch,
        capsys,
        NotImplementedError("No operator found for `memory_efficient_attention_forward`"),
        sdpa_error=RuntimeError("CUDA out of memory"),
    )
    assert result["kernel_unavailable"]
    assert "SDPA 检测也未通过" in result["error"]
    assert "CUDA out of memory" in result["error"]
    assert "可继续使用 SDPA" not in result["error"]
    sdpa.assert_called_once()


def test_xformers_success_does_not_run_fallback_probe(monkeypatch, capsys):
    result, sdpa, tensor = run_xformers_probe(monkeypatch, capsys, None)
    assert result == {"importable": True, "kernel_tested": True, "error": None}
    tensor.float.return_value.sum.return_value.backward.assert_called_once()
    sdpa.assert_not_called()


@pytest.mark.parametrize("package", ["torch", "arbitrary-package", "tensorboard;curl bad"])
def test_api_has_a_package_allowlist(env, package):
    assert env.client.post("/api/environment/operations", json={"package": package}).status_code == 422
    assert not env.installer.calls


def test_protected_framework_plan_is_rejected(env):
    env.installer.rows = [FakeInstaller.row("torch", "9.0"), FakeInstaller.row("tensorboard", "1.0")]
    op = start(env)
    assert op.status == "failed"
    assert "protected runtime torch" in op.error
    assert env.client.post(f"/api/environment/operations/{op.id}/apply").status_code == 409
    assert env.versions["torch"] == "2.5.1"


def test_dependency_plan_rejects_source_archives_and_external_sources(env):
    row = FakeInstaller.row("tensorboard", "1.0")
    row["download_info"]["url"] = "https://attacker.example/tensorboard.whl"
    env.installer.rows = [row]
    assert "unsupported wheel source" in start(env).error
    row["download_info"]["url"] = "https://files.pythonhosted.org/packages/tensorboard.tar.gz"
    assert "Source builds are disabled" in start(env).error


def test_corrupt_download_cannot_be_installed(env):
    op = start(env)
    env.installer.corrupt = True
    env.manager.apply(op.id)
    done = wait_status(env.manager, op.id, ("failed", "completed"))
    assert done.status == "failed" and "digest" in done.error
    assert not any("--no-index" in args for args in env.installer.calls)
    assert not maintenance_blocked(env.context.db)


def test_failed_retry_keeps_an_existing_restart_requirement(env):
    env.context.db.set_kv("environment.maintenance", {"blocked": True, "restart_required": True})
    op = start(env)
    env.installer.corrupt = True
    env.manager.apply(op.id)
    assert wait_status(env.manager, op.id, ("failed",)).status == "failed"
    assert maintenance_blocked(env.context.db)
    assert env.manager.status()["restart_required"]


def test_failed_mutation_requires_restart_and_has_concrete_error(env):
    op = start(env)
    env.installer.fail_install = True
    env.manager.apply(op.id)
    done = wait_status(env.manager, op.id, ("failed", "completed"))
    assert done.status == "failed" and "locked DLL" in done.error
    assert done.restart_required
    assert maintenance_blocked(env.context.db)


def test_runtime_probe_failure_never_reports_usable_install(env):
    op = start(env)
    env.probe.side_effect = lambda: {"tensorboard": {"importable": False, "error": "broken import"}}
    env.manager.apply(op.id)
    done = wait_status(env.manager, op.id, ("failed", "completed"))
    assert done.status == "failed" and "broken import" in done.error
    assert done.restart_required


def test_installed_xformers_with_unavailable_kernel_stays_failed_and_cannot_become_default(env, monkeypatch):
    monkeypatch.setattr(env.manager, "validate_wheel", Mock())
    op = start(env, package="xformers")
    diagnostic = "xFormers 已安装并可导入，但当前 wheel 没有可用于此 GPU（SM120）的注意力计算内核。"
    env.probe.side_effect = lambda: {
        "xformers": {
            "importable": True,
            "kernel_tested": False,
            "kernel_unavailable": True,
            "error": diagnostic,
        }
    }
    env.manager.apply(op.id)
    done = wait_status(env.manager, op.id, ("failed", "completed"))
    assert done.status == "failed" and done.error == diagnostic
    assert done.restart_required
    assert "xformers" in env.versions
    package = next(
        p for p in env.client.get("/api/environment").json()["packages"] if p["name"] == "xformers"
    )
    assert package["importable"] and not package["available"] and not package["kernel_tested"]
    assert package["error"] == diagnostic
    assert (
        env.client.put("/api/environment/settings", json={"attention_default": "xformers"}).status_code == 422
    )
    assert environment_attention_default(env.context) == "auto"


def test_training_or_cache_blocks_plan_and_apply(env):
    op = start(env)
    job(env)
    assert env.client.post(f"/api/environment/operations/{op.id}/apply").status_code == 409
    assert env.client.post("/api/environment/operations", json={"package": "tensorboard"}).status_code == 409
    # A terminal DB status does not authorize modification until the process really exits.
    env.context.db.update("jobs", "job", {"status": "paused"})
    env.context.supervisor._procs["job"] = Mock(poll=lambda: None)
    assert env.client.post(f"/api/environment/operations/{op.id}/apply").status_code == 409


def test_maintenance_prevents_supervisor_launch(env, monkeypatch):
    job(env, status="queued")
    env.context.db.set_kv("environment.maintenance", {"blocked": True})
    launch = Mock()
    monkeypatch.setattr(env.context.supervisor, "_launch", launch)
    env.context.supervisor._tick()
    launch.assert_not_called()


def test_environment_change_invalidates_ready_plan(env):
    op = start(env)
    env.versions["torch"] = "2.6.0"
    result = env.client.post(f"/api/environment/operations/{op.id}/apply")
    assert result.status_code == 409 and "Environment changed" in result.text


def test_windows_flash_and_sage_require_explicit_compatible_wheel(env):
    for package in ("flash-attn", "sageattention"):
        response = env.client.post("/api/environment/operations", json={"package": package})
        assert response.status_code == 422 and "prebuilt wheel" in response.text
    assert not env.installer.calls


def test_uninstall_selected_backend_resets_default(env):
    env.versions["xformers"] = "1.0"
    assert (
        env.client.put("/api/environment/settings", json={"attention_default": "xformers"}).status_code == 200
    )
    op = start(env, package="xformers", action="uninstall")
    env.manager.apply(op.id)
    done = wait_status(env.manager, op.id, ("completed", "failed"))
    assert done.status == "completed", done.error
    assert environment_attention_default(env.context) == "auto"
    assert "xformers" not in env.versions


def test_defaults_require_real_available_backend_and_cpu_status_is_truthful(env):
    assert (
        env.client.put("/api/environment/settings", json={"attention_default": "xformers"}).status_code == 422
    )
    env.runtime["cuda_available"] = False
    env.versions["xformers"] = "1.0"
    status = env.client.get("/api/environment").json()
    package = next(p for p in status["packages"] if p["name"] == "xformers")
    assert package["version"] == "1.0" and not package["available"]
    assert package["reason"] == "requires_cuda"


def test_attention_default_reaches_new_projects_without_overwriting_saved_choice(env):
    from ypuddin.server.routes_core import router as core_router
    from ypuddin.server.routes_work import router as work_router

    env.client.app.state.ctx = env.context
    env.client.app.include_router(core_router, prefix="/api")
    env.client.app.include_router(work_router, prefix="/api")
    existing = env.client.post("/api/projects", json={"name": "existing"}).json()["id"]
    saved = env.client.get(f"/api/projects/{existing}/config").json()
    saved["model"]["attention"] = "auto"
    assert env.client.put(f"/api/projects/{existing}/config", json=saved).status_code == 200
    assert env.client.put("/api/environment/settings", json={"attention_default": "sdpa"}).status_code == 200
    assert env.client.get("/api/config/defaults").json()["model"]["attention"] == "sdpa"
    new = env.client.post("/api/projects", json={"name": "new"}).json()["id"]
    assert env.client.get(f"/api/projects/{new}/config").json()["model"]["attention"] == "sdpa"
    assert env.client.get(f"/api/projects/{existing}/config").json()["model"]["attention"] == "auto"


def test_running_job_defers_cuda_probe(env):
    job(env)
    status = env.client.get("/api/environment?refresh=true").json()
    assert status["probe_deferred"]
    env.probe.assert_not_called()


def test_probe_holds_launch_maintenance_and_restores_existing_state(env):
    observed = []
    env.probe.side_effect = lambda: observed.append(maintenance_blocked(env.context.db)) or {}
    env.manager.status()
    assert observed == [True]
    assert not maintenance_blocked(env.context.db)
    env.context.db.set_kv("environment.maintenance", {"blocked": True, "restart_required": True})
    env.manager.status(refresh=True)
    assert maintenance_blocked(env.context.db)


def test_cancel_plan_and_uninstall_require_existing_package(env):
    op = start(env)
    assert env.client.post(f"/api/environment/operations/{op.id}/cancel").json()["status"] == "cancelled"
    assert env.client.post(f"/api/environment/operations/{op.id}/apply").status_code == 409
    assert (
        env.client.post(
            "/api/environment/operations", json={"package": "wandb", "action": "uninstall"}
        ).status_code
        == 422
    )


@pytest.mark.parametrize("package", ["wandb", "onnxruntime", "onnxruntime-gpu"])
def test_removed_external_logging_and_automatic_tagging_are_not_installable(env, package):
    status = env.client.get("/api/environment").json()
    assert package not in {item["name"] for item in status["packages"]}
    assert env.client.post("/api/environment/operations", json={"package": package}).status_code == 422
    assert not env.installer.calls


def wheel_bytes(tmp_path, filename, name, *, version="1.0", requires=()):
    path = tmp_path / filename
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{name}-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
            + "".join(f"Requires-Dist: {req}\n" for req in requires),
        )
    return path.read_bytes()


def test_wheel_upload_validates_contents_tags_and_paths(env, tmp_path):
    filename = "tensorboard-1.0-py3-none-any.whl"
    content = wheel_bytes(tmp_path, filename, "tensorboard")
    response = env.client.post("/api/environment/wheels", files={"file": (filename, content)})
    assert response.status_code == 201, response.text
    assert response.json()["package"] == "tensorboard"
    assert "path" not in response.json()
    for name in ("../" + filename, "..\\" + filename, "setup.py"):
        assert env.client.post("/api/environment/wheels", files={"file": (name, content)}).status_code == 422
    mismatched = wheel_bytes(tmp_path, filename, "torch")
    assert (
        env.client.post("/api/environment/wheels", files={"file": (filename, mismatched)}).status_code == 422
    )
    wrong_abi = "tensorboard-1.0-cp399-cp399-win_amd64.whl"
    assert env.client.post("/api/environment/wheels", files={"file": (wrong_abi, content)}).status_code == 422


def test_accelerator_wheel_requires_verifiable_torch_cuda_identity(env, tmp_path):
    tag = next(sys_tags())
    filename = f"flash_attn-1.0-{tag}.whl"
    content = wheel_bytes(tmp_path, filename, "flash-attn")
    error = env.client.post("/api/environment/wheels", files={"file": (filename, content)})
    assert error.status_code == 422 and "Cannot verify" in error.text
    bad_version = "1.0+cu126torch2.5"
    bad_file = f"flash_attn-{bad_version}-{tag}.whl"
    content = wheel_bytes(tmp_path, bad_file, "flash-attn", version=bad_version)
    error = env.client.post("/api/environment/wheels", files={"file": (bad_file, content)})
    assert error.status_code == 422 and "CUDA build tag" in error.text
    good_version = "1.0+cu128torch2.5"
    good_file = f"flash_attn-{good_version}-{tag}.whl"
    content = wheel_bytes(tmp_path, good_file, "flash-attn", version=good_version)
    result = env.client.post("/api/environment/wheels", files={"file": (good_file, content)})
    assert result.status_code == 201, result.text
    content = wheel_bytes(tmp_path, filename, "flash-attn", requires=["torch==2.9.0"])
    error = env.client.post("/api/environment/wheels", files={"file": (filename, content)})
    assert error.status_code == 422 and "current protected runtime" in error.text


def test_xformers_uses_compiled_cuda_metadata_and_declared_torch_range(env, tmp_path):
    filename = f"xformers-1.0-{next(sys_tags())}.whl"
    path = tmp_path / filename
    for cuda, expected in ((1208, 201), (1206, 422)):
        wheel_bytes(tmp_path, filename, "xformers", requires=["torch>=2.4"])
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr(
                "xformers/cpp_lib.json", json.dumps({"version": {"cuda": cuda, "torch": "2.4.0+cu128"}})
            )
        response = env.client.post("/api/environment/wheels", files={"file": (filename, path.read_bytes())})
        assert response.status_code == expected, response.text


def test_request_cannot_inject_pip_flags_or_nonversion_requirements():
    for value in ("--extra-index-url=http://bad", "git+http://bad", "1.0; arbitrary", ">=1"):
        with pytest.raises(ValueError):
            EnvironmentRequest(package="tensorboard", version=value)


def test_regularization_reservation_and_environment_apply_share_admission_lock(env):
    op = start(env)
    entered, completed = threading.Event(), threading.Event()
    errors = []

    def apply():
        entered.set()
        try:
            env.manager.apply(op.id)
        except Exception as error:
            errors.append(error)
        finally:
            completed.set()

    with env.context.db.lock:
        env.context.db.set_kv("regularization.reservation", {"id": "reg_pending"})
        worker = threading.Thread(target=apply)
        worker.start()
        assert entered.wait(1) and not completed.wait(0.05)
    worker.join(3)
    assert completed.is_set() and errors and errors[0].status == 409
    assert "regularization" in str(errors[0])
    assert maintenance_blocked(env.context.db)
    assert not any("install" in call and "--dry-run" not in call for call in env.installer.calls)


def test_dismissed_result_is_persistent_but_preserves_history(env):
    from ypuddin.server.environment import EnvironmentOperation

    op = EnvironmentOperation(
        id="env_dismiss_test",
        package="xformers",
        action="install",
        status="failed",
        created_at=now(),
        updated_at=now(),
        error="old kernel failure",
        logs=["keep diagnostic log"],
    )
    env.context.db.set_kv("environment.operation." + op.id, op.model_dump())
    response = env.client.post(f"/api/environment/operations/{op.id}/dismiss")
    assert response.status_code == 200
    stored = env.context.db.get_kv("environment.operation." + op.id)
    assert stored["dismissed_at"] > 0
    assert stored["status"] == "failed"
    assert stored["error"] == "old kernel failure"
    assert stored["logs"] == ["keep diagnostic log"]
    assert env.client.get("/api/environment/operations").json()[0]["dismissed_at"] == stored["dismissed_at"]


@pytest.mark.parametrize("status", ["planning", "ready", "installing", "verifying"])
def test_cannot_dismiss_unfinished_install(env, status):
    from ypuddin.server.environment import EnvironmentOperation

    op = EnvironmentOperation(
        id="env_busy_dismiss",
        package="xformers",
        action="install",
        status=status,
        created_at=now(),
        updated_at=now(),
    )
    env.context.db.set_kv("environment.operation." + op.id, op.model_dump())
    assert env.client.post(f"/api/environment/operations/{op.id}/dismiss").status_code == 409
    assert env.manager.get(op.id).dismissed_at is None
