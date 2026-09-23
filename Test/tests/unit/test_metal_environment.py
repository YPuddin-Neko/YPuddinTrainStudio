"""Apple extension admission, immutable runtime, reviewed wheels and real probe contract."""

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from packaging.utils import parse_wheel_filename

from Test.tests.unit.test_environment import env as env
from Test.tests.unit.test_environment import wait_status, wheel_bytes
from ypuddin.server import metal_attention_catalog as metal
from ypuddin.server.environment import EnvironmentRequest, environment_attention_default
from ypuddin.server.torch_environments import OPTIONAL_EXTENSIONS


@pytest.fixture
def apple(env, monkeypatch, tmp_path):
    env.manager.profile = "macos-mps"
    env.runtime.update(
        platform="Darwin", machine="arm64", python="3.12.12", macos_version="15.7.9",
        torch="2.13.0", cuda_runtime=None, hip_runtime=None, cuda_available=False,
        mps_available=True, gpu_capability=None,
    )
    env.versions["torch"] = "2.13.0"
    filename = "mtlattn-0.4.1-cp312-cp312-macosx_13_0_universal2.whl"
    content = wheel_bytes(tmp_path, filename, "mtlattn", version="0.4.1", requires=["torch"])
    sha = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(metal, "WHEELS", {filename: (len(content), sha)})
    monkeypatch.setattr("ypuddin.server.environment.sys_tags", lambda: parse_wheel_filename(filename)[3])
    env.installer.rows = [{
        "metadata": {"name": "mtlattn", "version": "0.4.1", "requires_dist": ["torch"]},
        "download_info": {
            "url": "https://files.pythonhosted.org/packages/" + filename,
            "archive_info": {"hashes": {"sha256": sha}},
        },
    }]
    original_run = env.installer.run

    def run(args, log, cancel, **kwargs):
        original_run(args, log, cancel, **kwargs)
        if "download" in args:
            (Path(args[args.index("--dest") + 1]) / filename).write_bytes(content)

    monkeypatch.setattr(env.installer, "run", run)
    env.metal_filename, env.metal_content = filename, content
    return env


def test_metal_package_version_is_pinned_and_other_versions_rejected():
    assert EnvironmentRequest(package="mtlattn").version == "0.4.1"
    assert "mtlattn" in OPTIONAL_EXTENSIONS
    for version in ["0.4.0", "0.4.2", "0.4.1+unreviewed"]:
        with pytest.raises(ValueError, match="only mtlattn 0.4.1"):
            EnvironmentRequest(package="mtlattn", version=version)


@pytest.mark.parametrize("changes,profile,reason", [
    ({"platform": "Windows"}, "macos-mps", "metal_requires_apple_silicon"),
    ({"machine": "x86_64"}, "macos-mps", "metal_requires_apple_silicon"),
    ({"hip_runtime": "6.3"}, "macos-mps", "metal_requires_apple_silicon"),
    ({}, "macos-cpu", "metal_requires_mps_profile"),
    ({"macos_version": "14.7"}, "macos-mps", "metal_requires_macos_15"),
    ({"python": "3.10.16"}, "macos-mps", "metal_requires_python_311_312"),
    ({"torch": "2.14.0"}, "macos-mps", "metal_requires_torch_2_13"),
    ({"mps_available": False}, "macos-mps", "metal_requires_mps"),
])
def test_metal_runtime_admission_is_shared_by_status_and_install(apple, changes, profile, reason):
    apple.runtime.update(changes)
    apple.manager.profile = profile
    status = next(p for p in apple.manager.status()["packages"] if p["name"] == "mtlattn")
    assert not status["supported"] and not status["available"] and status["reason"] == reason
    response = apple.client.post("/api/environment/operations", json={"package": "mtlattn"})
    assert response.status_code == 422 and metal.MESSAGES[reason] in response.text
    assert apple.installer.calls == []


def test_reviewed_metal_install_protects_torch_and_requires_kernel_before_default(apple):
    before = apple.versions.copy()
    response = apple.client.post("/api/environment/operations", json={"package": "mtlattn"})
    assert response.status_code == 202, response.text
    op = wait_status(apple.manager, response.json()["id"])
    assert op.status == "ready", op.error
    assert op.plan[0]["filename"] == apple.metal_filename
    plan_call = apple.installer.calls[0]
    assert plan_call[-1] == "mtlattn==0.4.1" and "--only-binary=:all:" in plan_call
    constraints = Path(plan_call[plan_call.index("--constraint") + 1]).read_text()
    assert "torch===2.13.0" in constraints and apple.versions == before
    apple.manager.apply(op.id)
    assert wait_status(apple.manager, op.id).status == "completed"
    assert apple.versions == {**before, "mtlattn": "0.4.1"}
    status = next(p for p in apple.manager.status()["packages"] if p["name"] == "mtlattn")
    assert status["available"] and status["backend"] == "metal_flash" and not status["wheel_required"]
    assert apple.client.put("/api/environment/settings", json={"attention_default": "metal_flash"}).status_code == 200


def test_installed_metal_with_failed_backward_is_not_available_or_default(apple):
    apple.versions["mtlattn"] = "0.4.1"
    cached = next(p for p in apple.manager.status(refresh=True)["packages"] if p["name"] == "mtlattn")
    assert cached["available"] and cached["kernel_tested"]
    apple.probe.reset_mock()
    apple.probe.side_effect = lambda: {"mtlattn": {
        "importable": True, "kernel_tested": False, "error": "native backward failed",
    }}
    status = next(p for p in apple.manager.status(refresh=True)["packages"] if p["name"] == "mtlattn")
    assert status["supported"] and status["importable"] and not status["available"]
    assert not status["kernel_tested"] and status["error"] == "native backward failed"
    apple.probe.assert_called_once()
    assert next(p for p in apple.manager.status()["packages"] if p["name"] == "mtlattn") == status
    apple.probe.assert_called_once()
    assert apple.client.put("/api/environment/settings", json={"attention_default": "metal_flash"}).status_code == 422
    assert apple.probe.call_count == 2


def test_metal_upload_accepts_reviewed_apple_wheel_without_cuda_tags_and_rejects_tampering(apple):
    response = apple.client.post("/api/environment/wheels", files={
        "file": (apple.metal_filename, apple.metal_content),
    })
    assert response.status_code == 201, response.text
    assert response.json()["package"] == "mtlattn"
    response = apple.client.post("/api/environment/wheels", files={
        "file": (apple.metal_filename, apple.metal_content + b"tampered"),
    })
    assert response.status_code == 422 and "SHA256" in response.text


@pytest.mark.parametrize("bad", ["hash", "torch", "future_version"])
def test_metal_plan_rejects_unreviewed_wheels_or_protected_framework_changes(apple, bad):
    if bad == "hash":
        apple.installer.rows[0]["download_info"]["archive_info"]["hashes"]["sha256"] = "0" * 64
    elif bad == "future_version":
        apple.installer.rows[0]["metadata"]["version"] = "0.4.2"
    else:
        apple.installer.rows.append({
            "metadata": {"name": "torch", "version": "2.12.0"}, "download_info": {},
        })
    before = apple.versions.copy()
    op = apple.manager.start(EnvironmentRequest(package="mtlattn"))
    assert wait_status(apple.manager, op.id).status == "failed"
    assert apple.versions == before and all("--dry-run" in call for call in apple.installer.calls)


def test_incompatible_runtime_can_uninstall_metal_and_clear_its_default(apple):
    apple.versions["mtlattn"] = "0.4.1"
    apple.context.db.set_kv("environment.settings", {"attention_default": "metal_flash"})
    apple.runtime["torch"] = "2.14.0"
    op = apple.manager.start(EnvironmentRequest(package="mtlattn", action="uninstall"))
    assert wait_status(apple.manager, op.id).status == "ready"
    apple.manager.apply(op.id)
    assert wait_status(apple.manager, op.id).status == "completed"
    assert "mtlattn" not in apple.versions
    assert apple.context.db.get_kv("environment.settings")["attention_default"] == "auto"


def test_metal_default_tracks_runtime_without_rewriting_saved_preference_or_configs(apple, monkeypatch):
    import importlib.metadata

    selected = {"profile": "macos-mps", "package": "0.4.1"}
    monkeypatch.setattr("ypuddin.server.environment.current_profile", lambda: selected["profile"])
    runtime = Mock(side_effect=lambda: apple.runtime.copy())
    monkeypatch.setattr("ypuddin.server.environment.runtime_info", runtime)

    def version(name):
        assert name == "mtlattn"
        if selected["package"] is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return selected["package"]

    monkeypatch.setattr("ypuddin.server.environment.importlib.metadata.version", version)
    saved_default = {"attention_default": "metal_flash"}
    saved_config = {"model": {"family": "anima", "attention": "metal_flash"}}
    apple.context.db.set_kv("environment.settings", saved_default)
    apple.context.db.set_kv("test.saved_model_config", saved_config)
    assert environment_attention_default(apple.context) == "metal_flash"
    selected["profile"] = "macos-cpu"
    runtime.reset_mock()
    assert environment_attention_default(apple.context) == "auto"
    runtime.assert_not_called()  # CPU defaults do not inspect an accelerator.
    selected["profile"] = "macos-mps"
    apple.runtime["torch"] = "2.14.0"
    assert environment_attention_default(apple.context) == "auto"
    apple.runtime["torch"] = "2.13.0"
    selected["package"] = None
    assert environment_attention_default(apple.context) == "auto"
    selected["package"] = "0.4.2"
    assert environment_attention_default(apple.context) == "auto"
    selected["package"] = "0.4.1"
    assert environment_attention_default(apple.context) == "metal_flash"
    assert apple.context.db.get_kv("environment.settings") == saved_default
    assert apple.context.db.get_kv("test.saved_model_config") == saved_config


@pytest.mark.parametrize("value", ["auto", "sdpa", "xformers", "flash_attn", "sage"])
def test_other_attention_defaults_do_not_gain_metal_runtime_checks(apple, monkeypatch, value):
    runtime = Mock(side_effect=AssertionError("Existing defaults do not require a runtime probe"))
    monkeypatch.setattr("ypuddin.server.environment.runtime_info", runtime)
    apple.context.db.set_kv("environment.settings", {"attention_default": value})
    assert environment_attention_default(apple.context) == value
    runtime.assert_not_called()


class _VarlenAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        return q + k + v

    @staticmethod
    def backward(ctx, grad):
        return grad, grad, grad


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.setattr(metal, "current_profile", lambda: "macos-mps")
    allocated = []

    def randn(*shape, **kwargs):
        assert kwargs.pop("device") == "mps" and kwargs["dtype"] == torch.float32
        tensor = torch.randn(*shape, **kwargs)
        allocated.append(tensor)
        return tensor

    fake = SimpleNamespace(
        randn=randn, enable_grad=torch.enable_grad, float32=torch.float32,
        isfinite=torch.isfinite, mps=SimpleNamespace(synchronize=Mock()),
    )
    core = SimpleNamespace(
        require_metal_flash=Mock(return_value={}), metal_flash_eligible=Mock(return_value=True),
        metal_flash_sdpa=Mock(side_effect=_VarlenAttnFn.apply),
    )
    monkeypatch.setitem(sys.modules, "ypuddin.models.metal_attention", core)
    return fake, core, allocated


def test_metal_probe_requires_native_backward_and_finite_qkv_gradients(probe):
    fake, core, tensors = probe
    result = metal.probe_metal_attention(fake)
    assert result == {"importable": True, "kernel_tested": True, "error": None}
    assert len(tensors) == 6 and all(t.grad is not None and torch.isfinite(t.grad).all() for t in tensors)
    assert [tuple(tensors[i].shape) for i in (0, 3)] == [(1, 2, 32, 64), (1, 2, 64, 128)]
    assert core.metal_flash_sdpa.call_count == fake.mps.synchronize.call_count == 2


def test_metal_probe_does_not_claim_success_for_native_sdpa_fallback(probe):
    fake, core, tensors = probe
    core.metal_flash_sdpa.side_effect = lambda q, k, v: q + k + v
    result = metal.probe_metal_attention(fake)
    assert result["importable"] and not result["kernel_tested"]
    assert "native varlen backward" in result["error"] and len(tensors) == 3


def test_metal_probe_runtime_failure_stops_before_allocating_gpu_tensors(probe):
    fake, core, tensors = probe
    core.require_metal_flash.side_effect = RuntimeError("PyTorch 2.13.x required")
    result = metal.probe_metal_attention(fake)
    assert not result["importable"] and not result["kernel_tested"]
    assert "2.13" in result["error"] and not tensors


@pytest.mark.parametrize("profile", ["macos-cpu", "windows-cpu", "linux-cpu"])
def test_metal_probe_never_imports_or_allocates_for_explicit_cpu(probe, monkeypatch, profile):
    fake, core, tensors = probe
    monkeypatch.setattr(metal, "current_profile", lambda: profile)
    result = metal.probe_metal_attention(fake)
    assert not result["kernel_tested"] and not tensors
    core.require_metal_flash.assert_not_called()
