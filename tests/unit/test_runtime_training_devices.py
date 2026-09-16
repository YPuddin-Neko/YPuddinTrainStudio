from types import SimpleNamespace

import pytest

from ypuddin.server.environment import runtime_info


@pytest.mark.parametrize(
    "system,profile,cuda,count,distributed,nccl,gloo,hip,expected_multi",
    [
        ("Windows", "windows-cuda", True, 2, True, True, True, None, True),
        ("Windows", "windows-cuda", True, 2, True, False, True, None, True),
        ("Windows", "windows-cuda", True, 2, True, False, False, None, False),
        ("Windows", "windows-cuda", True, 2, True, True, True, "6.3", False),
        ("Windows", "windows-cpu", False, 8, True, False, True, None, False),
        ("Windows", "windows-cpu", True, 2, True, True, True, None, False),
        ("Windows", "windows-cuda", False, 0, False, False, True, None, False),
        ("Linux", "linux-cuda", True, 2, True, True, True, None, True),
        ("Linux", "linux-cuda", True, 1, True, True, True, None, False),
        ("Linux", "linux-cuda", True, 2, False, True, True, None, False),
        ("Linux", "linux-cuda", True, 2, True, False, True, None, False),
        ("Linux", "linux-dtk", True, 2, True, True, True, "6.3.26093", True),
        ("Linux", "linux-cpu", True, 2, True, True, True, None, False),
        ("Linux", "linux-cpu", False, 8, True, False, True, None, False),
        ("Darwin", "macos-mps", False, 0, False, False, False, None, False),
    ],
)
def test_runtime_distinguishes_pytorch_build_capabilities_from_trainer_support(
    monkeypatch, system, profile, cuda, count, distributed, nccl, gloo, hip, expected_multi
):
    import torch

    from ypuddin.server import environment, hardware

    monkeypatch.setattr(
        environment,
        "platform",
        SimpleNamespace(system=lambda: system, python_version=lambda: "3.12.10", machine=lambda: "AMD64"),
    )
    monkeypatch.setattr(environment, "current_profile", lambda: profile)
    monkeypatch.setattr(environment.dtk_catalog, "system_info", lambda: {})
    monkeypatch.setattr(environment.dtk_catalog, "driver_version", lambda: None)
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))
    monkeypatch.setattr(
        torch,
        "distributed",
        SimpleNamespace(
            is_available=lambda: distributed, is_nccl_available=lambda: nccl, is_gloo_available=lambda: gloo
        ),
    )
    monkeypatch.setattr(hardware, "gpu_info", lambda **kwargs: [{"name": "Driver-visible GPU"}] * 2)
    result = runtime_info()
    assert result["cuda_device_count"] == (count if cuda else 0)
    assert result["distributed_available"] == distributed
    assert result["nccl_available"] == (distributed and nccl)
    assert result["environment_profile"] == profile
    assert result["platform"] == system
    assert result["multi_gpu_training"] is expected_multi
    assert result["gloo_available"] == (distributed and gloo)
    expected_backend = ("gloo" if system == "Windows" else "nccl") if expected_multi else None
    assert result["multi_gpu_backend"] == expected_backend
    assert result["multi_gpu_probe_required"] is (expected_multi and system == "Windows")
    assert result["training_device_policy"] == ("exclusive_devices" if expected_multi else "single_device")
    # Driver inventory must not turn CPU Torch into CUDA-capable Torch.
    assert len(result["gpus"]) == 2
    assert result["cuda_available"] is cuda
