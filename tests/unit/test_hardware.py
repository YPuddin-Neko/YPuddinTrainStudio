"""Driver telemetry works without tying GPU power to another optional metric."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ypuddin.server import hardware as hw


@pytest.fixture(autouse=True)
def isolate_native_apple_sensors(monkeypatch):
    monkeypatch.setattr(hw, "_apple_gpu_sensors", lambda: {"power_w": None, "temp_c": None})


@pytest.fixture
def cuda(monkeypatch):
    monkeypatch.setattr(hw.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(hw.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        hw.torch.cuda,
        "get_device_properties",
        lambda i: SimpleNamespace(name="RTX Test", total_memory=24 * 2**30, uuid="GPU-test"),
    )
    monkeypatch.setattr(hw.torch.cuda, "mem_get_info", lambda i: (20 * 2**30, 24 * 2**30))
    monkeypatch.setattr(hw, "_nvidia_smi", lambda: [])


def test_unsupported_utilization_does_not_hide_power(cuda, monkeypatch):
    nvml = SimpleNamespace(
        nvmlInit=Mock(),
        nvmlShutdown=Mock(),
        nvmlDeviceGetHandleByUUID=Mock(return_value="handle"),
        nvmlDeviceGetUtilizationRates=Mock(side_effect=RuntimeError("unsupported")),
        nvmlDeviceGetTemperature=Mock(return_value=42),
        nvmlDeviceGetPowerUsage=Mock(return_value=185_500),
        nvmlDeviceGetEnforcedPowerLimit=Mock(return_value=450_000),
        NVML_TEMPERATURE_GPU=0,
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    gpu = hw.gpu_info()[0]
    assert gpu["util_pct"] is None
    assert gpu["power_w"] == 185.5 and gpu["power_limit_w"] == 450
    assert gpu["telemetry_source"] == "nvml"
    nvml.nvmlDeviceGetHandleByUUID.assert_called_once_with("GPU-test")
    nvml.nvmlShutdown.assert_called_once()


def test_smi_fallback_matches_uuid_not_reordered_index(cuda, monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr(
        hw,
        "_nvidia_smi",
        lambda: [
            {"index": 0, "uuid": "GPU-other", "name": "RTX Test", "power_w": 400},
            {
                "index": 1,
                "uuid": "GPU-test",
                "name": "RTX Test",
                "power_w": 0,
                "power_limit_w": 450,
                "util_pct": 0,
            },
        ],
    )
    gpu = hw.gpu_info()[0]
    assert gpu["index"] == 0 and gpu["power_w"] == 0 and gpu["util_pct"] == 0
    assert gpu["telemetry_source"] == "nvidia-smi"


def test_missing_power_has_explanation(cuda, monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    gpu = hw.gpu_info()[0]
    assert gpu["power_w"] is None
    assert gpu["telemetry_note"] == "nvidia_power_unavailable"
    assert gpu["mem_free_mb"] == 20480


def test_driver_detected_gpu_is_diagnostic_only_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(hw.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(hw.torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(hw, "_nvidia_smi", lambda: [{"index": 0, "name": "RTX Test", "power_w": 90}])
    assert hw.gpu_info() == []  # Scheduling never claims an unavailable CUDA device.
    gpu = hw.gpu_info(include_unavailable=True)[0]
    assert gpu["cuda_available"] is False and gpu["device"] is None
    assert gpu["power_w"] == 90 and gpu["telemetry_note"] == "cuda_runtime_unavailable"


def test_smi_csv_unsupported_fields_and_probe_cache(monkeypatch):
    monkeypatch.setattr(hw, "_smi_cache", (0, []))
    monkeypatch.setattr(hw.shutil, "which", lambda name: "nvidia-smi")
    run = Mock(
        return_value=SimpleNamespace(
            stdout="0, GPU-a, RTX Test, 24576, 1200, [N/A], 42, 183.25, 450\n1, GPU-b, RTX Other, 8192, 0, 0, N/A, [Not Supported], 150\n"
        )
    )
    monkeypatch.setattr(hw.subprocess, "run", run)
    rows = hw._nvidia_smi()
    assert rows[0]["power_w"] == 183.25 and rows[0]["util_pct"] is None
    assert rows[1]["power_w"] is None and rows[1]["util_pct"] == 0
    assert rows[0]["mem_free_mb"] == 23376
    assert hw._nvidia_smi() == rows
    run.assert_called_once()
    assert run.call_args.kwargs["timeout"] == 2


def test_nvml_without_uuid_uses_unique_name_not_cuda_index(cuda, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(
        hw.torch.cuda,
        "get_device_properties",
        lambda i: SimpleNamespace(name="RTX Test", total_memory=24 * 2**30),
    )
    nvml = SimpleNamespace(
        nvmlInit=Mock(),
        nvmlShutdown=Mock(),
        nvmlDeviceGetCount=lambda: 2,
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetName=lambda i: [b"Other GPU", b"RTX Test"][i],
        nvmlDeviceGetUtilizationRates=lambda h: SimpleNamespace(gpu=10),
        nvmlDeviceGetTemperature=lambda h, _: 45,
        nvmlDeviceGetPowerUsage=lambda h: [10_000, 180_000][h],
        nvmlDeviceGetEnforcedPowerLimit=lambda h: 300_000,
        NVML_TEMPERATURE_GPU=0,
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    assert hw.gpu_info()[0]["power_w"] == 180
    nvml.nvmlDeviceGetName = lambda i: b"RTX Test"
    assert hw.gpu_info()[0]["power_w"] is None  # Duplicate names without identity stay unknown.


@pytest.mark.parametrize("mps", [False, True])
def test_ram_and_mps_use_available_memory_from_one_snapshot(monkeypatch, tmp_path, mps):
    from ypuddin.server.routes_core import system_stats

    memory = Mock(return_value=SimpleNamespace(total=100 * 2**20, available=75 * 2**20, used=19 * 2**20))
    monkeypatch.setattr(hw.psutil, "virtual_memory", memory)
    monkeypatch.setattr(hw.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(hw.torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(hw, "_apple_name", lambda: "Apple Test GPU")
    monkeypatch.setattr(hw, "_nvidia_smi", lambda: [])

    stats = system_stats(tmp_path)
    assert stats["ram"] == {"used_mb": 25, "total_mb": 100}
    memory.assert_called_once_with()
    if mps:
        gpu = stats["gpus"][0]
        assert gpu["mem_used_mb"] == stats["ram"]["used_mb"]
        assert gpu["mem_total_mb"] == stats["ram"]["total_mb"]
        assert gpu["mem_free_mb"] == 75 and gpu["memory_scope"] == "unified_system"
    else:
        assert stats["gpus"] == []


def test_standalone_mps_inventory_still_samples_system_memory(monkeypatch):
    memory = Mock(return_value=SimpleNamespace(total=100 * 2**20, available=75 * 2**20))
    monkeypatch.setattr(hw.psutil, "virtual_memory", memory)
    monkeypatch.setattr(hw.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(hw.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(hw, "_apple_name", lambda: "Apple Test GPU")
    assert hw.gpu_info()[0]["mem_used_mb"] == 25
    memory.assert_called_once_with()
