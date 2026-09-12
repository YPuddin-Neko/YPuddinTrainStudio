"""Driver percentages stay distinct from shared memory and process allocations."""

import plistlib
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ypuddin.server import hardware as hw


def accelerator(value=46, **overrides):
    return {
        "IOClass": "AGXAcceleratorG16G",
        "IORegistryEntryID": 123,
        "model": "Apple M4",
        "PerformanceStatistics": {
            "Device Utilization %": value,
            "Renderer Utilization %": 91,
            "Tiler Utilization %": 80,
        },
        **overrides,
    }


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.setattr(hw.sys, "platform", "darwin")
    monkeypatch.setattr(hw, "_apple_gpu_cache", None)
    now = [100.0]
    monkeypatch.setattr(hw.time, "monotonic", lambda: now[0])
    run = Mock(return_value=SimpleNamespace(stdout=plistlib.dumps([accelerator()])))
    monkeypatch.setattr(hw.subprocess, "run", run)
    return now, run


@pytest.mark.parametrize("value", [0, 0.5, 46, 100])
def test_driver_percentage_is_not_rescaled_or_renderer_sum(probe, value):
    _, run = probe
    run.return_value.stdout = plistlib.dumps([accelerator(value)])
    assert hw._apple_gpu_utilization() == value
    args, kwargs = run.call_args
    assert args[0] == ["/usr/sbin/ioreg", "-r", "-c", "IOAccelerator", "-d", "1", "-a"]
    assert kwargs["timeout"] == 2 and kwargs["check"] is True


@pytest.mark.parametrize("value", [-1, 101, float("nan"), float("inf"), True, "46"])
def test_invalid_driver_values_are_unknown(probe, value):
    _, run = probe
    run.return_value.stdout = plistlib.dumps([accelerator(value)])
    assert hw._apple_gpu_utilization() is None


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        ["not a device"],
        [accelerator(IOClass="IntelAccelerator")],
        [accelerator(PerformanceStatistics={"Renderer Utilization %": 50})],
        [accelerator(PerformanceStatistics="unavailable")],
        [accelerator(), accelerator(IORegistryEntryID=456)],
    ],
)
def test_missing_or_ambiguous_apple_device_is_not_guessed(probe, payload):
    _, run = probe
    run.return_value.stdout = plistlib.dumps(payload)
    assert hw._apple_gpu_utilization() is None


@pytest.mark.parametrize("payload", [b"", b"not a plist", b'<?xml version="1.0"?><plist><dict>'])
def test_malformed_registry_output_does_not_break_stats(probe, payload):
    _, run = probe
    run.return_value.stdout = payload
    assert hw._apple_gpu_utilization() is None


@pytest.mark.parametrize(
    "error", [OSError("unavailable"), subprocess.TimeoutExpired("ioreg", 2), subprocess.CalledProcessError(1, "ioreg")]
)
def test_probe_failure_expires_old_value_and_is_cached(probe, error):
    now, run = probe
    assert hw._apple_gpu_utilization() == 46
    assert hw._apple_gpu_utilization() == 46
    run.assert_called_once()
    now[0] += 2.1
    run.side_effect = error
    assert hw._apple_gpu_utilization() is None
    assert hw._apple_gpu_utilization() is None
    assert run.call_count == 2
    now[0] += 2.1
    run.side_effect = None
    run.return_value.stdout = plistlib.dumps([accelerator(0)])
    assert hw._apple_gpu_utilization() == 0
    assert run.call_count == 3


def test_non_mac_never_launches_ioreg(probe, monkeypatch):
    _, run = probe
    monkeypatch.setattr(hw.sys, "platform", "linux")
    assert hw._apple_gpu_utilization() is None
    run.assert_not_called()


@pytest.mark.parametrize("util", [0, 46, None])
def test_mps_stats_keep_gpu_load_separate_from_unified_memory(monkeypatch, util):
    monkeypatch.setattr(hw.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(hw.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(hw, "_apple_name", lambda: "Apple M4 GPU")
    monkeypatch.setattr(hw, "_apple_gpu_utilization", lambda: util)
    memory = SimpleNamespace(total=100 * 2**20, available=8 * 2**20)
    result = hw.gpu_info(system_memory=memory)[0]
    assert result["util_pct"] == util
    assert result["mem_used_mb"] == 92 and result["mem_total_mb"] == 100
    assert result["memory_scope"] == "unified_system"
    assert result["power_w"] is None and result["temp_c"] is None
    assert result["telemetry_source"] == ("ioreg" if util is not None else "mps")
