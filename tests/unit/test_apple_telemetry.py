"""Driver percentages stay distinct from shared memory and process allocations."""

import json
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
    "error",
    [
        OSError("unavailable"),
        subprocess.TimeoutExpired("ioreg", 2),
        subprocess.CalledProcessError(1, "ioreg"),
    ],
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
    monkeypatch.setattr(hw, "_apple_gpu_sensors", lambda: sensor_reading(None, None))
    memory = SimpleNamespace(total=100 * 2**20, available=8 * 2**20)
    result = hw.gpu_info(system_memory=memory)[0]
    assert result["util_pct"] == util
    assert result["mem_used_mb"] == 92 and result["mem_total_mb"] == 100
    assert result["memory_scope"] == "unified_system"
    assert result["power_w"] is None and result["temp_c"] is None
    assert result["telemetry_source"] == ("ioreg" if util is not None else "mps")


def sensor_reading(power=0.37, temperature=47.2):
    return {
        "power_w": power,
        "temp_c": temperature,
        "temp_max_c": 50.3 if temperature is not None else None,
        "temp_sensor_count": 8 if temperature is not None else 0,
        "power_sample_seconds": 0.253 if power is not None else None,
    }


@pytest.fixture
def sensors(monkeypatch):
    monkeypatch.setattr(hw.sys, "platform", "darwin")
    monkeypatch.setattr(hw, "_apple_sensors_cache", None)
    monkeypatch.setattr(hw, "_apple_name", lambda: "Apple M4 GPU")
    now = [100.0]
    monkeypatch.setattr(hw.time, "monotonic", lambda: now[0])
    run = Mock(return_value=SimpleNamespace(stdout=json.dumps(sensor_reading())))
    monkeypatch.setattr(hw.subprocess, "run", run)
    return now, run


def test_sensor_worker_is_isolated_bounded_and_shared(sensors):
    _, run = sensors
    value = hw._apple_gpu_sensors()
    assert value == sensor_reading()
    value["power_w"] = 999  # Callers cannot mutate the shared snapshot.
    assert hw._apple_gpu_sensors()["power_w"] == 0.37
    run.assert_called_once()
    args, options = run.call_args
    assert args[0][0:2] == [hw.sys.executable, "-I"]
    assert args[0][2].endswith("/apple_sensors.py")
    assert args[0][3:] == ["--chip", "Apple M4 GPU"]
    assert options["timeout"] == 2 and options["check"] is True


@pytest.mark.parametrize(
    "payload",
    [
        sensor_reading(0),
        sensor_reading(None),
        sensor_reading(0.37, None),
    ],
)
def test_zero_and_independent_missing_sensors_survive(sensors, payload):
    _, run = sensors
    run.return_value.stdout = json.dumps(payload)
    assert hw._apple_gpu_sensors() == payload


@pytest.mark.parametrize(
    "update",
    [
        {"power_w": -1},
        {"power_w": True},
        {"power_w": "0.3"},
        {"power_w": float("inf")},
        {"power_sample_seconds": 0},
        {"power_sample_seconds": None},
    ],
)
def test_invalid_power_never_hides_valid_temperature(sensors, update):
    _, run = sensors
    run.return_value.stdout = json.dumps({**sensor_reading(), **update})
    reading = hw._apple_gpu_sensors()
    assert reading["power_w"] is None and reading["power_sample_seconds"] is None
    assert reading["temp_c"] == 47.2


@pytest.mark.parametrize(
    "update",
    [
        {"temp_c": 0},
        {"temp_c": float("nan")},
        {"temp_max_c": 40},
        {"temp_max_c": 200},
        {"temp_sensor_count": 0},
        {"temp_sensor_count": True},
    ],
)
def test_invalid_temperature_never_hides_valid_power(sensors, update):
    _, run = sensors
    run.return_value.stdout = json.dumps({**sensor_reading(), **update})
    reading = hw._apple_gpu_sensors()
    assert reading["temp_c"] is None and reading["temp_max_c"] is None and reading["temp_sensor_count"] == 0
    assert reading["power_w"] == 0.37


@pytest.mark.parametrize(
    "failure",
    [
        OSError("missing"),
        subprocess.TimeoutExpired("worker", 2),
        subprocess.CalledProcessError(-11, "worker"),
        "malformed",
        "[]",
        "null",
    ],
)
def test_sensor_native_crash_timeout_and_malformed_data_expire_previous_sample(sensors, failure):
    now, run = sensors
    assert hw._apple_gpu_sensors()["power_w"] == 0.37
    now[0] += 2.1
    if isinstance(failure, Exception):
        run.side_effect = failure
    else:
        run.return_value.stdout = failure
    assert hw._apple_gpu_sensors() == sensor_reading(None, None)
    assert hw._apple_gpu_sensors() == sensor_reading(None, None)
    assert run.call_count == 2
    now[0] += 2.1
    run.side_effect = None
    run.return_value.stdout = json.dumps(sensor_reading(0))
    assert hw._apple_gpu_sensors()["power_w"] == 0


def test_sensor_worker_not_started_on_non_mac(sensors, monkeypatch):
    _, run = sensors
    monkeypatch.setattr(hw.sys, "platform", "win32")
    assert hw._apple_gpu_sensors() == sensor_reading(None, None)
    run.assert_not_called()


def test_mps_api_preserves_sensor_semantics_and_zero_power(monkeypatch):
    from ypuddin.server.models import GpuStats

    monkeypatch.setattr(hw.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(hw.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(hw, "_apple_name", lambda: "Apple M4 GPU")
    monkeypatch.setattr(hw, "_apple_gpu_utilization", lambda: 12)
    monkeypatch.setattr(hw, "_apple_gpu_sensors", lambda: sensor_reading(0))
    row = hw.gpu_info(system_memory=SimpleNamespace(total=100 * 2**20, available=8 * 2**20))[0]
    public = GpuStats.model_validate(row).model_dump(exclude_unset=True)
    assert public["power_w"] == 0 and public["power_estimated"] is True
    assert public["power_source"] == "ioreport" and public["power_sample_seconds"] == 0.253
    assert public["temp_c"] == 47.2 and public["temp_max_c"] == 50.3 and public["temp_sensor_count"] == 8
    assert public["temperature_source"] == "smc" and public["telemetry_note"] is None
    assert public["util_pct"] == 12 and public["mem_used_mb"] == 92
