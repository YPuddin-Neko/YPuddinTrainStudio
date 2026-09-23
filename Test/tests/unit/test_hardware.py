"""Driver telemetry works without tying GPU power to another optional metric."""

import sys
import uuid
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ypuddin.server import hardware as hw


@pytest.fixture(autouse=True)
def isolate_native_apple_sensors(monkeypatch, tmp_path):
    monkeypatch.setattr(hw, "_apple_gpu_sensors", lambda: {"power_w": None, "temp_c": None})
    monkeypatch.setattr(hw, "_DRM_SYSFS", tmp_path / "sys-drm")
    monkeypatch.setattr(hw, "_DRM_DEVICES", tmp_path / "dev-dri")
    monkeypatch.setattr(hw, "_hip_sensors_cache", None)


@pytest.fixture
def cuda(monkeypatch):
    monkeypatch.setattr(hw.torch.version, "hip", None)
    monkeypatch.setattr(hw.torch.version, "cuda", "12.8")
    monkeypatch.setattr(hw.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(hw.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        hw.torch.cuda,
        "get_device_properties",
        lambda i: SimpleNamespace(name="RTX Test", total_memory=24 * 2**30, uuid="GPU-test"),
    )
    monkeypatch.setattr(hw.torch.cuda, "mem_get_info", lambda i: (20 * 2**30, 24 * 2**30))
    monkeypatch.setattr(hw, "_nvidia_smi", lambda: [])

    def isolated_inventory():
        rows = []
        for i in range(hw.torch.cuda.device_count()):
            prop = hw.torch.cuda.get_device_properties(i)
            rows.append(
                {
                    "index": i,
                    "name": prop.name,
                    "mem_total_mb": round(prop.total_memory / 2**20),
                    "uuid": str(prop.uuid) if getattr(prop, "uuid", None) else None,
                    "pci_bus_id": getattr(prop, "pci_bus_id", None),
                    "compute_capability": [8, 9],
                }
            )
        return rows

    monkeypatch.setattr(hw, "_cuda_inventory", isolated_inventory)


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
    assert gpu["mem_free_mb"] is None  # No driver reading means available VRAM is unknown.


def test_cuda_monitoring_never_initializes_a_server_context_and_uses_physical_vram(cuda, monkeypatch):
    monkeypatch.setattr(
        hw,
        "_cuda_inventory",
        lambda: [
            {
                "index": 0,
                "name": "RTX Test",
                "mem_total_mb": 24576,
                "uuid": "GPU-test",
                "compute_capability": [8, 9],
            }
        ],
    )
    for name in ("get_device_properties", "get_device_capability", "mem_get_info"):
        monkeypatch.setattr(hw.torch.cuda, name, Mock(side_effect=AssertionError("CUDA context in server")))
    monkeypatch.setitem(
        sys.modules,
        "pynvml",
        SimpleNamespace(
            nvmlInit=lambda: None,
            nvmlShutdown=lambda: None,
            nvmlDeviceGetHandleByUUID=lambda uuid: "handle",
            nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(
                total=49140 * 2**20,
                used=402 * 2**20,
                free=48738 * 2**20,
            ),
        ),
    )
    first = hw.gpu_info()[0]
    second = hw.gpu_info()[0]
    assert first["mem_total_mb"] == second["mem_total_mb"] == 49140
    assert first["mem_used_mb"] == 402 and first["mem_free_mb"] == 48738
    from ypuddin.server.environment import runtime_info

    assert runtime_info()["gpu_capability"] == [8, 9]


def test_smi_physical_memory_replaces_torch_allocatable_capacity(cuda, monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr(
        hw,
        "_nvidia_smi",
        lambda: [
            {
                "index": 0,
                "uuid": "GPU-test",
                "name": "RTX Test",
                "mem_total_mb": 49140,
                "mem_used_mb": 402,
                "mem_free_mb": 48738,
            }
        ],
    )
    gpu = hw.gpu_info()[0]
    assert gpu["mem_total_mb"] == 49140 and gpu["mem_free_mb"] == 48738


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
        hw, "_cuda_inventory", lambda: [{"index": 0, "name": "RTX Test", "mem_total_mb": 24576}]
    )
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


@pytest.mark.parametrize("profile,kind", [("linux-dtk", "dtk"), ("legacy", "rocm")])
def test_hip_inventory_never_reads_nvidia_telemetry_or_invents_metrics(cuda, monkeypatch, profile, kind):
    monkeypatch.setenv("YPUDDIN_ENV_PROFILE", profile)
    monkeypatch.setattr(hw.torch.version, "hip", "6.2.0")
    monkeypatch.setattr(hw.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        hw.torch.cuda,
        "get_device_properties",
        lambda i: SimpleNamespace(name=f"BW GPU {i}", total_memory=64 * 2**30, uuid=f"DCU-{i}"),
    )
    monkeypatch.setattr(hw.torch.cuda, "mem_get_info", lambda i: ((60 - i) * 2**30, 64 * 2**30))
    monkeypatch.setattr(hw, "_nvml_metrics", lambda *a: pytest.fail("HIP must never call NVML"))
    monkeypatch.setattr(hw, "_nvidia_smi", lambda: pytest.fail("HIP must never call nvidia-smi"))
    gpus = hw.gpu_info(include_unavailable=True)
    assert len(gpus) == 2
    for i, gpu in enumerate(gpus):
        assert gpu["index"] == i and gpu["name"] == f"BW GPU {i}"
        assert gpu["kind"] == kind and gpu["device"] == f"cuda:{i}"
        assert gpu["hip_runtime"] == "6.2.0" and gpu["telemetry_source"] == "torch-hip"
        assert gpu["mem_total_mb"] == 65536 and gpu["mem_used_mb"] is None
        assert gpu["mem_free_mb"] is None
        assert all(gpu[key] is None for key in ("util_pct", "power_w", "power_limit_w", "temp_c"))
        assert gpu["telemetry_note"] == "hip_driver_metrics_unavailable"


def test_hip_memory_probe_failure_stays_unknown_without_nvidia_fallback(cuda, monkeypatch):
    monkeypatch.setattr(hw.torch.version, "hip", "6.2")
    monkeypatch.setattr(hw.torch.cuda, "mem_get_info", Mock(side_effect=RuntimeError("unsupported")))
    monkeypatch.setattr(hw, "_nvidia_smi", lambda: pytest.fail("HIP must never call nvidia-smi"))
    gpu = hw.gpu_info()[0]
    assert gpu["mem_total_mb"] == 24576
    assert gpu["mem_used_mb"] is None and gpu.get("mem_free_mb") is None


def test_unavailable_hip_runtime_does_not_fall_back_to_unrelated_nvidia_inventory(monkeypatch):
    monkeypatch.setattr(hw.torch.version, "hip", "6.2")
    monkeypatch.setattr(hw.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(hw.torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(hw, "_cuda_inventory", lambda: [])
    monkeypatch.setattr(hw, "_nvidia_smi", lambda: pytest.fail("HIP must never call nvidia-smi"))
    assert hw.gpu_info(include_unavailable=True) == []


def test_hip_inventory_is_serialized_by_health_stats_and_system_info(cuda, monkeypatch, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from ypuddin.server.routes_core import router

    monkeypatch.setenv("YPUDDIN_ENV_PROFILE", "linux-dtk")
    monkeypatch.setattr(hw.torch.version, "hip", "6.2.0")
    monkeypatch.setattr(hw.torch.version, "cuda", None)
    monkeypatch.setattr(hw, "_nvml_metrics", lambda *a: pytest.fail("HIP must never call NVML"))
    app = FastAPI()
    app.state.ctx = SimpleNamespace(data_root=tmp_path)
    app.include_router(router, prefix="/api")
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["hip"] == "6.2.0" and health.json()["hip_available"] is True
        assert health.json()["gpus"][0]["kind"] == "dtk"
        stats = client.get("/api/system/stats")
        assert stats.status_code == 200
        gpu = stats.json()["gpus"][0]
        assert gpu["kind"] == "dtk" and gpu["hip_runtime"] == "6.2.0"
        assert gpu["power_w"] is None and gpu["util_pct"] is None
        info = client.get("/api/system/info")
        assert info.status_code == 200
        assert info.json()["hip"] == "6.2.0" and info.json()["hip_available"] is True


@pytest.fixture
def hip_sysfs(cuda, monkeypatch, tmp_path):
    monkeypatch.setattr(hw.sys, "platform", "linux")
    monkeypatch.setattr(hw.torch.version, "hip", "6.3.26093")
    monkeypatch.setenv("YPUDDIN_ENV_PROFILE", "linux-dtk")
    denied = set()
    monkeypatch.setattr(hw, "_drm_accessible", lambda node: node.name not in denied)
    hw._DRM_DEVICES.mkdir()

    def card(bus, identity, node, *, allocated=True, vendor="0x1d94", driver="hycu"):
        device = tmp_path / "pci" / bus
        sensor = device / "hwmon" / "hwmon8"
        sensor.mkdir(parents=True)
        values = {
            "vendor": vendor,
            "unique_id": identity,
            "gpu_busy_percent": "37",
            "mem_info_vram_total": str(65520 * 2**20),
            "mem_info_vram_used": str(2048 * 2**20),
            "hwmon/hwmon8/name": driver,
            "hwmon/hwmon8/power1_average": "83500000",
            "hwmon/hwmon8/power1_cap": "1000000000",
            "hwmon/hwmon8/temp1_label": "edge",
            "hwmon/hwmon8/temp1_input": "53000",
            "hwmon/hwmon8/power2_average": "21000000",
        }
        for filename, value in values.items():
            (device / filename).write_text(value)
        node_dir = hw._DRM_SYSFS / node
        node_dir.mkdir(parents=True)
        (node_dir / "device").symlink_to(device, target_is_directory=True)
        if allocated:
            (hw._DRM_DEVICES / node).touch()
        return device

    return card, denied


def _hip_props(identity, **kwargs):
    return SimpleNamespace(
        name="BW", total_memory=65520 * 2**20, uuid=uuid.UUID(bytes=identity.encode()), **kwargs
    )


@pytest.mark.parametrize(
    "profile,vendor,driver", [("linux-dtk", "0x1d94", "hycu"), ("legacy", "0x1002", "amdgpu")]
)
def test_hip_driver_telemetry_matches_uuid_after_masks_and_shared_node_reordering(
    hip_sysfs, monkeypatch, profile, vendor, driver
):
    card, _ = hip_sysfs
    first = card("0000:36:00.0", "0014ba8a38d43021", "renderD129", vendor=vendor, driver=driver)
    second = card("0000:55:00.0", "0014ba57aa024101", "renderD130", vendor=vendor, driver=driver)
    card("0000:16:00.0", "0014ba8a38d43022", "renderD128", allocated=False)
    (second / "gpu_busy_percent").write_text("0")  # A real idle reading is not missing data.
    (second / "hwmon/hwmon8/power1_average").write_text("72000000")
    (second / "mem_info_vram_used").write_text(str(4096 * 2**20))
    monkeypatch.setenv("YPUDDIN_ENV_PROFILE", profile)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1,0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "3,2")
    monkeypatch.setattr(hw.torch.cuda, "device_count", lambda: 2)
    props = [_hip_props("0014ba57aa024101"), _hip_props("0014ba8a38d43021")]
    monkeypatch.setattr(hw.torch.cuda, "get_device_properties", lambda i: props[i])
    monkeypatch.setattr(hw.torch.cuda, "mem_get_info", lambda i: (60000 * 2**20, 65520 * 2**20))
    gpus = hw.gpu_info()
    assert [g["index"] for g in gpus] == [0, 1]
    assert [g["name"] for g in gpus] == ["BW", "BW"]
    assert [g["util_pct"] for g in gpus] == [0, 37]
    assert [g["power_w"] for g in gpus] == [72, 83.5]  # Never sum component power channels.
    assert [g["mem_used_mb"] for g in gpus] == [4096, 2048]
    assert [g["mem_free_mb"] for g in gpus] == [61424, 63472]
    assert all(g["mem_total_mb"] == 65520 and g["temp_c"] == 53 for g in gpus)
    assert all(g["telemetry_source"] == "torch-hip+sysfs" and g["telemetry_note"] is None for g in gpus)
    assert all(g["power_source"] == "hwmon" and g["power_estimated"] is False for g in gpus)
    assert all(g["temperature_source"] == "hwmon-edge" for g in gpus)
    # Cache identity changes immediately with Torch's visible order, not after expiry.
    props.reverse()
    assert [g["power_w"] for g in hw.gpu_info()] == [83.5, 72]
    assert first.exists()


@pytest.mark.parametrize(
    "failure", ["unknown_uuid", "duplicate_uuid", "unallocated", "denied", "other_vendor"]
)
def test_hip_sensors_never_attach_an_unproven_or_unallocated_device(hip_sysfs, monkeypatch, failure):
    card, denied = hip_sysfs
    identity = "0014ba8a38d43021"
    card(
        "0000:36:00.0",
        identity,
        "renderD129",
        allocated=failure != "unallocated",
        vendor="0x10de" if failure == "other_vendor" else "0x1d94",
    )
    if failure == "duplicate_uuid":
        card("0000:55:00.0", identity, "renderD130")
    if failure == "denied":
        denied.add("renderD129")
    monkeypatch.setattr(
        hw.torch.cuda,
        "get_device_properties",
        lambda _: _hip_props("0014ba8a38d43022" if failure == "unknown_uuid" else identity),
    )
    gpu = hw.gpu_info()[0]
    assert gpu["telemetry_source"] == "torch-hip"
    assert gpu["power_w"] is None and gpu["util_pct"] is None and gpu["temp_c"] is None
    assert gpu["mem_used_mb"] is None  # Unmatched driver memory stays unknown.


def test_hip_explicit_pci_identity_can_match_when_uuid_is_unavailable(hip_sysfs, monkeypatch):
    card, _ = hip_sysfs
    card("0000:55:00.0", "0014ba57aa024101", "renderD130")
    monkeypatch.setattr(
        hw.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            name="BW", total_memory=65520 * 2**20, uuid=None, pci_bus_id="0000:55:00.0"
        ),
    )
    assert hw.gpu_info()[0]["power_w"] == 83.5


def test_hip_sysfs_cache_is_bounded_and_failed_readings_replace_previous_values(hip_sysfs, monkeypatch):
    card, _ = hip_sysfs
    device = card("0000:36:00.0", "0014ba8a38d43021", "renderD129")
    clock = [100.0]
    monkeypatch.setattr(hw.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(hw.torch.cuda, "get_device_properties", lambda _: _hip_props("0014ba8a38d43021"))
    read = Mock(wraps=hw._hip_sensor_reading)
    monkeypatch.setattr(hw, "_hip_sensor_reading", read)
    assert hw.gpu_info()[0]["power_w"] == 83.5
    (device / "hwmon/hwmon8/power1_average").write_text("NaN")
    (device / "gpu_busy_percent").write_text("101")
    (device / "hwmon/hwmon8/temp1_input").write_text("0")
    (device / "mem_info_vram_used").write_text(str(70000 * 2**20))
    assert hw.gpu_info()[0]["power_w"] == 83.5
    assert read.call_count == 1
    clock[0] += 2.01
    gpu = hw.gpu_info()[0]
    assert read.call_count == 2
    assert gpu["power_w"] is None and gpu["util_pct"] is None and gpu["temp_c"] is None
    assert gpu["mem_used_mb"] is None
    assert gpu["power_limit_w"] == 1000  # A missing field does not hide independent metrics.


def test_drm_access_check_rejects_device_cgroup_denial(monkeypatch):
    node = Mock()
    node.stat.return_value = SimpleNamespace(st_mode=hw.stat.S_IFCHR)
    open_device = Mock(side_effect=PermissionError("device cgroup denies access"))
    monkeypatch.setattr(hw.os, "open", open_device)
    assert not hw._drm_accessible(node)
    open_device.assert_called_once()


def test_hip_driver_metrics_reach_system_stats_and_old_hwmon_without_label(hip_sysfs, monkeypatch, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from ypuddin.server.routes_core import router

    card, _ = hip_sysfs
    device = card("0000:36:00.0", "0014ba8a38d43021", "renderD129", vendor="0x1002", driver="amdgpu")
    (device / "hwmon/hwmon8/temp1_label").unlink()
    monkeypatch.setattr(hw.torch.cuda, "get_device_properties", lambda _: _hip_props("0014ba8a38d43021"))
    app = FastAPI()
    app.state.ctx = SimpleNamespace(data_root=tmp_path)
    app.include_router(router, prefix="/api")
    with TestClient(app) as client:
        response = client.get("/api/system/stats")
        assert response.status_code == 200
        gpu = response.json()["gpus"][0]
        assert gpu["power_w"] == 83.5 and gpu["power_source"] == "hwmon"
        assert gpu["power_estimated"] is False
        assert gpu["temp_c"] == 53 and gpu["temperature_source"] == "hwmon-edge"
        assert gpu["util_pct"] == 37 and gpu["mem_used_mb"] == 2048
        assert gpu["telemetry_note"] is None
