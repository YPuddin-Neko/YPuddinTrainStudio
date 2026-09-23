"""Normal UI reads never allocate accelerator memory in the HTTP process."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from ypuddin.server import create_app, environment, hardware


@pytest.mark.parametrize(
    "profile,hip",
    [
        ("windows-cuda", None),
        ("linux-cuda", None),
        ("linux-dtk", "6.3"),
        ("legacy", "6.3"),
        ("macos-mps", None),
        ("windows-cpu", None),
        ("linux-cpu", None),
        ("macos-cpu", None),
    ],
)
def test_platform_ui_reads_do_not_initialize_or_allocate_gpu(monkeypatch, tmp_path, profile, hip):
    torch = hardware.torch
    monkeypatch.setenv("YPUDDIN_ENV_PROFILE", profile)
    cpu = profile.endswith("-cpu")
    apple = profile == "macos-mps"
    forbidden = Mock(side_effect=AssertionError("GPU allocation or initialization in HTTP process"))
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.version, "cuda", None if hip or apple else "12.8")
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "device_count", forbidden)
    for name in (
        "_lazy_init",
        "get_device_properties",
        "get_device_capability",
        "mem_get_info",
        "current_device",
        "set_device",
        "synchronize",
        "empty_cache",
        "memory_allocated",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(torch.backends.mps, "is_available", forbidden if cpu else lambda: apple)
    for name in (
        "current_allocated_memory",
        "driver_allocated_memory",
        "recommended_max_memory",
        "synchronize",
        "empty_cache",
    ):
        monkeypatch.setattr(torch.mps, name, forbidden)
    monkeypatch.setattr(
        hardware,
        "_cuda_inventory",
        forbidden
        if cpu or apple
        else lambda: [
            {
                "index": 0,
                "name": "Test GPU",
                "uuid": "GPU-test",
                "mem_total_mb": 24576,
                "compute_capability": [8, 9],
            },
        ],
    )
    monkeypatch.setattr(hardware, "_nvml_metrics", lambda entries: None)
    monkeypatch.setattr(hardware, "_nvidia_smi", forbidden if cpu or apple or hip else lambda: [])
    monkeypatch.setattr(hardware, "_hip_sysfs_metrics", lambda entries: [{} for _ in entries])
    monkeypatch.setattr(hardware, "_apple_name", lambda: "Apple Test GPU")
    monkeypatch.setattr(hardware, "_apple_gpu_utilization", lambda: 0)
    monkeypatch.setattr(hardware, "_apple_gpu_sensors", lambda: {"power_w": None, "temp_c": None})
    monkeypatch.setattr(environment.dtk_catalog, "system_info", lambda: {})
    monkeypatch.setattr(environment.dtk_catalog, "driver_version", lambda: None)
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    app.state.environment.probe = forbidden
    client = TestClient(app)
    try:
        for route in (
            "/health",
            "/system/stats",
            "/system/info",
            "/families",
            "/queue/devices",
            "/environment",
        ):
            response = client.get("/api" + route)
            assert response.status_code == 200, (route, response.text)
        runtime = client.get("/api/environment").json()["runtime"]
        if cpu:
            assert runtime["compute_backend"] == "cpu" and not runtime["cuda_available"]
            assert client.get("/api/queue/devices").json()["devices"] == []
        assert client.get("/api/environment").json()["sdpa"] is None
        forbidden.assert_not_called()
    finally:
        client.close()
        for manager in (
            app.state.regularization,
            app.state.dataset_pipeline,
            app.state.environment,
            app.state.model_downloads,
            app.state.torch_environments,
            app.state.ctx.versions,
        ):
            manager.close()
        app.state.ctx.db.close()


def test_device_identity_runs_once_in_a_disposable_process(monkeypatch):
    hardware._cuda_inventory.cache_clear()
    run = Mock(
        return_value=SimpleNamespace(
            stdout='[{"index":0,"uuid":"GPU-a","name":"A","mem_total_mb":24576,"compute_capability":[8,9]}]'
        )
    )
    monkeypatch.setattr(hardware.subprocess, "run", run)
    try:
        assert hardware._cuda_inventory()[0]["uuid"] == "GPU-a"
        hardware._cuda_inventory()
        run.assert_called_once()
        command = run.call_args.args[0]
        assert command[:3] == [hardware.sys.executable, "-I", "-c"]
        assert "get_device_properties" in command[3]
        assert "mem_get_info" not in command[3] and "torch.randn" not in command[3]
    finally:
        hardware._cuda_inventory.cache_clear()
