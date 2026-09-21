"""Windows branch simulation on the host OS; no Windows binaries or GPU workers run."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from fastapi.testclient import TestClient

from ypuddin.config import load_config
from ypuddin.server import create_app
from ypuddin.server import supervisor as module


@pytest.fixture
def windows_queue(tmp_path, monkeypatch):
    inventory = [
        {"device": f"cuda:{index}", "name": f"GPU {index}", "mem_free_mb": 24000, "mem_total_mb": 24576}
        for index in range(2)
    ]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="7,3")
    environment.pop("HIP_VISIBLE_DEVICES", None)
    # Replacing the module reference avoids changing global os.name: pathlib and
    # pytest must continue using the real host filesystem implementation.
    monkeypatch.setattr(module, "os", SimpleNamespace(**(vars(os) | {"name": "nt", "environ": environment})))
    monkeypatch.setattr(module, "sys", SimpleNamespace(**(vars(sys) | {"platform": "win32"})))
    monkeypatch.setattr(module, "current_profile", lambda: "windows-cuda")
    monkeypatch.setattr(module, "gpu_info", lambda: inventory)
    monkeypatch.setattr("ypuddin.server.routes_work.gpu_info", lambda: inventory)
    monkeypatch.setattr(torch.version, "hip", None)
    launched = []

    def popen(command, **kwargs):
        process = Mock(pid=610000 + len(launched), returncode=None)
        process.poll.return_value = None
        launched.append(SimpleNamespace(command=command, kwargs=kwargs, process=process))
        return process

    kill = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(
        module,
        "subprocess",
        SimpleNamespace(
            **(vars(subprocess) | {"CREATE_NEW_PROCESS_GROUP": 512, "Popen": popen, "run": kill})
        ),
    )
    root = tmp_path / "训练 空间 & 本地"
    app = create_app(root, frontend_dist=root / "no-ui")
    client = TestClient(app)
    context = app.state.ctx
    context.supervisor.python = r"C:\训练 环境\Scripts\python.exe"
    yield SimpleNamespace(
        client=client,
        context=context,
        supervisor=context.supervisor,
        environment=environment,
        launched=launched,
        kill=kill,
    )
    context.supervisor._procs.clear()
    context.supervisor._devices.clear()
    client.close()
    context.db.close()


def enqueue(queue, image_dataset, device):
    response = queue.client.post(
        "/api/jobs",
        json={
            "name": "Windows 路径与选卡模拟",
            "type": "train",
            "gpu_devices": [device],
            "config": {
                "model": {"family": "toy", "dtype": "fp32"},
                "dataset": {
                    "sources": [{"path": str(image_dataset)}],
                    "resolutions": [64],
                    "bucket_step": 16,
                    "num_workers": 0,
                },
                "loop": {"gpu_count": 1, "epochs": 1, "mixed_precision": "no"},
            },
        },
    )
    assert response.status_code == 201, response.text
    return queue.context.db.fetchone("SELECT * FROM jobs WHERE id=?", (response.json()["id"],))


def test_windows_launch_preserves_unicode_space_arguments_and_process_group(windows_queue, image_dataset):
    queue = windows_queue
    row = enqueue(queue, image_dataset, "cuda:1")
    queue.supervisor._tick()
    launched = queue.launched[0]
    config_path = Path(row["run_dir"]) / "job-config.toml"
    assert launched.command == [
        r"C:\训练 环境\Scripts\python.exe",
        "-m",
        "ypuddin.cli",
        "train",
        str(config_path),
        "--device",
        "cuda:0",
    ]
    assert "训练 空间 & 本地" in str(config_path)
    assert launched.kwargs["cwd"] == row["run_dir"]
    assert launched.kwargs["creationflags"] == 512
    assert "start_new_session" not in launched.kwargs
    assert not launched.kwargs.get("shell", False)
    assert launched.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert queue.environment["CUDA_VISIBLE_DEVICES"] == "7,3"
    assert load_config(config_path).dataset.sources[0].path == str(image_dataset)


@pytest.mark.parametrize("use_libuv", [None, "1"])
def test_windows_ddp_launch_preserves_masks_and_requires_per_job_probe(
    windows_queue, image_dataset, monkeypatch, use_libuv
):
    queue = windows_queue
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_gloo_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_nccl_available", lambda: False)
    if use_libuv is None:
        queue.environment.pop("USE_LIBUV", None)
    else:
        queue.environment["USE_LIBUV"] = use_libuv
    row = enqueue(queue, image_dataset, "cuda:0")
    config = json.loads(row["config_json"])
    config["loop"]["gpu_count"] = 2
    response = queue.client.post(
        "/api/jobs", json={"name": "DDP 两卡", "config": config, "gpu_devices": ["cuda:1", "cuda:0"]}
    )
    assert response.status_code == 201, response.text
    dual = queue.context.db.fetchone("SELECT * FROM jobs WHERE id=?", (response.json()["id"],))
    queue.supervisor._launch(dual, device=("cuda:1", "cuda:0"))
    launched = queue.launched[-1]
    assert launched.command[1:6] == [
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
    ]
    assert launched.command[-2:] == ["--device", "cuda"]
    assert launched.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "3,7"
    assert launched.kwargs["env"]["YPUDDIN_DISTRIBUTED_BACKEND"] == "gloo"
    assert launched.kwargs["env"]["USE_LIBUV"] == (use_libuv or "0")
    assert queue.environment.get("USE_LIBUV") == use_libuv
    assert launched.kwargs["creationflags"] == 512
    progress = json.loads(
        queue.context.db.fetchone("SELECT progress_json FROM jobs WHERE id=?", (dual["id"],))["progress_json"]
    )
    assert progress["phase"] == "checking_communication"


def test_windows_ddp_build_admission_never_implies_fsdp_support(windows_queue, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_gloo_available", lambda: True)
    inventory = module.gpu_info()
    assert module.training_device_error(2, inventory) is None  # workers must still pass the real CUDA probe
    assert "FSDP" in module.training_device_error(2, inventory, strategy="fsdp")
    monkeypatch.setattr(torch.distributed, "is_gloo_available", lambda: False)
    assert "Gloo" in module.training_device_error(2, inventory)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert "CUDA PyTorch" in module.training_device_error(2, inventory)


@pytest.mark.parametrize("inherited", [None, "7,3", "GPU-first,GPU-second"])
def test_windows_two_jobs_keep_masks_and_wait_for_actual_exit(windows_queue, image_dataset, inherited):
    queue = windows_queue
    if inherited is None:
        queue.environment.pop("CUDA_VISIBLE_DEVICES")
    else:
        queue.environment["CUDA_VISIBLE_DEVICES"] = inherited
    first = enqueue(queue, image_dataset, "cuda:0")
    waiting = enqueue(queue, image_dataset, "cuda:0")
    other = enqueue(queue, image_dataset, "cuda:1")
    queue.supervisor._tick()
    assert set(queue.supervisor._procs) == {first["id"], other["id"]}
    masks = inherited.split(",") if inherited else ["0", "1"]
    assert [item.kwargs["env"]["CUDA_VISIBLE_DEVICES"] for item in queue.launched] == masks
    assert all(item.command[-2:] == ["--device", "cuda:0"] for item in queue.launched)
    assert all(item.kwargs["creationflags"] == 512 for item in queue.launched)
    assert (
        queue.context.db.fetchone("SELECT status FROM jobs WHERE id=?", (waiting["id"],))["status"]
        == "queued"
    )
    queue.supervisor._set_terminal(first["id"], "run.finished")
    queue.supervisor._tick()
    assert len(queue.launched) == 2  # A completion event does not prove its process exited.
    queue.launched[0].process.poll.return_value = 0
    queue.launched[0].process.returncode = 0
    queue.supervisor._tick()
    assert set(queue.supervisor._procs) == {waiting["id"], other["id"]}
    assert queue.supervisor._devices[other["id"]] == "cuda:1"
    assert queue.launched[1].process.poll() is None
    assert queue.launched[-1].kwargs["env"]["CUDA_VISIBLE_DEVICES"] == masks[0]


@pytest.mark.parametrize(
    "command,control,status",
    [("save", "save", "running"), ("pause", "pause", "pausing"), ("cancel", "stop", "cancelling")],
)
def test_windows_file_controls_do_not_interrupt_another_job(
    windows_queue, image_dataset, command, control, status
):
    queue = windows_queue
    first = enqueue(queue, image_dataset, "cuda:0")
    other = enqueue(queue, image_dataset, "cuda:1")
    queue.supervisor._tick()
    result = queue.supervisor.request(first["id"], command)
    assert result["status"] == status
    assert (Path(first["run_dir"]) / "control" / control).is_file()
    assert not (Path(other["run_dir"]) / "control").exists()
    assert (
        queue.context.db.fetchone("SELECT status FROM jobs WHERE id=?", (other["id"],))["status"] == "running"
    )
    assert set(queue.supervisor._procs) == {first["id"], other["id"]}
    queue.kill.assert_not_called()
    for item in queue.launched:
        item.process.send_signal.assert_not_called()
        item.process.terminate.assert_not_called()


def test_windows_force_cancel_targets_only_owned_pid_and_retains_reservation(windows_queue, image_dataset):
    queue = windows_queue
    first = enqueue(queue, image_dataset, "cuda:0")
    other = enqueue(queue, image_dataset, "cuda:1")
    queue.supervisor._tick()
    process = queue.supervisor._procs[first["id"]]
    queue.supervisor._force_kill(first["id"], process)
    queue.kill.assert_called_once_with(
        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert queue.supervisor._devices == {first["id"]: "cuda:0", other["id"]: "cuda:1"}
    assert queue.supervisor._procs[other["id"]].poll() is None


@pytest.mark.parametrize("stale", ["replaced", "exited", "removed"])
def test_windows_stale_force_cancel_cannot_kill_replacement(windows_queue, image_dataset, stale):
    queue = windows_queue
    row = enqueue(queue, image_dataset, "cuda:0")
    queue.supervisor._tick()
    original = queue.supervisor._procs[row["id"]]
    if stale == "replaced":
        replacement = Mock(pid=original.pid, poll=Mock(return_value=None))
        queue.supervisor._procs[row["id"]] = replacement
    elif stale == "exited":
        original.poll.return_value = 0
    else:
        queue.supervisor._procs.pop(row["id"])
    queue.supervisor._force_kill(row["id"], original)
    queue.kill.assert_not_called()


def test_windows_single_job_multigpu_without_cuda_remains_rejected(windows_queue, image_dataset, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    queue = windows_queue
    row = enqueue(queue, image_dataset, "cuda:0")
    config = json.loads(row["config_json"])
    config["loop"]["gpu_count"] = 2
    response = queue.client.post(
        "/api/jobs",
        json={"name": "缺少 CUDA 时拒绝两卡", "config": config, "gpu_devices": ["cuda:0", "cuda:1"]},
    )
    assert response.status_code == 400
    assert "CUDA PyTorch" in str(response.json())
    assert not queue.launched
