"""Multi-GPU admission, masked device ownership and worker-tree cleanup."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import psutil
import pytest
import torch
from fastapi.testclient import TestClient

from ypuddin.config import load_config
from ypuddin.server import create_app
from ypuddin.server import supervisor as module
from ypuddin.server.supervisor import worker_device_environment


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr("ypuddin.server.routes_work.gpu_info", lambda: [])
    monkeypatch.setattr(module, "current_profile", lambda: "linux-cuda")
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    client = TestClient(app)
    yield client, app.state.ctx
    client.close()
    app.state.ctx.db.close()


def inventory(count=3):
    return [
        {"device": f"cuda:{i}", "mem_free_mb": 8000 - i * 1000, "mem_total_mb": 8000} for i in range(count)
    ]


def job(service, image_dataset):
    client, ctx = service
    cfg = {
        "model": {"family": "toy", "dtype": "fp32"},
        "dataset": {
            "sources": [{"path": str(image_dataset)}],
            "resolutions": [64],
            "bucket_step": 16,
            "num_workers": 0,
        },
        "loop": {"epochs": 1, "mixed_precision": "no"},
    }
    response = client.post("/api/jobs", json={"name": "ownership", "type": "train", "config": cfg})
    assert response.status_code == 201, response.text
    row = ctx.db.fetchone("SELECT * FROM jobs WHERE id=?", (response.json()["id"],))
    cfg = json.loads(row["config_json"])
    cfg["loop"]["gpu_count"] = 2
    row["config_json"] = json.dumps(cfg)
    ctx.db.update("jobs", row["id"], {"config_json": row["config_json"]})
    return row


def capable(monkeypatch):
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_nccl_available", lambda: True)


def test_multigpu_reservation_is_atomic_and_never_overlaps(service, image_dataset, monkeypatch):
    row = job(service, image_dataset)
    _, ctx = service
    capable(monkeypatch)
    monkeypatch.setattr(module, "gpu_info", inventory)
    ctx.supervisor._devices["other"] = ("cuda:0", "cuda:2")
    assert ctx.supervisor._choose_device(row) is None
    assert ctx.supervisor._devices["other"] == ("cuda:0", "cuda:2")
    ctx.supervisor._devices["other"] = "cuda:2"
    assert ctx.supervisor._choose_device(row) == ("cuda:0", "cuda:1")
    row["progress_json"] = json.dumps({"estimated_peak_mb": 7500})
    assert ctx.supervisor._choose_device(row) is None  # both cards must fit
    assert ctx.supervisor._choose_device(row, check_memory=False) == ("cuda:0", "cuda:1")


@pytest.mark.parametrize("platform,count,nccl", [("linux", 1, True), ("linux", 2, False), ("win32", 2, True)])
def test_multigpu_preflight_rejects_missing_hardware_or_communication(
    service, image_dataset, monkeypatch, platform, count, nccl
):
    row = job(service, image_dataset)
    capable(monkeypatch)
    monkeypatch.setattr(module.sys, "platform", platform)
    monkeypatch.setattr(torch.distributed, "is_nccl_available", lambda: nccl)
    monkeypatch.setattr("ypuddin.server.routes_work.gpu_info", lambda: inventory(count))
    response = service[0].post(
        "/api/jobs", json={"name": "two cards", "config": json.loads(row["config_json"])}
    )
    assert response.status_code == 400
    assert response.json()["error"]["details"]["errors"][0]["loc"] == "loop.gpu_count"


def test_queued_multigpu_cannot_fall_back_to_cpu(service, image_dataset, monkeypatch):
    row = job(service, image_dataset)
    capable(monkeypatch)
    monkeypatch.setattr(module, "gpu_info", lambda: [])
    assert service[1].supervisor._choose_device(row) is None
    assert service[1].db.fetchone("SELECT status FROM jobs WHERE id=?", (row["id"],))["status"] == "failed"


def test_torchrun_launch_reserves_all_cards_and_uses_relative_worker_devices(
    service, image_dataset, monkeypatch
):
    row = job(service, image_dataset)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,2,7")
    proc = Mock(pid=987654)
    popen = Mock(return_value=proc)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    service[1].supervisor._launch(row, device=("cuda:2", "cuda:0"))
    args, kwargs = popen.call_args
    assert args[0][:6] == [
        service[1].supervisor.python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
    ]
    assert args[0][-2:] == ["--device", "cuda"]
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "7,5"
    assert kwargs["start_new_session"] is True
    assert load_config(Path(row["run_dir"]) / "job-config.toml").loop.gpu_count == 2
    assert service[1].supervisor._devices[row["id"]] == ("cuda:2", "cuda:0")
    progress = json.loads(
        service[1].db.fetchone("SELECT progress_json FROM jobs WHERE id=?", (row["id"],))["progress_json"]
    )
    assert progress["devices"] == ["cuda:2", "cuda:0"]
    assert progress["device"] == "cuda:2, cuda:0"
    service[1].supervisor._procs.clear()


def test_hip_mask_preserves_rocr_scope_and_translates_inherited_hip(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "6.3")
    env = {
        "ROCR_VISIBLE_DEVICES": "GPU-first,GPU-second,GPU-third",
        "HIP_VISIBLE_DEVICES": "2,0",
        "CUDA_VISIBLE_DEVICES": "2,0",
    }
    result = worker_device_environment(("cuda:1", "cuda:0"), env)
    assert result["ROCR_VISIBLE_DEVICES"] == env["ROCR_VISIBLE_DEVICES"]
    assert result["HIP_VISIBLE_DEVICES"] == result["CUDA_VISIBLE_DEVICES"] == "0,2"
    assert env["HIP_VISIBLE_DEVICES"] == "2,0"
    with pytest.raises(ValueError, match="visibility changed"):
        worker_device_environment(("cuda:2",), env)


def test_cache_of_multigpu_project_stays_single_card(service, image_dataset, monkeypatch):
    row = job(service, image_dataset)
    config = json.loads(row["config_json"])
    # A cache request must not require the project's future training GPU count.
    response = service[0].post("/api/jobs", json={"name": "cache", "type": "cache", "config": config})
    assert response.status_code == 201, response.text
    stored = service[1].db.fetchone("SELECT * FROM jobs WHERE id=?", (response.json()["id"],))
    assert json.loads(stored["config_json"])["loop"]["gpu_count"] == 1
    popen = Mock(return_value=Mock(pid=987655))
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    service[1].supervisor._launch(stored, device="cuda:1")
    assert "torch.distributed.run" not in popen.call_args.args[0]
    assert popen.call_args.args[0][-2:] == ["--device", "cuda:1"]
    service[1].supervisor._procs.clear()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group cleanup")
def test_force_cancel_terminates_torchrun_style_worker_tree(service, tmp_path):
    child_pid = tmp_path / "child.pid"
    program = 'import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)'
    proc = subprocess.Popen([sys.executable, "-c", program, str(child_pid)], start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not child_pid.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert child_pid.exists()
        child = psutil.Process(int(child_pid.read_text()))
        service[1].supervisor._procs["job"] = proc
        service[1].supervisor._force_kill("job", proc)
        proc.wait(timeout=5)
        deadline = time.monotonic() + 5
        while child.is_running() and child.status() != psutil.STATUS_ZOMBIE and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    finally:
        service[1].supervisor._procs.clear()
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)


@pytest.mark.skipif(os.name == "nt", reason="POSIX torchrun worker sessions")
def test_force_cancel_terminates_real_torchrun_sessions_and_preserves_other_jobs(service, tmp_path):
    worker = tmp_path / "idle_worker.py"
    worker.write_text(
        "import json,os,pathlib,subprocess,sys,time\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'],start_new_session=True)\n"
        "path=pathlib.Path(sys.argv[1])/('rank-'+os.environ['RANK']+'.json')\n"
        "path.write_text(json.dumps({'pid':os.getpid(),'pgid':os.getpgrp(),'child':child.pid}))\n"
        "time.sleep(120)\n"
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "HIP_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1"}
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True
    )
    tracked = []
    with (tmp_path / "torchrun.log").open("w") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                str(worker),
                str(tmp_path),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 30
            paths = [tmp_path / f"rank-{rank}.json" for rank in range(2)]
            while not all(path.exists() for path in paths) and time.monotonic() < deadline:
                assert proc.poll() is None, (tmp_path / "torchrun.log").read_text()
                time.sleep(0.05)
            assert all(path.exists() for path in paths), (tmp_path / "torchrun.log").read_text()
            ranks = [json.loads(path.read_text()) for path in paths]
            assert all(rank["pgid"] == rank["pid"] != proc.pid for rank in ranks)
            tracked = [psutil.Process(pid) for rank in ranks for pid in (rank["pid"], rank["child"])]
            service[1].supervisor._procs["torchrun-cancel"] = proc
            service[1].supervisor._force_kill("torchrun-cancel", proc)
            proc.wait(timeout=5)
            deadline = time.monotonic() + 5
            while (
                any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in tracked)
                and time.monotonic() < deadline
            ):
                time.sleep(0.02)
            assert all(not p.is_running() or p.status() == psutil.STATUS_ZOMBIE for p in tracked)
            assert unrelated.poll() is None
        finally:
            service[1].supervisor._procs.clear()
            if proc.poll() is None:
                tracked.extend(psutil.Process(proc.pid).children(recursive=True))
                proc.kill()
            for process in tracked:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
            proc.wait(timeout=5)
            unrelated.kill()
            unrelated.wait(timeout=5)


def test_plan_reports_gpu_admission_without_hiding_dataset_preview(service, image_dataset, monkeypatch):
    row = job(service, image_dataset)
    monkeypatch.setattr("ypuddin.server.routes_core.gpu_info", lambda: [])
    response = service[0].post("/api/plan", json={"config": json.loads(row["config_json"])})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["ok"] is False
    assert result["images"] > 0
    assert result["distributed"]["world_size"] == 2
    assert any(error["loc"] == "loop.gpu_count" for error in result["errors"])


@pytest.mark.skipif(os.name == "nt", reason="POSIX torchrun worker exit status")
def test_rank_zero_finished_cannot_hide_another_real_torchrun_rank_failure(service, image_dataset, tmp_path):
    row = job(service, image_dataset)
    supervisor = service[1].supervisor
    run_dir = Path(row["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    gate = run_dir / "allow-rank-one-failure"
    worker = tmp_path / "finish_then_fail.py"
    worker.write_text(
        "import json,os,pathlib,sys,time\n"
        "root=pathlib.Path(sys.argv[1])\n"
        "if os.environ['RANK']=='0':\n"
        " (root/'events.jsonl').write_text(json.dumps({'type':'run.finished','seq':1,'ts':time.time(),'step':1})+'\\n')\n"
        "else:\n"
        " deadline=time.monotonic()+60\n"
        " while not (root/'allow-rank-one-failure').exists() and time.monotonic()<deadline: time.sleep(.02)\n"
        " raise SystemExit(17)\n"
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "HIP_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1"}
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    with (run_dir / "run.log").open("w") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                str(worker),
                str(run_dir),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        supervisor._procs[row["id"]] = proc
        supervisor._set_status(row["id"], "running", pid=proc.pid)
        try:
            deadline = time.monotonic() + 30
            while not (run_dir / "events.jsonl").exists() and time.monotonic() < deadline:
                assert proc.poll() is None
                time.sleep(0.05)
            assert (run_dir / "events.jsonl").exists()
            supervisor._pump_events(row["id"])
            assert (
                service[1].db.fetchone("SELECT status FROM jobs WHERE id=?", (row["id"],))["status"]
                == "completed"
            )
            gate.touch()
            assert proc.wait(timeout=30) != 0
            supervisor._on_exit(row["id"], proc.returncode)
            result = service[0].get("/api/jobs/" + row["id"]).json()
            assert result["status"] == "failed"
            assert result["exit_code"] != 0
            assert "after reporting completion" in result["error"]
        finally:
            if proc.poll() is None:
                supervisor._kill_process_tree(proc)
            proc.wait(timeout=5)
            supervisor._procs.clear()


@pytest.mark.parametrize(
    "event,status,code",
    [("run.paused", "paused", 1), ("run.stopped", "cancelled", 1), ("run.finished", "completed", 0)],
)
def test_worker_exit_preserves_pause_cancel_and_success(service, image_dataset, event, status, code):
    row = job(service, image_dataset)
    supervisor = service[1].supervisor
    supervisor._set_terminal(row["id"], event)
    supervisor._on_exit(row["id"], code)
    result = service[0].get("/api/jobs/" + row["id"]).json()
    assert result["status"] == status and result["exit_code"] == code


def test_previous_paused_process_exit_does_not_fail_queued_resume(service, image_dataset):
    row = job(service, image_dataset)
    supervisor = service[1].supervisor
    supervisor._set_terminal(row["id"], "run.paused")
    supervisor.request(row["id"], "resume")
    supervisor._on_exit(row["id"], 1)
    assert service[0].get("/api/jobs/" + row["id"]).json()["status"] == "queued"
