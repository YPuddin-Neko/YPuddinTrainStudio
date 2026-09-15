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
        {"device": f"cuda:{i}", "name": f"GPU {i}", "mem_free_mb": 8000 - i * 1000, "mem_total_mb": 8000}
        for i in range(count)
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
    service[1].db.update("jobs", row["id"], {"exit_code": 0})
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
    assert service[1].db.fetchone("SELECT exit_code FROM jobs WHERE id=?", (row["id"],))["exit_code"] is None
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


@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
def test_cache_of_multigpu_project_stays_single_card(service, image_dataset, monkeypatch, strategy):
    row = job(service, image_dataset)
    config = json.loads(row["config_json"])
    config["loop"]["distributed_strategy"] = strategy
    if strategy == "fsdp":
        config["training"]["mode"] = "full"
    # A cache request must not require the project's future training GPU count.
    response = service[0].post("/api/jobs", json={"name": "cache", "type": "cache", "config": config})
    assert response.status_code == 201, response.text
    stored = service[1].db.fetchone("SELECT * FROM jobs WHERE id=?", (response.json()["id"],))
    assert json.loads(stored["config_json"])["loop"]["gpu_count"] == 1
    assert json.loads(stored["config_json"])["loop"]["distributed_strategy"] == "ddp"
    # Older queued cache jobs may still contain the project's original FSDP
    # settings. The worker boundary must normalize those independently too.
    stored["config_json"] = json.dumps(config)
    popen = Mock(return_value=Mock(pid=987655))
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    service[1].supervisor._launch(stored, device="cuda:1")
    assert "torch.distributed.run" not in popen.call_args.args[0]
    assert popen.call_args.args[0][-2:] == ["--device", "cuda:0"]
    assert popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    worker = load_config(Path(stored["run_dir"]) / "job-config.toml")
    assert worker.loop.gpu_count == 1
    assert worker.loop.distributed_strategy == "ddp"
    service[1].supervisor._procs.clear()


def single_job(service, image_dataset, devices):
    row = job(service, image_dataset)
    config = json.loads(row["config_json"])
    config["loop"]["gpu_count"] = 1
    patch = {
        "config_json": json.dumps(config),
        "gpu_devices_json": json.dumps(devices),
        "progress_json": "{}",
    }
    service[1].db.update("jobs", row["id"], patch)
    return row | patch


def test_default_queue_starts_independent_cards_and_skips_blocked_head(service, image_dataset, monkeypatch):
    first = single_job(service, image_dataset, ["cuda:0"])
    blocked = single_job(service, image_dataset, ["cuda:0"])
    other = single_job(service, image_dataset, ["cuda:1"])
    monkeypatch.setattr(module, "gpu_info", lambda: inventory(2))
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    spawned = []

    def launch(*args, **kwargs):
        proc = Mock(pid=900000 + len(spawned))
        proc.poll.return_value = None
        spawned.append((args, kwargs, proc))
        return proc

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    sup = service[1].supervisor
    sup._tick()
    assert set(sup._procs) == {first["id"], other["id"]}
    assert [kwargs["env"]["CUDA_VISIBLE_DEVICES"] for _, kwargs, _ in spawned] == ["0", "1"]
    assert all(args[0][-2:] == ["--device", "cuda:0"] for args, _, _ in spawned)
    waiting = service[1].db.fetchone("SELECT * FROM jobs WHERE id=?", (blocked["id"],))
    assert waiting["status"] == "queued"
    assert "cuda:0" in json.loads(waiting["progress_json"])["wait_reason"]
    # A terminal event does not free the card until its process really exits.
    sup._set_terminal(first["id"], "run.finished")
    sup._tick()
    assert len(spawned) == 2
    spawned[0][2].poll.return_value = 0
    spawned[0][2].returncode = 0
    sup._tick()
    assert set(sup._procs) == {blocked["id"], other["id"]}
    assert sup._devices[other["id"]] == "cuda:1"
    assert spawned[1][2].poll.return_value is None
    sup._procs.clear()


def test_explicit_concurrency_limit_can_be_returned_to_automatic(service, image_dataset, monkeypatch):
    first = single_job(service, image_dataset, ["cuda:0"])
    other = single_job(service, image_dataset, ["cuda:1"])
    monkeypatch.setattr(module, "gpu_info", lambda: inventory(2))
    proc = Mock(pid=900100)
    proc.poll.return_value = None
    monkeypatch.setattr(module.subprocess, "Popen", Mock(return_value=proc))
    client, ctx = service
    assert client.get("/api/queue/settings").json()["max_concurrent"] is None
    assert client.put("/api/queue/settings", json={"max_concurrent": 1}).status_code == 200
    ctx.supervisor._tick()
    assert list(ctx.supervisor._procs) == [first["id"]]
    assert client.put("/api/queue/settings", json={"max_concurrent": None}).status_code == 200
    ctx.supervisor._tick()
    assert set(ctx.supervisor._procs) == {first["id"], other["id"]}
    ctx.supervisor._procs.clear()


def test_requested_device_validation_patch_inventory_and_retry(service, image_dataset, monkeypatch):
    row = single_job(service, image_dataset, [])
    client, ctx = service
    monkeypatch.setattr("ypuddin.server.routes_work.gpu_info", lambda: inventory(2))
    monkeypatch.setattr(module, "gpu_info", lambda: inventory(2))
    for devices in (["cuda:2"], ["cuda:0", "cuda:1"], ["cuda:0", "cuda:0"], ["cuda:-1"]):
        response = client.patch(f"/api/jobs/{row['id']}", json={"gpu_devices": devices})
        assert response.status_code == 422, response.text
    response = client.patch(f"/api/jobs/{row['id']}", json={"gpu_devices": ["cuda:1"]})
    assert response.status_code == 200
    assert response.json()["gpu_devices"] == ["cuda:1"]
    assert "gpu_devices_json" not in response.json()
    stored = ctx.db.fetchone("SELECT * FROM jobs WHERE id=?", (row["id"],))
    assert ctx.supervisor._choose_device(stored) == "cuda:1"
    cloned = ctx.supervisor.clone(stored)
    assert json.loads(cloned["gpu_devices_json"]) == ["cuda:1"]
    ctx.supervisor._devices[row["id"]] = "cuda:1"
    ctx.db.update("jobs", row["id"], {"status": "running"})
    cards = client.get("/api/queue/devices").json()["devices"]
    assert cards[0]["job_id"] is None
    assert cards[1]["job_id"] == row["id"]
    assert cards[1]["job_name"] == "ownership"
    assert client.patch(f"/api/jobs/{row['id']}", json={"gpu_devices": []}).status_code == 409
    # Removing a card never quietly moves a pinned task to a different GPU or CPU.
    ctx.supervisor._devices.clear()
    monkeypatch.setattr(module, "gpu_info", lambda: [])
    assert ctx.supervisor._choose_device(cloned) is None
    assert ctx.db.fetchone("SELECT status FROM jobs WHERE id=?", (cloned["id"],))["status"] == "failed"


def test_old_job_retains_compute_mode_and_original_rng_device_on_resume(service, image_dataset, monkeypatch):
    row = single_job(service, image_dataset, ["cuda:0"])
    config = json.loads(row["config_json"])
    config["loop"].pop("deterministic", None)
    row["config_json"] = json.dumps(config)
    row["progress_json"] = json.dumps({"devices": ["cuda:1"], "device": "cuda:1"})
    row["resume_from"] = str(Path(row["run_dir"]) / "legacy-state")
    popen = Mock(return_value=Mock(pid=900021))
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    service[1].supervisor._launch(row, device="cuda:0")
    assert load_config(Path(row["run_dir"]) / "job-config.toml").loop.deterministic is False
    assert popen.call_args.kwargs["env"]["YPUDDIN_LEGACY_CUDA_RNG_INDEX"] == "1"
    service[1].supervisor._procs.clear()


def test_gpu_patch_during_hardware_probe_is_honoured_at_launch(service, image_dataset, monkeypatch):
    row = single_job(service, image_dataset, ["cuda:0"])
    client, ctx = service
    monkeypatch.setattr(module, "gpu_info", lambda: inventory(2))
    monkeypatch.setattr("ypuddin.server.routes_work.gpu_info", lambda: inventory(2))
    original_choose = ctx.supervisor._choose_device

    def interleaved_patch(job, **kwargs):
        allocation = original_choose(job, **kwargs)
        response = client.patch(f"/api/jobs/{job['id']}", json={"gpu_devices": ["cuda:1"]})
        assert response.status_code == 200, response.text
        assert response.json()["gpu_devices"] == ["cuda:1"]
        return allocation

    launch = Mock()
    monkeypatch.setattr(ctx.supervisor, "_choose_device", interleaved_patch)
    monkeypatch.setattr(ctx.supervisor, "_launch", launch)
    ctx.supervisor._tick()
    if not launch.called:
        assert ctx.db.fetchone("SELECT status FROM jobs WHERE id=?", (row["id"],))["status"] == "queued"
        monkeypatch.setattr(ctx.supervisor, "_choose_device", original_choose)
        ctx.supervisor._tick()
    launch.assert_called_once()
    assert json.loads(launch.call_args.args[0]["gpu_devices_json"]) == ["cuda:1"]
    assert launch.call_args.kwargs["device"] == "cuda:1"


def test_failed_first_legacy_resume_keeps_checkpoint_rng_origin(service, image_dataset, monkeypatch):
    row = single_job(service, image_dataset, ["cuda:0"])
    _, ctx = service
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)
    popen = Mock(return_value=Mock(pid=987654))
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    state = Path(row["run_dir"]) / "legacy-state"
    state.mkdir(parents=True)
    (state / "state.json").write_text("{}")
    ctx.db.update(
        "jobs",
        row["id"],
        {
            "status": "paused",
            "resume_from": str(state),
            "progress_json": json.dumps({"device": "cuda:1", "step": 8}),
        },
    )
    first = ctx.db.fetchone("SELECT * FROM jobs WHERE id=?", (row["id"],))
    ctx.supervisor._launch(first, device="cuda:0")
    assert popen.call_args.kwargs["env"]["YPUDDIN_LEGACY_CUDA_RNG_INDEX"] == "1"
    # Simulate an import/model preparation failure before reading or replacing the old state.
    ctx.supervisor._on_exit(row["id"], 1)
    ctx.supervisor._procs.clear()
    ctx.supervisor._devices.clear()
    ctx.supervisor.request(row["id"], "resume")
    second = ctx.db.fetchone("SELECT * FROM jobs WHERE id=?", (row["id"],))
    assert second["resume_from"] == str(state)
    ctx.supervisor._launch(second, device="cuda:0")
    assert popen.call_args.kwargs["env"]["YPUDDIN_LEGACY_CUDA_RNG_INDEX"] == "1"
    assert (state / "state.json").read_text() == "{}"
    ctx.supervisor._procs.clear()


@pytest.mark.parametrize("feedback", ["missing", "waiting"])
def test_stale_admission_feedback_after_gpu_patch_defers_to_next_tick(
    service, image_dataset, monkeypatch, feedback
):
    row = single_job(service, image_dataset, ["cuda:1"])
    client, ctx = service
    ctx.db.set_kv("queue.settings", {"max_concurrent": 2})
    cards = inventory(1 if feedback == "missing" else 2)
    monkeypatch.setattr("ypuddin.server.routes_work.gpu_info", lambda: cards)
    if feedback == "waiting":
        ctx.supervisor._devices["other"] = "cuda:1"
    previous = {"device": "cuda:1", "devices": ["cuda:1"], "step": 8, "phase": "paused"}
    ctx.db.update("jobs", row["id"], {"progress_json": json.dumps(previous)})

    def probe():
        response = client.patch(f"/api/jobs/{row['id']}", json={"gpu_devices": ["cuda:0"]})
        assert response.status_code == 200, response.text
        return cards

    monkeypatch.setattr(module, "gpu_info", probe)
    popen = Mock(return_value=Mock(pid=987655))
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    ctx.supervisor._tick()
    current = client.get(f"/api/jobs/{row['id']}").json()
    assert current["status"] == "queued" and current["error"] is None
    assert current["gpu_devices"] == ["cuda:0"] and current["progress"] == previous
    popen.assert_not_called()
    monkeypatch.setattr(module, "gpu_info", lambda: cards)
    ctx.supervisor._tick()
    popen.assert_called_once()
    current = client.get(f"/api/jobs/{row['id']}").json()
    assert current["status"] == "running" and current["progress"]["devices"] == ["cuda:0"]
    assert ctx.supervisor._devices[row["id"]] == "cuda:0"
    ctx.supervisor._procs.clear()
    ctx.supervisor._devices.clear()


@pytest.mark.parametrize("feedback", ["missing", "waiting"])
@pytest.mark.parametrize("command,status", [("pause", "paused"), ("cancel", "cancelled")])
def test_stale_admission_feedback_keeps_user_control_and_previous_device(
    service, image_dataset, monkeypatch, feedback, command, status
):
    row = single_job(service, image_dataset, ["cuda:1"])
    client, ctx = service
    previous = {"device": "cuda:0", "devices": ["cuda:0"], "step": 7}
    ctx.db.update("jobs", row["id"], {"progress_json": json.dumps(previous)})
    if feedback == "waiting":
        ctx.supervisor._devices["other"] = "cuda:1"

    def probe():
        assert client.post(f"/api/jobs/{row['id']}/{command}", json={}).status_code == 200
        return inventory(1 if feedback == "missing" else 2)

    monkeypatch.setattr(module, "gpu_info", probe)
    assert ctx.supervisor._choose_device(row) is None
    current = client.get(f"/api/jobs/{row['id']}").json()
    assert current["status"] == status and current["error"] is None
    assert current["progress"] == previous


def test_waiting_feedback_preserves_live_progress_and_actual_device(service, image_dataset, monkeypatch):
    row = single_job(service, image_dataset, ["cuda:1"])
    client, ctx = service
    ctx.supervisor._devices["other"] = "cuda:1"
    latest = {"device": "cuda:0", "devices": ["cuda:0"], "step": 9}

    def probe():
        ctx.db.update("jobs", row["id"], {"progress_json": json.dumps(latest)})
        return inventory(2)

    monkeypatch.setattr(module, "gpu_info", probe)
    assert ctx.supervisor._choose_device(row) is None
    current = client.get(f"/api/jobs/{row['id']}").json()
    assert current["status"] == "queued"
    assert current["progress"]["phase"] == "waiting_for_device"
    assert "cuda:1" in current["progress"]["wait_reason"]
    assert {key: current["progress"][key] for key in latest} == latest


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


@pytest.mark.parametrize(
    "phases",
    [
        None,
        {},
        {"text_cache": None},
        {"vae_cache": 1200, "text_cache": None},
        {"text_cache": float("inf")},
        {"text_cache": -1},
        {"text_cache": "22000"},
    ],
)
def test_cache_with_unknown_phase_estimate_never_inherits_training_peak(
    service, image_dataset, monkeypatch, phases
):
    row = job(service, image_dataset)
    config = json.loads(row["config_json"])
    config["training"]["mode"] = "full"
    config["loop"]["distributed_strategy"] = "fsdp"
    monkeypatch.setattr(
        "ypuddin.train.plan.plan",
        lambda *_args, **_kwargs: {
            "ok": True,
            "errors": [],
            "warnings": [],
            "memory": {"peak_mb_estimate": 191 * 1024, "cache_phase_peak_mb_estimates": phases},
        },
    )
    response = service[0].post(
        "/api/jobs", json={"type": "cache", "name": "unknown cache estimate", "config": config}
    )
    assert response.status_code == 201, response.text
    cached = service[1].db.fetchone("SELECT * FROM jobs WHERE id=?", (response.json()["id"],))
    assert json.loads(cached["progress_json"])["estimated_peak_mb"] is None
    monkeypatch.setattr(module, "gpu_info", lambda: inventory(1))
    assert service[1].supervisor._choose_device(cached) == "cuda:0"
