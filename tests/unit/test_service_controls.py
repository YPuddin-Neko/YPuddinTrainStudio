"""Service contracts that affect checkpoint safety, resource ownership and saved configuration."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from ypuddin.server import create_app, hardware
from ypuddin.server import supervisor as supervisor_module


@pytest.fixture
def api(tmp_path, monkeypatch):
    # No lifespan: requests and DB contracts are exercised without launching workers.
    monkeypatch.setattr("ypuddin.server.routes_work.gpu_info", lambda: [])
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    client = TestClient(app)
    yield client, app.state.ctx
    client.close()
    app.state.ctx.db.close()


def config(dataset):
    return {
        "model": {"family": "toy", "dtype": "fp32"},
        "dataset": {
            "sources": [{"path": str(dataset)}],
            "resolutions": [64],
            "bucket_step": 16,
            "num_workers": 0,
        },
        "loop": {"epochs": 1, "mixed_precision": "no"},
    }


def create_job(api, dataset, **fields):
    client, ctx = api
    response = client.post(
        "/api/jobs", json={"type": "train", "name": "test", "config": config(dataset), **fields}
    )
    assert response.status_code == 201, response.text
    return ctx.db.fetchone("SELECT * FROM jobs WHERE id=?", (response.json()["id"],))


def test_config_roundtrip_and_invalid_input(api):
    client, _ = api
    defaults = client.get("/api/config/defaults").json()
    defaults["dataset"]["caption"]["trigger_word"] = "训练词"
    defaults["loop"].update(epochs=None, max_steps=3)
    defaults["sampling"]["every_epochs"] = None
    defaults["validation"]["every_epochs"] = None
    defaults["checkpoint"]["save_every_epochs"] = None
    defaults["logging"]["wandb"] = {"project": "ypuddin", "run_name": "my run", "entity": None}
    for fmt in ("toml", "json"):
        output = client.post("/api/config/export", json={"config": defaults, "format": fmt})
        assert output.status_code == 200, output.text
        restored = client.post("/api/config/import", json={"text": output.json()["text"], "format": fmt})
        assert restored.json() == defaults
    bad = client.post("/api/config/import", json={"text": "broken=[", "format": "toml"})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "config.parse"
    defaults["optimizer"]["fused_backward"] = True
    bad = client.post("/api/config/export", json={"config": defaults, "format": "toml"})
    assert bad.status_code == 400 and "fused_backward" in bad.text


def test_job_preflight_paths_and_isolated_events(api, image_dataset, monkeypatch):
    client, ctx = api
    monkeypatch.chdir(image_dataset.parent)
    cfg = config("data")
    cfg["logging"] = {"events_path": "foreign-events.jsonl"}
    job = create_job(api, image_dataset, config=cfg)
    stored = json.loads(job["config_json"])
    assert stored["dataset"]["sources"][0]["path"] == str(image_dataset)
    assert stored["logging"]["events_path"] == str(Path(job["run_dir"]) / "events.jsonl")
    assert stored["dataset"]["cache_dir"] == str(ctx.cache_dir(None))
    cfg["checkpoint"] = {"resume": "missing-state"}
    bad = client.post("/api/jobs", json={"name": "invalid", "config": cfg})
    assert bad.status_code == 400 and "checkpoint.resume" in bad.text
    cfg["dataset"]["sources"] = []
    bad = client.post("/api/jobs", json={"name": "invalid", "config": cfg})
    assert bad.status_code == 400 and "dataset" in bad.text


def test_settings_new_paths_preserve_old_jobs_and_delete_custom_run(api, image_dataset, tmp_path):
    client, ctx = api
    old = create_job(api, image_dataset)
    old_run = Path(old["run_dir"])
    old_samples = Path(old["samples_dir"] or old_run / "samples")
    old_run.mkdir(parents=True, exist_ok=True)
    old_samples.mkdir(parents=True, exist_ok=True)
    old_weights = old_run / "old-weights"
    old_sample = old_samples / "old-sample.png"
    old_weights.write_bytes(b"existing weights")
    old_sample.write_bytes(b"existing sample")
    paths = {name: str(tmp_path / name) for name in ("cache_dir", "output_dir", "models_dir")}
    result = client.put("/api/settings", json={"paths": paths, "server": {"port": 9123}})
    assert result.status_code == 200, result.text
    pid = client.post("/api/projects", json={"name": "custom"}).json()["id"]
    new = create_job(api, image_dataset, project_id=pid)
    assert Path(new["run_dir"]) == tmp_path / "output_dir" / pid / "v1" / new["id"]
    expected_samples = ctx.data_root / "project" / pid / "v1" / "samples" / new["id"]
    assert Path(new["samples_dir"]) == expected_samples
    assert json.loads(new["config_json"])["sampling"]["output_dir"] == str(expected_samples)
    assert json.loads(new["config_json"])["dataset"]["cache_dir"] == str(
        tmp_path / "cache_dir" / pid / new["version_id"]
    )
    saved_old = ctx.db.fetchone("SELECT * FROM jobs WHERE id=?", (old["id"],))
    assert {key: saved_old[key] for key in ("run_dir", "samples_dir", "config_json")} == {
        key: old[key] for key in ("run_dir", "samples_dir", "config_json")
    }
    assert client.put("/api/settings", json={"paths": {"data_root": str(tmp_path)}}).status_code == 400
    assert client.put("/api/settings", json={"server": {"port": 99999}}).status_code == 400
    run = Path(new["run_dir"])
    run.mkdir(parents=True)
    (run / "weights").write_text("keep until explicit deletion")
    unrelated = run.parent / "user-file"
    unrelated.write_text("preserve")
    assert client.delete(f"/api/projects/{pid}?delete_files=true").status_code == 409
    ctx.db.update("jobs", new["id"], {"status": "completed"})
    assert client.delete(f"/api/projects/{pid}?delete_files=true").status_code == 200
    assert not run.exists() and unrelated.exists()
    assert old_weights.read_bytes() == b"existing weights"
    assert old_sample.read_bytes() == b"existing sample"


def test_settings_relocated_root_and_concurrent_patch(api, tmp_path):
    _, ctx = api
    saved = ctx.settings()
    saved["paths"]["data_root"] = str(tmp_path / "old-machine")
    ctx.settings_path.write_text(json.dumps(saved))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(ctx.save_settings, [{"ui": {"language": "en"}}, {"server": {"port": 9123}}]))
    result = ctx.settings()
    assert result["paths"]["data_root"] == str(ctx.data_root)
    assert result["ui"]["language"] == "en" and result["server"]["port"] == 9123


def test_device_admission_exclusive_and_queue_validation(api, image_dataset, monkeypatch):
    client, ctx = api
    job = create_job(api, image_dataset)
    job["progress_json"] = json.dumps({"estimated_peak_mb": 2000})
    monkeypatch.setattr(
        supervisor_module,
        "gpu_info",
        lambda: [
            {"device": "cuda:0", "mem_free_mb": 8000},
            {"device": "cuda:1", "mem_free_mb": 1000},
        ],
    )
    s = ctx.supervisor
    assert s._choose_device(job) == "cuda:0"
    s._devices["other"] = "cuda:0"
    assert s._choose_device(job) is None
    assert s._choose_device(job, check_memory=False) == "cuda:1"
    assert client.put("/api/queue/settings", json={"max_concurrent": 0}).status_code == 422
    assert (
        client.put("/api/queue/settings", json={"held": True, "memory_admission": False}).json()[
            "memory_admission"
        ]
        is False
    )


def test_preparing_pause_preserves_resume_and_clears_stale_phase(api, image_dataset, tmp_path):
    _, ctx = api
    job = create_job(api, image_dataset)
    state = tmp_path / "state-source"
    state.mkdir()
    (state / "state.json").write_text("{}")
    s = ctx.supervisor
    ctx.db.update("jobs", job["id"], {"status": "running", "resume_from": str(state)})
    s._handle_event(job["id"], {"type": "run.paused", "preparing": True})
    resumed = s.request(job["id"], "resume")
    assert resumed["resume_from"] == str(state)
    s._handle_event(job["id"], {"type": "run.prepared", "total_steps": 10, "steps_per_epoch": 5})
    ready = ctx.db.fetchone("SELECT * FROM jobs WHERE id=?", (job["id"],))
    assert ready["resume_from"] is None and not json.loads(ready["progress_json"])["preparing"]
    s._handle_event(job["id"], {"type": "run.failed", "error": "specific failure"})
    assert ctx.db.fetchone("SELECT error FROM jobs WHERE id=?", (job["id"],))["error"] == "specific failure"
    retry = s.clone(ready)
    cfg = json.loads(retry["config_json"])
    assert cfg["checkpoint"]["resume"] is None
    assert cfg["logging"]["events_path"] == str(Path(retry["run_dir"]) / "events.jsonl")


@pytest.mark.asyncio
async def test_cancel_from_sync_worker_and_stale_kill_timer(api, image_dataset):
    client, ctx = api
    job = create_job(api, image_dataset)
    s = ctx.supervisor
    s._event_loop = asyncio.get_running_loop()
    old, new = Mock(), Mock()
    old.poll.return_value = new.poll.return_value = None
    s._procs[job["id"]] = old
    ctx.db.update("jobs", job["id"], {"status": "running"})
    # This is a FastAPI synchronous route, hence it runs in a worker with no event loop.
    response = await asyncio.to_thread(client.post, f"/api/jobs/{job['id']}/cancel")
    assert response.status_code == 200 and response.json()["status"] == "cancelling"
    s._procs[job["id"]] = new
    s._force_kill(job["id"], old)
    new.kill.assert_not_called()
    s._procs.clear()


@pytest.mark.asyncio
async def test_shutdown_preserves_reported_terminal_before_process_exit(api, image_dataset):
    _, ctx = api
    job = create_job(api, image_dataset)
    s = ctx.supervisor
    run = Path(job["run_dir"])
    run.mkdir(parents=True)
    (run / "events.jsonl").write_text('{"type":"run.finished"}\n')
    ctx.db.update("jobs", job["id"], {"status": "running"})
    proc = Mock()
    proc.poll.side_effect = [None, 0]
    proc.returncode = 0
    s._procs[job["id"]] = proc
    await s.stop()
    final = ctx.db.fetchone("SELECT * FROM jobs WHERE id=?", (job["id"],))
    assert final["status"] == "completed" and final["exit_code"] == 0
    assert not (run / "control" / "pause").exists()


def test_mps_stats_are_unified_memory_without_fake_telemetry(monkeypatch):
    monkeypatch.setattr(hardware.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(hardware.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(
        hardware.psutil, "virtual_memory", lambda: SimpleNamespace(total=16 * 2**30, available=10 * 2**30)
    )
    monkeypatch.setattr(hardware, "_apple_name", lambda: "Apple Test GPU")
    stats = hardware.gpu_info()[0]
    assert stats["device"] == "mps" and stats["memory_scope"] == "unified_system"
    assert stats["mem_used_mb"] == 6144 and stats["mem_free_mb"] == 10240
    assert stats["util_pct"] is None and stats["temp_c"] is None and stats["power_w"] is None


def test_memory_metric_survives_rest_and_rotated_files_are_hidden(api, image_dataset):
    client, ctx = api
    job = create_job(api, image_dataset)
    run = Path(job["run_dir"])
    run.mkdir(parents=True)
    weights = run / "step-1.safetensors"
    weights.write_bytes(b"placeholder")
    event = {
        "type": "checkpoint.saved",
        "path": str(weights),
        "kind": "weights",
        "step": 1,
        "ts": 1,
        "ema": True,
    }
    step = {
        "type": "step",
        "step": 1,
        "epoch": 0,
        "loss": 1,
        "vram_mb": 2.5,
        "vram_metric": "current_allocated",
    }
    (run / "events.jsonl").write_text(json.dumps(step) + "\n" + json.dumps(event) + "\n")
    ctx.supervisor._handle_event(job["id"], step)
    ctx.supervisor._handle_event(job["id"], event)
    assert client.get(f"/api/jobs/{job['id']}").json()["progress"]["vram_metric"] == "current_allocated"
    assert client.get(f"/api/jobs/{job['id']}/metrics").json()["vram_metric"] == "current_allocated"
    assert client.get(f"/api/jobs/{job['id']}/checkpoints").json()[0]["ema"] is True
    weights.unlink()
    assert client.get(f"/api/jobs/{job['id']}/checkpoints").json() == []
    assert client.get("/api/artifacts").json() == []


@pytest.mark.parametrize("change", ["pause", "delete"])
def test_queue_rechecks_ownership_after_hardware_selection(api, image_dataset, monkeypatch, change):
    client, ctx = api
    job = create_job(api, image_dataset)
    launched = Mock()

    def select_device(*_args, **_kwargs):
        if change == "pause":
            ctx.supervisor.request(job["id"], "pause")
        else:
            assert client.delete(f"/api/jobs/{job['id']}").status_code == 200
        return "cpu"

    monkeypatch.setattr(ctx.supervisor, "_choose_device", select_device)
    monkeypatch.setattr(ctx.supervisor, "_launch", launched)
    ctx.supervisor._tick()
    launched.assert_not_called()
