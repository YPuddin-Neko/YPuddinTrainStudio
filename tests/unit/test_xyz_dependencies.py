"""XYZ history owns its source record; active workers also own checkpoint references."""

import hashlib
import json
import shutil
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.unit.test_xyz_sampling import body, trained  # noqa: F401
from ypuddin.server import create_app, xyz
from ypuddin.server.db import now


@pytest.fixture
def api(tmp_path, trained, monkeypatch):  # noqa: F811
    monkeypatch.setattr("ypuddin.server.supervisor.gpu_info", lambda: [])
    app = create_app(tmp_path / "studio", poll_interval=0.02)
    context = app.state.ctx
    config, original = trained
    # Every deletion test owns its copy, never the module's trained checkpoint.
    run = context.data_root / "runs" / "source"
    run.mkdir(parents=True)
    path = run / original.name
    shutil.copy2(original, path)
    config = config.model_copy(deep=True)
    config.checkpoint.output_dir = str(run)
    (run / "source-config.json").write_text(json.dumps(config.to_dict()))
    context.db.insert(
        "jobs",
        {
            "id": "source",
            "type": "train",
            "name": "trained",
            "status": "completed",
            "created_at": now(),
            "run_dir": str(run),
            "config_json": json.dumps(config.to_dict()),
        },
    )
    context.db.insert(
        "artifacts",
        {
            "id": "a_trained",
            "job_id": "source",
            "name": path.name,
            "path": str(path),
            "kind": "weights",
            "step": 2,
            "created_at": now(),
        },
    )
    context.db.set_kv("queue.settings", {"held": True, "max_concurrent": 1})
    with TestClient(app) as client:
        yield client, context, path


def enqueue(api, **patch):
    response = api[0].post("/api/jobs/source/xyz", json=body(**patch))
    assert response.status_code == 202, response.text
    return response.json()["id"]


def attach_project(api):
    client, context, _ = api
    response = client.post("/api/projects", json={"name": "XYZ deletion", "family": "toy"})
    assert response.status_code == 201, response.text
    project = response.json()
    owner = {"project_id": project["id"], "version_id": project["active_version_id"]}
    context.db.update("jobs", "source", owner)
    context.db.update("artifacts", "a_trained", owner)
    return project["id"]


@pytest.mark.parametrize("status", ["queued", "scheduled", "running", "pausing", "cancelling"])
def test_active_xyz_protects_only_referenced_checkpoints_and_source(api, status):
    client, context, path = api
    other = path.with_name("unselected.safetensors")
    shutil.copy2(path, other)
    context.db.insert(
        "artifacts",
        {
            "id": "a_unselected",
            "job_id": "source",
            "name": other.name,
            "path": str(other),
            "kind": "weights",
            "created_at": now(),
        },
    )
    jid = enqueue(api)
    context.db.update("jobs", jid, {"status": status})
    source = context.db.fetchone("SELECT * FROM jobs WHERE id='source'")
    artifact = context.db.fetchone("SELECT * FROM artifacts WHERE id='a_trained'")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for delete_files in (False, True):
        rejected = client.delete("/api/jobs/source", params={"delete_files": delete_files})
        assert rejected.status_code == 409, rejected.text
        assert rejected.json()["error"]["code"] == "job.xyz_dependencies"
        rejected = client.delete("/api/artifacts/a_trained", params={"delete_file": delete_files})
        assert rejected.status_code == 409, rejected.text
        assert rejected.json()["error"]["code"] == "artifact.xyz_dependencies"
    assert context.db.fetchone("SELECT * FROM jobs WHERE id='source'") == source
    assert context.db.fetchone("SELECT * FROM artifacts WHERE id='a_trained'") == artifact
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert client.delete("/api/artifacts/a_unselected?delete_file=true").status_code == 200
    assert not other.exists()


def test_checkpoint_axis_protects_each_selected_weight(api):
    client, context, path = api
    second = path.with_name("step1.safetensors")
    shutil.copy2(path, second)
    context.db.insert(
        "artifacts",
        {
            "id": "a_second",
            "job_id": "source",
            "name": second.name,
            "path": str(second),
            "kind": "weights",
            "step": 1,
            "created_at": now(),
        },
    )
    enqueue(api, x={"key": "checkpoint", "values": ["a_trained", "a_second"]})
    for aid, weight in (("a_trained", path), ("a_second", second)):
        response = client.delete(f"/api/artifacts/{aid}?delete_file=true")
        assert response.status_code == 409, response.text
        assert weight.is_file()


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_terminal_event_does_not_release_weights_before_worker_exit(api, monkeypatch, status):
    client, context, path = api
    jid = enqueue(api)
    context.db.update("jobs", jid, {"status": status})
    monkeypatch.setattr(context.supervisor, "is_running", lambda candidate: candidate == jid)
    assert client.delete("/api/artifacts/a_trained?delete_file=true").status_code == 409
    assert path.is_file()
    monkeypatch.setattr(context.supervisor, "is_running", lambda _: False)
    assert client.delete("/api/artifacts/a_trained?delete_file=true").status_code == 200
    assert not path.exists()
    assert client.delete("/api/jobs/source").status_code == 409


def test_queued_cancel_releases_weights_but_history_keeps_source_until_xyz_deleted(api):
    client, _, path = api
    jid = enqueue(api)
    cancelled = client.post(f"/api/xyz/{jid}/cancel")
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
    assert client.delete("/api/artifacts/a_trained?delete_file=true").status_code == 200
    assert not path.exists()
    assert client.delete("/api/jobs/source?delete_files=true").status_code == 409
    assert client.get("/api/jobs/source/xyz/options").status_code == 200
    assert client.get("/api/jobs/source/xyz").json()[0]["id"] == jid
    assert client.delete(f"/api/jobs/{jid}?delete_files=true").status_code == 200
    assert client.get("/api/jobs/source/xyz").json() == []
    assert client.delete("/api/jobs/source?delete_files=true").status_code == 200
    assert not path.parent.exists()


def test_real_completed_xyz_keeps_manifest_and_png_after_weight_removal(api):
    client, context, path = api
    jid = enqueue(api)
    source_config = client.get("/api/jobs/source/config").json()
    assert client.put("/api/queue/settings", json={"held": False, "max_concurrent": 1}).status_code == 200
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        task = client.get(f"/api/xyz/{jid}").json()
        if task["status"] in {"completed", "failed", "cancelled"} and not context.supervisor.is_running(jid):
            break
        time.sleep(0.05)
    assert task["status"] == "completed", task
    assert not context.supervisor.is_running(jid)
    assert context.db.fetchone("SELECT exit_code FROM jobs WHERE id=?", (jid,))["exit_code"] == 0
    assert task["done"] == 2 and task["manifest"]["complete"]
    published = task["manifest"]["cells"] + task["manifest"]["grids"]
    images = {entry["url"]: client.get(entry["url"]).content for entry in published}
    assert all(content.startswith(b"\x89PNG\r\n\x1a\n") for content in images.values())
    samples = Path(context.db.fetchone("SELECT samples_dir FROM jobs WHERE id=?", (jid,))["samples_dir"])
    manifest = (samples / "manifest.json").read_bytes()
    assert client.delete("/api/artifacts/a_trained?delete_file=true").status_code == 200
    assert not path.exists()
    assert client.delete("/api/jobs/source?delete_files=true").status_code == 409
    assert client.get("/api/jobs/source/config").json() == source_config
    assert client.get("/api/jobs/source/xyz/options").json()["checkpoints"] == []
    assert client.get("/api/jobs/source/xyz").json()[0]["manifest"] == task["manifest"]
    assert (samples / "manifest.json").read_bytes() == manifest
    assert {url: client.get(url).content for url in images} == images
    assert client.delete(f"/api/jobs/{jid}?delete_files=true").status_code == 200
    assert not samples.exists()
    assert client.delete("/api/jobs/source?delete_files=true").status_code == 200


@pytest.mark.parametrize("delete_files", [False, True])
def test_train_without_xyz_retains_existing_deletion_behavior(api, delete_files):
    client, _, path = api
    assert client.delete("/api/jobs/source", params={"delete_files": delete_files}).status_code == 200
    assert client.get("/api/jobs/source").status_code == 404
    assert path.exists() is not delete_files


def test_project_delete_blocks_active_xyz_then_cascades_its_complete_history(api):
    client, context, path = api
    pid = attach_project(api)
    jid = enqueue(api)
    assert client.patch(f"/api/projects/{pid}", json={"archived": True}).status_code == 200
    assert client.delete(f"/api/projects/{pid}?delete_files=true").status_code == 409
    assert client.get("/api/jobs/source").status_code == 200
    assert client.get(f"/api/xyz/{jid}").status_code == 200
    context.db.update("jobs", jid, {"status": "completed"})
    assert client.delete(f"/api/projects/{pid}?delete_files=true").status_code == 200
    assert client.get(f"/api/xyz/{jid}").status_code == 404
    assert client.get("/api/jobs/source").status_code == 404
    assert context.db.fetchall("SELECT * FROM jobs") == []
    assert context.db.fetchall("SELECT * FROM artifacts") == []
    assert not path.exists()


@pytest.mark.parametrize("delete_target", ["job", "artifact", "project"])
def test_deletion_during_xyz_validation_cannot_publish_dangling_references(api, monkeypatch, delete_target):
    client, context, _ = api
    pid = attach_project(api) if delete_target == "project" else None
    validated = threading.Event()
    release = threading.Event()
    original = xyz.file_signature

    def pause_after_validation(path):
        signature = original(path)
        validated.set()
        assert release.wait(timeout=10), "deletion did not release XYZ validation"
        return signature

    monkeypatch.setattr(xyz, "file_signature", pause_after_validation)
    responses = []
    errors = []

    def submit():
        try:
            responses.append(client.post("/api/jobs/source/xyz", json=body()))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=submit)
    thread.start()
    try:
        assert validated.wait(timeout=10), "XYZ validation did not inspect the checkpoint"
        if delete_target == "project":
            assert client.patch(f"/api/projects/{pid}", json={"archived": True}).status_code == 200
            response = client.delete(f"/api/projects/{pid}?delete_files=true")
        else:
            url = (
                "/api/jobs/source?delete_files=true"
                if delete_target == "job"
                else "/api/artifacts/a_trained?delete_file=true"
            )
            response = client.delete(url)
        assert response.status_code == 200, response.text
    finally:
        release.set()
        thread.join(timeout=10)
    assert not thread.is_alive() and not errors, errors
    assert responses[0].status_code == (422 if delete_target == "artifact" else 404), responses[0].text
    assert context.db.fetchall("SELECT * FROM jobs WHERE type='xyz'") == []
