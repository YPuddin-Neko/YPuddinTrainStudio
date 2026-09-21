"""Archive stops new work while preserving historical job and XY reads."""

import json
from unittest.mock import Mock

import pytest

from Test.tests.unit import test_supervisor_multigpu as gpu_tests
from Test.tests.unit import test_xyz_sampling as xyz_tests
from ypuddin.server import supervisor as module

api = xyz_tests.api
trained = xyz_tests.trained
gpu_service = gpu_tests.service


def attach_source(client, context):
    project = client.post("/api/projects", json={"name": "Archived work", "family": "toy"}).json()
    ownership = {"project_id": project["id"], "version_id": project["active_version_id"]}
    context.db.update("jobs", "source", ownership)
    context.db.update("artifacts", "a_trained", ownership)
    return project


def test_project_archive_blocks_new_work_and_resume_but_keeps_history(api):
    client, context = api
    project = attach_source(client, context)
    existing, _ = xyz_tests.create(api)
    config = client.get("/api/jobs/source/config").json()
    artifact = client.get("/api/artifacts/a_trained/download").content
    before_count = context.db.fetchone("SELECT count(*) AS n FROM jobs")["n"]
    assert client.patch(f"/api/projects/{project['id']}", json={"archived": True}).status_code == 200
    for kind in ("train", "cache"):
        response = client.post(
            "/api/jobs",
            json={
                "type": kind,
                "name": "blocked",
                "project_id": project["id"],
                "config": config,
            },
        )
        assert response.status_code == 409, response.text
        assert response.json()["error"]["code"] == "project.archived"
    response = client.post("/api/jobs/source/xyz", json=xyz_tests.body())
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "project.archived"
    context.db.update("jobs", "source", {"status": "paused"})
    for command in ("resume", "retry"):
        response = client.post(f"/api/jobs/source/{command}", json={})
        assert response.status_code == 409, response.text
    assert context.db.fetchone("SELECT count(*) AS n FROM jobs")["n"] == before_count
    assert client.get("/api/jobs/source").json()["status"] == "paused"
    assert client.get("/api/jobs/source/config").json() == config
    assert client.get("/api/artifacts/a_trained/download").content == artifact
    assert client.get("/api/jobs/source/xyz/options").status_code == 200
    assert [item["id"] for item in client.get("/api/jobs/source/xyz").json()] == [existing]
    assert client.get(f"/api/xyz/{existing}").status_code == 200
    assert client.patch(f"/api/projects/{project['id']}", json={"archived": False}).status_code == 200
    assert client.post("/api/jobs/source/xyz", json=xyz_tests.body()).status_code == 202
    assert client.post("/api/jobs/source/resume", json={}).status_code == 200


def test_archived_version_xyz_returns_conflict_and_keeps_history_readable(api):
    client, context = api
    project = attach_source(client, context)
    existing, _ = xyz_tests.create(api)
    context.db.update("project_versions", project["active_version_id"], {"archived": 1})
    response = client.post("/api/jobs/source/xyz", json=xyz_tests.body())
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "version.busy"
    assert client.get("/api/jobs/source/xyz/options").status_code == 200
    assert client.get(f"/api/xyz/{existing}").status_code == 200


@pytest.mark.parametrize("status", ["queued", "scheduled"])
def test_queued_job_cannot_launch_after_project_archive(gpu_service, image_dataset, monkeypatch, status):
    client, context = gpu_service
    row = gpu_tests.single_job(gpu_service, image_dataset, ["cuda:0"])
    project = client.post("/api/projects", json={"name": "Queued archive", "family": "toy"}).json()
    context.db.update(
        "jobs",
        row["id"],
        {
            "project_id": project["id"],
            "version_id": project["active_version_id"],
            "status": status,
            "scheduled_at": 1,
        },
    )
    before_config = json.loads(row["config_json"])
    assert client.patch(f"/api/projects/{project['id']}", json={"archived": True}).status_code == 200
    monkeypatch.setattr(module, "gpu_info", lambda: gpu_tests.inventory(2))
    launch = Mock()
    monkeypatch.setattr(module.subprocess, "Popen", launch)
    context.supervisor._tick()
    launch.assert_not_called()
    assert not context.supervisor._devices and not context.supervisor._procs
    current = client.get(f"/api/jobs/{row['id']}").json()
    assert current["status"] == "failed" and "归档" in current["error"]
    assert client.get(f"/api/jobs/{row['id']}/config").json() == before_config
