import io
import json
from pathlib import Path, PureWindowsPath

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ypuddin.config import TrainConfig
from ypuddin.server import create_app, routes_core, routes_work
from ypuddin.server.output_binding import project_weight_name
from ypuddin.train import Trainer


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(routes_work, "gpu_info", lambda: [])
    monkeypatch.setattr(routes_core, "gpu_info", lambda: [])
    monkeypatch.setattr(routes_work, "_cache_stats", lambda c, row: {})
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "no-ui")
    client = TestClient(app)
    project = client.post("/api/projects", json={"name": "测试角色 / 星光", "family": "toy"}).json()
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(buffer, "PNG")
    response = client.post(
        f"/api/projects/{project['id']}/datasets/upload",
        files={"files": ("p.png", buffer.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    yield client, app.state.ctx, project
    app.state.regularization.close()
    app.state.dataset_pipeline.close()
    app.state.ctx.versions.close()
    client.close()
    app.state.ctx.db.close()


@pytest.mark.parametrize("name", ["CON", "COM1.txt", " . ", "\x00", 'a/b\\c:*?"<>|', "中文🙂" * 100])
def test_generated_names_are_safe_on_windows_and_byte_limited(name):
    result = project_weight_name(name, "p_fallback", 12)
    assert result.endswith("_v12")
    assert len(result.encode("utf-8")) <= 180
    assert not PureWindowsPath(result).is_reserved()
    TrainConfig.model_validate({"checkpoint": {"name": result}})
    assert "/" not in result and "\\" not in result


def test_readonly_binding_and_real_enqueued_toy_export_match_without_rewriting_recipe(api):
    client, c, p = api
    cfg_url = f"/api/projects/{p['id']}/config"
    cfg = client.get(cfg_url).json()
    original = c.config_path(p["id"]).read_bytes()
    cfg["loop"].update(epochs=None, max_steps=1, mixed_precision="no")
    cfg["dataset"]["num_workers"] = 0
    cfg["sampling"]["enabled"] = False
    cfg["checkpoint"].update(name="", save_every_epochs=None)
    response = client.post(f"/api/projects/{p['id']}/output-binding", json={"config": cfg})
    assert response.status_code == 200, response.text
    binding = response.json()
    assert binding["name"] == "测试角色 _ 星光_v1"
    assert binding["automatic_name"] and binding["inherits_output_dir"]
    assert binding["directory_template"] == str(c.version_dir(p["id"]) / "output" / "{job_id}")
    assert not Path(binding["directory_template"]).exists()
    assert client.post("/api/plan", json={"config": cfg, "project_id": p["id"]}).json()["ok"]
    job_response = client.post(
        "/api/jobs", json={"name": "export proof", "project_id": p["id"], "config": cfg}
    )
    assert job_response.status_code == 201, job_response.text
    job = job_response.json()
    row = c.db.fetchone("SELECT * FROM jobs WHERE id=?", (job["id"],))
    snapshot = json.loads(row["config_json"])
    assert snapshot["checkpoint"]["name"] == binding["name"]
    assert row["run_dir"] == binding["directory_template"].replace("{job_id}", job["id"])
    assert c.config_path(p["id"]).read_bytes() == original
    assert cfg["checkpoint"]["name"] == ""
    assert Trainer(TrainConfig.model_validate(snapshot), device="cpu").run() == "finished"
    assert (Path(row["run_dir"]) / f"{binding['name']}-final.safetensors").is_file()
    frozen = row["config_json"]
    client.patch(f"/api/projects/{p['id']}", json={"name": "改名"})
    assert c.db.fetchone("SELECT config_json FROM jobs WHERE id=?", (job["id"],))["config_json"] == frozen
    assert (
        client.post(f"/api/projects/{p['id']}/output-binding", json={"config": cfg}).json()["name"]
        == "改名_v1"
    )


def test_custom_name_and_output_roots_are_preserved_and_preview_is_version_scoped(api, tmp_path):
    client, c, p = api
    cfg = client.get(f"/api/projects/{p['id']}/config").json()
    custom = str(tmp_path / "custom")
    cfg["checkpoint"].update(name="my_existing_recipe", output_dir=custom)
    endpoint = f"/api/projects/{p['id']}/output-binding"
    result = client.post(endpoint, json={"config": cfg}).json()
    assert result["name"] == "my_existing_recipe" and not result["automatic_name"]
    assert not result["inherits_output_dir"]
    assert result["directory_template"] == str(Path(custom) / p["id"] / "v1" / "{job_id}")
    job = client.post("/api/jobs", json={"name": "custom", "project_id": p["id"], "config": cfg})
    assert job.status_code == 201, job.text
    old = c.db.fetchone("SELECT * FROM jobs WHERE id=?", (job.json()["id"],))
    replay = json.loads(old["config_json"])
    assert (
        client.post(endpoint, json={"config": replay}).json()["directory_template"]
        == result["directory_template"]
    )
    c.save_settings({"paths": {"output_mode": "custom", "output_dir": str(tmp_path / "global")}})
    cfg["checkpoint"]["output_dir"] = "outputs/run"
    assert client.post(endpoint, json={"config": cfg}).json()["directory_template"] == str(
        tmp_path / "global" / p["id"] / "v1" / "{job_id}"
    )
    assert (
        c.db.fetchone("SELECT run_dir FROM jobs WHERE id=?", (job.json()["id"],))["run_dir"] == old["run_dir"]
    )


@pytest.mark.parametrize("checkpoint", [None, {"output_dir": 3}, {"name": "bad/name"}])
def test_bad_binding_config_reports_field_error_instead_of_500(api, checkpoint):
    client, _, p = api
    response = client.post(
        f"/api/projects/{p['id']}/output-binding", json={"config": {"checkpoint": checkpoint}}
    )
    assert response.status_code == 422, response.text
