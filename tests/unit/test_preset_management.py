"""Standalone preset writes must not overwrite another preset or corrupt an existing file."""

import pytest
from fastapi.testclient import TestClient

from ypuddin.server import create_app


@pytest.fixture
def api(tmp_path):
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "missing")
    client = TestClient(app)
    yield client, app.state.ctx
    client.close()
    app.state.ctx.db.close()


def test_create_conflicts_and_builtin_readonly(api):
    client, context = api
    original = {"name": "人物参数", "description": "first", "config": {"loop": {"epochs": 7}}}
    assert client.post("/api/presets", json=original).status_code == 200
    assert client.post("/api/presets", json={**original, "description": "lost"}).status_code == 409
    assert client.get("/api/presets/人物参数").json()["description"] == "first"
    assert client.post("/api/presets", json={**original, "name": "anima-lokr-default"}).status_code == 403
    assert not (context.data_root / "presets/anima-lokr-default.json").exists()
    assert client.post("/api/presets", json={**original, "name": "CaseTest"}).status_code == 200
    assert client.post("/api/presets", json={**original, "name": "casetest"}).status_code == 409


def test_invalid_update_preserves_saved_config_and_valid_update_is_explicit(api):
    client, context = api
    original = {"name": "recipe", "config": {"loop": {"epochs": 2}}}
    assert client.post("/api/presets", json=original).status_code == 200
    path = context.data_root / "presets/recipe.json"
    before = path.read_bytes()
    invalid = client.put("/api/presets/recipe", json={"name": "ignored", "config": {"loop": {"epochs": -2}}})
    assert invalid.status_code == 400
    assert invalid.json()["error"]["details"]["errors"][0]["loc"] == "loop.epochs"
    assert path.read_bytes() == before
    malformed = client.put("/api/presets/recipe", json={"name": "recipe", "config": {"model": {"family": []}}})
    assert malformed.status_code == 400
    assert malformed.json()["error"]["details"]["errors"][0]["loc"] == "model.family"
    assert path.read_bytes() == before
    updated = client.put("/api/presets/recipe", json={"name": "cannot-rename", "description": "updated", "config": {"loop": {"epochs": 3}}})
    assert updated.status_code == 200
    assert updated.json()["name"] == "recipe"
    assert updated.json()["config"] == {"loop": {"epochs": 3}}
    assert not list(path.parent.glob("*.tmp"))
    assert client.delete("/api/presets/recipe").status_code == 200
    assert client.get("/api/presets/recipe").status_code == 404
    assert client.put("/api/presets/missing", json=original).status_code == 404


@pytest.mark.parametrize("name", ["../outside", "CON", "a/b", "a\\b", "-_-", "a" * 129])
def test_names_cannot_escape_managed_preset_directory(api, name):
    client, context = api
    response = client.post("/api/presets", json={"name": name, "config": {}})
    assert response.status_code == 400
    assert not list((context.data_root / "presets").glob("*.json"))


def test_defaults_follow_family_without_cuda_recommendations(api):
    client, _ = api
    krea = client.get("/api/config/defaults?family=krea2")
    assert krea.status_code == 200
    config = krea.json()
    assert config["model"]["family"] == "krea2"
    assert config["dataset"]["text_encoding"] == "cached"
    assert config["memory"]["base_precision"] == "auto"
    assert config["sampling"]["steps"] == 28
    assert client.get("/api/config/defaults?family=flux").status_code == 422
