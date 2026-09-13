"""Standalone preset writes must not overwrite another preset or corrupt an existing file."""

import json

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


def test_create_conflicts_preserve_saved_custom_presets(api):
    client, _ = api
    original = {"name": "人物参数", "description": "first", "config": {"loop": {"epochs": 7}}}
    assert client.post("/api/presets", json=original).status_code == 200
    assert client.post("/api/presets", json={**original, "description": "lost"}).status_code == 409
    assert client.get("/api/presets/人物参数").json()["description"] == "first"
    assert client.post("/api/presets", json={**original, "name": "CaseTest"}).status_code == 200
    assert client.post("/api/presets", json={**original, "name": "casetest"}).status_code == 409


@pytest.mark.parametrize(
    "name", ["anima-lokr-default", "anima-lora-16", "krea2-lokr-default", "krea2-lora-32", "toy-smoke"]
)
def test_former_builtin_names_are_only_user_owned_presets(api, name):
    client, context = api
    assert client.get("/api/presets").json() == []
    for response in (
        client.get(f"/api/presets/{name}"),
        client.post(f"/api/presets/{name}/resolve", json={"config": {}}),
        client.put(f"/api/presets/{name}", json={"name": name, "config": {}}),
        client.delete(f"/api/presets/{name}"),
    ):
        assert response.status_code == 404, response.text
        assert response.json()["error"]["code"] == "preset.not_found"
    body = {"name": name, "description": "My own settings", "config": {"loop": {"epochs": 7}}}
    created = client.post("/api/presets", json=body)
    assert created.status_code == 200, created.text
    assert created.json()["builtin"] is False
    assert (context.data_root / "presets" / f"{name}.json").is_file()
    assert client.get("/api/presets").json() == [created.json()]
    assert client.get(f"/api/presets/{name}").json() == created.json()
    resolved = client.post(f"/api/presets/{name}/resolve", json={"config": {"loop": {"seed": 42}}})
    assert resolved.status_code == 200 and resolved.json()["ok"] is True
    assert resolved.json()["config"]["loop"]["epochs"] == 7
    assert resolved.json()["config"]["loop"]["seed"] == 42
    updated = client.put(f"/api/presets/{name}", json={**body, "config": {"loop": {"epochs": 3}}})
    assert updated.status_code == 200 and updated.json()["config"]["loop"]["epochs"] == 3
    assert client.delete(f"/api/presets/{name}").status_code == 200
    assert client.get("/api/presets").json() == []
    assert client.get(f"/api/presets/{name}").status_code == 404


def test_existing_custom_files_survive_without_being_shadowed_by_former_builtins(api):
    client, context = api
    directory = context.data_root / "presets"
    directory.mkdir(exist_ok=True)
    before = {}
    for name in ("anima-lokr-default", "自己的参数"):
        data = {"description": "Existing user file", "config": {"adapter": {"rank": 11}}}
        path = directory / f"{name}.json"
        path.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
        before[path] = path.read_bytes()
    listed = client.get("/api/presets").json()
    assert {row["name"] for row in listed} == {path.stem for path in before}
    for row in listed:
        assert row["builtin"] is False
        assert row["config"]["adapter"]["rank"] == 11
        assert client.get(f"/api/presets/{row['name']}").json()["config"] == row["config"]
    assert all(path.read_bytes() == content for path, content in before.items())


def test_removing_training_presets_keeps_family_targets_and_defaults(api):
    client, _ = api
    assert client.get("/api/presets").json() == []
    families = client.get("/api/families").json()
    for name in ("anima", "krea2", "toy"):
        family = next(item for item in families if item["name"] == name)
        target_names = {preset["name"] for preset in family["presets"]}
        assert target_names and family["default_preset"] in target_names
        defaults = client.get(f"/api/config/defaults?family={name}")
        assert defaults.status_code == 200
        assert defaults.json()["model"]["family"] == name
        assert defaults.json()["adapter"]["preset"] in target_names


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
    for family in ("flux2",):
        response = client.get(f"/api/config/defaults?family={family}")
        assert response.status_code == 200
        config = response.json()
        assert config["model"]["family"] == family
        assert config["dataset"]["text_encoding"] == "cached"
        assert config["sampling"]["steps"] is None  # resolved from actual dev/schnell/Klein weights
        assert config["sampling"]["cfg"] is None
    assert client.get("/api/config/defaults?family=flux").status_code == 422
    assert client.get("/api/config/defaults?family=flux3").status_code == 422
