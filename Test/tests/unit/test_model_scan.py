"""Scanning keeps complete pipelines and sharded components as single assets."""

import pytest

from Test.tests.unit.test_model_inspection import hf_sdxl_directory
from Test.tests.unit.test_sdxl_model_registration import api as _api

api = _api


@pytest.mark.parametrize("sharded", [False, True])
def test_scan_registers_pipeline_root_once_and_never_internal_shards(api, tmp_path, sharded):
    client, _ = api
    root = hf_sdxl_directory(tmp_path / "sdxl", sharded=sharded)
    response = client.post("/api/models/scan", json={"path": str(tmp_path), "family": "sdxl"})
    assert response.status_code == 200, response.text
    assert [(row["path"], row["kind"]) for row in response.json()] == [(str(root), "dit")]
    assert client.post("/api/models/scan", json={"path": str(root)}).json() == []
    assert len(client.get("/api/models").json()) == 1


def test_scan_incomplete_pipeline_does_not_publish_its_partial_weights(api, tmp_path):
    client, _ = api
    root = hf_sdxl_directory(tmp_path / "sdxl", sharded=True)
    (root / "tokenizer_2" / "vocab.json").unlink()
    response = client.post("/api/models/scan", json={"path": str(root), "family": "sdxl"})
    assert response.status_code == 200, response.text
    assert response.json() == []
    assert client.get("/api/models").json() == []


def test_scan_standalone_sharded_unet_preserves_component_directory(api, tmp_path):
    client, _ = api
    root = hf_sdxl_directory(tmp_path / "sdxl", sharded=True) / "unet"
    response = client.post("/api/models/scan", json={"path": str(root), "family": "sdxl"})
    assert response.status_code == 200, response.text
    assert [(row["path"], row["kind"]) for row in response.json()] == [(str(root), "dit")]


def test_plain_config_does_not_hide_independent_checkpoints(api, tmp_path):
    from Test.tests.unit.test_model_inspection import weights

    client, _ = api
    models = tmp_path / "collection"
    models.mkdir()
    (models / "config.json").write_text('{"note":"collection metadata"}')
    paths = [weights(models / name) for name in ("first.safetensors", "second.safetensors")]
    for path in paths:
        assert client.post("/api/models/inspect", json={"path": str(path)}).json()["family"] == "anima"
    response = client.post("/api/models/scan", json={"path": str(models), "family": "anima"})
    assert response.status_code == 200, response.text
    assert {row["path"] for row in response.json()} == {str(path) for path in paths}


def test_plain_config_does_not_hide_nested_independent_weights(api, tmp_path):
    from Test.tests.unit.test_model_inspection import weights

    client, _ = api
    models = tmp_path / "collection"
    models.mkdir()
    (models / "config.json").write_text('{"note":"collection metadata"}')
    direct = weights(models / "direct.safetensors")
    nested = models / "nested"
    nested.mkdir()
    child = weights(nested / "child.safetensors")
    response = client.post("/api/models/scan", json={"path": str(models)})
    assert response.status_code == 200, response.text
    assert {row["path"] for row in response.json()} == {str(direct), str(child)}


def test_invalid_shard_index_never_falls_back_to_internal_weights(api, tmp_path):
    import json

    from Test.tests.unit.test_model_inspection import weights

    client, _ = api
    models = tmp_path / "shards"
    models.mkdir()
    weights(models / "plausible_model.safetensors")
    (models / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"missing.weight": "missing.safetensors"}})
    )
    response = client.post("/api/models/scan", json={"path": str(models)})
    assert response.status_code == 200, response.text
    assert response.json() == []
    assert client.get("/api/models").json() == []


def test_recognized_unsharded_component_does_not_register_its_weight_file(api, tmp_path):
    client, _ = api
    component = hf_sdxl_directory(tmp_path / "sdxl") / "unet"
    response = client.post("/api/models/scan", json={"path": str(component), "family": "sdxl"})
    assert response.status_code == 200, response.text
    assert [(row["path"], row["kind"]) for row in response.json()] == [(str(component), "dit")]
    assert client.post("/api/models/scan", json={"path": str(component), "family": "sdxl"}).json() == []
    assert len(client.get("/api/models").json()) == 1
