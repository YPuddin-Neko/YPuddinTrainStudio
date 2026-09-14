"""Converted adapter formats remain valid inputs; full models never are."""

import io
import json
import zipfile

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import load_file, save_file

from ypuddin.server import create_app
from ypuddin.server.db import now


@pytest.mark.parametrize("kind", ["weights", "comfyui", "kohya"])
def test_raw_and_converted_adapter_kinds_can_be_converted_again(tmp_path, kind):
    app = create_app(tmp_path / "studio")
    source = tmp_path / f"{kind}.safetensors"
    tensors = {
        "diffusion_model.block.linear.lora_A.weight": torch.arange(8.0).reshape(2, 4),
        "diffusion_model.block.linear.lora_B.weight": torch.arange(8.0).reshape(4, 2),
    }
    save_file(tensors, source, metadata={"ypuddin.family": "anima"})
    app.state.ctx.db.insert(
        "artifacts",
        {
            "id": "source",
            "name": source.name,
            "path": str(source),
            "kind": kind,
            "size": source.stat().st_size,
            "created_at": now(),
        },
    )
    with TestClient(app) as client:
        result = client.post("/api/artifacts/source/convert", json={"format": "kohya"})
        assert result.status_code == 200, result.text
        converted = result.json()
        assert converted["kind"] == "kohya"
        loaded = load_file(converted["path"])
        torch.testing.assert_close(
            loaded["lora_unet_block_linear.lora_down.weight"],
            tensors["diffusion_model.block.linear.lora_A.weight"],
        )
        # A second, different conversion starts from the actual newly registered artifact.
        repeated = client.post(f"/api/artifacts/{converted['id']}/convert", json={"format": "comfyui"})
        assert repeated.status_code == 200, repeated.text
        assert repeated.json()["kind"] == "comfyui"
        torch.testing.assert_close(
            load_file(repeated.json()["path"])["lora_unet_block_linear.lora_down.weight"],
            tensors["diffusion_model.block.linear.lora_A.weight"],
        )


def test_model_export_download_name_and_kind_are_not_adapter_format(tmp_path):
    app = create_app(tmp_path / "studio")
    source = tmp_path / "character-final.model"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "format": "ypuddin-full-model-v1",
                "family": "anima",
                "components": {"backbone": "model.safetensors"},
            }
        )
    )
    (source / "model.safetensors").write_bytes(b"model bytes")
    app.state.ctx.db.insert(
        "artifacts",
        {
            "id": "model",
            "name": source.name,
            "path": str(source),
            "kind": "model",
            "size": 11,
            "created_at": now(),
        },
    )
    with TestClient(app) as client:
        response = client.get("/api/artifacts/model/download")
        assert response.status_code == 200
        assert 'filename="character-final.model.zip"' in response.headers["content-disposition"]
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            assert archive.read("character-final.model/model.safetensors") == b"model bytes"
        result = client.post("/api/artifacts/model/convert", json={"format": "comfyui"})
        assert result.status_code == 422
        assert result.json()["error"]["code"] == "artifact.kind"
        assert (source / "model.safetensors").read_bytes() == b"model bytes"
