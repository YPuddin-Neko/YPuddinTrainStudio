import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from ypuddin.config import ModelConfig
from ypuddin.models import get_family
from ypuddin.server import create_app
from ypuddin.server.family_config import initial_family_config


@pytest.fixture
def api(tmp_path):
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "missing")
    client = TestClient(app)
    try:
        yield client, app.state.ctx
    finally:
        app.state.regularization.close()
        app.state.dataset_pipeline.close()
        app.state.ctx.versions.close()
        client.close()
        app.state.ctx.db.close()


def bundle(path):
    save_file(
        {
            "model.diffusion_model.input_blocks.0.0.weight": torch.zeros(2, 4, 1, 1),
            "model.diffusion_model.label_emb.0.0.weight": torch.zeros(2, 2816),
            "first_stage_model.encoder.weight": torch.zeros(1),
            "first_stage_model.decoder.weight": torch.zeros(1),
            "first_stage_model.quant_conv.weight": torch.zeros(8, 8, 1, 1),
            "first_stage_model.post_quant_conv.weight": torch.zeros(4, 4, 1, 1),
        },
        str(path),
    )
    return path


def test_sdxl_inspection_scan_and_local_dit_default_registration(api, tmp_path):
    client, context = api
    folder = tmp_path / "models"
    folder.mkdir()
    path = bundle(folder / "misleading_vae.safetensors")
    detected = client.post("/api/models/inspect", json={"path": str(path)})
    assert detected.status_code == 200, detected.text
    assert detected.json()["family"] == "sdxl" and detected.json()["kind"] == "dit"
    assert client.get("/api/models").json() == []
    scanned = client.post("/api/models/scan", json={"path": str(folder), "family": "anima"})
    assert scanned.status_code == 200, scanned.text
    assert [(row["family"], row["kind"]) for row in scanned.json()] == [("sdxl", "dit")]
    registered = client.post(
        "/api/models", json={"family": "sdxl", "kind": "dit", "path": str(path), "is_default": True}
    )
    assert registered.status_code == 200, registered.text
    assert registered.json()["id"] == scanned.json()[0]["id"]
    recipe = initial_family_config(context, "sdxl")
    assert recipe["model"]["dit_path"] == str(path)
    assert not recipe["model"].get("text_encoder_path")
    assert not recipe["model"].get("text_encoder_2_path")
    assert not recipe["model"].get("vae_path")
    info = client.get("/api/families/sdxl")
    assert info.status_code == 200, info.text
    required = [weight["field"] for weight in info.json()["weights"] if weight["required"]]
    assert required == ["dit_path"]  # Bundle components need no separate registry rows.


@pytest.mark.parametrize(
    "kind,field", [("dit", "dit_path"), ("vae", "vae_path"), ("text_encoder_2", "text_encoder_2_path")]
)
def test_sdxl_directory_roles_and_defaults_are_retained(api, tmp_path, kind, field):
    client, context = api
    path = tmp_path / kind
    path.mkdir()
    # Registration records a reviewed path; full directory inspection has its own tests.
    response = client.post(
        "/api/models", json={"family": "sdxl", "kind": kind, "path": str(path), "is_default": True}
    )
    assert response.status_code == 200, response.text
    assert response.json()["path"] == str(path) and response.json()["is_default"] is True
    assert initial_family_config(context, "sdxl")["model"][field] == str(path)
    path.rmdir()
    assert not initial_family_config(context, "sdxl")["model"].get(field)


def test_independent_sdxl_unet_without_encoders_has_explicit_validation_error(tmp_path):
    path = tmp_path / "unet.safetensors"
    save_file(
        {"conv_in.weight": torch.zeros(2, 4, 1, 1), "add_embedding.linear_1.weight": torch.zeros(2, 2816)},
        str(path),
    )
    errors = get_family("sdxl").validate_config(ModelConfig(family="sdxl", dit_path=str(path)))
    assert any(
        "text_encoder" in error and ("missing" in error or "no " in error or "contain" in error)
        for error in errors
    ), errors
