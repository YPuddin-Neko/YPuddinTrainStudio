import json

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from ypuddin.server import create_app
from ypuddin.server.model_inspection import inspect_model


def weights(path, family="anima", dtype=torch.bfloat16):
    keys = (
        ["x_embedder.proj.1.weight", "llm_adapter.fc.weight", "blocks.0.self_attn.weight"]
        if family == "anima"
        else ["first.weight", "txtfusion.projector.weight", "blocks.0.attn.wq.weight"]
    )
    save_file({key: torch.zeros(2, 2, dtype=dtype) for key in keys}, str(path))
    return path


@pytest.mark.parametrize("family", ["anima", "krea2"])
def test_real_headers_identify_structure_and_dtype_without_loading_tensors(tmp_path, monkeypatch, family):
    path = weights(tmp_path / "misleading_fp32_vae.safetensors", family)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("pickle must not be loaded"))
    import safetensors.torch

    monkeypatch.setattr(
        safetensors.torch, "load_file", lambda *a, **k: pytest.fail("tensor payload must not be loaded")
    )
    result = inspect_model(path)
    assert result["family"] == family and result["kind"] == "dit"
    assert result["dtype"] == "bf16" and result["confidence"] == "high"


def test_shared_vae_has_candidates_and_unknown_matrix_does_not_use_filename(tmp_path):
    path = tmp_path / "vae.safetensors"
    save_file(
        {
            "encoder.weight": torch.zeros(1),
            "decoder.weight": torch.zeros(1),
            "quant_conv.weight": torch.zeros(32, 32, 1, 1, 1),
            "post_quant_conv.weight": torch.zeros(16, 16, 1, 1, 1),
        },
        str(path),
    )
    result = inspect_model(path)
    assert result["kind"] == "vae" and result["family"] is None
    assert result["family_candidates"] == ["anima", "krea2"]
    unknown = tmp_path / "anima_bf16_dit.safetensors"
    save_file({"mystery.weight": torch.zeros(2, 2)}, str(unknown))
    result = inspect_model(unknown)
    assert result["family"] is result["kind"] is None
    assert result["dtype"] == "fp32"


def test_hf_shards_use_config_and_actual_precision_instead_of_config_dtype(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_vl", "torch_dtype": "bfloat16"}))
    save_file(
        {"model.language_model.embed_tokens.weight": torch.zeros(8, 16, dtype=torch.float16)},
        str(tmp_path / "a.safetensors"),
    )
    save_file(
        {"model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(16, 16, dtype=torch.float16)},
        str(tmp_path / "b.safetensors"),
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.language_model.embed_tokens.weight": "a.safetensors",
                    "model.language_model.layers.0.self_attn.q_proj.weight": "b.safetensors",
                }
            }
        )
    )
    result = inspect_model(tmp_path)
    assert result["family"] == "krea2" and result["kind"] == "text_encoder"
    assert result["dtype"] == "fp16" and result["files_inspected"] == 2
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"bad": "../escape.safetensors"}})
    )
    with pytest.raises(ValueError, match="missing or outside"):
        inspect_model(tmp_path)


def test_fp8_scaled_and_mixed_matrix_precision_are_explicit(tmp_path):
    path = tmp_path / "scaled.safetensors"
    save_file(
        {
            "first.weight": torch.zeros(2, 2, dtype=torch.float8_e4m3fn),
            "first.scale_weight": torch.ones(1),
            "txtfusion.projector.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
            "blocks.0.attn.wq.weight": torch.zeros(2, 2, dtype=torch.float8_e4m3fn),
        },
        str(path),
    )
    assert inspect_model(path)["dtype"] == "fp8"
    save_file({"a.weight": torch.zeros(2, 2), "b.weight": torch.zeros(2, 2, dtype=torch.bfloat16)}, str(path))
    assert inspect_model(path)["dtype"] == "mixed"


def test_rejects_pickle_invalid_headers_and_symlink_escape(tmp_path):
    path = tmp_path / "unsafe.bin"
    path.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="pickle"):
        inspect_model(path)
    path = tmp_path / "oversized.safetensors"
    path.write_bytes((40 * 1024 * 1024).to_bytes(8, "little"))
    with pytest.raises(ValueError, match="header"):
        inspect_model(path)
    folder = tmp_path / "folder"
    folder.mkdir()
    real = weights(tmp_path / "outside.safetensors")
    try:
        (folder / "escape.safetensors").symlink_to(real)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="outside"):
        inspect_model(folder)


def test_inspect_api_and_scan_do_not_register_unknown_or_filename_guesses(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    known = weights(model_dir / "random_name.safetensors", "krea2", torch.float16)
    save_file({"unknown.weight": torch.zeros(2, 2)}, str(model_dir / "anima-base-bf16.safetensors"))
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "missing")
    client = TestClient(app)
    try:
        response = client.post("/api/models/inspect", json={"path": str(known)})
        assert response.status_code == 200, response.text
        assert response.json()["family"] == "krea2"
        assert client.get("/api/models").json() == []
        scanned = client.post("/api/models/scan", json={"path": str(model_dir), "family": "anima"})
        assert scanned.status_code == 200, scanned.text
        assert [(x["family"], x["dtype"]) for x in scanned.json()] == [("krea2", "fp16")]
        assert client.post("/api/models/inspect", json={"path": str(tmp_path / "missing")}).status_code == 422
    finally:
        app.state.regularization.close()
        app.state.dataset_pipeline.close()
        app.state.ctx.versions.close()
        client.close()
        app.state.ctx.db.close()
