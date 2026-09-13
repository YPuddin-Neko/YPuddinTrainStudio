import json
import math

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from ypuddin.server import create_app
from ypuddin.server.model_inspection import inspect_model


def sparse_headers(path, shapes):
    """Real safetensors headers with sparse zero payloads, without allocating giant CLIP tensors."""
    header, offset = {}, 0
    for name, shape in shapes.items():
        end = offset + math.prod(shape) * 2
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, end]}
        offset = end
    payload = json.dumps(header).encode()
    payload += b" " * (-len(payload) % 8)
    with path.open("wb") as stream:
        stream.write(len(payload).to_bytes(8, "little"))
        stream.write(payload)
        stream.truncate(8 + len(payload) + offset)
    return path


SDXL_UNET = {"conv_in.weight": [2, 4, 3, 3], "add_embedding.linear_1.weight": [2, 2816]}
SDXL_VAE = {
    "encoder.conv_in.weight": [2, 3, 3, 3],
    "decoder.conv_out.weight": [3, 2, 3, 3],
    "quant_conv.weight": [8, 8, 1, 1],
    "post_quant_conv.weight": [4, 4, 1, 1],
}


def hf_sdxl_directory(root, *, sharded=False):
    components = {
        "unet": (
            "diffusers",
            "UNet2DConditionModel",
            SDXL_UNET,
            {
                "in_channels": 4,
                "out_channels": 4,
                "cross_attention_dim": 2048,
                "addition_embed_type": "text_time",
                "projection_class_embeddings_input_dim": 2816,
            },
        ),
        "text_encoder": (
            "transformers",
            "CLIPTextModel",
            {"text_model.embeddings.token_embedding.weight": [49408, 768]},
            {"hidden_size": 768, "vocab_size": 49408},
        ),
        "text_encoder_2": (
            "transformers",
            "CLIPTextModelWithProjection",
            {
                "text_model.embeddings.token_embedding.weight": [49408, 1280],
                "text_projection.weight": [1280, 1280],
            },
            {"hidden_size": 1280, "vocab_size": 49408, "projection_dim": 1280},
        ),
        "vae": ("diffusers", "AutoencoderKL", SDXL_VAE, {"latent_channels": 4}),
    }
    root.mkdir()
    index = {"_class_name": "StableDiffusionXLPipeline"}
    for name, (library, model_class, shapes, config) in components.items():
        index[name] = [library, model_class]
        folder = root / name
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps(config))
        if sharded and name == "unet":
            mapping = {}
            for i, (key, shape) in enumerate(shapes.items()):
                filename = f"diffusion_pytorch_model-{i + 1:05d}-of-00002.safetensors"
                sparse_headers(folder / filename, {key: shape})
                mapping[key] = filename
            (folder / "diffusion_pytorch_model.safetensors.index.json").write_text(
                json.dumps({"weight_map": mapping})
            )
        else:
            filename = (
                "model.safetensors"
                if name.startswith("text_encoder")
                else "diffusion_pytorch_model.safetensors"
            )
            sparse_headers(folder / filename, shapes)
    for name in ("tokenizer", "tokenizer_2"):
        index[name] = ["transformers", "CLIPTokenizer"]
        folder = root / name
        folder.mkdir()
        (folder / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "CLIPTokenizer"}))
        (folder / "vocab.json").write_text(json.dumps({"a": 0}))
        (folder / "merges.txt").write_text("#version: 0.2\na b\n")
    (root / "model_index.json").write_text(json.dumps(index))
    return root


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


def test_sdxl_bundle_takes_priority_over_embedded_vae_and_both_clips(tmp_path, monkeypatch):
    from safetensors import safe_open

    from ypuddin.server.model_downloads import check_component

    shapes = {
        "model.diffusion_model.input_blocks.0.0.weight": [2, 4, 3, 3],
        "model.diffusion_model.label_emb.0.0.weight": [2, 2816],
        **{f"first_stage_model.{key}": value for key, value in SDXL_VAE.items()},
        "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight": [49408, 768],
        "conditioner.embedders.1.model.token_embedding.weight": [49408, 1280],
        "conditioner.embedders.1.model.text_projection": [1280, 1280],
    }
    path = sparse_headers(tmp_path / "misleading_vae.safetensors", shapes)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("pickle must not be loaded"))
    import safetensors.torch

    monkeypatch.setattr(
        safetensors.torch, "load_file", lambda *a, **k: pytest.fail("no tensor payload loads")
    )
    assert inspect_model(path)["family"] == "sdxl"
    assert inspect_model(path)["kind"] == "dit"
    with safe_open(str(path), framework="pt", device="cpu") as weights:
        check_component(weights, "sdxl", "dit")
        for family, kind in (("sdxl", "vae"), ("anima", "dit"), ("krea2", "dit")):
            with pytest.raises(ValueError, match="looks like sdxl dit"):
                check_component(weights, family, kind)


@pytest.mark.parametrize(
    "role,shapes",
    [
        ("text_encoder", {"text_model.embeddings.token_embedding.weight": [49408, 768]}),
        ("text_encoder", {"embeddings.token_embedding.weight": [49408, 768]}),
        (
            "text_encoder",
            {
                "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight": [
                    49408,
                    768,
                ]
            },
        ),
        (
            "text_encoder_2",
            {
                "text_model.embeddings.token_embedding.weight": [49408, 1280],
                "text_projection.weight": [1280, 1280],
            },
        ),
        ("text_encoder_2", {"token_embedding.weight": [49408, 1280], "text_projection": [1280, 1280]}),
        (
            "text_encoder_2",
            {
                "conditioner.embedders.1.model.token_embedding.weight": [49408, 1280],
                "conditioner.embedders.1.model.text_projection": [1280, 1280],
            },
        ),
    ],
)
def test_clip_roles_match_download_checker_without_guessing_family(tmp_path, role, shapes):
    from safetensors import safe_open

    from ypuddin.server.model_downloads import check_component

    path = sparse_headers(tmp_path / "clip.safetensors", shapes)
    result = inspect_model(path)
    assert result["kind"] == role
    assert result["family_candidates"] == (["sdxl", "flux"] if role == "text_encoder" else ["sdxl"])
    assert result["family"] is None  # CLIP-L/G are shared beyond SDXL.
    with safe_open(str(path), framework="pt", device="cpu") as weights:
        check_component(weights, "sdxl", role)
        if role == "text_encoder":
            check_component(weights, "flux", role)
        else:
            with pytest.raises(ValueError, match="looks like shared"):
                check_component(weights, "flux", role)
        with pytest.raises(ValueError, match="looks like shared"):
            check_component(weights, "sdxl", "text_encoder" if role == "text_encoder_2" else "text_encoder_2")


def test_wrong_unet_and_clip_projection_geometry_are_not_sdxl(tmp_path):
    from ypuddin.server.model_inspection import sdxl_component

    assert sdxl_component({**SDXL_UNET, "add_embedding.linear_1.weight": [2, 2560]}) is None
    assert sdxl_component({**SDXL_UNET, "conv_in.weight": [2, 9, 3, 3]}) is None
    path = sparse_headers(
        tmp_path / "clip_g.safetensors",
        {
            "text_model.embeddings.token_embedding.weight": [49408, 1280],
            "text_projection.weight": [768, 1280],
        },
    )
    assert inspect_model(path)["kind"] is None


@pytest.mark.parametrize("sharded", [False, True])
def test_complete_hf_sdxl_directory_and_components_preserve_directory_path(tmp_path, sharded):
    root = hf_sdxl_directory(tmp_path / "sdxl", sharded=sharded)
    result = inspect_model(root)
    assert result["family"] == "sdxl" and result["kind"] == "dit"
    assert result["path"] == str(root) and result["confidence"] == "high"
    assert result["files_inspected"] == (5 if sharded else 4)
    assert result["dtype"] == "bf16"
    for role in ("unet", "vae"):
        assert inspect_model(root / role)["path"] == str(root / role)


@pytest.mark.parametrize(
    "missing",
    [
        "unet/config.json",
        "text_encoder/model.safetensors",
        "text_encoder_2/model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "tokenizer/merges.txt",
        "tokenizer_2/vocab.json",
    ],
)
def test_hf_sdxl_directory_rejects_missing_local_components(tmp_path, missing):
    root = hf_sdxl_directory(tmp_path / "sdxl")
    (root / missing).unlink()
    with pytest.raises(ValueError, match="SDXL"):
        inspect_model(root)


def test_hf_sdxl_rejects_pickle_only_component_and_bad_config(tmp_path, monkeypatch):
    root = hf_sdxl_directory(tmp_path / "sdxl")
    model = root / "text_encoder/model.safetensors"
    model.rename(model.with_suffix(".bin"))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("pickle must not be loaded"))
    with pytest.raises(ValueError, match="safetensors"):
        inspect_model(root)
    (root / "unet/config.json").write_text(json.dumps({"in_channels": 9}))
    with pytest.raises(ValueError, match="incompatible"):
        inspect_model(root)


def test_diffusers_shard_index_validates_actual_tensor_locations_and_missing_shards(tmp_path):
    root = hf_sdxl_directory(tmp_path / "sdxl", sharded=True)
    index = root / "unet/diffusion_pytorch_model.safetensors.index.json"
    payload = json.loads(index.read_text())
    keys = list(payload["weight_map"])
    payload["weight_map"][keys[0]], payload["weight_map"][keys[1]] = (
        payload["weight_map"][keys[1]],
        payload["weight_map"][keys[0]],
    )
    index.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="tensor locations"):
        inspect_model(root)
    (index.parent / payload["weight_map"][keys[0]]).unlink()
    with pytest.raises(ValueError, match="missing or outside"):
        inspect_model(root)


@pytest.mark.parametrize("asset", ["model_index.json", "unet/config.json", "tokenizer_2/vocab.json"])
def test_hf_metadata_symlinks_cannot_escape_selected_directory(tmp_path, asset):
    root = hf_sdxl_directory(tmp_path / "sdxl")
    outside = tmp_path / "outside.json"
    (root / asset).replace(outside)
    try:
        (root / asset).symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="outside"):
        inspect_model(root)


def test_hf_tokenizer_json_format_and_allowed_component_boundary(tmp_path):
    root = hf_sdxl_directory(tmp_path / "sdxl")
    for name in ("tokenizer", "tokenizer_2"):
        (root / name / "vocab.json").unlink()
        (root / name / "merges.txt").unlink()
        (root / name / "tokenizer.json").write_text(json.dumps({"model": {"type": "BPE", "vocab": {"a": 0}}}))
    assert inspect_model(root)["family"] == "sdxl"
    forbidden = root / "text_encoder_2"
    with pytest.raises(ValueError, match="outside"):
        inspect_model(root, allowed=lambda path: not path.is_relative_to(forbidden))


@pytest.mark.parametrize(
    "filename", ["../escape.safetensors", "C:\\weights\\outside.safetensors", "weights.bin"]
)
def test_diffusers_index_rejects_unsafe_or_pickle_shards(tmp_path, filename):
    (tmp_path / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"conv_in.weight": filename}})
    )
    with pytest.raises(ValueError, match="missing or outside"):
        inspect_model(tmp_path)
