"""Sparse, valid safetensors metadata exercises inspection without loading model tensors."""

import json
import math

import pytest
from fastapi.testclient import TestClient
from safetensors import safe_open

from ypuddin.server import create_app
from ypuddin.server.model_downloads import check_component
from ypuddin.server.model_inspection import inspect_model


def sparse_headers(path, shapes):
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


def transformer_shapes(family="flux", width=3072, native=True, guidance=True):
    flux1 = family == "flux"
    shapes = {
        "x_embedder.weight" if native else "img_in.weight": [width, 64 if flux1 else 128],
        "context_embedder.weight" if native else "txt_in.weight": [
            width,
            4096 if flux1 else {3072: 7680, 4096: 12288, 6144: 15360}[width],
        ],
        "transformer_blocks.0.attn.to_q.weight" if native else "double_blocks.0.img_attn.qkv.weight": [2, 2],
        "single_transformer_blocks.0.attn.to_q.weight" if native else "single_blocks.0.linear1.weight": [
            2,
            2,
        ],
    }
    if not flux1:
        shapes[
            "single_stream_modulation.linear.weight" if native else "single_stream_modulation.lin.weight"
        ] = [2, 2]
    if guidance:
        shapes[
            "time_text_embed.guidance_embedder.linear_1.weight" if native else "guidance_in.in_layer.weight"
        ] = [2, 2]
    return shapes


def vae_shapes(flux2=False):
    result = {
        "encoder.conv_out.weight": [64 if flux2 else 32, 2, 3, 3],
        "decoder.conv_in.weight": [2, 32 if flux2 else 16, 3, 3],
    }
    if flux2:
        result.update({"bn.running_mean": [128], "bn.running_var": [128]})
    return result


CLIP = {"text_model.embeddings.token_embedding.weight": [49408, 768]}
T5 = {"shared.weight": [32128, 4096], "encoder.block.0.layer.0.SelfAttention.q.weight": [2, 2]}


def decoder_shapes(hidden, vocab=151936):
    return {
        "model.embed_tokens.weight": [vocab, hidden],
        "model.layers.0.self_attn.q_proj.weight": [2, 2],
        "model.layers.0.self_attn.q_norm.weight": [2],
    }


def hf_flux_directory(root, variant="flux", sharded=False):
    root.mkdir()
    flux1, klein = variant == "flux", variant.startswith("klein")
    width = 3072 if variant in ("flux", "klein4") else 4096 if klein else 6144
    hidden = 2560 if variant == "klein4" else 4096 if klein else 5120
    index = {"_class_name": "FluxPipeline" if flux1 else "Flux2KleinPipeline" if klein else "Flux2Pipeline"}
    if klein:
        index["is_distilled"] = False
    components = {
        "transformer": (
            "diffusers",
            "FluxTransformer2DModel" if flux1 else "Flux2Transformer2DModel",
            transformer_shapes("flux" if flux1 else "flux2", width, guidance=not klein),
            {
                "in_channels": 64 if flux1 else 128,
                "joint_attention_dim": 4096 if flux1 else hidden * 3,
                "pooled_projection_dim": 768,
            },
        ),
        "text_encoder": (
            "transformers",
            "CLIPTextModel" if flux1 else "Qwen3ForCausalLM" if klein else "Mistral3ForConditionalGeneration",
            CLIP if flux1 else decoder_shapes(hidden, 151936 if klein else 131072),
            {"hidden_size": 768}
            if flux1
            else {"model_type": "qwen3", "hidden_size": hidden}
            if klein
            else {"model_type": "mistral3", "text_config": {"model_type": "mistral", "hidden_size": hidden}},
        ),
        "vae": (
            "diffusers",
            "AutoencoderKL" if flux1 else "AutoencoderKLFlux2",
            vae_shapes(not flux1),
            {"latent_channels": 16 if flux1 else 32},
        ),
    }
    if flux1:
        components["text_encoder_2"] = (
            "transformers",
            "T5EncoderModel",
            T5,
            {"model_type": "t5", "d_model": 4096},
        )
    for name, (library, cls, shapes, config) in components.items():
        folder = root / name
        folder.mkdir()
        index[name] = [library, cls]
        (folder / "config.json").write_text(json.dumps(config))
        if sharded and name == "transformer":
            mapping = {}
            entries = list(shapes.items())
            for i, items in enumerate((entries[:2], entries[2:])):
                filename = f"diffusion_pytorch_model-{i + 1:05d}-of-00002.safetensors"
                sparse_headers(folder / filename, dict(items))
                mapping.update({key: filename for key, _ in items})
            (folder / "diffusion_pytorch_model.safetensors.index.json").write_text(
                json.dumps({"weight_map": mapping})
            )
        else:
            sparse_headers(
                folder
                / (
                    "model.safetensors"
                    if name.startswith("text_encoder")
                    else "diffusion_pytorch_model.safetensors"
                ),
                shapes,
            )
    for name, cls in (
        {"tokenizer": "CLIPTokenizer", "tokenizer_2": "T5TokenizerFast"}
        if flux1
        else {"tokenizer": "Qwen2TokenizerFast" if klein else "PixtralProcessor"}
    ).items():
        folder = root / name
        folder.mkdir()
        index[name] = ["transformers", cls]
        config = {"tokenizer_class": cls, "_name_or_path": "/upstream/provenance/not/a/runtime/reference"}
        if not flux1:
            config["chat_template"] = "{{ messages }}"
        (folder / "tokenizer_config.json").write_text(json.dumps(config))
        (folder / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"a": 0}}}))
    (root / "model_index.json").write_text(json.dumps(index))
    return root


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("family,width", [("flux", 3072), ("flux2", 6144), ("flux2", 3072), ("flux2", 4096)])
def test_original_and_diffusers_headers_identify_architecture_not_filename(
    tmp_path, monkeypatch, native, family, width
):
    import safetensors.torch

    monkeypatch.setattr(safetensors.torch, "load_file", lambda *a, **k: pytest.fail("must not load tensors"))
    shapes = transformer_shapes(family, width, native, guidance=width == 6144 or family == "flux")
    if not native:
        shapes = {f"model.diffusion_model.{key}": value for key, value in shapes.items()}
    path = sparse_headers(tmp_path / "sdxl_schnell_klein_distilled.safetensors", shapes)
    result = inspect_model(path)
    assert result["family"] == family and result["kind"] == "dit" and result["dtype"] == "bf16"
    if family == "flux2" and width != 6144:
        assert any("base" in warning and "distilled" in warning for warning in result["warnings"])
    with safe_open(str(path), framework="pt", device="cpu") as weights:
        check_component(weights, family, "dit")
        with pytest.raises(ValueError, match="looks like"):
            check_component(weights, "flux2" if family == "flux" else "flux", "dit")


@pytest.mark.parametrize("mutation", ["fill", "text_width", "missing_stream", "missing_modulation"])
def test_wrong_flux_geometry_is_not_accepted_from_name(tmp_path, mutation):
    shapes = transformer_shapes("flux2" if mutation == "missing_modulation" else "flux")
    if mutation == "fill":
        shapes["x_embedder.weight"][1] = 384
    elif mutation == "text_width":
        shapes["context_embedder.weight"][1] = 7680
    elif mutation == "missing_stream":
        del shapes["single_transformer_blocks.0.attn.to_q.weight"]
    else:
        del shapes["single_stream_modulation.linear.weight"]
    assert inspect_model(sparse_headers(tmp_path / "flux2_dev.safetensors", shapes))["family"] is None


def test_flux1_schnell_guidance_absence_does_not_become_flux2(tmp_path):
    result = inspect_model(sparse_headers(tmp_path / "flux2.safetensors", transformer_shapes(guidance=False)))
    assert result["family"] == "flux"
    assert "guidance=False" in result["evidence"][0]


@pytest.mark.parametrize(
    "kind,shapes,family",
    [
        ("text_encoder", CLIP, "flux"),
        ("text_encoder_2", T5, "flux"),
        ("vae", vae_shapes(), "flux"),
        ("vae", vae_shapes(True), "flux2"),
    ],
)
def test_shared_components_keep_role_and_compatible_families(tmp_path, kind, shapes, family):
    path = sparse_headers(tmp_path / "component.safetensors", shapes)
    result = inspect_model(path)
    assert result["kind"] == kind and family in result["family_candidates"]
    with safe_open(str(path), framework="pt", device="cpu") as weights:
        check_component(weights, family, kind)
        with pytest.raises(ValueError, match="looks like"):
            check_component(weights, "anima", kind)
        with pytest.raises(ValueError, match="looks like"):
            check_component(weights, family, "dit")


@pytest.mark.parametrize(
    "hidden,vocab,model_type,family,candidates",
    [
        (2560, 151936, None, None, ["krea2", "flux2"]),
        (2560, 151936, "qwen3", "flux2", ["flux2"]),
        (2560, 151936, "qwen3_vl", "krea2", ["krea2"]),
        (4096, 151936, "qwen3", "flux2", ["flux2"]),
        (5120, 131072, "mistral3", "flux2", ["flux2"]),
    ],
)
def test_qwen_and_mistral_need_config_to_resolve_shared_shapes(
    tmp_path, hidden, vocab, model_type, family, candidates
):
    path = sparse_headers(tmp_path / "krea2_anima_flux.safetensors", decoder_shapes(hidden, vocab))
    if model_type:
        (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    result = inspect_model(path)
    assert (
        result["family"] == family
        and result["family_candidates"] == candidates
        and result["kind"] == "text_encoder"
    )
    with safe_open(str(path), framework="pt", device="cpu") as weights:
        with pytest.raises(ValueError, match="完整本地 HF 目录"):
            check_component(weights, "flux2", "text_encoder")


@pytest.mark.parametrize("variant", ["flux", "dev", "klein4", "klein9"])
@pytest.mark.parametrize("sharded", [False, True])
def test_complete_hf_flux_roots_and_components_keep_directory(tmp_path, variant, sharded):
    root = hf_flux_directory(tmp_path / variant, variant, sharded)
    result = inspect_model(root)
    assert result["path"] == str(root) and result["family"] == ("flux" if variant == "flux" else "flux2")
    assert result["kind"] == "dit" and result["confidence"] == "high"
    assert result["files_inspected"] == (4 if variant == "flux" else 3) + int(sharded)
    for role in ("transformer", "vae", "text_encoder"):
        assert inspect_model(root / role)["path"] == str(root / role)


@pytest.mark.parametrize(
    "missing",
    [
        "transformer/config.json",
        "text_encoder/model.safetensors",
        "text_encoder_2/model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "tokenizer_2/tokenizer.json",
    ],
)
def test_flux_pipeline_missing_component_is_not_a_complete_model(tmp_path, missing):
    root = hf_flux_directory(tmp_path / "flux")
    (root / missing).unlink()
    with pytest.raises(ValueError, match="flux"):
        inspect_model(root)


@pytest.mark.parametrize("variant", ["dev", "klein4"])
def test_flux2_local_chat_template_required_and_disk_template_accepted(tmp_path, variant):
    root = hf_flux_directory(tmp_path / variant, variant)
    config = root / "tokenizer/tokenizer_config.json"
    config.write_text('{"tokenizer_class":"AutoTokenizer"}')
    with pytest.raises(ValueError, match="chat template"):
        inspect_model(root)
    templates = root / "tokenizer/chat_templates"
    templates.mkdir()
    (templates / "default.jinja").write_text("{{ messages }}")
    assert inspect_model(root)["family"] == "flux2"


@pytest.mark.parametrize("reference", ["../escape.json", "C:\\weights\\tokenizer.json", "/tmp/escape.json"])
@pytest.mark.parametrize("component", ["text_encoder", "tokenizer"])
def test_runtime_asset_references_cannot_escape_component(tmp_path, reference, component):
    root = hf_flux_directory(tmp_path / "flux")
    config = root / component / ("config.json" if component == "text_encoder" else "tokenizer_config.json")
    data = json.loads(config.read_text())
    data["tokenizer_file"] = reference
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="outside|relative"):
        inspect_model(root)


@pytest.mark.parametrize(
    "asset",
    [
        "transformer/config.json",
        "text_encoder/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/chat_templates/default.jinja",
    ],
)
def test_pipeline_metadata_symlinks_are_contained_without_allowed_callback(tmp_path, asset):
    root = hf_flux_directory(tmp_path / "klein4", "klein4")
    path = root / asset
    if "chat_templates" in asset:
        path.parent.mkdir()
        path.write_text("{{ messages }}")
    outside = tmp_path / "outside.json"
    path.replace(outside)
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="outside"):
        inspect_model(root)


def test_incompatible_pipeline_config_and_unsupported_variants_are_explicit(tmp_path):
    root = hf_flux_directory(tmp_path / "flux")
    config = root / "transformer/config.json"
    data = json.loads(config.read_text())
    data["in_channels"] = 384
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="incompatible geometry"):
        inspect_model(root)
    index = root / "model_index.json"
    data = json.loads(index.read_text())
    data["_class_name"] = "FluxKontextPipeline"
    index.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unsupported FLUX pipeline"):
        inspect_model(root)


def test_distilled_klein_is_identified_but_never_claimed_as_base(tmp_path):
    root = hf_flux_directory(tmp_path / "base_name_is_not_evidence", "klein4")
    index = root / "model_index.json"
    data = json.loads(index.read_text())
    data["is_distilled"] = True
    index.write_text(json.dumps(data))
    result = inspect_model(root)
    assert result["family"] == "flux2"
    assert any("is_distilled=true" in warning for warning in result["warnings"])


def test_api_inspection_and_scan_register_pipeline_root_once(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    root = hf_flux_directory(model_dir / "unhelpful_name", "flux", sharded=True)
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "missing")
    client = TestClient(app)
    try:
        response = client.post("/api/models/inspect", json={"path": str(root)})
        assert response.status_code == 200, response.text
        assert response.json()["path"] == str(root)
        response = client.post("/api/models/scan", json={"path": str(model_dir), "family": "anima"})
        assert response.status_code == 200, response.text
        assert [(row["path"], row["family"], row["kind"]) for row in response.json()] == [
            (str(root), "flux", "dit")
        ]
    finally:
        app.state.regularization.close()
        app.state.dataset_pipeline.close()
        app.state.ctx.versions.close()
        client.close()
        app.state.ctx.db.close()


@pytest.mark.parametrize("field,value", [("_class_name", []), ("tokenizer", ["transformers", []])])
def test_malformed_model_index_produces_validation_error(tmp_path, field, value):
    root = hf_flux_directory(tmp_path / "flux")
    path = root / "model_index.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="model index"):
        inspect_model(root)


def test_flux2_declared_mistral3_cannot_hide_decoder_only_config(tmp_path):
    root = hf_flux_directory(tmp_path / "dev", "dev")
    path = root / "text_encoder/config.json"
    data = json.loads(path.read_text())
    data["model_type"] = "mistral"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="incompatible text encoder"):
        inspect_model(root)


def test_flux2_download_rejected_before_network_or_registration(tmp_path):
    app = create_app(tmp_path / "studio", frontend_dist=tmp_path / "missing")
    with TestClient(app) as client:
        app.state.model_downloads.opener.open = lambda *a, **k: pytest.fail("incomplete TE must not download")
        response = client.post(
            "/api/models/downloads",
            json={
                "family": "flux2",
                "kind": "text_encoder",
                "repo_id": "example/model",
                "filename": "model.safetensors",
            },
        )
        assert response.status_code == 422, response.text
        assert "complete local HF directory" in response.text
        assert client.get("/api/models/downloads").json() == []
        assert client.get("/api/models").json() == []
