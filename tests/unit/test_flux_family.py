"""Reduced real FLUX.1 networks, offline components and the complete CPU training lifecycle."""

from __future__ import annotations

import json
import shutil

import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

pytest.importorskip("diffusers")
pytest.importorskip("transformers")
pytest.importorskip("accelerate")
from diffusers import AutoencoderKL, FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizerFast,
    T5Config,
    T5EncoderModel,
    T5TokenizerFast,
)

from ypuddin.adapters import inject, load_adapter_file
from ypuddin.config import AdapterConfig, MemoryConfig, ModelConfig, TrainConfig
from ypuddin.models.flux.family import FluxFamily, image_ids, pack_latents, unpack_latents
from ypuddin.models.flux.loading import ASSETS, load_component


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module", params=[True, False], ids=["dev", "schnell"])
def tiny_pipeline(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("flux-real-networks")
    torch.manual_seed(12)
    transformer = FluxTransformer2DModel(
        in_channels=64,
        out_channels=64,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=4,
        joint_attention_dim=32,
        pooled_projection_dim=32,
        guidance_embeds=request.param,
        axes_dims_rope=(2, 2, 4),
    )
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(8, 8, 8, 8),
        layers_per_block=1,
        latent_channels=16,
        norm_num_groups=4,
        sample_size=64,
        scaling_factor=0.42,
        shift_factor=0.15,
        use_quant_conv=False,
        use_post_quant_conv=False,
    )
    clip = CLIPTextModel(
        CLIPTextConfig(
            vocab_size=49408,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            max_position_embeddings=77,
        )
    )
    t5 = T5EncoderModel(
        T5Config(
            vocab_size=32128,
            d_model=32,
            d_ff=64,
            d_kv=8,
            num_heads=4,
            num_layers=1,
            feed_forward_proj="gated-gelu",
            tie_word_embeddings=False,
        )
    )
    for model, component in (
        (transformer, "transformer"),
        (vae, "vae"),
        (clip, "text_encoder"),
        (t5, "text_encoder_2"),
    ):
        model.save_pretrained(root / component, safe_serialization=True)
    CLIPTokenizerFast.from_pretrained(
        str(ASSETS / "sdxl/assets/tokenizer"), local_files_only=True
    ).save_pretrained(root / "tokenizer")
    T5TokenizerFast.from_pretrained(
        str(ASSETS / "anima/assets/t5_old"), local_files_only=True
    ).save_pretrained(root / "tokenizer_2")
    (root / "model_index.json").write_text(json.dumps({"_class_name": "FluxPipeline"}))
    return root


def _load(root, *, checkpoint=False):
    return FluxFamily().load(
        ModelConfig(family="flux", dit_path=str(root)),
        MemoryConfig(activation_checkpointing="block" if checkpoint else "none"),
        device="cpu",
        dtype=torch.float32,
        backbone_device="cpu",
    )


def test_lazy_loading_padding_and_real_encoder_cache(tiny_pipeline):
    loaded = _load(tiny_pipeline)
    assert all(p.is_meta for p in loaded.backbone.parameters())
    assert not loaded.text.models and loaded.latent.vae is None
    captions = ["a red cat", ""]
    cond = loaded.text.encode(captions, "cpu")
    max_len = 512 if loaded.extra["variant"] == "dev" else 256
    assert cond["embeds"].shape == (2, max_len, 32)
    assert cond["pooled"].shape == (2, 32)
    outputs = []
    for tokenizer, model, length in zip(
        loaded.text.tokenizers, loaded.text.models, (77, max_len), strict=True
    ):
        ids = tokenizer(
            captions, padding="max_length", max_length=length, truncation=True, return_tensors="pt"
        ).input_ids
        outputs.append(model(ids, output_hidden_states=False))
    torch.testing.assert_close(cond["pooled"], outputs[0].pooler_output)
    torch.testing.assert_close(cond["embeds"], outputs[1].last_hidden_state)
    assert loaded.text.tokenizers[1]("a red cat")["input_ids"] == [3, 9, 1131, 1712, 1]
    assert cond["embeds"][1, -1].abs().sum() > 0
    entries = loaded.text.encode_for_cache(captions)
    loaded.text.unload()
    assert not loaded.text.models and all(p.is_meta for p in loaded.backbone.parameters())
    cached = loaded.text.cond_from_cache(entries, "cpu")
    for key in cond.tensors:
        torch.testing.assert_close(cached[key], cond[key], rtol=0, atol=0)
    FluxFamily().materialize_backbone(loaded)
    assert not any(p.is_meta for p in loaded.backbone.parameters())
    assert not loaded.text.models and loaded.latent.vae is None


def test_real_ae_normalization_and_unload(tiny_pipeline):
    latent = _load(tiny_pipeline).latent
    pixels = torch.randn(1, 3, 64, 64).clamp(-1, 1)
    torch.manual_seed(7)
    encoded = latent.encode(pixels)
    assert encoded.shape == (1, 16, 8, 8) and encoded.dtype == torch.float32
    torch.manual_seed(7)
    expected = (latent.vae.encode(pixels).latent_dist.sample() - 0.15) * 0.42
    torch.testing.assert_close(encoded, expected)
    decoded = latent.decode(encoded)
    torch.testing.assert_close(decoded, latent.vae.decode(encoded / 0.42 + 0.15).sample.clamp(-1, 1))
    latent.unload()
    assert latent.vae is None
    torch.testing.assert_close(latent.decode(encoded), decoded, rtol=0, atol=0)


def test_packing_ids_and_forward_match_diffusers(tiny_pipeline):
    family, loaded = FluxFamily(), _load(tiny_pipeline)
    cond = loaded.text.encode(["a cat"], "cpu")
    x, t = torch.randn(1, 16, 8, 12), torch.tensor([0.31])
    tokens = pack_latents(x)
    torch.testing.assert_close(tokens, FluxPipeline._pack_latents(x, 1, 16, 8, 12))
    torch.testing.assert_close(unpack_latents(tokens, 8, 12), x)
    ids = image_ids(8, 12, device="cpu", dtype=torch.float32)
    torch.testing.assert_close(ids, FluxPipeline._prepare_latent_image_ids(1, 4, 6, "cpu", torch.float32))
    actual = family.forward(loaded, x, t, cond)
    guided = loaded.extra["variant"] == "dev"
    kwargs = dict(
        encoder_hidden_states=cond["embeds"],
        pooled_projections=cond["pooled"],
        timestep=t,
        img_ids=ids,
        txt_ids=torch.zeros(cond["embeds"].shape[1], 3),
        return_dict=False,
    )
    expected = loaded.backbone(tokens, guidance=torch.ones(1) if guided else None, **kwargs)[0]
    torch.testing.assert_close(actual, unpack_latents(expected, 8, 12))
    preview = family.forward(loaded, x, t, cond, inference=True, guidance=3.5)
    expected = loaded.backbone(tokens, guidance=torch.tensor([3.5]) if guided else None, **kwargs)[0]
    torch.testing.assert_close(preview, unpack_latents(expected, 8, 12))
    if guided:
        assert not torch.equal(preview, actual)
    else:
        torch.testing.assert_close(preview, actual)
    defaults = family.sampling_defaults(loaded)
    assert defaults.steps == (28 if guided else 4)
    if not guided:
        assert family.sampling_shift_for_model(loaded, 1024) == 1
    assert family.sampling_shift(256) == pytest.approx(torch.exp(torch.tensor(0.5)).item())
    assert family.sampling_shift(4096) == pytest.approx(torch.exp(torch.tensor(1.15)).item())


@pytest.mark.parametrize("algo", ["lora", "lokr"])
def test_native_checkpointing_gradient_matches_regular_forward(tiny_pipeline, algo):
    family = FluxFamily()
    loaded = _load(tiny_pipeline)
    cond = loaded.text.encode(["a cat"], "cpu")
    family.materialize_backbone(loaded)
    config = AdapterConfig(algo=algo, rank=4, alpha=4, factor=4, preset="attn-mlp")
    torch.manual_seed(24)
    adapters = inject(
        loaded.backbone, config, family.presets()[config.preset], prefix=FluxFamily.spec.adapter_prefix
    )
    loaded.backbone.train()
    x = torch.randn(1, 16, 4, 4, requires_grad=True)
    first = family.forward(loaded, x, torch.tensor([0.4]), cond)
    first.square().mean().backward()
    gradients = {
        name: p.grad.clone()
        for name, p in loaded.backbone.named_parameters()
        if p.requires_grad and p.grad is not None
    }
    loaded.backbone.zero_grad(set_to_none=True)
    x.grad = None
    loaded.backbone.enable_gradient_checkpointing()
    second = family.forward(loaded, x, torch.tensor([0.4]), cond)
    second.square().mean().backward()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    for name, p in loaded.backbone.named_parameters():
        if name in gradients:
            torch.testing.assert_close(p.grad, gradients[name], rtol=1e-5, atol=1e-7)
        elif not p.requires_grad:
            assert p.grad is None
    assert gradients and all(torch.isfinite(v).all() for v in gradients.values())
    for layer in adapters.layers.values():
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.adapter.parameters())


def test_native_single_files_load_without_network_and_strict_missing(tiny_pipeline, tmp_path):
    for component in ("transformer", "text_encoder", "text_encoder_2", "vae"):
        source = tiny_pipeline / component
        directory = tmp_path / component
        directory.mkdir()
        shutil.copy(source / "config.json", directory / "config.json")
        original = next(source.glob("*.safetensors"))
        file = directory / "local.safetensors"
        shutil.copy(original, file)
        expected = load_component(source, component, device="cpu", dtype=torch.float32)
        actual = load_component(file, component, device="cpu", dtype=torch.float32)
        for key, tensor in actual.state_dict().items():
            torch.testing.assert_close(tensor, expected.state_dict()[key], rtol=0, atol=0)
    state = load_file(str(tmp_path / "transformer/local.safetensors"))
    del state["proj_out.weight"]
    save_file(state, tmp_path / "transformer/local.safetensors")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_component(
            tmp_path / "transformer/local.safetensors", "transformer", device="cpu", dtype=torch.float32
        )


def test_real_trainer_cache_materialize_train_preview_save_reload(tiny_pipeline, tmp_path):
    from ypuddin.train import Trainer

    data = tmp_path / "images"
    data.mkdir()
    Image.new("RGB", (80, 64), (150, 40, 20)).save(data / "cat.png")
    (data / "cat.txt").write_text("a red cat")
    config = TrainConfig.model_validate(
        {
            "model": {"family": "flux", "dit_path": str(tiny_pipeline), "dtype": "fp32"},
            "dataset": {
                "sources": [{"path": str(data)}],
                "resolutions": [64],
                "batch_size": 1,
                "num_workers": 0,
                "text_encoding": "cached",
            },
            "adapter": {"algo": "lora", "rank": 4, "alpha": 4, "preset": "attn-only"},
            "memory": {"activation_checkpointing": "block"},
            "objective": {"timestep_sampling": "uniform", "weighting": "none"},
            "loop": {"max_steps": 2, "mixed_precision": "no"},
            "sampling": {
                "steps": 2,
                "width": 64,
                "height": 64,
                "cfg": 1,
                "prompts": [{"prompt": "a red cat"}],
            },
            "checkpoint": {"output_dir": str(tmp_path / "run"), "save_on_finish": True},
        }
    )
    trainer = Trainer(config, device="cpu")
    trainer.prepare()
    assert trainer.text_mode == "cached"
    assert not trainer.loaded.text.models and trainer.loaded.latent.vae is None
    assert not any(p.is_meta for p in trainer.loaded.backbone.parameters())
    before = {k: t.clone() for k, t in trainer.adapters.training_state_dict().items()}
    assert trainer.run() == "finished" and trainer.progress.step == 2
    assert any(not torch.equal(before[k], v) for k, v in trainer.adapters.training_state_dict().items())
    assert torch.isfinite(torch.tensor(trainer.progress.extra["train_loss"]["loss"]))
    samples = trainer.sample_images("verify")
    assert samples and Image.open(samples[0]).size == (64, 64)
    assert trainer.loaded.latent.vae is None and not trainer.loaded.text.models
    exports = list((tmp_path / "run").glob("*.safetensors"))
    assert exports
    tensors, metadata = load_adapter_file(exports[0])
    reloaded = _load(tiny_pipeline)
    FluxFamily().materialize_backbone(reloaded)
    fresh = inject(
        reloaded.backbone,
        config.adapter,
        FluxFamily().presets()[config.adapter.preset],
        prefix=FluxFamily.spec.adapter_prefix,
    )
    fresh.load_state(tensors, strict=True)
    assert metadata and len(fresh.layers) == len(trainer.adapters.layers)
    assert all(torch.isfinite(t).all() for t in tensors.values())


def test_rejects_explicit_other_tasks_and_unsupported_memory(tiny_pipeline, tmp_path):
    root = tmp_path / "kontext"
    root.mkdir()
    for component in ("transformer", "text_encoder", "text_encoder_2", "vae", "tokenizer", "tokenizer_2"):
        (root / component).symlink_to(tiny_pipeline / component, target_is_directory=True)
    (root / "model_index.json").write_text(json.dumps({"_class_name": "FluxKontextPipeline"}))
    assert any(
        "Kontext" in s for s in FluxFamily().validate_config(ModelConfig(family="flux", dit_path=str(root)))
    )
    with pytest.raises(ValueError, match="block swap"):
        FluxFamily().load(
            ModelConfig(family="flux", dit_path=str(tiny_pipeline)),
            MemoryConfig(blocks_to_swap=1),
            device="cpu",
            dtype=torch.float32,
        )
    assert "block_swap" not in FluxFamily.spec.capabilities


def test_exported_attention_lora_is_parsed_by_real_diffusers_loader(tiny_pipeline):
    from diffusers.loaders import FluxLoraLoaderMixin

    family, loaded = FluxFamily(), _load(tiny_pipeline)
    family.materialize_backbone(loaded)
    cfg = AdapterConfig(algo="lora", rank=4, alpha=2, preset="attn-only")
    adapters = inject(loaded.backbone, cfg, family.presets()[cfg.preset], prefix=family.spec.adapter_prefix)
    with torch.no_grad():
        for layer in adapters.layers.values():
            layer.adapter.up.normal_(0, 0.02)
    state, _ = adapters.export_state()
    converted = FluxLoraLoaderMixin.lora_state_dict(state)
    assert len(converted) == len(adapters.layers) * 2
    for name, layer in adapters.layers.items():
        down = converted[f"transformer.{name}.lora_A.weight"]
        up = converted[f"transformer.{name}.lora_B.weight"]
        torch.testing.assert_close(up @ down, layer.adapter.delta_weight(), rtol=1e-5, atol=1e-7)


def test_fp8_storage_scaling_and_adapter_gradients_use_real_network(tiny_pipeline, tmp_path):
    from ypuddin.adapters.frozen import FrozenLinear

    source = tiny_pipeline / "transformer"
    state = load_file(str(source / "diffusion_pytorch_model.safetensors"))
    reference = {}
    # Real serialized FP8 tensors with a non-unit scale, including fused-independent projections.
    for key, value in list(state.items()):
        if value.ndim == 2 and key.endswith(".weight"):
            state[key] = (value / 0.5).to(torch.float8_e4m3fn)
            state[key.removesuffix(".weight") + ".scale_weight"] = torch.tensor(0.5)
            reference[key] = state[key].float() * 0.5
        else:
            reference[key] = value
    file = tmp_path / "fp8.safetensors"
    save_file(state, file)
    shutil.copy(source / "config.json", tmp_path / "config.json")
    model = load_component(file, "transformer", device="cpu", dtype=torch.float32)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, reference[key], rtol=0, atol=0)
    loaded = _load(tiny_pipeline)
    loaded.backbone = model
    loaded.extra["pending_backbone"] = None
    family = FluxFamily()
    cfg = AdapterConfig(algo="lora", rank=4, alpha=4, preset="attn-only")
    adapters = inject(
        model, cfg, family.presets()[cfg.preset], prefix=family.spec.adapter_prefix, base_precision="fp8_e4m3"
    )
    assert all(
        isinstance(layer.base, FrozenLinear) and layer.base.weight.dtype == torch.float8_e4m3fn
        for layer in adapters.layers.values()
    )
    model.train()
    output = family.forward(
        loaded, torch.randn(1, 16, 4, 4), torch.tensor([0.3]), loaded.text.encode(["a cat"], "cpu")
    )
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    for layer in adapters.layers.values():
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for p in layer.adapter.parameters()
        )


def test_sharded_local_transformer_and_caption_fingerprint_change(tiny_pipeline, tmp_path):
    source = tiny_pipeline / "transformer"
    state = load_file(str(source / "diffusion_pytorch_model.safetensors"))
    shutil.copy(source / "config.json", tmp_path / "config.json")
    keys = list(state)
    mapping = {}
    for i, part in enumerate((keys[::2], keys[1::2])):
        name = f"diffusion_pytorch_model-{i}.safetensors"
        save_file({key: state[key] for key in part}, tmp_path / name)
        mapping.update({key: name for key in part})
    (tmp_path / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping})
    )
    actual = load_component(tmp_path, "transformer", device="cpu", dtype=torch.float32)
    for key, value in actual.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    from ypuddin.models.flux.text import FluxText

    paths = (tiny_pipeline / "text_encoder", tiny_pipeline / "text_encoder_2")
    tokens = (tiny_pipeline / "tokenizer", tiny_pipeline / "tokenizer_2")
    a = FluxText(paths, tokens, max_len=256, device="cpu", dtype=torch.float32)
    b = FluxText(paths, tokens, max_len=512, device="cpu", dtype=torch.float32)
    assert a.fingerprint != b.fingerprint
    assert not a.models and not b.models


def test_adapter_target_discovery_works_before_any_model_is_selected():
    from ypuddin.adapters.rules import resolve_targets

    family = FluxFamily()
    model = family.meta_backbone(ModelConfig(family="flux"))
    assert all(p.is_meta for p in model.parameters())
    names = family.linear_module_names()
    config = AdapterConfig(preset="attn-mlp")
    targets = resolve_targets(names, config, family.presets()[config.preset])
    assert any(t.name.startswith("transformer_blocks.18.") for t in targets)
    assert any(t.name.startswith("single_transformer_blocks.37.") for t in targets)
    assert all(not t.name.startswith("time_text_embed") for t in targets)
