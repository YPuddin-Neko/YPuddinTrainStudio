"""Real reduced Diffusers/Transformers networks, local weights and an end-to-end CPU Trainer run."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

pytest.importorskip("diffusers")
pytest.importorskip("transformers")
pytest.importorskip("accelerate")

from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from ypuddin.adapters import inject, load_adapter_file
from ypuddin.config import AdapterConfig, MemoryConfig, ModelConfig, TrainConfig
from ypuddin.models.sdxl.family import SDXLFamily
from ypuddin.models.sdxl.loading import (
    ASSETS,
    check_component_storage,
    load_clip,
    load_diffusers_component,
    read_component,
)


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def tiny_pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("sdxl-real-networks")
    torch.manual_seed(12)
    unet = UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        block_out_channels=(32, 64),
        layers_per_block=1,
        cross_attention_dim=80,
        attention_head_dim=(4, 4),
        norm_num_groups=4,
        addition_embed_type="text_time",
        addition_time_embed_dim=8,
        projection_class_embeddings_input_dim=80,
        use_linear_projection=True,
    )
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(8, 8, 8, 8),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=4,
        sample_size=64,
        scaling_factor=0.42,
    )
    common = dict(vocab_size=49408, num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=77)
    clip_l = CLIPTextModel(CLIPTextConfig(hidden_size=32, intermediate_size=64, **common))
    clip_g = CLIPTextModelWithProjection(
        CLIPTextConfig(hidden_size=48, intermediate_size=96, projection_dim=32, **common)
    )
    tokenizer = CLIPTokenizer.from_pretrained(str(ASSETS / "tokenizer"), local_files_only=True)
    tokenizer_2 = CLIPTokenizer.from_pretrained(str(ASSETS / "tokenizer_2"), local_files_only=True)
    pipe = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=clip_l,
        text_encoder_2=clip_g,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"),
        add_watermarker=False,
    )
    pipe.save_pretrained(root, safe_serialization=True)
    return root


def _load(root: Path, *, checkpoint=False):
    return SDXLFamily().load(
        ModelConfig(family="sdxl", dit_path=str(root)),
        MemoryConfig(activation_checkpointing="block" if checkpoint else "none"),
        device="cpu",
        dtype=torch.float32,
        backbone_device="cpu",
    )


def test_real_dual_clip_padding_pooling_cache_and_reload(tiny_pipeline):
    loaded = _load(tiny_pipeline)
    text = loaded.text
    assert text.models == [] and loaded.latent.vae is None
    captions = ["a red cat", ""]
    cond = text.encode(captions, "cpu")
    outputs = []
    for tokenizer, model in zip(text.tokenizers, text.models, strict=True):
        ids = tokenizer(
            captions, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        ).input_ids
        outputs.append(model(ids, output_hidden_states=True))
    torch.testing.assert_close(cond["embeds"], torch.cat([x.hidden_states[-2] for x in outputs], dim=-1))
    torch.testing.assert_close(cond["pooled"], outputs[1].text_embeds)
    assert cond["embeds"].shape == (2, 77, 80)
    assert cond["pooled"].shape == (2, 32)
    assert cond["embeds"][1, -1].abs().sum() > 0  # padding of an empty prompt is preserved
    entries = text.encode_for_cache(captions)
    text.unload()
    assert not text.models
    for actual in (text.cond_from_cache(entries, "cpu"), text.encode(captions, "cpu")):
        for key in cond.tensors:
            torch.testing.assert_close(actual[key], cond[key], rtol=0, atol=0)


def test_vae_real_encode_decode_scale_fp32_and_reload(tiny_pipeline):
    latent = _load(tiny_pipeline).latent
    pixels = torch.randn(1, 3, 64, 64).clamp(-1, 1)
    torch.manual_seed(7)
    encoded = latent.encode(pixels.to(torch.bfloat16))
    assert encoded.shape == (1, 4, 8, 8)
    assert encoded.dtype == torch.float32
    assert all(parameter.dtype == torch.float32 for parameter in latent.vae.parameters())
    torch.manual_seed(7)
    expected = latent.vae.encode(pixels.to(torch.bfloat16).float()).latent_dist.sample() * 0.42
    torch.testing.assert_close(encoded, expected)
    decoded = latent.decode(encoded)
    expected = latent.vae.decode(encoded / 0.42).sample.clamp(-1, 1)
    torch.testing.assert_close(decoded, expected)
    latent.unload()
    assert latent.vae is None
    torch.testing.assert_close(latent.decode(encoded), decoded, rtol=0, atol=0)


def test_forward_matches_native_unet_geometry_and_continuous_sampling_time(tiny_pipeline):
    family, loaded = SDXLFamily(), _load(tiny_pipeline)
    cond = loaded.text.encode(["a cat"], "cpu")
    noisy = torch.randn(1, 4, 8, 8)
    t = torch.tensor([0.3217])
    geometry = {
        "original_size": torch.tensor([[96, 80]]),
        "crop_top_left": torch.tensor([[8, 3]]),
        "target_size": torch.tensor([[64, 64]]),
    }
    kwargs = dict(
        encoder_hidden_states=cond["embeds"],
        added_cond_kwargs={
            "text_embeds": cond["pooled"],
            "time_ids": torch.tensor([[96, 80, 8, 3, 64, 64]]).float(),
        },
        return_dict=False,
    )
    torch.testing.assert_close(
        family.forward(loaded, noisy, t, cond, geometry=geometry),
        loaded.backbone(noisy, torch.tensor([321]), **kwargs)[0],
    )
    torch.testing.assert_close(
        family.forward(loaded, noisy, t, cond, geometry=geometry, inference=True),
        loaded.backbone(noisy, t * 1000, **kwargs)[0],
    )
    with pytest.raises(ValueError, match="geometry.original_size"):
        family.forward(loaded, noisy, t, cond, geometry={**geometry, "original_size": torch.zeros(2)})


@pytest.mark.parametrize("algo", ["lora", "lokr"])
def test_real_checkpointed_unet_trains_adapters_and_exports_standard_keys(tiny_pipeline, algo):
    family, loaded = SDXLFamily(), _load(tiny_pipeline, checkpoint=True)
    config = AdapterConfig(algo=algo, rank=4, alpha=4, factor=4, preset="attn-mlp")
    adapters = inject(loaded.backbone, config, family.presets()[config.preset], prefix="lora_unet")
    loaded.backbone.train()
    output = family.forward(
        loaded,
        torch.randn(1, 4, 8, 8, requires_grad=True),
        torch.tensor([0.3]),
        loaded.text.encode(["a cat"], "cpu"),
    )
    output.square().mean().backward()
    assert adapters.layers
    for layer in adapters.layers.values():
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for p in layer.adapter.parameters()
        )
    assert not any(p.grad is not None for n, p in loaded.backbone.named_parameters() if "adapter" not in n)
    exported, _ = adapters.export_state()
    assert any(k.startswith("lora_unet_down_blocks_") for k in exported)
    assert all(".base." not in k and "_orig_mod" not in k for k in exported)


def test_native_single_file_clip_override_uses_real_weights(tiny_pipeline, tmp_path):
    source = tiny_pipeline / "text_encoder"
    shutil.copy(source / "config.json", tmp_path / "config.json")
    file = tmp_path / "clip.safetensors"
    shutil.copy(source / "model.safetensors", file)
    actual = load_clip(file, "text_encoder", device="cpu", dtype=torch.float32)
    expected = CLIPTextModel.from_pretrained(source, local_files_only=True, use_safetensors=True)
    ids = torch.tensor([[49406, 320, 49407]])
    torch.testing.assert_close(actual(ids).last_hidden_state, expected(ids).last_hidden_state, rtol=0, atol=0)


@pytest.mark.parametrize("component", ["text_encoder", "text_encoder_2", "unet", "vae"])
@pytest.mark.parametrize("damage", ["missing", "unexpected"])
def test_hf_component_rejects_incomplete_or_unrecognized_weights(tiny_pipeline, tmp_path, component, damage):
    path = tmp_path / component
    shutil.copytree(tiny_pipeline / component, path)
    weight_file = next(path.glob("*.safetensors"))
    weights = load_file(weight_file)
    if damage == "missing":
        key = (
            next(key for key in weights if key.endswith("self_attn.k_proj.weight"))
            if component.startswith("text_encoder")
            else "encoder.conv_in.weight"
            if component == "vae"
            else "conv_in.weight"
        )
        del weights[key]
    else:
        weights["unrecognized_tensor.weight"] = torch.ones(1)
    save_file(weights, weight_file)
    loader = load_clip if component.startswith("text_encoder") else load_diffusers_component
    with pytest.raises(ValueError, match=f"SDXL {component} weights do not match config"):
        loader(path, component, device="cpu", dtype=torch.float32)


def test_real_model_identity_survives_relocation_but_tracks_weights_and_semantics(tiny_pipeline, tmp_path):
    from ypuddin.train import Trainer

    relocated = tmp_path / "copied-model"
    shutil.copytree(tiny_pipeline, relocated)

    def identity(path, *, prediction_type="epsilon"):
        config = TrainConfig.model_validate(
            {
                "model": {"family": "sdxl", "dit_path": str(path), "prediction_type": prediction_type},
                "checkpoint": {"output_dir": str(tmp_path / "run")},
            }
        )
        trainer = Trainer(config, device="cpu")
        try:
            trainer.family = SDXLFamily()
            trainer.loaded = _load(path)
            return trainer._model_identity(), trainer.loaded.extra["dit_config"].get("_name_or_path")
        finally:
            trainer.emitter.close()

    original, original_path = identity(tiny_pipeline)
    moved, moved_path = identity(relocated)
    assert original_path != moved_path  # Real Diffusers loader records the component directory.
    assert original == moved
    assert identity(relocated, prediction_type="v_prediction")[0] != original
    weight_file = relocated / "unet" / "diffusion_pytorch_model.safetensors"
    weights = load_file(weight_file)
    weights["conv_in.weight"][0, 0, 0, 0] += 1
    save_file(weights, weight_file)
    assert identity(relocated)[0] != original


def test_missing_embedded_encoders_rejected_before_weight_loading(tmp_path):
    file = tmp_path / "unet-only.safetensors"
    save_file({"conv_in.weight": torch.zeros(2, 4, 3, 3)}, file)
    with pytest.raises(ValueError, match="no text_encoder"):
        read_component(file, "text_encoder")
    issues = SDXLFamily().validate_config(ModelConfig(family="sdxl", dit_path=str(file)))
    assert any("no text_encoder" in issue for issue in issues)
    assert any("no vae" in issue for issue in issues)


@pytest.mark.parametrize("directory", [False, True])
def test_fp8_checkpoint_storage_rejected_without_casting_away_scaling(tmp_path, directory):
    file = tmp_path / "diffusion_pytorch_model.safetensors"
    key = "conv_in.weight" if directory else "model.diffusion_model.input_blocks.0.0.weight"
    save_file({key: torch.zeros(2, 4, 3, 3, dtype=torch.float8_e4m3fn)}, file)
    path = tmp_path if directory else file
    with pytest.raises(ValueError, match="FP8 checkpoint storage"):
        check_component_storage(path, "unet")
    if not directory:
        with pytest.raises(ValueError, match="FP8 checkpoint storage"):
            read_component(path, "unet")


@pytest.mark.parametrize(
    "prediction_type,sampler,zero_snr", [("epsilon", "euler", False), ("v_prediction", "heun", True)]
)
def test_real_trainer_cache_train_preview_save_and_reload(
    tiny_pipeline, tmp_path, prediction_type, sampler, zero_snr
):
    from ypuddin.train import Trainer

    data = tmp_path / "images"
    data.mkdir()
    Image.new("RGB", (80, 64), (150, 40, 20)).save(data / "cat.png")
    (data / "cat.txt").write_text("a red cat")
    config = TrainConfig.model_validate(
        {
            "model": {
                "family": "sdxl",
                "dit_path": str(tiny_pipeline),
                "dtype": "fp32",
                "prediction_type": prediction_type,
                "zero_terminal_snr": zero_snr,
            },
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
                "sampler": sampler,
                "scheduler": "uniform",
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
    before = {k: t.clone() for k, t in trainer.adapters.training_state_dict().items()}
    assert trainer.run() == "finished" and trainer.progress.step == 2
    assert any(not torch.equal(before[k], v) for k, v in trainer.adapters.training_state_dict().items())
    assert torch.isfinite(torch.tensor(trainer.progress.extra["train_loss"]["loss"]))
    samples = trainer.sample_images("verify")
    assert samples and Image.open(samples[0]).size == (64, 64)
    assert trainer.loaded.latent.vae is None  # preview restores the cache-training residency
    exports = list((tmp_path / "run").glob("*.safetensors"))
    assert exports
    tensors, metadata = load_adapter_file(exports[0])
    reloaded = _load(tiny_pipeline)
    fresh = inject(
        reloaded.backbone, config.adapter, SDXLFamily().presets()[config.adapter.preset], prefix="lora_unet"
    )
    fresh.load_state(tensors, strict=True)
    assert metadata and len(fresh.layers) == len(trainer.adapters.layers)
    assert all(torch.isfinite(t).all() for t in load_file(exports[0]).values())


def test_bundled_assets_have_source_hashes_and_offline_configs():
    import hashlib

    manifest = json.loads((ASSETS / "manifest.json").read_text())
    assert manifest["repository"] == "stabilityai/stable-diffusion-xl-base-1.0"
    for name, entry in manifest["files"].items():
        assert hashlib.sha256((ASSETS / name).read_bytes()).hexdigest() == entry["sha256"]


def test_incomplete_model_still_has_a_latent_spec_for_data_planning(tmp_path):
    from ypuddin.train.plan import plan

    family = SDXLFamily()
    for path in (None, str(tmp_path / "not-chosen.safetensors")):
        cfg = ModelConfig(family="sdxl", dit_path=path)
        assert family.latent_fingerprint(cfg, dtype=torch.float32) == family.spec.latent.fingerprint
    Image.new("RGB", (64, 64), "red").save(tmp_path / "training.png")
    (tmp_path / "training.txt").write_text("a red image")
    result = plan(
        {
            "model": {"family": "sdxl"},
            "dataset": {"sources": [{"path": str(tmp_path)}], "resolutions": [64]},
            "objective": {"timestep_sampling": "uniform"},
            "sampling": {"enabled": False},
        },
        device="cpu",
    )
    assert not result["ok"] and any("dit_path" in item["msg"] for item in result["errors"])
    assert result["images"] == 1 and result["buckets"]


def _ldm_unet_key(key: str) -> str:
    """Serialize the real two-stage fixture in the original SDXL checkpoint namespace."""
    direct = {
        "conv_in": "input_blocks.0.0",
        "conv_norm_out": "out.0",
        "conv_out": "out.2",
        "time_embedding.linear_1": "time_embed.0",
        "time_embedding.linear_2": "time_embed.2",
        "add_embedding.linear_1": "label_emb.0.0",
        "add_embedding.linear_2": "label_emb.0.2",
    }
    for source, target in direct.items():
        if key.startswith(source + "."):
            return target + key[len(source) :]
    match = re.match(r"(down|up)_blocks\.(\d+)\.(resnets|attentions)\.(\d+)\.(.*)", key)
    if match:
        direction, block, kind, layer, suffix = match.groups()
        index = 2 * int(block) + int(layer) + (1 if direction == "down" else 0)
        key = f"{'input' if direction == 'down' else 'output'}_blocks.{index}.{0 if kind == 'resnets' else 1}.{suffix}"
        if kind == "attentions":
            return key
    elif key.startswith("down_blocks.0.downsamplers.0.conv."):
        return key.replace("down_blocks.0.downsamplers.0.conv.", "input_blocks.2.0.op.")
    elif key.startswith("up_blocks.0.upsamplers.0.conv."):
        return key.replace("up_blocks.0.upsamplers.0.conv.", "output_blocks.1.2.conv.")
    elif key.startswith("mid_block.attentions.0."):
        return key.replace("mid_block.attentions.0.", "middle_block.1.")
    else:
        key = key.replace("mid_block.resnets.0.", "middle_block.0.").replace(
            "mid_block.resnets.1.", "middle_block.2."
        )
    for source, target in {
        "norm1": "in_layers.0",
        "conv1": "in_layers.2",
        "norm2": "out_layers.0",
        "conv2": "out_layers.3",
        "time_emb_proj": "emb_layers.1",
        "conv_shortcut": "skip_connection",
    }.items():
        key = key.replace("." + source + ".", "." + target + ".")
    return key


def _ldm_vae_key(key: str) -> str:
    from diffusers.loaders.single_file_utils import DIFFUSERS_TO_LDM_MAPPING

    if key in DIFFUSERS_TO_LDM_MAPPING["vae"]:
        return DIFFUSERS_TO_LDM_MAPPING["vae"][key]
    key = re.sub(r"encoder.down_blocks\.(\d+)\.resnets\.", r"encoder.down.\1.block.", key)
    key = re.sub(r"encoder.down_blocks\.(\d+)\.downsamplers.0\.", r"encoder.down.\1.downsample.", key)
    key = re.sub(r"decoder.up_blocks\.(\d+)\.resnets\.", lambda m: f"decoder.up.{3 - int(m[1])}.block.", key)
    key = re.sub(
        r"decoder.up_blocks\.(\d+)\.upsamplers.0\.", lambda m: f"decoder.up.{3 - int(m[1])}.upsample.", key
    )
    key = key.replace("mid_block.resnets.0.", "mid.block_1.").replace("mid_block.resnets.1.", "mid.block_2.")
    if "mid_block.attentions.0." in key:
        key = key.replace("mid_block.attentions.0.", "mid.attn_1.")
        for source, target in {
            "group_norm": "norm",
            "to_q": "q",
            "to_k": "k",
            "to_v": "v",
            "to_out.0": "proj_out",
        }.items():
            key = key.replace("." + source + ".", "." + target + ".")
    return key.replace("conv_shortcut", "nin_shortcut")


@pytest.fixture(scope="module")
def full_ldm_checkpoint(tiny_pipeline, tmp_path_factory):
    from diffusers.loaders.single_file_utils import DIFFUSERS_TO_LDM_MAPPING

    root = tmp_path_factory.mktemp("sdxl-ldm-checkpoint")
    state = {}
    for component in ("unet", "vae", "text_encoder", "text_encoder_2"):
        source = tiny_pipeline / component
        target = root / "sdxl_config" / component
        target.mkdir(parents=True)
        shutil.copy(source / "config.json", target / "config.json")
        weights = load_file(next(source.glob("*.safetensors")))
        for key, tensor in weights.items():
            if component == "unet":
                mapped = "model.diffusion_model." + _ldm_unet_key(key)
            elif component == "vae":
                mapped = "first_stage_model." + _ldm_vae_key(key)
                if "mid.attn_1" in mapped and tensor.ndim == 2:
                    tensor = tensor[:, :, None, None]
            elif component == "text_encoder":
                mapped = "conditioner.embedders.0.transformer." + (
                    key if key.startswith("text_model.") else "text_model." + key
                )
            else:
                mapping = DIFFUSERS_TO_LDM_MAPPING["openclip"]
                if key in mapping["layers"]:
                    mapped = mapping["layers"][key]
                    if key == "text_projection.weight":
                        tensor = tensor.T.contiguous()
                elif any("." + projection + "." in key for projection in ("q_proj", "k_proj", "v_proj")):
                    if ".q_proj." not in key:
                        continue
                    tensor = torch.cat(
                        [weights[key.replace(".q_proj.", f".{p}.")] for p in ("q_proj", "k_proj", "v_proj")]
                    )
                    mapped = key.replace(".q_proj.weight", ".in_proj_weight").replace(
                        ".q_proj.bias", ".in_proj_bias"
                    )
                    for src, dst in mapping["transformer"].items():
                        mapped = mapped.replace(src, dst)
                    mapped = "transformer." + mapped
                else:
                    mapped = key
                    for src, dst in mapping["transformer"].items():
                        mapped = mapped.replace(src, dst)
                    mapped = "transformer." + mapped
                mapped = "conditioner.embedders.1.model." + mapped
            state[mapped] = tensor.contiguous()
    path = root / "complete.safetensors"
    save_file(state, path)
    return path


def test_full_ldm_checkpoint_embedded_components_match_real_hf_pipeline(tiny_pipeline, full_ldm_checkpoint):
    """Exercise the actual upstream LDM conversions with all four real component networks."""
    family = SDXLFamily()
    actual = _load(full_ldm_checkpoint)
    expected = _load(tiny_pipeline)
    assert actual.text.models == [] and actual.latent.vae is None
    assert actual.backbone.state_dict().keys() == expected.backbone.state_dict().keys()
    for key, tensor in actual.backbone.state_dict().items():
        torch.testing.assert_close(tensor, expected.backbone.state_dict()[key], rtol=0, atol=0)
    cond, cond_ref = actual.text.encode(["a cat"], "cpu"), expected.text.encode(["a cat"], "cpu")
    for key in cond.tensors:
        torch.testing.assert_close(cond[key], cond_ref[key], rtol=0, atol=0)
    pixels = torch.randn(1, 3, 64, 64).clamp(-1, 1)
    torch.manual_seed(5)
    latent = actual.latent.encode(pixels)
    torch.manual_seed(5)
    torch.testing.assert_close(latent, expected.latent.encode(pixels), rtol=0, atol=0)
    torch.testing.assert_close(actual.latent.decode(latent), expected.latent.decode(latent), rtol=0, atol=0)
    t = torch.tensor([0.45])
    torch.testing.assert_close(
        family.forward(actual, latent, t, cond), family.forward(expected, latent, t, cond_ref), rtol=0, atol=0
    )


@pytest.mark.parametrize("backbone,text", [(True, False), (False, True), (True, True)])
def test_full_finetune_native_components_and_reload(tiny_pipeline, tmp_path, backbone, text):
    from ypuddin.config import load_config
    from ypuddin.train import Trainer

    data = tmp_path / "full-data"
    data.mkdir()
    Image.new("RGB", (64, 64), (120, 45, 80)).save(data / "cat.png")
    (data / "cat.txt").write_text("a red cat")
    cfg = TrainConfig.model_validate(
        {
            "training": {"mode": "full", "train_backbone": backbone, "train_text_encoder": text},
            "model": {"family": "sdxl", "dit_path": str(tiny_pipeline), "dtype": "fp32"},
            "dataset": {
                "sources": [{"path": str(data)}],
                "resolutions": [64],
                "num_workers": 0,
                "batch_size": 1,
                "text_encoding": "online" if text else "cached",
            },
            "memory": {"offload_text_encoder": not text, "activation_checkpointing": "block"},
            "objective": {"timestep_sampling": "uniform"},
            "loop": {"max_steps": 2, "mixed_precision": "no"},
            "sampling": {
                "enabled": True,
                "steps": 2,
                "cfg": 1,
                "width": 64,
                "height": 64,
                "prompts": [{"prompt": "a red cat", "width": 64, "height": 64, "steps": 2}],
            },
            "checkpoint": {
                "output_dir": str(tmp_path / "full-run"),
                "name": "full",
                "save_dtype": "fp32",
                "save_state_every_steps": 1,
            },
        }
    )
    trainer = Trainer(cfg, device="cpu")
    trainer.prepare()
    from ypuddin.train.plan import plan

    planned = plan(cfg, device="cpu")
    assert planned["ok"], planned["errors"]
    assert planned["params"]["trainable"] == trainer.adapters.num_params()
    before = trainer.adapters.training_state_dict()
    assert trainer.run() == "finished"
    after = trainer.adapters.training_state_dict()
    for component in trainer.adapters.modules:
        assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith(component + ".")), (
            component
        )
    assert trainer.sample_images("full-preview")
    artifact = tmp_path / "full-run/full-final.model"
    restored_cfg = load_config(artifact / "config.toml")
    restored_cfg.checkpoint.output_dir = str(tmp_path / "reloaded-run")
    restored_cfg.loop.max_steps = 1
    restored = Trainer(restored_cfg, device="cpu")
    restored.prepare()
    for k, value in restored.adapters.training_state_dict().items():
        torch.testing.assert_close(value, after[k], rtol=0, atol=0)
    assert restored.run() == "finished"

    resume_cfg = cfg.model_copy(deep=True)
    resume_cfg.checkpoint.output_dir = str(tmp_path / "resumed-run")
    resume_cfg.checkpoint.resume = str(tmp_path / "full-run/state-1")
    resumed = Trainer(resume_cfg, device="cpu")
    assert resumed.run() == "finished"
    for k, value in resumed.adapters.training_state_dict().items():
        torch.testing.assert_close(value, after[k], rtol=0, atol=0)
