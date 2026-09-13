"""Local reduced real Flux2, Qwen3/Mistral3 and VAE networks; no downloaded weights."""

from __future__ import annotations

import json
import re
import shutil

import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

pytest.importorskip("diffusers")
pytest.importorskip("transformers")
pytest.importorskip("accelerate")

from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Pipeline, Flux2Transformer2DModel
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from ypuddin.adapters import inject, load_adapter_file
from ypuddin.config import AdapterConfig, MemoryConfig, ModelConfig, TrainConfig
from ypuddin.memory.block_swap import BlockSwapper
from ypuddin.models.base import TextCond
from ypuddin.models.flux2.family import Flux2Family, image_ids, text_ids
from ypuddin.models.flux2.latent import Flux2Latent, patchify, unpatchify
from ypuddin.models.flux2.loading import load_transformer, resolve_variant, transformer_config, weight_files


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="module")
def tiny_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("flux2-real")
    torch.manual_seed(19)
    transformer = Flux2Transformer2DModel(
        in_channels=128,
        num_layers=2,
        num_single_layers=3,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=24,
        timestep_guidance_channels=8,
        axes_dims_rope=(2, 2, 2, 2),
        mlp_ratio=2,
        guidance_embeds=False,
    )
    transformer.save_pretrained(root / "transformer", safe_serialization=True)
    vae = AutoencoderKLFlux2(
        block_out_channels=(8, 8, 8, 8),
        layers_per_block=1,
        norm_num_groups=4,
        latent_channels=32,
        sample_size=64,
    )
    vae.bn.running_mean.copy_(torch.linspace(-0.4, 0.8, 128))
    vae.bn.running_var.copy_(torch.linspace(0.2, 1.8, 128))
    vae.save_pretrained(root / "vae", safe_serialization=True)
    encoder = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=27,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=1024,
        )
    )
    encoder.save_pretrained(root / "text_encoder", safe_serialization=True)
    tokenizer = Tokenizer(
        WordLevel(
            {"<unk>": 0, "<pad>": 1, "cat": 2, "red": 3, "user": 4, "assistant": 5, "a": 6}, unk_token="<unk>"
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>")
    fast.chat_template = "{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}{% if add_generation_prompt %}assistant{% endif %}"
    fast.save_pretrained(root / "tokenizer")
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "Flux2KleinPipeline", "is_distilled": False})
    )
    return root


def load(root, *, checkpoint=False, swap=0):
    return Flux2Family().load(
        ModelConfig(family="flux2", dit_path=str(root)),
        MemoryConfig(activation_checkpointing="block" if checkpoint else "none", blocks_to_swap=swap),
        device="cpu",
        dtype=torch.float32,
        backbone_device="cpu",
    )


def test_lazy_phase_and_real_text_cache(tiny_root):
    loaded, family = load(tiny_root), Flux2Family()
    assert next(loaded.backbone.parameters()).is_meta
    assert loaded.text.model is None and loaded.latent.vae is None
    captions = ["a red cat", ""]
    cond = loaded.text.encode(captions, "cpu")
    oracle = Flux2KleinPipeline._get_qwen3_prompt_embeds(loaded.text.model, loaded.text.tokenizer, captions)
    torch.testing.assert_close(cond["embeds"], oracle, rtol=0, atol=0)
    assert cond["embeds"].shape == (2, 512, 24)
    assert cond["embeds"][1, -1].abs().sum() > 0  # retain meaningful padding
    entries = loaded.text.encode_for_cache(captions)
    family.materialize_backbone(loaded)
    assert loaded.text.model is None and loaded.latent.vae is None
    assert not any(p.is_meta for p in loaded.backbone.parameters())
    torch.testing.assert_close(
        loaded.text.cond_from_cache(entries, "cpu")["embeds"], cond["embeds"], rtol=0, atol=0
    )
    previous = loaded.backbone
    family.materialize_backbone(loaded)
    assert previous is loaded.backbone
    with pytest.raises(ValueError, match="cache"):
        loaded.text.cond_from_cache([{"embeds": torch.zeros(511, 24)}], "cpu")


def test_vae_exact_bn_mode_patch_roundtrip(tiny_root):
    latent = load(tiny_root).latent
    pixels = torch.randn(1, 3, 64, 96).clamp(-1, 1)
    z = latent.encode(pixels)
    vae = latent.vae
    raw = vae.encode(pixels).latent_dist.mode()
    expected = Flux2Pipeline._patchify_latents(raw)
    expected = (expected - vae.bn.running_mean[None, :, None, None]) / (
        vae.bn.running_var[None, :, None, None] + vae.config.batch_norm_eps
    ).sqrt()
    torch.testing.assert_close(z, expected, rtol=0, atol=0)
    assert z.shape == (1, 128, 4, 6)
    torch.testing.assert_close(unpatchify(patchify(raw)), raw, rtol=0, atol=0)
    torch.testing.assert_close(latent.decode(z), vae.decode(raw).sample.clamp(-1, 1))
    assert not vae.training and int(vae.bn.num_batches_tracked) == 0
    latent.unload()
    torch.testing.assert_close(latent.encode(pixels), z, rtol=0, atol=0)


def test_ids_and_unit_time_native_forward(tiny_root):
    family, loaded = Flux2Family(), load(tiny_root)
    family.materialize_backbone(loaded)
    x, embeds, t = torch.randn(2, 128, 3, 5), torch.randn(2, 7, 24), torch.tensor([0.0, 0.8317])
    torch.testing.assert_close(image_ids(x), Flux2Pipeline._prepare_latent_ids(x).float())
    torch.testing.assert_close(text_ids(embeds), Flux2Pipeline._prepare_text_ids(embeds).float())
    expected = loaded.backbone(
        Flux2Pipeline._pack_latents(x),
        embeds,
        t,
        image_ids(x),
        text_ids(embeds),
        guidance=None,
        return_dict=False,
    )[0]
    actual = family.forward(loaded, x, t, TextCond({"embeds": embeds}), guidance=9)
    torch.testing.assert_close(actual, expected.transpose(1, 2).reshape_as(x), rtol=0, atol=0)
    assert family.sampling_defaults(loaded).cfg == 4 and family.sampling_defaults(loaded).guidance is None


@pytest.mark.parametrize("algo", ["lora", "lokr", "lokr-full"])
@pytest.mark.parametrize("input_gradient", [False, True])
def test_checkpoint_swap_real_blocks_gradient_equivalence(tiny_root, algo, input_gradient):
    family = Flux2Family()
    baseline, swapped = load(tiny_root, checkpoint=True), load(tiny_root, checkpoint=True, swap=5)
    for loaded in (baseline, swapped):
        family.materialize_backbone(loaded)
    config = AdapterConfig(
        algo="lokr" if algo == "lokr-full" else algo,
        rank="full" if algo == "lokr-full" else 2,
        alpha=2,
        factor=2,
        preset="attn-mlp",
    )
    torch.manual_seed(82)
    a = inject(baseline.backbone, config, family.presets()[config.preset])
    torch.manual_seed(82)
    b = inject(swapped.backbone, config, family.presets()[config.preset])
    swapper = BlockSwapper(family.memory_layout(swapped).blocks, 5, "cpu")
    x = torch.randn(1, 128, 2, 3)
    cond = TextCond({"embeds": torch.randn(1, 7, 24)})
    grads = []
    for loaded in (baseline, swapped):
        loaded.backbone.train()
        noisy = x.detach().clone().requires_grad_(input_gradient)
        out = family.forward(loaded, noisy, torch.tensor([0.6]), cond)
        out.square().mean().backward()
        grads.append(noisy.grad)
    if input_gradient:
        torch.testing.assert_close(grads[0], grads[1], rtol=0, atol=0)
    else:
        assert grads == [None, None]
    nonzero = 0
    for (name, p), (other, q) in zip(
        baseline.backbone.named_parameters(), swapped.backbone.named_parameters(), strict=True
    ):
        assert name == other
        if p.requires_grad:
            assert p.grad is not None and q.grad is not None
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
            assert torch.isfinite(q.grad).all()
            nonzero += int(q.grad.abs().sum() > 0)
        else:
            assert p.grad is None and q.grad is None
    assert nonzero >= len(b.layers)
    swapper.release_all()
    assert not swapper._resident and not swapper._deferred
    swapper.set_forward_only(True)
    with torch.no_grad():
        torch.testing.assert_close(
            family.forward(baseline, x, torch.tensor([0.6]), cond),
            family.forward(swapped, x, torch.tensor([0.6]), cond),
            rtol=0,
            atol=0,
        )
    assert not swapper._resident
    swapper.remove()
    assert len(a.layers) == len(b.layers)


def test_ambiguity_and_distilled_rejection(tmp_path):
    cfg = dict(in_channels=128, guidance_embeds=False, joint_attention_dim=7680)
    with pytest.raises(ValueError, match="share shapes"):
        resolve_variant(tmp_path, cfg)
    assert resolve_variant(tmp_path, cfg, "klein-base-4b") == "klein-base-4b"
    assert (
        resolve_variant(tmp_path, {**cfg, "joint_attention_dim": 12288}, "klein-base-9b") == "klein-base-9b"
    )
    with pytest.raises(ValueError, match="does not match"):
        resolve_variant(tmp_path, cfg, "dev")
    (tmp_path / "model_index.json").write_text('{"is_distilled": true}')
    with pytest.raises(ValueError, match="distilled"):
        resolve_variant(tmp_path, cfg, "klein-base-4b")


@pytest.mark.parametrize(
    "shard", ["../outside.safetensors", "C:/outside.safetensors", "a.bin", "missing.safetensors"]
)
def test_shard_paths_are_safe(tmp_path, shard):
    (tmp_path / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": shard}})
    )
    with pytest.raises(ValueError):
        weight_files(tmp_path)


@pytest.mark.parametrize("fixture", ["tiny_root", "tiny_dev_root"])
def test_real_native_trainer_cache_train_preview_save_reload(fixture, request, tmp_path):
    from ypuddin.train import Trainer

    tiny_root = request.getfixturevalue(fixture)

    data = tmp_path / "data"
    data.mkdir()
    Image.new("RGB", (80, 64), (110, 50, 20)).save(data / "cat.png")
    (data / "cat.txt").write_text("a red cat")
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "flux2", "dit_path": str(tiny_root), "dtype": "fp32"},
            "dataset": {
                "sources": [{"path": str(data)}],
                "resolutions": [64],
                "num_workers": 0,
                "batch_size": 1,
                "text_encoding": "cached",
            },
            "adapter": {"algo": "lora", "rank": 2, "alpha": 2, "preset": "attn-only"},
            "memory": {"activation_checkpointing": "block", "blocks_to_swap": 3},
            "loop": {"max_steps": 2, "mixed_precision": "no"},
            "sampling": {
                "enabled": True,
                "steps": 2,
                "cfg": 2,
                "width": 64,
                "height": 64,
                "prompts": [{"prompt": "cat"}],
            },
            "checkpoint": {"output_dir": str(tmp_path / "output"), "save_on_finish": True},
        }
    )
    trainer = Trainer(cfg, device="cpu")
    trainer.prepare()
    before = {k: v.clone() for k, v in trainer.adapters.training_state_dict().items()}
    assert trainer.run() == "finished" and trainer.progress.step == 2
    assert any(not torch.equal(before[k], v) for k, v in trainer.adapters.training_state_dict().items())
    samples = trainer.sample_images("verify")
    assert samples and Image.open(samples[0]).size == (64, 64)
    assert trainer.loaded.text.model is None and trainer.loaded.latent.vae is None
    exports = list((tmp_path / "output").glob("*.safetensors"))
    tensors, _ = load_adapter_file(exports[0])
    fresh = load(tiny_root)
    Flux2Family().materialize_backbone(fresh)
    adapters = inject(
        fresh.backbone,
        cfg.adapter,
        Flux2Family().presets()[cfg.adapter.preset],
        prefix=Flux2Family.spec.adapter_prefix,
    )
    adapters.load_state(tensors, strict=True)
    assert all(torch.isfinite(v).all() for v in tensors.values())


@pytest.fixture(scope="module")
def tiny_dev_root(tiny_root, tmp_path_factory):
    from transformers import (
        Mistral3Config,
        Mistral3ForConditionalGeneration,
        PixtralImageProcessor,
        PixtralProcessor,
    )

    root = tmp_path_factory.mktemp("flux2-dev-real")
    shutil.copytree(tiny_root / "vae", root / "vae")
    config = dict(Flux2Transformer2DModel.load_config(tiny_root / "transformer"))
    config["guidance_embeds"] = True
    transformer = Flux2Transformer2DModel.from_config(config)
    transformer.save_pretrained(root / "transformer", safe_serialization=True)
    encoder = Mistral3ForConditionalGeneration(
        Mistral3Config(
            text_config={
                "model_type": "mistral",
                "vocab_size": 16,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 30,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
            },
            vision_config={
                "model_type": "pixtral",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "head_dim": 4,
                "image_size": 16,
                "patch_size": 8,
            },
            image_token_index=10,
        )
    )
    encoder.save_pretrained(root / "text_encoder", safe_serialization=True)
    tokenizer = Tokenizer(
        WordLevel(
            {
                "<unk>": 0,
                "cat": 2,
                "user": 4,
                "system": 5,
                "[IMG]": 10,
                "<pad>": 11,
                "[IMG_BREAK]": 12,
                "[IMG_END]": 13,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>")
    processor = PixtralProcessor(
        PixtralImageProcessor(),
        fast,
        chat_template="{% for m in messages %}{{ m.role }} {% for c in m.content %}{{ c.text }} {% endfor %}{% endfor %}",
    )
    processor.save_pretrained(root / "tokenizer")
    (root / "model_index.json").write_text('{"_class_name": "Flux2Pipeline"}')
    return root


def test_real_mistral_processor_hidden_layers_and_dev_guidance(tiny_dev_root):
    from diffusers.pipelines.flux2.pipeline_flux2 import format_input

    loaded, family = load(tiny_dev_root), Flux2Family()
    text = loaded.text
    cond = text.encode(["cat [IMG]"], "cpu")
    tokens = text.tokenizer.apply_chat_template(
        format_input(["cat [IMG]"]),
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    outputs = text.model(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        output_hidden_states=True,
        use_cache=False,
    )
    expected = torch.cat([outputs.hidden_states[k] for k in (10, 20, 30)], dim=-1)
    torch.testing.assert_close(cond["embeds"], expected, rtol=0, atol=0)
    assert type(text.tokenizer).__name__ == "PixtralProcessor"
    assert text.tokenizer.tokenizer.pad_token_id == 11
    assert cond["embeds"].shape == (1, 512, 24)
    family.materialize_backbone(loaded)
    assert text.model is None
    x, t = torch.randn(1, 128, 2, 2), torch.tensor([0.4132])
    for extra, value in (({}, 1), ({"inference": True}, 4), ({"guidance": 2.5}, 2.5)):
        expected = loaded.backbone(
            Flux2Pipeline._pack_latents(x),
            cond["embeds"],
            t,
            image_ids(x),
            text_ids(cond["embeds"]),
            guidance=torch.tensor([value], dtype=torch.float32),
            return_dict=False,
        )[0]
        torch.testing.assert_close(
            family.forward(loaded, x, t, cond, **extra),
            expected.transpose(1, 2).reshape_as(x),
            rtol=0,
            atol=0,
        )
    defaults = family.sampling_defaults(loaded)
    assert (defaults.cfg, defaults.guidance, defaults.steps) == (1, 4, 50)


def _original_transformer(state):
    out = {}
    plain = {
        "x_embedder": "img_in",
        "context_embedder": "txt_in",
        "time_guidance_embed.timestep_embedder.linear_1": "time_in.in_layer",
        "time_guidance_embed.timestep_embedder.linear_2": "time_in.out_layer",
        "time_guidance_embed.guidance_embedder.linear_1": "guidance_in.in_layer",
        "time_guidance_embed.guidance_embedder.linear_2": "guidance_in.out_layer",
        "double_stream_modulation_img.linear": "double_stream_modulation_img.lin",
        "double_stream_modulation_txt.linear": "double_stream_modulation_txt.lin",
        "single_stream_modulation.linear": "single_stream_modulation.lin",
        "proj_out": "final_layer.linear",
    }
    double = {
        "attn.to_out.0": "img_attn.proj",
        "attn.to_add_out": "txt_attn.proj",
        "ff.linear_in": "img_mlp.0",
        "ff.linear_out": "img_mlp.2",
        "ff_context.linear_in": "txt_mlp.0",
        "ff_context.linear_out": "txt_mlp.2",
        "attn.norm_q": "img_attn.norm.query_norm",
        "attn.norm_k": "img_attn.norm.key_norm",
        "attn.norm_added_q": "txt_attn.norm.query_norm",
        "attn.norm_added_k": "txt_attn.norm.key_norm",
    }
    single = {
        "attn.to_qkv_mlp_proj": "linear1",
        "attn.to_out": "linear2",
        "attn.norm_q": "norm.query_norm",
        "attn.norm_k": "norm.key_norm",
    }
    for key, value in state.items():
        module, _, suffix = key.rpartition(".")
        if module in plain:
            out[plain[module] + "." + suffix] = value
        elif module == "norm_out.linear":
            scale, shift = value.chunk(2)
            out["final_layer.adaLN_modulation.1.weight"] = torch.cat((shift, scale))
        elif module.startswith("transformer_blocks."):
            _, index, name = module.split(".", 2)
            if name in double:
                suffix = "scale" if "norm_" in name else suffix
                out[f"double_blocks.{index}.{double[name]}.{suffix}"] = value
            elif name in ("attn.to_q", "attn.add_q_proj"):
                names = (
                    ("to_q", "to_k", "to_v")
                    if name == "attn.to_q"
                    else ("add_q_proj", "add_k_proj", "add_v_proj")
                )
                side = "img" if name == "attn.to_q" else "txt"
                out[f"double_blocks.{index}.{side}_attn.qkv.weight"] = torch.cat(
                    [state[f"transformer_blocks.{index}.attn.{n}.weight"] for n in names]
                )
        elif module.startswith("single_transformer_blocks."):
            _, index, name = module.split(".", 2)
            suffix = "scale" if "norm_" in name else suffix
            out[f"single_blocks.{index}.{single[name]}.{suffix}"] = value
        else:
            raise AssertionError(key)
    return out


@pytest.mark.parametrize("dev", [False, True])
def test_original_single_transformer_exact_conversion(tiny_root, tiny_dev_root, tmp_path, dev):
    source = (tiny_dev_root if dev else tiny_root) / "transformer"
    expected = Flux2Transformer2DModel.from_pretrained(source, local_files_only=True)
    state = _original_transformer(expected.state_dict())
    file = tmp_path / "original.safetensors"
    save_file({"model.diffusion_model." + key: value.contiguous() for key, value in state.items()}, file)
    shutil.copy(source / "config.json", tmp_path / "config.json")
    actual = load_transformer(file, transformer_config(file), dtype=torch.float32, device="cpu")
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for key, value in actual.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[key], rtol=0, atol=0)


def test_original_single_vae_exact_conversion(tiny_root, tmp_path):
    from diffusers.loaders.single_file_utils import DIFFUSERS_TO_LDM_MAPPING

    state = load_file(tiny_root / "vae" / "diffusion_pytorch_model.safetensors")
    original = {}
    for key, value in state.items():
        if key.startswith("quant_conv."):
            mapped = "encoder." + key
        elif key.startswith("post_quant_conv."):
            mapped = "decoder." + key
        elif key in DIFFUSERS_TO_LDM_MAPPING["vae"]:
            mapped = DIFFUSERS_TO_LDM_MAPPING["vae"][key]
        else:
            mapped = re.sub(r"encoder.down_blocks\.(\d+)\.resnets\.", r"encoder.down.\1.block.", key)
            mapped = re.sub(
                r"encoder.down_blocks\.(\d+)\.downsamplers.0\.", r"encoder.down.\1.downsample.", mapped
            )
            mapped = re.sub(
                r"decoder.up_blocks\.(\d+)\.resnets\.", lambda m: f"decoder.up.{3 - int(m[1])}.block.", mapped
            )
            mapped = re.sub(
                r"decoder.up_blocks\.(\d+)\.upsamplers.0\.",
                lambda m: f"decoder.up.{3 - int(m[1])}.upsample.",
                mapped,
            )
            mapped = mapped.replace("mid_block.resnets.0.", "mid.block_1.").replace(
                "mid_block.resnets.1.", "mid.block_2."
            )
            if "mid_block.attentions.0." in mapped:
                mapped = mapped.replace("mid_block.attentions.0.", "mid.attn_1.")
                for src, dst in {
                    "group_norm": "norm",
                    "to_q": "q",
                    "to_k": "k",
                    "to_v": "v",
                    "to_out.0": "proj_out",
                }.items():
                    mapped = mapped.replace("." + src + ".", "." + dst + ".")
                if value.ndim == 2:
                    value = value[:, :, None, None]
            mapped = mapped.replace("conv_shortcut", "nin_shortcut")
        original[mapped] = value.contiguous()
    save_file(original, tmp_path / "ae.safetensors")
    shutil.copy(tiny_root / "vae" / "config.json", tmp_path / "config.json")
    latent = Flux2Latent(tmp_path / "ae.safetensors")
    actual = latent._ensure().state_dict()
    assert actual.keys() == state.keys()
    for key in state:
        torch.testing.assert_close(actual[key], state[key], rtol=0, atol=0)


@pytest.mark.parametrize("tokens,steps", [(16, 2), (1024, 50), (6400, 30)])
def test_empirical_shift_matches_diffusers_euler_schedule(tokens, steps):
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu

    from ypuddin.sampling.dispatch import noise_schedule

    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    scheduler.set_timesteps(
        steps, sigmas=np.linspace(1, 1 / steps, steps), mu=compute_empirical_mu(tokens, steps)
    )
    shift = Flux2Family().sampling_shift_for_model(None, tokens, steps=steps)
    torch.testing.assert_close(noise_schedule(steps, shift=shift), scheduler.sigmas, rtol=1e-6, atol=1e-7)


def test_fp8_storage_rejected_without_losing_scales(tmp_path):
    from ypuddin.models.flux2.loading import shapes

    path = tmp_path / "fp8.safetensors"
    save_file({"img_in.weight": torch.zeros(16, 128, dtype=torch.float8_e4m3fn)}, path)
    with pytest.raises(ValueError, match="FP8"):
        shapes(path)


@pytest.mark.parametrize(
    "memory",
    [
        {"blocks_to_swap": 1},
        {"base_precision": "fp8_e4m3"},
        {"activation_checkpointing": "unsloth"},
        {"compile": True},
    ],
)
def test_unsupported_memory_options_fail_explicitly(tiny_root, memory):
    with pytest.raises(ValueError, match="FLUX.2"):
        Flux2Family().load(
            ModelConfig(family="flux2", dit_path=str(tiny_root)),
            MemoryConfig(**memory),
            device="cpu",
            dtype=torch.float32,
        )


def test_original_official_variant_header_geometry(tmp_path):
    for width, variant, context, blocks, single in (
        (6144, "dev", 15360, 8, 48),
        (3072, "klein-base-4b", 7680, 5, 20),
        (4096, "klein-base-9b", 12288, 8, 24),
    ):
        file = tmp_path / (variant + ".safetensors")
        save_file({"img_in.weight": torch.zeros(width, 128)}, file)
        cfg = transformer_config(file)
        assert (cfg["joint_attention_dim"], cfg["num_layers"], cfg["num_single_layers"]) == (
            context,
            blocks,
            single,
        )
        assert resolve_variant(tmp_path, cfg, variant) == variant


def test_lora_explicit_peft_conversion_loads_real_diffusers(tiny_root):
    pytest.importorskip("peft")
    from ypuddin.adapters.convert import kohya_to_comfy

    family, loaded = Flux2Family(), load(tiny_root)
    family.materialize_backbone(loaded)
    adapters = inject(
        loaded.backbone,
        AdapterConfig(algo="lora", rank=2, alpha=1, preset="attn-mlp"),
        family.presets()["attn-mlp"],
        prefix=family.spec.adapter_prefix,
    )
    with torch.no_grad():
        for layer in adapters.layers.values():
            layer.adapter.up.normal_(std=0.05)
    tensors, _ = adapters.export_state()
    assert all(key.startswith("lora_transformer_") for key in tensors)
    peft = kohya_to_comfy(
        tensors, adapters.layers, prefix=family.spec.adapter_prefix, comfy_prefix="transformer"
    )
    # PEFT defaults alpha=rank; fold each exported alpha/rank into B.
    for key in list(peft):
        if key.endswith(".alpha"):
            module = key[:-6]
            rank = peft[module + ".lora_A.weight"].shape[0]
            peft[module + ".lora_B.weight"] *= float(peft.pop(key)) / rank
    oracle = load(tiny_root)
    family.materialize_backbone(oracle)
    oracle.backbone.load_lora_adapter(peft, prefix="transformer")
    x, t, cond = torch.randn(1, 128, 2, 3), torch.tensor([0.44]), TextCond({"embeds": torch.randn(1, 7, 24)})
    with torch.no_grad():
        torch.testing.assert_close(
            family.forward(loaded, x, t, cond), family.forward(oracle, x, t, cond), rtol=2e-5, atol=2e-6
        )


def test_real_hf_shards_load_strictly_and_missing_weight_rejected(tiny_root, tmp_path):
    model = Flux2Transformer2DModel.from_pretrained(tiny_root / "transformer", local_files_only=True)
    model.save_pretrained(tmp_path, safe_serialization=True, max_shard_size="20KB")
    assert len(weight_files(tmp_path)) > 1
    actual = load_transformer(tmp_path, transformer_config(tmp_path), device="cpu", dtype=torch.float32)
    for key, value in actual.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    # A directory containing an incomplete state dict must never quietly random-initialize it.
    broken = tmp_path / "incomplete"
    broken.mkdir()
    shutil.copy(tmp_path / "config.json", broken / "config.json")
    state = dict(model.state_dict())
    state.pop("x_embedder.weight")
    save_file(state, broken / "diffusion_pytorch_model.safetensors")
    with pytest.raises(ValueError, match="do not match"):
        load_transformer(broken, transformer_config(broken), device="cpu", dtype=torch.float32)
