"""Offline FLUX.1 components. Header inspection and meta construction precede weight loading."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

ASSETS = Path(__file__).resolve().parent.parent
DEFAULTS = {
    "transformer": dict(
        patch_size=1,
        in_channels=64,
        out_channels=64,
        num_layers=19,
        num_single_layers=38,
        attention_head_dim=128,
        num_attention_heads=24,
        joint_attention_dim=4096,
        pooled_projection_dim=768,
        guidance_embeds=True,
        axes_dims_rope=[16, 56, 56],
    ),
    "text_encoder": dict(
        vocab_size=49408,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        bos_token_id=49406,
        eos_token_id=49407,
        pad_token_id=1,
    ),
    "text_encoder_2": dict(
        vocab_size=32128,
        d_model=4096,
        d_ff=10240,
        d_kv=64,
        num_heads=64,
        num_layers=24,
        feed_forward_proj="gated-gelu",
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        relative_attention_num_buckets=32,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    ),
    "vae": dict(
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D"] * 4,
        up_block_types=["UpDecoderBlock2D"] * 4,
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        latent_channels=16,
        norm_num_groups=32,
        sample_size=256,
        scaling_factor=0.3611,
        shift_factor=0.1159,
        use_quant_conv=False,
        use_post_quant_conv=False,
    ),
}


def component_path(root: Path, component: str, override: str | None = None) -> Path:
    if override:
        return Path(override).expanduser()
    candidate = root / component
    if candidate.is_dir():
        return candidate
    if component == "transformer":
        return root
    raise ValueError(
        f"FLUX.1 {component} is required; choose its local weights or a complete pipeline directory"
    )


def weight_files(path: Path, component: str) -> list[Path]:
    if path.is_file():
        if path.suffix != ".safetensors":
            raise ValueError(f"FLUX.1 {component} requires safetensors weights: {path}")
        return [path]
    base = (
        "diffusion_pytorch_model.safetensors" if component in {"transformer", "vae"} else "model.safetensors"
    )
    if (path / base).is_file():
        return [path / base]
    index = path / (base + ".index.json")
    if not index.is_file():
        raise ValueError(f"FLUX.1 {component} has no local safetensors weights: {path}")
    mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    files = []
    for name in sorted(set(mapping.values())):
        file = path / name
        if not file.resolve().is_relative_to(path.resolve()) or not file.is_file():
            raise ValueError(f"FLUX.1 {component} has an invalid or missing shard: {name}")
        files.append(file)
    if not files:
        raise ValueError(f"FLUX.1 {component} weight index is empty")
    return files


def strip_prefix(key: str, component: str) -> str:
    prefixes = ("model.diffusion_model.", "diffusion_model.") if component == "transformer" else ()
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def header(path: Path, component: str) -> dict[str, tuple[int, ...]]:
    result = {}
    for file in weight_files(path, component):
        with safe_open(str(file), framework="pt", device="cpu") as weights:
            for key in weights.keys():
                result[strip_prefix(key, component)] = tuple(weights.get_slice(key).get_shape())
    return result


def config_file(path: Path, component: str) -> Path | None:
    options = (
        [path / "config.json"]
        if path.is_dir()
        else [path.with_suffix(".json"), path.parent / component / "config.json", path.parent / "config.json"]
    )
    return next((p for p in options if p.is_file()), None)


def component_config(path: Path, component: str) -> dict:
    file = config_file(path, component)
    config = json.loads(file.read_text(encoding="utf-8")) if file else dict(DEFAULTS[component])
    if component == "transformer":
        keys = header(path, component)
        native = "x_embedder.weight" in keys
        image_key = "x_embedder.weight" if native else "img_in.weight"
        if image_key not in keys:
            raise ValueError("FLUX.1 checkpoint has no transformer image projection")
        guided = any(k.startswith(("guidance_in.", "time_text_embed.guidance_embedder.")) for k in keys)
        if file and bool(config.get("guidance_embeds", False)) != guided:
            raise ValueError("FLUX.1 guidance_embeds config does not match the checkpoint")
        config["guidance_embeds"] = guided
        if not file:
            if keys[image_key] != (3072, 64):
                raise ValueError("A nonstandard FLUX.1 transformer requires its adjacent config.json")
            prefix = "transformer_blocks." if native else "double_blocks."
            single = "single_transformer_blocks." if native else "single_blocks."
            config["num_layers"] = (
                max((int(k.split(".")[1]) for k in keys if k.startswith(prefix)), default=-1) + 1
            )
            config["num_single_layers"] = (
                max((int(k.split(".")[1]) for k in keys if k.startswith(single)), default=-1) + 1
            )
        if (
            config.get("in_channels") != 64
            or config.get("out_channels", 64) not in (None, 64)
            or keys[image_key][1] != 64
        ):
            raise ValueError(
                "FLUX.1 text-to-image requires 64 packed channels; Fill/Control models are unsupported"
            )
        if config.get("patch_size", 1) != 1:
            raise ValueError("FLUX.1 requires a transformer patch_size of 1 after latent packing")
        if not config.get("num_layers") or not config.get("num_single_layers"):
            raise ValueError("FLUX.1 requires both double-stream and single-stream blocks")
    return config


def read_state(path: Path, component: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    state = {}
    for file in weight_files(path, component):
        for key, value in load_file(str(file), device="cpu").items():
            key = strip_prefix(key, component)
            if key in state:
                raise ValueError(f"Duplicate FLUX.1 weight: {key}")
            state[key] = value
    # Dequantize one tensor at a time; do not retain a second full floating-point checkpoint.
    # Per-tensor ComfyUI scale is applied before original fused projections are split.
    for key in list(state):
        if key not in state or key.endswith((".scale_weight", ".weight_scale")) or key == "scaled_fp8":
            continue
        value = state[key]
        if value.is_floating_point():
            scale_keys = [
                key.removesuffix(".weight") + suffix for suffix in (".scale_weight", ".weight_scale")
            ]
            scales = [state.pop(k) for k in scale_keys if k in state]
            if len(scales) > 1 or (scales and scales[0].numel() != 1):
                raise ValueError(f"Unsupported FLUX.1 quantization scale: {key}")
            value = value.to(dtype)
            if scales:
                value = value * scales[0].to(dtype)
            state[key] = value
    state.pop("scaled_fp8", None)
    return state


def convert_original(state: dict[str, torch.Tensor], config: dict) -> dict[str, torch.Tensor]:
    """BFL namespaces -> Diffusers, including fused QKV and final shift/scale order.

    Mapping follows Diffusers' Apache-2.0 single-file converter. Split sizes come
    from the actual config instead of its hard-coded 3072-wide standard network.
    """
    result = {}

    def move(source, target, swap=False):
        for suffix in ("weight", "bias"):
            tensor = state.pop(f"{source}.{suffix}")
            if swap:
                tensor = torch.cat(tensor.chunk(2)[::-1], dim=0)
            result[f"{target}.{suffix}"] = tensor

    for source, target in (
        ("time_in", "timestep_embedder"),
        ("vector_in", "text_embedder"),
        ("guidance_in", "guidance_embedder"),
    ):
        if source == "guidance_in" and not config["guidance_embeds"]:
            continue
        for before, after in (("in_layer", "linear_1"), ("out_layer", "linear_2")):
            move(f"{source}.{before}", f"time_text_embed.{target}.{after}")
    move("img_in", "x_embedder")
    move("txt_in", "context_embedder")
    for i in range(config["num_layers"]):
        source, target = f"double_blocks.{i}", f"transformer_blocks.{i}"
        for stream, norm, ff, proj, added in (
            ("img", "norm1", "ff", "to_out.0", False),
            ("txt", "norm1_context", "ff_context", "to_add_out", True),
        ):
            move(f"{source}.{stream}_mod.lin", f"{target}.{norm}.linear")
            move(f"{source}.{stream}_mlp.0", f"{target}.{ff}.net.0.proj")
            move(f"{source}.{stream}_mlp.2", f"{target}.{ff}.net.2")
            move(f"{source}.{stream}_attn.proj", f"{target}.attn.{proj}")
            for suffix in ("weight", "bias"):
                qkv = state.pop(f"{source}.{stream}_attn.qkv.{suffix}").chunk(3, dim=0)
                for letter, value in zip("qkv", qkv, strict=True):
                    name = f"add_{letter}_proj" if added else f"to_{letter}"
                    result[f"{target}.attn.{name}.{suffix}"] = value
            for before, after in (("query_norm", "q"), ("key_norm", "k")):
                result[f"{target}.attn.norm_{'added_' if added else ''}{after}.weight"] = state.pop(
                    f"{source}.{stream}_attn.norm.{before}.scale"
                )
    width = config["num_attention_heads"] * config["attention_head_dim"]
    for i in range(config["num_single_layers"]):
        source, target = f"single_blocks.{i}", f"single_transformer_blocks.{i}"
        move(f"{source}.modulation.lin", f"{target}.norm.linear")
        move(f"{source}.linear2", f"{target}.proj_out")
        for suffix in ("weight", "bias"):
            fused = state.pop(f"{source}.linear1.{suffix}")
            parts = fused.split((width, width, width, fused.shape[0] - 3 * width), dim=0)
            for name, value in zip(("attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"), parts, strict=True):
                result[f"{target}.{name}.{suffix}"] = value
        for before, after in (("query_norm", "q"), ("key_norm", "k")):
            result[f"{target}.attn.norm_{after}.weight"] = state.pop(f"{source}.norm.{before}.scale")
    move("final_layer.linear", "proj_out")
    move("final_layer.adaLN_modulation.1", "norm_out.linear", swap=True)
    if state:
        raise ValueError(f"Unsupported FLUX.1 checkpoint tensors: {sorted(state)[:5]}")
    return result


def create_meta(component: str, config: dict) -> nn.Module:
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKL, FluxTransformer2DModel
    from transformers import CLIPTextConfig, CLIPTextModel, T5Config, T5EncoderModel

    with init_empty_weights():
        if component == "transformer":
            return FluxTransformer2DModel.from_config(config)
        if component == "vae":
            return AutoencoderKL.from_config(config)
        if component == "text_encoder":
            return CLIPTextModel(CLIPTextConfig.from_dict(config))
        return T5EncoderModel(T5Config.from_dict(config))


def load_component(path: Path, component: str, *, device, dtype, config: dict | None = None) -> nn.Module:
    config = config or component_config(path, component)
    model = create_meta(component, config)
    state = read_state(path, component, dtype)
    if component == "transformer" and "img_in.weight" in state:
        state = convert_original(state, config)
    elif component == "vae" and "encoder.down.0.block.0.norm1.weight" in state:
        from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

        state = convert_ldm_vae_checkpoint(state, config)
    elif component == "text_encoder":
        if not hasattr(model, "text_model"):
            state = {key.removeprefix("text_model."): value for key, value in state.items()}
        # A CLIP projection is unused by FLUX, which takes pooler_output directly.
        for key in ("text_projection.weight", "text_projection", "logit_scale", "logit_bias"):
            state.pop(key, None)
    elif component == "text_encoder_2":
        for a, b in (
            ("shared.weight", "encoder.embed_tokens.weight"),
            ("encoder.embed_tokens.weight", "shared.weight"),
        ):
            if a not in state and b in state:
                state[a] = state[b]
    model.load_state_dict(state, strict=True, assign=True)
    if any(p.is_meta for p in model.parameters()):
        raise ValueError(f"FLUX.1 {component} contains missing weights")
    return model.to(device=device, dtype=dtype).requires_grad_(False).eval()
