"""Offline safetensors loading for FLUX.2 dev and explicitly identified Klein base.

Geometry follows BFL flux2/model.py. Original tensor conversion delegates to the
installed Apache-2.0 Diffusers implementation; no reference trainer code is vendored.
"""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath

import torch
from safetensors import safe_open

VARIANTS = {
    "dev": dict(
        num_layers=8,
        num_single_layers=48,
        num_attention_heads=48,
        joint_attention_dim=15360,
        guidance_embeds=True,
    ),
    "klein-base-4b": dict(
        num_layers=5,
        num_single_layers=20,
        num_attention_heads=24,
        joint_attention_dim=7680,
        guidance_embeds=False,
    ),
    "klein-base-9b": dict(
        num_layers=8,
        num_single_layers=24,
        num_attention_heads=32,
        joint_attention_dim=12288,
        guidance_embeds=False,
    ),
}


def read_json(path: Path) -> dict:
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError(f"FLUX.2 JSON asset is too large: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"FLUX.2 JSON asset must be an object: {path}")
    return value


def component(root: str | Path, name: str, override: str | None = None) -> Path:
    if override:
        return Path(override).expanduser()
    root = Path(root).expanduser()
    if (root / name).is_dir():
        return root / name
    if name == "transformer":
        return root
    raise ValueError(f"FLUX.2 {name} needs a local component directory or an explicit model path")


def weight_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".safetensors":
            raise ValueError(f"FLUX.2 only loads safetensors weights: {path}")
        return [path]
    indexes = [
        p
        for p in (
            path / "diffusion_pytorch_model.safetensors.index.json",
            path / "model.safetensors.index.json",
        )
        if p.is_file()
    ]
    if len(indexes) > 1:
        raise ValueError(f"Ambiguous FLUX.2 shard indexes: {path}")
    if indexes:
        mapping = read_json(indexes[0]).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"Invalid FLUX.2 shard index: {indexes[0]}")
        names = set(mapping.values())
        if len(names) > 1024:
            raise ValueError("Too many FLUX.2 shards")
        files = []
        for name in names:
            if (
                not isinstance(name, str)
                or PureWindowsPath(name).is_absolute()
                or "\\" in name
                or Path(name).is_absolute()
            ):
                raise ValueError("FLUX.2 shard path must be local and relative")
            file = (path / name).resolve()
            if not file.is_relative_to(path.resolve()) or file.suffix != ".safetensors" or not file.is_file():
                raise ValueError(f"Missing or unsafe FLUX.2 shard: {name}")
            files.append(file)
        return sorted(files)
    files = [
        p for p in (path / "diffusion_pytorch_model.safetensors", path / "model.safetensors") if p.is_file()
    ]
    if len(files) != 1:
        raise ValueError(f"FLUX.2 needs one safetensors file or shard index: {path}")
    return files


def shapes(path: Path) -> dict[str, tuple[int, ...]]:
    result = {}
    for file in weight_files(path):
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_slice(key)
                if tensor.get_dtype().startswith("F8") or key.endswith((".scale_weight", ".weight_scale")):
                    raise ValueError(
                        "FLUX.2 prequantized/FP8 checkpoints are unsupported; select BF16/FP16 weights"
                    )
                if key in result:
                    raise ValueError(f"Duplicate FLUX.2 tensor: {key}")
                result[key] = tuple(tensor.get_shape())
    return result


def transformer_config(path: Path) -> dict:
    config_file = (path if path.is_dir() else path.parent) / "config.json"
    if config_file.is_file():
        cfg = read_json(config_file)
        if cfg.get("_class_name") == "Flux2Transformer2DModel":
            return cfg
    keys = shapes(path)
    image_key = next((k for k in ("img_in.weight", "model.diffusion_model.img_in.weight") if k in keys), None)
    if image_key is None or keys[image_key][1] != 128:
        raise ValueError("FLUX.2 transformer requires Diffusers config.json or an original BFL checkpoint")
    variant = {6144: "dev", 3072: "klein-base-4b", 4096: "klein-base-9b"}.get(keys[image_key][0])
    if variant is None:
        raise ValueError("Unrecognized FLUX.2 transformer width")
    return dict(
        in_channels=128,
        attention_head_dim=128,
        axes_dims_rope=[32] * 4,
        timestep_guidance_channels=256,
        mlp_ratio=3.0,
        rope_theta=2000,
        **VARIANTS[variant],
    )


def resolve_variant(root: Path, config: dict, requested: str = "auto") -> str:
    if config.get("in_channels", 128) != 128 or config.get("out_channels") not in (None, 128):
        raise ValueError("FLUX.2 requires 128 patchified latent channels")
    inferred = (
        "dev"
        if config.get("guidance_embeds", True)
        else ("klein-base-9b" if config.get("joint_attention_dim") == 12288 else "klein-base-4b")
    )
    manifest_file = root / "model_index.json"
    manifest = read_json(manifest_file) if manifest_file.is_file() else {}
    if manifest.get("is_distilled") is True:
        raise ValueError("FLUX.2 Klein distilled/KV variants are not supported; choose Klein base")
    if requested != "auto":
        if requested not in VARIANTS or requested != inferred:
            raise ValueError("FLUX.2 selected variant does not match the transformer configuration")
        return requested
    if inferred != "dev" and manifest.get("is_distilled") is not False:
        raise ValueError(
            "Klein base and distilled weights share shapes; select flux2_variant explicitly or use an HF base directory with is_distilled=false"
        )
    return inferred


def load_transformer(path: Path, config: dict, *, dtype, device):
    from diffusers import Flux2Transformer2DModel

    if path.is_dir():
        model, info = Flux2Transformer2DModel.from_pretrained(
            str(path),
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            output_loading_info=True,
        )
        if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"FLUX.2 transformer weights do not match config: {info}")
    else:
        from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers
        from safetensors.torch import load_file

        state = {k.removeprefix("model.diffusion_model."): v for k, v in load_file(str(path)).items()}
        if "img_in.weight" in state:
            state = convert_flux2_transformer_checkpoint_to_diffusers(state)
        with torch.device("meta"):
            model = Flux2Transformer2DModel.from_config(config)
        model.load_state_dict(
            {k: v.to(dtype) if v.is_floating_point() else v for k, v in state.items()},
            strict=True,
            assign=True,
        )
    return model.to(device=device, dtype=dtype).requires_grad_(False).eval()
