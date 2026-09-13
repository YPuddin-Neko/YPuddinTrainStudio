"""Load local SDXL components without materializing unused checkpoint components.

The standard LDM/A1111 checkpoint is converted by Diffusers, not by a parallel key mapper.
Component configs and tokenizers are bundled for ordinary SDXL base checkpoints; a local
Diffusers directory or adjacent ``sdxl_config/`` can supply custom component geometries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

ASSETS = Path(__file__).parent / "assets"
PREFIXES = {
    "unet": "model.diffusion_model.",
    "vae": "first_stage_model.",
    "text_encoder": "conditioner.embedders.0.transformer.",
    "text_encoder_2": "conditioner.embedders.1.model.",
}


def component_path(checkpoint: str | Path, component: str, override: str | Path | None = None) -> Path:
    path = Path(override or checkpoint).expanduser()
    if path.is_dir() and (path / component).is_dir():
        return path / component
    return path


def config_location(path: Path, component: str) -> tuple[Path, str]:
    if path.is_dir() and (path / "config.json").is_file():
        return path, ""
    parent = path if path.is_dir() else path.parent
    if (parent / "sdxl_config" / component / "config.json").is_file():
        return parent / "sdxl_config", component
    if (parent / component / "config.json").is_file():
        return parent, component
    if path.is_file() and (parent / "config.json").is_file():
        return parent, ""
    return ASSETS, component


def component_config(path: Path, component: str) -> dict[str, Any]:
    root, subfolder = config_location(path, component)
    return json.loads((root / subfolder / "config.json").read_text(encoding="utf-8"))


def config_asset(path: Path, component: str) -> Path:
    root, subfolder = config_location(path, component)
    return root / subfolder / "config.json"


def component_keys(keys: list[str], component: str) -> list[str]:
    selected = [key for key in keys if key.startswith(PREFIXES[component])]
    if selected:
        return selected
    native_markers = {
        "unet": ("conv_in.weight",),
        "vae": ("encoder.conv_in.weight",),
        "text_encoder": ("text_model.embeddings.token_embedding.weight", "embeddings.token_embedding.weight"),
        "text_encoder_2": ("text_model.embeddings.token_embedding.weight",),
    }
    if any(key.startswith(tuple(PREFIXES.values())) for key in keys) or not any(
        marker in keys for marker in native_markers[component]
    ):
        raise ValueError(f"SDXL checkpoint has no {component}; supply its external path")
    if component == "text_encoder_2" and "text_projection.weight" not in keys:
        raise ValueError("SDXL CLIP-G checkpoint has no text_projection.weight")
    return keys


def check_component_storage(path: Path, component: str) -> None:
    """Reject unsupported FP8 storage from headers before Diffusers can cast away its scales."""
    from safetensors import safe_open

    if path.is_dir():
        base = "diffusion_pytorch_model.safetensors" if component in {"unet", "vae"} else "model.safetensors"
        index = path / (base + ".index.json")
        if (path / base).is_file():
            files = [path / base]
        elif index.is_file():
            files = [
                path / shard
                for shard in set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
            ]
        else:
            return  # The directory loader/validator reports missing standard weights.
    else:
        files = [path] if path.suffix.lower() == ".safetensors" else []
    for file_path in files:
        with safe_open(str(file_path), framework="pt", device="cpu") as file:
            keys = list(file.keys()) if path.is_dir() else component_keys(list(file.keys()), component)
            if any(file.get_slice(key).get_dtype().startswith("F8") for key in keys):
                raise ValueError("SDXL FP8 checkpoint storage is unsupported; use BF16/FP16/FP32 weights")


def read_component(path: Path, component: str) -> dict[str, Tensor]:
    """Keep original LDM prefixes for the upstream converter; select before reading safetensors."""
    if path.suffix.lower() == ".safetensors":
        from safetensors import safe_open

        check_component_storage(path, component)
        with safe_open(str(path), framework="pt", device="cpu") as file:
            keys = list(file.keys())
            return {key: file.get_tensor(key) for key in component_keys(keys, component)}
    if path.suffix.lower() not in {".ckpt", ".pt", ".bin"}:
        raise ValueError(f"Unsupported SDXL component format: {path.suffix}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    if not isinstance(state, dict):
        raise ValueError(f"SDXL checkpoint {path.name} is not a state dictionary")
    selected = {key: state[key] for key in component_keys(list(state), component)}
    if any(isinstance(value, Tensor) and "float8" in str(value.dtype) for value in selected.values()):
        raise ValueError("SDXL FP8 checkpoint storage is unsupported; use BF16/FP16/FP32 weights")
    return selected


def _ready(model: nn.Module, component: str, device: torch.device | str, dtype: torch.dtype) -> nn.Module:
    missing = [name for name, value in (*model.named_parameters(), *model.named_buffers()) if value.is_meta]
    if missing:
        raise ValueError(f"SDXL {component} is missing checkpoint tensors: {missing[:5]}")
    model.to(device=device, dtype=dtype).requires_grad_(False)
    return model.eval()


def _check_loading_info(info: dict[str, Any], component: str) -> None:
    # Transformers can initialize absent tensors on CPU, so a meta-tensor check alone
    # does not prevent silently caching conditioning from partially random weights.
    issues = {
        key: info[key]
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if info.get(key)
    }
    if issues:
        raise ValueError(f"SDXL {component} weights do not match config: {issues}")


def load_diffusers_component(
    path: Path, component: str, *, device: torch.device | str, dtype: torch.dtype
) -> nn.Module:
    from diffusers import AutoencoderKL, UNet2DConditionModel

    cls = UNet2DConditionModel if component == "unet" else AutoencoderKL
    if path.is_dir():
        check_component_storage(path, component)
        model, info = cls.from_pretrained(
            str(path),
            torch_dtype=dtype,
            local_files_only=True,
            use_safetensors=True,
            output_loading_info=True,
        )
        _check_loading_info(info, component)
    else:
        root, subfolder = config_location(path, component)
        model = cls.from_single_file(
            read_component(path, component),
            config=str(root),
            subfolder=subfolder,
            torch_dtype=dtype,
            local_files_only=True,
        )
    return _ready(model, component, device, dtype)


def load_clip(path: Path, component: str, *, device: torch.device | str, dtype: torch.dtype) -> nn.Module:
    from transformers import CLIPTextModel, CLIPTextModelWithProjection

    cls = CLIPTextModel if component == "text_encoder" else CLIPTextModelWithProjection
    if path.is_dir():
        check_component_storage(path, component)
        model, info = cls.from_pretrained(
            str(path),
            torch_dtype=dtype,
            local_files_only=True,
            use_safetensors=True,
            output_loading_info=True,
        )
        _check_loading_info(info, component)
    else:
        state = read_component(path, component)
        root, subfolder = config_location(path, component)
        if any(key.startswith(PREFIXES[component]) for key in state):
            from diffusers.loaders.single_file_utils import create_diffusers_clip_model_from_ldm

            model = create_diffusers_clip_model_from_ldm(
                cls, state, config=str(root), subfolder=subfolder, torch_dtype=dtype, local_files_only=True
            )
        else:
            config = cls.config_class.from_pretrained(str(root), subfolder=subfolder, local_files_only=True)
            from accelerate import init_empty_weights

            with init_empty_weights():
                model = cls(config)
            # CLIP-L was flattened in Transformers 5.6; projection CLIP retains its wrapper.
            if not hasattr(model, "text_model"):
                state = {key.removeprefix("text_model."): value for key, value in state.items()}
            model.load_state_dict(state, strict=True, assign=True)
    return _ready(model, component, device, dtype)
