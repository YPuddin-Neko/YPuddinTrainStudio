"""FLUX.2's 32-channel VAE, 2x2 spatial packing and frozen batchnorm statistics."""

from pathlib import Path

import torch
from torch import Tensor

from ypuddin.models.base import LatentPipeline
from ypuddin.models.fingerprints import content_fingerprint
from ypuddin.models.memory import release_model_memory

from .loading import read_json, shapes


def patchify(x: Tensor) -> Tensor:
    b, c, h, w = x.shape
    if h % 2 or w % 2:
        raise ValueError("FLUX.2 VAE latent height and width must be even")
    return x.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


def unpatchify(x: Tensor) -> Tensor:
    b, c, h, w = x.shape
    return x.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)


class Flux2Latent(LatentPipeline):
    channels, stride = 128, 16

    def __init__(self, path: Path, *, device="cpu"):
        self.path, self.device = path, torch.device(device)
        self.dtype, self.vae = torch.float32, None
        config_file = (path if path.is_dir() else path.parent) / "config.json"
        self.config = read_json(config_file) if config_file.is_file() else {}
        if self.config and self.config.get("_class_name") != "AutoencoderKLFlux2":
            raise ValueError("FLUX.2 requires AutoencoderKLFlux2, not a FLUX.1 or SDXL VAE")
        if (
            self.config.get("latent_channels", 32) != 32
            or len(self.config.get("block_out_channels", [0] * 4)) != 4
        ):
            raise ValueError("FLUX.2 VAE must have 32 channels and spatial stride 8 before patchifying")
        headers = shapes(path)
        if headers.get("bn.running_mean") != (128,) or headers.get("bn.running_var") != (128,):
            raise ValueError("FLUX.2 VAE requires 128-channel batchnorm running statistics")
        self.fingerprint = content_fingerprint(
            [path, *([config_file] if config_file.is_file() else [])],
            namespace="flux2-vae-mode-fp32-patch2-bn-v1",
        )

    def _ensure(self):
        if self.vae is None:
            from diffusers import AutoencoderKLFlux2

            if self.path.is_dir():
                model, info = AutoencoderKLFlux2.from_pretrained(
                    str(self.path),
                    local_files_only=True,
                    use_safetensors=True,
                    torch_dtype=self.dtype,
                    output_loading_info=True,
                )
                if any(
                    info.get(key)
                    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                ):
                    raise ValueError(f"FLUX.2 VAE weights do not match config: {info}")
            else:
                from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
                from safetensors.torch import load_file

                state = load_file(str(self.path))
                with torch.device("meta"):
                    model = AutoencoderKLFlux2.from_config(self.config)
                if "encoder.down.0.block.0.norm1.weight" in state:
                    bn = {k: v for k, v in state.items() if k.startswith("bn.")}
                    # BFL nests these projections inside encoder/decoder; LDM's
                    # converter expects them at the autoencoder root.
                    for prefix, target in (
                        ("encoder.quant_conv", "quant_conv"),
                        ("decoder.post_quant_conv", "post_quant_conv"),
                    ):
                        for suffix in ("weight", "bias"):
                            if f"{prefix}.{suffix}" in state:
                                state[f"{target}.{suffix}"] = state.pop(f"{prefix}.{suffix}")
                    state = convert_ldm_vae_checkpoint(state, model.config)
                    state.update(bn)
                model.load_state_dict(state, strict=True, assign=True)
            self.vae = model.to(device=self.device, dtype=self.dtype).requires_grad_(False).eval()
        return self.vae

    def to(self, device):
        self.device = torch.device(device)
        if self.vae is not None:
            self.vae.to(self.device)

    def unload(self):
        if self.vae is None:
            return
        self.vae = None
        release_model_memory(self.device)

    @torch.no_grad()
    def encode(self, pixels: Tensor) -> Tensor:
        with torch.autocast(self.device.type, enabled=False):
            model = self._ensure()
            z = patchify(model.encode(pixels.to(self.device, self.dtype)).latent_dist.mode())
            mean = model.bn.running_mean.view(1, -1, 1, 1)
            std = (model.bn.running_var.view(1, -1, 1, 1) + model.config.batch_norm_eps).sqrt()
            return (z - mean) / std

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        with torch.autocast(self.device.type, enabled=False):
            model = self._ensure()
            mean = model.bn.running_mean.view(1, -1, 1, 1)
            std = (model.bn.running_var.view(1, -1, 1, 1) + model.config.batch_norm_eps).sqrt()
            z = unpatchify(latents.to(self.device, self.dtype) * std + mean)
            return model.decode(z).sample.float().clamp(-1, 1)
