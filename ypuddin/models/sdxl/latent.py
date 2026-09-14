"""Lazily loaded SDXL AutoencoderKL; FP32 encoding/decoding and checkpoint scaling factor."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from ypuddin.models.base import LatentPipeline
from ypuddin.models.fingerprints import content_fingerprint
from ypuddin.models.memory import release_model_memory

from .loading import component_config, config_asset, load_diffusers_component


class SDXLLatent(LatentPipeline):
    channels = 4
    stride = 8

    def __init__(self, path: Path, *, device: torch.device | str = "cpu"):
        self.path, self.device = path, torch.device(device)
        self.dtype = torch.float32
        self.vae: nn.Module | None = None
        config = component_config(path, "vae")
        self.channels = int(config["latent_channels"])
        self.stride = 2 ** (len(config["block_out_channels"]) - 1)
        self.scale = float(config.get("scaling_factor", 0.13025))
        if self.channels != 4 or self.stride != 8:
            raise ValueError("SDXL requires a VAE with 4 latent channels and spatial stride 8")
        if self.scale <= 0:
            raise ValueError("SDXL VAE scaling_factor must be positive")
        self.fingerprint = content_fingerprint(
            [path, config_asset(path, "vae")],
            namespace=f"sdxl-vae-posterior-sample-fp32-v1:scale={self.scale}",
        )

    def _ensure(self) -> nn.Module:
        if self.vae is None:
            self.vae = load_diffusers_component(self.path, "vae", device=self.device, dtype=self.dtype)
        return self.vae

    def to(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        if self.vae is not None:
            self.vae.to(self.device)

    def unload(self) -> None:
        if self.vae is None:
            return
        self.vae = None
        release_model_memory(self.device)

    @torch.no_grad()
    def encode(self, pixels: Tensor) -> Tensor:
        with torch.autocast(device_type=self.device.type, enabled=False):
            return self._ensure().encode(pixels.to(self.device, self.dtype)).latent_dist.sample() * self.scale

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        with torch.autocast(device_type=self.device.type, enabled=False):
            return (
                self._ensure()
                .decode(latents.to(self.device, self.dtype) / self.scale)
                .sample.float()
                .clamp(-1, 1)
            )
