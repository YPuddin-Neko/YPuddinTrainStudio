"""FLUX.1 AutoencoderKL with checkpoint-owned shift/scale and FP32 VAE operations."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from ypuddin.models.base import LatentPipeline
from ypuddin.models.fingerprints import content_fingerprint

from .loading import component_config, config_file, load_component


class FluxLatent(LatentPipeline):
    channels, stride = 16, 8

    def __init__(self, path: Path, *, device="cpu"):
        self.path, self.device, self.dtype = path, torch.device(device), torch.float32
        self.config = component_config(path, "vae")
        self.scale = float(self.config["scaling_factor"])
        self.shift = float(self.config["shift_factor"])
        if self.config["latent_channels"] != 16 or len(self.config["block_out_channels"]) != 4:
            raise ValueError("FLUX.1 requires a 16-channel AE with spatial stride 8")
        if not math.isfinite(self.scale) or self.scale <= 0 or not math.isfinite(self.shift):
            raise ValueError("FLUX.1 AE scaling_factor and shift_factor are invalid")
        file = config_file(path, "vae")
        self.fingerprint = content_fingerprint(
            [path, *([file] if file else [])],
            namespace="flux1-ae-posterior-sample-fp32-v1:" + json.dumps(self.config, sort_keys=True),
        )
        self.vae = None

    def _ensure(self):
        if self.vae is None:
            self.vae = load_component(
                self.path, "vae", device=self.device, dtype=self.dtype, config=self.config
            )
        return self.vae

    @torch.no_grad()
    def encode(self, pixels):
        with torch.autocast(device_type=self.device.type, enabled=False):
            raw = self._ensure().encode(pixels.to(self.device, self.dtype)).latent_dist.sample()
            return (raw - self.shift) * self.scale

    @torch.no_grad()
    def decode(self, latents):
        with torch.autocast(device_type=self.device.type, enabled=False):
            raw = latents.to(self.device, self.dtype) / self.scale + self.shift
            return self._ensure().decode(raw).sample.float().clamp(-1, 1)

    def to(self, device):
        self.device = torch.device(device)
        if self.vae is not None:
            self.vae.to(self.device)

    def unload(self):
        self.vae = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
