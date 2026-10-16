"""Batched FP32 FLUX.3 image encoding with frozen normalization statistics."""

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import Tensor

from ypuddin.models.base import LatentPipeline
from ypuddin.models.fingerprints import content_fingerprint
from ypuddin.models.memory import release_model_memory

from .autoencoder import AutoEncoder


class Flux3Latent(LatentPipeline):
    channels, stride = 128, 16

    def __init__(self, path: str | Path, *, device="cpu"):
        self.path = self.validate_path(path)
        self.device, self.dtype = torch.device(device), torch.float32
        self.vae = None
        self.fingerprint = content_fingerprint(
            [self.path], namespace="flux3-vae-mean-fp32-patch2-bn1e-4-v1",
        )

    @staticmethod
    def validate_path(path: str | Path) -> Path:
        path = Path(path).expanduser()
        path = path / "ae.safetensors" if path.is_dir() else path
        if path.suffix != ".safetensors" or not path.is_file():
            raise ValueError("FLUX.3 requires a local ae.safetensors file")
        with torch.device("meta"):
            expected = AutoEncoder().state_dict()
        with safe_open(str(path), framework="pt") as tensors:
            keys = set(tensors.keys())
            if keys != set(expected):
                missing, unexpected = sorted(set(expected) - keys), sorted(keys - set(expected))
                raise ValueError(f"FLUX.3 VAE weights do not match: missing={missing}, unexpected={unexpected}")
            for key, value in expected.items():
                tensor = tensors.get_slice(key)
                if tuple(tensor.get_shape()) != tuple(value.shape):
                    raise ValueError(f"FLUX.3 VAE weight shape does not match: {key}")
                allowed = {"F32", "F16", "BF16", "F64"} if value.is_floating_point() else {"I64"}
                if tensor.get_dtype() not in allowed:
                    raise ValueError(f"FLUX.3 VAE weight dtype is unsupported: {key}")
        return path

    def _ensure(self):
        if self.vae is None:
            with torch.device("meta"):
                model = AutoEncoder()
            state = load_file(str(self.path), device="cpu")
            model.load_state_dict(state, strict=True, assign=True)
            self.vae = model.to(device=self.device, dtype=self.dtype).requires_grad_(False).eval()
        return self.vae

    def to(self, device):
        self.device = torch.device(device)
        if self.vae is not None:
            self.vae.to(self.device)

    def unload(self):
        if self.vae is not None:
            self.vae = None
            release_model_memory(self.device)

    @torch.no_grad()
    def encode(self, pixels: Tensor) -> Tensor:
        if pixels.ndim != 4 or pixels.shape[1] != 3 or pixels.shape[0] < 1:
            raise ValueError("FLUX.3 VAE expects pixels shaped [batch, 3, height, width]")
        if min(pixels.shape[2:]) < self.stride or any(side % self.stride for side in pixels.shape[2:]):
            raise ValueError("FLUX.3 image dimensions must be positive multiples of 16")
        with torch.autocast(self.device.type, enabled=False):
            model = self._ensure()
            mean, _ = model.encoder(pixels.to(self.device, self.dtype)).chunk(2, dim=1)
            batch, channels, height, width = mean.shape
            z = mean.reshape(batch, channels, height // 2, 2, width // 2, 2)
            z = z.permute(0, 1, 3, 5, 2, 4).reshape(batch, self.channels, height // 2, width // 2)
            return model.bn(z)

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        if latents.ndim != 4 or latents.shape[1] != self.channels or min(latents.shape) < 1:
            raise ValueError("FLUX.3 VAE expects latents shaped [batch, 128, height, width]")
        with torch.autocast(self.device.type, enabled=False):
            model = self._ensure()
            z = latents.to(self.device, self.dtype)
            batch, _, height, width = z.shape
            z = z * (model.bn.running_var[None, :, None, None] + model.bn_eps).sqrt()
            z = z + model.bn.running_mean[None, :, None, None]
            z = z.reshape(batch, self.channels // 4, 2, 2, height, width)
            z = z.permute(0, 1, 4, 2, 5, 3).reshape(batch, self.channels // 4, height * 2, width * 2)
            return model.decoder(z)
