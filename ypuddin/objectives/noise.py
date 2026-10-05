"""Training noise shared by DDPM and rectified-flow objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ypuddin.config import ObjectiveConfig


def random_normal_like(tensor: Tensor, generator: torch.Generator | None) -> Tensor:
    """Generate on the RNG's device, then transfer; the trainer's CPU RNG works on CUDA/MPS too."""
    device = generator.device if generator is not None else tensor.device
    return torch.randn(tensor.shape, generator=generator, device=device, dtype=tensor.dtype).to(tensor.device)


def pyramid_noise(
    noise: Tensor, *, iterations: int, discount: float, generator: torch.Generator | None
) -> Tensor:
    """sd-scripts' bilinear noise pyramid, using the checkpointed training RNG."""
    height, width = noise.shape[-2:]
    dtype = noise.dtype
    # Accumulate and normalize low-precision latents in fp32.
    pyramid = noise.float() if dtype in (torch.float16, torch.bfloat16) else noise
    rng_device = generator.device if generator is not None else noise.device
    for level in range(iterations):
        ratio = 2 + 2 * torch.rand((), generator=generator, device=rng_device).item()
        scaled_height = max(1, int(height / ratio**level))
        scaled_width = max(1, int(width / ratio**level))
        layer = random_normal_like(pyramid[:, :, :scaled_height, :scaled_width], generator)
        pyramid = pyramid + F.interpolate(
            layer, size=(height, width), mode="bilinear", align_corners=False
        ) * discount**level
        if scaled_height == 1 or scaled_width == 1:
            break
    if pyramid.numel() > 1:
        deviation = pyramid.std()
        # Scalar and constant fields have no usable sample variance.
        divisor = torch.where(deviation > torch.finfo(pyramid.dtype).eps, deviation, 1)
        pyramid = pyramid / divisor
    return pyramid.to(dtype)


def sample_training_noise(
    x0: Tensor, cfg: ObjectiveConfig, generator: torch.Generator | None
) -> Tensor:
    if (cfg.noise_offset or cfg.multires_noise_iterations) and (
        x0.ndim != 4 or min(x0.shape) < 1
    ):
        raise ValueError("Noise offset and multiresolution noise require nonempty NCHW latents")
    noise = random_normal_like(x0, generator)
    if cfg.noise_offset:
        noise = noise + cfg.noise_offset * random_normal_like(noise[:, :, :1, :1], generator)
    elif cfg.multires_noise_iterations:
        noise = pyramid_noise(
            noise,
            iterations=cfg.multires_noise_iterations,
            discount=cfg.multires_noise_discount,
            generator=generator,
        )
    return noise
