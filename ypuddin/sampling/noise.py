"""A seed's noise drawn the way ComfyUI or A1111 WebUI draws it, so a preview seed starts the same image there."""

from __future__ import annotations

import torch
from torch import Tensor

NOISE_SOURCES = ("comfyui", "a1111")


class SeedNoise:
    """The starting noise of a batch, and the fresh noise stochastic samplers add on the way.

    ``comfyui`` draws the whole batch's starting noise from ``generator`` (ComfyUI's ``prepare_noise``, the
    CPU generator seeded with the seed) and the later noise from a second generator with the same seed
    on the sampling device, seed + 1 on the CPU (its ``default_noise_sampler``).

    ``a1111`` follows A1111 WebUI with its default GPU random source: image i of the batch has its own
    generator seeded with seed + i on the GPU, on the CPU on Apple and CPU machines, and every later
    draw continues it.
    """

    def __init__(
        self, generator: torch.Generator | None, source: str = "comfyui", device: torch.device | str = "cpu"
    ) -> None:
        if source not in NOISE_SOURCES:
            raise ValueError(f"unknown noise source: {source}")
        if generator is None:
            generator = torch.Generator(device="cpu")
            generator.seed()
        self.generator, self.source, self.device = generator, source, torch.device(device)
        self.seed = generator.initial_seed()
        self._later: torch.Generator | None = None
        self._images: list[torch.Generator] = []

    def first(self, shape: tuple[int, ...]) -> Tensor:
        """The starting noise, FP32 on the sampling device."""
        if self.source == "comfyui":
            noise = torch.randn(
                shape, generator=self.generator, device=self.generator.device, dtype=torch.float32
            )
            return noise.to(self.device)
        where = self.device if self.device.type == "cuda" else torch.device("cpu")
        self._images = [torch.Generator(device=where).manual_seed(self.seed + i) for i in range(shape[0])]
        return self._per_image(shape, torch.float32)

    def next(self, shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> Tensor:
        """Noise for one stochastic step, on the sampling device."""
        if self.source == "a1111":
            return self._per_image(shape, dtype)
        if self._later is None:
            seed = self.seed + 1 if self.device.type == "cpu" else self.seed
            self._later = torch.Generator(device=self.device).manual_seed(seed)
        return torch.randn(shape, generator=self._later, device=self.device, dtype=dtype)

    def _per_image(self, shape: tuple[int, ...], dtype: torch.dtype) -> Tensor:
        if len(self._images) != shape[0]:
            raise ValueError("A1111 noise needs its starting noise drawn first")
        noise = [torch.randn(shape[1:], generator=g, device=g.device, dtype=dtype) for g in self._images]
        return torch.stack(noise).to(self.device)
