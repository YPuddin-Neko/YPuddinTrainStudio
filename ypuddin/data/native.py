"""Model-independent native sizes and bounded, heterogeneous logical batches.

Unlike sequence packing, this path does not change the model's attention operator.
One logical image batch is evaluated as homogeneous microbatches; the trainer
weights their gradients by image count. No image padding or repeated tail samples.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .sampler import BucketBatchSampler


@dataclass(frozen=True)
class NativeSize:
    width: int
    height: int
    scale: float

    @property
    def downscaled(self) -> bool:
        return self.scale < 1.0 - 1e-9


def native_size(
    width: int,
    height: int,
    *,
    align: int,
    max_pixels: int,
    max_side: int,
    overflow: str = "downscale",
) -> NativeSize:
    """Preserve source scale unless a budget is exceeded, then align by cropping.

    Never enlarge, pad, distort the aspect ratio, or silently drop an oversized
    image. The caller applies resize-to-cover plus an alignment-only centre crop.
    Pixel budgets bound tensor geometry, not total model/GPU memory.
    """
    if min(width, height) < align:
        raise ValueError(f"image {width}x{height} is smaller than model alignment {align}")
    if max_pixels < align * align or max_side < align:
        raise ValueError("native resolution budget is smaller than model alignment")
    aligned_w, aligned_h = width // align * align, height // align * align
    if aligned_w * aligned_h <= max_pixels and max(aligned_w, aligned_h) <= max_side:
        return NativeSize(aligned_w, aligned_h, 1.0)
    if overflow == "error":
        raise ValueError(
            f"image {width}x{height} exceeds native budget {max_pixels} pixels / {max_side}px side"
        )
    scale = min(1.0, math.sqrt(max_pixels / (width * height)), max_side / width, max_side / height)
    target_w, target_h = int(width * scale) // align * align, int(height * scale) // align * align
    if min(target_w, target_h) < align:
        raise ValueError(
            f"image {width}x{height} cannot fit the native budget without enlarging its short side"
        )
    return NativeSize(target_w, target_h, scale)


class NativeBatchSampler(BucketBatchSampler):
    """Shuffle each image once, preserving a fixed logical batch size across shapes."""

    def plan(self, epoch: int | None = None) -> list[list[int]]:
        epoch = self.epoch if epoch is None else epoch
        if self._plan_cache and self._plan_cache[0] == epoch:
            return self._plan_cache[1]
        indices = list(range(len(self.bucket_of)))
        random.Random(self.seed * 1_000_003 + epoch).shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        if self.world_size > 1:
            usable = len(batches) - len(batches) % self.world_size
            batches = batches[self.rank : usable : self.world_size]
        self._plan_cache = epoch, batches
        return batches


def microbatch_indices(shapes: Sequence[tuple[int, int]], max_pixels: int) -> list[list[int]]:
    """Same-size groups fit within the configured simultaneous pixel budget."""
    groups: dict[tuple[int, int], list[int]] = {}
    for i, shape in enumerate(shapes):
        if shape[0] * shape[1] > max_pixels:
            raise ValueError(f"native image {shape} exceeds the microbatch pixel budget")
        groups.setdefault(shape, []).append(i)
    result = []
    for (width, height), indices in groups.items():
        capacity = max_pixels // (width * height)
        result.extend(indices[i : i + capacity] for i in range(0, len(indices), capacity))
    return result


def collate_native(samples: list[dict[str, Any]], *, max_pixels: int) -> dict[str, Any]:
    from .dataset import collate

    groups = microbatch_indices([sample["bucket"] for sample in samples], max_pixels)
    return {
        "caption": [sample["caption"] for sample in samples],
        "microbatches": [collate([samples[i] for i in group]) for group in groups],
    }
