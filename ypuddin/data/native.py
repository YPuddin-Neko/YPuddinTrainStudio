"""Model-independent native sizes and bounded, heterogeneous logical batches.

Unlike sequence packing, this path does not change the model's attention operator.
One logical image batch is evaluated as homogeneous microbatches; the trainer
weights their gradients by image count. Legacy crop and whole-image padding use
separate geometry; neither path repeats tail samples to fill a batch.
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
    image_fit: str = "crop",
) -> NativeSize:
    """Preserve source scale unless a budget is exceeded, then align as requested.

    The legacy default floor-aligns with a centre crop. Whole-image mode ceil-aligns
    and excludes padding from loss. Neither enlarges the source or silently drops
    oversized images. Pixel budgets bound canvas geometry, not total model/GPU memory.
    """
    if image_fit == "pad":
        return _native_pad_size(width, height, align, max_pixels, max_side, overflow)
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


def _native_pad_size(
    width: int, height: int, align: int, max_pixels: int, max_side: int, overflow: str
) -> NativeSize:
    """Ceil-align the whole image, budgeting the canvas including its padding."""
    if min(width, height, align) <= 0:
        raise ValueError("native image dimensions and alignment must be positive")
    if max_pixels < align * align or max_side < align:
        raise ValueError("native resolution budget is smaller than model alignment")

    def canvas(scale: float) -> tuple[int, int]:
        return tuple(((max(1, round(side * scale)) + align - 1) // align) * align for side in (width, height))

    def fits(size: tuple[int, int]) -> bool:
        return size[0] * size[1] <= max_pixels and max(size) <= max_side

    original_canvas = canvas(1.0)
    if fits(original_canvas):
        return NativeSize(*original_canvas, 1.0)
    if overflow == "error":
        raise ValueError(
            f"image {width}x{height} including alignment padding exceeds native budget "
            f"{max_pixels} pixels / {max_side}px side"
        )
    # Canvas dimensions are monotone stair functions of scale. Search the greatest
    # supported scale; rounding/filling above is identical to the pixel transform.
    low, high = 0.0, 1.0
    for _ in range(60):
        middle = (low + high) / 2
        if fits(canvas(middle)):
            low = middle
        else:
            high = middle
    return NativeSize(*canvas(low), low)


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
