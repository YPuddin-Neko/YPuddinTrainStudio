"""Search native pixel budgets without assuming memory is monotone in the budget."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from ypuddin.config import DatasetConfig
from ypuddin.data.buckets import Bucket
from ypuddin.data.dataset import DataLayout
from ypuddin.data.native import NativeSize, native_size


class NativeVramGeometry:
    """The selected training/validation images, grouped before candidate sizing."""

    def __init__(self, layout: DataLayout, dataset: DatasetConfig):
        self.layout, self.dataset = layout, dataset
        self.align = layout.bucket_manager.align
        self.originals: dict[tuple[int, int], dict[str, Any]] = {}
        for items, validation in ((layout.items, False), (layout.validation_items, True)):
            for item in items:
                key = (item.record.width, item.record.height)
                entry = self.originals.setdefault(key, {"items": 0, "images": set(), "validation": set()})
                entry["validation" if validation else "images"].add(item.record.path)
                entry["items"] += not validation
        # Each source canvas has a monotone size even when the memory estimate does not.
        self._plateaus: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}

    def size(self, original: tuple[int, int], pixels: int) -> NativeSize:
        return native_size(
            *original, align=self.align, max_pixels=pixels, max_side=self.dataset.native_max_side,
            overflow=self.dataset.native_overflow, image_fit=self.dataset.image_fit, auto_area=True,
        )

    def shapes(self, pixels: int) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], dict[str, Any]]]:
        counts: Counter = Counter()
        images: dict[tuple[int, int], dict[str, Any]] = {}
        for original, entry in self.originals.items():
            size = self.size(original, pixels)
            key = (size.width, size.height)
            if entry["items"]:
                counts[key] += entry["items"]
            target = images.setdefault(key, {"items": 0, "images": set(), "validation": set()})
            target["items"] += entry["items"]
            target["images"].update(entry["images"])
            target["validation"].update(entry["validation"])
        return dict(counts), images

    def interval_start(self, pixels: int) -> int:
        """Lowest budget with the same canvases and simultaneous image counts."""
        minimum = lower = self.align * self.align
        for original in self.originals:
            size = self.size(original, pixels)
            shape = (size.width, size.height)
            cached = self._plateaus.get(original)
            if cached is not None and cached[0] == shape:
                start = cached[1]
            else:
                low, high = minimum, pixels
                while low < high:
                    middle = (low + high) // 2
                    try:
                        candidate = self.size(original, middle)
                        same = (candidate.width, candidate.height) == shape
                    except ValueError:
                        same = False
                    if same:
                        high = middle
                    else:
                        low = middle + 1
                start = low
                self._plateaus[original] = (shape, start)
            lower = max(lower, start)
            area = shape[0] * shape[1]
            simultaneous = min(self.dataset.batch_size, max(1, pixels // area))
            lower = max(lower, simultaneous * area)
        return lower

    def resolved_layout(self, pixels: int) -> DataLayout:
        def resized(item):
            size = self.size((item.record.width, item.record.height), pixels)
            return replace(item, bucket=Bucket(size.width, size.height, 0), native_scale=size.scale)

        return replace(
            self.layout,
            items=[resized(item) for item in self.layout.items],
            validation_items=[resized(item) for item in self.layout.validation_items],
            native_max_pixels=pixels,
        )


def maximum_fitting_pixels(
    geometry: NativeVramGeometry,
    upper: int,
    evaluate: Callable[[int, dict, dict], float],
    budget_mb: float,
    *,
    training_lower_bound: Callable[[int, dict, dict], float] | None = None,
) -> int | None:
    """Visit constant-geometry/microbatch intervals from largest to smallest.

    Geometry jumps can reduce microbatch counts or split VAE encoding groups. A failed
    candidate therefore skips only its unchanged interval, never every larger resolution.
    """
    pixels = upper
    minimum = geometry.align * geometry.align
    try:
        counts, images = geometry.shapes(upper)
    except ValueError:
        return None
    if evaluate(upper, counts, images) <= budget_mb:
        return upper
    if training_lower_bound is not None:
        # One image's training activations grow with its canvas. This lower bound
        # excludes VAE grouping/tiling and cannot discard a fitting larger budget.
        low, high = minimum - 1, upper
        while low < high:
            middle = (low + high + 1) // 2
            try:
                counts, images = geometry.shapes(middle)
                fits_bound = training_lower_bound(middle, counts, images) <= budget_mb
            except ValueError:
                fits_bound = True  # Minimum-short-side errors occur below usable canvases.
            if fits_bound:
                low = middle
            else:
                high = middle - 1
        pixels = low
    while pixels >= minimum:
        try:
            counts, images = geometry.shapes(pixels)
        except ValueError:
            # Shrinking further cannot restore an image's minimum aligned short side.
            return None
        if pixels != upper and evaluate(pixels, counts, images) <= budget_mb:
            return pixels
        pixels = geometry.interval_start(pixels) - 1
    return None
