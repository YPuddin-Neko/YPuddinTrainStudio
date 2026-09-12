"""Aspect-ratio bucketing."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Bucket:
    width: int
    height: int
    base: int  # base resolution this bucket was generated for

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect(self) -> float:
        return self.width / self.height

    @property
    def key(self) -> tuple[int, int]:
        return (self.width, self.height)


class BucketManager:
    """Generates buckets for each base resolution and assigns images to the closest aspect ratio."""

    def __init__(
        self,
        resolutions: list[int],
        *,
        align: int = 16,
        step: int | None = None,
        aspect_ratio_limit: float = 2.0,
        area_tolerance: float = 0.10,
        no_upscale: bool = False,
    ) -> None:
        self.resolutions = sorted(set(resolutions), reverse=True)
        self.align = align
        # Buckets live on a coarser grid than the alignment (64 px by default, like most trainers)
        # so a 1024 base yields ~40 buckets instead of hundreds of near-duplicates.
        step = max(align, 64) if step is None else step
        if step % align != 0:
            raise ValueError(f"bucket step {step} must be a multiple of align {align}")
        self.step = step
        self.ar_limit = aspect_ratio_limit
        self.area_tol = area_tolerance
        self.no_upscale = no_upscale
        self.buckets: dict[int, list[Bucket]] = {r: self._generate(r) for r in self.resolutions}

    def _generate(self, base: int) -> list[Bucket]:
        area = base * base
        lo, hi = area * (1 - self.area_tol), area * (1 + self.area_tol)
        a = self.step
        min_side = max(a, int(math.floor(base / math.sqrt(self.ar_limit) / a)) * a)
        max_side = int(math.ceil(base * math.sqrt(self.ar_limit) / a)) * a
        out: dict[tuple[int, int], Bucket] = {}
        for w in range(min_side, max_side + 1, a):
            # pick the h that brings the area closest to base² and check tolerance / aspect
            h = int(round(area / w / a)) * a
            for hh in (h - a, h, h + a):
                if hh < a:
                    continue
                if not (lo <= w * hh <= hi):
                    continue
                if max(w / hh, hh / w) > self.ar_limit + 1e-9:
                    continue
                out[(w, hh)] = Bucket(w, hh, base)
        square = max(a, int(round(base / a)) * a)
        out.setdefault((square, square), Bucket(square, square, base))
        return sorted(out.values(), key=lambda b: b.aspect)

    def all_buckets(self) -> list[Bucket]:
        return [b for r in self.resolutions for b in self.buckets[r]]

    def assign(self, width: int, height: int, base: int) -> Bucket:
        """Closest-aspect bucket for ``base``; with ``no_upscale`` small images get a shrunk bucket."""
        ar = width / height
        cands = self.buckets[base]
        best = min(cands, key=lambda b: (abs(math.log(b.aspect) - math.log(ar)), abs(b.area - base * base)))
        if self.no_upscale and (width < best.width or height < best.height):
            scale = min(width / best.width, height / best.height)
            w = max(self.align, int(best.width * scale // self.align) * self.align)
            h = max(self.align, int(best.height * scale // self.align) * self.align)
            return Bucket(w, h, base)
        return best

    def describe(self) -> list[dict]:
        return [
            {"base": b.base, "w": b.width, "h": b.height, "aspect": round(b.aspect, 4)}
            for b in self.all_buckets()
        ]


def fit_crop(src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int, int, int, int, int]:
    """Resize-to-cover then center-crop. Returns ``(resize_w, resize_h, left, top, right, bottom)``."""
    scale = max(dst_w / src_w, dst_h / src_h)
    rw, rh = max(dst_w, int(round(src_w * scale))), max(dst_h, int(round(src_h * scale)))
    left = (rw - dst_w) // 2
    top = (rh - dst_h) // 2
    return rw, rh, left, top, left + dst_w, top + dst_h


def fit_pad(
    src_w: int, src_h: int, dst_w: int, dst_h: int, *, max_scale: float | None = None
) -> tuple[int, int, int, int, int, int]:
    """Contain the complete image; return resized size and its rectangle on the canvas.

    The optional scale ceiling preserves small/native images without upscaling. Integer
    rounding is shared by RGB, masks, cache identity and planning, never inferred twice.
    """
    if min(src_w, src_h, dst_w, dst_h) <= 0:
        raise ValueError("image and canvas dimensions must be positive")
    scale = min(dst_w / src_w, dst_h / src_h)
    if max_scale is not None:
        if not math.isfinite(max_scale) or max_scale <= 0:
            raise ValueError("maximum image scale must be positive and finite")
        # A native planner budgets the rounded pixel dimensions. Preserve its scale
        # when rounding fits, rather than shrinking again at a half-pixel boundary.
        scale = (
            max_scale
            if round(src_w * max_scale) <= dst_w and round(src_h * max_scale) <= dst_h
            else min(scale, max_scale)
        )
    rw = max(1, min(dst_w, round(src_w * scale)))
    rh = max(1, min(dst_h, round(src_h * scale)))
    left, top = (dst_w - rw) // 2, (dst_h - rh) // 2
    return rw, rh, left, top, left + rw, top + rh
