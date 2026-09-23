"""Aspect-ratio bucketing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

# Changing bucket membership or selection changes sample order and geometry.
# Include this identity in bucket-mode resume fingerprints, never latent keys.
BUCKET_POLICY = "complete-grid-pixel-fit-v2"


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
    """Complete area-constrained grids, ranked by the actual image transform."""

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
        if align <= 0 or any(r <= 0 for r in self.resolutions):
            raise ValueError("resolution and alignment must be positive")
        if not math.isfinite(aspect_ratio_limit) or aspect_ratio_limit < 1:
            raise ValueError("aspect ratio limit must be finite and at least 1")
        if not math.isfinite(area_tolerance) or not 0 <= area_tolerance < 1:
            raise ValueError("area tolerance must be finite and within [0, 1)")
        self.align = align
        # Buckets live on a coarser grid than the alignment (64 px by default, like most trainers)
        # so a 1024 base yields ~40 buckets instead of hundreds of near-duplicates.
        step = max(align, 64) if step is None else step
        if step <= 0 or step % align != 0:
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
        # Derive bounds from the whole area band, including its upper edge.
        # Enumerate each feasible height interval, not just neighbours of base²/w.
        # This visits all legal grid points without scanning a full square grid.
        min_side = max(a, math.floor(math.sqrt(lo / self.ar_limit) / a) * a)
        max_side = math.ceil(math.sqrt(hi * self.ar_limit) / a) * a
        out: dict[tuple[int, int], Bucket] = {}
        for w in range(min_side, max_side + 1, a):
            low_h = max(a, math.floor(max(lo / w, w / self.ar_limit) / a) * a)
            high_h = math.ceil(min(hi / w, w * self.ar_limit) / a) * a
            for hh in range(low_h, high_h + 1, a):
                if not (lo <= w * hh <= hi):
                    continue
                if max(w / hh, hh / w) > self.ar_limit + 1e-9:
                    continue
                out[(w, hh)] = Bucket(w, hh, base)
        if not out:
            # Fallback for resolutions smaller than the bucket step,
            # or an area band with no aligned point. Do not add an out-of-band
            # square when legal buckets already exist.
            square = max(a, int(round(base / a)) * a)
            out[(square, square)] = Bucket(square, square, base)
        return sorted(out.values(), key=lambda b: (b.aspect, b.area, b.width, b.height))

    def all_buckets(self) -> list[Bucket]:
        return [b for r in self.resolutions for b in self.buckets[r]]

    def assign(self, width: int, height: int, base: int, *, image_fit: str = "crop") -> Bucket:
        """Minimize real crop/padding, then unnecessary scaling, then area deviation.

        Ranking uses the same integer geometry as RGB/masks and the data plan.
        """
        if min(width, height) <= 0 or image_fit not in {"crop", "pad"}:
            raise ValueError("positive image dimensions and crop/pad image fit are required")
        if self.no_upscale and image_fit == "crop" and min(width, height) < self.align:
            raise ValueError("原图短边小于模型对齐尺寸，无法在不放大的同时裁切填满；请使用保留完整画面")
        cands = self.buckets[base]
        if self.no_upscale:
            # Rank AFTER the same no-upscale transformation for every candidate;
            # floor alignment can change which shape best preserves this image.
            shapes = set()
            for b in cands:
                scale = min(Fraction(1), Fraction(width, b.width), Fraction(height, b.height))
                w = max(self.align, (b.width * scale // self.align) * self.align)
                h = max(self.align, (b.height * scale // self.align) * self.align)
                shapes.add((w, h))
            cands = [Bucket(w, h, base) for w, h in sorted(shapes)]

        def score(b: Bucket):
            if image_fit == "pad":
                rw, rh, *_ = fit_pad(
                    width, height, b.width, b.height, max_scale=1 if self.no_upscale else None
                )
                loss = Fraction(b.area - rw * rh, b.area)
            else:
                rw, rh, *_ = fit_crop(width, height, b.width, b.height)
                loss = Fraction(rw * rh - b.area, rw * rh)
            # Equal geometry: keep more original sampling resolution, then avoid
            # enlargement. Fractions avoid orientation-dependent float tie breaks.
            retained = Fraction(min(width, rw) * min(height, rh), width * height)
            enlarged = Fraction(max(width, rw) * max(height, rh), width * height)
            oriented = (b.width, b.height) if width >= height else (b.height, b.width)
            return loss, -retained, enlarged, abs(b.area - base * base), *oriented

        return min(cands, key=score)

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
