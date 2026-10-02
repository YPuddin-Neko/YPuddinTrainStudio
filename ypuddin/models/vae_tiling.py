"""Spatial VAE tiles with overlap blending and one image resident per invocation."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

VAE_TILE_PIXELS = 512
CACHE_TILE_PIXELS = 1024
TILE_OVERLAP_PIXELS = 128
CACHE_TILE_THRESHOLD = 4 * 1024 * 1024


def tiled_latent_fingerprint(fingerprint: str, *, vae_tiling: bool, cache_encode_tiled: bool) -> str:
    if vae_tiling:
        fingerprint += f"|vae-spatial-v1:{VAE_TILE_PIXELS}:{TILE_OVERLAP_PIXELS}"
    if cache_encode_tiled:
        fingerprint += f"|cache-spatial-v1:{CACHE_TILE_PIXELS}:{TILE_OVERLAP_PIXELS}:gt{CACHE_TILE_THRESHOLD}"
    return fingerprint


def _starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, tile - overlap))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def _feather(length: int, before: int, after: int) -> Tensor:
    weights = torch.ones(length, dtype=torch.float32)
    if before:
        weights[:before] *= torch.arange(1, before + 1, dtype=torch.float32) / (before + 1)
    if after:
        weights[-after:] *= torch.arange(after, 0, -1, dtype=torch.float32) / (after + 1)
    return weights


@torch.no_grad()
def spatial_tiled_apply(
    value: Tensor,
    operation: Callable[[Tensor], Tensor],
    *,
    tile: int,
    overlap: int,
    scale_num: int = 1,
    scale_den: int = 1,
) -> Tensor:
    """Apply a spatial encoder/decoder and blend in FP32 on CPU to bound GPU memory.

    Tile dimensions use the input's units; the scale converts their coordinates
    to output units. Boundary tiles retain their full size. Images that already
    fit in one tile bypass blending and retain the operation's exact output.
    """
    if value.ndim not in (4, 5) or not value.shape[0]:
        raise ValueError("VAE tiles require a nonempty batch of images or videos")
    if tile <= overlap or overlap < 0 or scale_num <= 0 or scale_den <= 0:
        raise ValueError("invalid VAE tile size, overlap or spatial scale")
    h, w = value.shape[-2:]
    if min(h, w) <= 0 or any(n % scale_den for n in (h, w, tile, overlap)):
        raise ValueError("VAE tile dimensions must align with the latent stride")
    ys, xs = _starts(h, tile, overlap), _starts(w, tile, overlap)
    def factor(n: int) -> int:
        return n * scale_num // scale_den

    def invoke(tile_value: Tensor) -> Tensor:
        with torch.autocast(device_type=value.device.type, enabled=False):
            return operation(tile_value)

    results = []
    output_device = None
    output_dtype = None
    for sample in value.split(1):
        if len(ys) == len(xs) == 1:
            result = invoke(sample)
            output_device, output_dtype = result.device, result.dtype
            results.append(result.cpu())
            continue
        accumulated = None
        coverage = torch.zeros(factor(h), factor(w), dtype=torch.float32)
        for yi, y in enumerate(ys):
            end_y = min(y + tile, h)
            height = factor(end_y - y)
            before_y = factor(max(0, ys[yi - 1] + tile - y)) if yi else 0
            after_y = factor(max(0, end_y - ys[yi + 1])) if yi + 1 < len(ys) else 0
            wy = _feather(height, before_y, after_y)
            for xi, x in enumerate(xs):
                end_x = min(x + tile, w)
                width = factor(end_x - x)
                before_x = factor(max(0, xs[xi - 1] + tile - x)) if xi else 0
                after_x = factor(max(0, end_x - xs[xi + 1])) if xi + 1 < len(xs) else 0
                patch = invoke(sample[..., y:end_y, x:end_x])
                if patch.shape[0] != 1 or patch.shape[-2:] != (height, width):
                    raise ValueError("VAE tile output does not match its spatial scale")
                output_device, output_dtype = patch.device, patch.dtype
                patch = patch.to(device="cpu", dtype=torch.float32)
                if accumulated is None:
                    accumulated = torch.zeros(*patch.shape[:-2], factor(h), factor(w), dtype=torch.float32)
                elif patch.shape[:-2] != accumulated.shape[:-2]:
                    raise ValueError("VAE tiles returned inconsistent channel or frame counts")
                mask = wy[:, None] * _feather(width, before_x, after_x)[None, :]
                oy, ox = factor(y), factor(x)
                accumulated[..., oy:oy + height, ox:ox + width] += patch * mask
                coverage[oy:oy + height, ox:ox + width] += mask
        accumulated.div_(coverage)
        results.append(accumulated.to(dtype=output_dtype))
    with torch.autocast(device_type="cpu", enabled=False):
        return torch.cat(results).to(device=output_device, dtype=output_dtype)
