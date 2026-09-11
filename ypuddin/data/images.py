"""Image / mask loading into tensors sized for a bucket."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor

from .buckets import fit_crop


def load_rgb(path: str) -> tuple[Image.Image, Image.Image | None]:
    """Open an image, composite transparency on white; returns ``(rgb, alpha_or_None)``."""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    alpha = None
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        alpha = im.getchannel("A")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB"), alpha


def load_alpha(path: str) -> Image.Image | None:
    """Load just the current alpha channel, without compositing or constructing RGB tensors."""
    with Image.open(path) as image:
        im = ImageOps.exif_transpose(image)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            return im.convert("RGBA").getchannel("A")
    return None


def to_bucket(
    im: Image.Image, width: int, height: int, *, flip: bool = False, resample=Image.LANCZOS
) -> Image.Image:
    rw, rh, left, top, right, bottom = fit_crop(im.width, im.height, width, height)
    out = im.resize((rw, rh), resample=resample).crop((left, top, right, bottom))
    if flip:
        out = ImageOps.mirror(out)
    return out


def to_native(
    im: Image.Image, width: int, height: int, *, scale: float, flip: bool = False, resample=Image.LANCZOS
) -> Image.Image:
    """Alignment-only crop at scale 1, or proportional downscale then crop.

    A native image already within budget must not be resampled just because both
    sides need alignment. RGB, alpha and sidecar masks use the same transform.
    """
    resized = (max(width, round(im.width * scale)), max(height, round(im.height * scale)))
    image = im if resized == im.size else im.resize(resized, resample=resample)
    left, top = (image.width - width) // 2, (image.height - height) // 2
    image = image.crop((left, top, left + width, top + height))
    return ImageOps.mirror(image) if flip else image


def pil_to_tensor(im: Image.Image) -> Tensor:
    """``(3, H, W)`` float32 in ``[-1, 1]``."""
    arr = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def load_mask(
    path: str | None,
    alpha: Image.Image | None,
    width: int,
    height: int,
    *,
    flip: bool = False,
    native_scale: float | None = None,
) -> Tensor | None:
    """Loss mask ``(H, W)`` in ``[0, 1]`` from a sidecar (grayscale) or the alpha channel."""
    src: Image.Image | None = None
    if path:
        with Image.open(path) as image:
            src = ImageOps.exif_transpose(image).convert("L")
    elif alpha is not None:
        src = alpha
    if src is None:
        return None
    m = (
        to_bucket(src, width, height, flip=flip, resample=Image.BILINEAR)
        if native_scale is None
        else to_native(src, width, height, scale=native_scale, flip=flip, resample=Image.BILINEAR)
    )
    return torch.from_numpy(np.asarray(m, dtype=np.float32) / 255.0)
