"""Image / mask loading into tensors sized for a bucket."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor

from .buckets import fit_crop, fit_pad
from .image_metadata import alpha_channel


def load_rgb(path: str) -> tuple[Image.Image, Image.Image | None]:
    """Open an image, composite transparency on white; returns ``(rgb, alpha_or_None)``."""
    with Image.open(path) as image:
        im = ImageOps.exif_transpose(image)
        alpha = alpha_channel(im)
        if alpha is not None:
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = im.convert("RGBA")
            im.putalpha(alpha)
            im = Image.alpha_composite(bg, im)
        return im.convert("RGB"), alpha


def load_alpha(path: str) -> Image.Image | None:
    """Load just the current alpha channel, without compositing or constructing RGB tensors."""
    with Image.open(path) as image:
        im = ImageOps.exif_transpose(image)
        return alpha_channel(im)


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


def to_padded(
    im: Image.Image,
    width: int,
    height: int,
    *,
    max_scale: float | None = None,
    flip: bool = False,
    resample=Image.LANCZOS,
    mask: bool = False,
) -> Image.Image:
    """Resize the complete image then pad; RGB repeats edges, loss masks use zero."""
    rw, rh, left, top, right, bottom = fit_pad(im.width, im.height, width, height, max_scale=max_scale)
    resized = im if im.size == (rw, rh) else im.resize((rw, rh), resample=resample)
    if mask:
        result = Image.new("L", (width, height), 0)
        result.paste(resized, (left, top))
    else:
        array = np.asarray(resized)
        padding = [(top, height - bottom), (left, width - right)] + [(0, 0)] * (array.ndim - 2)
        result = Image.fromarray(np.pad(array, padding, mode="edge"))
    return ImageOps.mirror(result) if flip else result


def valid_image_mask(
    source_width: int, source_height: int, width: int, height: int, *, max_scale: float | None, flip: bool
) -> Tensor:
    """Whole-image validity, independent of the user's optional training mask."""
    _, _, left, top, right, bottom = fit_pad(source_width, source_height, width, height, max_scale=max_scale)
    result = torch.zeros((height, width), dtype=torch.float32)
    result[top:bottom, left:right] = 1
    return result.flip(-1) if flip else result


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
    image_fit: str = "crop",
    max_scale: float | None = None,
    source_size: tuple[int, int] | None = None,
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
    if image_fit == "pad" and source_size is not None and src.size != source_size:
        raise ValueError(f"mask dimensions {src.size} do not match image dimensions {source_size}")
    if image_fit == "pad":
        m = to_padded(src, width, height, max_scale=max_scale, flip=flip, resample=Image.BILINEAR, mask=True)
        return torch.from_numpy(np.asarray(m, dtype=np.float32) / 255.0)
    m = (
        to_bucket(src, width, height, flip=flip, resample=Image.BILINEAR)
        if native_scale is None
        else to_native(src, width, height, scale=native_scale, flip=flip, resample=Image.BILINEAR)
    )
    return torch.from_numpy(np.asarray(m, dtype=np.float32) / 255.0)
