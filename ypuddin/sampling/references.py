"""Single-reference preprocessing shared by previews and model tests."""

from __future__ import annotations

from pathlib import Path

import torch

from ypuddin.data.images import load_rgb, pil_to_tensor, to_padded


def validate_reference_images(family: str, task: str, paths: list[str]) -> None:
    if len(paths) > 1:
        raise ValueError("当前仅支持一张参考图。")
    if paths and family not in {"flux2", "flux3"}:
        raise ValueError("当前模型不支持参考图编辑。")
    if task == "edit" and len(paths) != 1:
        raise ValueError("编辑任务需要一张参考图。")


def fit_reference_image(path: str | Path, width: int, height: int):
    image, alpha = load_rgb(str(path))
    try:
        return to_padded(image, width, height)
    finally:
        image.close()
        if alpha is not None:
            alpha.close()


@torch.no_grad()
def encode_reference_latents(loaded, reference_images: list[str], *, width: int, height: int,
                             device, dtype) -> torch.Tensor | None:
    """Encode one clean reference; the caller controls subsequent VAE residency."""
    if not reference_images:
        return None
    if len(reference_images) != 1:
        raise ValueError("当前仅支持一张参考图。")
    with fit_reference_image(reference_images[0], width, height) as image:
        pixels = pil_to_tensor(image).unsqueeze(0).to(device)
    loaded.latent.to(device)
    return loaded.latent.encode(pixels).to(device=device, dtype=dtype)


def reference_for_batch(reference_latents: torch.Tensor | None, batch_size: int) -> torch.Tensor | None:
    if reference_latents is None or len(reference_latents) == batch_size:
        return reference_latents
    if batch_size == 2 * len(reference_latents):
        return torch.cat([reference_latents, reference_latents])
    raise ValueError("参考图批次与目标图批次不符。")
