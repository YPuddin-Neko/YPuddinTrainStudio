"""Euler sampler for rectified flow with classifier-free guidance (used for training previews)."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from ypuddin.objectives import mobius_shift

from .noise import SeedNoise


def flow_schedule(steps: int, shift: float = 1.0, device: torch.device | str = "cpu") -> Tensor:
    """Descending timesteps ``t_0 = 1 > ... > t_steps = 0`` with an optional Möbius shift."""
    t = torch.linspace(1.0, 0.0, steps + 1, device=device)
    if shift != 1.0:
        t = mobius_shift(t, shift)
        t[0], t[-1] = 1.0, 0.0
    return t


def comfyui_denoised(predict, predict_uncond, x: Tensor, sigma: Tensor, cfg: float) -> Tensor:
    """RF model output and CFG in ComfyUI's denoised-space operation order."""
    batch_time = sigma.expand(x.shape[0]).float()
    denoised = x - predict(x, batch_time).to(x.dtype) * sigma
    if cfg != 1.0 and predict_uncond is not None:
        uncond = x - predict_uncond(x, batch_time).to(x.dtype) * sigma
        denoised = uncond + (denoised - uncond) * cfg
    return denoised


@torch.no_grad()
def euler_sample(
    predict: Callable[[Tensor, Tensor], Tensor],
    shape: tuple[int, ...],
    *,
    steps: int,
    shift: float = 1.0,
    cfg: float = 1.0,
    predict_uncond: Callable[[Tensor, Tensor], Tensor] | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    on_step: Callable[[int, int], None] | None = None,
    noise: SeedNoise | None = None,
) -> Tensor:
    """``predict(x_t, t) -> v``; integrates ``dx/dt = v`` from ``t=1`` (noise) down to ``t=0``.

    ``on_step(done, total)`` is called after every integration step (progress reporting). ``noise`` gives
    the starting noise; without it ``generator`` does, as in ComfyUI.
    """
    start = noise if noise is not None else SeedNoise(generator, device=device)
    x = start.first(shape).to(device=device, dtype=dtype)
    ts = flow_schedule(steps, shift, device=device)
    for i in range(steps):
        t_cur, t_next = ts[i], ts[i + 1]
        if start.source == "comfyui":
            denoised = comfyui_denoised(predict, predict_uncond, x, t_cur, cfg)
            direction = (x - denoised) / t_cur
            x = x + direction * (t_next - t_cur)
            if on_step is not None:
                on_step(i + 1, steps)
            continue
        tb = t_cur.expand(shape[0]).float()
        v = predict(x, tb)
        if cfg != 1.0 and predict_uncond is not None:
            v_u = predict_uncond(x, tb)
            v = v_u + cfg * (v - v_u)
        x = x + (t_next - t_cur).to(x.dtype) * v.to(x.dtype)
        if on_step is not None:
            on_step(i + 1, steps)
    return x
