"""Shared rectified-flow preview dispatch; sampler and noise schedule are independent."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

from .euler import euler_sample, flow_schedule
from .noise import SeedNoise

SAMPLERS = ("euler", "euler_ancestral", "heun", "er_sde")
SCHEDULERS = ("uniform", "simple", "sgm_uniform", "normal")


def noise_schedule(
    steps: int,
    shift: float = 1.0,
    scheduler: str = "uniform",
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Flow times, including terminal zero. SGM/normal use the discrete-flow sigma endpoints.

    ``uniform`` preserves the original continuous schedule. ``simple`` subsamples a
    1000-point discrete grid. SGM excludes the low endpoint, while normal includes it.
    The latter two follow the flow timestep convention (timestep does not undo shift).
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 1000:
        raise ValueError("sampling steps must be an integer between 1 and 1000")
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("sampling shift must be finite and positive")
    if scheduler not in SCHEDULERS:
        raise ValueError(f"unknown sampling scheduler: {scheduler}")
    if scheduler == "uniform" and dtype == torch.float32:
        # Keep existing Euler previews identical, including float32 rounding.
        result = flow_schedule(steps, shift, device=device)
    else:
        if scheduler == "uniform":
            times = torch.linspace(1, 0, steps + 1, dtype=torch.float64)
        elif scheduler == "simple":
            indices = torch.floor(torch.arange(steps, dtype=torch.float64) * (1000 / steps))
            times = torch.cat(((1000 - indices) / 1000, torch.zeros(1, dtype=torch.float64)))
        else:
            low = shift / (999 + shift)
            count = steps + 1 if scheduler == "sgm_uniform" else steps
            times = torch.linspace(1, low, count, dtype=torch.float64)
            if scheduler == "sgm_uniform":
                times = times[:-1]
            times = torch.cat((times, torch.zeros(1, dtype=torch.float64)))
        # Algebraically equivalent to shift*t/(1+(shift-1)*t), stable for large shift.
        result = torch.where(times == 0, 0, times / (times + (1 - times) / shift))
        result[0], result[-1] = 1, 0
        result = result.to(device=device, dtype=dtype)
    if not torch.isfinite(result).all() or not (result[:-1] > result[1:]).all():
        raise ValueError("sampling shift is too extreme to form distinct finite timesteps")
    return result


@torch.no_grad()
def sample(
    predict: Callable[[Tensor, Tensor], Tensor],
    shape: tuple[int, ...],
    *,
    steps: int,
    shift: float = 1.0,
    cfg: float = 1.0,
    sampler: str = "euler",
    scheduler: str = "uniform",
    er_sde_order: int = 3,
    er_sde_s_noise: float = 1.0,
    predict_uncond: Callable[[Tensor, Tensor], Tensor] | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    on_step: Callable[[int, int], None] | None = None,
    noise: str = "comfyui",
) -> Tensor:
    """Generate a preview from velocity predictions using the selected integration method.

    ``noise`` names whose way the seed in ``generator`` becomes noise (``SeedNoise``).
    """
    if sampler not in SAMPLERS:
        raise ValueError(f"unknown sampler: {sampler}")
    if not math.isfinite(cfg) or cfg < 0:
        raise ValueError("sampling cfg must be finite and nonnegative")
    if er_sde_order not in (1, 2, 3) or isinstance(er_sde_order, bool):
        raise ValueError("ER-SDE order must be 1, 2 or 3")
    if not math.isfinite(er_sde_s_noise) or not 0 <= er_sde_s_noise <= 1:
        raise ValueError("ER-SDE noise strength must be finite and between 0 and 1")
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("sampling shape must contain positive dimensions")
    if generator is None:
        generator = torch.Generator(device="cpu")
        generator.seed()
    seeded = SeedNoise(generator, noise, device)
    times = noise_schedule(
        steps, shift, scheduler, dtype=torch.float64 if sampler == "er_sde" else torch.float32
    )
    if sampler == "er_sde":
        from .er_sde import er_sde_sample

        # Finite SNR is required by the solver. Do not alter the other samplers' t=1 start.
        start = 0.9999 / (0.9999 + 0.0001 / shift)
        times[0] = start
        return er_sde_sample(
            predict,
            shape,
            sigmas=times,
            cfg=cfg,
            predict_uncond=predict_uncond,
            generator=generator,
            device=device,
            dtype=dtype,
            on_step=on_step,
            max_order=er_sde_order,
            s_noise=er_sde_s_noise,
            noise=seeded,
        )
    if sampler == "euler" and scheduler == "uniform":
        return euler_sample(
            predict,
            shape,
            steps=steps,
            shift=shift,
            cfg=cfg,
            predict_uncond=predict_uncond,
            generator=generator,
            device=device,
            dtype=dtype,
            on_step=on_step,
            noise=seeded,
        )
    times = times.to(device)
    x = seeded.first(shape).to(device=device, dtype=dtype)

    def velocity(latent: Tensor, time: Tensor) -> Tensor:
        batch_time = time.expand(shape[0]).float()
        v = predict(latent, batch_time)
        if cfg != 1 and predict_uncond is not None:
            uncond = predict_uncond(latent, batch_time)
            v = uncond + cfg * (v - uncond)
        return v.to(dtype)

    for index, (current, following) in enumerate(zip(times[:-1], times[1:], strict=True)):
        if sampler == "euler_ancestral":
            x = _ancestral_step(x, current, following, velocity(x, current), seeded)
            if on_step is not None:
                on_step(index + 1, steps)
            continue
        delta = (following - current).to(dtype)
        first = velocity(x, current)
        proposed = x + delta * first
        if sampler == "heun" and index < steps - 1:
            x = x + delta * (first + velocity(proposed, following)) / 2
        else:
            x = proposed
        if on_step is not None:
            on_step(index + 1, steps)
    return x


def _ancestral_step(x: Tensor, current: Tensor, following: Tensor, v: Tensor, seeded: SeedNoise) -> Tensor:
    """One rectified-flow Euler ancestral step (eta 1), ComfyUI's ``sample_euler_ancestral_RF``: step down
    to ``following² / current`` toward the denoised estimate, then add fresh noise back up to ``following``."""
    denoised = x - current * v
    if float(following) == 0:
        return denoised
    # ComfyUI's downstep ratio at eta 1, kept in its order of float operations.
    down = following * (1 + (following / current - 1))
    ratio = down / current
    x = ratio * x + (1 - ratio) * denoised
    alpha_next, alpha_down = 1 - following, 1 - down
    renoise = (following.square() - down.square() * alpha_next.square() / alpha_down.square()).sqrt()
    return (alpha_next / alpha_down) * x + seeded.next(tuple(x.shape)).to(x.dtype) * renoise
