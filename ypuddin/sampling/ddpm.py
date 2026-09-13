"""Euler/Heun previews for variance-preserving DDPM denoisers.

The solver state is y = x0 + sigma * noise, while predict receives the VP input
x_t = y / sqrt(1 + sigma**2). Its unit time is continuous_index / N. SDXL's
forward(inference=True) must use sampling_timestep, whereas training/validation
use objectives.ddpm.unit_to_timesteps (which floors to the training index).

``uniform`` means Diffusers Euler/Heun ``timestep_spacing='linspace'`` with linear
sigma interpolation and terminal sigma zero. Other existing flow schedule names
are rejected. Equations and the finite zero-SNR endpoint follow Diffusers 0.40:
https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/schedulers/scheduling_euler_discrete.py
https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/schedulers/scheduling_heun_discrete.py
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor

from ypuddin.config import ObjectiveConfig
from ypuddin.objectives.ddpm import DDPMObjective, unit_to_timesteps


def sampling_timestep(t: Tensor, num_train_timesteps: int = 1000) -> Tensor:
    """Continuous denoiser index for an explicitly marked inference forward.

    Shares DDPM's unit domain/boundary checks without its training-only floor.
    In particular, a model evaluation at index 499.5 receives unit t=0.4995.
    """
    unit_to_timesteps(t, num_train_timesteps)
    return (t.float() * num_train_timesteps).clamp(max=num_train_timesteps - 1)


def _schedule(obj: DDPMObjective, steps: int) -> tuple[Tensor, Tensor]:
    alpha_bar = obj.alphas_cumprod.clone()
    if obj.zero_terminal_snr:
        # EulerDiscreteScheduler uses this finite inference endpoint to avoid
        # infinite sigma. Keep the training schedule's exact terminal alpha=0.
        alpha_bar[-1] = 2**-24
    table = ((1 - alpha_bar) / alpha_bar).sqrt()
    # Float64 construction then float32 agrees with Diffusers' numpy linspace.
    # Like that schedule, steps=1 selects index 0 (not the highest training index).
    indices = torch.linspace(0, obj.num_train_timesteps - 1, steps, dtype=torch.float64).flip(0).float()
    low = indices.floor().long()
    high = (low + 1).clamp(max=obj.num_train_timesteps - 1)
    fraction = indices.double() - low
    sigmas = torch.lerp(table[low].double(), table[high].double(), fraction).float()
    return indices, torch.cat((sigmas, torch.zeros(1)))


@torch.no_grad()
def sample_ddpm(
    predict: Callable[[Tensor, Tensor], Tensor],
    shape: tuple[int, ...],
    *,
    prediction_type: Literal["epsilon", "v_prediction"],
    zero_terminal_snr: bool = False,
    num_train_timesteps: int = 1000,
    steps: int,
    sampler: str = "euler",
    scheduler: str = "uniform",
    cfg: float = 1.0,
    predict_uncond: Callable[[Tensor, Tensor], Tensor] | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    on_step: Callable[[int, int], None] | None = None,
    shift: float = 1.0,
    er_sde_order: int = 3,
    er_sde_s_noise: float = 1.0,
) -> Tensor:
    """Return clean latents from epsilon/v predictions; never silently fall back.

    Euler uses ``steps`` model evaluations; Heun uses ``2*steps-1`` (terminal
    Euler step). Guidance may double this. on_step fires once per logical step,
    matching training preview progress/cancellation. Integration uses float32;
    model inputs and returned latents use dtype. An explicit CPU RNG is supported
    for every execution device, and only initial noise consumes random numbers.

    ER-SDE arguments remain accepted/validated for dispatcher compatibility, but
    are inactive for Euler/Heun. Selecting ER-SDE itself is unsupported.
    """
    if sampler not in ("euler", "heun"):
        raise ValueError(f"DDPM sampler {sampler!r} is unsupported; choose 'euler' or 'heun'")
    if scheduler != "uniform":
        raise ValueError(f"DDPM scheduler {scheduler!r} is unsupported; choose 'uniform' (linspace)")
    if not math.isfinite(shift) or shift != 1:
        raise ValueError("DDPM sampling shift must be 1; flow sigma shifts are not supported")
    if not math.isfinite(cfg) or cfg < 0:
        raise ValueError("sampling cfg must be finite and nonnegative")
    if cfg != 1 and predict_uncond is None:
        raise ValueError("DDPM guidance with cfg != 1 requires predict_uncond")
    if er_sde_order not in (1, 2, 3) or isinstance(er_sde_order, bool):
        raise ValueError("ER-SDE order must be 1, 2 or 3")
    if not math.isfinite(er_sde_s_noise) or not 0 <= er_sde_s_noise <= 1:
        raise ValueError("ER-SDE noise strength must be finite and between 0 and 1")
    if len(shape) < 2 or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in shape):
        raise ValueError("sampling shape must contain positive integer batch and feature dimensions")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("DDPM sampling dtype must be fp16, bf16, fp32 or fp64")
    obj = DDPMObjective(
        ObjectiveConfig(timestep_sampling="uniform"),
        prediction_type=prediction_type,
        zero_terminal_snr=zero_terminal_snr,
        num_train_timesteps=num_train_timesteps,
    )
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= num_train_timesteps:
        raise ValueError("DDPM sampling steps must be an integer between 1 and num_train_timesteps")
    indices, sigmas = _schedule(obj, steps)
    sigmas = sigmas.to(device)
    times = (indices / num_train_timesteps).to(device)
    noise_device = generator.device if generator is not None else torch.device("cpu")
    y = torch.randn(shape, generator=generator, device=noise_device, dtype=torch.float32).to(device)
    y = y * sigmas[0]

    def derivative(state: Tensor, sigma: Tensor, time: Tensor) -> Tensor:
        root = (1 + sigma.square()).sqrt()
        model_input = (state / root).to(dtype)
        batch_time = time.expand(shape[0]).float()
        prediction = predict(model_input, batch_time)
        if prediction.shape != state.shape:
            raise ValueError("DDPM prediction shape must match sampling shape")
        prediction = prediction.float()
        if cfg != 1:
            uncond = predict_uncond(model_input, batch_time)
            if uncond.shape != state.shape:
                raise ValueError("DDPM unconditional prediction shape must match sampling shape")
            uncond = uncond.float()
            prediction = uncond + cfg * (prediction - uncond)
        if prediction_type == "epsilon":
            denoised = state - sigma * prediction
        else:
            denoised = state / (1 + sigma.square()) - (sigma / root) * prediction
        return (state - denoised) / sigma

    for index in range(steps):
        current, following = sigmas[index], sigmas[index + 1]
        delta = following - current
        first = derivative(y, current, times[index])
        proposed = y + delta * first
        if sampler == "heun" and index < steps - 1:
            second = derivative(proposed, following, times[index + 1])
            y = y + delta * ((first + second) / 2)
        else:
            y = proposed
        if on_step is not None:
            on_step(index + 1, steps)
    return y.to(dtype)
