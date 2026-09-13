"""Discrete DDPM training behind the trainer's unit-timestep objective interface.

The schedule follows diffusers' DDPMScheduler (scaled_linear, add_noise,
get_velocity and rescale_zero_terminal_snr), also used by sd-scripts SDXL:
https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py

Public t is in [0, 1]; index = min(floor(t * N), N - 1). Unlike rectified flow,
t=0 is the first *noisy* training step, and t=1 is pure noise only when the
schedule is explicitly rescaled to zero terminal SNR. Model families must use
unit_to_timesteps too when passing the training timestep to their denoiser.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor

from ypuddin.config import ObjectiveConfig

from .flow import TimestepSampler, elementwise_loss, random_normal_like, reduce_loss


def _validate_step_count(num_train_timesteps: int) -> None:
    if isinstance(num_train_timesteps, bool) or not isinstance(num_train_timesteps, int):
        raise ValueError("num_train_timesteps must be an integer >= 2")
    if num_train_timesteps < 2:
        raise ValueError("num_train_timesteps must be an integer >= 2")


def unit_to_timesteps(t: Tensor, num_train_timesteps: int = 1000) -> Tensor:
    """Convert unit t into DDPM indices, preserving equally wide sampling bins.

    Exact 1 selects the last step. Non-finite/out-of-range values are rejected,
    rather than silently treating flow sigmas or already-scaled indices as t.
    """
    _validate_step_count(num_train_timesteps)
    if t.is_complex() or not torch.isfinite(t).all() or ((t < 0) | (t > 1)).any():
        raise ValueError("DDPM timesteps must be finite unit values in [0, 1]")
    # Preserve float64 boundary probes; avoid doing the index arithmetic in bf16/fp16.
    unit = t if t.dtype == torch.float64 else t.float()
    return (unit * num_train_timesteps).floor().long().clamp(max=num_train_timesteps - 1)


def _betas(num_train_timesteps: int, zero_terminal_snr: bool) -> Tensor:
    betas = torch.linspace(0.00085**0.5, 0.012**0.5, num_train_timesteps, dtype=torch.float32).square()
    if zero_terminal_snr:
        # Algorithm 1 of https://arxiv.org/abs/2305.08891: translate sqrt(alpha_bar)
        # to end at zero, then scale it to preserve the initial noise level.
        root_alpha_bar = (1 - betas).cumprod(0).sqrt()
        first, last = root_alpha_bar[0].clone(), root_alpha_bar[-1].clone()
        rescaled = ((root_alpha_bar - last) * (first / (first - last))).square()
        alphas = torch.cat((rescaled[:1], rescaled[1:] / rescaled[:-1]))
        betas = 1 - alphas
    return betas


class DDPMObjective:
    """Epsilon or v-prediction with the same prepare/loss contract as Objective.

    Only uniform/logit_normal timestep sampling and unweighted loss are currently
    supported. Flow-specific weighting (including snr_like) is not DDPM Min-SNR;
    reject it explicitly. Inactive sampler fields, e.g. cfg.shift for uniform,
    remain inactive, matching ObjectiveConfig's existing conditional semantics.
    """

    def __init__(
        self,
        cfg: ObjectiveConfig,
        *,
        prediction_type: Literal["epsilon", "v_prediction"] = "epsilon",
        zero_terminal_snr: bool = False,
        num_train_timesteps: int = 1000,
    ):
        if prediction_type not in ("epsilon", "v_prediction"):
            raise ValueError("DDPM prediction_type must be 'epsilon' or 'v_prediction'")
        if not isinstance(zero_terminal_snr, bool):
            raise ValueError("zero_terminal_snr must be a bool")
        _validate_step_count(num_train_timesteps)
        if cfg.timestep_sampling not in ("uniform", "logit_normal"):
            raise ValueError(
                "DDPM objective.timestep_sampling must be 'uniform' or 'logit_normal'; "
                "flow shift/resolution_shift/mode/cosmap sampling is not supported"
            )
        if cfg.weighting != "none":
            raise ValueError("DDPM objective.weighting must be 'none'; flow weighting is not DDPM Min-SNR")
        self.cfg = cfg.model_copy(deep=True)
        self.prediction_type = prediction_type
        self.zero_terminal_snr = zero_terminal_snr
        self.num_train_timesteps = num_train_timesteps
        self.betas = _betas(num_train_timesteps, zero_terminal_snr)
        self.alphas_cumprod = (1 - self.betas).cumprod(0)
        self.sampler = TimestepSampler(self.cfg)

    def sample_t(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        num_tokens: int | None = None,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        return self.sampler.sample(batch_size, generator=generator, num_tokens=num_tokens, device=device)

    def timesteps(self, t: Tensor) -> Tensor:
        return unit_to_timesteps(t, self.num_train_timesteps)

    def _coefficients(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim < 2 or not x.is_floating_point():
            raise ValueError("DDPM samples must be floating tensors with a batch and feature dimensions")
        if t.ndim != 1 or len(t) != len(x):
            raise ValueError("DDPM t must contain exactly one unit timestep per sample")
        indices = self.timesteps(t).to(x.device)
        alpha_bar = self.alphas_cumprod.to(x.device)[indices]
        shape = (len(x),) + (1,) * (x.ndim - 1)
        # Compute square roots before casting: bf16 alpha_bar can otherwise round
        # the first step to 1 and erroneously remove its noise entirely.
        signal = alpha_bar.sqrt().to(x.dtype).view(shape)
        sigma = (1 - alpha_bar).sqrt().to(x.dtype).view(shape)
        return signal, sigma

    def prepare(
        self, x0: Tensor, t: Tensor, *, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        signal, sigma = self._coefficients(x0, t)
        noise = random_normal_like(x0, generator)
        input_noise = noise
        if self.cfg.ip_noise_gamma:
            input_noise = noise + self.cfg.ip_noise_gamma * random_normal_like(noise, generator)
        x_t = signal * x0 + sigma * input_noise
        target = noise if self.prediction_type == "epsilon" else signal * noise - sigma * x0
        return x_t, target, noise

    def loss(
        self,
        pred: Tensor,
        target: Tensor,
        t: Tensor,
        *,
        mask: Tensor | None = None,
        sample_weight: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if pred.shape != target.shape:
            raise ValueError("DDPM prediction and target shapes must match")
        if t.ndim != 1 or len(t) != len(pred):
            raise ValueError("DDPM t must contain exactly one unit timestep per sample")
        self.timesteps(t)  # Validate even though unweighted loss does not use the index.
        per_elem = elementwise_loss(pred, target, self.cfg.loss, self.cfg.huber_c)
        # Share the existing mask normalization and mean-over-batch prior weights.
        return reduce_loss(per_elem, t, self.cfg, mask=mask, sample_weight=sample_weight)

    def describe(self) -> dict[str, Any]:
        return {
            **self.cfg.model_dump(),
            "type": "ddpm",
            "prediction_type": self.prediction_type,
            "zero_terminal_snr": self.zero_terminal_snr,
            "num_train_timesteps": self.num_train_timesteps,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "beta_end": 0.012,
        }
