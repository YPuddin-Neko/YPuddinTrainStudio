"""Rectified-flow objective: ``x_t = (1-t)·x0 + t·ε``, target ``v = ε − x0``, plus timestep samplers,
loss functions and per-timestep weighting.

Invariants (shared with every model family): ``t ∈ (0, 1)``, ``t = 1`` is pure noise, ``t`` is
fp32 and the backbone receives it unscaled.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from ypuddin.config import ObjectiveConfig

# --------------------------------------------------------------------------- timestep transforms


def mobius_shift(t: Tensor, shift: float) -> Tensor:
    """``t' = s·t / (1 + (s−1)·t)``; ``s > 1`` pushes mass toward high noise."""
    return shift * t / (1 + (shift - 1) * t)


def resolution_shift_value(num_tokens: int, base_tokens: int = 256, max_tokens: int = 4096, base_shift: float = 0.5, max_shift: float = 1.15) -> float:
    """Flux-style ``mu`` interpolated on the image token count; returned as a multiplicative shift ``exp(mu)``."""
    m = (max_shift - base_shift) / (max_tokens - base_tokens)
    mu = m * num_tokens + (base_shift - m * base_tokens)
    return math.exp(mu)


class TimestepSampler:
    """Draws ``t`` for a batch. Stateless; ``icdf`` lets validation pin quantiles deterministically."""

    def __init__(self, cfg: ObjectiveConfig):
        self.cfg = cfg

    # ---- base distribution on (0,1) before shift
    def _base_from_uniform(self, u: Tensor) -> Tensor:
        cfg = self.cfg
        mode = cfg.timestep_sampling
        if mode == "uniform":
            return u
        if mode in ("logit_normal", "shift", "resolution_shift"):
            z = torch.erfinv(2 * u - 1) * math.sqrt(2) * cfg.logit_std + cfg.logit_mean  # normal icdf
            return torch.sigmoid(z)
        if mode == "mode":
            # SD3 "mode" weighting: t = 1 - u - s·(cos²(πu/2) - 1 + u)
            s = cfg.mode_scale
            return 1 - u - s * (torch.cos(math.pi / 2 * u) ** 2 - 1 + u)
        if mode == "cosmap":
            # icdf of the CosMap density 2 / (π(1 - 2t + 2t²))
            return 1 - 1 / (torch.tan(math.pi / 2 * u) + 1)
        raise ValueError(mode)

    def _apply_shift(self, t: Tensor, num_tokens: int | None) -> Tensor:
        mode = self.cfg.timestep_sampling
        if mode == "shift":
            t = mobius_shift(t, self.cfg.shift)
        elif mode == "resolution_shift":
            t = mobius_shift(t, resolution_shift_value(num_tokens or 1024))
        lo, hi = self.cfg.t_min, self.cfg.t_max
        return t.clamp(max(lo, 1e-5), min(hi, 1 - 1e-5))

    def sample(self, batch_size: int, *, generator: torch.Generator | None = None, num_tokens: int | None = None, device: torch.device | str = "cpu") -> Tensor:
        if self.cfg.stratified and batch_size > 1:
            u = (torch.arange(batch_size, dtype=torch.float32) + torch.rand(batch_size, generator=generator)) / batch_size
            u = u[torch.randperm(batch_size, generator=generator)]
        else:
            u = torch.rand(batch_size, generator=generator)
        u = u.clamp(1e-6, 1 - 1e-6)
        return self._apply_shift(self._base_from_uniform(u), num_tokens).to(device)

    def icdf(self, quantiles: list[float], *, num_tokens: int | None = None) -> Tensor:
        u = torch.tensor(quantiles, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
        return self._apply_shift(self._base_from_uniform(u), num_tokens)


# --------------------------------------------------------------------------- noising / target


def noisy_input_and_target(x0: Tensor, noise: Tensor, t: Tensor, *, ip_noise_gamma: float = 0.0, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
    """Returns ``(x_t, target)`` with ``t`` broadcast over non-batch dims."""
    tb = t.to(x0.dtype).view(-1, *([1] * (x0.dim() - 1)))
    eps = noise
    if ip_noise_gamma > 0:
        eps = eps + ip_noise_gamma * torch.randn(noise.shape, generator=generator, device=noise.device, dtype=noise.dtype)
    x_t = (1 - tb) * x0 + tb * eps
    target = noise - x0
    return x_t, target


# --------------------------------------------------------------------------- loss


def elementwise_loss(pred: Tensor, target: Tensor, kind: str, c: float) -> Tensor:
    pred, target = pred.float(), target.float()
    if kind == "mse":
        return (pred - target) ** 2
    diff = pred - target
    if kind == "huber":
        return torch.where(diff.abs() <= c, 0.5 * diff**2, c * (diff.abs() - 0.5 * c))
    if kind == "pseudo_huber":
        return 2 * c * (torch.sqrt(diff**2 + c**2) - c)
    raise ValueError(kind)


def timestep_weight(t: Tensor, scheme: str, snr_gamma: float = 5.0) -> Tensor:
    """Per-sample multiplicative loss weight as a function of ``t``."""
    t = t.float()
    if scheme == "none":
        return torch.ones_like(t)
    if scheme == "sigma_sqrt":
        return t.clamp(min=1e-3) ** -2.0
    if scheme == "cosmap":
        return 2 / (math.pi * (1 - 2 * t + 2 * t**2))
    if scheme == "snr_like":
        snr = ((1 - t) / t.clamp(min=1e-4)) ** 2
        return torch.minimum(snr, torch.full_like(snr, snr_gamma)) / snr_gamma  # in (0,1]
    if scheme == "cosmos":
        return t**2 + (1 - t) ** 2
    raise ValueError(scheme)


def reduce_loss(per_elem: Tensor, t: Tensor, cfg: ObjectiveConfig, *, mask: Tensor | None = None, sample_weight: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Mean over non-batch dims (mask-aware), times timestep and per-sample weights.

    Returns ``(loss_scalar, per_sample_unweighted)``; the latter feeds validation / diagnostics.
    """
    if mask is not None:
        m = mask.to(per_elem.dtype)
        while m.dim() < per_elem.dim():
            m = m.unsqueeze(1)
        m = m.expand_as(per_elem)
        per_sample = (per_elem * m).flatten(1).sum(1) / m.flatten(1).sum(1).clamp(min=1.0)
    else:
        per_sample = per_elem.flatten(1).mean(1)
    w = timestep_weight(t, cfg.weighting, cfg.snr_gamma).to(per_sample.device)
    if sample_weight is not None:
        w = w * sample_weight.to(per_sample.device, per_sample.dtype)
    return (per_sample * w).mean(), per_sample.detach()


class Objective:
    """Bundles sampler + noising + loss so the trainer only sees ``prepare``/``loss``."""

    def __init__(self, cfg: ObjectiveConfig):
        self.cfg = cfg
        self.sampler = TimestepSampler(cfg)

    def sample_t(self, batch_size: int, *, generator: torch.Generator | None = None, num_tokens: int | None = None, device: torch.device | str = "cpu") -> Tensor:
        return self.sampler.sample(batch_size, generator=generator, num_tokens=num_tokens, device=device)

    def prepare(self, x0: Tensor, t: Tensor, *, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor, Tensor]:
        noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
        x_t, target = noisy_input_and_target(x0, noise, t, ip_noise_gamma=self.cfg.ip_noise_gamma, generator=generator)
        return x_t, target, noise

    def loss(self, pred: Tensor, target: Tensor, t: Tensor, *, mask: Tensor | None = None, sample_weight: Tensor | None = None) -> tuple[Tensor, Tensor]:
        per_elem = elementwise_loss(pred, target, self.cfg.loss, self.cfg.huber_c)
        return reduce_loss(per_elem, t, self.cfg, mask=mask, sample_weight=sample_weight)

    def describe(self) -> dict[str, Any]:
        return self.cfg.model_dump()
