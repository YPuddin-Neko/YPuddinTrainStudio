"""ER-SDE-3 preview sampling for rectified-flow velocity models.

Derived from Qinpeng Cui's MIT-licensed ER-SDE Solver, ``vp_3_order_taylor``
(commit 25ca6c2e6b065754a55694dc6941a3259fa0f082), with RF parameterization
and stable numerical coefficients. See ER_SDE_LICENSE.txt and ER_SDE_PROVENANCE.md.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache

import numpy as np
import torch
from torch import Tensor

from .noise import SeedNoise

_LOG_10 = math.log(10.0)


def _log_phi(value: float) -> float:
    """log(lambda * (exp(lambda**0.3) + 10)), without constructing exp(lambda**0.3)."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError("ER-SDE lambda must be positive and finite")
    power = value**0.3
    return math.log(value) + float(np.logaddexp(power, _LOG_10))


def _log_g_increment(low: float, log_ratio: float) -> float:
    """log(g(low * exp(log_ratio)) / g(low)), accurate for nearby lambdas."""
    power = low**0.3
    delta = power * math.expm1(0.3 * log_ratio)
    if delta <= 50.0:
        return math.log1p(math.expm1(delta) / (1.0 + 10.0 * math.exp(-power)))
    return float(np.logaddexp(power + delta, _LOG_10) - np.logaddexp(power, _LOG_10))


def _log_ratio(high: float, low: float) -> float:
    ratio_minus_one = (high - low) / low
    return math.log1p(ratio_minus_one) if math.isfinite(ratio_minus_one) else math.log(high) - math.log(low)


@lru_cache(maxsize=2)
def _gauss_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(order)


def _integrate_pair(function, left: float, right: float, tolerance: float = 2e-12, depth: int = 0):
    """Adaptive Gauss-Legendre quadrature of two bounded scalar integrands, on CPU float64."""
    estimates = []
    for order in (16, 32):
        nodes, weights = _gauss_rule(order)
        half = (right - left) / 2.0
        mid = (left + right) / 2.0
        values = np.asarray([function(mid + half * node) for node in nodes], dtype=np.float64)
        estimates.append(half * np.sum(weights[:, None] * values, axis=0))
    if float(np.max(np.abs(estimates[1] - estimates[0]))) <= tolerance:
        return estimates[1]
    if depth >= 20:
        raise ValueError("ER-SDE coefficient integration did not converge for this timestep schedule")
    mid = (left + right) / 2.0
    return _integrate_pair(function, left, mid, tolerance / 2.0, depth + 1) + _integrate_pair(
        function, mid, right, tolerance / 2.0, depth + 1
    )


def _moments(high: float, low: float) -> tuple[float, float]:
    """K0=int(1-R)dq, K1=int((1-q)(1-R))dq; R=phi(low)/phi(low+q*(high-low))."""
    span = high - low
    relative_span = span / low

    def log_inverse_ratio(q: float) -> float:
        if q == 0:
            return 0.0
        log_relative = (
            math.log1p(q * relative_span)
            if math.isfinite(relative_span)
            else float(np.logaddexp(0.0, math.log(span) + math.log(q) - math.log(low)))
        )
        return log_relative + _log_g_increment(low, log_relative)

    end = log_inverse_ratio(1.0)
    if end <= 1.0:
        # Integrate the complement directly: no loss of precision when R is nearly one.
        def complement(q):
            value = -math.expm1(-log_inverse_ratio(q))
            return value, (1.0 - q) * value

        values = _integrate_pair(complement, 0.0, 1.0)
        return float(values[0]), float(values[1])

    cutoff = 1.0
    if end > 40.0:
        left, right = 0.0, 1.0
        for _ in range(80):
            mid = (left + right) / 2.0
            if log_inverse_ratio(mid) < 40.0:
                left = mid
            else:
                right = mid
        cutoff = right  # Omitted R tail is bounded by exp(-40), below the quadrature tolerance.

    def ratio(q):
        value = math.exp(-log_inverse_ratio(q))
        return value, (1.0 - q) * value

    integrals = _integrate_pair(ratio, 0.0, cutoff)
    return 1.0 - float(integrals[0]), 0.5 - float(integrals[1])


def _finite(tensor: Tensor, label: str) -> Tensor:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"ER-SDE {label} contains NaN or infinity")
    return tensor


@torch.no_grad()
def er_sde_sample(
    predict: Callable[[Tensor, Tensor], Tensor],
    shape: tuple[int, ...],
    *,
    sigmas: Tensor,
    cfg: float = 1.0,
    predict_uncond: Callable[[Tensor, Tensor], Tensor] | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    on_step: Callable[[int, int], None] | None = None,
    max_order: int = 3,
    s_noise: float = 1.0,
    noise: SeedNoise | None = None,
) -> Tensor:
    """Sample with the author's ER-SDE Taylor update, using ``predict(x,t) -> velocity``.

    ``sigmas`` are RF times, strictly decreasing from ``0 < t0 < 1`` to exactly zero.
    Denoised=x-t*v, alpha=1-t, lambda=t/alpha. The caller owns the schedule. ``noise``
    (``SeedNoise``) gives the initial unit Gaussian and every stochastic increment; without
    it they come from the CPU ``generator`` (or a fresh private one) as in ComfyUI. Global
    training RNG state is never consumed.

    Start-up uses orders 1, 2, then 3; max_order can cap this at 1 or 2. The terminal
    zero step returns denoised directly. s_noise scales stochastic increments; zero
    retains ER-SDE drift and is not a replacement for an ODE solver. CFG follows the
    preview convention: without predict_uncond, predict is already fully conditioned.
    Accumulation is FP32 (FP64 when requested), model inputs/output follow dtype.
    on_step(done,total) executes after each completed step; its exceptions propagate.
    """
    if not shape or any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in shape):
        raise ValueError("ER-SDE shape must contain positive integer dimensions, including batch")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("ER-SDE requires a floating-point dtype")
    if not isinstance(max_order, int) or isinstance(max_order, bool) or max_order not in (1, 2, 3):
        raise ValueError("ER-SDE max_order must be 1, 2 or 3")
    if not math.isfinite(cfg) or not math.isfinite(s_noise) or s_noise < 0:
        raise ValueError("ER-SDE cfg must be finite and s_noise must be finite and nonnegative")
    if (
        not isinstance(sigmas, Tensor)
        or sigmas.ndim != 1
        or sigmas.numel() < 2
        or not sigmas.is_floating_point()
    ):
        raise ValueError(
            "ER-SDE sigmas must be a one-dimensional floating-point tensor with at least two entries"
        )
    times = sigmas.detach().to(device="cpu", dtype=torch.float64).tolist()
    if (
        not all(math.isfinite(value) for value in times)
        or times[-1] != 0.0
        or not 0.0 < times[0] < 1.0
        or any(first <= second for first, second in zip(times, times[1:], strict=False))
    ):
        raise ValueError("ER-SDE times must strictly decrease from 0 < t0 < 1 to exactly zero")
    if noise is None:
        if generator is not None and generator.device.type != "cpu":
            raise ValueError("ER-SDE requires a CPU random generator")
        noise = SeedNoise(generator, device=device)
    device = torch.device(device)
    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32

    def velocity(function, model_x, batch_t):
        value = function(model_x, batch_t)
        if (
            not isinstance(value, Tensor)
            or tuple(value.shape) != tuple(shape)
            or not value.is_floating_point()
        ):
            raise ValueError("ER-SDE velocity prediction must be a floating-point tensor matching shape")
        if value.device != device:
            raise ValueError("ER-SDE velocity prediction must remain on the sampling device")
        return _finite(value.to(dtype=work_dtype), "velocity prediction")

    x = noise.first(shape).to(dtype=work_dtype)
    device = x.device  # Canonicalize unspecified CUDA indices to the actual device.
    old_denoised = previous_delta = None
    old_lambda = previous_span = None
    total = len(times) - 1
    for index, (current, following) in enumerate(zip(times, times[1:], strict=False)):
        model_x = _finite(x.to(dtype=dtype), "model input")
        batch_t = torch.full(
            (shape[0],),
            current,
            device=device,
            dtype=torch.float64 if dtype == torch.float64 else torch.float32,
        )
        if cfg == 0 and predict_uncond is not None:
            v = velocity(predict_uncond, model_x, batch_t)
        else:
            v = velocity(predict, model_x, batch_t)
            if cfg != 1.0 and predict_uncond is not None:
                uncond = velocity(predict_uncond, model_x, batch_t)
                v = _finite(uncond + cfg * (v - uncond), "guided velocity")
        denoised = _finite(x - current * v, "denoised prediction")
        if following == 0:
            x = denoised
        else:
            alpha_next = 1.0 - following
            current_lambda = current / (1.0 - current)
            next_lambda = following / alpha_next
            log_lambda_ratio = _log_ratio(current_lambda, next_lambda)
            log_g_ratio = _log_g_increment(next_lambda, log_lambda_ratio)
            rho = math.exp(-log_g_ratio)
            phi_complement = -math.expm1(-log_lambda_ratio - log_g_ratio)
            updated = (following / current * rho) * x + (alpha_next * phi_complement) * denoised
            delta = None
            if max_order >= 2 and old_denoised is not None:
                span = current_lambda - next_lambda
                history_span = old_lambda - current_lambda
                k0, k1 = _moments(current_lambda, next_lambda)
                delta = denoised - old_denoised
                updated = updated + (alpha_next * span / history_span * k0) * delta
                if max_order == 3 and previous_delta is not None:
                    scale = alpha_next * 2.0 * (span / (history_span + previous_span)) * k1
                    updated = (
                        updated
                        + (scale * span / history_span) * delta
                        - (scale * span / previous_span) * previous_delta
                    )
                previous_span = history_span
            if s_noise:
                deviation = following * math.sqrt(-math.expm1(-2.0 * log_g_ratio)) * s_noise
                if deviation:
                    updated = updated + deviation * noise.next(shape, work_dtype)
            x = _finite(updated, "updated latent")
            old_lambda = current_lambda
            previous_delta = delta
            old_denoised = denoised
        if on_step is not None:
            on_step(index + 1, total)
    return _finite(x.to(dtype=dtype), "output")
