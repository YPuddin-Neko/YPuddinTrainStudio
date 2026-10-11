"""Online, image-normalized RF timestep allocation in physical ``t`` coordinates.

InfoNoise density: https://arxiv.org/html/2602.18647v2 (Eq. 15, Appendix B.6).
For RF velocity MSE, the sigma-to-t Jacobian gives g(sigma)*t*(1-t)*M_v.
This implementation uses a fixed configured gate, binned HT estimates and a
baseline/uniform mixture; it does not estimate the paper's dynamic onset.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor


class AdaptiveTimestepSampler:
    """A binned mixture with optimizer-transaction statistics.

    The caller owns the CPU generator and checkpoints its state. Statistics are
    unweighted per-image velocity MSE; regularization examples are excluded.
    A successful step globally sums statistics before the next sampling refresh.
    """

    STATE_VERSION = 1
    FORMULA = "rf-velocity-physical-t-ht-v1"
    _FIELDS = (
        "timestep_sampling",
        "shift",
        "logit_mean",
        "logit_std",
        "t_min",
        "t_max",
        "stratified",
        "resolution_reference_tokens",
        "infonoise_bins",
        "infonoise_warmup_steps",
        "infonoise_refresh_every",
        "infonoise_ema_decay",
        "infonoise_exploration",
        "infonoise_adaptive_mix",
        "infonoise_min_samples",
        "infonoise_min_coverage",
        "infonoise_gate_sigma",
    )

    def __init__(self, cfg: Any, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        values = {name: getattr(cfg, name) for name in self._FIELDS}
        self.cfg = cfg = SimpleNamespace(**values)
        self.recipe = dict(values)
        if cfg.timestep_sampling != "resolution_sqrt":
            self.recipe.pop("resolution_reference_tokens")
        if cfg.timestep_sampling not in {"shift", "resolution_sqrt"}:
            self.recipe.pop("shift")
        if cfg.timestep_sampling == "uniform":
            self.recipe.pop("logit_mean")
            self.recipe.pop("logit_std")
        if cfg.timestep_sampling not in {"uniform", "logit_normal", "shift", "resolution_sqrt"}:
            raise ValueError("InfoNoise requires uniform, logit_normal, shift, or resolution_sqrt")
        self.k = int(cfg.infonoise_bins)
        self.lo, self.hi = max(float(cfg.t_min), 0.001), min(float(cfg.t_max), 0.999)
        if self.k < 2 or not self.lo < self.hi:
            raise ValueError("InfoNoise requires at least two bins and a nonempty timestep interval")
        if not 0 < cfg.infonoise_exploration < 1 or not 0 <= cfg.infonoise_adaptive_mix <= 1:
            raise ValueError("Invalid InfoNoise mixture")
        if not 0 <= cfg.infonoise_ema_decay < 1 or not 0 < cfg.infonoise_min_coverage <= 1:
            raise ValueError("Invalid InfoNoise smoothing or coverage")
        if cfg.infonoise_refresh_every < 1 or cfg.infonoise_warmup_steps < 0 or cfg.infonoise_min_samples < 1:
            raise ValueError("Invalid InfoNoise update cadence")
        if cfg.logit_std <= 0 or cfg.shift <= 0 or cfg.resolution_reference_tokens <= 0:
            raise ValueError("Invalid InfoNoise baseline")
        if cfg.infonoise_gate_sigma < 0 or not all(
            not isinstance(value, (float, int)) or math.isfinite(value) for value in self.recipe.values()
        ):
            raise ValueError("InfoNoise parameters must be finite")
        self.width = (self.hi - self.lo) / self.k
        self.edges = torch.linspace(self.lo, self.hi, self.k + 1, dtype=torch.float64)
        self.dtype = torch.float32 if self.device.type == "mps" else torch.float64
        # [HT sums (K), observation counts (K), valid N, invalid, empty mask, regularization]
        self._size = 2 * self.k + 4
        self._pending = torch.zeros(self._size, device=self.device, dtype=self.dtype)
        self.device = self._pending.device
        self._interval = torch.zeros_like(self._pending)
        self._candidate: Tensor | None = None
        self._has_pending = False
        self._last_t: Tensor | None = None
        self._last_q: Tensor | None = None
        self._ema = torch.zeros(self.k, dtype=torch.float64)
        self._counts = torch.zeros(self.k, dtype=torch.float64)
        self._mass = torch.full((self.k,), 1 / self.k, dtype=torch.float64)
        self._cdf = self._make_cdf(self._mass)
        self._totals = torch.zeros(4, dtype=torch.float64)
        self._steps = self._refreshes = self._statistics_step = 0
        self._alpha = 0.0
        self._reason = "warmup"
        self._ema_ready = False
        self._baseline_cache: dict[int | None, Tensor] = {}
        self._last_num_tokens: int | None = None

    @staticmethod
    def _make_cdf(mass: Tensor) -> Tensor:
        cdf = torch.cat((torch.zeros(1, dtype=torch.float64), mass.cumsum(0)))
        cdf[-1] = 1.0
        return cdf

    def _shift(self, num_tokens: int | None) -> float:
        mode = self.cfg.timestep_sampling
        if mode == "resolution_sqrt":
            if num_tokens is None or isinstance(num_tokens, bool) or num_tokens <= 0:
                raise ValueError("resolution_sqrt requires the actual positive image token count")
            return self.cfg.shift * math.sqrt(num_tokens / self.cfg.resolution_reference_tokens)
        return self.cfg.shift if mode == "shift" else 1.0

    def _baseline(self, num_tokens: int | None) -> Tensor:
        key = num_tokens if self.cfg.timestep_sampling == "resolution_sqrt" else None
        if key in self._baseline_cache:
            return self._baseline_cache[key]
        shift = self._shift(num_tokens)
        if self.cfg.timestep_sampling == "uniform":
            cdf = self.edges.clone()
        else:
            z = (torch.logit(self.edges) - math.log(shift) - self.cfg.logit_mean) / self.cfg.logit_std
            cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
        # Preserve clipped baseline bin masses; endpoints are spread within their bin.
        cdf[0], cdf[-1] = 0.0, 1.0
        mass = torch.diff(cdf).clamp_min(0)
        mass /= mass.sum()
        self._baseline_cache[key] = mass
        return mass

    def _mixture(self, num_tokens: int | None) -> Tensor:
        explore = self.cfg.infonoise_exploration
        mass = (1 - explore) * ((1 - self._alpha) * self._baseline(num_tokens) + self._alpha * self._mass)
        return mass + explore / self.k

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        num_tokens: int | None = None,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if batch_size < 1:
            raise ValueError("A timestep batch must contain at least one sample")
        if generator is not None and generator.device.type != "cpu":
            raise ValueError("InfoNoise uses the caller's CPU generator")
        if self._candidate is not None:
            raise RuntimeError("Accept or reject the current observation before sampling again")
        mass = self._mixture(num_tokens)
        self._last_num_tokens = num_tokens
        cdf = self._make_cdf(mass)
        if self.cfg.stratified and batch_size > 1:
            u = torch.arange(batch_size, dtype=torch.float64) + torch.rand(batch_size, generator=generator)
            u = (u / batch_size)[torch.randperm(batch_size, generator=generator)]
        else:
            u = torch.rand(batch_size, generator=generator).double()
        indices = torch.searchsorted(cdf[1:], u, right=True).clamp_max(self.k - 1)
        fraction = (u - cdf[indices]) / mass[indices]
        t = (self.lo + self.width * (indices + fraction)).float()
        target_device = self.device if device is None else torch.device(device)
        self._last_t = t.to(self.device)
        self._last_q = mass[indices].to(device=self.device, dtype=self.dtype)
        return t.to(target_device)

    @torch.no_grad()
    def record(
        self,
        t: Tensor,
        per_sample_mse: Tensor,
        is_reg: Tensor | list[bool],
        valid_mask: Tensor | None = None,
    ) -> None:
        if self._last_t is None or self._last_q is None:
            raise RuntimeError("record requires a preceding training sample")
        if self._candidate is not None:
            raise RuntimeError("Accept or reject the current observation before recording again")
        t = t.detach().to(device=self.device, dtype=self.dtype)
        mse = per_sample_mse.detach().to(device=self.device, dtype=self.dtype)
        reg = torch.as_tensor(is_reg, device=self.device, dtype=torch.bool)
        if t.ndim != 1 or mse.shape != t.shape or reg.shape != t.shape or self._last_t.shape != t.shape:
            raise ValueError("InfoNoise requires one timestep, MSE, and regularization flag per image")
        mask = torch.ones_like(reg) if valid_mask is None else valid_mask.detach().to(self.device).bool()
        if mask.shape != t.shape:
            raise ValueError("valid_mask must contain one validity flag per image")
        t3 = t.pow(3)
        gate = t3 / (t3 + (self.cfg.infonoise_gate_sigma * (1 - t)).pow(3))
        value = gate * t * (1 - t) * mse / self._last_q
        finite = torch.isfinite(value) & torch.isfinite(mse) & (mse >= 0)
        finite &= (t >= self.lo - 1e-7) & (t <= self.hi + 1e-7) & (t == self._last_t.to(self.dtype))
        valid = ~reg & mask & finite
        # Invalid inputs have no scatter index or contribution; no device synchronization is needed.
        indices = ((torch.nan_to_num(t, nan=self.lo) - self.lo) / self.width).long().clamp(0, self.k - 1)
        stats = torch.zeros_like(self._pending)
        stats[: self.k].scatter_add_(0, indices, torch.where(valid, value, torch.zeros_like(value)))
        stats[self.k : 2 * self.k].scatter_add_(0, indices, valid.to(self.dtype))
        stats[2 * self.k] = valid.sum()
        stats[2 * self.k + 1] = (~reg & mask & ~finite).sum()
        stats[2 * self.k + 2] = (~reg & ~mask).sum()
        stats[2 * self.k + 3] = reg.sum()
        self._candidate = stats
        self._last_t = self._last_q = None

    def accept_observation(self) -> None:
        if self._candidate is None:
            raise RuntimeError("There is no current InfoNoise observation")
        self._pending.add_(self._candidate)
        self._candidate = None
        self._has_pending = True

    def reject_observation(self) -> None:
        self._candidate = None
        self._last_t = self._last_q = None

    def discard_pending(self) -> None:
        self.reject_observation()
        self._pending.zero_()
        self._has_pending = False

    @torch.no_grad()
    def commit_step(self, reduce_sum: Callable[[Tensor], Tensor | None] | None = None) -> bool:
        """Called by every rank after a successful optimizer update, including zero-main ranks."""
        if self._candidate is not None or self._last_t is not None:
            raise RuntimeError("Resolve the current sample and observation before committing")
        stats = self._pending.clone()
        if reduce_sum is not None:
            reduced = reduce_sum(stats)
            if reduced is not None:
                stats = reduced
            if stats.shape != self._pending.shape or stats.device != self.device or stats.dtype != self.dtype:
                raise ValueError("InfoNoise reduction must preserve the statistics shape, device, and dtype")
        self._interval.add_(stats)
        self._pending.zero_()
        self._has_pending = False
        self._steps += 1
        if self._steps % self.cfg.infonoise_refresh_every:
            return False
        self._refresh()
        return True

    def _refresh(self) -> None:
        values = self._interval.detach().to(device="cpu", dtype=torch.float64, copy=True)
        self._interval.zero_()
        self._refreshes += 1
        self._statistics_step = self._steps
        if not torch.isfinite(values).all() or (values < 0).any():
            self._alpha, self._reason = 0.0, "nonfinite_statistics"
            return
        self._counts += values[self.k : 2 * self.k]
        self._totals += values[2 * self.k :]
        n = float(values[2 * self.k])
        if n <= 0:
            self._alpha, self._reason = 0.0, "no_main_observations"
            return
        estimate = values[: self.k] * (self.width / n)
        observed = self._counts > 0
        if not self._ema_ready:
            # Unseen bins start from the observed mean, a finite-sample prior rather than measured risk.
            prior = estimate[values[self.k : 2 * self.k] > 0].mean()
            self._ema.fill_(float(prior))
            self._ema_ready = True
        decay = self.cfg.infonoise_ema_decay
        self._ema.mul_(decay).add_(estimate, alpha=1 - decay)
        candidate = self._ema.clone()
        if observed.any():
            candidate[~observed] = candidate[observed].mean()
        total = float(candidate.sum())
        if not math.isfinite(total) or total <= 0:
            self._alpha, self._reason = 0.0, "zero_information"
            return
        self._mass = candidate / total
        self._cdf = self._make_cdf(self._mass)
        coverage = float(observed.double().mean())
        if self._steps < self.cfg.infonoise_warmup_steps:
            self._alpha, self._reason = 0.0, "warmup"
        elif self._totals[0] < self.cfg.infonoise_min_samples:
            self._alpha, self._reason = 0.0, "insufficient_samples"
        elif coverage < self.cfg.infonoise_min_coverage:
            self._alpha, self._reason = 0.0, "insufficient_coverage"
        else:
            self._alpha = self.cfg.infonoise_adaptive_mix * coverage
            self._reason = "active" if self._alpha else "adaptive_mix_zero"

    def status(self, num_tokens: int | None = None) -> dict[str, Any]:
        """CPU snapshot; counts/distribution reflect ``statistics_step``, not unrefreshed observations."""
        result: dict[str, Any] = {
            "kind": "infonoise",
            "formula": self.FORMULA,
            "coordinate": "physical_t",
            "optimizer_updates": self._steps,
            "statistics_step": self._statistics_step,
            "refreshes": self._refreshes,
            "state": self._reason,
            "reason": self._reason,
            "stage": "fallback"
            if self._reason in {"zero_information", "nonfinite_statistics", "no_main_observations"}
            else "active"
            if self._alpha > 0
            else "warmup"
            if self._steps < self.cfg.infonoise_warmup_steps
            else "collecting",
            "active": self._alpha > 0,
            "coverage": float((self._counts > 0).double().mean()),
            "observed_bins": int((self._counts > 0).sum()),
            "bins": self.k,
            "main_samples": int(self._totals[0]),
            "invalid_samples": int(self._totals[1]),
            "empty_mask_samples": int(self._totals[2]),
            "regularization_samples": int(self._totals[3]),
            "warmup_steps": self.cfg.infonoise_warmup_steps,
            "warmup_completed_steps": min(self._steps, self.cfg.infonoise_warmup_steps),
            "min_samples": self.cfg.infonoise_min_samples,
            "min_coverage": self.cfg.infonoise_min_coverage,
            "adaptive_fraction": (1 - self.cfg.infonoise_exploration) * self._alpha,
            "baseline_fraction": (1 - self.cfg.infonoise_exploration) * (1 - self._alpha),
            "exploration_fraction": self.cfg.infonoise_exploration,
            "bin_edges": self.edges.tolist(),
            "learned_mass": self._mass.tolist(),
            "gate_coordinate": "sigma=t/(1-t)",
            "gate_sigma": self.cfg.infonoise_gate_sigma,
            "gate_exponent": 3,
        }
        tokens = num_tokens if num_tokens is not None else self._last_num_tokens
        if tokens is not None or self.cfg.timestep_sampling != "resolution_sqrt":
            result["effective_mass"] = self._mixture(tokens).tolist()
            result["baseline_shift"] = self._shift(tokens)
            result["effective_mass_num_tokens"] = tokens
            result["effective_mass_scope"] = (
                "resolution_independent"
                if self.cfg.timestep_sampling != "resolution_sqrt"
                else "requested_tokens"
                if num_tokens is not None
                else "last_local_sample_tokens"
            )
        return result

    def state_dict(self) -> dict[str, Any]:
        if self._candidate is not None or self._has_pending or self._last_t is not None:
            raise RuntimeError("Checkpoint InfoNoise only at a completed optimizer boundary")
        return {
            "version": self.STATE_VERSION,
            "formula": self.FORMULA,
            "recipe": dict(self.recipe),
            "rng_owner": "caller_cpu_generator",
            "edges": self.edges.tolist(),
            "interval": self._interval.detach().cpu().double().tolist(),
            "ema": self._ema.tolist(),
            "counts": self._counts.tolist(),
            "totals": self._totals.tolist(),
            "mass": self._mass.tolist(),
            "cdf": self._cdf.tolist(),
            "ema_ready": self._ema_ready,
            "steps": self._steps,
            "refreshes": self._refreshes,
            "statistics_step": self._statistics_step,
            "alpha": self._alpha,
            "reason": self._reason,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = set(self.state_dict())
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError("Incomplete or unexpected InfoNoise checkpoint fields")
        if (
            type(state["version"]) is not int
            or state["version"] != self.STATE_VERSION
            or state["formula"] != self.FORMULA
            or state["recipe"] != self.recipe
            or state["rng_owner"] != "caller_cpu_generator"
        ):
            raise ValueError("InfoNoise checkpoint semantics differ from the current configuration")

        def vector(name: str, length: int, *, integral: bool = False) -> Tensor:
            value = state[name]
            if (
                not isinstance(value, list)
                or len(value) != length
                or any(isinstance(x, bool) or not isinstance(x, (float, int)) for x in value)
            ):
                raise ValueError(f"Invalid InfoNoise {name}")
            tensor = torch.tensor(value, dtype=torch.float64)
            if not torch.isfinite(tensor).all() or (tensor < 0).any():
                raise ValueError(f"Invalid InfoNoise {name}")
            if integral and not torch.equal(tensor, tensor.round()):
                raise ValueError(f"Invalid InfoNoise {name} counts")
            return tensor

        edges, interval = vector("edges", self.k + 1), vector("interval", self._size)
        ema, counts, totals = (
            vector("ema", self.k),
            vector("counts", self.k, integral=True),
            vector("totals", 4, integral=True),
        )
        mass, cdf = vector("mass", self.k), vector("cdf", self.k + 1)
        if not torch.equal(edges, self.edges) or not torch.allclose(
            cdf, self._make_cdf(mass), atol=1e-12, rtol=0
        ):
            raise ValueError("Invalid InfoNoise bin or CDF layout")
        if not math.isclose(float(mass.sum()), 1, abs_tol=1e-12) or (torch.diff(cdf) < 0).any():
            raise ValueError("Invalid InfoNoise probability mass")
        if (
            not torch.equal(interval[self.k :], interval[self.k :].round())
            or float(interval[self.k : 2 * self.k].sum()) != float(interval[2 * self.k])
            or float(counts.sum()) != float(totals[0])
        ):
            raise ValueError("Invalid InfoNoise observation counts")
        for name in ("steps", "refreshes", "statistics_step"):
            if type(state[name]) is not int or state[name] < 0:
                raise ValueError(f"Invalid InfoNoise {name}")
        steps, refreshes, statistics_step = state["steps"], state["refreshes"], state["statistics_step"]
        if (
            refreshes != steps // self.cfg.infonoise_refresh_every
            or statistics_step != refreshes * self.cfg.infonoise_refresh_every
        ):
            raise ValueError("Invalid InfoNoise refresh history")
        alpha = state["alpha"]
        if (
            type(alpha) not in (int, float)
            or not math.isfinite(alpha)
            or not 0 <= alpha <= self.cfg.infonoise_adaptive_mix
            or type(state["ema_ready"]) is not bool
            or state["reason"]
            not in {
                "warmup",
                "nonfinite_statistics",
                "no_main_observations",
                "zero_information",
                "insufficient_samples",
                "insufficient_coverage",
                "active",
                "adaptive_mix_zero",
            }
        ):
            raise ValueError("Invalid InfoNoise adaptation state")
        if (alpha > 0) != (state["reason"] == "active"):
            raise ValueError("Invalid InfoNoise active state")
        if alpha > 0 and (
            steps < self.cfg.infonoise_warmup_steps
            or totals[0] < self.cfg.infonoise_min_samples
            or float((counts > 0).double().mean()) < self.cfg.infonoise_min_coverage
            or not state["ema_ready"]
        ):
            raise ValueError("InfoNoise active state lacks its required observations")
        if alpha > 0 and not math.isclose(
            alpha,
            self.cfg.infonoise_adaptive_mix * float((counts > 0).double().mean()),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("InfoNoise active mixture differs from its coverage")
        self._interval.copy_(interval.to(device=self.device, dtype=self.dtype))
        self._ema, self._counts, self._totals, self._mass, self._cdf = ema, counts, totals, mass, cdf
        self._steps, self._refreshes, self._statistics_step = steps, refreshes, statistics_step
        self._alpha, self._reason, self._ema_ready = float(alpha), state["reason"], state["ema_ready"]
        self.discard_pending()
