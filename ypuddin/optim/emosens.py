"""EmoSens v3.9.3 with explicit update loss and resumable FP32 optimizer state.

Adapted from muooon/EmoSens, commit bc925f9ab366a2fc8b20e5b5205d7082a8097c98.
Apache-2.0; see EMOSENS_NOTICE.md and EMOSENS_LICENSE.txt for attribution and changes.
"""

from __future__ import annotations

import logging
import math
from copy import deepcopy

import torch
from torch.optim import Optimizer

log = logging.getLogger(__name__)


class EmoSens(Optimizer):
    """Generate one step size from the loss of each completed gradient update."""

    _FORMAT = {"algorithm": "emosens", "upstream": "bc925f9ab366a2fc8b20e5b5205d7082a8097c98", "version": 1}
    _GROUP_IDENTITY = ("name", "param_names", "emosens_scale", "emosens_shapes", "emosens_dtypes", "betas", "eps", "weight_decay")

    def __init__(self, params, lr=1.0, eps=1e-8, betas=(0.9, 0.995), weight_decay=0.01,
                 stopcoef=0.04, use_shadow=False, notify=True):
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("EmoSens initial scale must be finite and positive")
        if not math.isfinite(eps) or eps <= 0 or not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("EmoSens requires positive eps and nonnegative weight decay")
        if len(betas) != 2 or any(not 0 <= beta < 1 for beta in betas):
            raise ValueError("EmoSens betas must be within [0, 1)")
        if not math.isfinite(stopcoef) or stopcoef < 0:
            raise ValueError("EmoSens stopcoef must be finite and nonnegative")
        self._settings = dict(lr=lr, stopcoef=stopcoef, use_shadow=use_shadow, notify=notify)
        self._pending_loss = None
        self.emoScope = lr
        self.base_scale, self.min_lim = 1e-4, 1e-8
        self.max_lim = min(max(min(lr, 1.0) * 3e-3, 3e-7), 3e-3)
        self.dNR_hist, self.noise_est, self.d_est, self.c_est = 1.0, 1.0, 0.02, 0.0
        self.loss_ema = {}
        self.should_stop = False
        self.updates = 0
        super().__init__(params, dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay))

    def add_param_group(self, group):
        group = dict(group)
        params = group["params"]
        params = [params] if isinstance(params, torch.Tensor) else list(params)
        group["params"] = params
        scale = group.get("lr", self.defaults["lr"])
        if scale not in (0, self._settings["lr"]):
            raise ValueError("EmoSens uses one initial scale; remove per-group learning rates")
        for p in params:
            if p.dtype not in (torch.float32, torch.bfloat16, torch.float16):
                raise ValueError("EmoSens requires FP32, BF16 or FP16 parameters")
        group.update(emosens_scale=scale, emosens_shapes=[list(p.shape) for p in params],
                     emosens_dtypes=[str(p.dtype) for p in params])
        super().add_param_group(group)

    def set_loss(self, loss):
        """Supply the unscaled, sample-weighted loss for this optimizer update."""
        value = float(loss.detach().item()) if isinstance(loss, torch.Tensor) else float(loss)
        if not math.isfinite(value):
            raise ValueError("EmoSens requires a finite update loss")
        self._pending_loss = value

    def _pulse(self, loss):
        ema = self.loss_ema
        for name, weight in (("short", 0.3), ("medium", 0.05), ("long", 0.01)):
            ema[name] = weight * loss + (1 - weight) * ema.get(name, loss)
        scale_l, scale_m = max(ema["long"], 1e-5), max(ema["medium"], 1e-5)
        diff = ema["long"] - ema["short"]
        dl, dm = diff / scale_l, diff / scale_m
        scalar = math.tanh(dm if abs(dl) >= 0.05 and abs(dm) * scale_m < abs(dl) * scale_l else dl)
        trust = math.copysign(1 - abs(scalar), scalar)
        self.noise_est = 0.97 * self.noise_est + 0.03 * abs(scalar)
        self.d_est = 0.97 * self.d_est + 0.03 * abs(trust)
        self.c_est = 0.7 * self.c_est + 0.3 * scalar
        noise = max(self.noise_est, 1e-10)
        now = ((abs(noise - self.d_est) + 0.1) / (abs(scalar - trust) + 0.1)) ** 2
        if now >= self.dNR_hist and trust >= 0.5:
            self.dNR_hist = min(now, self.dNR_hist * 1.5)
        elif -0.5 <= trust <= 0.5:
            self.dNR_hist = now * 0.8
        chain = self.emoScope * max(100.0 ** self.c_est, 1e-3)
        pulse = float(max(min(self.dNR_hist * (chain * self.base_scale), self.max_lim), self.min_lim))
        return pulse, scalar, trust, scale_m

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
            self.set_loss(loss)
        active = [(g, p) for g in self.param_groups if g["emosens_scale"] != 0
                  for p in g["params"] if p.grad is not None]
        if not active:
            self._pending_loss = None
            return loss
        if self._pending_loss is None:
            raise ValueError("EmoSens requires set_loss(update_loss) or a loss-returning closure before each step")
        if any(p.grad.is_sparse for _, p in active):
            raise ValueError("EmoSens does not support sparse gradients")
        pulse, scalar, trust, scale_m = self._pulse(self._pending_loss)
        self._pending_loss = None
        shadow = self._settings["use_shadow"]
        ratio = 1 - abs(scalar) if shadow and abs(scalar) > 0.625 else 0.0
        for group, p in active:
            state = self.state[p]
            if not state:
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                if p.dtype != torch.float32:
                    state["master_param"] = p.detach().float().clone()
                if shadow:
                    state["shadow"] = p.detach().float().clone()
            weight = state.get("master_param", p)
            grad = p.grad.float()
            if shadow:
                if ratio > 0:
                    weight.mul_(1 - ratio).add_(state["shadow"], alpha=abs(trust))
                else:
                    state["shadow"].lerp_(weight, 0.1 * abs(trust))
            beta1, beta2 = group["betas"]
            m, v = state["exp_avg"], state["exp_avg_sq"]
            m.mul_(beta1).add_(grad, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            denom = v.sqrt().add_(group["eps"])
            if group["weight_decay"]:
                weight.mul_(1 - group["weight_decay"] * pulse)
            weight.addcdiv_(m, denom, value=-pulse)
            if weight is not p:
                p.copy_(weight)
        for group in self.param_groups:
            group["lr"] = pulse if group["emosens_scale"] != 0 else 0.0
        ready = self.d_est - self.noise_est >= 0.3 and scale_m <= self._settings["stopcoef"]
        if ready and not self.should_stop and self._settings["notify"]:
            log.info("EmoSens: stability threshold reached (loss EMA %.6g); training continues", scale_m)
        self.should_stop = ready
        self.updates += 1
        return loss

    def get_learning_rates(self):
        return [float(group["lr"]) if self.updates else 0.0 for group in self.param_groups]

    def state_dict(self):
        result = super().state_dict()
        result["emosens"] = {
            "format": dict(self._FORMAT), "settings": dict(self._settings),
            "loss_ema": dict(self.loss_ema), "updates": self.updates, "should_stop": self.should_stop,
            **{name: getattr(self, name) for name in ("emoScope", "dNR_hist", "noise_est", "d_est", "c_est")},
        }
        return result

    def load_state_dict(self, state_dict):
        control = state_dict.get("emosens", {})
        if control.get("format") != self._FORMAT or control.get("settings") != self._settings:
            raise ValueError("EmoSens checkpoint version or initial settings differ from this optimizer")
        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("EmoSens checkpoint parameter groups differ")
        for saved, current in zip(saved_groups, self.param_groups, strict=True):
            if any(saved.get(key) != current.get(key) for key in self._GROUP_IDENTITY):
                raise ValueError("EmoSens checkpoint parameter identity or hyperparameters differ")
            if len(saved["params"]) != len(current["params"]):
                raise ValueError("EmoSens checkpoint parameter count differs")
            for saved_id, p in zip(saved["params"], current["params"], strict=True):
                buffers = state_dict["state"].get(saved_id, {})
                if not buffers:
                    continue
                required = {"exp_avg", "exp_avg_sq"}
                if p.dtype != torch.float32:
                    required.add("master_param")
                if self._settings["use_shadow"]:
                    required.add("shadow")
                if set(buffers) != required or any(
                    not isinstance(v, torch.Tensor) or v.shape != p.shape or v.dtype != torch.float32
                    for v in buffers.values()
                ):
                    raise ValueError("EmoSens checkpoint requires matching FP32 moment, master and shadow buffers")
        updates = control.get("updates")
        ema = control.get("loss_ema")
        names = ("emoScope", "dNR_hist", "noise_est", "d_est", "c_est")
        if (not isinstance(updates, int) or updates < 0 or not isinstance(ema, dict)
                or not isinstance(control.get("should_stop"), bool)
                or set(ema) != ({"short", "medium", "long"} if updates else set())
                or any(not math.isfinite(control.get(name, math.nan)) for name in names)
                or any(not math.isfinite(v) for v in ema.values())):
            raise ValueError("EmoSens checkpoint loss controller is incomplete")
        super().load_state_dict({"state": state_dict["state"], "param_groups": saved_groups})
        # Optimizer.load_state_dict casts buffers to parameter dtype; retain FP32 values.
        for saved, current in zip(saved_groups, self.param_groups, strict=True):
            for saved_id, p in zip(saved["params"], current["params"], strict=True):
                if saved_id in state_dict["state"]:
                    self.state[p] = {key: value.to(device=p.device, dtype=torch.float32, copy=True)
                                     for key, value in state_dict["state"][saved_id].items()}
        for name in names:
            setattr(self, name, control[name])
        self.loss_ema = deepcopy(ema)
        self.updates, self.should_stop = updates, bool(control["should_stop"])
        self._pending_loss = None
