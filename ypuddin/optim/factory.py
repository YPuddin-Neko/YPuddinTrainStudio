"""Optimizer factory, Kahan-compensated bf16 wrapper and LR schedulers."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ypuddin.config import OptimizerConfig, SchedulerConfig

# --------------------------------------------------------------------------- optimizers

_BUILTIN: dict[str, str] = {
    "adamw": "torch.optim.AdamW",
    "adam": "torch.optim.Adam",
    "sgd": "torch.optim.SGD",
    "adamw8bit": "bitsandbytes.optim.AdamW8bit",
    "lion": "lion_pytorch.Lion",
    "lion8bit": "bitsandbytes.optim.Lion8bit",
    "prodigy": "prodigyopt.Prodigy",
    "prodigy_plus_sf": "prodigyplus.ProdigyPlusScheduleFree",
    "adafactor": "transformers.optimization.Adafactor",
    "came": "pytorch_optimizer.CAME",
    "adamw_sf": "schedulefree.AdamWScheduleFree",
}

SCHEDULE_FREE = {"prodigy_plus_sf", "adamw_sf"}


def _import_class(path: str) -> type:
    module, _, name = path.rpartition(".")
    try:
        return getattr(importlib.import_module(module), name)
    except ImportError as e:  # pragma: no cover - depends on optional packages
        raise ImportError(f"optimizer {path!r} needs an optional dependency: {e}") from e


def build_optimizer(cfg: OptimizerConfig, param_groups: list[dict[str, Any]]) -> Optimizer:
    key = cfg.type.lower()
    path = _BUILTIN.get(key, cfg.type)
    cls = _import_class(path)
    kwargs: dict[str, Any] = {"lr": cfg.lr, "weight_decay": cfg.weight_decay}
    if key in ("adamw", "adam", "adamw8bit", "prodigy", "prodigy_plus_sf", "adamw_sf", "came"):
        kwargs["betas"] = tuple(cfg.betas)
    if key in ("adamw", "adam", "adamw8bit", "adamw_sf"):
        kwargs["eps"] = cfg.eps
    if key == "adafactor":
        kwargs.update({"scale_parameter": False, "relative_step": False, "warmup_init": False})
    if key == "sgd":
        kwargs.pop("weight_decay", None)
        kwargs["weight_decay"] = cfg.weight_decay
    kwargs.update(cfg.args)
    groups = [dict(g) for g in param_groups]
    for g in groups:
        g.pop("name", None)
    opt = cls(groups, **kwargs)
    if cfg.kahan:
        opt = KahanWrapper(opt)
    return opt


def is_schedule_free(cfg: OptimizerConfig) -> bool:
    return cfg.type.lower() in SCHEDULE_FREE or "schedulefree" in cfg.type.lower()


class KahanWrapper(Optimizer):
    """Kahan summation for low-precision parameters.

    The inner optimizer updates fp32 shadows of every bf16/fp16 parameter; the shadow is then
    rounded back into the low-precision tensor and the rounding error is carried to the next
    step. Parameters already in fp32 are passed through untouched.
    """

    def __init__(self, inner: Optimizer):
        self.inner = inner
        self.param_groups = inner.param_groups
        self.defaults = inner.defaults
        self.state = inner.state
        self._shadow: dict[Tensor, Tensor] = {}
        self._comp: dict[Tensor, Tensor] = {}
        for g in self.param_groups:
            for p in g["params"]:
                if p.dtype in (torch.bfloat16, torch.float16):
                    self._shadow[p] = p.detach().to(torch.float32).clone()
                    self._comp[p] = torch.zeros_like(self._shadow[p])

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> Any:
        # Run the inner step on fp32 shadows so state is accumulated in full precision.
        swapped = []
        for p, shadow in self._shadow.items():
            if p.grad is None:
                continue
            low = p.data
            p.data = shadow
            p.grad = p.grad.to(torch.float32)
            swapped.append((p, low))
        loss = self.inner.step(closure)
        for p, low in swapped:
            shadow = p.data
            comp = self._comp[p]
            # value we want to store = shadow; representable = round(shadow + comp)
            target = shadow + comp
            new_low = target.to(low.dtype)
            comp.copy_(target - new_low.to(torch.float32))
            low.copy_(new_low)
            p.data = low
            p.grad = None
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.inner.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        sd = self.inner.state_dict()
        sd["kahan"] = {
            "shadow": [s.clone() for s in self._shadow.values()],
            "comp": [c.clone() for c in self._comp.values()],
        }
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        kahan = state_dict.pop("kahan", None)
        self.inner.load_state_dict(state_dict)
        if kahan:
            for s, saved in zip(self._shadow.values(), kahan["shadow"], strict=True):
                s.copy_(saved)
            for c, saved in zip(self._comp.values(), kahan["comp"], strict=True):
                c.copy_(saved)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


# --------------------------------------------------------------------------- schedulers


def _resolve_steps(value: float | None, total: int) -> int:
    if value is None:
        return 0
    return int(round(value * total)) if 0 < value < 1 else int(value)


def build_scheduler(cfg: SchedulerConfig, optimizer: Optimizer, total_steps: int) -> LambdaLR:
    warmup = _resolve_steps(cfg.warmup_steps, total_steps)
    floor = cfg.min_lr_ratio
    decay_steps = (
        _resolve_steps(cfg.decay_steps, total_steps)
        if cfg.decay_steps is not None
        else max(1, total_steps // 10)
    )
    main = max(1, total_steps - warmup)

    def lam(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        p = min(1.0, (step - warmup) / main)
        if cfg.type == "constant":
            return 1.0
        if cfg.type == "linear":
            return floor + (1 - floor) * (1 - p)
        if cfg.type == "cosine":
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))
        if cfg.type == "cosine_restarts":
            pc = (p * cfg.num_cycles) % 1.0
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * pc))
        if cfg.type == "polynomial":
            return floor + (1 - floor) * (1 - p) ** cfg.power
        if cfg.type == "rex":  # reflected exponential: slow start, fast end
            return floor + (1 - floor) * (1 - p) / (0.5 + 0.5 * (1 - p))
        if cfg.type == "warmup_stable_decay":
            stable_end = max(0, total_steps - warmup - decay_steps)
            s = step - warmup
            if s < stable_end:
                return 1.0
            q = min(1.0, (s - stable_end + 1) / max(1, decay_steps))
            return floor + (1 - floor) * (1 - q)
        raise ValueError(cfg.type)

    return LambdaLR(optimizer, lam)
