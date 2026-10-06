"""Optimizer factory, Kahan-compensated bf16 wrapper and LR schedulers."""

from __future__ import annotations

import importlib
import math
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ypuddin.config import OptimizerConfig, SchedulerConfig
from ypuddin.config.optimizer_rules import optimizer_key, optimizer_policy, optimizer_specific_fields

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
    "automagic": "ypuddin.optim.automagic.Automagic",
}

SCHEDULE_FREE = {"prodigy_plus_sf", "adamw_sf"}


def optimizer_packages() -> dict[str, str]:
    """The top-level package each built-in optimizer imports, for those PyTorch and YPuddin do not ship."""
    packages = {key: path.split(".", 1)[0] for key, path in _BUILTIN.items()}
    return {key: package for key, package in packages.items() if package not in {"torch", "ypuddin"}}


def _validate_managed_learning_rates(key: str, lr: float, groups: list[dict[str, Any]]) -> None:
    expected = optimizer_policy(key).get("fixed", {}).get("optimizer.lr")
    if expected is None:
        return
    if lr != expected:
        raise ValueError(f"{key} uses the managed learning-rate setting {expected:g}")
    for group in groups:
        group_lr = group.get("lr", lr)
        if group_lr not in (0, expected):
            name = group.get("name", "unnamed")
            raise ValueError(
                f"{key} parameter group {name!r} overrides the managed learning rate; "
                f"remove per-group/per-layer rates and adapter LR scaling (expected {expected:g})"
            )


def _import_class(path: str) -> type:
    module, _, name = path.rpartition(".")
    try:
        return getattr(importlib.import_module(module), name)
    except ImportError as e:  # pragma: no cover - depends on optional packages
        raise ImportError(f"optimizer {path!r} needs an optional dependency: {e}") from e


def build_optimizer(cfg: OptimizerConfig, param_groups: list[dict[str, Any]]) -> Optimizer:
    # Assignment validation cannot observe in-place changes to args/group_lr.
    cfg = OptimizerConfig.model_validate(cfg.model_dump())
    if cfg.fused_backward or cfg.args.get("fused_back_pass", False):
        raise ValueError("optimizer.fused_backward is not implemented; use the normal optimizer step")
    key = optimizer_key(cfg.type)
    if getattr(cfg, "cpu_offload", False):
        if key != "adamw" or cfg.kahan:
            raise ValueError("CPU optimizer offload currently requires AdamW without Kahan compensation")
    if cfg.kahan and (is_schedule_free(cfg) or key == "prodigy_plus_sf"):
        raise ValueError("optimizer.kahan cannot be combined with a schedule-free optimizer")
    if key == "automagic" and cfg.kahan:
        raise ValueError("Automagic already preserves low-precision updates with FP32 master weights")
    path = _BUILTIN.get(key, cfg.type)
    cls = _import_class(path)
    kwargs: dict[str, Any] = {"lr": cfg.lr, "weight_decay": cfg.weight_decay}
    if key in (
        "adamw",
        "adam",
        "adamw8bit",
        "lion",
        "lion8bit",
        "prodigy",
        "prodigy_plus_sf",
        "adamw_sf",
        "came",
    ):
        kwargs["betas"] = tuple(cfg.betas)
    if key == "came":
        # The shared beta control has two values; CAME additionally requires
        # its confidence-statistic decay (the upstream default is 0.9999).
        kwargs["betas"] = (*cfg.betas, 0.9999)
    if key in ("adamw", "adam", "adamw8bit", "adamw_sf", "prodigy", "prodigy_plus_sf", "automagic"):
        kwargs["eps"] = cfg.eps
    if key == "adafactor":
        kwargs.update({"scale_parameter": False, "relative_step": False, "warmup_init": False})
    if key == "sgd":
        kwargs.pop("weight_decay", None)
        kwargs["weight_decay"] = cfg.weight_decay
    kwargs.update(cfg.args)
    for name in optimizer_specific_fields(key):
        # Known fields have been migrated out of legacy args by config validation.
        if hasattr(cfg, name):
            value = getattr(cfg, name)
            kwargs[name] = math.inf if name == "growth_rate" and value is None else value
    groups = [dict(g) for g in param_groups]
    _validate_managed_learning_rates(key, kwargs["lr"], groups)
    if getattr(cfg, "cpu_offload", False):
        from .cpu_offload import CPUOffloadAdamW

        opt = CPUOffloadAdamW(groups, **kwargs)
    else:
        opt = cls(groups, **kwargs)
    validate_optimizer_runtime(cfg, opt)
    if cfg.kahan:
        opt = KahanWrapper(opt)
    return opt


def load_optimizer_state(cfg: OptimizerConfig, optimizer: Optimizer, state: dict[str, Any]) -> None:
    """Load PPSF statistics without PyTorch casting them to the parameter dtype.

    PPSF deliberately keeps p0/s in BF16 even with FP32 parameters. The generic
    loader promotes these buffers, changing subsequent accumulator rounding.
    Keep the loader's device placement but restore the saved tensor precision.
    """
    optimizer.load_state_dict(state)
    if optimizer_key(cfg.type) != "prodigy_plus_sf":
        return
    for saved_group, group in zip(state["param_groups"], optimizer.param_groups, strict=True):
        for saved_id, parameter in zip(saved_group["params"], group["params"], strict=True):
            for name, saved in state["state"].get(saved_id, {}).items():
                current = optimizer.state[parameter].get(name)
                if isinstance(saved, Tensor) and isinstance(current, Tensor):
                    optimizer.state[parameter][name] = saved.to(device=current.device, copy=True)


def is_schedule_free(cfg: OptimizerConfig) -> bool:
    key = optimizer_key(cfg.type)
    if key == "prodigy_plus_sf":
        return bool(getattr(cfg, "use_schedulefree", cfg.args.get("use_schedulefree", True)))
    return key in SCHEDULE_FREE or "schedulefree" in key


def manages_learning_rate(cfg: OptimizerConfig) -> bool:
    """Whether the optimizer must run without an external LR scheduler."""
    return optimizer_key(cfg.type) == "automagic" or is_schedule_free(cfg)


def optimizer_rate_snapshot(optimizer: Optimizer) -> list[dict[str, Any]]:
    """Capture scalar group settings before optimizers advance their D estimates."""
    keys = ("lr", "d", "shared_d", "split_groups", "split_groups_mean", "betas", "use_bias_correction", "k")
    return [{key: group[key] for key in keys if key in group} for group in optimizer.param_groups]


def optimizer_learning_rates(
    cfg: OptimizerConfig,
    optimizer: Optimizer,
    *,
    before_step: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Report adaptive scalar rates, using the completed update when snapshotted.

    These are optimizer step-size scales, not a measurement of weight changes.
    Automagic reports its mean coordinate rate. PPSF includes its SF mixing
    factor; Prodigy includes bias correction when enabled.
    """
    key = optimizer_key(cfg.type)
    if key == "automagic":
        rates = optimizer.get_learning_rates()
    else:
        rates = []
        for index, group in enumerate(optimizer.param_groups):
            source = before_step[index] if before_step is not None else group
            rate = float(source["lr"])
            if key == "prodigy":
                rate *= float(source.get("d", 0))
                if source.get("use_bias_correction"):
                    beta1, beta2 = source["betas"]
                    step = int(source.get("k", 0)) + 1
                    rate *= math.sqrt(1 - beta2**step) / (1 - beta1**step)
            elif key == "prodigy_plus_sf":
                shared = source.get("shared_d")
                use_shared = (
                    source.get("split_groups") and source.get("split_groups_mean") and shared is not None
                )
                d = shared if use_shared else source.get("d", 0)
                rate = float(d) * float(group.get("effective_lr", rate))
            rates.append(rate)
    names = [str(group.get("name", index)) for index, group in enumerate(optimizer.param_groups)]
    counts = Counter(names)
    result = {}
    for index, (name, rate) in enumerate(zip(names, rates, strict=True)):
        key = name
        if counts[name] > 1:
            key = f"{name}/{index + 1}"
            while key in counts or key in result:
                key += f"/{index + 1}"
        result[key] = rate
    return result


def optimizer_hyperparameter_snapshot(cfg: OptimizerConfig, optimizer: Optimizer) -> list[dict[str, Any]]:
    """Preserve the requested per-group settings before loading checkpoint state."""
    keys = {"name", "betas", "eps", "weight_decay", "optimiser_version"}
    keys.update(optimizer_specific_fields(cfg.type))
    keys.update(cfg.args)
    # A scheduler changes group.lr; its initial_lr remains the requested base.
    keys.discard("lr")
    return [{key: deepcopy(group[key]) for key in keys if key in group} for group in optimizer.param_groups]


def validate_optimizer_runtime(
    cfg: OptimizerConfig,
    optimizer: Optimizer,
    *,
    expected_groups: list[dict[str, Any]] | None = None,
) -> None:
    """Reject checkpoint overrides of protected settings before another update."""
    key = optimizer_key(cfg.type)
    groups = optimizer.param_groups
    if not manages_learning_rate(cfg):
        # A supported cosine/linear scheduler legitimately changes the current
        # rate. Validate its protected base, not its resumed position on the curve.
        groups = [{**group, "lr": group.get("initial_lr", group["lr"])} for group in groups]
    _validate_managed_learning_rates(key, cfg.lr, groups)
    for group in optimizer.param_groups:
        if key in ("prodigy", "prodigy_plus_sf") and not {"d", "d0", "k"} <= group.keys():
            raise ValueError("Checkpoint optimizer does not match the selected adaptive optimizer")
        if key == "prodigy_plus_sf":
            if group.get("optimiser_version") != getattr(optimizer, "VERSION", None):
                raise ValueError(
                    "Checkpoint PPSF optimizer version is incompatible; load adapter weights to start a new optimizer"
                )
            if group.get("use_schedulefree") != is_schedule_free(cfg):
                raise ValueError(
                    "Checkpoint Schedule-Free mode differs; resume with the original optimizer mode"
                )
            if group.get("fused_back_pass", False):
                raise ValueError("Checkpoint requests fused backward, which is not supported")
    if expected_groups is not None:
        for expected, group in zip(expected_groups, optimizer.param_groups, strict=True):
            for name, value in expected.items():
                actual = group.get(name)
                if isinstance(actual, (tuple, list)) and isinstance(value, (tuple, list)):
                    actual, value = tuple(actual), tuple(value)
                if actual != value:
                    raise ValueError(
                        f"Checkpoint optimizer setting {name!r} differs from the current configuration; "
                        "resume with the original settings, or load adapter weights to start a new optimizer"
                    )


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
        state_dict = dict(state_dict)
        kahan = state_dict.pop("kahan", None)
        # Optimizer.load_state_dict casts moments to the current parameter dtype. Our optimizer
        # actually updates fp32 shadows, so loading against bf16 parameters would destroy precision
        # and fail the following Adam step with mixed moment/gradient dtypes.
        originals = [(p, p.data) for p in self._shadow]
        try:
            for p, _low in originals:
                p.data = self._shadow[p]
            self.inner.load_state_dict(state_dict)
        finally:
            for p, low in originals:
                p.data = low
        self.param_groups = self.inner.param_groups
        self.state = self.inner.state
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
