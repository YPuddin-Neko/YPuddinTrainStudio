"""Muon matrix updates and auxiliary AdamW with FP32 optimizer state.

The Newton–Schulz kernel follows Keller Jordan's MIT-licensed implementation.
See MUON_NOTICE.md for the fixed source and integration choices.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer


def _orthogonalize(update: Tensor, steps: int) -> Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = update.bfloat16()
    transposed = update.shape[0] > update.shape[1]
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        polynomial = b * gram + c * gram @ gram
        x = a * x + polynomial @ x
    if transposed:
        x = x.mT
    x *= max(1, update.shape[0] / update.shape[1]) ** 0.5
    return x


class Muon(Optimizer):
    """Apply Muon to explicit matrix groups and AdamW to auxiliary groups.

    Each group supplies ``muon_role`` and ordered ``param_names``. Group learning
    rates are final rates; auxiliary scaling belongs to the parameter grouper.
    """

    _FORMAT = {"algorithm": "muon_with_aux_adamw", "version": 1}
    _IDENTITY_KEYS = (
        "name",
        "muon_role",
        "param_names",
        "muon_shapes",
        "muon_dtypes",
        "muon_base_lr",
        "momentum",
        "nesterov",
        "ns_steps",
        "weight_decay",
        "betas",
        "eps",
        "aux_lr_ratio",
    )

    def __init__(
        self,
        params: Iterable[dict[str, Any]],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
        self._validate_options(defaults)
        super().__init__(params, defaults)

    @staticmethod
    def _validate_options(group: dict[str, Any]) -> None:
        if not isinstance(group, dict) or not {
            "lr",
            "weight_decay",
            "momentum",
            "eps",
            "nesterov",
            "ns_steps",
            "betas",
        }.issubset(group):
            raise ValueError("Muon group is missing required optimizer settings")
        for key in ("lr", "weight_decay", "momentum", "eps"):
            value = group[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Muon {key} must be a finite number")
        if group["lr"] < 0 or group["weight_decay"] < 0 or group["eps"] <= 0:
            raise ValueError("Muon requires nonnegative lr/weight_decay and positive eps")
        if not 0 <= group["momentum"] < 1 or not isinstance(group["nesterov"], bool):
            raise ValueError("Muon requires momentum in [0, 1) and boolean nesterov")
        if type(group["ns_steps"]) is not int or not 1 <= group["ns_steps"] <= 10:
            raise ValueError("Muon ns_steps must be an integer from 1 to 10")
        betas = group["betas"]
        if (
            not isinstance(betas, (tuple, list))
            or len(betas) != 2
            or any(
                isinstance(beta, bool)
                or not isinstance(beta, (int, float))
                or not math.isfinite(beta)
                or not 0 <= beta < 1
                for beta in betas
            )
        ):
            raise ValueError("Muon AdamW betas must contain two values in [0, 1)")

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if not isinstance(param_group, dict):
            raise ValueError("Muon requires explicit parameter groups")
        group = {**self.defaults, **param_group}
        self._validate_options(group)
        if group.get("muon_role") not in ("muon", "adamw"):
            raise ValueError("Muon groups require muon_role='muon' or 'adamw'")
        parameters = list(group["params"])
        names = group.get("param_names")
        if (
            not isinstance(names, (tuple, list))
            or len(names) != len(parameters)
            or any(not isinstance(name, str) or not name for name in names)
        ):
            raise ValueError("Muon groups require one stable param_names entry per parameter")
        old_ids = {id(p) for old in self.param_groups for p in old["params"]}
        old_names = {name for old in self.param_groups for name in old["param_names"]}
        if len(set(names)) != len(names) or old_names.intersection(names):
            raise ValueError("Muon parameter names must be unique")
        for parameter in parameters:
            if not isinstance(parameter, Tensor) or parameter.layout != torch.strided:
                raise ValueError("Muon requires dense parameters")
            if parameter.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise ValueError("Muon supports only FP32, FP16, and BF16 parameters")
            if not parameter.numel() or (group["muon_role"] == "muon" and parameter.ndim != 2):
                raise ValueError("Muon matrix groups require nonempty 2D parameters")
            if id(parameter) in old_ids:
                raise ValueError("Muon does not accept duplicate parameters")
            old_ids.add(id(parameter))
        if not isinstance(group.get("name", ""), str):
            raise ValueError("Muon group name must be a string")
        group.update(
            params=parameters,
            param_names=list(names),
            name=group.get("name", ""),
            betas=tuple(group["betas"]),
            muon_base_lr=group["lr"],
            muon_shapes=[list(p.shape) for p in parameters],
            muon_dtypes=[str(p.dtype) for p in parameters],
        )
        super().add_param_group(group)

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Validate every active gradient before creating state or updating weights.
        finite_by_device: dict[torch.device, list[Tensor]] = {}
        for group in self.param_groups:
            self._validate_options(group)
            if group["lr"] == 0:
                continue
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None or not parameter.requires_grad:
                    continue
                if grad.layout != torch.strided:
                    raise RuntimeError("Muon does not support sparse gradients")
                finite_by_device.setdefault(grad.device, []).append(self._finite_input(parameter, group))
        if any(not bool(torch.stack(values).all()) for values in finite_by_device.values()):
            raise ValueError("Muon received a non-finite gradient or BF16 update; no parameters were updated")
        del finite_by_device
        for group in self.param_groups:
            if group["lr"] == 0:
                continue
            for parameter in group["params"]:
                if parameter.grad is None or not parameter.requires_grad:
                    continue
                self._update(parameter, group)
        return loss

    def _finite_input(self, parameter: Tensor, group: dict[str, Any]) -> Tensor:
        grad = parameter.grad.detach()
        finite = torch.isfinite(grad).all()
        if group["muon_role"] == "muon":
            grad = grad.float()
            saved_momentum = self.state.get(parameter, {}).get("momentum_buffer")
            next_momentum = torch.zeros_like(grad) if saved_momentum is None else saved_momentum.clone()
            next_momentum.lerp_(grad, 1 - group["momentum"])
            update = grad.lerp(next_momentum, group["momentum"]) if group["nesterov"] else next_momentum
            # Finite FP32 values can overflow when rounded to BF16. Check the
            # actual next NS input without advancing the saved momentum.
            finite = finite & torch.isfinite(update.bfloat16()).all()
        return finite

    def _update(self, parameter: Tensor, group: dict[str, Any]) -> None:
        grad = parameter.grad.detach().float()
        state = self.state[parameter]
        if not state:
            state["step"] = 0
            keys = ("momentum_buffer",) if group["muon_role"] == "muon" else ("exp_avg", "exp_avg_sq")
            for key in keys:
                state[key] = torch.zeros_like(parameter, dtype=torch.float32)
            if parameter.dtype != torch.float32:
                state["master_param"] = parameter.detach().float().clone()
        state["step"] += 1
        if group["muon_role"] == "muon":
            momentum = state["momentum_buffer"]
            momentum.lerp_(grad, 1 - group["momentum"])
            update = grad.lerp(momentum, group["momentum"]) if group["nesterov"] else momentum
            update = _orthogonalize(update, group["ns_steps"])
        else:
            beta1, beta2 = group["betas"]
            state["exp_avg"].lerp_(grad, 1 - beta1)
            state["exp_avg_sq"].lerp_(grad.square(), 1 - beta2)
            numerator = state["exp_avg"] / (1 - beta1 ** state["step"])
            denominator = (state["exp_avg_sq"] / (1 - beta2 ** state["step"])).sqrt_().add_(group["eps"])
            update = numerator.div_(denominator)
        data = state.get("master_param", parameter)
        data.mul_(1 - group["lr"] * group["weight_decay"])
        data.add_(update, alpha=-group["lr"])
        if data is not parameter:
            parameter.copy_(data)

    def state_dict(self) -> dict[str, Any]:
        result = super().state_dict()
        result["muon_format"] = dict(self._FORMAT)
        return result

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("muon_format") != self._FORMAT:
            raise ValueError("Checkpoint does not contain a compatible Muon optimizer format")
        saved_groups = state_dict.get("param_groups")
        saved_states = state_dict.get("state")
        if (
            not isinstance(saved_groups, list)
            or len(saved_groups) != len(self.param_groups)
            or not isinstance(saved_states, dict)
        ):
            raise ValueError("Muon checkpoint parameter groups do not match")
        restored = defaultdict(dict)
        groups = []
        seen = set()
        for saved, current in zip(saved_groups, self.param_groups, strict=True):
            self._validate_options(saved)
            for key in self._IDENTITY_KEYS:
                if saved.get(key) != current.get(key):
                    raise ValueError(f"Muon checkpoint {key} does not match")
            if saved.get("initial_lr", saved["muon_base_lr"]) != saved["muon_base_lr"]:
                raise ValueError("Muon checkpoint initial_lr does not match the base learning rate")
            ids = saved.get("params")
            if not isinstance(ids, list) or len(ids) != len(current["params"]):
                raise ValueError("Muon checkpoint parameter count does not match")
            if current["muon_shapes"] != [list(p.shape) for p in current["params"]] or current[
                "muon_dtypes"
            ] != [str(p.dtype) for p in current["params"]]:
                raise ValueError("Muon parameter shape or dtype changed after construction")
            for saved_id, parameter in zip(ids, current["params"], strict=True):
                if type(saved_id) is not int or saved_id in seen:
                    raise ValueError("Muon checkpoint contains duplicate or invalid parameter IDs")
                seen.add(saved_id)
                if saved_id not in saved_states:
                    continue
                values = saved_states[saved_id]
                keys = (
                    {"step", "momentum_buffer"}
                    if current["muon_role"] == "muon"
                    else {"step", "exp_avg", "exp_avg_sq"}
                )
                if parameter.dtype != torch.float32:
                    keys.add("master_param")
                if not isinstance(values, dict) or set(values) != keys:
                    raise ValueError("Muon checkpoint state fields do not match the parameter role")
                if type(values["step"]) is not int or values["step"] < 1:
                    raise ValueError("Muon checkpoint step must be a positive integer")
                target = {"step": values["step"]}
                for key in keys - {"step"}:
                    value = values[key]
                    if (
                        not isinstance(value, Tensor)
                        or value.layout != torch.strided
                        or value.dtype != torch.float32
                        or value.shape != parameter.shape
                    ):
                        raise ValueError(f"Muon checkpoint {key} shape or FP32 dtype does not match")
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError(f"Muon checkpoint {key} contains non-finite values")
                    if key == "exp_avg_sq" and bool((value < 0).any()):
                        raise ValueError("Muon checkpoint second moment must be nonnegative")
                    target[key] = value.detach().to(device=parameter.device).clone()
                restored[parameter] = target
            group = deepcopy({key: value for key, value in saved.items() if key != "params"})
            group["params"] = current["params"]
            groups.append(group)
        if set(saved_states) - seen:
            raise ValueError("Muon checkpoint contains state for unknown parameters")
        # Avoid Optimizer.load_state_dict casting FP32 state through low-precision weights.
        self.__setstate__({"state": restored, "param_groups": groups})
