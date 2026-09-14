"""Automagic v1 with FP32 optimizer state and ordinary accumulated-gradient steps.

Adapted from Ostris' AI Toolkit, MIT License, Copyright (c) 2024 Ostris, LLC.
See NOTICE.md for the pinned source, license, and implementation differences.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer


class Automagic(Optimizer):
    """Adapt each element's learning rate by agreement between update directions.

    Matrix second moments are factored over the final two dimensions. Learning-rate
    masks and moments stay in FP32; low-precision parameters also have FP32 master
    weights. There are no backward hooks, parameter swapping, or quantized states.
    ``lr`` is the initial mask value, not an external learning-rate multiplier.
    """

    manages_learning_rate = True

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-6,
        min_lr: float = 1e-7,
        max_lr: float = 1e-3,
        lr_bump: float = 1e-6,
        eps: float = 1e-30,
        clip_threshold: float = 1.0,
        beta2: float = 0.999,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=lr,
            min_lr=min_lr,
            max_lr=max_lr,
            lr_bump=lr_bump,
            eps=eps,
            clip_threshold=clip_threshold,
            beta2=beta2,
            weight_decay=weight_decay,
        )
        self._validate_group(defaults)
        super().__init__(params, defaults)

    @staticmethod
    def _validate_group(group: dict[str, Any]) -> None:
        for name in ("min_lr", "max_lr", "lr_bump", "eps", "clip_threshold"):
            value = group[name]
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Automagic {name} must be finite and positive")
        if not math.isfinite(group["lr"]) or group["lr"] < 0:
            raise ValueError("Automagic lr must be finite and nonnegative")
        if group["lr"] != 0 and not group["min_lr"] <= group["lr"] <= group["max_lr"]:
            raise ValueError("Automagic requires min_lr <= initial lr <= max_lr")
        if not math.isfinite(group["beta2"]) or not 0 <= group["beta2"] < 1:
            raise ValueError("Automagic beta2 must be in [0, 1)")
        if not math.isfinite(group["weight_decay"]) or group["weight_decay"] < 0:
            raise ValueError("Automagic weight_decay must be finite and nonnegative")

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        group = {**self.defaults, **param_group}
        self._validate_group(group)
        super().add_param_group(param_group)
        for parameter in self.param_groups[-1]["params"]:
            if parameter.dtype not in (torch.float32, torch.bfloat16, torch.float16):
                raise ValueError("Automagic supports float32, bfloat16, and float16 parameters")
            if parameter.numel() == 0:
                raise ValueError("Automagic does not support empty parameters")

    @staticmethod
    def _rms(tensor: Tensor) -> Tensor:
        return tensor.norm(2) / math.sqrt(tensor.numel())

    def _initialize_state(self, parameter: Tensor, group: dict[str, Any]) -> dict[str, Any]:
        state = self.state[parameter]
        state["step"] = 0
        state["lr_mask"] = torch.full_like(parameter, group["lr"], dtype=torch.float32)
        state["last_polarity"] = torch.zeros_like(parameter, dtype=torch.bool)
        if parameter.ndim >= 2:
            state["exp_avg_sq_row"] = torch.zeros(
                parameter.shape[:-1], device=parameter.device, dtype=torch.float32
            )
            state["exp_avg_sq_col"] = torch.zeros(
                parameter.shape[:-2] + parameter.shape[-1:], device=parameter.device, dtype=torch.float32
            )
        else:
            state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
        if parameter.dtype != torch.float32:
            state["master_param"] = parameter.detach().float().clone()
        return state

    def get_learning_rates(self) -> list[float]:
        """Report the element-weighted mean adaptive rate for each parameter group."""
        rates = []
        for group in self.param_groups:
            if group["lr"] == 0:
                rates.append(0.0)
                continue
            sums_by_device: dict[torch.device, list[Tensor]] = {}
            count = 0
            for parameter in group["params"]:
                mask = self.state.get(parameter, {}).get("lr_mask")
                if mask is not None:
                    sums_by_device.setdefault(mask.device, []).append(mask.sum())
                    count += mask.numel()
            # Synchronize once per group/device, not once for every adapter tensor.
            rate_sum = sum(float(torch.stack(sums).sum()) for sums in sums_by_device.values())
            rates.append(rate_sum / count if count else float(group["lr"]))
        return rates

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["lr"] == 0:
                continue
            for parameter in group["params"]:
                if parameter.grad is None or not parameter.requires_grad:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("Automagic does not support sparse gradients")
                grad = parameter.grad.detach().float()
                state = self.state.get(parameter) or self._initialize_state(parameter, group)
                state["step"] += 1
                beta2 = group["beta2"]
                squared = grad.square().add_(group["eps"])
                if parameter.ndim >= 2:
                    row, col = state["exp_avg_sq_row"], state["exp_avg_sq_col"]
                    row.mul_(beta2).add_(squared.mean(dim=-1), alpha=1 - beta2)
                    col.mul_(beta2).add_(squared.mean(dim=-2), alpha=1 - beta2)
                    row_factor = (row / row.mean(dim=-1, keepdim=True)).rsqrt().unsqueeze(-1)
                    update = row_factor * col.rsqrt().unsqueeze(-2) * grad
                else:
                    moment = state["exp_avg_sq"]
                    moment.mul_(beta2).add_(squared, alpha=1 - beta2)
                    update = moment.rsqrt() * grad
                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1))

                polarity = update > 0
                mask = state["lr_mask"]
                # Preserve the upstream first-step rule (previous signs start false).
                mask.add_(
                    torch.where(state["last_polarity"] == polarity, group["lr_bump"], -group["lr_bump"])
                )
                mask.clamp_(min=group["min_lr"], max=group["max_lr"])
                state["last_polarity"].copy_(polarity)
                data = state.get("master_param", parameter)
                if group["weight_decay"]:
                    data.add_(data * mask, alpha=-group["weight_decay"])
                data.addcmul_(update, mask, value=-1)
                if data is not parameter:
                    parameter.copy_(data)
        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Torch normally casts optimizer state to the parameter dtype. Preserve
        # FP32 masks/moments/masters and bool signs from the original serialized
        # tensors, mapped by parameter-group IDs (including parameters without grads).
        super().load_state_dict(state_dict)
        for saved_group, group in zip(state_dict["param_groups"], self.param_groups, strict=True):
            self._validate_group(group)
            for saved_id, parameter in zip(saved_group["params"], group["params"], strict=True):
                saved = state_dict["state"].get(saved_id)
                if not saved:
                    continue
                if "lr_mask" not in saved or "last_polarity" not in saved:
                    raise ValueError("Checkpoint is missing Automagic adaptive-rate state")
                restored = self.state[parameter]
                for key, value in saved.items():
                    if isinstance(value, Tensor):
                        dtype = torch.bool if key == "last_polarity" else torch.float32
                        restored[key] = value.detach().to(device=parameter.device, dtype=dtype).clone()
                if restored["lr_mask"].shape != parameter.shape:
                    raise ValueError("Automagic checkpoint parameter shape does not match")
                if parameter.dtype != torch.float32 and "master_param" not in restored:
                    restored["master_param"] = parameter.detach().float().clone()
                elif parameter.dtype == torch.float32:
                    restored.pop("master_param", None)
