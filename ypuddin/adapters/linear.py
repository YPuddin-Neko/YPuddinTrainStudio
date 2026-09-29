"""``AdaptedLayer``: a frozen layer plus its adapter; ``AdaptedLinear`` replaces a target ``nn.Linear``."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import AdapterModule, probability
from .dora import DoRA
from .frozen import FrozenLinear

Mode = str  # "bypass" | "merged"


def resolve_mode(requested: str, adapter: AdapterModule, dora: bool, *, grouped: bool = False) -> Mode:
    if requested not in ("auto", "bypass", "merged"):
        raise ValueError(f"unknown adapter mode {requested!r}")
    if adapter.needs_bypass and (dora or requested == "merged" or grouped):
        # T-LoRA masks ranks per sample, which one merged weight cannot express.
        raise ValueError(
            f"{adapter.kind} changes its ranks per sample and must run in bypass mode, without DoRA"
            + (" or grouped convolutions" if grouped else "")
        )
    if dora:
        merged_only = "DoRA needs the merged weight"
    elif grouped:
        merged_only = "a grouped convolution needs the merged weight"
    elif not adapter.supports_bypass:
        merged_only = f"{adapter.kind} has no bypass form"
    else:
        return "bypass" if requested == "auto" else requested
    if requested == "bypass":
        raise ValueError(f"bypass mode unavailable: {merged_only}")
    return "merged"


def merged_bias(
    bias: Tensor | None, adapter: AdapterModule, multiplier: float, dtype: torch.dtype
) -> Tensor | None:
    """The layer bias plus the adapter's own bias delta (LyCORIS Full trains the bias)."""
    delta = adapter.delta_bias()
    if delta is None:
        return None if bias is None else bias.to(dtype)
    return (bias.to(torch.float32) + multiplier * delta.to(torch.float32)).to(dtype)


class AdaptedLayer(nn.Module):
    """Runs the adapter in bypass (``base(x) + adapter(x)``) or merged (``W₀ + ΔW``) mode, with DoRA,
    module dropout and a multiplier. Subclasses provide the frozen weight and the layer's operation."""

    def __init__(
        self,
        base: nn.Module,
        adapter: AdapterModule,
        *,
        mode: str = "auto",
        dora: bool = False,
        dora_axis: str = "output",
        module_dropout: float = 0.0,
        name: str = "",
        grouped: bool = False,
    ) -> None:
        super().__init__()
        self.base = base
        self.adapter = adapter
        self.name = name
        self.module_dropout_p = probability("module_dropout", module_dropout)
        self.multiplier = 1.0
        self.mode: Mode = resolve_mode(mode, adapter, dora, grouped=grouped)
        # Full keeps W₀ and b₀; OrthoLoRA takes its principal subspace; T-LoRA draws its start on the layer's device.
        bind = getattr(adapter, "bind_base", None)
        weight = self.frozen_weight(torch.float32) if dora or bind is not None else None
        self.dora = DoRA(weight, dtype=adapter.param_dtype, axis=dora_axis) if dora else None
        if bind is not None:
            bind(weight, bias=base.bias)

    def frozen_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        """The frozen weight in ``dtype``; without one, as stored (FP8 is dequantized to FP32)."""
        raise NotImplementedError

    def _layer_op(self, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        """The layer's own operation with ``weight`` and ``bias``."""
        raise NotImplementedError

    def unwrapped(self) -> nn.Module:
        """A plain module equal to the frozen layer."""
        raise NotImplementedError

    @property
    def weight(self) -> Tensor:
        return self.base.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base.bias

    # ----------------------------------------------------------------- forward
    def merged_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        delta = self.adapter().float()
        if self.multiplier != 1.0:
            delta = delta * self.multiplier
        # Adding the stored weight to the FP32 delta promotes it without an FP32 copy.
        w = self.frozen_weight() + delta
        if self.dora is not None:
            w = self.dora(w)
        return w if dtype is None else w.to(dtype)

    def forward(self, x: Tensor) -> Tensor:
        if self.module_dropout_p > 0 and self.training and torch.rand(()).item() < self.module_dropout_p:
            return self.base(x)
        if self.multiplier == 0.0 and self.dora is None:
            return self.base(x)
        if self.mode == "bypass":
            delta = self.adapter(x)
            if self.multiplier != 1.0:
                delta = delta * self.multiplier
            return self.base(x) + delta
        bias = merged_bias(self.base.bias, self.adapter, self.multiplier, x.dtype)
        return self._layer_op(x, self.merged_weight(x.dtype), bias)

    def extra_repr(self) -> str:
        return f"name={self.name}, mode={self.mode}, algo={self.adapter.kind}, dora={self.dora is not None}"


class AdaptedLinear(AdaptedLayer):
    def __init__(self, base: FrozenLinear, adapter: AdapterModule, **options: Any) -> None:
        if tuple(base.weight.shape) != adapter.weight_shape:
            raise ValueError("adapter shape does not match the base layer")
        super().__init__(base, adapter, **options)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def frozen_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        return self.base.dequant(torch.float32 if dtype is None and self.base.is_fp8 else dtype)

    def _layer_op(self, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        return F.linear(x, weight, bias)

    def unwrapped(self) -> nn.Module:
        return self.base.to_linear()
