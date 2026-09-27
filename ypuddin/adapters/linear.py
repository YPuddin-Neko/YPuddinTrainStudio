"""``AdaptedLinear``: a real submodule that replaces the target ``nn.Linear``."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import AdapterModule
from .dora import DoRA
from .frozen import FrozenLinear

Mode = str  # "bypass" | "merged"


def resolve_mode(requested: str, adapter: AdapterModule, base: FrozenLinear, dora: bool) -> Mode:
    if requested not in ("auto", "bypass", "merged"):
        raise ValueError(f"unknown adapter mode {requested!r}")
    if getattr(adapter, "needs_bypass", False) and (dora or requested == "merged"):
        # T-LoRA masks ranks per sample, which one merged weight cannot express.
        raise ValueError(
            f"{adapter.kind} changes its ranks per sample and must run in bypass mode, without DoRA"
        )
    if dora or not adapter.supports_bypass:
        if requested == "bypass":
            reason = "DoRA needs the merged weight" if dora else f"{adapter.kind} has no bypass form"
            raise ValueError(f"bypass mode unavailable: {reason}")
        return "merged"
    return "bypass" if requested == "auto" else requested


class AdaptedLinear(nn.Module):
    def __init__(
        self,
        base: FrozenLinear,
        adapter: AdapterModule,
        *,
        mode: str = "auto",
        dora: bool = False,
        dora_axis: str = "output",
        module_dropout: float = 0.0,
        name: str = "",
    ) -> None:
        super().__init__()
        if (base.out_features, base.in_features) != (adapter.out_features, adapter.in_features):
            raise ValueError("adapter shape does not match the base layer")
        self.base = base
        self.adapter = adapter
        self.name = name
        self.module_dropout_p = float(module_dropout)
        self.multiplier = 1.0
        self.dora = (
            DoRA(base.dequant(torch.float32), dtype=adapter.param_dtype, axis=dora_axis) if dora else None
        )
        self.mode: Mode = resolve_mode(mode, adapter, base, dora)
        # Full keeps W₀; OrthoLoRA takes its principal subspace; T-LoRA draws its start on the layer's device.
        bind = getattr(adapter, "bind_base", None)
        if callable(bind):
            bind(base.dequant(torch.float32))

    # ----------------------------------------------------------------- nn.Linear-compatible surface
    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> Tensor:
        return self.base.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base.bias

    # ----------------------------------------------------------------- forward
    def merged_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        w = self.base.dequant(torch.float32) + self.multiplier * self.adapter().to(torch.float32)
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
        w = self.merged_weight(x.dtype)
        b = None if self.base.bias is None else self.base.bias.to(x.dtype)
        return F.linear(x, w, b)

    def extra_repr(self) -> str:
        return f"name={self.name}, mode={self.mode}, algo={self.adapter.kind}, dora={self.dora is not None}"
