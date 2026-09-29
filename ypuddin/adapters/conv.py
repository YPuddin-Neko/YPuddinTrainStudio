"""``AdaptedConv``: a real submodule that replaces a target ``nn.Conv1d``/``Conv2d``/``Conv3d``.

The frozen base stays the original convolution, so its stride, padding, dilation and padding mode
are applied unchanged. Bypass adds the adapter's own convolution; merged convolves once with
``W₀ + ΔW``. Grouped convolutions only run merged.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import AdapterModule, ConvGeometry, probability
from .dora import DoRA
from .linear import Mode, merged_bias, resolve_mode

CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)


class AdaptedConv(nn.Module):
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
    ) -> None:
        super().__init__()
        if not isinstance(base, CONV_TYPES):
            raise TypeError(f"AdaptedConv wraps a convolution, got {type(base).__name__}")
        if tuple(base.weight.shape) != adapter.weight_shape:
            raise ValueError("adapter shape does not match the base layer")
        base.requires_grad_(False)
        self.base = base
        self.adapter = adapter
        self.name = name
        self.module_dropout_p = probability("module_dropout", module_dropout)
        self.multiplier = 1.0
        weight = base.weight.detach().to(torch.float32)
        self.dora = DoRA(weight, dtype=adapter.param_dtype, axis=dora_axis) if dora else None
        self.mode: Mode = resolve_mode(mode, adapter, base, dora, grouped=base.groups != 1)
        adapter.bind_geometry(ConvGeometry.of(base))
        bind = getattr(adapter, "bind_base", None)
        if callable(bind):
            bind(weight, bias=base.bias)

    # ----------------------------------------------------------------- nn.Conv-compatible surface
    @property
    def in_channels(self) -> int:
        return self.base.in_channels

    @property
    def out_channels(self) -> int:
        return self.base.out_channels

    @property
    def kernel_size(self) -> tuple[int, ...]:
        return self.base.kernel_size

    @property
    def groups(self) -> int:
        return self.base.groups

    @property
    def weight(self) -> Tensor:
        return self.base.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base.bias

    def frozen_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        return self.base.weight if dtype is None else self.base.weight.to(dtype)

    # ----------------------------------------------------------------- forward
    def merged_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        w = self.base.weight.to(torch.float32) + self.multiplier * self.adapter().to(torch.float32)
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
        return self.base._conv_forward(x, self.merged_weight(x.dtype), bias)

    def extra_repr(self) -> str:
        return f"name={self.name}, mode={self.mode}, algo={self.adapter.kind}, dora={self.dora is not None}"
