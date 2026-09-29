"""``AdaptedConv``: replaces a target ``nn.Conv1d``/``Conv2d``/``Conv3d``.

The frozen base stays the original convolution, so its stride, padding, dilation and padding mode
apply unchanged. Grouped convolutions only run merged.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .base import AdapterModule, ConvGeometry
from .linear import AdaptedLayer

CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)


class AdaptedConv(AdaptedLayer):
    def __init__(self, base: nn.Module, adapter: AdapterModule, **options: Any) -> None:
        if not isinstance(base, CONV_TYPES):
            raise TypeError(f"AdaptedConv wraps a convolution, got {type(base).__name__}")
        if tuple(base.weight.shape) != adapter.weight_shape:
            raise ValueError("adapter shape does not match the base layer")
        base.requires_grad_(False)
        adapter.bind_geometry(ConvGeometry.of(base))
        super().__init__(base, adapter, grouped=base.groups != 1, **options)

    def frozen_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        return self.base.weight if dtype is None else self.base.weight.to(dtype)

    def _layer_op(self, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        return self.base._conv_forward(x, weight, bias)

    def unwrapped(self) -> nn.Module:
        return self.base
