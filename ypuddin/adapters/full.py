"""Full: train the layer weight, and its bias when it has one, directly.

``ΔW = W - W₀`` and ``Δb = b - b₀`` are exported as LyCORIS ``diff`` and ``diff_b``; ComfyUI and
LyCORIS apply both. A convolution's ``diff`` keeps the kernel's shape.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from .base import AdapterModule, resolve_kernel


class Full(AdapterModule):
    kind = "full"
    supports_bypass = False

    def __init__(
        self,
        out_features: int,
        in_features: int,
        *,
        kernel: tuple[int, ...] = (),
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(out_features, in_features, kernel=kernel, dtype=dtype)
        self.weight = nn.Parameter(torch.zeros(self.weight_shape, dtype=dtype))
        self.register_buffer("base_weight", torch.zeros(self.weight_shape, dtype=dtype), persistent=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
            self.register_buffer("base_bias", torch.zeros(out_features, dtype=dtype), persistent=False)
        else:
            self.register_parameter("bias", None)
            self.register_buffer("base_bias", None, persistent=False)
        self._pending_diff = False

    @torch.no_grad()
    def bind_base(self, base_weight: Tensor, bias: Tensor | None = None) -> None:
        """Snapshot ``W₀`` (and ``b₀``); start from them, or from them plus a loaded diff."""
        if self.bias is not None and bias is None:
            raise ValueError("LyCORIS Full trains the layer bias, but the layer has none")
        for param, snapshot, value in (
            (self.weight, "base_weight", base_weight.reshape(self.weight_shape)),
            (self.bias, "base_bias", bias),
        ):
            if param is None:
                continue
            value = value.detach().to(device=param.device, dtype=self.param_dtype)
            setattr(self, snapshot, value.clone())
            if self._pending_diff:
                param.add_(value)
            else:
                param.copy_(value)
        self._pending_diff = False

    def delta_weight(self) -> Tensor:
        return self.weight - self.base_weight.to(self.weight.device)

    def delta_bias(self) -> Tensor | None:
        return None if self.bias is None else self.bias - self.base_bias.to(self.bias.device)

    @torch.no_grad()
    def export_tensors(self) -> dict[str, Tensor]:
        out = {"diff": self.delta_weight().detach().clone()}
        if self.bias is not None:
            out["diff_b"] = self.delta_bias().detach().clone()
        return out

    def extra_metadata(self) -> dict[str, Any]:
        return {"algo": "full", **self._shape_metadata()}

    @classmethod
    def from_tensors(
        cls, tensors: dict[str, Tensor], meta: dict[str, Any] | None = None, **kwargs: Any
    ) -> Full:
        diff = tensors["diff"]
        diff_b = tensors.get("diff_b")
        kernel = resolve_kernel(meta, kwargs, tuple(diff.shape[2:]))
        dtype = kwargs.pop("dtype", torch.float32)
        mod = cls(
            int(diff.shape[0]),
            diff[0].numel() // math.prod(kernel),
            kernel=kernel,
            bias=diff_b is not None,
            dtype=dtype,
        )
        with torch.no_grad():
            mod.weight.copy_(diff.reshape(mod.weight.shape).to(dtype))  # W₀ + diff once bind_base() runs
            if diff_b is not None:
                mod.bias.copy_(diff_b.reshape(mod.bias.shape).to(dtype))
        mod._pending_diff = True
        return mod

    def param_kinds(self) -> dict[str, str]:
        return {"weight": "full", "bias": "full"}
