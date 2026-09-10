"""Full: train the layer weight directly; ``ΔW = W - W₀``. Exported as LyCORIS ``diff``."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .base import AdapterModule


class Full(AdapterModule):
    kind = "full"
    supports_bypass = False

    def __init__(
        self,
        out_features: int,
        in_features: int,
        *,
        base_weight: Tensor | None = None,
        dtype: torch.dtype = torch.float32,
        **_: Any,
    ) -> None:
        super().__init__(out_features, in_features, dtype=dtype)
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, dtype=dtype))
        self.register_buffer(
            "base_weight", torch.zeros(out_features, in_features, dtype=dtype), persistent=False
        )
        self._pending_diff = False
        if base_weight is not None:
            self.bind_base(base_weight)

    @torch.no_grad()
    def bind_base(self, base_weight: Tensor) -> None:
        """Snapshot ``W₀`` (fp32); start from ``W₀`` or from ``W₀ + diff`` when a diff was loaded."""
        w = base_weight.detach().to(device=self.weight.device, dtype=self.param_dtype)
        self.base_weight = w.clone()
        if self._pending_diff:
            self.weight.add_(w)
            self._pending_diff = False
        else:
            self.weight.copy_(w)

    def delta_weight(self) -> Tensor:
        return self.weight - self.base_weight.to(self.weight.device)

    @torch.no_grad()
    def export_tensors(self) -> dict[str, Tensor]:
        return {"diff": self.delta_weight().detach().clone()}

    def extra_metadata(self) -> dict[str, Any]:
        return {"algo": "full", "out_features": self.out_features, "in_features": self.in_features}

    @classmethod
    def from_tensors(
        cls, tensors: dict[str, Tensor], meta: dict[str, Any] | None = None, **kwargs: Any
    ) -> Full:
        diff = tensors["diff"]
        dtype = kwargs.pop("dtype", torch.float32)
        mod = cls(int(diff.shape[0]), int(diff.shape[1]), dtype=dtype)
        with torch.no_grad():
            mod.weight.copy_(diff.to(dtype))  # becomes W₀ + diff once bind_base() runs
        mod._pending_diff = True
        return mod

    def param_kinds(self) -> dict[str, str]:
        return {"weight": "full"}
