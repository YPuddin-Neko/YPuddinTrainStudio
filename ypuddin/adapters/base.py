"""Common contract for adapter algorithms.

Every algorithm produces a weight delta ``ΔW`` of shape ``(out, in)`` and may provide a
structured fast path ``delta_apply(x) == F.linear(x, ΔW)``. ``scale``, ``scalar`` and
dropout are applied in exactly one place per path so that bypass, merged and exported
weights always agree.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def kaiming_uniform_(t: Tensor) -> Tensor:
    return nn.init.kaiming_uniform_(t, a=math.sqrt(5))


def compute_scale(alpha: float, rank: int | None, rs_lora: bool) -> float:
    """``alpha / rank`` (or ``alpha / sqrt(rank)`` for rsLoRA); ``1.0`` when there is no rank."""
    if rank is None:
        return 1.0
    return float(alpha) / (math.sqrt(rank) if rs_lora else rank)


class AdapterModule(nn.Module, ABC):
    kind: str = "base"
    supports_bypass: bool = False

    def __init__(
        self,
        out_features: int,
        in_features: int,
        *,
        dropout: float = 0.0,
        rank_dropout: float = 0.0,
        init: str = "default",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.out_features = int(out_features)
        self.in_features = int(in_features)
        self.dropout_p = float(dropout)
        self.rank_dropout_p = float(rank_dropout)
        self.init_mode = init
        self.param_dtype = dtype
        if init == "scalar":
            self.scalar: nn.Parameter | None = nn.Parameter(torch.zeros((), dtype=dtype))
        else:
            self.scalar = None

    def forward(self, x: Tensor | None = None) -> Tensor:
        """Keep computations inside Module hooks for distributed parameter residency."""
        return self.delta_weight() if x is None else self.delta_apply(x)

    # ----------------------------------------------------------------- scaling
    @property
    def effective_scalar(self) -> Tensor | float:
        return self.scalar if self.scalar is not None else 1.0

    def _rank_mask(self, rank: int, device: torch.device, dtype: torch.dtype) -> Tensor | None:
        """Mask over the rank axis (compensated by ``1/(1-p)``), only while training."""
        if not self.training or self.rank_dropout_p <= 0:
            return None
        keep = (torch.rand(rank, device=device) >= self.rank_dropout_p).to(dtype)
        return keep / (1.0 - self.rank_dropout_p)

    def _output_dropout(self, y: Tensor) -> Tensor:
        if self.training and self.dropout_p > 0:
            return F.dropout(y, self.dropout_p, training=True)
        return y

    # ----------------------------------------------------------------- contract
    @abstractmethod
    def delta_weight(self) -> Tensor:
        """Full ``ΔW`` of shape ``(out, in)`` including scale and scalar (never multiplier)."""

    def delta_apply(self, x: Tensor) -> Tensor:
        """``F.linear(x, ΔW)``; subclasses override with a structured fast path."""
        return F.linear(x, self.delta_weight().to(x.dtype))

    @abstractmethod
    def export_tensors(self) -> dict[str, Tensor]:
        """Kohya/LyCORIS key suffix -> tensor, with ``scalar`` folded in and ``alpha`` encoded."""

    @abstractmethod
    def extra_metadata(self) -> dict[str, Any]:
        """Explicit structural metadata (shapes, rank, factorization) stored alongside weights."""

    @classmethod
    @abstractmethod
    def from_tensors(
        cls, tensors: dict[str, Tensor], meta: dict[str, Any] | None = None, **kwargs: Any
    ) -> AdapterModule:
        """Rebuild a module from exported tensors (+ optional explicit metadata)."""

    def param_kinds(self) -> dict[str, str]:
        """Parameter name -> kind label used by ``lr_scale`` / weight-decay rules (e.g. ``w1``, ``up``)."""
        return {name: name.split("_")[0] for name, _ in self.named_parameters(recurse=False)}

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
