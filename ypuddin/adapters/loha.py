"""LoHa: ``ΔW = scale · (w1_a @ w1_b) ⊙ (w2_a @ w2_b)`` (Hadamard product of two low-rank matrices).

The Hadamard structure does not factor through the input, so LoHa always runs on the merged
path. A custom autograd function recomputes the two products in backward instead of caching
both (LyCORIS trick) to halve activation memory for large layers. A convolution's ``w*_b`` span
``in·k…``, the layout LyCORIS exports without Tucker decomposition.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from .base import AdapterModule, compute_scale, resolve_kernel


class _HadaWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w1a, w1b, w2a, w2b):  # type: ignore[override]
        ctx.save_for_backward(w1a, w1b, w2a, w2b)
        return (w1a @ w1b) * (w2a @ w2b)

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        w1a, w1b, w2a, w2b = ctx.saved_tensors
        p1 = w1a @ w1b
        p2 = w2a @ w2b
        g1 = grad_out * p2
        g2 = grad_out * p1
        return g1 @ w1b.T, w1a.T @ g1, g2 @ w2b.T, w2a.T @ g2


class LoHa(AdapterModule):
    kind = "loha"
    supports_bypass = False

    def __init__(
        self,
        out_features: int,
        in_features: int,
        *,
        rank: int = 16,
        alpha: float = 16.0,
        rs_lora: bool = False,
        kernel: tuple[int, ...] = (),
        dropout: float = 0.0,
        rank_dropout: float = 0.0,
        init: str = "default",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            out_features,
            in_features,
            kernel=kernel,
            dropout=dropout,
            rank_dropout=rank_dropout,
            init=init,
            dtype=dtype,
        )
        self.rank = int(rank)
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        self.alpha = float(alpha)
        self.rs_lora = bool(rs_lora)
        self.scale = compute_scale(self.alpha, self.rank, self.rs_lora)
        self.w1_a = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype))
        self.w1_b = nn.Parameter(torch.empty(self.rank, self.fan_in, dtype=dtype))
        self.w2_a = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype))
        self.w2_b = nn.Parameter(torch.empty(self.rank, self.fan_in, dtype=dtype))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.w1_b, std=1.0)
        nn.init.normal_(self.w1_a, std=0.1)
        nn.init.normal_(self.w2_b, std=1.0)
        if self.init_mode == "scalar":
            nn.init.normal_(self.w2_a, std=0.1)
        else:
            self.w2_a.zero_()
        if self.scalar is not None:
            self.scalar.zero_()

    def delta_weight(self) -> Tensor:
        mask = self._rank_mask(self.rank, self.w1_a.device, self.w1_a.dtype)
        w1_a = self.w1_a if mask is None else self.w1_a * mask
        w = _HadaWeight.apply(w1_a, self.w1_b, self.w2_a, self.w2_b)
        return self._as_weight(self._scaled(w))

    def delta_apply(self, x: Tensor) -> Tensor:
        return self._output_dropout(super().delta_apply(x))

    @torch.no_grad()
    def export_tensors(self) -> dict[str, Tensor]:
        scalar = float(self.effective_scalar) if isinstance(self.effective_scalar, Tensor) else 1.0
        return {
            "hada_w1_a": (self.w1_a * scalar).detach().clone(),
            "hada_w1_b": self.w1_b.detach().clone(),
            "hada_w2_a": self.w2_a.detach().clone(),
            "hada_w2_b": self.w2_b.detach().clone(),
            "alpha": torch.tensor(self.scale * self.rank, dtype=torch.float32),
        }

    def extra_metadata(self) -> dict[str, Any]:
        return {
            "algo": "loha",
            "rank": self.rank,
            "alpha": self.alpha,
            "rs_lora": self.rs_lora,
            **self._shape_metadata(),
        }

    @classmethod
    def from_tensors(
        cls, tensors: dict[str, Tensor], meta: dict[str, Any] | None = None, **kwargs: Any
    ) -> LoHa:
        w1_a, w1_b = tensors["hada_w1_a"], tensors["hada_w1_b"]
        kernel = resolve_kernel(meta, kwargs)
        rank = int(w1_a.shape[1])
        alpha_file = float(tensors["alpha"].item()) if "alpha" in tensors else float(rank)
        dtype = kwargs.pop("dtype", torch.float32)
        mod = cls(
            int(w1_a.shape[0]),
            int(w1_b.shape[1]) // math.prod(kernel),
            rank=rank,
            alpha=alpha_file,
            rs_lora=False,
            kernel=kernel,
            dtype=dtype,
            **kwargs,
        )
        with torch.no_grad():
            for name in ("w1_a", "w1_b", "w2_a", "w2_b"):
                getattr(mod, name).copy_(tensors[f"hada_{name}"].to(dtype))
        return mod
