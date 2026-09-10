"""LoRA: ``ΔW = scale · up @ down`` with ``down: (r, in)``, ``up: (out, r)``."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .base import AdapterModule, compute_scale, kaiming_uniform_


class LoRA(AdapterModule):
    kind = "lora"
    supports_bypass = True

    def __init__(
        self,
        out_features: int,
        in_features: int,
        *,
        rank: int = 16,
        alpha: float = 16.0,
        rs_lora: bool = False,
        dropout: float = 0.0,
        rank_dropout: float = 0.0,
        init: str = "default",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            out_features, in_features, dropout=dropout, rank_dropout=rank_dropout, init=init, dtype=dtype
        )
        self.rank = int(rank)
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        self.alpha = float(alpha)
        self.rs_lora = bool(rs_lora)
        self.scale = compute_scale(self.alpha, self.rank, self.rs_lora)
        self.down = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype))
        self.up = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        kaiming_uniform_(self.down)
        if self.init_mode == "scalar":
            kaiming_uniform_(self.up)
        else:
            self.up.zero_()
        if self.scalar is not None:
            self.scalar.zero_()

    def _gain(self) -> Tensor | float:
        return self.scale * self.effective_scalar

    def delta_weight(self) -> Tensor:
        mask = self._rank_mask(self.rank, self.up.device, self.up.dtype)
        up = self.up if mask is None else self.up * mask
        return (up @ self.down) * self._gain()

    def delta_apply(self, x: Tensor) -> Tensor:
        dt = x.dtype
        h = x @ self.down.to(dt).transpose(0, 1)
        mask = self._rank_mask(self.rank, x.device, dt)
        if mask is not None:
            h = h * mask
        y = h @ self.up.to(dt).transpose(0, 1)
        gain = self._gain()
        if isinstance(gain, Tensor):
            gain = gain.to(dt)
        return self._output_dropout(y * gain)

    @torch.no_grad()
    def export_tensors(self) -> dict[str, Tensor]:
        scalar = float(self.effective_scalar) if isinstance(self.effective_scalar, Tensor) else 1.0
        return {
            "lora_down.weight": self.down.detach().clone(),
            "lora_up.weight": (self.up * scalar).detach().clone(),
            "alpha": torch.tensor(self.scale * self.rank, dtype=torch.float32),
        }

    def extra_metadata(self) -> dict[str, Any]:
        return {
            "algo": "lora",
            "rank": self.rank,
            "alpha": self.alpha,
            "rs_lora": self.rs_lora,
            "out_features": self.out_features,
            "in_features": self.in_features,
        }

    @classmethod
    def from_tensors(
        cls, tensors: dict[str, Tensor], meta: dict[str, Any] | None = None, **kwargs: Any
    ) -> LoRA:
        down, up = tensors["lora_down.weight"], tensors["lora_up.weight"]
        if down.dim() > 2:  # conv-style 1x1 kernels from other trainers
            down, up = down.flatten(1), up.flatten(1)
        rank = int(down.shape[0])
        alpha_file = float(tensors["alpha"].item()) if "alpha" in tensors else float(rank)
        dtype = kwargs.pop("dtype", torch.float32)
        mod = cls(
            int(up.shape[0]),
            int(down.shape[1]),
            rank=rank,
            alpha=alpha_file,
            rs_lora=False,
            dtype=dtype,
            **kwargs,
        )
        with torch.no_grad():
            mod.down.copy_(down.to(dtype))
            mod.up.copy_(up.to(dtype))
        return mod

    def param_kinds(self) -> dict[str, str]:
        return {name: name for name, _ in self.named_parameters(recurse=False)}
