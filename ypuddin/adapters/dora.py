"""DoRA weight decomposition: ``W' = m · (W₀ + ΔW) / ||W₀ + ΔW||_row`` with trainable magnitude ``m``."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DoRA(nn.Module):
    def __init__(self, base_weight: Tensor, dtype: torch.dtype = torch.float32):
        super().__init__()
        norm = (
            base_weight.detach().to(torch.float32).reshape(base_weight.shape[0], -1).norm(dim=1, keepdim=True)
        )
        self.dora_scale = nn.Parameter(norm.to(dtype))

    def rescale(self, weight: Tensor) -> Tensor:
        w32 = weight.to(torch.float32)
        norm = w32.reshape(w32.shape[0], -1).norm(dim=1, keepdim=True) + torch.finfo(torch.float32).eps
        return (w32 * (self.dora_scale.to(torch.float32) / norm)).to(weight.dtype)

    @torch.no_grad()
    def export_tensor(self) -> Tensor:
        return self.dora_scale.detach().clone()

    @torch.no_grad()
    def load_tensor(self, t: Tensor) -> None:
        self.dora_scale.copy_(t.reshape(self.dora_scale.shape).to(self.dora_scale.dtype))
