"""DoRA weight decomposition: ``W' = m · (W₀ + ΔW) / ||W₀ + ΔW||`` with trainable magnitude ``m``.

``input`` (the default) keeps one magnitude per input column, stored ``(1, in)``: ComfyUI, Forge and
A1111 compute it as trained. ``output`` keeps one per output row, stored ``(out, 1)``: LyCORIS's
default, which its loaders read, where for output rows ComfyUI and Forge divide by ``||W₀||`` and
A1111 by the column norm. The stored shape names the axis. A convolution's magnitude spans its kernel too,
``(out, 1, 1…)`` or ``(1, in, 1…)``, as LyCORIS and ComfyUI store it.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

AXES = ("input", "output")
_AXIS_NAMES = {"output": "输出通道", "input": "输入通道"}


def magnitude_axis(scale: Tensor) -> str:
    """``input`` for a ``(1, in…)`` magnitude, ``output`` for ``(out, 1…)`` or a flat one."""
    return "input" if scale.dim() >= 2 and scale.shape[0] == 1 and scale.shape[1] > 1 else "output"


def weight_norm(weight: Tensor, axis: str) -> Tensor:
    """Norms per output channel ``(out, 1…)`` or per input channel ``(1, in, 1…)``."""
    if weight.dim() == 2:
        return weight.norm(dim=1 if axis == "output" else 0, keepdim=True)
    kept = 0 if axis == "output" else 1
    return torch.linalg.vector_norm(weight, dim=[d for d in range(weight.dim()) if d != kept], keepdim=True)


def decompose(weight: Tensor, scale: Tensor) -> Tensor:
    """``weight`` rescaled to the magnitude ``scale`` along the axis its shape names, in FP32."""
    w32 = weight.to(torch.float32)
    norm = weight_norm(w32, magnitude_axis(scale)) + torch.finfo(torch.float32).eps
    return (w32 * (scale.to(w32.device, torch.float32).reshape(norm.shape) / norm)).to(weight.dtype)


class DoRA(nn.Module):
    def __init__(self, base_weight: Tensor, dtype: torch.dtype = torch.float32, axis: str = "input"):
        super().__init__()
        if axis not in AXES:
            raise ValueError(f"unknown DoRA axis {axis!r}")
        self.axis = axis
        self.dora_scale = nn.Parameter(weight_norm(base_weight.detach().to(torch.float32), axis).to(dtype))

    def forward(self, weight: Tensor) -> Tensor:
        return self.rescale(weight)

    def rescale(self, weight: Tensor) -> Tensor:
        return decompose(weight, self.dora_scale)

    @torch.no_grad()
    def export_tensor(self) -> Tensor:
        return self.dora_scale.detach().clone()

    @torch.no_grad()
    def load_tensor(self, t: Tensor) -> None:
        # A square layer's (n, 1) and (1, n) hold as many values; never reshape one axis into the other.
        if t.dim() >= 2 and t.shape != self.dora_scale.shape and magnitude_axis(t) != self.axis:
            raise ValueError(
                f"权重文件中的 DoRA 按{_AXIS_NAMES[magnitude_axis(t)]}计算，当前为{_AXIS_NAMES[self.axis]}；"
                "请把“DoRA 计算方向”改为与文件一致"
            )
        if t.numel() != self.dora_scale.numel():
            raise ValueError(
                f"DoRA magnitude {tuple(t.shape)} does not fit the layer's {tuple(self.dora_scale.shape)}"
            )
        self.dora_scale.copy_(t.reshape(self.dora_scale.shape).to(self.dora_scale.dtype))
