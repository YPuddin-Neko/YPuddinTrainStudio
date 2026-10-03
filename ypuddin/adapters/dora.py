"""DoRA weight decomposition: ``W' = m · (W₀ + ΔW) / ||W₀ + ΔW||`` with trainable magnitude ``m``.

``input`` (the default) keeps one magnitude per input column, stored ``(1, in)``.
``output`` keeps one per output row, stored ``(out, 1)``. Convolutions store
``(1, in, 1…)`` or ``(out, 1, 1…)``. Standard mode uses the merged norm in FP32.
ComfyUI mode follows its 0.38.1 floating-point loader: input uses the merged
norm, output uses the base norm, with epsilon in the selected merge dtype.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import Tensor, nn

AXES = ("input", "output")
COMPUTE_MODES = ("standard", "comfyui")
MERGE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
MERGE_PRECISIONS = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
_AXIS_NAMES = {"output": "输出通道", "input": "输入通道"}


def auto_merge_dtype(base: nn.Module, compute_dtype: torch.dtype | None = None) -> torch.dtype:
    """The automatic merge precision: the base weight's dtype; FP8 storage merges in the compute dtype."""
    return (compute_dtype or torch.float32) if getattr(base, "is_fp8", False) else base.weight.dtype


def resolve_merge_dtype(
    selected: str, base: nn.Module, compute_dtype: torch.dtype | None = None
) -> torch.dtype:
    """``adapter.dora_merge_dtype`` for one layer: ``auto`` or an explicit ``bf16`` / ``fp16`` / ``fp32``."""
    return auto_merge_dtype(base, compute_dtype) if selected == "auto" else MERGE_PRECISIONS[selected]


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


def loader_weight_norm(weight: Tensor, axis: str) -> Tensor:
    """Preserve the loader's reshape/reduction order, including convolution kernels.

    Follows ComfyUI's ``weight_decompose``; see ypuddin/sampling/ADAPTER_FUSION_NOTICE.md.
    """
    matrix = weight if axis == "output" else weight.transpose(0, 1)
    norm = matrix.reshape(matrix.shape[0], -1).norm(dim=1, keepdim=True)
    norm = norm.reshape(matrix.shape[0], *[1] * (weight.ndim - 1))
    return norm if axis == "output" else norm.transpose(0, 1)


def loader_decompose(base: Tensor, delta: Tensor, magnitude: Tensor, *, alpha: float, strength: float) -> Tensor:
    """``base`` with the unscaled ``delta`` and a ``magnitude`` in ``base.dtype``, in ComfyUI's
    ``weight_decompose`` order (see ypuddin/sampling/ADAPTER_FUSION_NOTICE.md): alpha scales the delta
    before normalisation and strength interpolates afterwards; the magnitude's first dimension names the
    axis and epsilon follows the base dtype. Differentiable; nothing is modified in place."""
    merged = base + (delta * alpha).to(base.dtype)
    # The shape check is the external loader's axis convention.
    axis = "output" if magnitude.shape[0] == base.shape[0] else "input"
    norm = loader_weight_norm(base if axis == "output" else merged, axis)
    norm = norm + torch.finfo(base.dtype).eps
    effective = merged * (magnitude / norm).to(base.dtype)
    return effective if strength == 1.0 else base + strength * (effective - base)


class DoRA(nn.Module):
    def __init__(
        self, base_weight: Tensor, dtype: torch.dtype = torch.float32, axis: str = "input", *,
        compute_mode: str = "standard", merge_dtype: torch.dtype | None = None,
        save_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if axis not in AXES:
            raise ValueError(f"unknown DoRA axis {axis!r}")
        if compute_mode not in COMPUTE_MODES:
            raise ValueError(f"unknown DoRA compute mode {compute_mode!r}")
        self.axis = axis
        self.compute_mode = compute_mode
        self.merge_dtype = base_weight.dtype if merge_dtype is None else merge_dtype
        self.save_dtype = dtype if save_dtype is None else save_dtype
        if compute_mode == "comfyui":
            if self.merge_dtype not in MERGE_DTYPES or self.save_dtype not in MERGE_DTYPES:
                raise ValueError("ComfyUI DoRA requires FP16, BF16 or FP32 merge and save dtypes")
            with nullcontext() if base_weight.device.type == "meta" else torch.autocast(device_type=base_weight.device.type, enabled=False):
                norm = loader_weight_norm(base_weight.detach().to(self.merge_dtype), axis)
                magnitude = norm + torch.finfo(self.merge_dtype).eps
        else:
            magnitude = weight_norm(base_weight.detach().to(torch.float32), axis)
        self.dora_scale = nn.Parameter(magnitude.to(dtype))

    def forward(
        self, weight: Tensor, *, base_weight: Tensor | None = None, alpha: float = 1.0,
        strength: float = 1.0,
    ) -> Tensor:
        if self.compute_mode == "comfyui":
            if base_weight is None:
                raise ValueError("ComfyUI DoRA requires the base weight and unscaled exported delta")
            return self.rescale_exported(base_weight, weight, alpha=alpha, strength=strength)
        return self.rescale(weight)

    def rescale_exported(
        self, base_weight: Tensor, delta: Tensor, *, alpha: float = 1.0, strength: float = 1.0,
    ) -> Tensor:
        """Differentiable saved-precision fusion in ComfyUI's order (see
        ypuddin/sampling/ADAPTER_FUSION_NOTICE.md); FP32 optimizer leaves stay attached."""
        with torch.autocast(device_type=base_weight.device.type, enabled=False):
            base = base_weight.to(self.merge_dtype)
            magnitude = self.dora_scale.to(self.save_dtype).to(base.dtype)
            return loader_decompose(base, delta, magnitude, alpha=alpha, strength=strength)

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
