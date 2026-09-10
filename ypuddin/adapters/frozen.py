"""Frozen base linear layer with optional fp8 storage (per-tensor scale)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

FP8_DTYPES: dict[str, torch.dtype] = {}
if hasattr(torch, "float8_e4m3fn"):
    FP8_DTYPES["fp8_e4m3"] = torch.float8_e4m3fn
if hasattr(torch, "float8_e5m2"):
    FP8_DTYPES["fp8_e5m2"] = torch.float8_e5m2

_FP8_MAX = {"fp8_e4m3": 448.0, "fp8_e5m2": 57344.0}

STORAGE_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def quantize_fp8(weight: Tensor, kind: str) -> tuple[Tensor, Tensor]:
    """Per-tensor symmetric quantization: ``q = round(w / scale)``, ``scale = amax / fp8_max``."""
    if kind not in FP8_DTYPES:
        raise RuntimeError(f"{kind} is not supported by this torch build")
    w32 = weight.detach().to(torch.float32)
    amax = w32.abs().amax().clamp(min=1e-12)
    scale = (amax / _FP8_MAX[kind]).to(torch.float32)
    q = (w32 / scale).clamp(-_FP8_MAX[kind], _FP8_MAX[kind]).to(FP8_DTYPES[kind])
    return q, scale


class FrozenLinear(nn.Module):
    """Holds ``weight``/``bias`` of a frozen ``nn.Linear`` as buffers.

    ``precision`` is one of ``keep`` (leave dtype as-is), ``bf16``/``fp16``/``fp32`` or
    ``fp8_e4m3``/``fp8_e5m2``. fp8 weights are dequantized on the fly (``w.to(dtype) * scale``),
    matching how ComfyUI consumes ``fp8_scaled`` checkpoints.
    """

    def __init__(
        self,
        weight: Tensor,
        bias: Tensor | None = None,
        *,
        precision: str = "keep",
        scale: Tensor | None = None,
    ):
        super().__init__()
        self.out_features, self.in_features = int(weight.shape[0]), int(weight.shape[1])
        self.precision = precision
        w = weight.detach()
        if precision in FP8_DTYPES:
            if w.dtype in FP8_DTYPES.values():
                if scale is None:
                    scale = torch.ones((), dtype=torch.float32, device=w.device)
                q = w
            else:
                q, scale = quantize_fp8(w, precision)
            self.register_buffer("weight", q.contiguous())
            self.register_buffer("weight_scale", scale.to(w.device))
        else:
            if precision != "keep":
                w = w.to(STORAGE_DTYPES[precision])
            self.register_buffer("weight", w.contiguous())
            self.weight_scale = None
        self.register_buffer("bias", None if bias is None else bias.detach().clone())

    @property
    def is_fp8(self) -> bool:
        return self.weight_scale is not None

    def dequant(self, dtype: torch.dtype | None = None) -> Tensor:
        if self.is_fp8:
            dt = dtype or torch.bfloat16
            return self.weight.to(dt) * self.weight_scale.to(dt)
        return self.weight if dtype is None else self.weight.to(dtype)

    def forward(self, x: Tensor) -> Tensor:
        w = self.dequant(x.dtype)
        b = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, w, b)

    @classmethod
    def from_linear(cls, linear: nn.Linear, precision: str = "keep") -> FrozenLinear:
        return cls(linear.weight.data, None if linear.bias is None else linear.bias.data, precision=precision)

    def to_linear(self, dtype: torch.dtype | None = None) -> nn.Linear:
        lin = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None, device="meta")
        w = self.dequant(dtype)
        lin.weight = nn.Parameter(w.detach().clone(), requires_grad=False)
        if self.bias is not None:
            lin.bias = nn.Parameter(self.bias.detach().clone().to(w.dtype), requires_grad=False)
        return lin

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, precision={self.precision}, dtype={self.weight.dtype}"
