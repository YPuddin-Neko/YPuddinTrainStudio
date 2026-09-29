"""Common contract for adapter algorithms.

Every algorithm produces a weight delta ``ΔW`` shaped like the layer's weight: ``(out, in)`` for a
linear layer, ``(out, in, *kernel)`` for a convolution, and may provide a structured fast path
``delta_apply(x)`` equal to applying the layer's operation with ``ΔW``. ``scale``, ``scalar`` and
dropout are applied in exactly one place per path so that bypass, merged and exported weights always
agree.

A convolution's low-rank factors see its weight as the matrix ``(out, in·k₁·k₂…)``, the layout
LyCORIS and kohya export; LoKr instead keeps the kernel in its second factor, as LyCORIS does.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_CONV = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}


def kaiming_uniform_(t: Tensor) -> Tensor:
    return nn.init.kaiming_uniform_(t, a=math.sqrt(5))


def compute_scale(alpha: float, rank: int | None, rs_lora: bool) -> float:
    """``alpha / rank`` (or ``alpha / sqrt(rank)`` for rsLoRA); ``1.0`` when there is no rank."""
    if rank is None:
        return 1.0
    return float(alpha) / (math.sqrt(rank) if rs_lora else rank)


def probability(name: str, value: float) -> float:
    """A dropout probability in ``[0, 1)``: at 1 nothing would be left to train."""
    value = float(value)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be at least 0 and below 1, got {value}")
    return value


@dataclass(frozen=True)
class ConvGeometry:
    """How the adapted layer convolves (stride, padding, dilation, padding mode); ``groups`` is 1."""

    dims: int
    stride: tuple[int, ...]
    padding: tuple[int, ...] | str
    dilation: tuple[int, ...]
    padding_mode: str = "zeros"
    reversed_padding: tuple[int, ...] = ()

    @classmethod
    def of(cls, conv: nn.Module) -> ConvGeometry:
        return cls(
            dims=len(conv.kernel_size),
            stride=tuple(conv.stride),
            padding=conv.padding if isinstance(conv.padding, str) else tuple(conv.padding),
            dilation=tuple(conv.dilation),
            padding_mode=conv.padding_mode,
            reversed_padding=tuple(getattr(conv, "_reversed_padding_repeated_twice", ())),
        )

    def conv(self, x: Tensor, weight: Tensor) -> Tensor:
        """The layer's own convolution, with ``weight`` in place of its kernel and no bias."""
        op = _CONV[self.dims]
        if self.padding_mode != "zeros":
            x = F.pad(x, self.reversed_padding, mode=self.padding_mode)
            return op(x, weight, None, self.stride, 0, self.dilation)
        return op(x, weight, None, self.stride, self.padding, self.dilation)

    def pointwise(self, h: Tensor, weight: Tensor) -> Tensor:
        """Mix channels with a ``(out, in)`` matrix, as a 1×1 convolution."""
        return _CONV[self.dims](h, weight.reshape(*weight.shape, *(1,) * self.dims))

    def channels(self, values: Tensor) -> Tensor:
        """A per-channel vector ``(c,)`` (or ``(batch, c)``) shaped to broadcast over a feature map."""
        return values.reshape(*values.shape[:-1] or (1,), values.shape[-1], *(1,) * self.dims)


class AdapterModule(nn.Module, ABC):
    kind: str = "base"
    supports_bypass: bool = False
    # Masks that differ per sample cannot be merged into one weight.
    needs_bypass: bool = False

    def __init__(
        self,
        out_features: int,
        in_features: int,
        *,
        kernel: tuple[int, ...] = (),
        dropout: float = 0.0,
        rank_dropout: float = 0.0,
        init: str = "default",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.out_features = int(out_features)
        self.in_features = int(in_features)
        self.kernel = tuple(int(size) for size in kernel)
        if any(size <= 0 for size in self.kernel):
            raise ValueError(f"kernel sizes must be positive, got {self.kernel}")
        self.dropout_p = probability("dropout", dropout)
        self.rank_dropout_p = probability("rank_dropout", rank_dropout)
        self.init_mode = init
        self.param_dtype = dtype
        self.geometry: ConvGeometry | None = None
        if init == "scalar":
            self.scalar: nn.Parameter | None = nn.Parameter(torch.zeros((), dtype=dtype))
        else:
            self.scalar = None

    def forward(self, x: Tensor | None = None) -> Tensor:
        """Keep computations inside Module hooks for distributed parameter residency."""
        return self.delta_weight() if x is None else self.delta_apply(x)

    # ----------------------------------------------------------------- shape
    @property
    def fan_in(self) -> int:
        """Inputs per output: ``in`` for a linear layer, ``in·k₁·k₂…`` for a convolution."""
        return self.in_features * math.prod(self.kernel)

    @property
    def weight_shape(self) -> tuple[int, ...]:
        return (self.out_features, self.in_features, *self.kernel)

    def bind_geometry(self, geometry: ConvGeometry) -> None:
        self.geometry = geometry

    def _geometry(self) -> ConvGeometry:
        if self.geometry is None:
            raise RuntimeError(
                f"{self.kind} convolution adapter has no layer geometry; wrap it in AdaptedConv"
            )
        return self.geometry

    def _as_weight(self, flat: Tensor) -> Tensor:
        """``(out, in·k…)`` as the layer's weight shape."""
        return flat.view(self.weight_shape)

    # ----------------------------------------------------------------- layer operations
    def _rank_in(self, x: Tensor, factor: Tensor) -> Tensor:
        """``x`` through an ``(r, in·k…)`` factor as the layer computes: ``x @ fᵀ``, or its convolution."""
        if not self.kernel:
            return x @ factor.transpose(0, 1)
        return self._geometry().conv(x, factor.view(factor.shape[0], self.in_features, *self.kernel))

    def _rank_out(self, h: Tensor, factor: Tensor) -> Tensor:
        """The channels of ``h`` mixed by an ``(n, r)`` factor: ``h @ fᵀ``, or a 1×1 convolution."""
        return self._geometry().pointwise(h, factor) if self.kernel else h @ factor.transpose(0, 1)

    def _per_rank(self, h: Tensor, values: Tensor) -> Tensor:
        """``h`` scaled per rank by ``(r,)`` values, or per sample and rank by ``(batch, r)``."""
        if self.kernel:
            return h * self._geometry().channels(values)
        if values.dim() == 2:
            values = values.view(values.shape[0], *(1,) * (h.dim() - 2), values.shape[1])
        return h * values

    # ----------------------------------------------------------------- scaling
    @property
    def effective_scalar(self) -> Tensor | float:
        return self.scalar if self.scalar is not None else 1.0

    def _gain(self) -> Tensor | float:
        return self.scale * self.effective_scalar

    def _scaled(self, t: Tensor) -> Tensor:
        """``t`` times the gain; a gain of exactly 1 costs nothing."""
        gain = self._gain()
        if isinstance(gain, Tensor):
            return t * gain.to(t.dtype)
        return t if gain == 1.0 else t * gain

    def _finish(self, y: Tensor) -> Tensor:
        """A bypass output: scaled, then the output dropout."""
        return self._output_dropout(self._scaled(y))

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
        """Full ``ΔW`` shaped like the layer weight, including scale and scalar (never multiplier)."""

    def delta_apply(self, x: Tensor) -> Tensor:
        """The layer's operation with ``ΔW``; subclasses override with a structured fast path."""
        weight = self.delta_weight().to(x.dtype)
        return self._geometry().conv(x, weight) if self.kernel else F.linear(x, weight)

    def delta_bias(self) -> Tensor | None:
        """``Δb`` for adapters that train the layer bias (LyCORIS Full)."""
        return None

    @abstractmethod
    def export_tensors(self) -> dict[str, Tensor]:
        """Kohya/LyCORIS key suffix -> tensor, with ``scalar`` folded in and ``alpha`` encoded."""

    def _lora_export(self, down: Tensor, up: Tensor, alpha: float) -> dict[str, Tensor]:
        """Plain LoRA tensors; a convolution's ``down`` keeps its kernel and ``up`` is 1×1 (LoCon)."""
        return {
            "lora_down.weight": down.reshape(down.shape[0], self.in_features, *self.kernel),
            "lora_up.weight": up.reshape(*up.shape, *(1,) * len(self.kernel)),
            "alpha": torch.tensor(float(alpha), dtype=torch.float32),
        }

    @abstractmethod
    def extra_metadata(self) -> dict[str, Any]:
        """Explicit structural metadata (shapes, rank, factorization) stored alongside weights."""

    def _shape_metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {"out_features": self.out_features, "in_features": self.in_features}
        if self.kernel:
            meta["kernel"] = list(self.kernel)
        return meta

    @classmethod
    @abstractmethod
    def from_tensors(
        cls, tensors: dict[str, Tensor], meta: dict[str, Any] | None = None, **kwargs: Any
    ) -> AdapterModule:
        """Rebuild a module from exported tensors (+ optional explicit metadata); ``kwargs`` may name
        the layer's ``kernel`` (see :func:`resolve_kernel`)."""

    def param_kinds(self) -> dict[str, str]:
        """Parameter name -> kind label used by ``lr_scale`` / weight-decay rules (e.g. ``w1``, ``up``)."""
        return {name: name.split("_")[0] for name, _ in self.named_parameters(recurse=False)}

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


TUCKER_KEYS = frozenset({"lora_mid.weight", "lokr_t1", "lokr_t2", "hada_t1", "hada_t2"})


def refuse_tucker(tensors: dict[str, Tensor]) -> None:
    """Files with Tucker-decomposed convolutions cannot be read."""
    if TUCKER_KEYS & tensors.keys():
        raise ValueError("文件使用 Tucker 分解（lora_mid / lokr_t2 / hada_t1），当前不支持读取")


def resolve_kernel(
    meta: dict[str, Any] | None, kwargs: dict[str, Any], fallback: tuple[int, ...] = ()
) -> tuple[int, ...]:
    """The target layer's kernel: the caller's (popped from ``kwargs``), the file metadata's, else
    ``fallback``. Files keep a low-rank convolution's kernel only inside its flattened factors."""
    kernel = kwargs.pop("kernel", None)
    if kernel is None and meta and meta.get("kernel") is not None:
        kernel = meta["kernel"]
    return tuple(int(size) for size in (fallback if kernel is None else kernel))
