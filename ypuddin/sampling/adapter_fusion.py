"""Inference-only fusion of exported adapters using ComfyUI 0.38.1 arithmetic.

Adapted from ComfyUI commit 20ca544ee0436721d8eb5f544665e490609f72c8,
under GPL-3.0; see ADAPTER_FUSION_NOTICE.md. The requested dtype is used for
both the base weight and intermediate values, as in its LowVramPatch path.
Factor reconstruction and DoRA decomposition are shared with training's
compatible DoRA mode (``ypuddin.adapters.base`` and ``ypuddin.adapters.dora``).
This module does not implement static patching's separate intermediate dtype,
stochastic rounding, model-strength patches, offsets or custom patch functions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import torch
from torch import Tensor

from ypuddin.adapters.base import exported_factor_delta
from ypuddin.adapters.dora import loader_decompose

COMFYUI_COMMIT = "20ca544ee0436721d8eb5f544665e490609f72c8"
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_Kind = Literal["lora", "lokr", "loha", "full"]


def _strength(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("adapter strength must be finite")
    return value


def _base_in(weight: Tensor, dtype: torch.dtype, *, copy: bool = False) -> Tensor:
    if dtype not in _DTYPES:
        raise ValueError("adapter fusion dtype must be float16, bfloat16 or float32")
    if not weight.is_floating_point():
        raise ValueError("adapter fusion requires a dequantized floating-point base weight")
    return weight.detach().to(dtype=dtype, copy=copy)


def _copy_base(weight: Tensor, dtype: torch.dtype) -> Tensor:
    return _base_in(weight, dtype, copy=True)


@torch.no_grad()
def fuse_adapter_delta(
    base_weight: Tensor,
    delta: Tensor,
    *,
    dtype: torch.dtype,
    alpha: float = 1.0,
    magnitude: Tensor | None = None,
    strength: float = 1.0,
) -> Tensor:
    """Apply an unscaled reconstructed delta without changing any input tensor.

    Keep alpha separate from delta: ordinary adapters multiply by strength and
    alpha together, while DoRA applies alpha before normalization and strength
    afterwards. The distinction affects low-precision rounding.
    """
    strength = _strength(strength)
    alpha = float(alpha)
    if not math.isfinite(alpha):
        raise ValueError("adapter alpha must be finite")
    with torch.autocast(device_type=base_weight.device.type, enabled=False):
        # DoRA builds new tensors; the plain update adds in place to a copy of the base.
        weight = _copy_base(base_weight, dtype) if magnitude is None else _base_in(base_weight, dtype)
        delta = delta.detach().to(device=weight.device, dtype=dtype)
        if delta.shape != weight.shape:
            raise ValueError("adapter delta does not match the base weight shape")
        if magnitude is not None:
            # Epsilon is tied to the fusion dtype, not the file's.
            scale = magnitude.detach().to(device=weight.device, dtype=dtype)
            return loader_decompose(weight, delta, scale, alpha=alpha, strength=strength)
        weight += ((strength * alpha) * delta).to(weight.dtype)
        return weight


@dataclass(frozen=True)
class ComfyAdapterFusion:
    """One exported layer. Scalar gains and rsLoRA scaling must already be folded by export.

    Construction detaches tensor views without duplicating their storage. The caller
    must keep these exported tensors stable while the object is in use. Fusion never
    mutates them or the supplied base; the returned tensor has no autograd graph.
    """

    kind: _Kind
    tensors: Mapping[str, Tensor]
    alpha: float | None

    @classmethod
    def from_tensors(cls, tensors: Mapping[str, Tensor], prefix: str = "") -> ComfyAdapterFusion:
        """Read suffix keys, or select ``prefix + '.'`` from a complete export."""
        if prefix:
            stem = prefix + "."
            selected = {key[len(stem):]: value for key, value in tensors.items() if key.startswith(stem)}
        else:
            selected = dict(tensors)
        if not selected:
            raise ValueError("adapter fusion received no layer tensors")
        unsupported = {"lora_mid.weight", "lokr_t2", "hada_t1", "hada_t2", "reshape_weight", "scalar"}
        if selected.keys() & unsupported:
            raise ValueError("adapter fusion requires exported factors without Tucker, reshape or unfolded scalar")
        kinds = [
            name for name, present in (
                ("lora", "lora_up.weight" in selected),
                ("lokr", "lokr_w1" in selected or "lokr_w1_a" in selected),
                ("loha", "hada_w1_a" in selected),
                ("full", "diff" in selected),
            ) if present
        ]
        if len(kinds) != 1:
            raise ValueError("adapter fusion requires exactly one supported exported algorithm per layer")
        kind = kinds[0]
        if kind == "lora":
            required = {"lora_up.weight", "lora_down.weight"}
        elif kind == "loha":
            required = {"hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"}
        elif kind == "lokr":
            required = set()
            for factor in ("lokr_w1", "lokr_w2"):
                if factor in selected:
                    if factor + "_a" in selected or factor + "_b" in selected:
                        raise ValueError(f"ambiguous adapter factor: {factor}")
                    required.add(factor)
                else:
                    required.update((factor + "_a", factor + "_b"))
        else:
            required = {"diff"}
        missing = required - selected.keys()
        if missing:
            raise ValueError(f"adapter fusion is missing tensors: {', '.join(sorted(missing))}")
        optional = {"diff_b"} if kind == "full" else {"alpha", "dora_scale"}
        extra = selected.keys() - required - optional
        if extra:
            raise ValueError(f"unsupported adapter tensors: {', '.join(sorted(extra))}")
        if any(not isinstance(value, Tensor) or not value.is_floating_point() for value in selected.values()):
            raise ValueError("adapter fusion tensors must be floating point")
        alpha = None
        if "alpha" in selected:
            if selected["alpha"].numel() != 1:
                raise ValueError("adapter alpha must be a scalar")
            alpha = float(selected["alpha"].item())
            if not math.isfinite(alpha):
                raise ValueError("adapter alpha must be finite")
        return cls(kind, MappingProxyType({key: value.detach() for key, value in selected.items()}), alpha)

    @property
    def nbytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.tensors.values())

    def to(self, device: torch.device | str) -> ComfyAdapterFusion:
        """The same layer with its saved tensors on ``device``, unchanged in dtype and value."""
        tensors = {key: value.to(device=device) for key, value in self.tensors.items()}
        return ComfyAdapterFusion(self.kind, MappingProxyType(tensors), self.alpha)

    @torch.no_grad()
    def reconstruct_delta(
        self, weight_shape: Sequence[int], *, device: torch.device | str, dtype: torch.dtype,
    ) -> tuple[Tensor, float]:
        """Return the unscaled delta and file alpha/rank, with no autograd graph."""
        if dtype not in _DTYPES:
            raise ValueError("adapter fusion dtype must be float16, bfloat16 or float32")
        device = torch.device(device)
        shape = tuple(weight_shape)
        with torch.autocast(device_type=device.type, enabled=False):
            def cast(name: str) -> Tensor:
                return self.tensors[name].to(device=device, dtype=dtype)

            if self.kind == "full":
                diff = cast("diff")
                if diff.shape != shape:
                    raise ValueError("Full adapter difference does not match the base weight shape")
                return diff.clone(), 1.0
            return exported_factor_delta(self.tensors, cast, self.alpha, shape)

    @torch.no_grad()
    def fuse(
        self, base_weight: Tensor, delta: Tensor, alpha: float, *, dtype: torch.dtype, strength: float = 1.0,
    ) -> Tensor:
        """Apply this layer's reconstructed delta to a new tensor, in the inference ``dtype``: Full is skipped
        at zero strength and DoRA uses the exported magnitude, as ComfyUI does."""
        strength = _strength(strength)
        if self.kind == "full" and strength == 0.0:
            return _copy_base(base_weight, dtype)
        return fuse_adapter_delta(
            base_weight, delta, dtype=dtype, alpha=alpha, magnitude=self.tensors.get("dora_scale"),
            strength=strength,
        )

    @torch.no_grad()
    def fuse_bias(self, base_bias: Tensor | None, *, dtype: torch.dtype, strength: float = 1.0) -> Tensor | None:
        """Apply Full's exported bias difference; other algorithms preserve the base bias."""
        strength = _strength(strength)
        diff = self.tensors.get("diff_b")
        if base_bias is None:
            if diff is not None:
                raise ValueError("Full adapter has a bias difference but the base layer has no bias")
            return None
        with torch.autocast(device_type=base_bias.device.type, enabled=False):
            bias = _copy_base(base_bias, dtype)
            if diff is not None:
                if diff.shape != bias.shape:
                    raise ValueError("Full adapter difference does not match the base bias shape")
                if strength != 0.0:
                    bias += strength * diff.to(device=bias.device, dtype=dtype)
            return bias
