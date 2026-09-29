"""LoKr: ``ΔW = scale · (W1 ⊗ W2)`` with optional low-rank factors.

Shapes for ``Linear(in -> out)`` with ``(a, b) = factorization(out)`` and ``(c, d) = factorization(in)``::

    w1: (a, c)   or   w1_a: (a, r), w1_b: (r, c)      (decompose_both)
    w2: (b, d)   or   w2_a: (b, r), w2_b: (r, d)      (rank < max(b, d) / 2)

A convolution factors its channels the same way and keeps the kernel in ``W2``, as LyCORIS does:
``w2: (b, d, *k)`` or ``w2_b: (r, d·k…)``; ``W1 ⊗ W2`` over the flattened ``(b, d·k…)`` is the kernel.

The bypass path uses ``(W1 ⊗ W2) vec(X) = vec(W1 · X · W2ᵀ)`` and never materializes ``ΔW``; a
convolution runs ``W2`` over each of the ``c`` input channel groups and mixes the groups with ``W1``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from .base import AdapterModule, compute_scale, kaiming_uniform_, resolve_kernel
from .factorize import factorization


class LoKr(AdapterModule):
    kind = "lokr"
    supports_bypass = True

    def __init__(
        self,
        out_features: int,
        in_features: int,
        *,
        rank: int | str | None = 16,
        alpha: float = 16.0,
        factor: int = -1,
        decompose_both: bool = False,
        rs_lora: bool = False,
        shape: tuple[tuple[int, int], tuple[int, int]] | None = None,
        w1_lowrank: bool | None = None,
        w2_lowrank: bool | None = None,
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
        if shape is None:
            (a, b) = factorization(out_features, factor)
            (c, d) = factorization(in_features, factor)
        else:
            (a, b), (c, d) = shape
            if a * b != out_features or c * d != in_features:
                raise ValueError(f"factorization {shape} does not match ({out_features}, {in_features})")
        self.a, self.b, self.c, self.d = a, b, c, d
        self.factor = int(factor)

        r = None if rank in (None, "full") else int(rank)
        if r is not None and r <= 0:
            raise ValueError("rank must be positive or 'full'")
        self.w2_lowrank = (r is not None and r < max(b, d) / 2) if w2_lowrank is None else bool(w2_lowrank)
        self.w1_lowrank = (
            (decompose_both and r is not None and r < max(a, c) / 2)
            if w1_lowrank is None
            else bool(w1_lowrank)
        )
        if (self.w1_lowrank or self.w2_lowrank) and r is None:
            raise ValueError("low-rank factors require an integer rank")
        self.rank = r if (self.w1_lowrank or self.w2_lowrank) else None
        self.alpha = float(alpha)
        self.rs_lora = bool(rs_lora)
        self.scale = compute_scale(self.alpha, self.rank, self.rs_lora)

        if self.w1_lowrank:
            self.w1_a = nn.Parameter(torch.empty(a, r, dtype=dtype))
            self.w1_b = nn.Parameter(torch.empty(r, c, dtype=dtype))
        else:
            self.w1 = nn.Parameter(torch.empty(a, c, dtype=dtype))
        if self.w2_lowrank:
            self.w2_a = nn.Parameter(torch.empty(b, r, dtype=dtype))
            self.w2_b = nn.Parameter(torch.empty(r, d * math.prod(self.kernel), dtype=dtype))
        else:
            self.w2 = nn.Parameter(torch.empty(b, d, *self.kernel, dtype=dtype))
        self.reset_parameters()

    # ----------------------------------------------------------------- init
    @torch.no_grad()
    def reset_parameters(self) -> None:
        random_all = self.init_mode == "scalar"
        if self.w1_lowrank:
            kaiming_uniform_(self.w1_a)
            kaiming_uniform_(self.w1_b)
        else:
            kaiming_uniform_(self.w1)
        if self.w2_lowrank:
            kaiming_uniform_(self.w2_a)
            if random_all:
                kaiming_uniform_(self.w2_b)
            else:
                self.w2_b.zero_()
        else:
            if random_all:
                kaiming_uniform_(self.w2)
            else:
                self.w2.zero_()
        if self.scalar is not None:
            self.scalar.zero_()

    # ----------------------------------------------------------------- factors
    def _w1(self) -> Tensor:
        return self.w1_a @ self.w1_b if self.w1_lowrank else self.w1

    def _w2(self, rank_mask: Tensor | None = None) -> Tensor:
        if not self.w2_lowrank:
            return self.w2
        w2_a = self.w2_a if rank_mask is None else self.w2_a * rank_mask
        return w2_a @ self.w2_b

    def delta_weight(self) -> Tensor:
        mask = self._rank_mask(self.rank, self.w2_a.device, self.w2_a.dtype) if self.w2_lowrank else None
        return self._as_weight(self._scaled(torch.kron(self._w1(), self._w2(mask).reshape(self.b, -1))))

    def delta_apply(self, x: Tensor) -> Tensor:
        if self.kernel:
            return self._conv_apply(x)
        lead = x.shape[:-1]
        dt = x.dtype
        X = x.reshape(*lead, self.c, self.d)
        if self.w2_lowrank:
            mask = self._rank_mask(self.rank, x.device, dt)
            w2_a = self.w2_a if mask is None else self.w2_a * mask
            H = (X @ self.w2_b.to(dt).transpose(0, 1)) @ w2_a.to(dt).transpose(0, 1)  # (…, c, b)
        else:
            H = X @ self.w2.to(dt).transpose(0, 1)
        H = H.transpose(-1, -2)  # (…, b, c)
        if self.w1_lowrank:
            Y = (H @ self.w1_b.to(dt).transpose(0, 1)) @ self.w1_a.to(dt).transpose(0, 1)  # (…, b, a)
        else:
            Y = H @ self.w1.to(dt).transpose(0, 1)
        return self._finish(Y.transpose(-1, -2).reshape(*lead, self.out_features))

    def _conv_apply(self, x: Tensor) -> Tensor:
        """Input channel ``k·d + l`` meets ``W2`` in group ``k``; ``W1`` then mixes the ``c`` groups."""
        dt = x.dtype
        geometry = self._geometry()
        n = x.shape[0]
        groups = x.reshape(n * self.c, self.d, *x.shape[2:])
        if self.w2_lowrank:
            mask = self._rank_mask(self.rank, x.device, dt)
            w2_a = self.w2_a if mask is None else self.w2_a * mask
            h = geometry.conv(groups, self.w2_b.to(dt).view(self.rank, self.d, *self.kernel))
            h = geometry.pointwise(h, w2_a.to(dt))
        else:
            h = geometry.conv(groups, self.w2.to(dt))
        y = torch.einsum("ik,nkm->nim", self._w1().to(dt), h.reshape(n, self.c, -1))
        return self._finish(y.reshape(n, self.out_features, *h.shape[2:]))

    # ----------------------------------------------------------------- io
    @torch.no_grad()
    def export_tensors(self) -> dict[str, Tensor]:
        """Kohya/LyCORIS layout. Third-party loaders compute ``alpha / rank`` with ``rank`` read
        from ``lokr_w2_b`` (or ``lokr_w1_b``) and use scale 1 when both factors are full, so:
        low-rank -> ``alpha_file = scale * rank`` and scalar folded into w1; full -> scale and
        scalar folded into w1 and ``alpha_file = 1``."""
        scalar = float(self.effective_scalar) if isinstance(self.effective_scalar, Tensor) else 1.0
        out: dict[str, Tensor] = {}
        fold = scalar if self.rank is not None else scalar * self.scale
        if self.w1_lowrank:
            out["lokr_w1_a"] = (self.w1_a * fold).detach().clone()
            out["lokr_w1_b"] = self.w1_b.detach().clone()
        else:
            out["lokr_w1"] = (self.w1 * fold).detach().clone()
        if self.w2_lowrank:
            out["lokr_w2_a"] = self.w2_a.detach().clone()
            out["lokr_w2_b"] = self.w2_b.detach().clone()
        else:
            out["lokr_w2"] = self.w2.detach().clone()
        alpha_file = self.scale * self.rank if self.rank is not None else 1.0
        out["alpha"] = torch.tensor(float(alpha_file), dtype=torch.float32)
        return out

    def extra_metadata(self) -> dict[str, Any]:
        return {
            "algo": "lokr",
            "shape": [[self.a, self.b], [self.c, self.d]],
            "rank": self.rank,
            "alpha": self.alpha,
            "factor": self.factor,
            "rs_lora": self.rs_lora,
            "w1_lowrank": self.w1_lowrank,
            "w2_lowrank": self.w2_lowrank,
            **self._shape_metadata(),
        }

    @classmethod
    def from_tensors(
        cls, tensors: dict[str, Tensor], meta: dict[str, Any] | None = None, **kwargs: Any
    ) -> LoKr:
        w1_lowrank = "lokr_w1_a" in tensors
        w2_lowrank = "lokr_w2_a" in tensors
        kernel = resolve_kernel(meta, kwargs, () if w2_lowrank else tuple(tensors["lokr_w2"].shape[2:]))
        if meta and "shape" in meta:
            (a, b), (c, d) = meta["shape"]
        else:
            # Infer from factor shapes: w1 (a, c), w2 (b, d, *k) (or their low-rank parts).
            a = tensors["lokr_w1_a"].shape[0] if w1_lowrank else tensors["lokr_w1"].shape[0]
            c = tensors["lokr_w1_b"].shape[1] if w1_lowrank else tensors["lokr_w1"].shape[1]
            b = tensors["lokr_w2_a"].shape[0] if w2_lowrank else tensors["lokr_w2"].shape[0]
            d = (
                tensors["lokr_w2_b"].shape[1] // math.prod(kernel)
                if w2_lowrank
                else tensors["lokr_w2"].shape[1]
            )
        out_features, in_features = a * b, c * d
        rank: int | None = None
        if w2_lowrank:
            rank = int(tensors["lokr_w2_b"].shape[0])
        elif w1_lowrank:
            rank = int(tensors["lokr_w1_b"].shape[0])
        alpha_file = float(tensors["alpha"].item()) if "alpha" in tensors else (rank if rank else 1.0)
        # Exported alpha already encodes the effective scale: scale = alpha_file / rank (rsLoRA folded).
        alpha = alpha_file if rank is not None else 1.0
        dtype = kwargs.pop("dtype", torch.float32)
        mod = cls(
            out_features,
            in_features,
            rank=rank if rank is not None else "full",
            alpha=alpha,
            factor=int(meta.get("factor", -1)) if meta else -1,
            shape=((a, b), (c, d)),
            w1_lowrank=w1_lowrank,
            w2_lowrank=w2_lowrank,
            rs_lora=False,
            kernel=kernel,
            dtype=dtype,
            **kwargs,
        )
        with torch.no_grad():
            for name in ("w1", "w1_a", "w1_b", "w2", "w2_a", "w2_b"):
                key = f"lokr_{name}"
                if key in tensors:
                    param = getattr(mod, name)
                    param.copy_(tensors[key].reshape(param.shape).to(dtype))
        return mod

    def param_kinds(self) -> dict[str, str]:
        kinds = {}
        for name, _ in self.named_parameters(recurse=False):
            kinds[name] = "scalar" if name == "scalar" else ("w1" if name.startswith("w1") else "w2")
        return kinds
