"""OrthoLoRA: an orthogonal update of each layer's principal subspace (PSOFT, arXiv 2505.11235).

With the top ``r`` singular triplets of the frozen weight, ``W₀ ≈ U diag(s) Vᵀ``, held fixed, the layer
learns ``C = diag(a) · R · diag(b)``, where ``R = (I + A)⁻¹(I - A)`` is the Cayley rotation of a
skew-symmetric ``A``, and adds ``ΔW = scale · U (C - I) diag(s) Vᵀ``. At the start ``A = 0`` and
``a = b = 1``, so ``ΔW = 0`` while every parameter already has a gradient. The rotation keeps the angles
between the principal directions and ``a``, ``b`` let their lengths change. ``ΔW`` has rank ``r`` and is
exported exactly as a plain LoRA. A convolution's weight is taken as the matrix ``(out, in·k…)``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .base import AdapterModule, compute_scale


def principal_subspace(weight: Tensor, rank: int) -> tuple[Tensor, Tensor, Tensor]:
    """Top ``rank`` singular triplets of ``weight``: ``U (out, r)``, ``s (r)``, ``V (in, r)``."""
    small = min(weight.shape)
    if small <= 1024 or rank * 4 >= small:
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        return u[:, :rank], s[:rank], vh[:rank].t()
    # Generous oversampling and power iterations find the top subspace almost exactly even for
    # slowly decaying spectra, at a fraction of the cost of a full decomposition of a large layer.
    u, s, v = torch.svd_lowrank(weight, q=min(small, 2 * rank + 64), niter=12)
    return u[:, :rank], s[:rank], v[:, :rank]


class OrthoLoRA(AdapterModule):
    kind = "ortho"
    supports_bypass = True

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
        if init == "scalar":
            raise ValueError(
                "OrthoLoRA starts from the layer's own singular vectors; the scalar init does not apply"
            )
        if int(rank) <= 0:
            raise ValueError("rank must be positive")
        # A layer has at most min(out, in·k…) principal directions.
        self.rank = min(int(rank), int(out_features), self.fan_in)
        self.alpha = float(alpha)
        self.rs_lora = bool(rs_lora)
        self.scale = compute_scale(self.alpha, self.rank, self.rs_lora)
        self.rotation = nn.Parameter(torch.zeros(self.rank, self.rank, dtype=dtype))
        self.out_scale = nn.Parameter(torch.ones(self.rank, dtype=dtype))
        self.in_scale = nn.Parameter(torch.ones(self.rank, dtype=dtype))
        self.register_buffer("basis_out", torch.zeros(out_features, self.rank, dtype=dtype))
        self.register_buffer("singular", torch.zeros(self.rank, dtype=dtype))
        self.register_buffer("basis_in", torch.zeros(self.rank, self.fan_in, dtype=dtype))
        self._ready = False

    # ----------------------------------------------------------------- init
    @torch.no_grad()
    def bind_base(self, base_weight: Tensor, bias: Tensor | None = None) -> None:
        """Take the principal subspace of the frozen weight; meta layers (planning) stay unset."""
        if self._ready or base_weight.device.type == "meta" or self.rotation.device.type == "meta":
            return
        where = base_weight.device if base_weight.device.type == "cuda" else torch.device("cpu")
        matrix = base_weight.detach().to(device=where, dtype=torch.float32).reshape(self.out_features, -1)
        u, s, v = principal_subspace(matrix, self.rank)
        device = self.basis_out.device
        self.basis_out.copy_(u.to(device=device, dtype=self.basis_out.dtype))
        self.singular.copy_(s.to(device=device, dtype=self.singular.dtype))
        self.basis_in.copy_(v.t().to(device=device, dtype=self.basis_in.dtype))
        self._ready = True

    def _ensure_ready(self) -> None:
        if not self._ready:
            raise RuntimeError("OrthoLoRA has no principal subspace yet; bind it to its base weight first")

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):  # noqa: ANN001
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        # A restored checkpoint carries the subspace it was trained in.
        if f"{prefix}basis_out" in state_dict:
            self._ready = True

    # ----------------------------------------------------------------- compute
    def _core(self) -> Tensor:
        """``C - I`` in float32."""
        a = self.rotation.float()
        eye = torch.eye(self.rank, device=a.device)
        rotation = torch.linalg.solve(eye + a - a.t(), eye - a + a.t())
        return self.out_scale.float()[:, None] * rotation * self.in_scale.float()[None, :] - eye

    def delta_weight(self) -> Tensor:
        self._ensure_ready()
        dt = self.basis_out.dtype
        keep = self._rank_mask(self.rank, self.basis_out.device, dt)
        left = self.basis_out if keep is None else self.basis_out * keep
        return self._as_weight(self._scaled(((left @ self._core().to(dt)) * self.singular) @ self.basis_in))

    def delta_apply(self, x: Tensor) -> Tensor:
        self._ensure_ready()
        dt = x.dtype
        h = self._per_rank(self._rank_in(x, self.basis_in.to(dt)), self.singular.to(dt))
        h = self._rank_out(h, self._core().to(dt))
        keep = self._rank_mask(self.rank, x.device, dt)
        if keep is not None:
            h = self._per_rank(h, keep)
        return self._finish(self._rank_out(h, self.basis_out.to(dt)))

    # ----------------------------------------------------------------- io
    @torch.no_grad()
    def export_tensors(self) -> dict[str, Tensor]:
        """The exact rank-``r`` plain LoRA; ``√s`` goes to both factors to keep them in range."""
        self._ensure_ready()
        root = self.singular.float().abs().sqrt()
        up = (self.basis_out.float() @ self._core()) * root[None, :] * self.scale
        down = root[:, None] * self.basis_in.float()
        return self._lora_export(down, up, self.rank)

    def extra_metadata(self) -> dict[str, Any]:
        """Describes the exported layer, a plain LoRA, so any LoRA reader rebuilds it."""
        return {
            "algo": "lora",
            "rank": self.rank,
            "alpha": float(self.rank),
            "rs_lora": False,
            **self._shape_metadata(),
            "trained_as": "ortho",
        }

    @classmethod
    def from_tensors(
        cls, tensors: dict[str, Tensor], meta: dict[str, Any] | None = None, **kwargs: Any
    ) -> OrthoLoRA:
        raise ValueError("OrthoLoRA cannot continue from an exported LoRA; resume from a recovery point")

    def param_kinds(self) -> dict[str, str]:
        return {"rotation": "rotation", "out_scale": "scale", "in_scale": "scale"}
