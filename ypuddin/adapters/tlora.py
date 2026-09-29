"""T-LoRA: LoRA whose usable rank follows each sample's noise level (arXiv 2507.05964).

At noise level ``t`` (0 clean, 1 pure noise) only the first ``r(t) = ⌊(1 - t)^p · (R - R_min)⌋ + R_min``
ranks act, so the noisiest steps, where a small dataset is learned too literally, train a few directions
and the clean end trains all of them. The mask is per sample and exists only while training; previews
and exported files use every rank.

With ``ortho`` the factors start orthonormal with small singular values, as the last singular
components of a random matrix in the paper's full method, and a frozen copy of that start is
subtracted, so the ranks start independent and ``ΔW`` starts at zero whatever the mask. The export
carries both terms exactly, as a plain LoRA of rank ``2R``. A convolution's factors see its weight as
the matrix ``(out, in·k…)``, as LoRA's do.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .base import AdapterModule, compute_scale, kaiming_uniform_
from .lora import from_lora_tensors


def active_ranks(level: Tensor, rank: int, min_rank: int, power: float) -> Tensor:
    """Ranks each sample may use at its noise level (0 clean … 1 pure noise), the paper's schedule."""
    level = level.detach().float().reshape(-1).clamp(0.0, 1.0)
    return torch.floor((1.0 - level).pow(power) * (rank - min_rank)) + min_rank


def rank_mask(level: Tensor, rank: int, min_rank: int, power: float) -> Tensor:
    """``(batch, rank)`` with ones on the ranks each sample uses."""
    active = active_ranks(level, rank, min_rank, power)
    return (torch.arange(rank, device=active.device) < active[:, None]).float()


# Singular values of a random matrix concentrate tightly, so layers of one shape share a draw.
_SPECTRA: dict[tuple[int, int, int], Tensor] = {}


def _smallest_singular_values(out_features: int, in_features: int, rank: int, device: torch.device) -> Tensor:
    """The ``rank`` smallest singular values of an ``N(0, 1/rank²)`` matrix of this shape, largest first."""
    key = (out_features, in_features, rank)
    if key not in _SPECTRA:
        # linalg on MPS is incomplete; the spectrum is tiny, so CPU is fine there.
        where = device if device.type == "cuda" else torch.device("cpu")
        sample = torch.randn(in_features, out_features, device=where) / rank
        _SPECTRA[key] = torch.linalg.svdvals(sample)[-rank:].cpu()
    return _SPECTRA[key].to(device)


def _orthonormal(rows: int, cols: int, device: torch.device) -> Tensor:
    """A random ``(rows, cols)`` matrix with orthonormal columns (``rows >= cols``)."""
    where = device if device.type == "cuda" else torch.device("cpu")
    q, r = torch.linalg.qr(torch.randn(rows, cols, device=where))
    # Fixing the signs makes the draw uniform over orthonormal frames.
    signs = torch.sign(torch.diagonal(r))
    signs[signs == 0] = 1
    return (q * signs).to(device)


class TLoRA(AdapterModule):
    kind = "tlora"
    supports_bypass = True
    needs_bypass = True

    def __init__(
        self,
        out_features: int,
        in_features: int,
        *,
        rank: int = 16,
        alpha: float = 16.0,
        rs_lora: bool = False,
        min_rank: int | None = None,
        power: float = 1.0,
        ortho: bool = True,
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
        if ortho and init == "scalar":
            raise ValueError("orthogonal T-LoRA starts from its own factors; the scalar init does not apply")
        self.alpha = float(alpha)
        self.rs_lora = bool(rs_lora)
        self.scale = compute_scale(self.alpha, self.rank, self.rs_lora)
        # The paper recommends about half the rank at the noisiest end.
        self.min_rank = max(1, min(self.rank, int(min_rank) if min_rank else self.rank // 2))
        self.power = float(power)
        if not self.power > 0:
            raise ValueError("power must be positive")
        self.ortho = bool(ortho)
        self.down = nn.Parameter(torch.empty(self.rank, self.fan_in, dtype=dtype))
        self.up = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype))
        if self.ortho:
            self.lam = nn.Parameter(torch.zeros(self.rank, dtype=dtype))
            self.register_buffer("down0", torch.zeros(self.rank, self.fan_in, dtype=dtype))
            self.register_buffer("up0", torch.zeros(out_features, self.rank, dtype=dtype))
            self.register_buffer("lam0", torch.zeros(self.rank, dtype=dtype))
        self._mask: Tensor | None = None
        self._ortho_ready = False
        self.reset_parameters()

    # ----------------------------------------------------------------- init
    @torch.no_grad()
    def reset_parameters(self) -> None:
        if self.ortho:
            # The orthonormal start is drawn once the module sits on its device (bind_base).
            self._ortho_ready = False
            return
        kaiming_uniform_(self.down)
        if self.init_mode == "scalar":
            kaiming_uniform_(self.up)
        else:
            self.up.zero_()
        if self.scalar is not None:
            self.scalar.zero_()

    @torch.no_grad()
    def bind_base(self, base_weight: Tensor, bias: Tensor | None = None) -> None:
        """Draw the orthonormal start on the layer's device; meta layers (planning) stay unset."""
        if not self.ortho or self._ortho_ready or self.down.device.type == "meta":
            return
        device = self.down.device
        up = _orthonormal(self.out_features, self.rank, device) if self.out_features >= self.rank else None
        down = _orthonormal(self.fan_in, self.rank, device) if self.fan_in >= self.rank else None
        if up is None or down is None:
            raise ValueError(
                "orthogonal T-LoRA needs a rank no larger than the layer's input and output size"
            )
        lam = _smallest_singular_values(self.out_features, self.fan_in, self.rank, device)
        for trained, frozen, value in (
            (self.up, self.up0, up),
            (self.down, self.down0, down.t()),
            (self.lam, self.lam0, lam),
        ):
            trained.copy_(value.to(trained.dtype))
            frozen.copy_(value.to(frozen.dtype))
        self._ortho_ready = True

    def _ensure_ready(self) -> None:
        if self.ortho and not self._ortho_ready:
            self.bind_base(self.up)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):  # noqa: ANN001
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        # A restored checkpoint carries its own start; it must not be redrawn.
        if self.ortho and f"{prefix}lam0" in state_dict:
            self._ortho_ready = True

    # ----------------------------------------------------------------- mask
    def set_mask(self, mask: Tensor | None) -> None:
        """Per-sample ``(batch, rank)`` mask of usable ranks; ``None`` uses every rank."""
        self._mask = mask

    def _sample_mask(self, x: Tensor) -> Tensor | None:
        mask = self._mask
        if mask is None or not self.training:
            return None
        if mask.shape[0] not in (1, x.shape[0]):
            raise RuntimeError(
                f"T-LoRA mask covers {mask.shape[0]} samples but the layer input has {x.shape[0]}"
            )
        return mask.to(device=x.device, dtype=x.dtype)

    # ----------------------------------------------------------------- compute
    def delta_weight(self) -> Tensor:
        """``ΔW`` with every rank, as previews and the export use it."""
        self._ensure_ready()
        keep = self._rank_mask(self.rank, self.up.device, self.up.dtype)
        up = self.up if keep is None else self.up * keep
        if not self.ortho:
            return self._as_weight(self._scaled(up @ self.down))
        up0 = self.up0 if keep is None else self.up0 * keep
        return self._as_weight(self._scaled((up * self.lam) @ self.down - (up0 * self.lam0) @ self.down0))

    def delta_apply(self, x: Tensor) -> Tensor:
        self._ensure_ready()
        dt = x.dtype
        weight = self._sample_mask(x)
        keep = self._rank_mask(self.rank, x.device, dt)
        if keep is not None:
            weight = keep if weight is None else weight * keep
        if not self.ortho:
            h = self._rank_in(x, self.down.to(dt))
            if weight is not None:
                h = self._per_rank(h, weight)
            return self._finish(self._rank_out(h, self.up.to(dt)))
        # Trained and frozen terms share one pass: [Q; Q₀] in, [P·λ, -P₀·λ₀] out.
        h = self._rank_in(x, torch.cat((self.down, self.down0)).to(dt))
        gains = torch.cat((self.lam, -self.lam0)).to(dt)
        h = self._per_rank(h, gains if weight is None else gains * torch.cat((weight, weight), dim=-1))
        return self._finish(self._rank_out(h, torch.cat((self.up, self.up0), dim=1).to(dt)))

    # ----------------------------------------------------------------- io
    @torch.no_grad()
    def export_tensors(self) -> dict[str, Tensor]:
        """A plain LoRA with every rank; the orthogonal form exports both terms exactly at rank ``2R``."""
        self._ensure_ready()
        if not self.ortho:
            scalar = float(self.effective_scalar) if isinstance(self.effective_scalar, Tensor) else 1.0
            up = (self.up * scalar).detach().clone()
            return self._lora_export(self.down.detach().clone(), up, self.scale * self.rank)
        root = self.scale**0.5
        lam, lam0 = self.lam.float(), self.lam0.float()
        size, size0 = lam.abs().sqrt(), lam0.abs().sqrt()
        up = torch.cat((self.up.float() * (torch.sign(lam) * size), -self.up0.float() * size0), dim=1) * root
        down = torch.cat((self.down.float() * size[:, None], self.down0.float() * size0[:, None])) * root
        return self._lora_export(down, up, down.shape[0])

    def extra_metadata(self) -> dict[str, Any]:
        """Describes the exported layer, a plain LoRA, so any LoRA reader rebuilds it."""
        rank = 2 * self.rank if self.ortho else self.rank
        return {
            "algo": "lora",
            "rank": rank,
            "alpha": float(rank) if self.ortho else self.alpha,
            "rs_lora": self.rs_lora and not self.ortho,
            **self._shape_metadata(),
            "trained_as": "tlora",
            "tlora_min_rank": self.min_rank,
            "tlora_power": self.power,
            "tlora_ortho": self.ortho,
        }

    @classmethod
    def from_tensors(
        cls, tensors: dict[str, Tensor], meta: dict[str, Any] | None = None, **kwargs: Any
    ) -> TLoRA:
        """Warm start from a plain LoRA; only the non-orthogonal form has the same factors."""
        meta = meta or {}
        if meta.get("tlora_ortho"):
            raise ValueError(
                "orthogonal T-LoRA cannot continue from an exported LoRA; resume from a recovery point"
            )
        return from_lora_tensors(
            cls,
            tensors,
            meta,
            kwargs,
            min_rank=meta.get("tlora_min_rank"),
            power=float(meta.get("tlora_power", 1.0)),
            ortho=False,
        )

    def param_kinds(self) -> dict[str, str]:
        return {"down": "down", "up": "up", "lam": "lambda", "scalar": "scalar"}
