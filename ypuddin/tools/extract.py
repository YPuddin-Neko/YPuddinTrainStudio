"""Extraction / resizing utilities operating on weight deltas.

* ``extract_lora``   : truncated SVD of ΔW -> (down, up)
* ``extract_lokr``   : nearest Kronecker product (Van Loan–Pitsianis) -> W1 ⊗ W2, optional low-rank W2
* ``resize_lora``    : SVD re-factorization of a LoRA pair to a new rank
* ``extract_from_state_dicts`` : diff two full model state dicts into a kohya-format adapter file
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

from ypuddin.adapters.factorize import factorization


@dataclass
class ExtractResult:
    tensors: dict[str, Tensor]  # kohya suffix -> tensor
    residual: float  # relative Frobenius residual ||ΔW - approx|| / ||ΔW||
    meta: dict


def _rel_residual(delta: Tensor, approx: Tensor) -> float:
    denom = delta.norm().item()
    return 0.0 if denom == 0 else (delta - approx).norm().item() / denom


def extract_lora(delta: Tensor, rank: int, *, alpha: float | None = None) -> ExtractResult:
    """``ΔW ≈ up @ down`` with ``down (r, in)``, ``up (out, r)``; alpha defaults to rank (scale 1)."""
    d = delta.detach().float()
    r = min(int(rank), min(d.shape))
    U, S, Vh = torch.linalg.svd(d, full_matrices=False)
    s = S[:r].sqrt()
    up = U[:, :r] * s[None, :]
    down = s[:, None] * Vh[:r]
    alpha = float(alpha if alpha is not None else r)
    scale = alpha / r
    approx = up @ down
    # exported weights must reproduce ΔW under scale = alpha / rank
    up = up / scale
    return ExtractResult(
        {"lora_down.weight": down, "lora_up.weight": up, "alpha": torch.tensor(alpha)},
        _rel_residual(d, approx),
        {"algo": "lora", "rank": r, "alpha": alpha},
    )


def nearest_kronecker(delta: Tensor, a: int, b: int, c: int, d: int) -> tuple[Tensor, Tensor, float]:
    """Best rank-1 Kronecker approximation ``ΔW ≈ W1 ⊗ W2`` with ``W1 (a,c)``, ``W2 (b,d)``.

    Rearranges ΔW ``(a·b, c·d)`` into ``R (a·c, b·d)`` with ``R[(i1,j1),(i2,j2)] = ΔW[i1·b+i2, j1·d+j2]``;
    the leading singular pair of ``R`` gives the factors (Van Loan & Pitsianis, 1993).
    """
    D = delta.detach().float().reshape(a, b, c, d).permute(0, 2, 1, 3).reshape(a * c, b * d)
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    w1 = (U[:, 0] * S[0]).reshape(a, c)
    w2 = Vh[0].reshape(b, d)
    approx = torch.kron(w1, w2)
    return w1, w2, _rel_residual(delta.float(), approx)


def extract_lokr(
    delta: Tensor, *, factor: int = -1, rank: int | str | None = "full", alpha: float | None = None
) -> ExtractResult:
    out_f, in_f = delta.shape
    a, b = factorization(out_f, factor)
    c, d = factorization(in_f, factor)
    w1, w2, residual = nearest_kronecker(delta, a, b, c, d)
    tensors: dict[str, Tensor] = {"lokr_w1": w1}
    meta = {"algo": "lokr", "shape": [[a, b], [c, d]], "factor": factor}
    r = None if rank in (None, "full") else int(rank)
    if r is not None and r < max(b, d) / 2:
        U, S, Vh = torch.linalg.svd(w2, full_matrices=False)
        s = S[:r].sqrt()
        w2_a = U[:, :r] * s[None, :]
        w2_b = s[:, None] * Vh[:r]
        alpha_v = float(alpha if alpha is not None else r)
        scale = alpha_v / r
        tensors["lokr_w2_a"] = w2_a / scale
        tensors["lokr_w2_b"] = w2_b
        tensors["alpha"] = torch.tensor(alpha_v)
        approx = torch.kron(w1, w2_a @ w2_b)
        residual = _rel_residual(delta.float(), approx)
        meta.update({"rank": r, "alpha": alpha_v})
    else:
        tensors["lokr_w2"] = w2
        tensors["alpha"] = torch.tensor(1.0)
        meta.update({"rank": None, "alpha": 1.0})
    return ExtractResult(tensors, residual, meta)


def resize_lora(down: Tensor, up: Tensor, alpha: float, new_rank: int) -> ExtractResult:
    """Re-factor ``scale · up @ down`` at a new rank (alpha := new rank, i.e. scale 1)."""
    old_rank = down.shape[0]
    delta = (up.float() @ down.float()) * (alpha / old_rank)
    return extract_lora(delta, new_rank)


def extract_from_state_dicts(
    base: dict[str, Tensor],
    tuned: dict[str, Tensor],
    *,
    algo: str = "lora",
    rank: int | str = 16,
    factor: int = -1,
    prefix: str = "lora_unet",
    include: Callable[[str], bool] | None = None,
    min_residual_improvement: float = 0.0,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, Tensor], dict[str, dict]]:
    """Diff two full models (same architecture) into a kohya-format adapter."""
    keys = [
        k
        for k in tuned
        if k in base and k.endswith(".weight") and tuned[k].dim() == 2 and (include is None or include(k))
    ]
    out: dict[str, Tensor] = {}
    report: dict[str, dict] = {}
    for i, k in enumerate(keys):
        delta = tuned[k].float() - base[k].float()
        if delta.abs().max() == 0:
            continue
        module = k[: -len(".weight")]
        kohya = f"{prefix}_{module.replace('.', '_')}"
        if algo == "lokr":
            res = extract_lokr(delta, factor=factor, rank=rank)
        else:
            res = extract_lora(delta, int(rank))
        for suffix, t in res.tensors.items():
            out[f"{kohya}.{suffix}"] = t.contiguous()
        report[module] = {"residual": res.residual, **res.meta}
        if progress:
            progress(i + 1, len(keys))
    return out, report


def iter_linear_deltas(base: dict[str, Tensor], tuned: dict[str, Tensor]) -> Iterable[tuple[str, Tensor]]:
    for k in tuned:
        if k in base and k.endswith(".weight") and tuned[k].dim() == 2:
            yield k[: -len(".weight")], tuned[k].float() - base[k].float()
