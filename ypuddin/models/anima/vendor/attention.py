# Vendored from kohya-ss/sd-scripts (Apache-2.0) at commit 4e62430; local modifications: import paths only + noted changes.
# Unified attention function supporting various implementations
# ruff: noqa  -- vendored third-party code, kept byte-close to upstream; not linted to project style
#
# Local modifications (see NOTICE.md):
#   * Only the PyTorch SDPA backend ("torch", alias "sdpa") is kept. The xformers / flash-attn / sageattention
#     branches and their optional imports were removed; selecting one of those modes raises NotImplementedError.
#   * In ``split_attn`` mode with no explicit ``seqlens`` the key/value tensors are no longer sliced to the *query*
#     length. Upstream did ``k[i:i+1, :q_len]`` which is a no-op for self-attention but silently truncated the text
#     context in cross-attention whenever the image token count was smaller than the text token count.
#   * ``attn_mode="sage"`` is accepted again through a tiny dispatch shim (``_sdpa``): unmasked, dropout-free calls go
#     to ``sageattention.sageattn`` when the package is installed; masked calls (text cross-attention) and dropout fall
#     back to PyTorch SDPA, which is exactly what upstream's sage branch did.
"""Minimal SDPA attention helper used by :mod:`ypuddin.models.anima.vendor.cosmos_dit`.

Call pattern (identical to sd-scripts ``library.attention``)::

    attn_params = AttentionParams.create_attention_params("torch", split_attn=False)
    out = attention([q, k, v], attn_params=attn_params)      # q/k/v: (B, L, H, D)  ->  out: (B, L_q, H*D)

``q`` may have a different sequence length than ``k``/``v`` (cross-attention).  ``attention_mask`` (when given via
``create_attention_params_from_mask``) is a boolean SDPA mask broadcastable to ``(B, H, L_q, L_kv)``.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch

_TORCH_MODES = ("torch", "sdpa", None)
_SAGE_MODES = ("sage", "sageattn")

try:  # optional, CUDA-only accelerator (int8 QK^T); never required
    from sageattention import sageattn as _sageattn  # type: ignore
except Exception:  # noqa: BLE001  (ImportError or a CUDA-less install failing at import time)
    _sageattn = None


def sage_available() -> bool:
    return _sageattn is not None


def _is_torch_mode(attn_mode: Optional[str]) -> bool:
    return attn_mode in _TORCH_MODES or attn_mode in _SAGE_MODES


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mode: Optional[str],
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """SDPA-compatible dispatch: [B, H, L, D] in, [B, H, L, D] out."""
    if attn_mode in _SAGE_MODES and attn_mask is None and dropout_p == 0.0 and q.is_cuda:
        if _sageattn is None:
            raise RuntimeError("attn_mode='sage' requires the sageattention package (pip install sageattention)")
        return _sageattn(q, k, v, tensor_layout="HND", is_causal=False)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)


@dataclass
class AttentionParams:
    attn_mode: Optional[str] = None
    split_attn: bool = False
    img_len: Optional[int] = None
    attention_mask: Optional[torch.Tensor] = None
    seqlens: Optional[torch.Tensor] = None
    cu_seqlens: Optional[torch.Tensor] = None
    max_seqlen: Optional[int] = None

    @property
    def supports_fp32(self) -> bool:
        # flash-attn (removed) was the only backend without fp32 support
        return self.attn_mode not in ["flash"]

    @property
    def requires_same_dtype(self) -> bool:
        # xformers (removed) was the only backend requiring identical q/k/v dtypes
        return self.attn_mode in ["xformers"]

    @staticmethod
    def create_attention_params(attn_mode: Optional[str], split_attn: bool) -> "AttentionParams":
        return AttentionParams(attn_mode, split_attn)

    @staticmethod
    def create_attention_params_from_mask(
        attn_mode: Optional[str], split_attn: bool, img_len: Optional[int], attention_mask: Optional[torch.Tensor]
    ) -> "AttentionParams":
        if attention_mask is None:
            # No attention mask provided: assume all tokens are valid
            return AttentionParams(attn_mode, split_attn, None, None, None, None, None)

        if not _is_torch_mode(attn_mode):
            raise NotImplementedError(f"Only the 'torch' attention backend is vendored; got attn_mode={attn_mode!r}")

        # Note: attention_mask is only for text tokens, not including image tokens
        seqlens = attention_mask.sum(dim=1).to(torch.int32) + img_len  # [B]
        max_seqlen = attention_mask.shape[1] + img_len

        if split_attn:
            # cu_seqlens is not needed for split attention
            return AttentionParams(attn_mode, split_attn, img_len, attention_mask, seqlens, None, max_seqlen)

        # Expand attention mask to include image tokens
        attention_mask = torch.nn.functional.pad(attention_mask, (img_len, 0), value=1)  # [B, img_len + L]
        attention_mask = attention_mask[:, None, None, :].to(torch.bool)  # [B, 1, 1, img_len + L]

        return AttentionParams(attn_mode, split_attn, img_len, attention_mask, seqlens, None, max_seqlen)


def attention(
    qkv_or_q: Union[torch.Tensor, list],
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    attn_params: Optional[AttentionParams] = None,
    drop_rate: float = 0.0,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention with variable sequence lengths.

    Handles batches with different sequence lengths by splitting and
    processing each sequence individually.

    Args:
        qkv_or_q: Query tensor [B, L, H, D]. or list of such tensors.
        k: Key tensor [B, L, H, D].
        v: Value tensor [B, L, H, D].
        attn_params: Attention parameters including mask and sequence lengths.
        drop_rate: Attention dropout rate.

    Returns:
        Attention output tensor [B, L, H*D].
    """
    if isinstance(qkv_or_q, list):
        q, k, v = qkv_or_q
        q: torch.Tensor = q
        qkv_or_q.clear()
        del qkv_or_q
    else:
        q: torch.Tensor = qkv_or_q
        del qkv_or_q
        assert k is not None and v is not None, "k and v must be provided if qkv_or_q is a tensor"
    if attn_params is None:
        attn_params = AttentionParams.create_attention_params("torch", False)

    if not _is_torch_mode(attn_params.attn_mode):
        raise NotImplementedError(
            f"Only the 'torch' (SDPA) and 'sage' attention backends are vendored; got attn_mode={attn_params.attn_mode!r}. "
            "xformers / flash-attn were removed from this copy (PyTorch SDPA already dispatches to flash kernels)."
        )

    # If split attn is False, attention mask is provided and all sequence lengths are same, we can trim the sequence
    seqlen_trimmed = False
    if not attn_params.split_attn and attn_params.attention_mask is not None and attn_params.seqlens is not None:
        if torch.all(attn_params.seqlens == attn_params.seqlens[0]):
            seqlen = attn_params.seqlens[0].item()
            q = q[:, :seqlen]
            k = k[:, :seqlen]
            v = v[:, :seqlen]
            max_seqlen = attn_params.max_seqlen
            attn_params = AttentionParams.create_attention_params(attn_params.attn_mode, False)  # do not in-place modify
            attn_params.max_seqlen = max_seqlen  # keep max_seqlen for padding
            seqlen_trimmed = True

    # Tensor layout for SDPA: [B, H, L, D]
    transpose_fn = lambda x: x.transpose(1, 2)  # noqa: E731
    # pad on sequence length dimension
    pad_fn = lambda x, pad_to: torch.nn.functional.pad(x, (0, 0, 0, pad_to - x.shape[-2]), value=0)  # noqa: E731

    # Process each batch element with its valid sequence lengths
    if attn_params.split_attn:
        if attn_params.seqlens is None:
            # No seqlens provided: every token is valid. q and k/v keep their own full lengths (cross-attention safe).
            max_seqlen = q.shape[1]
            q = [transpose_fn(q[i : i + 1]) for i in range(len(q))]
            k = [transpose_fn(k[i : i + 1]) for i in range(len(k))]
            v = [transpose_fn(v[i : i + 1]) for i in range(len(v))]
        else:
            max_seqlen = attn_params.max_seqlen
            q = [transpose_fn(q[i : i + 1, : attn_params.seqlens[i]]) for i in range(len(q))]
            k = [transpose_fn(k[i : i + 1, : attn_params.seqlens[i]]) for i in range(len(k))]
            v = [transpose_fn(v[i : i + 1, : attn_params.seqlens[i]]) for i in range(len(v))]

        x = []
        for i in range(len(q)):
            x_i = _sdpa(q[i], k[i], v[i], attn_params.attn_mode, dropout_p=drop_rate)
            q[i] = None
            k[i] = None
            v[i] = None
            x.append(pad_fn(x_i, max_seqlen))  # B, H, L, D
        x = torch.cat(x, dim=0)
        del q, k, v
    else:
        q = transpose_fn(q)
        k = transpose_fn(k)
        v = transpose_fn(v)
        x = _sdpa(q, k, v, attn_params.attn_mode, attn_mask=attn_params.attention_mask, dropout_p=drop_rate)
        del q, k, v

    x = transpose_fn(x)  # [B, L, H, D]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, L, H*D]

    if seqlen_trimmed:
        x = torch.nn.functional.pad(x, (0, 0, 0, attn_params.max_seqlen - x.shape[1]), value=0)  # pad back to max_seqlen

    return x


__all__ = ["AttentionParams", "attention", "sage_available"]
