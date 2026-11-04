"""FLUX.3 self-attention over packed, isolated sample sequences."""

import torch
import torch.nn.functional as F
from torch import Tensor


def _compatible(q: Tensor, k: Tensor, v: Tensor) -> bool:
    return (q.is_cuda and torch.version.hip is None
            and q.dtype in (torch.float16, torch.bfloat16) and q.dtype == k.dtype == v.dtype
            and q.device == k.device == v.device and q.shape[-1] <= 256 and q.shape[-1] % 8 == 0)


def _flash_varlen_function():
    from flash_attn import flash_attn_varlen_func

    return flash_attn_varlen_func


def _varlen_attention(q: Tensor, k: Tensor, v: Tensor, bounds: list[tuple[int, int]]) -> Tensor:
    _, heads, length, dim = q.shape
    cumulative, maximum, end = [0], 0, 0
    for start, stop in bounds:
        if start != end or stop < start or stop > length:
            raise ValueError("FLUX.3 attention bounds must partition tokens in their original order")
        end = stop
        if stop > start:
            cumulative.append(stop)
            maximum = max(maximum, stop - start)
    if maximum == 0:
        # The DiT deliberately runs projections on one empty-mode phantom token.
        # It does not enter attention or the joint stream.
        return q.new_zeros(length, heads * dim)
    if end != length:
        raise ValueError("FLUX.3 attention bounds do not cover all real tokens")
    cu = torch.tensor(cumulative, device=q.device, dtype=torch.int32)
    tensors = [value[0].transpose(0, 1).contiguous() for value in (q, k, v)]
    output = _flash_varlen_function()(
        *tensors, cu, cu, maximum, maximum, dropout_p=0.0, softmax_scale=None,
        causal=False, deterministic=torch.are_deterministic_algorithms_enabled(),
    )
    return output.reshape(length, heads * dim)


def packed_attention(q: Tensor, k: Tensor, v: Tensor, bounds: list[tuple[int, int]],
                     mode: str = "sdpa") -> Tensor:
    """Q/K already include the original normalization and RoPE: (1,H,total,D)."""
    if mode == "flash_attn":
        if not _compatible(q, k, v):
            raise NotImplementedError("FLUX.3 FA2 varlen requires NVIDIA CUDA FP16/BF16 with head dim <=256 divisible by 8")
        return _varlen_attention(q, k, v, bounds)
    elif mode != "sdpa":
        raise ValueError(f"Unsupported FLUX.3 attention mode: {mode}")
    # Preserve the original default path and its exact ordering.
    _, heads, length, dim = q.shape
    out = torch.zeros(length, heads * dim, device=q.device, dtype=q.dtype)
    for start, end in bounds:
        if start == end:
            continue
        chunk = F.scaled_dot_product_attention(q[:, :, start:end], k[:, :, start:end], v[:, :, start:end])
        out[start:end] = chunk[0].transpose(0, 1).reshape(end - start, heads * dim)
    return out
