"""One tiny Triton kernel on the GPU, for the environment check of an installed Triton.

Imported only by the probe process after ``import triton`` succeeded: ``triton.jit`` reads the
kernel's source, so it has to live in a module file.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out_ptr + offsets, tl.load(x_ptr + offsets, mask=mask) + tl.load(y_ptr + offsets, mask=mask), mask=mask)


def run(device: str = "cuda") -> None:
    x = torch.rand(1000, device=device)
    y = torch.rand_like(x)
    out = torch.empty_like(x)
    _add[(triton.cdiv(x.numel(), 256),)](x, y, out, x.numel(), BLOCK=256)
    torch.cuda.synchronize()
    if not torch.equal(out, x + y):
        raise RuntimeError("Triton 内核的计算结果不正确。")
