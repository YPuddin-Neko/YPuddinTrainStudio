"""Standard DoRA: FP32 linear weights, strength=1, first-order gradients."""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

EPSILON = torch.finfo(torch.float32).eps


@triton.jit
def _forward(base, delta, magnitude, output, raw_norm,
             ROWS: tl.constexpr, COLS: tl.constexpr, INPUT_AXIS: tl.constexpr,
             EPS: tl.constexpr, BLOCK: tl.constexpr):
    kept = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    if INPUT_AXIS:
        offset = k * COLS + kept
        valid = k < ROWS
    else:
        offset = kept * COLS + k
        valid = k < COLS
    merged = tl.load(base + offset, valid, 0) + tl.load(delta + offset, valid, 0)
    r = tl.sqrt(tl.sum(merged * merged, 0))
    d = tl.load(magnitude + kept)
    result = merged * (d / (r + EPS))
    tl.store(raw_norm + kept, r)
    tl.store(output + offset, result, valid)


@triton.jit
def _backward(grad, base, delta, magnitude, raw_norm, grad_merged, grad_magnitude,
              ROWS: tl.constexpr, COLS: tl.constexpr, INPUT_AXIS: tl.constexpr,
              EPS: tl.constexpr, BLOCK: tl.constexpr):
    kept = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    if INPUT_AXIS:
        offset = k * COLS + kept
        valid = k < ROWS
    else:
        offset = kept * COLS + k
        valid = k < COLS
    merged = tl.load(base + offset, valid, 0) + tl.load(delta + offset, valid, 0)
    g = tl.load(grad + offset, valid, 0)
    r = tl.load(raw_norm + kept)
    d = tl.load(magnitude + kept)
    denominator = r + EPS
    q = tl.sum(g * merged, 0)
    safe_r = tl.where(r > 0, r, 1.0)
    correction = d * q / (safe_r * (denominator * denominator))
    correction = tl.where(r > 0, correction, 0.0)
    result = g * (d / denominator) - correction * merged
    tl.store(grad_merged + offset, result, valid)
    tl.store(grad_magnitude + kept, q / denominator)


class StandardDoRA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, delta, magnitude, input_axis):
        rows, cols = base.shape
        kept, reduced = (cols, rows) if input_axis else (rows, cols)
        output = torch.empty_like(base)
        raw_norm = torch.empty_like(magnitude)
        _forward[(kept,)](
            base, delta, magnitude, output, raw_norm, rows, cols, input_axis,
            EPSILON, triton.next_power_of_2(reduced), num_warps=4,
            enable_fp_fusion=False,
        )
        ctx.save_for_backward(base, delta, magnitude, raw_norm)
        ctx.input_axis = input_axis
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        base, delta, magnitude, raw_norm = ctx.saved_tensors
        rows, cols = base.shape
        kept, reduced = (cols, rows) if ctx.input_axis else (rows, cols)
        grad_merged, grad_magnitude = torch.empty_like(base), torch.empty_like(magnitude)
        _backward[(kept,)](
            grad.contiguous(), base, delta, magnitude, raw_norm, grad_merged,
            grad_magnitude, rows, cols, ctx.input_axis, EPSILON,
            triton.next_power_of_2(reduced), num_warps=4, enable_fp_fusion=False,
        )
        return grad_merged, grad_merged, grad_magnitude, None


def standard_dora(base, delta, magnitude, *, axis="input"):
    if axis not in {"input", "output"} or base.ndim != 2:
        raise ValueError("standard linear DoRA requires input/output axis and a 2-D weight")
    expected = (1, base.shape[1]) if axis == "input" else (base.shape[0], 1)
    if delta.shape != base.shape or magnitude.shape != expected:
        raise ValueError("base/delta/magnitude shapes do not match the selected axis")
    if not all(t.device == base.device and t.is_cuda and t.dtype == torch.float32
               and t.is_contiguous() for t in (base, delta, magnitude)):
        raise ValueError("DoRA requires contiguous FP32 tensors on one CUDA device")
    if not all(1 <= size <= 4096 for size in base.shape):
        raise ValueError("DoRA supports weight dimensions 1..4096")
    return StandardDoRA.apply(base, delta, magnitude, axis == "input")
