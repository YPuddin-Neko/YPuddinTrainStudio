"""Native BF16 Linear forward with FP32 backward contractions.

Keep forward operand rounding, layout and dispatch unchanged. Return gradients
in their original operand dtype, including BF16 FSDP all-gather parameters;
FP32 reduce-scatter does not remove that BF16 gradient rounding boundary.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable

from ypuddin.adapters.frozen import FrozenLinear
from ypuddin.config.compute_policy import BF16_LINEAR_BACKWARD_IMPLEMENTATION_ID

_MARKER = "_ypuddin_bf16_linear_backward"


class _BF16LinearFP32Backward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, forward_grad_enabled):
        ctx.device_type = x.device.type
        ctx.input_dtype = x.dtype
        ctx.weight_dtype = weight.dtype
        ctx.bias_dtype = None if bias is None else bias.dtype
        # Save actual forward operands, not the original FP32 values that were
        # rounded away by autocast. The Function still takes FP32 master params.
        with torch.autocast(device_type=x.device.type, enabled=False):
            effective_x = x.to(torch.bfloat16)
            effective_weight = weight.to(torch.bfloat16)
            effective_bias = None if bias is None else bias.to(torch.bfloat16)
            # Autograd.Function.forward runs under no_grad. Preserve the
            # metadata of autocast-created operands: matmul uses requires_grad
            # to choose folded mm versus broadcast bmm for noncontiguous inputs.
            # Changing that would make this a forward-dispatch experiment too.
            for source, effective in [(x, effective_x), (weight, effective_weight), (bias, effective_bias)]:
                if source is not None and effective is not source:
                    effective.requires_grad_(source.requires_grad and forward_grad_enabled)
            out = F.linear(effective_x, effective_weight, effective_bias)
        ctx.save_for_backward(effective_x, effective_weight)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        dx = dw = db = None
        with torch.autocast(device_type=ctx.device_type, enabled=False):
            dy = grad_output.float()
            if ctx.needs_input_grad[0]:
                dx = (dy @ weight.float()).to(ctx.input_dtype)
            if ctx.needs_input_grad[1]:
                dw = dy.reshape(-1, weight.shape[0]).T @ x.reshape(-1, weight.shape[1]).float()
                dw = dw.to(ctx.weight_dtype)
            if ctx.bias_dtype is not None and ctx.needs_input_grad[2]:
                db = dy.reshape(-1, weight.shape[0]).sum(dim=0).to(ctx.bias_dtype)
        return dx, dw, db, None


def _uses_bf16(x, weight):
    if torch.is_autocast_enabled(x.device.type):
        return torch.get_autocast_dtype(x.device.type) == torch.bfloat16
    return x.dtype == weight.dtype == torch.bfloat16


def install_linear_bf16_forward_fp32_backward(backbone):
    """Return (restore, counts). Patch only Linear/FrozenLinear instances below backbone."""
    records = []
    counts = {"nn_linear": 0, "frozen_linear": 0}
    modules = [module for module in backbone.modules() if isinstance(module, (nn.Linear, FrozenLinear))]
    if not modules:
        raise ValueError("BF16 Linear backward policy found no Linear modules")
    if any(_MARKER in module.__dict__ for module in modules):
        raise ValueError("BF16 Linear backward policy is already installed")

    def make_forward(original, frozen):
        def forward(module, x):
            # Evaluation has no backward to change. Preserve its exact native
            # dispatch, including autocast caching/grad-mode special cases.
            if not torch.is_grad_enabled():
                return original(x)
            weight = module.dequant(x.dtype) if frozen else module.weight
            bias = module.bias
            if frozen and bias is not None:
                bias = bias.to(x.dtype)
            if not _uses_bf16(x, weight):
                return original(x)
            return _BF16LinearFP32Backward.apply(x, weight, bias, torch.is_grad_enabled())

        return forward

    for module in modules:
        frozen = isinstance(module, FrozenLinear)
        had_instance_forward = "forward" in module.__dict__
        previous_instance_forward = module.__dict__.get("forward")
        original = module.forward
        replacement = types.MethodType(make_forward(original, frozen), module)
        module.forward = replacement
        module.__dict__[_MARKER] = (BF16_LINEAR_BACKWARD_IMPLEMENTATION_ID, replacement)
        records.append((module, replacement, had_instance_forward, previous_instance_forward))
        counts["frozen_linear" if frozen else "nn_linear"] += 1

    def restore():
        for module, replacement, existed, previous in records:
            if module.__dict__.get("forward") is replacement:
                if existed:
                    module.forward = previous
                else:
                    module.__dict__.pop("forward", None)
            marker = module.__dict__.get(_MARKER)
            if marker is not None and marker[1] is replacement:
                module.__dict__.pop(_MARKER)

    return restore, counts


def validate_linear_backward_installation(backbone, expected_counts):
    """Verify real instance forwards, without touching parameters or FSDP hooks."""
    counts = {"nn_linear": 0, "frozen_linear": 0}
    for module in backbone.modules():
        frozen = isinstance(module, FrozenLinear)
        if not frozen and not isinstance(module, nn.Linear):
            continue
        marker = module.__dict__.get(_MARKER)
        if (
            marker is None
            or marker[0] != BF16_LINEAR_BACKWARD_IMPLEMENTATION_ID
            or module.__dict__.get("forward") is not marker[1]
        ):
            raise ValueError("BF16 Linear backward policy is missing or its forward was replaced")
        counts["frozen_linear" if frozen else "nn_linear"] += 1
    if counts != expected_counts or not sum(counts.values()):
        raise ValueError("BF16 Linear backward installation does not match the selected backbone")
    return counts


def validate_lokr_bypass_backbone(backbone):
    """Check resolved adapters, including rule overrides and already-frozen weights."""
    from ypuddin.adapters.linear import AdaptedLinear
    from ypuddin.adapters.lokr import LoKr

    count = 0
    for module in backbone.modules():
        if isinstance(module, FrozenLinear) and module.is_fp8:
            raise ValueError("SDXL BF16 算子策略尚不支持 FP8 冻结权重")
        if not isinstance(module, AdaptedLinear):
            continue
        if (
            type(module.adapter) is not LoKr
            or module.mode != "bypass"
            or module.dora is not None
            or any(parameter.dtype != torch.float32 for parameter in module.adapter.parameters())
        ):
            raise ValueError("SDXL BF16 算子策略需要实际 LoKr bypass、无 DoRA、FP32 适配器参数")
        count += 1
    if not count:
        raise ValueError("SDXL BF16 算子策略未找到实际 LoKr 适配器")
    return count
