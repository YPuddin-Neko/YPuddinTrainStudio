"""Versioned FP16-rounded backbone/LoRA contractions for DTK single-GPU training.

AMP, GradScaler, parameter storage and all intermediate dtype boundaries are
preserved. No-grad evaluation retains its native implementation.
"""

import types

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable

from ypuddin.adapters.frozen import FrozenLinear
from ypuddin.adapters.linear import AdaptedLinear
from ypuddin.adapters.lora import LoRA
from ypuddin.config.compute_policy import FP16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID

_MARKER = "_ypuddin_fp16_adapter_compute"


class _FP16OperandsFP32Compute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.input_dtype, ctx.weight_dtype = x.dtype, weight.dtype
        ctx.bias_dtype = None if bias is None else bias.dtype
        ctx.device_type = x.device.type
        with torch.autocast(x.device.type, enabled=False):
            a, b = x.half(), weight.half()
            c = None if bias is None else bias.half()
            # Preserve original grad metadata for noncontiguous matmul dispatch.
            af, bf = a.float(), b.float()
            cf = None if c is None else c.float()
            for src, dst in [(x, af), (weight, bf), (bias, cf)]:
                if src is not None:
                    dst.requires_grad_(src.requires_grad)
            out = F.linear(af, bf, cf).half()
        ctx.save_for_backward(a, b)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        a, b = ctx.saved_tensors
        dx = dw = db = None
        with torch.autocast(ctx.device_type, enabled=False):
            g = gradient.float()
            if ctx.needs_input_grad[0]:
                dx = (g @ b.float()).to(ctx.input_dtype)
            if ctx.needs_input_grad[1]:
                dw = (g.reshape(-1, b.shape[0]).T @ a.reshape(-1, b.shape[1]).float()).to(ctx.weight_dtype)
            if ctx.bias_dtype is not None and ctx.needs_input_grad[2]:
                db = g.reshape(-1, b.shape[0]).sum(0).to(ctx.bias_dtype)
        return dx, dw, db


def _uses_fp16(x, weight):
    if torch.is_autocast_enabled(x.device.type):
        return torch.get_autocast_dtype(x.device.type) == torch.float16
    return x.dtype == weight.dtype == torch.float16


def install_fp16_adapter_compute(backbone):
    records = []
    counts = {"base_modules": 0, "lora_modules": 0}

    def bind_base(original, frozen):
        def forward(module, x):
            if not torch.is_grad_enabled():
                return original(x)
            weight = module.dequant(x.dtype) if frozen else module.weight
            bias = module.bias
            if frozen and bias is not None:
                bias = bias.to(x.dtype)
            if not _uses_fp16(x, weight):
                return original(x)
            return _FP16OperandsFP32Compute.apply(x, weight, bias)

        return forward

    def bind_lora(original):
        def delta_apply(adapter, x):
            if not torch.is_grad_enabled():
                return original(x)
            dt = x.dtype
            down = adapter.down.to(dt)
            if not _uses_fp16(x, down):
                return original(x)
            h = _FP16OperandsFP32Compute.apply(x, down, None)
            mask = adapter._rank_mask(adapter.rank, x.device, dt)
            if mask is not None:
                h = h * mask
            y = _FP16OperandsFP32Compute.apply(h, adapter.up.to(dt), None)
            gain = adapter._gain()
            if isinstance(gain, torch.Tensor):
                gain = gain.to(dt)
            return adapter._output_dropout(y * gain)

        return delta_apply

    def patch(module, name, fn):
        old = module.__dict__.get(name)
        existed = name in module.__dict__
        replacement = types.MethodType(fn, module)
        setattr(module, name, replacement)
        module.__dict__[_MARKER] = (FP16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID, name, replacement)
        records.append((module, name, replacement, existed, old))

    modules = _validated_modules(backbone)
    if any(_MARKER in module.__dict__ for module in modules):
        raise ValueError("FP16 算子计算策略不能重复安装")
    for module in backbone.modules():
        if isinstance(module, (nn.Linear, FrozenLinear)):
            patch(module, "forward", bind_base(module.forward, isinstance(module, FrozenLinear)))
            counts["base_modules"] += 1
        if isinstance(module, AdaptedLinear):
            patch(module.adapter, "delta_apply", bind_lora(module.adapter.delta_apply))
            counts["lora_modules"] += 1

    def restore():
        for module, name, replacement, existed, old in reversed(records):
            if module.__dict__.get(name) is replacement:
                if existed:
                    setattr(module, name, old)
                else:
                    module.__dict__.pop(name, None)
            if module.__dict__.get(_MARKER) == (
                FP16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID,
                name,
                replacement,
            ):
                module.__dict__.pop(_MARKER)

    return restore, counts


def _validated_modules(backbone):
    modules = []
    bases = adapters = 0
    for module in backbone.modules():
        if isinstance(module, FrozenLinear) and module.is_fp8:
            raise ValueError("FP16 可复现策略不支持 FP8 冻结权重")
        if isinstance(module, (nn.Linear, FrozenLinear)):
            modules.append(module)
            bases += 1
        if isinstance(module, AdaptedLinear):
            if (
                type(module.adapter) is not LoRA
                or module.mode != "bypass"
                or module.dora is not None
                or any(p.dtype != torch.float32 for p in module.adapter.parameters())
            ):
                raise ValueError("FP16 可复现策略需要 LoRA bypass、FP32 适配器参数且不使用 DoRA")
            modules.append(module.adapter)
            adapters += 1
    if not bases or not adapters:
        raise ValueError("FP16 可复现策略未找到完整的主模型和 LoRA")
    return modules


def validate_fp16_adapter_compute(backbone, expected_counts):
    counts = {"base_modules": 0, "lora_modules": 0}
    for module in _validated_modules(backbone):
        name = "delta_apply" if type(module) is LoRA else "forward"
        marker = module.__dict__.get(_MARKER)
        if (
            not marker
            or marker[:2] != (FP16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID, name)
            or module.__dict__.get(name) is not marker[2]
        ):
            raise ValueError("FP16 计算策略未完整安装或算子已被替换")
        counts["lora_modules" if name == "delta_apply" else "base_modules"] += 1
    if counts != expected_counts:
        raise ValueError("FP16 计算策略的实际模块数量不一致")
    return counts
