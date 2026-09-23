"""Versioned BF16/FP16 Linear preview contractions, scoped to backbone prediction.

The normal Linear forward still determines native mm/bmm/addmm dispatch and
rounds autocast operands. Only the selected half-precision contractions use FP32
math, then return the original half dtype. Adapter factors, text encoders, VAE, training and validation
are outside this context.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from ypuddin.adapters.frozen import FrozenLinear
from ypuddin.config.compute_policy import (
    BF16_LINEAR_PREVIEW_IMPLEMENTATION_ID,
    FP16_LINEAR_PREVIEW_IMPLEMENTATION_ID,
)

_enabled = ContextVar("ypuddin_preview_linear_compute", default=None)
_preview_backbones = ContextVar("ypuddin_preview_preview_backbones", default=())


@contextmanager
def preview_linear_compute(policy, backbone=None):
    implementation = (policy or {}).get("preview_linear_implementation")
    if implementation == FP16_LINEAR_PREVIEW_IMPLEMENTATION_ID:
        if (
            policy.get("preview_operator_components") != ["backbone"]
            or policy.get("preview_linear_forward") != "fp16-rounded-operands-fp32-contraction-fp16-output"
            or backbone is None
        ):
            raise ValueError("预览 FP16 Linear 计算策略不完整")
        token = _enabled.set(implementation)
        try:
            with _preview_hooks(backbone, implementation):
                yield
        finally:
            _enabled.reset(token)
        return
    if implementation is not None and (
        implementation != BF16_LINEAR_PREVIEW_IMPLEMENTATION_ID
        or policy.get("preview_operator_components") != ["backbone"]
        or policy.get("preview_linear_forward") != "bf16-rounded-operands-fp32-contraction-bf16-output"
    ):
        raise ValueError("预览 Linear 计算策略不完整或版本不受支持")
    token = _enabled.set(implementation)
    try:
        if (
            implementation == BF16_LINEAR_PREVIEW_IMPLEMENTATION_ID
            and "linear_backward_implementation" not in policy
        ):
            if backbone is None:
                raise ValueError("预览 BF16 Linear 计算策略需要主模型")
            with _preview_hooks(backbone, implementation):
                yield
        else:
            yield
    finally:
        _enabled.reset(token)


class _BF16PreviewContractions(TorchDispatchMode):
    operand_dtype = torch.bfloat16
    implementation = BF16_LINEAR_PREVIEW_IMPLEMENTATION_ID

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        operands = [arg for arg in args if isinstance(arg, torch.Tensor)]
        if (
            _enabled.get() == self.implementation
            and not torch.is_grad_enabled()
            and str(func) in {"aten.mm.default", "aten.bmm.default", "aten.addmm.default"}
            and operands
            and all(arg.dtype == self.operand_dtype for arg in operands)
        ):
            with torch.autocast(operands[0].device.type, enabled=False):
                result = func(
                    *(arg.float() if isinstance(arg, torch.Tensor) else arg for arg in args), **kwargs
                )
            return result.to(self.operand_dtype)
        return func(*args, **kwargs)


class _FP16PreviewContractions(_BF16PreviewContractions):
    operand_dtype = torch.float16

    implementation = FP16_LINEAR_PREVIEW_IMPLEMENTATION_ID


@contextmanager
def _preview_hooks(backbone, implementation):
    """Scope half-precision math to base Linear calls in this preview only.

    Preview-only policies attach temporary hooks because they do not install
    or alter any training operator.
    """
    active = _preview_backbones.get()
    key = (id(backbone), implementation)
    if torch.is_grad_enabled() or key in active:
        yield
        return
    handles, entered = [], []
    token = _preview_backbones.set((*active, key))

    def before(_module, _args):
        mode = (
            _FP16PreviewContractions
            if implementation == FP16_LINEAR_PREVIEW_IMPLEMENTATION_ID
            else _BF16PreviewContractions
        )()
        mode.__enter__()
        entered.append(mode)

    def after(_module, _args, _result):
        entered.pop().__exit__(None, None, None)

    try:
        for module in backbone.modules():
            if isinstance(module, (nn.Linear, FrozenLinear)):
                handles.append(module.register_forward_pre_hook(before))
                handles.append(module.register_forward_hook(after, always_call=True))
        if not handles:
            raise ValueError("预览 Linear 策略没有找到主模型线性层")
        yield
    finally:
        for handle in handles:
            handle.remove()
        while entered:
            entered.pop().__exit__(None, None, None)
        _preview_backbones.reset(token)


def linear_preview_forward(original, x):
    if _enabled.get() != BF16_LINEAR_PREVIEW_IMPLEMENTATION_ID:
        return original(x)
    with _BF16PreviewContractions():
        return original(x)
