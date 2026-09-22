"""Versioned BF16 Linear preview contractions, scoped to backbone prediction.

The normal Linear forward still determines native mm/bmm/addmm dispatch and
rounds autocast operands. Only its dispatched BF16 contractions use FP32 math,
then return BF16. Adapter factors, text encoders, VAE, training and validation
are outside this context.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from ypuddin.config.compute_policy import BF16_LINEAR_PREVIEW_IMPLEMENTATION_ID

_enabled = ContextVar("ypuddin_preview_linear_compute", default=False)


@contextmanager
def preview_linear_compute(policy):
    implementation = (policy or {}).get("preview_linear_implementation")
    if implementation is not None and (
        implementation != BF16_LINEAR_PREVIEW_IMPLEMENTATION_ID
        or policy.get("preview_operator_components") != ["backbone"]
        or policy.get("preview_linear_forward") != "bf16-rounded-operands-fp32-contraction-bf16-output"
    ):
        raise ValueError("预览 Linear 计算策略不完整或版本不受支持")
    token = _enabled.set(implementation is not None)
    try:
        yield
    finally:
        _enabled.reset(token)


class _BF16PreviewContractions(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        operands = [arg for arg in args if isinstance(arg, torch.Tensor)]
        if (
            not torch.is_grad_enabled()
            and str(func) in {"aten.mm.default", "aten.bmm.default", "aten.addmm.default"}
            and operands
            and all(arg.dtype == torch.bfloat16 for arg in operands)
        ):
            with torch.autocast(operands[0].device.type, enabled=False):
                result = func(
                    *(arg.float() if isinstance(arg, torch.Tensor) else arg for arg in args), **kwargs
                )
            return result.bfloat16()
        return func(*args, **kwargs)


def linear_preview_forward(original, x):
    if not _enabled.get():
        return original(x)
    with _BF16PreviewContractions():
        return original(x)
