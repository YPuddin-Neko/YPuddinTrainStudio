"""Scoped frozen Qwen3 linear contractions for the versioned DTK 9B recipe."""

from contextlib import contextmanager

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from ypuddin.config.compute_policy import KLEIN9B_TEXT_COMPUTE_ID


class _LinearContractions(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        operands = [arg for arg in args if isinstance(arg, torch.Tensor)]
        if (
            str(func) in {"aten.mm.default", "aten.bmm.default", "aten.addmm.default"}
            and operands
            and all(arg.dtype == torch.bfloat16 for arg in operands)
        ):
            with torch.autocast(operands[0].device.type, enabled=False):
                result = func(
                    *(arg.float() if isinstance(arg, torch.Tensor) else arg for arg in args), **kwargs
                )
            return result.to(torch.bfloat16)
        return func(*args, **kwargs)


@contextmanager
def frozen_text_compute(model, implementation):
    if implementation is None:
        yield
        return
    if implementation != KLEIN9B_TEXT_COMPUTE_ID:
        raise ValueError("Unsupported frozen Qwen3 compute policy")
    if torch.is_grad_enabled() or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Qwen3 cache compute policy requires a frozen no-grad encoder")
    handles, entered = [], []

    def before(_module, _args):
        mode = _LinearContractions()
        mode.__enter__()
        entered.append(mode)

    def after(_module, _args, _result):
        entered.pop().__exit__(None, None, None)

    try:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                handles.append(module.register_forward_pre_hook(before))
                handles.append(module.register_forward_hook(after, always_call=True))
        if not handles:
            raise ValueError("Qwen3 cache compute policy found no linear layers")
        yield
    finally:
        for handle in handles:
            handle.remove()
        while entered:
            entered.pop().__exit__(None, None, None)
