"""Adapter-specific FSDP storage and portable export helpers."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from ypuddin.adapters.dora import DoRA
from ypuddin.adapters.frozen import FrozenLinear
from ypuddin.adapters.inject import ALGOS, AdapterSet


def shardable_frozen_weights(model):
    """Promote frozen linear buffers to non-trainable parameters, without casting.

    FSDP shards parameters, never buffers. Keeping the original dtype avoids
    silently expanding a BF16 base model to FP32 just to shard it.
    """
    if any(isinstance(module, FrozenLinear) and module.is_fp8 for module in model.modules()):
        raise ValueError("适配器显存分片暂不支持 FP8 底模，请使用 BF16、FP16 或 FP32 底模")
    supported = {torch.float32, torch.float16, torch.bfloat16}
    if any(p.dtype not in supported for p in model.parameters()):
        raise ValueError("适配器显存分片仅支持 BF16、FP16 或 FP32 参数")
    for module in model.modules():
        if isinstance(module, FrozenLinear):
            for name in ("weight", "bias"):
                value = module._buffers.get(name)
                if value is not None:
                    del module._buffers[name]
                    module.register_parameter(name, nn.Parameter(value, requires_grad=False))


def adapter_modules(adapters: AdapterSet):
    """Only small adapter states belong in a resumable adapter checkpoint."""
    return {
        f"{name}.{kind}": module
        for name, layer in adapters.layers.items()
        for kind, module in (("adapter", layer.adapter), ("dora", layer.dora))
        if module is not None
    }


def gathered_adapter_export(adapters: AdapterSet):
    """All ranks gather one adapter at a time; fold scales on detached CPU copies.

    Export code may multiply matrices or convert scalar tensors to Python values;
    running it directly on sharded DTensors would change those semantics.
    """
    tensors = {}
    for name, layer in adapters.layers.items():
        prefix = adapters.export_key(name)
        for kind, module in (("adapter", layer.adapter), ("dora", layer.dora)):
            if module is None:
                continue
            # FSDP's dynamic class cannot be copy.copy()'d: its __new__ calls
            # the original constructor. Export needs only the plain algorithm.
            local = object.__new__(ALGOS[module.kind] if kind == "adapter" else DoRA)
            local.__dict__ = module.__dict__.copy()
            local._modules = {}
            local._parameters = {key: None for key in module._parameters}
            local._buffers = {key: None for key in module._buffers}
            for key, value in module.named_parameters(recurse=False):
                tensor = value.full_tensor() if isinstance(value, DTensor) else value
                local._parameters[key] = nn.Parameter(tensor.detach().cpu().clone(), requires_grad=False)
            for key, value in module.named_buffers(recurse=False):
                tensor = value.full_tensor() if isinstance(value, DTensor) else value
                local._buffers[key] = tensor.detach().cpu().clone()
            values = local.export_tensors() if kind == "adapter" else {"dora_scale": local.export_tensor()}
            tensors.update({f"{prefix}.{key}": value for key, value in values.items()})
    return tensors
