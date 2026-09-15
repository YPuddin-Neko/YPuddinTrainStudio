"""Optimizer admission and native Adafactor state initialization for FSDP2.

Transformers Adafactor creates its factored states with ``torch.zeros(shape)``.
Those are ordinary tensors even when its parameter is a DTensor.  Initialize
only these states before its unchanged step so updates retain the original
algorithm and the global parameter shape.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard


def _parameters(optimizer):
    return [parameter for group in optimizer.param_groups for parameter in group["params"]]


def _check_parameter(parameter: torch.Tensor) -> None:
    if not isinstance(parameter, DTensor):
        raise ValueError("FSDP2 optimizer requires DTensor parameters; build it after sharding")
    if parameter.device_mesh.ndim != 1 or not isinstance(parameter.placements[0], Shard):
        raise ValueError("FSDP2 optimizer currently requires one-dimensional parameter sharding")
    if parameter.ndim == 0 or parameter.dtype != torch.float32:
        raise ValueError("FSDP2 optimizer requires non-scalar FP32 master parameters")
    if parameter.shape[parameter.placements[0].dim] < parameter.device_mesh.size():
        raise ValueError("FSDP2 optimizer requires nonempty shards; select another parameter shard dimension")


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(math.prod(shape[index + 1 :]) for index in range(len(shape)))


def _factored_zero(gradient: DTensor, reduced_dim: int) -> DTensor:
    """Make the reduced state without allocating or gathering the full gradient."""
    dim = reduced_dim % gradient.ndim
    shard_dim = gradient.placements[0].dim
    shape = tuple(size for index, size in enumerate(gradient.shape) if index != dim)
    if dim == shard_dim:
        placements = (Replicate(),)
        local_shape = shape
    else:
        placements = (Shard(shard_dim - int(dim < shard_dim)),)
        local_shape = tuple(size for index, size in enumerate(gradient.to_local().shape) if index != dim)
    local = torch.zeros(local_shape, device=gradient.to_local().device, dtype=torch.float32)
    return DTensor.from_local(
        local,
        device_mesh=gradient.device_mesh,
        placements=placements,
        run_check=False,
        shape=torch.Size(shape),
        stride=_contiguous_stride(shape),
    )


def _check_state(value: Any, expected: DTensor, name: str) -> None:
    if (
        not isinstance(value, DTensor)
        or value.shape != expected.shape
        or value.dtype != torch.float32
        or value.device_mesh != expected.device_mesh
        or tuple(value.placements) != tuple(expected.placements)
    ):
        raise ValueError(f"Adafactor checkpoint state {name!r} has incompatible FSDP2 layout")


def _prepare_adafactor_step(optimizer, args, kwargs):
    closure = args[1] if len(args) > 1 else kwargs.get("closure")
    if closure is not None:
        raise ValueError("FSDP2 Adafactor does not support optimizer closures")
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if not isinstance(parameter, DTensor):
                # The trainer reduces these explicitly replicated small parameters.
                # Leave their ordinary Adafactor state and update entirely native.
                continue
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            if not isinstance(gradient, DTensor) or gradient.is_sparse:
                raise ValueError("FSDP2 Adafactor requires dense DTensor gradients")
            _check_parameter(parameter)
            if gradient.shape != parameter.shape or gradient.placements != parameter.placements:
                raise ValueError("FSDP2 Adafactor gradient layout differs from its parameter")
            state = optimizer.state[parameter]
            expected = {}
            if group["beta1"] is not None:
                expected["exp_avg"] = gradient
            if gradient.ndim >= 2:
                expected["exp_avg_sq_row"] = _factored_zero(gradient, -1)
                expected["exp_avg_sq_col"] = _factored_zero(gradient, -2)
            else:
                expected["exp_avg_sq"] = gradient
            if not state:
                state.update(step=0, RMS=0)
                for name, template in expected.items():
                    state[name] = torch.zeros_like(template, dtype=torch.float32)
            else:
                if "step" not in state or "RMS" not in state:
                    raise ValueError("Adafactor checkpoint state is incomplete")
                for name, template in expected.items():
                    _check_state(state.get(name), template, name)


def prepare_sharded_optimizer(optimizer):
    """Validate a supported optimizer and install the native Adafactor state hook.

    Call after FSDP2 placement and optimizer creation, including after loading a
    checkpoint into a newly created optimizer.  Repeated calls are idempotent.
    This function does not replace the optimizer or change its hyperparameters.
    FP32 parameters smaller than the mesh may remain ordinary replicated tensors;
    the caller must synchronize their gradients before stepping the optimizer.
    """
    from transformers.optimization import Adafactor

    if type(optimizer) not in {Adafactor, torch.optim.SGD, torch.optim.AdamW}:
        raise ValueError("FSDP2 currently supports native SGD, AdamW and Transformers Adafactor")
    parameters = _parameters(optimizer)
    if not parameters:
        raise ValueError("FSDP2 optimizer has no parameters")
    sharded = [parameter for parameter in parameters if isinstance(parameter, DTensor)]
    if not sharded:
        raise ValueError("FSDP2 optimizer requires DTensor parameters; build it after sharding")
    mesh = sharded[0].device_mesh
    for parameter in parameters:
        if isinstance(parameter, DTensor):
            _check_parameter(parameter)
            if parameter.device_mesh != mesh:
                raise ValueError("FSDP2 optimizer parameters must use the same device mesh")
        elif parameter.dtype != torch.float32 or parameter.numel() >= mesh.size():
            raise ValueError("Only FP32 parameters smaller than the FSDP2 mesh may remain replicated")
    if isinstance(optimizer, Adafactor) and not hasattr(optimizer, "_ypuddin_fsdp2_state_hook"):
        optimizer._ypuddin_fsdp2_state_hook = optimizer.register_step_pre_hook(_prepare_adafactor_step)
    return optimizer


def sharded_optimizer_state_bytes(optimizer) -> dict[str, Any]:
    """Report local stored state bytes without materializing global DTensors."""
    records = []
    total = 0
    for parameter_index, parameter in enumerate(_parameters(optimizer)):
        for name, value in optimizer.state.get(parameter, {}).items():
            if not isinstance(value, torch.Tensor):
                continue
            local = value.to_local() if isinstance(value, DTensor) else value
            nbytes = local.numel() * local.element_size()
            total += nbytes
            records.append(
                {
                    "parameter_index": parameter_index,
                    "name": name,
                    "global_shape": list(value.shape),
                    "local_shape": list(local.shape),
                    "placements": [str(p) for p in value.placements] if isinstance(value, DTensor) else [],
                    "local_bytes": nbytes,
                }
            )
    return {"local_state_bytes": total, "states": records}
