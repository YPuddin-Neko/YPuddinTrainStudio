"""Shape-based storage bounds for adapter weight reconstruction and its backward pass."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict

import torch
from torch import nn

from ypuddin.adapters.linear import AdaptedLayer


class ModeMemory(TypedDict):
    retained_bytes: int
    workspace_bytes: int


class AdapterMemory(TypedDict):
    none: ModeMemory
    block: ModeMemory
    unsloth: ModeMemory
    unsupported: list[str]


@dataclass
class _LayerMemory:
    retained: int = 0
    workspace: int = 0
    cache: int = 0
    saved_cache: int = 0


def _size(dtype: torch.dtype) -> int:
    return dtype.itemsize


def _layer_memory(
    layer: AdaptedLayer, activation_dtype: torch.dtype, autocast_enabled: bool
) -> tuple[_LayerMemory, str | None]:
    adapter = layer.adapter
    if layer.mode != "merged" or layer.multiplier == 0:
        # The family's activation estimate covers the bypass path's token-dependent tensors.
        return _LayerMemory(), None
    n = layer.weight.numel()
    op_bytes = _size(activation_dtype)
    params = dict(adapter.named_parameters(recurse=False))
    param_dtype = next(iter(params.values())).dtype
    param_bytes = _size(param_dtype)
    amp = autocast_enabled and activation_dtype in (torch.float16, torch.bfloat16)
    mm_dtype = activation_dtype if amp else param_dtype
    dora = layer.dora
    exported = dora is not None and dora.compute_mode == "comfyui"
    merge_dtype = dora.merge_dtype if exported else torch.float32
    merge_bytes = _size(merge_dtype)
    result = _LayerMemory(retained=n * op_bytes)
    unsupported = None

    def factor(name: str, *, copied: bool = False, matmul: bool = True) -> None:
        parameter = params[name]
        dtype = merge_dtype if exported else mm_dtype if matmul else parameter.dtype
        # Export sometimes multiplies a factor by the scalar 1 before casting; it is still a new tensor.
        converted = parameter.dtype != dtype or (
            exported and parameter.dtype != dora.save_dtype
        )
        byte_count = parameter.numel() * _size(dtype)
        if copied or converted:
            result.retained += byte_count
        if not exported and amp and matmul and parameter.dtype == torch.float32 and not copied:
            result.cache += byte_count
            result.saved_cache += byte_count

    masked = bool(adapter.rank_dropout_p)
    scalar = adapter.scalar is not None
    if adapter.kind == "lora":
        factor("down")
        factor("up", copied=exported or masked)
        delta_dtype = merge_dtype if exported else mm_dtype
    elif adapter.kind == "lokr":
        factor_dtypes = []
        for part in ("w1", "w2"):
            lowrank = getattr(adapter, f"{part}_lowrank")
            if lowrank:
                factor(f"{part}_a", copied=(exported and part == "w1") or (masked and part == "w2"))
                factor(f"{part}_b")
                count = (adapter.a * adapter.c) if part == "w1" else (adapter.b * adapter.d)
                if part == "w2":
                    count *= adapter.fan_in // adapter.in_features
                dtype = merge_dtype if exported else mm_dtype
                result.retained += count * _size(dtype)  # The reconstructed factors saved by kron.
            else:
                factor(part, copied=exported and part == "w1", matmul=False)
                dtype = merge_dtype if exported else params[part].dtype
            factor_dtypes.append(dtype)
        delta_dtype = torch.promote_types(*factor_dtypes)
    elif adapter.kind == "loha":
        if exported:
            for name in ("w1_a", "w1_b", "w2_a", "w2_b"):
                factor(name, copied=name == "w1_a")
            # Export reconstruction uses two ordinary matmuls, then a Hadamard product.
            result.retained += 2 * n * merge_bytes
        else:
            # _HadaWeight saves the original factors and recomputes both products in backward.
            if masked:
                result.retained += params["w1_a"].numel() * param_bytes
            if amp and param_dtype == torch.float32:
                result.cache += sum(p.numel() for p in params.values()) * _size(mm_dtype)
        delta_dtype = merge_dtype if exported else mm_dtype
    elif adapter.kind == "full" and dora is None:
        delta_dtype = param_dtype
    else:
        unsupported = f"{layer.name or adapter.kind}: {adapter.kind} merged adapter"
        # Unknown reconstruction keeps a conservative dense budget and reports missing coverage.
        result.retained += sum(p.numel() * p.element_size() for p in params.values()) + 2 * n * 4
        delta_dtype = param_dtype

    if masked:
        rank = getattr(adapter, "rank", None)
        if rank is not None:
            result.retained += rank * param_bytes
    if scalar:
        # Standard _scaled needs its unscaled delta for the scalar gradient; export folds the
        # scalar into one factor instead, whose original parameter storage is already counted.
        result.retained += 0 if exported else n * _size(delta_dtype)
        result.retained += max(param_bytes, merge_bytes)
        if exported and masked and adapter.kind in ("lora", "loha"):
            # Scalar folding also saves the preceding rank-masked factor in parameter precision.
            name = "up" if adapter.kind == "lora" else "w1_a"
            result.retained += params[name].numel() * param_bytes

    if dora is not None:
        result.retained += n * merge_bytes  # merged, shared by normalization and rescaling backward.
        channels = dora.dora_scale.numel()
        # Norm, norm+epsilon and ratio are new; the magnitude is a parameter unless cast for export.
        result.retained += 3 * channels * merge_bytes
        magnitude = dora.dora_scale
        if magnitude.dtype != merge_dtype or (exported and magnitude.dtype != dora.save_dtype):
            result.retained += channels * merge_bytes

    # Forward: delta, merged, and (for DoRA) rescaled weight coexist before the final cast.
    forward_dense = (3 if dora is not None else 2) * n * merge_bytes
    if activation_dtype != merge_dtype:
        forward_dense += n * op_bytes
    saved_dense = n * op_bytes + (n * merge_bytes if dora is not None else 0)
    result.workspace = max(0, forward_dense - saved_dense)
    # A floating-point base cast exists only in the exported path; FP8 dequantization has its
    # own planner budget. Standard mode promotes the addition without making a base copy.
    if exported and not getattr(layer.base, "is_fp8", False) and layer.weight.dtype != merge_dtype:
        result.workspace += n * merge_bytes

    if dora is not None:
        # Rescaling/norm backward can retain incoming grad, direct weight grad and the reduction
        # product together. The saved merged weight remains; the linear's effective weight is freed.
        result.workspace = max(result.workspace, n * max(0, 3 * merge_bytes - op_bytes))
        result.workspace += 4 * dora.dora_scale.numel() * merge_bytes
    if adapter.kind == "loha" and not exported:
        # _HadaWeight.backward holds p1, p2, g1, g2 together with incoming grad_out.
        backward_dense = n * (4 * param_bytes + _size(delta_dtype))
        # All four parameter gradients are returned together, before accumulation releases them.
        backward_dense += sum(p.numel() * p.element_size() for p in params.values())
        result.workspace = max(result.workspace, backward_dense - n * op_bytes)
    elif adapter.kind == "lokr":
        # kron backward can hold grad_delta and both broadcast products before sum-to-size;
        # the first factor's reduced gradient can coexist with the second full-size product.
        factor_elements = adapter.a * adapter.c
        backward_dense = (3 * n + factor_elements) * _size(delta_dtype)
        result.workspace = max(result.workspace, backward_dense - n * op_bytes)
    # A large-rank factor gradient can exceed a dense weight. Its incoming gradient and cast to
    # parameter precision coexist before accumulation; existing .grad storage is counted elsewhere.
    gradient_bytes = _size(merge_dtype if exported else delta_dtype)
    factor_gradient_cast = max(
        (p.numel() * (p.element_size() + gradient_bytes) for p in params.values()), default=0
    )
    result.workspace = max(result.workspace, factor_gradient_cast)
    if unsupported:
        result.workspace = max(result.workspace, 4 * n * 4)
    return result, unsupported


def adapter_memory(
    backbone: nn.Module,
    blocks: Sequence[nn.Module],
    activation_dtype: torch.dtype,
    *,
    autocast_enabled: bool = True,
) -> AdapterMemory:
    """Additional adapter storage per device, using full layer shapes even with FSDP.

    Existing parameters/base weights, FP8 dequantization and bypass activations are excluded.
    Checkpoint modes retain adapters outside the checkpointed blocks plus one active block.
    Autocast factor caches can instead span every block until the original forward exits.
    Workspace is an operation-lifetime bound, not CUDA/HIP allocator or kernel workspace.
    """
    owners = {id(module): index for index, block in enumerate(blocks) for module in block.modules()}
    grouped: dict[int | None, list[_LayerMemory]] = {None: []}
    unsupported = []
    for module in backbone.modules():
        if not isinstance(module, AdaptedLayer):
            continue
        memory, issue = _layer_memory(module, activation_dtype, autocast_enabled)
        grouped.setdefault(owners.get(id(module)), []).append(memory)
        if issue:
            unsupported.append(issue)
    outside = grouped.pop(None)
    all_layers = outside + [layer for layers in grouped.values() for layer in layers]

    def saved(layers: list[_LayerMemory]) -> int:
        return sum(layer.retained for layer in layers)

    def saved_and_cache(layers: list[_LayerMemory]) -> int:
        return sum(layer.retained + layer.cache - layer.saved_cache for layer in layers)

    active_block = max((saved_and_cache(layers) for layers in grouped.values()), default=0)
    forward_cache = sum(layer.cache for layers in grouped.values() for layer in layers)
    checkpointed = max(saved(outside) + active_block, saved_and_cache(outside) + forward_cache)
    workspace = max((layer.workspace for layer in all_layers), default=0)
    return {
        "none": {"retained_bytes": saved_and_cache(all_layers), "workspace_bytes": workspace},
        "block": {"retained_bytes": checkpointed, "workspace_bytes": workspace},
        "unsloth": {"retained_bytes": checkpointed, "workspace_bytes": workspace},
        "unsupported": unsupported,
    }
