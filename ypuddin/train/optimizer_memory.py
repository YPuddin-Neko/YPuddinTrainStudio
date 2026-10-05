"""Persistent optimizer tensor storage from parameter shapes, without loading weights."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from importlib import metadata
from typing import Any

import torch

from ypuddin.config import OptimizerConfig
from ypuddin.config.optimizer_rules import optimizer_key


def optimizer_state_bytes(cfg: OptimizerConfig, param_groups: Sequence[Mapping[str, Any]]) -> int | None:
    """Estimate PPSF 2.x persistent state; return None for an unrecognized optimizer/version.

    Groups are the ones passed to ``build_optimizer`` and may contain meta parameters.
    Initial values are unknown, so p0 reserves the full sample even for zero-initialized weights.
    Parameter and gradient storage, and temporary optimizer-step workspaces, are excluded.
    """
    if getattr(cfg, "cpu_offload", False):
        return 0
    if optimizer_key(cfg.type) != "prodigy_plus_sf":
        return None
    try:
        installed = metadata.version("prodigy-plus-schedule-free")
    except metadata.PackageNotFoundError:
        return None
    if not re.fullmatch(r"2(?:\.\d+)+(?:[-_.]?(?:a|b|rc|post|dev)\d+)*(?:\+[a-z0-9]+(?:[-_.][a-z0-9]+)*)?", installed, re.I):
        return None

    # Match factory validation, including legacy options edited in cfg.args in place.
    cfg = OptimizerConfig.model_validate(cfg.model_dump())
    total = 0
    groups = 0
    seen: set[int] = set()
    for group in param_groups:
        parameters = group["params"]
        if not parameters:
            continue
        groups += 1
        factored = group.get("factored", cfg.factored) and not group.get("use_focus", cfg.use_focus)
        factored_fp32 = group.get("factored_fp32", cfg.factored_fp32)
        use_speed = group.get("use_speed", cfg.use_speed)
        for parameter in parameters:
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            shape = tuple(parameter.shape)
            count = math.prod(shape)
            element_bytes = parameter.element_size()
            sample_bytes = 2 if parameter.dtype == torch.float32 else element_bytes
            # PPSF 2.x samples every eleventh element; ordinary Prodigy's slice_p does not apply.
            sampled = (count + 10) // 11
            total += count * element_bytes  # z (Schedule-Free) or exp_avg
            total += max(sampled, 1) * sample_bytes  # p0 is at least a scalar
            if not use_speed:
                total += sampled * sample_bytes
            if factored and len(shape) >= 2 and any(size >= 32 for size in shape):
                axes = sorted(range(len(shape)), key=lambda axis: (shape[axis], axis))[-2:]
                second_moment = sum(math.prod(size for axis, size in enumerate(shape) if axis != reduced) for reduced in axes)
                total += second_moment * (4 if factored_fp32 else element_bytes)
            else:
                total += count * element_bytes
    # The constructor allocates two FP32 running statistics per group, or one shared pair.
    return total + (groups if cfg.split_groups else min(groups, 1)) * 8


def cpu_offload_memory_bytes(cfg: OptimizerConfig, param_groups: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """CPU tensor storage for AdamW offload; excludes allocator and step workspaces."""
    shadows = moments = gradients = 0
    seen: set[int] = set()
    if getattr(cfg, "cpu_offload", False):
        for group in param_groups:
            amsgrad = group.get("amsgrad", cfg.args.get("amsgrad", False))
            for parameter in group["params"]:
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                nbytes = parameter.numel() * 4
                shadows += nbytes
                gradients += nbytes
                moments += nbytes * (3 if amsgrad else 2) + 4
    return {"parameter_bytes": shadows, "state_bytes": moments, "gradient_bytes": gradients}
