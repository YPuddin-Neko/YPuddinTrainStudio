"""Optimizer storage and step tensors from parameter shapes, without loading weights."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from importlib import metadata
from typing import Any

import torch

from ypuddin.config import OptimizerConfig
from ypuddin.config.optimizer_rules import optimizer_key


def _factored_elements(shape: tuple[int, ...]) -> int:
    return math.prod(shape[:-1]) + math.prod(shape[:-2] + shape[-1:])


def _bitsandbytes_layout() -> tuple[int, int, bool]:
    """Read allocation constants without importing the optional native backend."""
    try:
        source = metadata.distribution("bitsandbytes").locate_file("bitsandbytes/optim/optimizer.py").read_text()
    except (metadata.PackageNotFoundError, OSError, UnicodeError, AttributeError):
        # A 256-element block has at least as many scale tensors as the older
        # 2048-element blocks. Older releases keep tensors below 4096 in FP32.
        return 256, 4096, False
    current = re.search(r"blocksize\s*=\s*(256|2048)\b", source)
    legacy = re.search(r"blocks\s*=\s*n\s*//\s*(256|2048)\b", source)
    match = current or legacy
    if match is None:
        return 256, 4096, False
    floor = 4096 if re.search(r"dtype\s*==\s*torch.uint8\s+and\s+p.numel\(\)\s*<\s*4096", source) else 0
    return int(match[1]), floor, True


def optimizer_memory_breakdown(
    cfg: OptimizerConfig, param_groups: Sequence[Mapping[str, Any]], *, device_type: str = "cuda",
) -> dict[str, Any]:
    """Count built-in state and bound tensor temporaries for an ordinary dense step.

    Parameter and gradient tensors are supplied separately by the caller. Kahan
    shadows are separate from the inner optimizer state. Initial values are
    unavailable on meta, so Prodigy p0 uses its nonzero-weight storage bound.
    Kernel scratch and allocator reservation are not tensor storage.
    Storage records describe initialized states, not their allocation order.
    CPU offload uses the separate host-memory helpers below.
    """
    cfg = OptimizerConfig.model_validate(cfg.model_dump())
    key = optimizer_key(cfg.type)
    result: dict[str, Any] = {
        "state_bytes": 0, "host_state_bytes": 0, "kahan_bytes": 0,
        "step_workspace_bytes": 0, "state_kind": "exact", "workspace_kind": "tensor_bound",
        "unknowns": [],
        "state_storage_bytes": None, "host_state_storage_bytes": None,
        "kahan_storage_bytes": None, "state_storage_records": None,
        "state_layout_kind": "unknown",
    }
    if cfg.cpu_offload:
        result["state_layout_kind"] = "cpu_offload_separate"
        return result
    supported = {"adam", "adamw", "sgd", "lion", "adamw8bit", "lion8bit", "adafactor",
                 "came", "automagic", "prodigy", "prodigy_plus_sf", "adamw_sf", "muon"}
    if key not in supported:
        return {**result, "state_bytes": None, "step_workspace_bytes": None,
                "state_kind": "unknown", "workspace_kind": "unknown", "unknowns": ["optimizer_class"]}
    if key == "muon" and any("muon_role" not in group for group in param_groups):
        from ypuddin.optim.muon_groups import prepare_muon_groups

        param_groups = prepare_muon_groups(cfg, param_groups)
    if key == "prodigy_plus_sf":
        try:
            installed = metadata.version("prodigy-plus-schedule-free")
        except metadata.PackageNotFoundError:
            installed = ""
        if not re.fullmatch(r"2(?:\.\d+)+(?:[-_.]?(?:a|b|rc|post|dev)\d+)*(?:\+[a-z0-9]+(?:[-_.][a-z0-9]+)*)?", installed, re.I):
            return {**result, "state_bytes": None, "step_workspace_bytes": None,
                    "state_kind": "unknown", "workspace_kind": "unknown", "unknowns": ["ppsf_version"]}
    state = host_state = workspace = kahan = kahan_gradients = 0
    schedulefree_retained = 0
    muon_parameters = 0
    previous = previous_live = 0
    bnb_block, bnb_floor, bnb_known = _bitsandbytes_layout() if key in {"adamw8bit", "lion8bit"} else (256, 0, True)
    nonempty_groups = 0
    nonempty_group_indices = []
    quantized = False
    seen: set[int] = set()
    records: list[dict[str, Any]] = []

    def record(
        role: str, shape: tuple[int, ...], dtype: torch.dtype, *,
        component: str = "state", kind: str = "exact", scope: str = "parameter",
        owner_group: int | None, owner_parameter: int | None,
    ) -> None:
        records.append({
            "component": component, "scope": scope,
            "group_index": owner_group, "parameter_index": owner_parameter,
            "role": role, "bytes": math.prod(shape) * dtype.itemsize,
            "shape": list(shape), "dtype": str(dtype),
            "placement": "host" if component == "host_state" else "device", "kind": kind,
        })

    for group_index, raw_group in enumerate(param_groups):
        group = {**cfg.args, **raw_group}
        if group["params"]:
            nonempty_groups += 1
            nonempty_group_indices.append(group_index)
        # Shared parameter objects own one optimizer state, even if repeated.
        parameters = []
        for parameter_index, p in enumerate(group["params"]):
            if id(p) not in seen:
                seen.add(id(p))
                parameters.append((parameter_index, p))
        if not parameters:
            continue
        sizes = []
        for parameter_index, p in parameters:
            n, shape = p.numel(), tuple(p.shape)
            low_precision = p.dtype in (torch.float16, torch.bfloat16)
            element = 4 if cfg.kahan and low_precision else p.element_size()
            dtype = torch.float32 if cfg.kahan and low_precision else p.dtype
            size = n * element
            sizes.append(size)

            def parameter_record(
                role: str, storage_shape: tuple[int, ...] = shape, storage_dtype: torch.dtype = dtype,
                *, component: str = "state", kind: str = "exact",
                owner_group: int = group_index, owner_parameter: int = parameter_index,
            ) -> None:
                record(role, storage_shape, storage_dtype, component=component, kind=kind,
                       owner_group=owner_group, owner_parameter=owner_parameter)

            if cfg.kahan and low_precision:
                kahan += n * 8
                kahan_gradients += n * (4 - p.element_size())
                parameter_record("shadow", shape, torch.float32, component="kahan")
                parameter_record("compensation", shape, torch.float32, component="kahan")
            if key == "muon":
                muon_parameters += 1
                roles = ("momentum_buffer",) if group["muon_role"] == "muon" else ("exp_avg", "exp_avg_sq")
                for role in roles:
                    state += n * 4
                    parameter_record(role, shape, torch.float32)
                if low_precision:
                    state += n * 4
                    parameter_record("master_param", shape, torch.float32)
                # State is lazy; count every parameter once initialized. Step is
                # a Python integer. _update releases temporaries per parameter.
                if group["muon_role"] == "muon":
                    # FP32 gradient/Nesterov copies, BF16 Newton-Schulz matrix
                    # expressions and Gram polynomials. Transpose the tall case.
                    workspace = max(workspace, n * 16 + min(shape) ** 2 * 8)
                else:
                    # FP32 gradient copy, corrected numerator and denominator.
                    workspace = max(workspace, n * 12)
            elif key in {"adam", "adamw"}:
                state += size * (3 if group.get("amsgrad", False) else 2)
                parameter_record("exp_avg")
                parameter_record("exp_avg_sq")
                if group.get("amsgrad", False):
                    parameter_record("max_exp_avg_sq")
                if device_type == "cpu" or group.get("capturable", False) or group.get("fused", False):
                    state += 4
                    parameter_record("step", (), torch.float32)
                else:
                    host_state += 4
                    parameter_record("step", (), torch.float32, component="host_state")
            elif key == "sgd":
                state += size if group.get("momentum", 0) else 0
                if group.get("momentum", 0):
                    parameter_record("momentum_buffer")
            elif key == "lion":
                state += size
                parameter_record("exp_avg")
                workspace = max(workspace, 2 * size)
            elif key in {"adamw8bit", "lion8bit"}:
                moments = 2 if key == "adamw8bit" else 1
                layout_kind = "exact" if bnb_known else "upper_bound"
                if n < max(bnb_floor, cfg.args.get("min_8bit_size", 4096)):
                    state += n * 4 * moments
                    for index in range(1, moments + 1):
                        parameter_record(f"state{index}", shape, torch.float32, kind=layout_kind)
                else:
                    quantized = True
                    state += n * moments
                    state += ((n + bnb_block - 1) // bnb_block * 4 if cfg.args.get("block_wise", True) else 8) * moments
                    for index in range(1, moments + 1):
                        parameter_record(f"state{index}", shape, torch.uint8, kind=layout_kind)
                        if cfg.args.get("block_wise", True):
                            parameter_record(f"absmax{index}", ((n + bnb_block - 1) // bnb_block,),
                                             torch.float32, kind=layout_kind)
                        else:
                            parameter_record(f"max{index}", (1,), torch.float32, kind=layout_kind)
                            parameter_record(f"new_max{index}", (1,), torch.float32, kind=layout_kind)
                if cfg.args.get("percentile_clipping", 100) < 100:
                    state += 100 * 4
                    parameter_record("gnorm_vec", (100,), torch.float32, kind=layout_kind)
                if cfg.args.get("max_unorm", 0) > 0:
                    state += 4
                    parameter_record("unorm_vec", (1,), torch.float32, kind=layout_kind)
            elif key == "adamw_sf":
                state += 2 * size
                parameter_record("z")
                parameter_record("exp_avg_sq")
            elif key == "adafactor":
                wide = max(4, element)
                factored = len(shape) >= 2
                factors = _factored_elements(shape) if factored else n
                first_moment = group.get("beta1") is not None
                state += (factors + (n if first_moment else 0) + 1) * wide
                wide_dtype = torch.float32 if element < 4 else dtype
                if first_moment:
                    parameter_record("exp_avg", shape, wide_dtype)
                if factored:
                    parameter_record("exp_avg_sq_row", shape[:-1], wide_dtype)
                    parameter_record("exp_avg_sq_col", shape[:-2] + shape[-1:], wide_dtype)
                else:
                    parameter_record("exp_avg_sq", shape, wide_dtype)
                parameter_record("RMS", (), wide_dtype)
                # Squared gradients and their add result overlap. Factored
                # reconstruction also retains the row/column factors.
                cast = 2 * n * wide if element < 4 else 0
                factor_bytes = _factored_elements(shape) * wide if factored else 0
                workspace = max(workspace, previous_live + cast + 2 * n * wide + factor_bytes + 2 * wide)
                previous_live = 0 if first_moment else n * wide
            elif key == "automagic":
                if group.get("lr", cfg.lr) == 0:
                    continue
                factors = _factored_elements(shape) if len(shape) >= 2 else n
                state += n * 5 + factors * 4 + (n * 4 if element < 4 else 0)
                parameter_record("lr_mask", shape, torch.float32)
                parameter_record("last_polarity", shape, torch.bool)
                if len(shape) >= 2:
                    parameter_record("exp_avg_sq_row", shape[:-1], torch.float32)
                    parameter_record("exp_avg_sq_col", shape[:-2] + shape[-1:], torch.float32)
                else:
                    parameter_record("exp_avg_sq", shape, torch.float32)
                if element < 4:
                    parameter_record("master_param", shape, torch.float32)
                cast = n * 4 if element < 4 else 0
                row = math.prod(shape[:-1]) * 4 if len(shape) >= 2 else 0
                # The prior update and polarity survive until reassignment;
                # current squared gradients and factor products overlap.
                workspace = max(workspace, previous_live + cast + 3 * n * 4 + row,
                                cast + 3 * n * 4 + 2 * n + row)
                previous_live = n * 5
            elif key == "came":
                factors = _factored_elements(shape) if len(shape) >= 2 else 0
                state += size + (2 * factors * element if factors else size) + element
                parameter_record("exp_avg")
                if factors:
                    for prefix in ("exp_avg_sq", "exp_avg_res"):
                        parameter_record(f"{prefix}_row", shape[:-1])
                        parameter_record(f"{prefix}_col", shape[:-2] + shape[-1:])
                else:
                    parameter_record("exp_avg_sq")
                parameter_record("RMS", ())
                if group.get("ams_bound", False):
                    state += size
                    parameter_record("max_exp_avg_sq")
                workspace = max(workspace, previous_live + 3 * size + factors * element)
                previous_live = 2 * size
            elif key == "prodigy":
                sample = (n + int(group.get("slice_p", cfg.slice_p)) - 1) // int(group.get("slice_p", cfg.slice_p))
                state += size * (2 if group.get("betas", cfg.betas)[0] else 1) + max(sample, 1) * element + sample * element
                if group.get("betas", cfg.betas)[0]:
                    parameter_record("exp_avg")
                parameter_record("exp_avg_sq")
                parameter_record("s", (sample,))
                parameter_record("p0", (max(sample, 1),), kind="upper_bound")
                workspace = max(workspace, previous + size, sample * element + element)
                previous = size
            elif key == "prodigy_plus_sf":
                sample = (n + 10) // 11
                sample_element = 2 if element == 4 else element
                state += size + max(sample, 1) * sample_element
                parameter_record("z" if group.get("use_schedulefree", cfg.use_schedulefree) else "exp_avg")
                sample_dtype = torch.bfloat16 if element == 4 else dtype
                parameter_record("p0", (max(sample, 1),), sample_dtype, kind="upper_bound")
                if not group.get("use_speed", cfg.use_speed):
                    state += sample * sample_element
                    parameter_record("s", (sample,), sample_dtype)
                factored = (group.get("factored", cfg.factored) and not group.get("use_focus", cfg.use_focus)
                            and len(shape) >= 2 and any(d >= 32 for d in shape))
                factors = 0
                if factored:
                    axes = sorted(range(len(shape)), key=lambda axis: (shape[axis], axis))[-2:]
                    factors = sum(math.prod(d for index, d in enumerate(shape) if index != axis) for axis in axes)
                    state += factors * (4 if group.get("factored_fp32", cfg.factored_fp32) else element)
                    factor_dtype = torch.float32 if group.get("factored_fp32", cfg.factored_fp32) else dtype
                    for role, axis in zip(("exp_avg_sq_row", "exp_avg_sq_col"), reversed(axes), strict=True):
                        factor_shape = tuple(1 if index == axis else d for index, d in enumerate(shape))
                        parameter_record(role, factor_shape, factor_dtype)
                else:
                    state += size
                    parameter_record("exp_avg_sq")
                # FP32 gradient copy and denominator; low-precision weights
                # also keep FP32 y, and z when Schedule-Free is enabled.
                copies = 2 + ((2 if group.get("use_schedulefree", cfg.use_schedulefree) else 1) if element < 4 else 0)
                schedulefree = group.get("use_schedulefree", cfg.use_schedulefree)
                extra = int(not schedulefree and (group.get("use_cautious", cfg.use_cautious) or group.get("use_grams", cfg.use_grams)))
                extra += int(not group.get("weight_decay_by_lr", cfg.weight_decay_by_lr) and bool(group.get("weight_decay", cfg.weight_decay)))
                scalar_work = 32 if group.get("use_orthograd", cfg.use_orthograd) else 0
                workspace = max(workspace, (copies + extra) * n * 4 + factors * 4 + sample * 8 + scalar_work)
        if key in {"adam", "adamw", "sgd", "adamw_sf"}:
            fused = bool(group.get("fused", False))
            foreach = group.get("foreach")
            if foreach is None:
                foreach = key == "adamw_sf" or (device_type in {"cuda", "xpu", "mtia", "privateuseone"} and not fused)
            maximize = bool(group.get("maximize", False))
            coupled = key == "adam" and bool(group.get("weight_decay", cfg.weight_decay)) and not group.get("decoupled_weight_decay", False)
            if fused:
                result["workspace_kind"] = "backend_dependent"
                result["unknowns"].append("fused_optimizer_kernel_scratch")
            elif key == "sgd":
                decay = bool(group.get("weight_decay", cfg.weight_decay))
                nesterov = bool(group.get("nesterov", False))
                copies = int(decay or maximize) + (int(nesterov) if not foreach else 0)
                if maximize and not foreach:
                    # The preceding negated/decayed gradient is still live
                    # while Python evaluates the next negation.
                    copies = max(copies, 2)
                workspace = max(workspace, copies * (sum(sizes) if foreach else max(sizes)))
            elif foreach:
                # Adam releases the optional gradient list before sqrt.
                workspace = max(workspace, sum(sizes) + schedulefree_retained + (4 if maximize else 0))
                if key == "adamw_sf":
                    schedulefree_retained = sum(sizes)
                if group.get("capturable", False):
                    workspace = max(workspace, sum(sizes) + len(sizes) * 32)
            else:
                previous = 0
                for size in sizes:
                    copies = 1 if key == "adamw_sf" else 2 + int(maximize or coupled)
                    workspace = max(workspace, max(previous, schedulefree_retained) + copies * size)
                    if maximize:
                        workspace = max(workspace, 2 * previous + size)
                    previous = size
                if key == "adamw_sf":
                    schedulefree_retained = sizes[-1]
    if key == "muon" and muon_parameters:
        # Finite checks retain one boolean scalar per gradient and a stacked
        # reduction. Include their storage in the conservative tensor bound.
        workspace += 2 * muon_parameters + 1
    if key in {"prodigy", "prodigy_plus_sf"}:
        result["state_kind"] = "upper_bound"
    if key == "prodigy_plus_sf":
        state += (nonempty_groups if cfg.split_groups else min(nonempty_groups, 1)) * 8
        for group_index in nonempty_group_indices if cfg.split_groups else nonempty_group_indices[:1]:
            for role in ("running_d_numerator", "running_d_denom"):
                record(role, (), torch.float32, scope="group", owner_group=group_index, owner_parameter=None)
    if key in {"adamw8bit", "lion8bit"}:
        if quantized:
            state += 256 * 4 * (2 if key == "adamw8bit" else 1)
            for index in range(1, (2 if key == "adamw8bit" else 1) + 1):
                record(f"qmap{index}", (256,), torch.float32, scope="optimizer",
                       kind="exact" if bnb_known else "upper_bound", owner_group=None, owner_parameter=None)
        result["workspace_kind"] = "backend_dependent"
        result["state_kind"] = "exact" if bnb_known else "upper_bound"
        result["unknowns"].append("bitsandbytes_kernel_scratch")
        if not bnb_known:
            result["unknowns"].append("bitsandbytes_state_layout_version")
    if key == "lion" and cfg.args.get("use_triton", False):
        result["workspace_kind"] = "backend_dependent"
        result["unknowns"].append("lion_triton_kernel_scratch")
    if cfg.args.get("differentiable", False):
        result["workspace_kind"] = "unknown"
        result["unknowns"].append("differentiable_optimizer_graph")
    if cfg.kahan and kahan:
        largest = max((p.numel() for g in param_groups for p in g["params"] if p.dtype in (torch.float16, torch.bfloat16)), default=0)
        # Every low-precision gradient is replaced before the inner step.
        # Rounding retains target, low copy, FP32 copy and subtraction output.
        workspace = kahan_gradients + max(workspace, largest * 14)
    result.update(state_bytes=state, host_state_bytes=host_state, kahan_bytes=kahan, step_workspace_bytes=workspace)
    result["state_storage_records"] = records
    result["state_layout_kind"] = "modeled"
    for component in ("state", "host_state", "kahan"):
        result[f"{component}_storage_bytes"] = [r["bytes"] for r in records if r["component"] == component]
    if result["workspace_kind"] == "unknown":
        result["step_workspace_bytes"] = None
    result["unknowns"] = sorted(set(result["unknowns"]))
    return result


def optimizer_state_bytes(cfg: OptimizerConfig, param_groups: Sequence[Mapping[str, Any]]) -> int | None:
    """Persistent device state, excluding parameters, gradients and Kahan shadows."""
    return optimizer_memory_breakdown(cfg, param_groups)["state_bytes"]


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


def cpu_offload_step_workspace_bytes(
    cfg: OptimizerConfig, param_groups: Sequence[Mapping[str, Any]],
) -> int:
    """Peak AdamW CPU temporaries; parameter groups execute sequentially."""
    if not cfg.cpu_offload:
        return 0
    peak = 0
    seen: set[int] = set()
    for group in param_groups:
        sizes = []
        for parameter in group["params"]:
            if id(parameter) not in seen:
                seen.add(id(parameter))
                sizes.append(parameter.numel() * 4)
        if not sizes:
            continue
        maximize = bool(group.get("maximize", cfg.args.get("maximize", False)))
        if group.get("foreach", cfg.args.get("foreach", False)):
            # Negated gradients are released before the square roots. The CPU
            # step increment can coexist with the negated-gradient group.
            peak = max(peak, sum(sizes) + (4 if maximize else 0))
        else:
            # CPU AdamW also takes this path when foreach is left as None.
            # The preceding denominator lives until the new sqrt/divide result
            # is assigned; maximize additionally retains a negated gradient.
            previous = 0
            for size in sizes:
                peak = max(peak, previous + (3 if maximize else 2) * size)
                if maximize:
                    peak = max(peak, 2 * previous + size)
                previous = size
    return peak


def validate_cpu_offload_memory(
    cfg: OptimizerConfig, param_groups: Sequence[Mapping[str, Any]], *, ema_bytes: int = 0,
) -> None:
    """Check live host memory before allocating shadows, including CLI and old queued jobs."""
    if not cfg.cpu_offload:
        return
    import psutil
    import torch.distributed as dist

    from .memory_budget import host_memory_error

    distributed = dist.is_available() and dist.is_initialized()
    world = dist.get_world_size() if distributed else 1
    required_mb = (
        sum(cpu_offload_memory_bytes(cfg, param_groups).values())
        + cpu_offload_step_workspace_bytes(cfg, param_groups)
        + ema_bytes
    ) * world / 2**20
    result = [None]
    if not distributed or dist.get_rank() == 0:
        try:
            result[0] = host_memory_error(required_mb, psutil.virtual_memory().available / 2**20)
        except (OSError, RuntimeError) as error:
            result[0] = {"loc": "memory.host", "msg": f"无法读取可用系统内存，不能检查 CPU 卸载预算：{error}"}
    # All ranks check the same snapshot before any rank starts allocating CPU copies.
    if distributed:
        dist.broadcast_object_list(result, src=0)
    if result[0] is not None:
        raise RuntimeError(result[0]["msg"])
