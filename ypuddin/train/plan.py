"""``plan``: validate a config and predict steps / buckets / parameter counts / VRAM without loading weights."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pydantic import TypeAdapter, ValidationError
from torch import nn

from ypuddin.adapters import inject
from ypuddin.adapters.frozen import FrozenLinear
from ypuddin.config import DatasetConfig, LoopConfig, ModelConfig, TrainConfig, ValidationConfig
from ypuddin.config.compute_policy import resolve_training_compute_config, validate_resume_compute_policy
from ypuddin.config.issues import validation_issues
from ypuddin.data import BucketBatchSampler, IndexDB
from ypuddin.data.dataset import DataConfigError, DataLayout, item_geometry, prepare_data_layout
from ypuddin.data.native import NativeBatchSampler, microbatch_indices, native_size
from ypuddin.models import get_family
from ypuddin.models.base import LatentSpec
from ypuddin.runtime_compile import CompileEnvironmentError, validate_compile_environment
from ypuddin.runtime_profiles import current_profile

from .activation_plan import backbone_memory_layout, estimate_backbone_memory
from .adapter_memory import adapter_memory
from .advice import value_advice
from .evaluation_memory import evaluation_snapshot_memory
from .evaluation_plan import evaluation_mode_stages, evaluation_operations, project_evaluation_stages
from .gradient_memory import gradient_workspace_bytes
from .memory_budget import HEADROOM
from .metal_compute import resolve_metal_attention_runtime, validate_metal_attention_resume
from .native_vram import NativeVramGeometry, maximum_fitting_pixels
from .optimizer_memory import optimizer_memory_breakdown
from .preview_memory import (
    cached_text_phase_capacity,
    preview_residency,
    preview_working_memory,
    text_capacity_complete,
    text_phase_capacity,
)
from .text_memory import frozen_text_phase
from .text_plan_inputs import text_plan_inputs
from .vae_memory import (
    VaeMemory,
    latent_cache_phase,
    online_encoding_phase,
    preview_decoding_phase,
    preview_shapes,
    vae_memory,
)

DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4, "fp8_e4m3": 1, "fp8_e5m2": 1, "keep": 2, "auto": 2}


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _frozen_storage_bytes(module: nn.Module) -> int:
    tensors = {id(t): t for t in (*module.parameters(), *module.buffers()) if not t.requires_grad}
    return sum(t.numel() * t.element_size() for t in tensors.values())


def _adapter_ema_bytes(adapters) -> int:
    """EMA stores FP32 copies of exported factors, including frozen export components."""
    from ypuddin.adapters.ortho import OrthoLoRA

    elements = 0
    with torch.no_grad(), torch.device("meta"):
        for layer in adapters.layers.values():
            adapter = layer.adapter
            if isinstance(adapter, OrthoLoRA):
                # The basis is bound only after real weights load; its plain LoRA export
                # has two rank-r factors and alpha, independently of those values.
                elements += adapter.rank * (adapter.out_features + adapter.fan_in) + 1
            else:
                elements += sum(t.numel() for t in adapter.export_tensors().values())
            if layer.dora is not None:
                elements += layer.dora.export_tensor().numel()
    return elements * 4


def _training_peak_mb(live_mb: float) -> float:
    # The allocator keeps reusable blocks in addition to live tensors. This is a
    # workspace allowance, separate from the device headroom used at admission.
    return live_mb + max(512.0, live_mb * 0.05)


def _resident_vae_phase(phase: dict[str, Any], resident_mb: float, *, capacity_host_mb: float = 0.0) -> dict[str, Any]:
    """Project one VAE invocation beside the tensors that coexist with it."""
    result = dict(phase)
    result["resident_mb_estimate"] = round(resident_mb, 1)
    result["allocated_peak_mb_estimate"] = round(resident_mb + phase["workspace_mb_estimate"])
    result["peak_mb_estimate"] = round(resident_mb + phase["reserved_workspace_mb_estimate"])
    result["capacity_peak_mb_estimate"] = round(resident_mb + capacity_host_mb + phase["capacity_workspace_mb_estimate"])
    return result


def _combined_vae_phase(calls: list[tuple[dict, float, float]]) -> dict | None:
    if not calls:
        return None
    projected = [_resident_vae_phase(phase, resident, capacity_host_mb=host) for phase, resident, host in calls]
    result = dict(max(projected, key=lambda phase: phase["allocated_peak_mb_estimate"]))
    for key in ("allocated_peak_mb_estimate", "peak_mb_estimate", "capacity_peak_mb_estimate"):
        result[key] = max(phase[key] for phase in projected)
    return result


def _plan_vae(cfg: TrainConfig, dtype: torch.dtype, device: str | torch.device | None) -> VaeMemory | None:
    """The run's image VAE; a geometry that cannot be read leaves the VAE phases out of the estimate."""
    try:
        return vae_memory(cfg, dtype, device=device, profile=current_profile())
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return None


def _vae_batches(
    image_shapes: dict[tuple[int, int], dict[str, Any]], ds: DatasetConfig, encode_batch: int,
    *, online_kind: str = "combined",
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    """Images per VAE call by shape: while caching, and while training or validating online.

    Caching groups up to ``encode_batch`` images of one shape, each flip a separate entry. Online
    encoding splits a training batch, native microbatch group or validation batch into calls of
    at most ``encode_batch`` images.
    """
    area_native = ds.resolution_mode == "native" and ds.native_batching_version in (1, 3)
    cache, online = {}, {}
    for (width, height), entry in image_shapes.items():
        entries = len(entry["images"]) * (2 if ds.flip else 1) + len(entry["validation"])
        cache[width, height] = min(encode_batch, entries)
        group = min(ds.batch_size, max(1, ds.native_max_pixels // (width * height))) if area_native else ds.batch_size
        validation_items = entry.get("validation_items", len(entry["validation"]))
        items = (entry["items"] if online_kind == "training" else validation_items
                 if online_kind == "validation" else max(entry["items"], validation_items))
        if items:
            online[width, height] = min(encode_batch, group, items)
    return cache, online


def _encoding_vae_phase(vae, batches, *, cached):
    encode = latent_cache_phase if cached else online_encoding_phase
    phase = encode(vae, batches)
    single = encode(vae, dict.fromkeys(batches, 1))
    if phase is not None and single is not None:
        for prefix in ("", "capacity_", "reserved_"):
            key = prefix + "workspace_mb_estimate"
            if single[key] < phase[key]:
                phase[prefix + "workspace_reductions_mb"]["memory.vae_encode_batch_size"] = single[key]
    return phase


def _fsdp_memory(backbone: nn.Module, blocks: list[nn.Module], cfg: TrainConfig) -> dict[str, Any]:
    """Account for actual per-rank FP32 shards and native optimizer-state layouts.

    Only persistent tensor sizes are exact. All-gather, reduce-scatter and
    optimizer temporaries are conservative workspace estimates, not measured
    allocator peaks. Use the training placement/group functions so odd shapes,
    single-row projections and shared/nested modules cannot drift from runtime.
    """
    from ypuddin.config.optimizer_rules import optimizer_key

    from .sharded import parameter_shard, sharding_groups

    world = cfg.loop.gpu_count
    key = optimizer_key(cfg.optimizer.type)
    if key not in {"adafactor", "adamw", "sgd"}:
        raise ValueError("显存分片请选择 AdamW、Adafactor 或 SGD")
    parameter_bytes = [0] * world
    trainable_bytes = [0] * world
    state_bytes = [0] * world
    largest_parameter_bytes = [0] * world
    local_elements: dict[int, list[int]] = {}
    replicated_elements = 0
    for parameter in backbone.parameters():
        shape = tuple(parameter.shape)
        axis = None if parameter.numel() < world else parameter_shard(parameter, world).dim
        if axis is None:
            replicated_elements += parameter.numel()
        sizes = []
        for rank in range(world):
            local_shape = list(shape)
            if axis is not None:
                chunk = math.ceil(shape[axis] / world)
                local_shape[axis] = max(0, min(chunk, shape[axis] - rank * chunk))
            elements = math.prod(local_shape)
            sizes.append(elements)
            parameter_bytes[rank] += elements * parameter.element_size()
            if not parameter.requires_grad:
                continue
            trainable_bytes[rank] += elements * parameter.element_size()
            largest_parameter_bytes[rank] = max(largest_parameter_bytes[rank], elements * 4)
            if key == "adafactor":
                if len(shape) >= 2:
                    # A reduction across the sharded dimension is replicated;
                    # reductions on any other dimension retain the local shard.
                    factors = 0
                    for reduced in (len(shape) - 1, len(shape) - 2):
                        source = shape if axis == reduced else local_shape
                        factors += math.prod(size for dim, size in enumerate(source) if dim != reduced)
                    state_bytes[rank] += factors * 4
                else:
                    state_bytes[rank] += elements * 4
                if cfg.optimizer.args.get("beta1") is not None:
                    state_bytes[rank] += elements * 4
                state_bytes[rank] += 4  # RMS is a replicated FP32 scalar after the first step.
            elif key == "adamw":
                state_bytes[rank] += elements * 4 * (3 if cfg.optimizer.args.get("amsgrad") else 2)
                if cfg.optimizer.args.get("capturable") or cfg.optimizer.args.get("fused"):
                    state_bytes[rank] += 4  # Otherwise the step scalar stays on CPU.
            elif cfg.optimizer.args.get("momentum", 0):
                state_bytes[rank] += elements * 4
        local_elements[id(parameter)] = sizes

    claimed = set()
    groups = []
    for name, module in [*sharding_groups(backbone, blocks), ("", backbone)]:
        members = []
        for parameter in module.parameters():
            if id(parameter) in claimed:
                continue
            claimed.add(id(parameter))
            if parameter.numel() >= world:
                members.append(parameter)
        if members:
            groups.append(
                {
                    "name": name or "backbone",
                    "parameter_count": sum(parameter.numel() for parameter in members),
                    "trainable_count": sum(p.numel() for p in members if p.requires_grad),
                    "storage_bytes": sum(p.numel() * p.element_size() for p in members),
                    "local_parameter_bytes_by_rank": [
                        sum(
                            local_elements[id(parameter)][rank] * parameter.element_size()
                            for parameter in members
                        )
                        for rank in range(world)
                    ],
                }
            )
    largest_groups = sorted((group["parameter_count"] for group in groups), reverse=True)
    largest_trainable_group = max((g["trainable_count"] for g in groups), default=0)
    largest_storage_group = max((g["storage_bytes"] for g in groups), default=0)
    compute_bytes = DTYPE_BYTES["fp32" if cfg.loop.mixed_precision == "no" else cfg.loop.mixed_precision]
    # Current and prefetched group gathers may coexist; backward also needs a
    # full FP32 reduction input and a local reduction output. Activations and
    # frozen text/VAE cache phases are accounted separately and are NOT divided.
    communication_bytes = sum(largest_groups[:2]) * compute_bytes + largest_trainable_group * 4
    communication_bytes += max((max(group["local_parameter_bytes_by_rank"]) for group in groups), default=0)
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in backbone.buffers())
    if key == "adafactor":
        optimizer_workspace = 2 * max(largest_parameter_bytes, default=0)
        state_layout = (
            "factored_with_first_moment" if cfg.optimizer.args.get("beta1") is not None else "factored"
        )
    else:
        # foreach=None permits automatic selection; reserve a tensor-list-sized
        # temporary unless the user explicitly disables it.
        optimizer_workspace = (
            max(trainable_bytes)
            if cfg.optimizer.args.get("foreach") is not False
            else max(largest_parameter_bytes, default=0)
        )
        state_layout = "adamw_amsgrad" if key == "adamw" and cfg.optimizer.args.get("amsgrad") else key
    return {
        "world_size": world,
        "optimizer": key,
        "optimizer_state_layout": state_layout,
        "global_parameter_bytes": sum(p.numel() * p.element_size() for p in backbone.parameters()),
        "local_trainable_bytes_by_rank": trainable_bytes,
        "local_parameter_bytes_by_rank": parameter_bytes,
        "optimizer_state_bytes_by_rank": state_bytes,
        "replicated_parameter_count": replicated_elements,
        "replicated_buffer_bytes": buffer_bytes,
        "communication_bytes_estimate": communication_bytes,
        "optimizer_workspace_bytes_estimate": optimizer_workspace,
        "initialization_bytes_estimate": max(parameter_bytes) + largest_storage_group + buffer_bytes,
        "groups": groups,
    }


@dataclass(frozen=True)
class _LayoutInputs:
    """Only the validated fields consumed by prepare_data_layout, never a training config."""

    dataset: DatasetConfig
    validation: ValidationConfig
    model: ModelConfig


def _append_data_plan(
    out: dict[str, Any],
    cfg: TrainConfig | _LayoutInputs,
    latent: LatentSpec,
    *,
    loop: LoopConfig | None,
    seed: int | None,
    index_db_path: str | Path | None,
    image_shapes: dict[tuple[int, int], dict[str, Any]] | None = None,
    prepared_layout: DataLayout | None = None,
    layout_result: list[DataLayout] | None = None,
) -> dict[tuple[int, int], int]:
    """Share exact image selection and geometry between full plans and incomplete drafts."""
    ds = cfg.dataset
    records = []
    items = []
    validation_images = 0
    index = None
    layout = None
    try:
        index = IndexDB(index_db_path) if index_db_path and prepared_layout is None else None
        layout = prepared_layout or prepare_data_layout(
            cfg, latent, index_db=index, allow_unresolved_native_vram=True,
        )
        if layout_result is not None:
            layout_result.append(layout)
        records, items = layout.records, layout.items
        validation_images = len(layout.validation_items)
        if image_shapes is not None:
            # What the VAE encodes: training items and images per shape, then validation images.
            for item, validation in [(item, False) for item in items] + [
                (item, True) for item in layout.validation_items
            ]:
                entry = image_shapes.setdefault(item.bucket.key, {
                    "items": 0, "images": set(), "validation": set(), "validation_items": 0,
                })
                entry["validation" if validation else "images"].add(item.record.path)
                entry["items"] += not validation
                entry["validation_items"] += validation
    except DataConfigError as e:
        out["errors"].append({"loc": e.loc, "msg": str(e)})
    except (OSError, ValueError) as e:
        out["errors"].append({"loc": "dataset.sources", "msg": str(e)})
    finally:
        if index is not None:
            index.close()
    if layout is None and (isinstance(cfg, _LayoutInputs) or (
        ds.resolution_mode == "native" and ds.native_max_pixels_mode in {"auto", "auto_vram"}
    )):
        return {}
    max_pixels = layout.native_max_pixels if layout and layout.native_max_pixels is not None else ds.native_max_pixels
    if layout is not None:
        # Count the shared training items after validation exclusion, before the
        # sampler distributes batches across ranks. Never use registry totals.
        source_paths: dict[int, set[str]] = {}
        source_items: dict[int, int] = {}
        for item in items:
            source_index = item.record.source_index
            source_paths.setdefault(source_index, set()).add(item.record.path)
            source_items[source_index] = source_items.get(source_index, 0) + 1
        out["source_balance"] = []
        for index, source in enumerate(ds.sources):
            images = len(source_paths.get(index, ()))
            out["source_balance"].append(
                {
                    "source_index": index,
                    "path": source.path,
                    "is_reg": source.is_reg,
                    "images": images,
                    "repeats": source.repeats,
                    "repeated_images": images * source.repeats,
                    "resolution_variants": 1
                    if ds.resolution_mode == "native"
                    else len(source.resolutions or ds.resolutions),
                    "items": source_items.get(index, 0),
                }
            )
    counts: dict[tuple[int, int], int] = {}
    # Batching groups items by shape; the preview also keeps the base resolution.
    shapes: dict[tuple[int, int, int], int] = {}
    for it in items:
        counts[it.bucket.key] = counts.get(it.bucket.key, 0) + 1
        shape = (it.bucket.base, *it.bucket.key)
        shapes[shape] = shapes.get(shape, 0) + 1
    native = ds.resolution_mode == "native"
    native_modern = native and ds.native_batching_version in (2, 3)
    area_native = native and ds.native_batching_version in (1, 3)
    batches = (
        math.ceil(len(items) / ds.batch_size)
        if native and not native_modern
        else sum(math.ceil(n / ds.batch_size) for n in counts.values())
    )
    out.update(
        {
            "images": len(records),
            "items": len(items),
            "captioned": sum(1 for r in records if r.caption_path),
            # Loss masks come from a .mask.png sidecar, else transparent pixels; unused unless masked loss is on.
            # A fully opaque alpha channel weights the whole image, as no mask does.
            "masks": {
                "enabled": ds.masked_loss,
                "files": sum(1 for r in records if r.mask_path),
                "alpha": sum(1 for r in records if not r.mask_path and r.has_transparency),
            },
            "validation_images": validation_images,
            "buckets": [
                {"base": base, "w": w, "h": h, "items": n, "batches": math.ceil(n / ds.batch_size)}
                for (base, w, h), n in sorted(shapes.items())
            ],
        }
    )
    if loop is not None:
        per_rank_batches = math.ceil(batches / loop.gpu_count) if native_modern else batches // loop.gpu_count
        steps_per_epoch = math.ceil(per_rank_batches / loop.grad_accum) if per_rank_batches else 0
        by_epochs = (loop.epochs or 10**9) * steps_per_epoch
        total_steps = min(by_epochs, loop.max_steps or 10**9) if steps_per_epoch else 0
        out.update(steps_per_epoch=steps_per_epoch, total_steps=total_steps, epochs=loop.epochs)
        out["distributed"] = {
            "world_size": loop.gpu_count,
            "strategy": loop.distributed_strategy if loop.gpu_count > 1 else "single",
            "parameter_storage": "sharded" if loop.distributed_strategy == "fsdp" else "replicated",
            "gradient_storage": "sharded" if loop.distributed_strategy == "fsdp" else "replicated",
            "optimizer_storage": "sharded" if loop.distributed_strategy == "fsdp" else "replicated",
            "per_device_batch_size": ds.batch_size,
            "effective_batch_size": ds.batch_size * loop.grad_accum * loop.gpu_count,
            "batches_per_rank": per_rank_batches,
            "tail_policy": ("empty_rank_slot" if native_modern else "drop_incomplete_rank_group")
            if loop.gpu_count > 1 else "keep",
            "dropped_samples": 0,
        }
        if loop.gpu_count > 1 and native_modern:
            if batches % loop.gpu_count and loop.distributed_strategy == "fsdp":
                out["errors"].append({
                    "loc": "loop.distributed_strategy",
                    "msg": "原生分辨率批次数不能被卡数整除时无法使用 FSDP；请改用 DDP 或单卡训练。",
                })
        if loop.gpu_count > 1 and not native_modern:
            sampler_type = NativeBatchSampler if native else BucketBatchSampler
            sampler_options = {"batching_version": ds.native_batching_version} if native else {}
            batch_plan = sampler_type(
                [item.bucket.key for item in items], ds.batch_size, seed=seed or 0, **sampler_options,
            ).plan()
            usable = len(batch_plan) - len(batch_plan) % loop.gpu_count
            dropped = sum(map(len, batch_plan[usable:]))
            out["distributed"]["dropped_samples"] = dropped
            if dropped:
                out["warnings"].append(
                    {
                        "code": "distributed.tail",
                        "msg": f"首轮有 {dropped} 张训练项因不足各卡同时处理一个批次而略过；每轮重新打乱，不复制图片补齐。",
                    }
                )
            if per_rank_batches == 0:
                out["errors"].append(
                    {"loc": "loop.gpu_count", "msg": "训练批次数少于卡数，请增加图片或降低每卡批量"}
                )
    geometry = {}
    # How each training size is reached: the source size and its resize, before the crop or padding.
    routes: dict[tuple[int, int, int], dict[tuple[int, int, int, int], set[str]]] = {}
    padded_images, cropped_images = set(), set()
    padding_pixels = total_pixels = 0
    for item in items:
        key = (item.record.path, item.bucket.key, item.max_scale)
        if key not in geometry:
            geometry[key] = item_geometry(item)
        entry = geometry[key]
        route = (
            entry["source_width"],
            entry["source_height"],
            entry["resized_width"],
            entry["resized_height"],
        )
        routes.setdefault((item.bucket.base, *item.bucket.key), {}).setdefault(route, set()).add(
            item.record.path
        )
        total_pixels += item.bucket.area
        padding_pixels += entry["padding_pixels"]
        if entry["padding_pixels"]:
            padded_images.add(item.record.path)
        if entry["cropped_pixels"]:
            cropped_images.add(item.record.path)
    out["image_fit"] = {
        "mode": ds.image_fit,
        "crop_anchor": ds.crop_anchor,
        "padded_images": len(padded_images),
        "cropped_images": len(cropped_images),
        "padding_pixels": padding_pixels,
        "total_pixels": total_pixels,
        "padding_fraction": padding_pixels / total_pixels if total_pixels else 0.0,
        "total_shapes": len(geometry),
        "truncated": len(geometry) > 100,
        "items": list(geometry.values())[:100],
    }
    for bucket in out["buckets"]:
        found = sorted(
            routes.get((bucket["base"], bucket["w"], bucket["h"]), {}).items(),
            key=lambda pair: (-len(pair[1]), pair[0]),
        )
        bucket["sources"] = [
            {"width": sw, "height": sh, "resized_width": rw, "resized_height": rh, "images": len(paths)}
            for (sw, sh, rw, rh), paths in found[:4]
        ]
        bucket["source_variants"] = len(found)
    if padding_pixels:
        out["warnings"].append(
            {
                "code": "images.padding",
                "msg": f"{len(padded_images)} images preserve the complete frame with padding "
                f"({padding_pixels / total_pixels:.1%} of training canvas pixels); padding is excluded from "
                "direct loss but remains visible context. Native mode or wider aspect buckets can reduce it.",
            }
        )
    if native:
        shape_keys = [item.bucket.key for item in items]
        forward_counts: dict[tuple[int, int], int] = {}
        synchronization_groups = 0
        if seed is not None:
            native_plan = NativeBatchSampler(
                shape_keys, ds.batch_size, seed=seed, batching_version=ds.native_batching_version,
            ).plan()
            if not area_native:
                for batch in native_plan:
                    key = shape_keys[batch[0]]
                    forward_counts[key] = forward_counts.get(key, 0) + 1
                if loop is not None and loop.gpu_count > 1:
                    synchronization_groups = (-len(native_plan)) % loop.gpu_count
            else:
                if loop is not None and loop.gpu_count > 1 and not native_modern:
                    native_plan = native_plan[: len(native_plan) - len(native_plan) % loop.gpu_count]
                native_groups = []
                for batch in native_plan:
                    shapes = [shape_keys[index] for index in batch]
                    groups = microbatch_indices(shapes, max_pixels, batching_version=ds.native_batching_version)
                    native_groups.append([(shapes[group[0]], len(group)) for group in groups])
                    for group in groups:
                        key = shapes[group[0]]
                        forward_counts[key] = forward_counts.get(key, 0) + 1
                if loop is not None and loop.gpu_count > 1:
                    for offset in range(0, len(native_groups), loop.gpu_count):
                        ranks = native_groups[offset : offset + loop.gpu_count]
                        slots = max(map(len, ranks))
                        if native_modern:
                            ranks.extend([] for _ in range(loop.gpu_count - len(ranks)))
                        for groups in ranks:
                            padding = slots - len(groups)
                            if groups and (not native_modern or loop.distributed_strategy == "fsdp"):
                                key, _ = min(groups, key=lambda entry: math.prod(entry[0]) * entry[1])
                                forward_counts[key] += padding
                            synchronization_groups += padding
        for bucket in out["buckets"]:
            bucket["batches"] = (
                forward_counts.get((bucket["w"], bucket["h"]), 0) if seed is not None else None
            )
        resized = (
            sum(
                native_size(
                    record.width,
                    record.height,
                    align=latent.align,
                    max_pixels=max_pixels,
                    max_side=ds.native_max_side,
                    overflow=ds.native_overflow,
                    image_fit=ds.image_fit,
                    auto_area=ds.native_max_pixels_mode in {"auto", "auto_vram"},
                ).downscaled
                for record in records
            )
            if items
            else 0
        )
        out["native"] = {
            "batching_version": ds.native_batching_version,
            "batch_mode": "area" if area_native else "physical",
            "microbatch_max_pixels": max_pixels if area_native else None,
            "images": len(records),
            "downscaled": resized,
            "sizes": len(counts),
            "logical_batches": batches,
            "max_pixels": max_pixels,
            "max_pixels_mode": ds.native_max_pixels_mode,
            "auto_max_pixels": layout.native_auto_max_pixels if layout else None,
            "auto_vram_max_pixels": None,
            "auto_vram_error": "请完善训练参数后计算显存面积上限。",
            "alignment": latent.align,
            "batch_size": ds.batch_size,
            "forward_groups": sum(forward_counts.values()) if seed is not None else None,
            "synchronization_groups": synchronization_groups,
        }
        if ds.native_batching_version == 3:
            execution_code = "native.execution.area"
            execution_message = (
                "Native resolution keeps same-size logical batches and splits their forward passes by the microbatch "
                "pixel budget; gradients are accumulated over the real images."
            )
        elif native_modern:
            execution_code = "native.execution"
            execution_message = (
                "Native resolution groups images with identical width and height into batches up to batch_size; "
                "the image area limit applies to each image."
            )
        else:
            execution_code = "native.execution.v1"
            execution_message = (
                "native resolution uses pixel-bounded shape groups and image-weighted gradient accumulation; "
                "pixel budget is not a VRAM guarantee"
            )
        out["warnings"].append({"code": execution_code, "msg": execution_message})
    uncaptioned = len(records) - out["captioned"]
    if uncaptioned and not any(s.class_prompt for s in ds.sources):
        out["warnings"].append(
            {
                "code": "captions.missing",
                "msg": f"{uncaptioned} images have no caption file and no class_prompt",
            }
        )
    if not native and any(n < ds.batch_size for n in counts.values()):
        out["warnings"].append(
            {
                "code": "buckets.small",
                "msg": "some buckets have fewer images than batch_size (tail batches will be smaller)",
            }
        )

    return counts


def _preview_invalid_config(
    raw: dict[str, Any], out: dict[str, Any], *, index_db_path: str | Path | None,
    image_shapes: dict[tuple[int, int], dict[str, Any]] | None = None,
    layout_result: list[DataLayout] | None = None,
) -> None:
    """Add data-only results without repairing the draft or weakening its training errors."""
    model = raw.get("model")
    family_name = model.get("family") if isinstance(model, dict) else None
    if not isinstance(family_name, str) or not family_name:
        out["errors"].append({"loc": "model.family", "msg": "model.family is required for a data preview"})
        return
    try:
        family = get_family(family_name)
    except KeyError as error:
        out["errors"].append({"loc": "model.family", "msg": str(error)})
        return
    fields: dict[str, Any] = {}
    for name, schema in (("dataset", DatasetConfig), ("validation", ValidationConfig)):
        try:
            fields[name] = schema.model_validate(raw.get(name, {}))
        except ValidationError as error:
            for issue in validation_issues(error, prefix=[name]):
                if issue not in out["errors"]:
                    out["errors"].append(issue)
    if len(fields) != 2:
        return
    dataset = fields["dataset"]
    if dataset.resolution_mode == "native" and dataset.native_batching_version != 1:
        dataset.native_batching_version = 3 if dataset.native_batch_mode == "area" else 2
    validation = fields["validation"]
    if validation.enabled and validation.split_ratio == 0 and not validation.sources:
        out["errors"].append(
            {"loc": "validation", "msg": "validation.enabled requires split_ratio > 0 or explicit sources"}
        )
        return
    try:
        loop = LoopConfig.model_validate(raw.get("loop", {}))
    except ValidationError:
        # Step estimates require a valid stop condition and accumulation count.
        loop = None
    try:
        loop_raw = raw.get("loop", {})
        seed = TypeAdapter(int).validate_python(
            loop_raw.get("seed", LoopConfig.model_fields["seed"].default)
            if isinstance(loop_raw, dict)
            else None
        )
    except ValidationError:
        seed = None
    _append_data_plan(
        out,
        _LayoutInputs(**fields, model=ModelConfig(family=family_name)),
        family.spec.latent,
        loop=loop,
        seed=seed,
        index_db_path=index_db_path,
        image_shapes=image_shapes,
        layout_result=layout_result,
    )


def _model_plan_draft(raw: dict[str, Any]) -> tuple[TrainConfig, dict[str, dict[str, Any]]]:
    """Validate estimator inputs independently without repairing an invalid training draft."""
    fields, issues = {}, {}
    for name, field in TrainConfig.model_fields.items():
        try:
            fields[name] = field.annotation.model_validate(raw.get(name, {}))
        except ValidationError as error:
            fields[name] = None
            issues[name] = validation_issues(error, prefix=[name])[0]
    # This object is private to the planner. Invalid sections remain absent;
    # their defaults must never produce estimates for a different configuration.
    return TrainConfig.model_construct(**fields), issues


def plan(
    cfg: TrainConfig | dict[str, Any],
    *,
    gpu_total_mb: float | None = None,
    gpu_memory_budget_mb: float | None = None,
    index_db_path: str | Path | None = None,
    device: str | torch.device | None = None,
    check_compile: bool = True,
) -> dict[str, Any]:
    """Plan without loading weights. None is an offline target-hardware estimate.

    Pass the execution device from CLI/service to enforce hardware constraints and account
    for CPU/MPS fp32 execution. This does not require target CUDA hardware to be locally present.
    Cache-only workers set check_compile=False because they never compile the model.
    """
    out: dict[str, Any] = {
        "ok": True,
        "errors": [],
        "warnings": [],
        "compute_policy": None,
        "source_balance": None,
        "dora": None,
    }
    draft = False
    memory_issue = None
    image_shapes: dict[tuple[int, int], dict[str, Any]] = {}
    data_layouts: list[DataLayout] = []
    try:
        cfg = TrainConfig.model_validate(cfg)
    except ValidationError as e:
        out.update(ok=False, errors=validation_issues(e))
        raw = cfg.to_dict() if isinstance(cfg, TrainConfig) else cfg
        if not isinstance(raw, dict):
            return out
        checkpoint = raw.get("checkpoint")
        if isinstance(checkpoint, dict) and checkpoint.get("resume"):
            # A partial recipe cannot authenticate the checkpoint's batch strategy.
            return out
        _preview_invalid_config(
            raw, out, index_db_path=index_db_path, image_shapes=image_shapes, layout_result=data_layouts,
        )
        cfg, field_issues = _model_plan_draft(raw)
        draft = True
        for name in ("model", "training", "adapter", "memory", "loop", "dataset"):
            if name in field_issues:
                out["memory"] = {"unavailable_issue": field_issues[name]}
                return out
        memory_issue = field_issues.get("optimizer") or field_issues.get("validation")
        if memory_issue is None and "buckets" not in out:
            memory_issue = next(
                (issue for issue in out["errors"] if issue["loc"].startswith(("dataset", "validation"))),
                out["errors"][0],
            )
    try:
        if cfg.dataset is not None and cfg.checkpoint is not None:
            from .native_batch_contract import resolve_native_batching_config

            cfg = resolve_native_batching_config(cfg)
        if (
            cfg.dataset is not None and cfg.dataset.resolution_mode == "native"
            and cfg.dataset.native_max_pixels_mode == "auto_vram"
            and cfg.checkpoint is not None and cfg.checkpoint.resume
        ):
            from .native_resolution import resolve_native_vram_config

            cfg = resolve_native_vram_config(cfg, device=device or "cpu")
    except (OSError, ValueError, TypeError, AttributeError, EOFError, RuntimeError) as error:
        issue = {"loc": "checkpoint.resume", "msg": str(error)}
        out["errors"].append(issue)
        out.update(ok=False, memory={"unavailable_issue": issue})
        return out
    try:
        family = get_family(cfg.model.family)
    except KeyError as e:
        issue = {"loc": "model.family", "msg": str(e)}
        if issue not in out["errors"]:
            out["errors"].append(issue)
        out.update(ok=False, memory={"unavailable_issue": issue})
        return out
    try:
        device_type = torch.device(device).type if device is not None else None
    except (RuntimeError, ValueError) as e:
        issue = {"loc": "device", "msg": str(e)}
        out["errors"].append(issue)
        out.update(ok=False, memory={"unavailable_issue": issue})
        return out
    if device_type not in (None, "cpu", "mps", "cuda"):
        out["errors"].append({"loc": "device", "msg": "only cpu, mps and cuda execution are supported"})
    if cfg.sampling is not None:
        out["warnings"].extend(value_advice(cfg, family.spec.latent.align))
    try:
        cfg, compute_policy = resolve_training_compute_config(cfg, device_type, current_profile())
    except ValueError as error:
        issue = {"loc": "model", "msg": str(error)}
        out["errors"].append(issue)
        out.update(ok=False, memory={"unavailable_issue": issue})
        return out
    out["compute_policy"] = compute_policy
    if check_compile:
        try:
            validate_compile_environment(cfg.memory.compile, device_type, isolated=True)
        except CompileEnvironmentError as error:
            out["errors"].append({"loc": "memory.compile", "msg": str(error)})
    metal_runtime = None
    if cfg.model.attention == "metal_flash" and device_type is not None:
        try:
            metal_runtime = resolve_metal_attention_runtime(cfg.model.attention, device_type)
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            out["errors"].append({"loc": "model.attention", "msg": str(error)})
    # Offline plans do not know which runtime recipe will apply. The execution
    # plan and trainer perform this check once the device is known.
    if cfg.checkpoint is not None and cfg.checkpoint.resume:
        metadata_path = Path(cfg.checkpoint.resume) / "state.json"
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if not isinstance(metadata, dict) or not isinstance(metadata.get("progress", {}), dict):
                    raise ValueError("训练状态的进度元数据格式无效")
                extra = metadata.get("progress", {}).get("extra", {})
                if not isinstance(extra, dict):
                    raise ValueError("训练状态的附加元数据格式无效")
                from .training_feature_contract import STATE_KEY, validate_training_feature_resume

                validate_training_feature_resume(cfg, extra.get(STATE_KEY))
                if device_type is not None:
                    validate_resume_compute_policy(compute_policy, extra.get("compute_policy"))
                    validate_metal_attention_resume(metal_runtime, extra.get("metal_attention_runtime"))
            except (OSError, UnicodeError, ValueError) as error:
                out["errors"].append({"loc": "checkpoint.resume", "msg": str(error)})
    caps = family.spec.capabilities
    checks = [
        (
            cfg.model.attention == "metal_flash" and cfg.model.attention not in family.spec.attention_backends,
            "model.attention",
            "Metal FlashAttention is not supported by this model family",
        ),
        (
            cfg.memory.blocks_to_swap > 0 and "block_swap" not in caps,
            "memory.blocks_to_swap",
            "family does not support block swap",
        ),
        (
            (cfg.dataset.masked_loss or cfg.dataset.image_fit == "pad") and "masked_loss" not in caps,
            "dataset.masked_loss",
            "family does not support masked loss, required to exclude image padding",
        ),
        (
            cfg.dataset.text_encoding == "online"
            and "online_text" not in caps
            and not cfg.training.train_text_encoder,
            "dataset.text_encoding",
            "family requires cached text encoding",
        ),
        (
            cfg.memory.activation_checkpointing != "none" and "activation_checkpointing" not in caps,
            "memory.activation_checkpointing",
            "family does not support activation checkpointing",
        ),
        (
            cfg.memory.compile and "compile" not in caps,
            "memory.compile",
            "family does not support compilation",
        ),
        (
            cfg.memory.compile and cfg.memory.blocks_to_swap > 0,
            "memory.compile",
            "compile cannot be combined with block swap",
        ),
        (
            cfg.model.attention == "metal_flash" and cfg.memory.compile,
            "memory.compile",
            "Metal FlashAttention does not support torch.compile",
        ),
        (
            device_type in ("cpu", "mps") and cfg.memory.base_precision.startswith("fp8"),
            "memory.base_precision",
            "fp8 base precision requires CUDA",
        ),
        (
            device_type in ("cpu", "mps") and cfg.model.attention in ("sage", "xformers", "flash_attn"),
            "model.attention",
            f"{cfg.model.attention} attention requires CUDA",
        ),
        (
            cfg.optimizer is not None
            and device_type in ("cpu", "mps")
            and "8bit" in cfg.optimizer.type.lower(),
            "optimizer.type",
            "8-bit optimizers require CUDA",
        ),
        (
            device_type in ("cpu", "mps") and cfg.loop.distributed_strategy == "fsdp",
            "loop.distributed_strategy",
            "显存分片训练需要至少两张 CUDA 或 DTK 显卡",
        ),
    ]
    out["errors"].extend({"loc": loc, "msg": message} for failed, loc, message in checks if failed)
    if cfg.objective is not None and cfg.sampling is not None:
        out["errors"].extend(family.training_options_errors(cfg))
    effective_dtype = (
        "fp32"
        if device_type in ("cpu", "mps")
        or (
            cfg.training.mode == "full"
            and (cfg.training.train_backbone or cfg.memory.base_precision == "fp32")
        )
        else cfg.model.dtype
    )
    if device_type == "mps" and (cfg.model.dtype != "fp32" or cfg.loop.mixed_precision != "no"):
        out["warnings"].append(
            {
                "code": "device.mps_fp32",
                "msg": "MPS training and this memory estimate use fp32 without autocast",
            }
        )
    try:
        out["errors"] += [{"loc": "model", "msg": message} for message in family.validate_config(cfg.model)]
    except (OSError, ValueError) as e:
        out["errors"].append({"loc": "model", "msg": str(e)})

    # ---- data
    ds = cfg.dataset
    native = ds.resolution_mode == "native"
    area_native = native and ds.native_batching_version in (1, 3)
    if draft:
        counts = {}
        for bucket in out.get("buckets", []):
            key = (bucket["w"], bucket["h"])
            counts[key] = counts.get(key, 0) + bucket["items"]
    else:
        counts = _append_data_plan(
            out,
            cfg,
            family.spec.latent,
            loop=cfg.loop,
            seed=cfg.loop.seed,
            index_db_path=index_db_path,
            image_shapes=image_shapes,
            layout_result=data_layouts,
        )

    if native and out.get("native") is not None:
        # Memory/cache estimates share the resolved runtime budget, while the
        # caller's saved custom value stays unchanged when auto is selected.
        ds = ds.model_copy(update={"native_max_pixels": out["native"]["max_pixels"]})
        cfg = cfg.model_copy(update={"dataset": ds})

    # ---- parameters (meta device, no weights)
    params: dict[str, Any] = {}
    memory: dict[str, Any] = {}
    meta_fn = getattr(family, "meta_backbone", None)
    if meta_fn is not None:
        error_loc = "model"
        try:
            with torch.device("meta"):
                backbone = meta_fn(cfg.model)
            base_params = _count_params(backbone)
            compute_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
                effective_dtype
            ]
            family.prepare_backbone_for_plan(backbone, cfg.model, compute_dtype)
            presets = family.presets()
            full_training = cfg.training.mode == "full"
            if not full_training and cfg.training.train_backbone and cfg.adapter.preset not in presets:
                issue = {"loc": "adapter.preset", "msg": f"unknown preset; available: {sorted(presets)}"}
                out["errors"].append(issue)
                memory["unavailable_issue"] = issue
            else:
                error_loc = "adapter"
                if full_training:
                    from ypuddin.models.training_parameters import text_modules_for_plan

                    from .training_modes import FullTrainingSet

                    backbone.requires_grad_(False)
                    modules = {}
                    if cfg.training.train_backbone:
                        modules["backbone"] = backbone
                    text_modules = {}
                    if cfg.training.train_text_encoder:
                        text_modules = text_modules_for_plan(family, cfg.model)
                        modules.update(text_modules)
                    aset = FullTrainingSet(modules, config=cfg)
                else:
                    from ypuddin.adapters.components import ComponentAdapterSet, inject_text_adapters

                    components = {}
                    backbone.requires_grad_(False)
                    # An invalid checkpoint draft can still count parameters; its precision report stays unresolved.
                    plan_save_dtype = cfg.checkpoint.save_dtype if cfg.checkpoint is not None else "fp32"
                    if cfg.training.train_backbone:
                        components["backbone"] = inject(
                            backbone,
                            cfg.adapter,
                            presets[cfg.adapter.preset],
                            prefix=family.spec.adapter_prefix,
                            base_precision=cfg.memory.base_precision
                            if cfg.memory.base_precision != "auto"
                            else "keep",
                            dora_save_dtype=plan_save_dtype,
                            compute_dtype=compute_dtype,
                        )
                    if cfg.training.train_text_encoder:
                        from ypuddin.models.training_parameters import text_modules_for_plan

                        text_modules = text_modules_for_plan(family, cfg.model)
                        for module in text_modules.values():
                            module.to(dtype=compute_dtype).requires_grad_(False)
                        components.update(inject_text_adapters(text_modules, cfg.adapter,
                                                              dora_save_dtype=plan_save_dtype,
                                                              compute_dtype=compute_dtype))
                        aset = ComponentAdapterSet(components)
                    else:
                        aset = components["backbone"]
                    from .dora_precision import precision_report, resume_errors

                    if cfg.checkpoint is not None:
                        out["dora"] = precision_report(aset, cfg, compute_dtype)
                        if device_type is not None:
                            out["errors"].extend(resume_errors(aset, cfg))
                        if out["dora"]["confirmation_required"]:
                            out["warnings"].append({
                                "code": "dora.precision_confirmation",
                                "msg": out["dora"]["confirmation_message"],
                            })
                params = {
                    "base": base_params,
                    "trainable": aset.num_params(),
                    "adapted_layers": len(aset.layers),
                    "by_algo": aset.summary()["by_algo"],
                    "training_mode": cfg.training.mode,
                }
                if native and ds.native_max_pixels_mode in {"auto", "auto_vram"} and out.get("native") is None:
                    error_loc = "dataset.native_max_pixels_mode"
                    raise ValueError("无法自动计算图像面积上限，请检查训练图片")
                if full_training:
                    params["components"] = {
                        component: sum(p.numel() for p in module.parameters() if p.requires_grad)
                        for component, module in aset.modules.items()
                    }
                    params["groups"] = aset.group_report()
                    unmatched = [key for key in cfg.optimizer.group_lr if not any(key in name for name in aset._names)]
                    if unmatched:
                        out["warnings"].append({"loc": "optimizer.group_lr", "msg": "未匹配到参数的学习率规则：" + "、".join(unmatched)})
                elif cfg.training.train_text_encoder:
                    params["components"] = {
                        component: item.num_params() for component, item in aset.components.items()
                    }
                if params["trainable"] == 0:
                    out["errors"].append(
                        {"loc": "training" if full_training else "adapter", "msg": "所选组件已全部冻结，请至少为一个组件设置非零学习率" if full_training else "adapter rules select no trainable parameters"}
                    )
                if memory_issue is None and cfg.loop.distributed_strategy == "fsdp":
                    from ypuddin.config.training_rules import distributed_training_errors

                    memory_issue = next(iter(distributed_training_errors(cfg)), None)
                if memory_issue is not None:
                    out.update(params=params, memory={"unavailable_issue": memory_issue})
                    return out
                error_loc = "memory"
                # A loader may retain native FP8 even when model.dtype is BF16.
                # Explicit base_precision applies only to selected adapter targets.
                weights_mb = _frozen_storage_bytes(backbone) / 2**20
                adapter_mb = (
                    aset.num_params()
                    * (4 if full_training or cfg.adapter.param_dtype == "fp32" else 2)
                    / 2**20
                )
                optimizer_mb = 0.0
                from .optimizer_memory import cpu_offload_memory_bytes, cpu_offload_step_workspace_bytes

                optimizer_groups = aset.param_groups(cfg.optimizer.lr, cfg.optimizer.weight_decay, cfg.optimizer.group_lr)
                optimizer_estimate = optimizer_memory_breakdown(
                    cfg.optimizer, optimizer_groups, device_type=device_type or "cuda",
                )
                optimizer_bytes = optimizer_estimate["state_bytes"]
                if optimizer_bytes is not None:
                    optimizer_mb = optimizer_bytes / 2**20
                cpu_optimizer = cpu_offload_memory_bytes(cfg.optimizer, optimizer_groups)
                gradients_mb = adapter_mb
                compensation_mb = optimizer_estimate["kahan_bytes"] / 2**20
                # EMA snapshots are FP32 exports on the host for both adapters and full training.
                ema_mb = 0.0
                cpu_ema_mb = 0.0
                if cfg.loop.ema:
                    if full_training:
                        cpu_ema_mb = sum(t.numel() * 4 for module in aset.modules.values() for t in module.state_dict().values()) / 2**20
                    else:
                        cpu_ema_mb = _adapter_ema_bytes(aset) / 2**20
                cpu_optimizer_mb = (
                    sum(cpu_optimizer.values()) + optimizer_estimate["host_state_bytes"]
                ) / 2**20
                cpu_workspace_mb = cpu_offload_step_workspace_bytes(cfg.optimizer, optimizer_groups) / 2**20
                # Each DDP process creates its optimizer shadows and CPU EMA copy.
                host_memory_mb = (cpu_optimizer_mb + cpu_workspace_mb + cpu_ema_mb) * cfg.loop.gpu_count
                host_resident_mb = (cpu_optimizer_mb + cpu_ema_mb) * cfg.loop.gpu_count
                host_step_workspace_mb = cpu_workspace_mb * cfg.loop.gpu_count
                unified_host_mb = host_resident_mb if device_type == "mps" else 0.0
                unified_step_workspace_mb = host_step_workspace_mb if device_type == "mps" else 0.0
                layout = (
                    family.memory_layout_meta(backbone) if hasattr(family, "memory_layout_meta") else None
                )
                sharding = None
                communication_mb = 0.0
                optimizer_workspace_mb = (optimizer_estimate["step_workspace_bytes"] or 0) / 2**20
                gradient_workspace_unknown = False
                if cfg.loop.distributed_strategy != "fsdp":
                    try:
                        norm_workspace = gradient_workspace_bytes(optimizer_groups, clip_norm=cfg.optimizer.grad_clip_norm)
                        # Norm/clip temporaries are released before optimizer.step.
                        optimizer_workspace_mb = max(optimizer_workspace_mb, norm_workspace / 2**20)
                    except (NotImplementedError, RuntimeError, ValueError):
                        gradient_workspace_unknown = True
                if cfg.loop.gpu_count > 1 and cfg.loop.distributed_strategy == "ddp":
                    # DDP keeps reduction buckets beside the accumulated parameter gradients.
                    if full_training:
                        communication_mb = adapter_mb
                    else:
                        ddp_parameters = {id(p): p for p in aset.parameters() if p.requires_grad}
                        communication_mb = sum(
                            p.numel() * p.element_size() for p in ddp_parameters.values()
                        ) / 2**20
                initialization_peak = None
                estimate_notes = []
                if cfg.loop.distributed_strategy == "fsdp":
                    if not full_training:
                        from .sharded_adapters import shardable_frozen_weights

                        shardable_frozen_weights(backbone)
                    sharding = _fsdp_memory(backbone, list(layout.blocks) if layout else [], cfg)
                    adapter_mb = max(sharding["local_parameter_bytes_by_rank"]) / 2**20
                    gradients_mb = max(sharding["local_trainable_bytes_by_rank"]) / 2**20
                    optimizer_mb = max(sharding["optimizer_state_bytes_by_rank"]) / 2**20
                    weights_mb = sharding["replicated_buffer_bytes"] / 2**20
                    communication_mb = sharding["communication_bytes_estimate"] / 2**20
                    optimizer_workspace_mb = sharding["optimizer_workspace_bytes_estimate"] / 2**20
                    initialization_peak = sharding["initialization_bytes_estimate"] / 2**20 + 512
                    estimate_notes = [
                        "显存按单张卡中占用最大的分片估算；主参数、梯度和优化器状态分片，少量小参数及缓冲区在各卡保留。",
                        "激活和文本、图片编码缓存按每卡计算，不随卡数平均分摊。",
                        "通信缓冲区、优化器临时张量和激活峰值是估算；分片不保证按卡数成倍提速。",
                    ]
                    if sharding["optimizer"] == "adafactor":
                        estimate_notes.insert(
                            1, "Adafactor 按实际行、列状态计算；设置 beta1 会额外保留一份 FP32 一阶动量。"
                        )
                if layout and cfg.memory.blocks_to_swap > len(layout.blocks):
                    out["errors"].append(
                        {
                            "loc": "memory.blocks_to_swap",
                            "msg": f"cannot swap more than {len(layout.blocks)} blocks",
                        }
                    )
                # Peak activations of the largest bucket under each checkpointing mode.
                # Only the modes this family implements are estimated or suggested. A mode it lacks
                # runs as block checkpointing (Krea 2) or is rejected by the family's own checks.
                supported_modes = (
                    "none",
                    *(mode for mode in ("block", "unsloth") if mode in family.spec.checkpointing_modes),
                )
                current_mode = (
                    cfg.memory.activation_checkpointing
                    if cfg.memory.activation_checkpointing in supported_modes
                    else "block"
                    if "block" in supported_modes
                    else "none"
                )
                activation_dtype = (
                    torch.float32
                    if device_type in ("cpu", "mps")
                    or (sharding is not None and cfg.loop.mixed_precision == "no")
                    else compute_dtype if cfg.loop.mixed_precision == "no"
                    else {"bf16": torch.bfloat16, "fp16": torch.float16}[cfg.loop.mixed_precision]
                )
                fusion_blocks = family.adapter_checkpoint_blocks_for_plan(backbone)
                if fusion_blocks is None:
                    fusion_blocks = list(layout.blocks) if layout else []
                fusion = adapter_memory(
                    backbone, fusion_blocks, activation_dtype,
                    autocast_enabled=cfg.loop.mixed_precision != "no" and device_type not in ("cpu", "mps"),
                )
                swapped_mb = 0.0
                swap_staging_mb = 0.0
                if layout and cfg.memory.blocks_to_swap and layout.blocks and device_type in (None, "cuda"):
                    # BlockSwapper owns the last N blocks' frozen tensors only;
                    # embeddings/text fusion and adapter parameters remain resident.
                    n_swap = min(cfg.memory.blocks_to_swap, len(layout.blocks))
                    block_sizes = [_frozen_storage_bytes(block) / 2**20 for block in layout.blocks[-n_swap:]]
                    swapped_mb = sum(block_sizes)
                    # Current block plus the prefetched next block can coexist.
                    swap_staging_mb = sum(sorted(block_sizes, reverse=True)[:2])
                dequant_mb = max(
                    (
                        2 * m.weight.numel() * DTYPE_BYTES[effective_dtype] / 2**20
                        for m in backbone.modules()
                        if isinstance(m, FrozenLinear) and m.is_fp8
                    ),
                    default=0.0,
                )
                text_mode = "online" if cfg.training.train_text_encoder else ds.text_encoding
                if text_mode == "auto":
                    text_mode = "online" if "online_text" in family.spec.capabilities else "cached"
                text_inputs = (
                    text_plan_inputs(cfg, family.spec.latent, data_layouts[0] if data_layouts else None)
                    if cfg.model.family in {"anima", "krea2", "sdxl", "flux2"} else {}
                )
                out["errors"].extend(issue for issue in text_inputs.get("errors", []) if issue not in out["errors"])
                text_preview_unknown = (
                    cfg.model.family == "sdxl" and text_inputs.get("sdxl_prompt_batches", []) is None
                )
                text_encoder_mb = 0.0
                text_profiles = {}
                text_capacity_unknown = False
                cache_text_capacity_unknown = False

                def profile_capacity(profile, *, cached_text=False):
                    nonlocal text_capacity_unknown
                    if not text_capacity_complete(profile, unified_memory=device_type == "mps", cached_text=cached_text):
                        text_capacity_unknown = True
                    capacity = cached_text_phase_capacity if cached_text else text_phase_capacity
                    return capacity(profile, unified_memory=device_type == "mps")

                def text_profile(batch_size: int, *, phase: str = "online", token_length: int | None = None,
                                 t5_token_length: int | None = None, prompt_spec: dict | None = None,
                                 include_condition: bool = True, cached_condition_kind: str = "training"):
                    if token_length is None:
                        token_length = text_inputs.get("anima_lengths", {}).get(phase, {}).get("qwen_tokens")
                    if t5_token_length is None:
                        t5_token_length = text_inputs.get("anima_lengths", {}).get(phase, {}).get("t5_tokens")
                    key = (batch_size, token_length, t5_token_length, json.dumps(prompt_spec, sort_keys=True),
                           include_condition, cached_condition_kind)
                    if key not in text_profiles:
                        text_profiles[key] = frozen_text_phase(
                            cfg, compute_dtype, batch_size=batch_size, device_type=device_type,
                            token_length=token_length,
                            t5_token_length=t5_token_length,
                            include_condition=include_condition,
                            cached_condition_kind=cached_condition_kind,
                            prompt_spec=prompt_spec,
                            attention_backend=(
                                "math" if current_profile() == "linux-dtk" and cfg.loop.deterministic else None
                            ),
                            linear_compute_dtype=(
                                torch.float32 if (compute_policy or {}).get("frozen_text_implementation") else None
                            ),
                        )
                    return text_profiles[key]

                if (
                    text_mode == "online"
                    and (not cfg.memory.offload_text_encoder or device_type == "cpu")
                    and not cfg.training.train_text_encoder
                ):
                    profile = text_profile(1)
                    text_encoder_mb = (
                        profile["weights_mb"] if profile is not None
                        else family.spec.text.encoder_params * DTYPE_BYTES[effective_dtype] / 2**20
                    )
                if cfg.training.mode == "adapter" and cfg.training.train_text_encoder:
                    text_encoder_mb = (
                        sum(_frozen_storage_bytes(module) for module in text_modules.values()) / 2**20
                    )
                if full_training and cfg.training.train_text_encoder:
                    text_encoder_mb = sum(_frozen_storage_bytes(module) for module in text_modules.values()) / 2**20
                offloaded_text_mb = 0.0
                if text_mode == "online" and cfg.memory.offload_text_encoder and device_type != "cpu":
                    profile = text_profile(1)
                    if profile is not None:
                        offloaded_text_mb = profile["weights_mb"]
                        host_memory_mb += offloaded_text_mb * cfg.loop.gpu_count
                unified_training_host_mb = unified_host_mb + (offloaded_text_mb if device_type == "mps" else 0.0)
                vae = _plan_vae(cfg, compute_dtype, device)
                latent_encoder_mb = latent_workspace_mb = 0.0
                if vae is not None and not ds.cache_latents:
                    # Online encoding loads the VAE with the first batch and keeps it for the run.
                    latent_encoder_mb = vae.weights_mb
                    if sharding is not None:
                        estimate_notes.append(
                            "关闭图像编码缓存后，完整 VAE 在每张卡上与训练状态同时驻留；"
                            "权重与在线编码工作区均计入单卡估算，不按卡数分摊。"
                        )
                optimizer_unknown = sharding is None and (
                    optimizer_estimate["state_bytes"] is None
                    or optimizer_estimate["step_workspace_bytes"] is None
                    or gradient_workspace_unknown
                )
                vae_unknown = vae is None and cfg.model.family in {"anima", "krea2", "sdxl", "flux2"}
                training_base = (
                    weights_mb
                    - swapped_mb
                    + text_encoder_mb
                    + latent_encoder_mb
                    + adapter_mb
                    + optimizer_mb
                    + gradients_mb
                    + compensation_mb
                    + ema_mb
                    + swap_staging_mb
                    + dequant_mb
                    + communication_mb
                    + (host_resident_mb if device_type == "cpu" else 0.0)
                )
                # Optimizer temporaries are created after backward releases its saved activations.
                optimizer_step_live = (
                    training_base + optimizer_workspace_mb - dequant_mb
                    + (host_step_workspace_mb if device_type == "cpu" else 0.0)
                )
                evaluation_ops = evaluation_operations(aset, cfg, optimizer_estimate, device_type=device_type or "cuda")
                evaluation_state = evaluation_snapshot_memory(
                    aset, cfg.optimizer, meta_device_type=device_type or "cuda",
                )
                if not evaluation_ops["context_active"]:
                    evaluation_state = dict(evaluation_state, enabled=False, snapshot_device_bytes=0, snapshot_host_bytes=0,
                                            restore_workspace_device_bytes=0, restore_workspace_host_bytes=0)
                evaluation_snapshot_mb = evaluation_state["snapshot_device_bytes"] / 2**20
                evaluation_host_mb = evaluation_state["snapshot_host_bytes"] / 2**20
                evaluation_restore_mb = evaluation_state["restore_workspace_device_bytes"] / 2**20
                evaluation_restore_host_mb = evaluation_state["restore_workspace_host_bytes"] / 2**20
                if device_type == "cpu":
                    evaluation_snapshot_mb += evaluation_host_mb
                    evaluation_restore_mb += evaluation_restore_host_mb
                evaluation_host_capacity_mb = evaluation_host_mb if device_type == "mps" else 0.0
                evaluation_restore_host_capacity_mb = evaluation_restore_host_mb if device_type == "mps" else 0.0
                host_memory_mb = max(host_memory_mb, host_resident_mb + offloaded_text_mb * cfg.loop.gpu_count
                                     + evaluation_host_mb + evaluation_restore_host_mb)
                evaluation_base_mb = training_base - gradients_mb - dequant_mb + evaluation_snapshot_mb
                evaluation_restore_live = evaluation_base_mb + evaluation_restore_mb
                evaluation_restore_capacity = (
                    evaluation_restore_live + unified_training_host_mb
                    + evaluation_host_capacity_mb + evaluation_restore_host_capacity_mb
                )
                # Export stages already contain their own raw/restore copies.
                evaluation_resident_mb = training_base - gradients_mb - dequant_mb
                evaluation_host_resident_mb = host_resident_mb + offloaded_text_mb * cfg.loop.gpu_count
                mode_projection = project_evaluation_stages(
                    evaluation_mode_stages(evaluation_ops["mode"], evaluation_state) if evaluation_ops["context_active"] else [],
                    resident_mb=evaluation_resident_mb, host_resident_mb=evaluation_host_resident_mb,
                    device_type=device_type or "cuda", processes=cfg.loop.gpu_count,
                )
                export_projection = project_evaluation_stages(
                    evaluation_ops["stages"], resident_mb=evaluation_resident_mb,
                    host_resident_mb=evaluation_host_resident_mb,
                    device_type=device_type or "cuda", processes=cfg.loop.gpu_count,
                )
                evaluation_restore_live = max(evaluation_restore_live, mode_projection["known_allocated_mb"])
                evaluation_restore_capacity = max(evaluation_restore_capacity, mode_projection["known_capacity_mb"])
                host_memory_mb = max(host_memory_mb, mode_projection["known_host_mb"], export_projection["known_host_mb"])
                evaluation_allocated_unknown = bool(mode_projection["allocated_unknowns"] or export_projection["allocated_unknowns"])
                evaluation_capacity_unknown = bool(mode_projection["capacity_unknowns"] or export_projection["capacity_unknowns"])
                evaluation_host_unknown = bool(mode_projection["host_unknowns"] or export_projection["host_unknowns"])
                fixed_cache_phases = family.cache_memory_estimate(cfg, compute_dtype)
                fixed_cache_capacity = dict(fixed_cache_phases)
                fixed_cache_allocated = dict(fixed_cache_phases)
                if text_mode == "cached" and not cfg.training.train_text_encoder:
                    batches = text_inputs.get("anima_cache_batches", [
                        {"batch_size": text_inputs.get("cache_batch_size", 16), "qwen_tokens": None},
                    ])
                    profiles = [text_profile(batch["batch_size"], phase="cache", token_length=batch["qwen_tokens"],
                                             t5_token_length=batch.get("t5_tokens"), include_condition=False)
                                for batch in batches]
                    if cfg.model.family == "sdxl":
                        profiles.extend(text_profile(1, phase="prompt", prompt_spec=spec)
                                        for spec in text_inputs.get("sdxl_prompt_batches") or [])
                    elif text_inputs.get("prompt_cache_batch_size"):
                        profiles.append(text_profile(text_inputs["prompt_cache_batch_size"], phase="prompt"))
                    profiles = [profile for profile in profiles if profile is not None]
                    if profiles:
                        cache_text_capacity_unknown = any(
                            not text_capacity_complete(profile, unified_memory=device_type == "mps")
                            for profile in profiles
                        )
                        text_cache_peak = max(profile["peak_mb_estimate"] for profile in profiles)
                        fixed_cache_phases["text_cache"] = _training_peak_mb(text_cache_peak)
                        fixed_cache_allocated["text_cache"] = text_cache_peak
                        fixed_cache_capacity["text_cache"] = _training_peak_mb(max(
                            text_phase_capacity(profile, unified_memory=device_type == "mps") for profile in profiles
                        ))
                preview_issue = next((issue for issue in text_inputs.get("errors", [])
                                      if issue["loc"].startswith("sampling.")), None)
                if "preview_items" in text_inputs:
                    previews = {(item["width"], item["height"]): 1
                                for item in text_inputs["preview_items"] or []
                                if cfg.sampling is not None and cfg.sampling.enabled}
                else:
                    try:
                        previews = preview_shapes(cfg, family.spec.latent.align)
                    except (OSError, ValueError) as error:
                        preview_issue = {"loc": "sampling.prompts_file", "msg": str(error)}
                        if preview_issue not in out["errors"]:
                            out["errors"].append(preview_issue)
                        previews = {}
                if initialization_peak is not None:
                    initialization_peak += text_encoder_mb
                preview_base_mb = evaluation_base_mb

                def prompt_profile(spec):
                    return text_profile(1, phase="prompt", token_length=spec.get("qwen_tokens"),
                                        t5_token_length=spec.get("t5_tokens"), prompt_spec=spec.get("prompt_spec"),
                                        cached_condition_kind="preview")

                preview_items = (text_inputs.get("preview_items", [
                        {"width": width, "height": height, "positive": {}, "negative": {}}
                        for width, height in previews
                    ]) or []) if previews else []
                preview_stages = preview_residency(
                    preview_items, prompt_profile, family.spec.latent,
                    cached_text=text_mode == "cached", unified_memory=device_type == "mps",
                )
                fixed_preview_inputs = []
                preview_text_peak = preview_text_capacity = 0.0
                for stage in preview_stages:
                    text_capacity_unknown |= not stage["text_capacity_complete"]
                    vae_resident = vae.weights_mb if vae is not None and ds.cache_latents else 0.0
                    text_resident = preview_base_mb - (text_encoder_mb if text_mode == "online" else 0.0)
                    if stage["previous_image"]:
                        text_resident += vae_resident
                    preview_text_peak = max(preview_text_peak, text_resident + stage["text_peak_mb"])
                    preview_text_capacity = max(
                        preview_text_capacity, text_resident + stage["text_capacity_mb"] + evaluation_host_capacity_mb,
                    )
                    if vae is not None:
                        phase = preview_decoding_phase(vae, {(stage["width"], stage["height"]): 1})
                        resident = preview_base_mb + vae_resident + stage["decoding_retained_mb"]
                        fixed_preview_inputs.append((phase, resident, unified_training_host_mb + evaluation_host_capacity_mb))
                fixed_preview = _combined_vae_phase(fixed_preview_inputs)
                activation_layout = backbone_memory_layout(
                    backbone, family, cfg, compute_dtype, activation_dtype, device_type or "cuda", layout=layout,
                )
                fusion_mb = {mode: (fusion[mode]["retained_bytes"] + fusion[mode]["workspace_bytes"]) / 2**20
                             for mode in supported_modes}
                preview_profile = preview_working_memory(
                    cfg, backbone, aset, preview_items, preview_stages,
                    dtype=activation_dtype, device_type=device_type or "cuda",
                    export_layout=evaluation_ops["export_layout"],
                    backbone_workspace=lambda width, height, batch, text_tokens: estimate_backbone_memory(
                        activation_layout, width=width, height=height, batch_size=batch,
                        checkpointing="none", text_tokens=text_tokens, training=False,
                    )["forward_workspace_bytes"],
                ) if preview_items else None
                preview_live = preview_base_mb + preview_profile["allocated_additional_mb"] if preview_profile else 0.0
                preview_host = evaluation_host_resident_mb + evaluation_host_mb + preview_profile["host_additional_mb"] if preview_profile else 0.0
                preview_capacity = preview_live + (preview_host if device_type == "mps" else 0.0)
                if preview_profile:
                    for stage, row in zip(preview_stages, preview_profile["rows"], strict=True):
                        for phase in ("sampling", "image_conversion", "pixel_clamp"):
                            cached_vae = vae.weights_mb if vae is not None and ds.cache_latents and (stage["previous_image"] or phase != "sampling") else 0.0
                            preview_live = max(preview_live, preview_base_mb + cached_vae + row[phase + "_additional_mb"])
                            capacity = preview_base_mb + cached_vae + row[phase + "_additional_mb"]
                            if device_type == "mps":
                                capacity = preview_base_mb + cached_vae + evaluation_host_resident_mb + evaluation_host_mb + row[phase + "_shared_additional_mb"]
                            preview_capacity = max(preview_capacity, capacity)
                        vae_resident = vae.weights_mb if vae is not None and ds.cache_latents and stage["previous_image"] else 0.0
                        text_resident = preview_base_mb - (text_encoder_mb if text_mode == "online" else 0.0) + vae_resident
                        preview_text_peak = max(preview_text_peak, text_resident + stage["text_peak_mb"] + row["text_context_device_mb"])
                        preview_text_capacity = max(preview_text_capacity, text_resident + stage["text_capacity_mb"]
                                                    + (row["text_context_shared_mb"] + unified_training_host_mb + evaluation_host_capacity_mb
                                                       if device_type == "mps" else row["text_context_device_mb"]))
                    fixed_preview_inputs = [
                        (phase, resident + row["decode_context_device_mb"],
                         host + (row["decode_context_host_mb"] if device_type == "mps" else 0.0))
                        for (phase, resident, host), row in zip(fixed_preview_inputs, preview_profile["rows"], strict=False)
                    ]
                    fixed_preview = _combined_vae_phase(fixed_preview_inputs)

                def condition_tokens(phase):
                    if cfg.model.family == "anima":
                        return text_inputs.get("anima_lengths", {}).get(phase, {}).get("t5_tokens", 512)
                    if cfg.model.family == "sdxl":
                        return cfg.model.sdxl_max_token_length + 2
                    return 512

                def evaluate_sizing(counts, image_shapes, ds):
                    adaptive_device_mb = adaptive_host_mb = adaptive_host_peak_mb = 0.0
                    adaptive_update_live = adaptive_update_capacity = 0.0
                    if cfg.objective.infonoise_enabled:
                        bins = cfg.objective.infonoise_bins
                        keys = len(counts) if cfg.objective.timestep_sampling == "resolution_sqrt" else 1
                        # Constructor arrays, cached baselines, and a bounded set of update vectors.
                        target_bytes = 4 if device_type == "mps" else 8
                        device_bytes = (4 * bins + 8) * target_bytes + ds.batch_size * 12
                        host_bytes = (5 * bins + 6 + max(1, keys) * bins) * 8
                        host_peak = host_bytes + bins * 128 + ds.batch_size * 96
                        if device_type == "cpu":
                            device_bytes += host_bytes
                            host_bytes = host_peak = 0
                        adaptive_device_mb, adaptive_host_mb = device_bytes / 2**20, host_bytes / 2**20
                        adaptive_host_peak_mb = host_peak / 2**20
                        adaptive_update_live = training_base - dequant_mb - gradients_mb + adaptive_device_mb
                        adaptive_update_capacity = adaptive_update_live + unified_training_host_mb
                        if device_type == "mps":
                            adaptive_update_capacity += adaptive_host_peak_mb
                    adaptive_shared_mb = adaptive_device_mb + (adaptive_host_mb if device_type == "mps" else 0.0)

                    act_by_bucket = []
                    auxiliary_act_peak = 0.0
                    act_by_mode = dict.fromkeys(supported_modes, 0.0)
                    backbone_by_mode = dict.fromkeys(supported_modes, 0.0)
                    text_encoding_peak = preview_text_peak + adaptive_device_mb
                    text_encoding_capacity = preview_text_capacity + adaptive_shared_mb
                    validation_live = validation_capacity = 0.0
                    hook_mb = caller_host_mb = validation_caller_host_mb = 0.0
                    cache_batches, training_batches = _vae_batches(
                        image_shapes, ds, cfg.memory.vae_encode_batch_size, online_kind="training",
                    )
                    _, validation_batches = _vae_batches(
                        image_shapes, ds, cfg.memory.vae_encode_batch_size, online_kind="validation",
                    )
                    for (w, h), available_items in sorted(counts.items()) or [((r, r), 0) for r in ds.resolutions]:
                        forward_batch = (
                            min(ds.batch_size, max(1, ds.native_max_pixels // (w * h)))
                            if area_native else ds.batch_size
                        )
                        if available_items:
                            forward_batch = min(forward_batch, available_items)
                        if text_mode == "online" and not cfg.training.train_text_encoder:
                            profile = text_profile(forward_batch)
                            if profile is not None:
                                text_encoding_peak = max(
                                    text_encoding_peak,
                                    training_base - dequant_mb - text_encoder_mb + profile["peak_mb_estimate"] + adaptive_device_mb,
                                )
                                text_encoding_capacity = max(
                                    text_encoding_capacity, training_base - dequant_mb - text_encoder_mb
                                    + profile_capacity(profile) + adaptive_shared_mb + unified_host_mb,
                                )
                        elif text_mode == "cached":
                            profile = text_profile(forward_batch, phase="cached_training")
                            if profile is not None:
                                text_encoding_peak = max(
                                    text_encoding_peak,
                                    training_base - dequant_mb + profile["cached_condition_workspace_mb_estimate"] + adaptive_device_mb,
                                )
                                text_encoding_capacity = max(
                                    text_encoding_capacity, training_base - dequant_mb
                                    + profile_capacity(profile, cached_text=True) + adaptive_shared_mb + unified_training_host_mb,
                                )
                        for mode in act_by_mode:
                            estimate = estimate_backbone_memory(
                                activation_layout, width=w, height=h, batch_size=forward_batch, checkpointing=mode,
                                text_tokens=condition_tokens("cached_training" if text_mode == "cached" else "online"),
                            )
                            dynamic_mb = (estimate["activation_bytes"] + estimate["input_bytes"]) / 2**20 + fusion_mb[mode]
                            act_by_mode[mode] = max(act_by_mode[mode], dynamic_mb)
                            backbone_by_mode[mode] = max(backbone_by_mode[mode], training_base + dynamic_mb + adaptive_device_mb)
                            if mode == current_mode:
                                auxiliary_act_peak = max(auxiliary_act_peak, estimate.get("auxiliary_activation_bytes", 0) / 2**20)
                                hook_mb = max(hook_mb, estimate["input_bytes"] / 2**20)
                                caller_host_mb = max(caller_host_mb, estimate["host_input_bytes"] / 2**20)
                                act_by_bucket.append({"w": w, "h": h, "mb": round(dynamic_mb)})
                    hook_capacity_mb = hook_mb + (caller_host_mb if device_type == "mps" else 0.0)
                    text_encoding_peak += hook_mb
                    text_encoding_capacity += hook_capacity_mb
                    adaptive_update_live += hook_mb
                    adaptive_update_capacity += hook_capacity_mb
                    for (w, h), entry in image_shapes.items():
                        count = entry.get("validation_items", len(entry["validation"]))
                        if not count:
                            continue
                        batch = min(count, ds.batch_size,
                                    max(1, ds.native_max_pixels // (w * h)) if area_native else ds.batch_size)
                        estimate = estimate_backbone_memory(
                            activation_layout, width=w, height=h, batch_size=batch, checkpointing="none",
                            text_tokens=condition_tokens("validation"), training=False,
                        )
                        live = (evaluation_base_mb + adaptive_device_mb
                                + (estimate["forward_workspace_bytes"] + estimate["input_bytes"]) / 2**20)
                        validation_live = max(validation_live, live)
                        validation_caller_host_mb = max(validation_caller_host_mb, estimate["host_input_bytes"] / 2**20)
                        validation_capacity = max(validation_capacity, live + unified_training_host_mb
                                                  + evaluation_host_capacity_mb + (adaptive_host_mb + validation_caller_host_mb
                                                                                  if device_type == "mps" else 0.0))
                    if not cfg.training.train_text_encoder:
                        validation_batch = max((
                            min(entry.get("validation_items", len(entry["validation"])), ds.batch_size,
                                max(1, ds.native_max_pixels // (w * h)) if area_native else ds.batch_size)
                            for (w, h), entry in image_shapes.items()
                        ), default=0)
                        if validation_batch:
                            profile = text_profile(validation_batch, phase="validation")
                            if profile is not None:
                                cached = text_mode == "cached"
                                resident = evaluation_base_mb - (0 if cached else text_encoder_mb) + adaptive_device_mb
                                key = "cached_condition_workspace_mb_estimate" if cached else "peak_mb_estimate"
                                text_encoding_peak = max(
                                    text_encoding_peak, resident + profile[key],
                                )
                                text_encoding_capacity = max(
                                    text_encoding_capacity, resident + profile_capacity(profile, cached_text=cached)
                                    + evaluation_host_capacity_mb + unified_host_mb
                                    + (adaptive_host_mb if device_type == "mps" else 0.0),
                                )
                                validation_live = max(validation_live, resident + profile[key])
                                validation_capacity = max(
                                    validation_capacity, resident + profile_capacity(profile, cached_text=cached)
                                    + unified_training_host_mb + evaluation_host_capacity_mb
                                    + (adaptive_host_mb if device_type == "mps" else 0.0),
                                )
                    if validation_batches:
                        validation_live = max(validation_live, evaluation_restore_live + adaptive_device_mb)
                        validation_capacity = max(validation_capacity, evaluation_restore_capacity + adaptive_shared_mb)
                    validation_live += hook_mb if validation_batches else 0.0
                    validation_capacity += hook_capacity_mb if validation_batches else 0.0
                    backbone_live = backbone_by_mode[current_mode]
                    common_live = max(
                        optimizer_step_live + adaptive_device_mb + hook_mb,
                        text_encoding_peak,
                        evaluation_restore_live + adaptive_device_mb + hook_mb,
                        validation_live,
                        export_projection["known_allocated_mb"] + adaptive_device_mb + hook_mb,
                        adaptive_update_live,
                        preview_live + adaptive_device_mb + hook_mb,
                    )
                    training_live = max(backbone_live, common_live)
                    training_peak = _training_peak_mb(training_live)
                    common_capacity = max(
                        optimizer_step_live + unified_training_host_mb + unified_step_workspace_mb + adaptive_shared_mb + hook_capacity_mb,
                        text_encoding_capacity,
                        evaluation_restore_capacity + adaptive_shared_mb + hook_capacity_mb,
                        validation_capacity,
                        export_projection["known_capacity_mb"] + adaptive_shared_mb + hook_capacity_mb,
                        adaptive_update_capacity,
                        preview_capacity + adaptive_shared_mb + hook_capacity_mb,
                    )
                    training_capacity = _training_peak_mb(max(
                        backbone_live + unified_training_host_mb
                        + (adaptive_host_mb + caller_host_mb if device_type == "mps" else 0.0), common_capacity,
                    ))
                    cache_phases = dict(fixed_cache_phases)
                    cache_capacity = dict(fixed_cache_capacity)
                    cache_allocated = dict(fixed_cache_allocated)
                    # The VAE's own phases: caching with nothing else on the device, or encoding and
                    # preview decoding beside the training state (no activations; gradients are
                    # cleared before previews).
                    image_encoding = preview_decoding = None
                    vae_phases = []
                    vae_capacity = []
                    vae_allocated = []
                    vae_phase_inputs = []
                    if vae is not None:
                        if ds.cache_latents:
                            image_encoding = _encoding_vae_phase(vae, cache_batches, cached=True)
                        else:
                            for batches, resident, host, validating in (
                                (training_batches, training_base - dequant_mb + adaptive_device_mb + hook_mb,
                                 unified_training_host_mb + (adaptive_host_mb + caller_host_mb if device_type == "mps" else 0.0), False),
                                (validation_batches, evaluation_base_mb + adaptive_device_mb + hook_mb,
                                 unified_training_host_mb + evaluation_host_capacity_mb
                                 + (adaptive_host_mb + caller_host_mb if device_type == "mps" else 0.0), True),
                            ):
                                phase = _encoding_vae_phase(vae, batches, cached=False)
                                if phase is not None:
                                    vae_phase_inputs.append((phase, resident, host))
                                    if validating:
                                        validation_live = max(validation_live, resident + phase["workspace_mb_estimate"])
                                        validation_capacity = max(
                                            validation_capacity, resident + host + phase["capacity_workspace_mb_estimate"],
                                        )
                            image_encoding = _combined_vae_phase(vae_phase_inputs)
                    if image_encoding is not None and ds.cache_latents:
                        image_encoding = _resident_vae_phase(image_encoding, vae.weights_mb)
                        vae_phase_inputs.append((image_encoding, vae.weights_mb, 0.0))
                        image_encoding["mode"] = "cached"
                        cache_phases["latent_cache"] = image_encoding["peak_mb_estimate"]
                        cache_capacity["latent_cache"] = image_encoding["capacity_peak_mb_estimate"]
                        cache_allocated["latent_cache"] = image_encoding["allocated_peak_mb_estimate"]
                    elif image_encoding is not None:
                        image_encoding["mode"] = "online"
                        vae_phases.append(image_encoding["peak_mb_estimate"])
                        vae_capacity.append(image_encoding["capacity_peak_mb_estimate"])
                        vae_allocated.append(image_encoding["allocated_peak_mb_estimate"])
                    if fixed_preview is not None:
                        adaptive_preview_inputs = [
                            (phase, resident + adaptive_device_mb + hook_mb,
                             host + (adaptive_host_mb + caller_host_mb if device_type == "mps" else 0.0))
                            for phase, resident, host in fixed_preview_inputs
                        ]
                        preview_decoding = _combined_vae_phase(adaptive_preview_inputs)
                        vae_phase_inputs.extend(adaptive_preview_inputs)
                        vae_phases.append(preview_decoding["peak_mb_estimate"])
                        vae_capacity.append(preview_decoding["capacity_peak_mb_estimate"])
                        vae_allocated.append(preview_decoding["allocated_peak_mb_estimate"])
                    peak = max([training_peak, *cache_phases.values(), *vae_phases])
                    capacity_peak = max([training_capacity, *cache_capacity.values(), *vae_capacity])
                    allocated_peak = max([training_live, *cache_allocated.values(), *vae_allocated])
                    if initialization_peak is not None:
                        peak = max(peak, initialization_peak)
                        capacity_peak = max(capacity_peak, initialization_peak)
                        allocated_peak = max(allocated_peak, initialization_peak - 512)
                    latent_workspace_mb = (
                        image_encoding["workspace_mb_estimate"]
                        if image_encoding is not None and image_encoding["mode"] == "online" else 0.0
                    )
                    # The overall peak after each VAE setting that lowers one of its phases.
                    vae_setting_peaks = {}
                    vae_setting_capacity = {}
                    for path in dict.fromkeys(
                        path
                        for phase, _, _ in vae_phase_inputs
                        for path in phase["reserved_workspace_reductions_mb"]
                    ):
                        values = [
                            training_peak,
                            initialization_peak or 0,
                            *(value for key, value in cache_phases.items() if key != "latent_cache"),
                        ]
                        for phase, resident, _ in vae_phase_inputs:
                            reduced = phase["reserved_workspace_reductions_mb"].get(path, phase["reserved_workspace_mb_estimate"])
                            values.append(resident + reduced)
                        vae_setting_peaks[path] = round(max(values))
                    for path in dict.fromkeys(
                        path for phase, _, _ in vae_phase_inputs
                        for path in phase["capacity_workspace_reductions_mb"]
                    ):
                        values = [training_capacity, initialization_peak or 0,
                                  *(value for key, value in cache_capacity.items() if key != "latent_cache")]
                        for phase, resident, host in vae_phase_inputs:
                            reduced = phase["capacity_workspace_reductions_mb"].get(path, phase["capacity_workspace_mb_estimate"])
                            values.append(resident + host + reduced)
                        vae_setting_capacity[path] = round(max(values))
                    return {
                        "act_by_bucket": act_by_bucket,
                        "auxiliary_act_peak": auxiliary_act_peak,
                        "act_by_mode": act_by_mode,
                        "backbone_by_mode": backbone_by_mode,
                        "adaptive_device_mb": adaptive_device_mb,
                        "adaptive_host_mb": adaptive_host_mb,
                        "adaptive_shared_mb": adaptive_shared_mb,
                        "adaptive_host_peak_mb": adaptive_host_peak_mb,
                        "adaptive_update_live": adaptive_update_live,
                        "adaptive_update_capacity": adaptive_update_capacity,
                        "caller_hook_mb": hook_mb,
                        "caller_host_mb": caller_host_mb,
                        "validation_caller_host_mb": validation_caller_host_mb,
                        "common_live": common_live,
                        "common_capacity": common_capacity,
                        "training_live": training_live,
                        "text_encoding_peak": text_encoding_peak,
                        "text_encoding_capacity": text_encoding_capacity,
                        "training_peak": training_peak,
                        "training_capacity": training_capacity,
                        "validation_live": validation_live,
                        "validation_capacity": validation_capacity,
                        "cache_phases": cache_phases,
                        "cache_capacity": cache_capacity,
                        "cache_allocated": cache_allocated,
                        "image_encoding": image_encoding,
                        "preview_decoding": preview_decoding,
                        "vae_phases": vae_phases,
                        "vae_capacity": vae_capacity,
                        "peak": peak,
                        "capacity_peak": capacity_peak,
                        "allocated_peak": allocated_peak,
                        "latent_workspace_mb": latent_workspace_mb,
                        "vae_setting_peaks": vae_setting_peaks,
                        "vae_setting_capacity": vae_setting_capacity,
                    }

                sized = evaluate_sizing(counts, image_shapes, ds)
                sized_dataset = ds
                if native and out.get("native") is not None:
                    native_out = out["native"]
                    resolved = ds.native_max_pixels_resolved if ds.native_max_pixels_mode == "auto_vram" else None
                    chosen = resolved
                    selection_error = None
                    if resolved is None:
                        if device_type not in {"cuda", "mps"}:
                            selection_error = "当前设备使用 CPU，无法按显存计算面积上限；请改用自动分辨率优先或自定义。"
                        elif type(gpu_total_mb) not in (int, float) or not math.isfinite(gpu_total_mb) or gpu_total_mb <= 0:
                            selection_error = "无法读取训练显卡的显存容量。"
                        elif cfg.training.train_text_encoder:
                            selection_error = "训练文本编码器时无法估算完整显存占用；请改用自动分辨率优先或自定义。"
                        elif preview_issue is not None:
                            selection_error = "无法读取预览提示词，请先修正提示词文件。"
                        elif text_preview_unknown:
                            selection_error = "无法读取预览提示词的编码长度，请检查本地分词器。"
                        elif text_capacity_unknown or cache_text_capacity_unknown:
                            selection_error = "无法估算当前文本条件的统一内存占用，请使用自定义面积上限。"
                        elif optimizer_unknown:
                            selection_error = "无法读取当前优化器的内存布局，请检查优化器版本或使用自定义面积上限。"
                        elif evaluation_capacity_unknown:
                            selection_error = "无法估算当前检查点保存的内存占用，请使用自定义面积上限。"
                        elif fusion["unsupported"]:
                            selection_error = "当前适配器无法估算完整显存占用；请改用自动分辨率优先或自定义。"
                        elif vae is None and cfg.model.family in {"anima", "krea2", "sdxl", "flux2"}:
                            selection_error = "无法估算当前 VAE 的显存占用，请检查模型文件或使用自定义面积上限。"
                        elif not data_layouts or native_out["auto_max_pixels"] is None:
                            selection_error = "无法读取训练图片尺寸，请检查数据集。"
                        else:
                            geometry = NativeVramGeometry(data_layouts[0], ds)
                            fixed_peak = max(
                                _training_peak_mb(max(
                                    training_base,
                                    optimizer_step_live + unified_step_workspace_mb,
                                ) + unified_training_host_mb),
                                initialization_peak or 0, *fixed_cache_capacity.values(),
                                _training_peak_mb(evaluation_restore_capacity),
                                _training_peak_mb(export_projection["known_capacity_mb"]),
                                fixed_preview["capacity_peak_mb_estimate"] if fixed_preview else 0,
                                vae.weights_mb + 512 if vae is not None and ds.cache_latents else 0,
                            )
                            budget = min(gpu_total_mb * HEADROOM, gpu_memory_budget_mb) if gpu_memory_budget_mb is not None else gpu_total_mb * HEADROOM

                            def candidate_peak(pixels, candidate_counts, candidate_images):
                                candidate_ds = ds.model_copy(update={"native_max_pixels": pixels})
                                estimate = evaluate_sizing(candidate_counts, candidate_images, candidate_ds)
                                return round(estimate["capacity_peak"])

                            def training_lower_bound(pixels, candidate_counts, candidate_images):
                                candidate_ds = ds.model_copy(update={"native_max_pixels": pixels, "batch_size": 1})
                                estimate = evaluate_sizing(candidate_counts, candidate_images, candidate_ds)
                                return round(estimate["training_capacity"])

                            if round(fixed_peak) <= budget:
                                chosen = maximum_fitting_pixels(
                                    geometry, native_out["auto_max_pixels"], candidate_peak, budget,
                                    training_lower_bound=training_lower_bound,
                                )
                            if chosen is None and selection_error is None:
                                selection_error = (
                                    "当前配置在显存预算内无法保留训练图片尺寸；请允许超限图片等比缩小，或调整训练参数。"
                                    if ds.native_overflow == "error"
                                    else "当前训练配置超出显存预算，降低图像面积仍无法容纳；请调整批大小、适配器或显存设置。"
                                )
                    native_out.update(auto_vram_max_pixels=chosen, auto_vram_error=selection_error)
                    if ds.native_max_pixels_mode == "auto_vram":
                        if selection_error is not None:
                            out["errors"].append({"loc": "dataset.native_max_pixels_mode", "msg": selection_error})
                        elif chosen is not None and data_layouts:
                            ds = ds.model_copy(update={"native_max_pixels": chosen, "native_max_pixels_resolved": chosen})
                            cfg = cfg.model_copy(update={"dataset": ds})
                            geometry = NativeVramGeometry(data_layouts[0], ds)
                            data_layout = geometry.resolved_layout(chosen)
                            image_shapes.clear()
                            out["warnings"] = [w for w in out["warnings"] if w.get("code") not in {
                                "images.padding", "native.execution", "native.execution.v1", "native.execution.area", "captions.missing", "buckets.small",
                                "distributed.tail",
                            }]
                            counts = _append_data_plan(
                                out, cfg, family.spec.latent, loop=cfg.loop, seed=cfg.loop.seed,
                                index_db_path=None, image_shapes=image_shapes, prepared_layout=data_layout,
                            )
                            out["native"].update(auto_vram_max_pixels=chosen, auto_vram_error=None)
                if ds is not sized_dataset:
                    sized = evaluate_sizing(counts, image_shapes, ds)
                act_by_bucket = sized["act_by_bucket"]
                auxiliary_act_peak = sized["auxiliary_act_peak"]
                act_by_mode = sized["act_by_mode"]
                training_live = sized["training_live"]
                training_peak = sized["training_peak"]
                training_capacity = sized["training_capacity"]
                cache_phases = sized["cache_phases"]
                image_encoding = sized["image_encoding"]
                preview_decoding = sized["preview_decoding"]
                vae_phases = sized["vae_phases"]
                peak = sized["peak"]
                capacity_peak = sized["capacity_peak"]
                latent_workspace_mb = sized["latent_workspace_mb"]
                vae_setting_peaks = sized["vae_setting_peaks"]
                adaptive_device_mb = sized["adaptive_device_mb"]
                adaptive_shared_mb = sized["adaptive_shared_mb"]
                host_memory_mb = max(
                    host_memory_mb + sized["adaptive_host_mb"],
                    evaluation_host_resident_mb + sized["adaptive_host_peak_mb"],
                    preview_host + sized["adaptive_host_mb"],
                )
                host_memory_mb += sized["caller_host_mb"]
                host_memory_mb = max(host_memory_mb, evaluation_host_resident_mb + evaluation_host_mb
                                     + sized["adaptive_host_mb"] + sized["validation_caller_host_mb"])
                total_unknown = bool(
                    cfg.training.train_text_encoder or preview_issue or text_preview_unknown
                    or optimizer_unknown or vae_unknown
                    or evaluation_allocated_unknown
                    or (preview_profile and preview_profile["export_layout_unknowns"])
                )
                training_unknown = (
                    cfg.training.train_text_encoder or optimizer_unknown
                    or (vae_unknown and not ds.cache_latents)
                    or evaluation_allocated_unknown
                    or (preview_profile and preview_profile["export_layout_unknowns"])
                )
                validation_unknown = bool(cfg.training.train_text_encoder
                                          or (vae_unknown and not ds.cache_latents)
                                          or mode_projection["allocated_unknowns"])

                def phase_estimates(values, *, capacity=False):
                    result = {
                        key: None if key == "text_cache" and (
                            text_preview_unknown or (capacity and cache_text_capacity_unknown)
                        ) else round(value)
                        for key, value in values.items()
                    }
                    if vae_unknown and ds.cache_latents:
                        result["latent_cache"] = None
                    return result

                memory = {
                    "estimate_scope": "per_device",
                    "communication_mb_estimate": round(communication_mb, 1),
                    "optimizer_workspace_mb_estimate": None
                    if gradient_workspace_unknown or (optimizer_unknown and optimizer_estimate["step_workspace_bytes"] is None)
                    else round(optimizer_workspace_mb, 1),
                    "evaluation_snapshot_mb_estimate": round(evaluation_snapshot_mb, 1),
                    "evaluation_host_snapshot_mb_estimate": round(evaluation_host_mb, 1),
                    "evaluation_restore_workspace_mb_estimate": round(evaluation_restore_mb, 1),
                    "evaluation_restore_host_workspace_mb_estimate": round(evaluation_restore_host_mb, 1),
                    "adapter_graph_mb_estimate": round(fusion[current_mode]["retained_bytes"] / 2**20, 1),
                    "adapter_workspace_mb_estimate": round(fusion[current_mode]["workspace_bytes"] / 2**20, 1),
                    "auxiliary_activations_mb_estimate": round(auxiliary_act_peak, 1),
                    "activation_estimate_method": "family_geometry_dtype_estimate",
                    "activation_includes_gradients": False,
                    "activation_includes_adapter_graph": True,
                    "validation_allocated_peak_mb_estimate": None if validation_unknown else round(sized["validation_live"]),
                    "validation_capacity_mb_estimate": None
                    if validation_unknown or text_capacity_unknown or mode_projection["capacity_unknowns"]
                    else round(sized["validation_capacity"]),
                    "training_runtime_mb_estimate": round(training_peak - training_live, 1),
                    "initialization_peak_mb_estimate": round(initialization_peak)
                    if initialization_peak is not None
                    else None,
                    "estimate_notes": estimate_notes,
                    "sharding": sharding,
                    "weights_mb": round(weights_mb),
                    "swapped_mb": round(swapped_mb),
                    "text_encoder_mb": round(text_encoder_mb),
                    "latent_encoder_mb": round(latent_encoder_mb, 1),
                    "latent_encoding_workspace_mb_estimate": round(latent_workspace_mb, 1),
                    "adapter_mb": round(adapter_mb, 1),
                    "optimizer_mb": None if optimizer_unknown and optimizer_bytes is None else round(optimizer_mb, 1),
                    "cpu_optimizer_mb": round(cpu_optimizer_mb, 1),
                    "cpu_ema_mb": round(cpu_ema_mb, 1),
                    "host_memory_mb_estimate": None if evaluation_host_unknown else round(host_memory_mb, 1),
                    "gpu_memory_budget_mb": gpu_memory_budget_mb,
                    "gradients_mb": round(gradients_mb, 1),
                    "compensation_mb": round(compensation_mb, 1),
                    "ema_mb": round(ema_mb, 1),
                    "swap_staging_mb": round(swap_staging_mb, 1),
                    "dequant_mb": round(dequant_mb, 1),
                    "training_peak_mb_estimate": None
                    if training_unknown
                    else round(training_peak),
                    "training_capacity_mb_estimate": None
                    if training_unknown or text_capacity_unknown or evaluation_capacity_unknown
                    else round(training_capacity),
                    "known_training_residency_mb": round(training_base - dequant_mb + adaptive_device_mb),
                    "unestimated_components": (
                        (["text_encoder_activations"] if cfg.training.train_text_encoder else [])
                        + (["adapter_reconstruction"] if fusion["unsupported"] else [])
                        + ([] if sharding is not None else optimizer_estimate["unknowns"])
                        + (["gradient_norm_workspace"] if gradient_workspace_unknown else [])
                        + (["compiled_kernel_workspace"] if cfg.memory.compile and device_type in {None, "cuda"} else [])
                        + ["text_" + issue["reason"] for issue in text_inputs.get("unknowns", [])]
                        + (["preview_shapes"] if preview_issue is not None else [])
                        + (["image_vae"] if vae_unknown else [])
                        + (["text_condition_host_staging"] if text_capacity_unknown or cache_text_capacity_unknown else [])
                        + sorted(set(mode_projection["allocated_unknowns"] + mode_projection["capacity_unknowns"]
                                     + mode_projection["host_unknowns"] + export_projection["allocated_unknowns"]
                                     + export_projection["capacity_unknowns"] + export_projection["host_unknowns"]))
                        + evaluation_ops["non_tensor_unknowns"]
                        + (preview_profile["export_layout_unknowns"] if preview_profile else [])
                    ),
                    "image_encoding": image_encoding,
                    "preview_decoding": preview_decoding,
                    "vae_setting_peak_mb_estimates": None
                    if total_unknown
                    else vae_setting_peaks,
                    "vae_setting_capacity_mb_estimates": None
                    if total_unknown or text_capacity_unknown or cache_text_capacity_unknown or evaluation_capacity_unknown
                    else sized["vae_setting_capacity"],
                    "cache_phase_peak_mb_estimates": phase_estimates(cache_phases),
                    "cache_phase_capacity_mb_estimates": phase_estimates(sized["cache_capacity"], capacity=True),
                    "cache_phase_allocated_mb_estimates": phase_estimates(sized["cache_allocated"]),
                    "activations_mb_by_bucket": act_by_bucket,
                    "peak_mb_estimate": None if total_unknown else round(peak),
                    "capacity_peak_mb_estimate": None
                    if total_unknown or text_capacity_unknown or cache_text_capacity_unknown or evaluation_capacity_unknown
                    else round(capacity_peak),
                    "allocated_peak_mb_estimate": None if total_unknown else round(sized["allocated_peak"]),
                    # The same estimate under each checkpointing mode, so a fix can say what it saves.
                    "checkpointing_peak_mb_estimates": None
                    if total_unknown or "activation_checkpointing" not in caps
                    else {
                        mode: round(
                            max(
                                _training_peak_mb(max(
                                    sized["backbone_by_mode"][mode],
                                    optimizer_step_live + adaptive_device_mb,
                                    sized["text_encoding_peak"],
                                    evaluation_restore_live + adaptive_device_mb,
                                    export_projection["known_allocated_mb"] + adaptive_device_mb,
                                    sized["validation_live"],
                                    sized["adaptive_update_live"],
                                    sized["common_live"],
                                )),
                                *cache_phases.values(),
                                *vae_phases,
                                initialization_peak or 0,
                            )
                        )
                        for mode, act in act_by_mode.items()
                    },
                    "checkpointing_capacity_mb_estimates": None
                    if (total_unknown or text_capacity_unknown or cache_text_capacity_unknown or evaluation_capacity_unknown
                        or "activation_checkpointing" not in caps)
                    else {
                        mode: round(max(
                            _training_peak_mb(max(
                                sized["backbone_by_mode"][mode] + unified_training_host_mb
                                + (sized["adaptive_host_mb"] + sized["caller_host_mb"] if device_type == "mps" else 0.0),
                                optimizer_step_live + unified_step_workspace_mb + unified_training_host_mb + adaptive_shared_mb,
                                sized["text_encoding_capacity"],
                                evaluation_restore_capacity + adaptive_shared_mb,
                                export_projection["known_capacity_mb"] + adaptive_shared_mb,
                                sized["validation_capacity"],
                                sized["adaptive_update_capacity"],
                                sized["common_capacity"],
                            )),
                            *sized["cache_capacity"].values(), *sized["vae_capacity"],
                            initialization_peak or 0,
                        ))
                        for mode, act in act_by_mode.items()
                    },
                    "gpu_total_mb": gpu_total_mb,
                    "heuristic": True,
                    "activation_checkpointing": cfg.memory.activation_checkpointing,
                    "device": str(device) if device is not None else None,
                    "effective_dtype": effective_dtype,
                    "suggestions": [],
                }
                tight = gpu_total_mb * 0.9 if gpu_total_mb and not cfg.training.train_text_encoder else None
                if tight:
                    # Each fix only where it lowers the phase that is over the budget.
                    for phase in (image_encoding, preview_decoding):
                        if phase is not None and phase["capacity_peak_mb_estimate"] > tight:
                            for path in phase["capacity_workspace_reductions_mb"]:
                                suggestion = (
                                    "set memory.vae_encode_batch_size = 1"
                                    if path == "memory.vae_encode_batch_size"
                                    else f"enable {path}"
                                )
                                if suggestion not in memory["suggestions"]:
                                    memory["suggestions"].append(suggestion)
                    if (
                        image_encoding is not None
                        and image_encoding["mode"] == "online"
                        and max(training_capacity, image_encoding["capacity_peak_mb_estimate"]) > tight
                    ):
                        memory["suggestions"].append("enable dataset.cache_latents")
                    if preview_decoding is not None and preview_decoding["capacity_peak_mb_estimate"] > tight:
                        memory["suggestions"].append("lower sampling.width / sampling.height")
                if tight and not training_unknown and not text_capacity_unknown and training_capacity > tight:
                    # Multi-GPU training rejects offloaded checkpoints and block swap.
                    multi_gpu = cfg.loop.gpu_count > 1 or cfg.loop.distributed_strategy == "fsdp"
                    if (
                        cfg.memory.activation_checkpointing == "none"
                        and act_by_mode["block"] < act_by_mode["none"]
                    ):
                        memory["suggestions"].append("set memory.activation_checkpointing = 'block'")
                    elif (
                        cfg.memory.activation_checkpointing == "block"
                        and not multi_gpu
                        and "unsloth" in family.spec.checkpointing_modes
                        and act_by_mode["unsloth"] < act_by_mode["block"]
                    ):
                        memory["suggestions"].append("set memory.activation_checkpointing = 'unsloth'")
                    if (
                        device_type != "mps"
                        and not multi_gpu
                        and "block_swap" in family.spec.capabilities
                        and not cfg.memory.blocks_to_swap
                        and cfg.training.mode != "full"
                    ):
                        memory["suggestions"].append("enable memory.blocks_to_swap")
                    if (
                        device_type in (None, "cuda")
                        and "8bit" not in cfg.optimizer.type
                        and sharding is None
                    ):
                        memory["suggestions"].append("use optimizer.type = 'adamw8bit'")
                    if sharding and (
                        sharding["optimizer"] != "adafactor" or cfg.optimizer.args.get("beta1") is not None
                    ):
                        memory["suggestions"].append(
                            "可改用 Adafactor 且不设置 beta1，以减少优化器状态；这会改变优化算法。"
                        )
                    if text_encoder_mb:
                        memory["suggestions"].append(
                            "set dataset.text_encoding = 'cached' (frees the text encoder)"
                        )
                if tight and capacity_peak > tight:
                    out["warnings"].append(
                        {
                            "code": "vram.tight",
                            "msg": f"estimated peak {capacity_peak:.0f} MB vs {gpu_total_mb:.0f} MB available",
                        }
                    )
        except Exception as e:  # noqa: BLE001
            issue = {
                "loc": error_loc,
                # The automatic area limit's own reason is already plain; others name the failed step.
                "msg": str(e)
                if error_loc == "dataset.native_max_pixels_mode"
                else f"could not prepare model/adapter plan: {e}",
            }
            out["errors"].append(issue)
            memory["unavailable_issue"] = issue
            if out.get("native") is not None:
                out["native"]["auto_vram_error"] = issue["msg"]
    out["params"] = params
    out["memory"] = memory
    out["text_encoding"] = (
        "online"
        if (cfg.dataset.text_encoding == "auto" and "online_text" in family.spec.capabilities)
        or cfg.dataset.text_encoding == "online"
        or cfg.training.train_text_encoder
        else "cached"
    )
    if native and cfg.dataset.native_max_pixels_mode == "auto_vram":
        native_result = out.get("native") or {}
        if native_result.get("auto_vram_max_pixels") is None:
            issue = {
                "loc": "dataset.native_max_pixels_mode",
                "msg": native_result.get("auto_vram_error") or "无法计算显存面积上限，请检查训练参数和数据集。",
            }
            if not any(error["loc"] == issue["loc"] for error in out["errors"]):
                out["errors"].append(issue)
    out["ok"] = not out["errors"]
    return out
