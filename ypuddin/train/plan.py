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

from .adapter_memory import adapter_memory
from .advice import value_advice
from .memory_budget import HEADROOM
from .metal_compute import resolve_metal_attention_runtime, validate_metal_attention_resume
from .native_vram import NativeVramGeometry, maximum_fitting_pixels
from .optimizer_memory import optimizer_state_bytes
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


def _training_peak_mb(live_mb: float) -> float:
    # The allocator keeps reusable blocks in addition to live tensors. This is a
    # workspace allowance, separate from the device headroom used at admission.
    return live_mb + max(512.0, live_mb * 0.05)


def _plan_vae(cfg: TrainConfig, dtype: torch.dtype, device: str | torch.device | None) -> VaeMemory | None:
    """The run's image VAE; a geometry that cannot be read leaves the VAE phases out of the estimate."""
    try:
        return vae_memory(cfg, dtype, device=device, profile=current_profile())
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return None


def _vae_batches(
    image_shapes: dict[tuple[int, int], dict[str, Any]], ds: DatasetConfig, encode_batch: int
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    """Images per VAE call by shape: while caching, and while training or validating online.

    Caching groups up to ``encode_batch`` images of one shape, each flip a separate entry. Online
    encoding splits a training batch, native microbatch group or validation batch into calls of
    at most ``encode_batch`` images.
    """
    native = ds.resolution_mode == "native"
    cache, online = {}, {}
    for (width, height), entry in image_shapes.items():
        entries = len(entry["images"]) * (2 if ds.flip else 1) + len(entry["validation"])
        cache[width, height] = min(encode_batch, entries)
        group = min(ds.batch_size, max(1, ds.native_max_pixels // (width * height))) if native else ds.batch_size
        online[width, height] = min(encode_batch, group, max(entry["items"], len(entry["validation"])))
    return cache, online


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
                entry = image_shapes.setdefault(item.bucket.key, {"items": 0, "images": set(), "validation": set()})
                entry["validation" if validation else "images"].add(item.record.path)
                entry["items"] += not validation
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
        # sampler drops incomplete multi-rank tail groups. Never use registry totals.
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
    batches = (
        math.ceil(len(items) / ds.batch_size)
        if native
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
        per_rank_batches = batches // loop.gpu_count
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
            "tail_policy": "drop_incomplete_rank_group" if loop.gpu_count > 1 else "keep",
            "dropped_samples": 0,
        }
        if loop.gpu_count > 1:
            sampler_type = NativeBatchSampler if native else BucketBatchSampler
            batch_plan = sampler_type(
                [item.bucket.key for item in items], ds.batch_size, seed=seed or 0
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
            native_plan = NativeBatchSampler(shape_keys, ds.batch_size, seed=seed).plan()
            if loop is not None and loop.gpu_count > 1:
                native_plan = native_plan[: len(native_plan) - len(native_plan) % loop.gpu_count]
            native_groups = []
            for batch in native_plan:
                shapes = [shape_keys[index] for index in batch]
                groups = microbatch_indices(shapes, max_pixels)
                native_groups.append([(shapes[group[0]], len(group)) for group in groups])
                for group in groups:
                    key = shapes[group[0]]
                    forward_counts[key] = forward_counts.get(key, 0) + 1
            if loop is not None and loop.gpu_count > 1:
                for offset in range(0, len(native_groups), loop.gpu_count):
                    ranks = native_groups[offset : offset + loop.gpu_count]
                    slots = max(map(len, ranks))
                    for groups in ranks:
                        padding = slots - len(groups)
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
        out["warnings"].append(
            {
                "code": "native.execution",
                "msg": "native resolution uses pixel-bounded shape groups and image-weighted gradient accumulation; pixel budget is not a VRAM guarantee",
            }
        )
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
        if (
            cfg.dataset is not None and cfg.dataset.resolution_mode == "native"
            and cfg.dataset.native_max_pixels_mode == "auto_vram"
            and cfg.checkpoint is not None and cfg.checkpoint.resume
        ):
            from .native_resolution import resolve_native_vram_config

            cfg = resolve_native_vram_config(cfg, device=device or "cpu")
    except (OSError, ValueError, TypeError, AttributeError) as error:
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
            validate_compile_environment(cfg.memory.compile, device_type)
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
    if cfg.checkpoint is not None and cfg.checkpoint.resume and device_type is not None:
        metadata_path = Path(cfg.checkpoint.resume) / "state.json"
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if not isinstance(metadata, dict) or not isinstance(metadata.get("progress", {}), dict):
                    raise ValueError("训练状态的进度元数据格式无效")
                extra = metadata.get("progress", {}).get("extra", {})
                if not isinstance(extra, dict):
                    raise ValueError("训练状态的附加元数据格式无效")
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
                    from types import SimpleNamespace

                    from ypuddin.models.training_parameters import text_parameter_count

                    if any(isinstance(layer, FrozenLinear) for layer in backbone.modules()):
                        raise ValueError("Full fine-tuning requires unquantized base weights")
                    backbone.to(dtype=torch.float32).requires_grad_(cfg.training.train_backbone)
                    encoder_params = (
                        text_parameter_count(family, cfg.model) if cfg.training.train_text_encoder else 0
                    )
                    trainable = (base_params if cfg.training.train_backbone else 0) + encoder_params
                    aset = SimpleNamespace(
                        num_params=lambda: trainable,
                        layers={},
                        summary=lambda: {
                            "by_algo": {
                                "full-model": int(cfg.training.train_backbone)
                                + int(cfg.training.train_text_encoder)
                            }
                        },
                    )
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
                        "backbone": base_params if cfg.training.train_backbone else 0,
                        "text_encoder": encoder_params,
                    }
                elif cfg.training.train_text_encoder:
                    params["components"] = {
                        component: item.num_params() for component, item in aset.components.items()
                    }
                if params["trainable"] == 0:
                    out["errors"].append(
                        {"loc": "adapter", "msg": "adapter rules select no trainable parameters"}
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
                weights_mb = (
                    0.0
                    if full_training and cfg.training.train_backbone
                    else _frozen_storage_bytes(backbone) / 2**20
                )
                adapter_mb = (
                    aset.num_params()
                    * (4 if full_training or cfg.adapter.param_dtype == "fp32" else 2)
                    / 2**20
                )
                optimizer_mb = aset.num_params() * 4 * (0.5 if "8bit" in cfg.optimizer.type else 2) / 2**20
                if not full_training:
                    optimizer_bytes = optimizer_state_bytes(
                        cfg.optimizer,
                        aset.param_groups(cfg.optimizer.lr, cfg.optimizer.weight_decay, cfg.optimizer.group_lr),
                    )
                    if optimizer_bytes is not None:
                        optimizer_mb = optimizer_bytes / 2**20
                gradients_mb = adapter_mb
                compensation_mb = adapter_mb if cfg.optimizer.kahan else 0.0
                ema_mb = adapter_mb if cfg.loop.ema else 0.0
                layout = (
                    family.memory_layout_meta(backbone) if hasattr(family, "memory_layout_meta") else None
                )
                sharding = None
                communication_mb = optimizer_workspace_mb = 0.0
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
                units = family.spec.activation_units
                n_blocks = len(layout.blocks) if layout else 1
                # A backbone measured as a whole already counts its width; the others scale by it.
                whole = dict(family.spec.backbone_activation_units)
                block_units = whole or {
                    "none": units * n_blocks,
                    # Each block keeps only its input; one block is recomputed at a time.
                    "block": min(units * n_blocks, n_blocks + units + 2),
                    # Block inputs wait in system memory, so only the recomputed block stays.
                    "unsloth": min(units * n_blocks, units + 3),
                }
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
                fusion_mb = {
                    mode: (fusion[mode]["retained_bytes"] + fusion[mode]["workspace_bytes"]) / 2**20
                    for mode in supported_modes
                }
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
                text_encoder_mb = 0.0
                if (
                    text_mode == "online"
                    and not cfg.memory.offload_text_encoder
                    and not cfg.training.train_text_encoder
                ):
                    text_encoder_mb = family.spec.text.encoder_params * DTYPE_BYTES[effective_dtype] / 2**20
                if cfg.training.mode == "adapter" and cfg.training.train_text_encoder:
                    text_encoder_mb = (
                        sum(_frozen_storage_bytes(module) for module in text_modules.values()) / 2**20
                    )
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
                    + optimizer_workspace_mb
                )
                fixed_cache_phases = family.cache_memory_estimate(cfg, compute_dtype)
                previews = preview_shapes(cfg, family.spec.latent.align)
                if initialization_peak is not None:
                    initialization_peak += text_encoder_mb
                fixed_preview = None
                if vae is not None and previews:
                    fixed_preview = preview_decoding_phase(vae, previews)
                    fixed_preview["peak_mb_estimate"] = round(
                        training_base - gradients_mb - optimizer_workspace_mb - dequant_mb
                        + (vae.weights_mb if ds.cache_latents else 0.0)
                        + fixed_preview["workspace_mb_estimate"] + 512
                    )
                auxiliary_by_batch = {}

                def evaluate_sizing(counts, image_shapes, ds):
                    act_by_bucket = []
                    act_peak = auxiliary_act_peak = 0.0
                    act_by_mode = dict.fromkeys(supported_modes, 0.0)
                    cache_batches, online_batches = _vae_batches(image_shapes, ds, cfg.memory.vae_encode_batch_size)
                    for (w, h), _n in sorted(counts.items()) or [((r, r), 0) for r in ds.resolutions]:
                        tokens = (w // family.spec.latent.align) * (h // family.spec.latent.align)
                        tokens = family.training_tokens_for_plan(tokens)
                        hidden = (
                            getattr(backbone, "dim", None)
                            or getattr(backbone, "model_channels", None)
                            or getattr(backbone, "inner_dim", None)
                            or getattr(getattr(backbone, "config", None), "features", 2048)
                        )
                        activation_bytes = torch.empty((), device="meta", dtype=activation_dtype).element_size()
                        forward_batch = (
                            min(ds.batch_size, max(1, ds.native_max_pixels // (w * h)))
                            if native
                            else ds.batch_size
                        )
                        unit_mb = tokens * (1 if whole else hidden) * activation_bytes * forward_batch / 2**20
                        if forward_batch not in auxiliary_by_batch:
                            auxiliary_by_batch[forward_batch] = family.auxiliary_activation_bytes_for_plan(
                                backbone, forward_batch, activation_dtype,
                            ) / 2**20
                        auxiliary_act_peak = max(auxiliary_act_peak, auxiliary_by_batch[forward_batch])
                        for mode in act_by_mode:
                            act_by_mode[mode] = max(act_by_mode[mode], unit_mb * block_units[mode])
                        act = unit_mb * block_units[current_mode]
                        act_by_bucket.append({"w": w, "h": h, "mb": round(act)})
                        act_peak = max(act_peak, act)
                    training_live = training_base + act_peak + auxiliary_act_peak + fusion_mb[current_mode]
                    training_peak = _training_peak_mb(training_live)
                    cache_phases = dict(fixed_cache_phases)
                    # The VAE's own phases: caching with nothing else on the device, or encoding and
                    # preview decoding beside the training state (no activations; gradients are
                    # cleared before previews).
                    image_encoding = preview_decoding = None
                    vae_phases = []
                    if vae is not None:
                        encode_phase = latent_cache_phase if ds.cache_latents else online_encoding_phase
                        batches = cache_batches if ds.cache_latents else online_batches
                        image_encoding = encode_phase(vae, batches)
                        single = encode_phase(vae, dict.fromkeys(batches, 1))
                        if single is not None and single["workspace_mb_estimate"] < image_encoding["workspace_mb_estimate"]:
                            image_encoding["workspace_reductions_mb"]["memory.vae_encode_batch_size"] = single[
                                "workspace_mb_estimate"
                            ]
                    if image_encoding is not None and ds.cache_latents:
                        image_encoding["mode"] = "cached"
                        cache_phases["latent_cache"] = image_encoding["peak_mb_estimate"]
                    elif image_encoding is not None:
                        image_encoding["mode"] = "online"
                        image_encoding["peak_mb_estimate"] = round(
                            training_base - optimizer_workspace_mb - dequant_mb
                            + image_encoding["workspace_mb_estimate"] + 512
                        )
                        vae_phases.append(image_encoding["peak_mb_estimate"])
                    if fixed_preview is not None:
                        preview_decoding = fixed_preview
                        vae_phases.append(preview_decoding["peak_mb_estimate"])
                    peak = max([training_peak, *cache_phases.values(), *vae_phases])
                    if initialization_peak is not None:
                        peak = max(peak, initialization_peak)
                    latent_workspace_mb = (
                        image_encoding["workspace_mb_estimate"]
                        if image_encoding is not None and image_encoding["mode"] == "online" else 0.0
                    )
                    # The overall peak after each VAE setting that lowers one of its phases.
                    vae_setting_peaks = {}
                    for path in dict.fromkeys(
                        path
                        for phase in (image_encoding, preview_decoding)
                        if phase is not None
                        for path in phase["workspace_reductions_mb"]
                    ):
                        values = [
                            training_peak,
                            initialization_peak or 0,
                            *(value for key, value in cache_phases.items() if key != "latent_cache"),
                        ]
                        for phase in (image_encoding, preview_decoding):
                            if phase is not None:
                                reduced = phase["workspace_reductions_mb"].get(path, phase["workspace_mb_estimate"])
                                values.append(phase["peak_mb_estimate"] - phase["workspace_mb_estimate"] + reduced)
                        vae_setting_peaks[path] = round(max(values))
                    return {
                        "act_by_bucket": act_by_bucket,
                        "auxiliary_act_peak": auxiliary_act_peak,
                        "act_by_mode": act_by_mode,
                        "training_live": training_live,
                        "training_peak": training_peak,
                        "cache_phases": cache_phases,
                        "image_encoding": image_encoding,
                        "preview_decoding": preview_decoding,
                        "vae_phases": vae_phases,
                        "peak": peak,
                        "latent_workspace_mb": latent_workspace_mb,
                        "vae_setting_peaks": vae_setting_peaks,
                    }

                if native and out.get("native") is not None:
                    native_out = out["native"]
                    resolved = ds.native_max_pixels_resolved if ds.native_max_pixels_mode == "auto_vram" else None
                    chosen = resolved
                    selection_error = None
                    if resolved is None:
                        if device_type not in {"cuda", "mps"}:
                            selection_error = "请先选择训练显卡，再计算显存面积上限。"
                        elif type(gpu_total_mb) not in (int, float) or not math.isfinite(gpu_total_mb) or gpu_total_mb <= 0:
                            selection_error = "无法读取训练显卡的显存容量。"
                        elif cfg.training.train_text_encoder:
                            selection_error = "训练文本编码器时无法计算显存面积上限，请选择按图片尺寸自动计算或自定义。"
                        elif fusion["unsupported"]:
                            selection_error = "当前适配器无法估算完整显存占用，请选择按图片尺寸自动计算或自定义。"
                        elif vae is None and cfg.model.family in {"anima", "krea2", "sdxl", "flux2"}:
                            selection_error = "无法估算当前 VAE 的显存占用，请检查模型文件或使用自定义面积上限。"
                        elif not data_layouts or native_out["auto_max_pixels"] is None:
                            selection_error = "无法读取训练图片尺寸，请检查数据集。"
                        else:
                            geometry = NativeVramGeometry(data_layouts[0], ds)
                            fixed_peak = max(
                                _training_peak_mb(training_base + fusion_mb[current_mode]),
                                initialization_peak or 0, *fixed_cache_phases.values(),
                                fixed_preview["peak_mb_estimate"] if fixed_preview else 0,
                                vae.weights_mb + 512 if vae is not None and ds.cache_latents else 0,
                            )
                            budget = gpu_total_mb * HEADROOM

                            def candidate_peak(pixels, candidate_counts, candidate_images):
                                candidate_ds = ds.model_copy(update={"native_max_pixels": pixels})
                                return round(evaluate_sizing(candidate_counts, candidate_images, candidate_ds)["peak"])

                            def training_lower_bound(pixels, candidate_counts, candidate_images):
                                candidate_ds = ds.model_copy(update={"native_max_pixels": pixels, "batch_size": 1})
                                return round(evaluate_sizing(candidate_counts, candidate_images, candidate_ds)["training_peak"])

                            if round(fixed_peak) <= budget:
                                chosen = maximum_fitting_pixels(
                                    geometry, native_out["auto_max_pixels"], candidate_peak, budget,
                                    training_lower_bound=training_lower_bound,
                                )
                            if chosen is None:
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
                                "images.padding", "native.execution", "captions.missing", "buckets.small",
                            }]
                            counts = _append_data_plan(
                                out, cfg, family.spec.latent, loop=cfg.loop, seed=cfg.loop.seed,
                                index_db_path=None, image_shapes=image_shapes, prepared_layout=data_layout,
                            )
                            out["native"].update(auto_vram_max_pixels=chosen, auto_vram_error=None)
                sized = evaluate_sizing(counts, image_shapes, ds)
                act_by_bucket = sized["act_by_bucket"]
                auxiliary_act_peak = sized["auxiliary_act_peak"]
                act_by_mode = sized["act_by_mode"]
                training_live = sized["training_live"]
                training_peak = sized["training_peak"]
                cache_phases = sized["cache_phases"]
                image_encoding = sized["image_encoding"]
                preview_decoding = sized["preview_decoding"]
                vae_phases = sized["vae_phases"]
                peak = sized["peak"]
                latent_workspace_mb = sized["latent_workspace_mb"]
                vae_setting_peaks = sized["vae_setting_peaks"]
                memory = {
                    "estimate_scope": "per_device",
                    "communication_mb_estimate": round(communication_mb, 1),
                    "optimizer_workspace_mb_estimate": round(optimizer_workspace_mb, 1),
                    "adapter_graph_mb_estimate": round(fusion[current_mode]["retained_bytes"] / 2**20, 1),
                    "adapter_workspace_mb_estimate": round(fusion[current_mode]["workspace_bytes"] / 2**20, 1),
                    "auxiliary_activations_mb_estimate": round(auxiliary_act_peak, 1),
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
                    "optimizer_mb": round(optimizer_mb, 1),
                    "gradients_mb": round(gradients_mb, 1),
                    "compensation_mb": round(compensation_mb, 1),
                    "ema_mb": round(ema_mb, 1),
                    "swap_staging_mb": round(swap_staging_mb, 1),
                    "dequant_mb": round(dequant_mb, 1),
                    "training_peak_mb_estimate": None
                    if cfg.training.train_text_encoder
                    else round(training_peak),
                    "known_training_residency_mb": round(
                        weights_mb
                        + text_encoder_mb
                        + latent_encoder_mb
                        + adapter_mb
                        + optimizer_mb
                        + gradients_mb
                        + compensation_mb
                        + ema_mb
                        + communication_mb
                    ),
                    "unestimated_components": (
                        (["text_encoder_activations"] if cfg.training.train_text_encoder else [])
                        + (["adapter_reconstruction"] if fusion["unsupported"] else [])
                    ),
                    "image_encoding": image_encoding,
                    "preview_decoding": preview_decoding,
                    "vae_setting_peak_mb_estimates": vae_setting_peaks,
                    "cache_phase_peak_mb_estimates": {
                        key: round(value) for key, value in cache_phases.items()
                    },
                    "activations_mb_by_bucket": act_by_bucket,
                    "peak_mb_estimate": None if cfg.training.train_text_encoder else round(peak),
                    # The same estimate under each checkpointing mode, so a fix can say what it saves.
                    "checkpointing_peak_mb_estimates": None
                    if cfg.training.train_text_encoder or "activation_checkpointing" not in caps
                    else {
                        mode: round(
                            max(
                                _training_peak_mb(training_base + act + auxiliary_act_peak + fusion_mb[mode]),
                                *cache_phases.values(),
                                *vae_phases,
                                initialization_peak or 0,
                            )
                        )
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
                        if phase is not None and phase["peak_mb_estimate"] > tight:
                            for path in phase["workspace_reductions_mb"]:
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
                        and max(training_peak, image_encoding["peak_mb_estimate"]) > tight
                    ):
                        memory["suggestions"].append("enable dataset.cache_latents")
                    if preview_decoding is not None and preview_decoding["peak_mb_estimate"] > tight:
                        memory["suggestions"].append("lower sampling.width / sampling.height")
                if tight and training_peak > tight:
                    # Multi-GPU training rejects offloaded checkpoints and block swap.
                    multi_gpu = cfg.loop.gpu_count > 1 or cfg.loop.distributed_strategy == "fsdp"
                    if cfg.memory.activation_checkpointing == "none":
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
                if tight and peak > tight:
                    out["warnings"].append(
                        {
                            "code": "vram.tight",
                            "msg": f"estimated peak {peak:.0f} MB vs {gpu_total_mb:.0f} MB available",
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
