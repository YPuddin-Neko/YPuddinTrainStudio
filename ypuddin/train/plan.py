"""``plan``: validate a config and predict steps / buckets / parameter counts / VRAM without loading weights."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pydantic import TypeAdapter, ValidationError
from torch import nn

from ypuddin.adapters import inject
from ypuddin.adapters.frozen import FrozenLinear
from ypuddin.config import DatasetConfig, LoopConfig, TrainConfig, ValidationConfig
from ypuddin.data import IndexDB
from ypuddin.data.dataset import DataConfigError, item_geometry, prepare_data_layout
from ypuddin.data.native import NativeBatchSampler, microbatch_indices, native_size
from ypuddin.models import get_family
from ypuddin.models.base import LatentSpec

DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4, "fp8_e4m3": 1, "fp8_e5m2": 1, "keep": 2, "auto": 2}


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _frozen_storage_bytes(module: nn.Module) -> int:
    tensors = {id(t): t for t in (*module.parameters(), *module.buffers()) if not t.requires_grad}
    return sum(t.numel() * t.element_size() for t in tensors.values())


@dataclass(frozen=True)
class _LayoutInputs:
    """Only the validated fields consumed by prepare_data_layout, never a training config."""

    dataset: DatasetConfig
    validation: ValidationConfig


def _append_data_plan(
    out: dict[str, Any],
    cfg: TrainConfig | _LayoutInputs,
    latent: LatentSpec,
    *,
    loop: LoopConfig | None,
    seed: int | None,
    index_db_path: str | Path | None,
) -> dict[tuple[int, int], int]:
    """Share exact image selection and geometry between full plans and incomplete drafts."""
    ds = cfg.dataset
    records = []
    items = []
    validation_images = 0
    index = None
    layout = None
    try:
        index = IndexDB(index_db_path) if index_db_path else None
        layout = prepare_data_layout(cfg, latent, index_db=index)
        records, items = layout.records, layout.items
        validation_images = len(layout.validation_items)
    except DataConfigError as e:
        out["errors"].append({"loc": e.loc, "msg": str(e)})
    except (OSError, ValueError) as e:
        out["errors"].append({"loc": "dataset.sources", "msg": str(e)})
    finally:
        if index is not None:
            index.close()
    if layout is None and isinstance(cfg, _LayoutInputs):
        return {}
    counts: dict[tuple[int, int], int] = {}
    for it in items:
        counts[it.bucket.key] = counts.get(it.bucket.key, 0) + 1
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
            "validation_images": validation_images,
            "buckets": [
                {"w": w, "h": h, "items": n, "batches": math.ceil(n / ds.batch_size)}
                for (w, h), n in sorted(counts.items())
            ],
        }
    )
    if loop is not None:
        steps_per_epoch = math.ceil(batches / loop.grad_accum) if batches else 0
        by_epochs = (loop.epochs or 10**9) * steps_per_epoch
        total_steps = min(by_epochs, loop.max_steps or 10**9) if steps_per_epoch else 0
        out.update(steps_per_epoch=steps_per_epoch, total_steps=total_steps, epochs=loop.epochs)
    geometry = {}
    padded_images, cropped_images = set(), set()
    padding_pixels = total_pixels = 0
    for item in items:
        key = (item.record.path, item.bucket.key, item.max_scale)
        if key not in geometry:
            geometry[key] = item_geometry(item)
        entry = geometry[key]
        total_pixels += item.bucket.area
        padding_pixels += entry["padding_pixels"]
        if entry["padding_pixels"]:
            padded_images.add(item.record.path)
        if entry["cropped_pixels"]:
            cropped_images.add(item.record.path)
    out["image_fit"] = {
        "mode": ds.image_fit,
        "padded_images": len(padded_images),
        "cropped_images": len(cropped_images),
        "padding_pixels": padding_pixels,
        "total_pixels": total_pixels,
        "padding_fraction": padding_pixels / total_pixels if total_pixels else 0.0,
        "total_shapes": len(geometry),
        "truncated": len(geometry) > 100,
        "items": list(geometry.values())[:100],
    }
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
        if seed is not None:
            for batch in NativeBatchSampler(shape_keys, ds.batch_size, seed=seed).plan():
                shapes = [shape_keys[index] for index in batch]
                for group in microbatch_indices(shapes, ds.native_max_pixels):
                    key = shapes[group[0]]
                    forward_counts[key] = forward_counts.get(key, 0) + 1
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
                    max_pixels=ds.native_max_pixels,
                    max_side=ds.native_max_side,
                    overflow=ds.native_overflow,
                    image_fit=ds.image_fit,
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
            "max_pixels": ds.native_max_pixels,
            "alignment": latent.align,
            "batch_size": ds.batch_size,
            "forward_groups": sum(forward_counts.values()) if seed is not None else None,
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
    raw: dict[str, Any], out: dict[str, Any], *, index_db_path: str | Path | None
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
            for entry in error.errors():
                issue = {"loc": ".".join([name, *(str(part) for part in entry["loc"])]), "msg": entry["msg"]}
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
        # A bad stop condition or accumulation value cannot produce an honest step estimate.
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
        _LayoutInputs(**fields),
        family.spec.latent,
        loop=loop,
        seed=seed,
        index_db_path=index_db_path,
    )


def plan(
    cfg: TrainConfig | dict[str, Any],
    *,
    gpu_total_mb: float | None = None,
    index_db_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Plan without loading weights. None is an offline target-hardware estimate.

    Pass the execution device from CLI/service to enforce hardware constraints and account
    for CPU/MPS fp32 execution. This does not require target CUDA hardware to be locally present.
    """
    out: dict[str, Any] = {"ok": True, "errors": [], "warnings": []}
    try:
        cfg = TrainConfig.model_validate(cfg)
    except ValidationError as e:
        out.update(
            ok=False,
            errors=[
                {"loc": ".".join(str(part) for part in error["loc"]), "msg": error["msg"]}
                for error in e.errors()
            ],
        )
        raw = cfg.to_dict() if isinstance(cfg, TrainConfig) else cfg
        if isinstance(raw, dict):
            _preview_invalid_config(raw, out, index_db_path=index_db_path)
        return out
    try:
        family = get_family(cfg.model.family)
    except KeyError as e:
        return {"ok": False, "errors": [{"loc": "model.family", "msg": str(e)}], "warnings": []}
    try:
        device_type = torch.device(device).type if device is not None else None
    except (RuntimeError, ValueError) as e:
        return {"ok": False, "errors": [{"loc": "device", "msg": str(e)}], "warnings": []}
    if device_type not in (None, "cpu", "mps", "cuda"):
        out["errors"].append({"loc": "device", "msg": "only cpu, mps and cuda execution are supported"})
    caps = family.spec.capabilities
    checks = [
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
            cfg.dataset.text_encoding == "online" and "online_text" not in caps,
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
            device_type in ("cpu", "mps") and "8bit" in cfg.optimizer.type.lower(),
            "optimizer.type",
            "8-bit optimizers require CUDA",
        ),
    ]
    out["errors"].extend({"loc": loc, "msg": message} for failed, loc, message in checks if failed)
    out["errors"].extend(family.training_options_errors(cfg))
    effective_dtype = "fp32" if device_type in ("cpu", "mps") else cfg.model.dtype
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
    counts = _append_data_plan(
        out,
        cfg,
        family.spec.latent,
        loop=cfg.loop,
        seed=cfg.loop.seed,
        index_db_path=index_db_path,
    )

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
            if cfg.adapter.preset not in presets:
                out["errors"].append(
                    {"loc": "adapter.preset", "msg": f"unknown preset; available: {sorted(presets)}"}
                )
            else:
                error_loc = "adapter"
                aset = inject(
                    backbone,
                    cfg.adapter,
                    presets[cfg.adapter.preset],
                    prefix=family.spec.adapter_prefix,
                    base_precision=cfg.memory.base_precision
                    if cfg.memory.base_precision != "auto"
                    else "keep",
                )
                params = {
                    "base": base_params,
                    "trainable": aset.num_params(),
                    "adapted_layers": len(aset.layers),
                    "by_algo": aset.summary()["by_algo"],
                }
                if params["trainable"] == 0:
                    out["errors"].append(
                        {"loc": "adapter", "msg": "adapter rules select no trainable parameters"}
                    )
                error_loc = "memory"
                # A loader may retain native FP8 even when model.dtype is BF16.
                # Explicit base_precision applies only to selected adapter targets.
                weights_mb = _frozen_storage_bytes(backbone) / 2**20
                adapter_mb = aset.num_params() * (4 if cfg.adapter.param_dtype == "fp32" else 2) / 2**20
                optimizer_mb = aset.num_params() * 4 * (0.5 if "8bit" in cfg.optimizer.type else 2) / 2**20
                gradients_mb = adapter_mb
                compensation_mb = adapter_mb if cfg.optimizer.kahan else 0.0
                ema_mb = adapter_mb if cfg.loop.ema else 0.0
                layout = (
                    family.memory_layout_meta(backbone) if hasattr(family, "memory_layout_meta") else None
                )
                if layout and cfg.memory.blocks_to_swap > len(layout.blocks):
                    out["errors"].append(
                        {
                            "loc": "memory.blocks_to_swap",
                            "msg": f"cannot swap more than {len(layout.blocks)} blocks",
                        }
                    )
                act_by_bucket = []
                for (w, h), _n in sorted(counts.items()) or [((r, r), 0) for r in ds.resolutions]:
                    tokens = (w // family.spec.latent.align) * (h // family.spec.latent.align)
                    tokens = family.training_tokens_for_plan(tokens)
                    hidden = (
                        getattr(backbone, "dim", None)
                        or getattr(backbone, "model_channels", None)
                        or getattr(getattr(backbone, "config", None), "features", 2048)
                    )
                    n_blocks = len(layout.blocks) if layout else 1
                    ckpt = cfg.memory.activation_checkpointing != "none"
                    activation_bytes = (
                        4
                        if device_type in ("cpu", "mps")
                        else DTYPE_BYTES[
                            cfg.model.dtype if cfg.loop.mixed_precision == "no" else cfg.loop.mixed_precision
                        ]
                    )
                    per_block = tokens * hidden * activation_bytes * (2 if ckpt else 14)
                    forward_batch = (
                        min(ds.batch_size, max(1, ds.native_max_pixels // (w * h)))
                        if native
                        else ds.batch_size
                    )
                    act = per_block * n_blocks * forward_batch / 2**20
                    act_by_bucket.append({"w": w, "h": h, "mb": round(act)})
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
                text_mode = ds.text_encoding
                if text_mode == "auto":
                    text_mode = "online" if "online_text" in family.spec.capabilities else "cached"
                text_encoder_mb = 0.0
                if text_mode == "online" and not cfg.memory.offload_text_encoder:
                    text_encoder_mb = family.spec.text.encoder_params * DTYPE_BYTES[effective_dtype] / 2**20
                training_peak = (
                    weights_mb
                    - swapped_mb
                    + text_encoder_mb
                    + adapter_mb
                    + optimizer_mb
                    + gradients_mb
                    + compensation_mb
                    + ema_mb
                    + swap_staging_mb
                    + dequant_mb
                    + (max(a["mb"] for a in act_by_bucket) if act_by_bucket else 0)
                    + 512
                )
                cache_phases = family.cache_memory_estimate(cfg, compute_dtype)
                peak = max(training_peak, *cache_phases.values()) if cache_phases else training_peak
                memory = {
                    "weights_mb": round(weights_mb),
                    "swapped_mb": round(swapped_mb),
                    "text_encoder_mb": round(text_encoder_mb),
                    "adapter_mb": round(adapter_mb, 1),
                    "optimizer_mb": round(optimizer_mb, 1),
                    "gradients_mb": round(gradients_mb, 1),
                    "compensation_mb": round(compensation_mb, 1),
                    "ema_mb": round(ema_mb, 1),
                    "swap_staging_mb": round(swap_staging_mb, 1),
                    "dequant_mb": round(dequant_mb, 1),
                    "training_peak_mb_estimate": round(training_peak),
                    "cache_phase_peak_mb_estimates": {
                        key: round(value) for key, value in cache_phases.items()
                    },
                    "activations_mb_by_bucket": act_by_bucket,
                    "peak_mb_estimate": round(peak),
                    "gpu_total_mb": gpu_total_mb,
                    "heuristic": True,
                    "device": str(device) if device is not None else None,
                    "effective_dtype": effective_dtype,
                    "suggestions": [],
                }
                if gpu_total_mb and peak > gpu_total_mb * 0.9:
                    if cfg.memory.activation_checkpointing == "none":
                        memory["suggestions"].append("set memory.activation_checkpointing = 'block'")
                    if (
                        device_type != "mps"
                        and "block_swap" in family.spec.capabilities
                        and not cfg.memory.blocks_to_swap
                    ):
                        memory["suggestions"].append("enable memory.blocks_to_swap")
                    if device_type in (None, "cuda") and "8bit" not in cfg.optimizer.type:
                        memory["suggestions"].append("use optimizer.type = 'adamw8bit'")
                    if text_encoder_mb:
                        memory["suggestions"].append(
                            "set dataset.text_encoding = 'cached' (frees the text encoder)"
                        )
                    out["warnings"].append(
                        {
                            "code": "vram.tight",
                            "msg": f"estimated peak {peak:.0f} MB vs {gpu_total_mb:.0f} MB available",
                        }
                    )
        except Exception as e:  # noqa: BLE001
            out["errors"].append({"loc": error_loc, "msg": f"could not prepare model/adapter plan: {e}"})
    out["params"] = params
    out["memory"] = memory
    out["text_encoding"] = (
        "online"
        if (cfg.dataset.text_encoding == "auto" and "online_text" in family.spec.capabilities)
        or cfg.dataset.text_encoding == "online"
        else "cached"
    )
    out["ok"] = not out["errors"]
    return out
