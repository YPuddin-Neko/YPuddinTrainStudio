"""``plan``: validate a config and predict steps / buckets / parameter counts / VRAM without loading weights."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from pydantic import ValidationError
from torch import nn

from ypuddin.adapters import inject
from ypuddin.config import TrainConfig
from ypuddin.data import IndexDB
from ypuddin.data.dataset import DataConfigError, prepare_data_layout
from ypuddin.data.native import NativeBatchSampler, microbatch_indices, native_size
from ypuddin.models import get_family

DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4, "fp8_e4m3": 1, "fp8_e5m2": 1, "keep": 2, "auto": 2}


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


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
        return {
            "ok": False,
            "errors": [
                {"loc": ".".join(str(part) for part in error["loc"]), "msg": error["msg"]}
                for error in e.errors()
            ],
            "warnings": [],
        }
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
            cfg.dataset.masked_loss and "masked_loss" not in caps,
            "dataset.masked_loss",
            "family does not support masked loss",
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
    records = []
    items = []
    validation_images = 0
    index = None
    try:
        index = IndexDB(index_db_path) if index_db_path else None
        layout = prepare_data_layout(cfg, family.spec.latent, index_db=index)
        records, items = layout.records, layout.items
        validation_images = len(layout.validation_items)
    except DataConfigError as e:
        out["errors"].append({"loc": e.loc, "msg": str(e)})
    except (OSError, ValueError) as e:
        out["errors"].append({"loc": "dataset.sources", "msg": str(e)})
    finally:
        if index is not None:
            index.close()
    counts: dict[tuple[int, int], int] = {}
    for it in items:
        counts[it.bucket.key] = counts.get(it.bucket.key, 0) + 1
    native = ds.resolution_mode == "native"
    batches = (
        math.ceil(len(items) / ds.batch_size)
        if native
        else sum(math.ceil(n / ds.batch_size) for n in counts.values())
    )
    steps_per_epoch = math.ceil(batches / cfg.loop.grad_accum) if batches else 0
    by_epochs = (cfg.loop.epochs or 10**9) * steps_per_epoch
    total_steps = min(by_epochs, cfg.loop.max_steps or 10**9) if steps_per_epoch else 0
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
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "epochs": cfg.loop.epochs,
        }
    )
    if native:
        shape_keys = [item.bucket.key for item in items]
        forward_counts: dict[tuple[int, int], int] = {}
        for batch in NativeBatchSampler(shape_keys, ds.batch_size, seed=cfg.loop.seed).plan():
            shapes = [shape_keys[index] for index in batch]
            for group in microbatch_indices(shapes, ds.native_max_pixels):
                key = shapes[group[0]]
                forward_counts[key] = forward_counts.get(key, 0) + 1
        for bucket in out["buckets"]:
            bucket["batches"] = forward_counts.get((bucket["w"], bucket["h"]), 0)
        resized = (
            sum(
                native_size(
                    record.width,
                    record.height,
                    align=family.spec.latent.align,
                    max_pixels=ds.native_max_pixels,
                    max_side=ds.native_max_side,
                    overflow=ds.native_overflow,
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
            "alignment": family.spec.latent.align,
            "batch_size": ds.batch_size,
            "forward_groups": sum(forward_counts.values()),
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
            presets = family.presets()
            if cfg.adapter.preset not in presets:
                out["errors"].append(
                    {"loc": "adapter.preset", "msg": f"unknown preset; available: {sorted(presets)}"}
                )
            else:
                error_loc = "adapter"
                aset = inject(
                    backbone, cfg.adapter, presets[cfg.adapter.preset], prefix=family.spec.adapter_prefix
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
                base_bytes = (
                    4
                    if device_type in ("cpu", "mps")
                    else DTYPE_BYTES[
                        cfg.memory.base_precision if cfg.memory.base_precision != "auto" else cfg.model.dtype
                    ]
                )
                weights_mb = base_params * base_bytes / 2**20
                adapter_mb = aset.num_params() * (4 if cfg.adapter.param_dtype == "fp32" else 2) / 2**20
                optimizer_mb = aset.num_params() * 4 * (0.5 if "8bit" in cfg.optimizer.type else 2) / 2**20
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
                    hidden = getattr(backbone, "dim", None) or getattr(backbone, "model_channels", 2048)
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
                if layout and cfg.memory.blocks_to_swap and layout.blocks and device_type != "mps":
                    swapped_mb = (
                        weights_mb * min(cfg.memory.blocks_to_swap, len(layout.blocks)) / len(layout.blocks)
                    )
                text_mode = ds.text_encoding
                if text_mode == "auto":
                    text_mode = "online" if "online_text" in family.spec.capabilities else "cached"
                text_encoder_mb = 0.0
                if text_mode == "online" and not cfg.memory.offload_text_encoder:
                    text_encoder_mb = family.spec.text.encoder_params * DTYPE_BYTES[effective_dtype] / 2**20
                peak = (
                    weights_mb
                    - swapped_mb
                    + text_encoder_mb
                    + adapter_mb
                    + optimizer_mb
                    + (max(a["mb"] for a in act_by_bucket) if act_by_bucket else 0)
                    + 512
                )
                memory = {
                    "weights_mb": round(weights_mb),
                    "swapped_mb": round(swapped_mb),
                    "text_encoder_mb": round(text_encoder_mb),
                    "adapter_mb": round(adapter_mb, 1),
                    "optimizer_mb": round(optimizer_mb, 1),
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
