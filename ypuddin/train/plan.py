"""``plan``: validate a config and predict steps / buckets / parameter counts / VRAM without loading weights."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ypuddin.adapters import inject
from ypuddin.config import TrainConfig
from ypuddin.data import BucketManager, expand_items, scan_sources
from ypuddin.models import get_family

DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4, "fp8_e4m3": 1, "fp8_e5m2": 1, "keep": 2, "auto": 2}


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def plan(
    cfg: TrainConfig, *, gpu_total_mb: float | None = None, index_db_path: str | Path | None = None
) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": True, "errors": [], "warnings": []}
    try:
        family = get_family(cfg.model.family)
    except KeyError as e:
        return {"ok": False, "errors": [{"loc": "model.family", "msg": str(e)}], "warnings": []}
    out["errors"] += [{"loc": "model", "msg": m} for m in family.validate_config(cfg.model)]

    # ---- data
    ds = cfg.dataset
    records = []
    if ds.sources:
        try:
            records = scan_sources(ds.sources)
        except FileNotFoundError as e:
            out["errors"].append({"loc": "dataset.sources", "msg": str(e)})
    if not records and ds.sources:
        out["errors"].append({"loc": "dataset.sources", "msg": "no images found"})
    bm = BucketManager(
        ds.resolutions,
        align=family.spec.latent.align,
        step=ds.bucket_step,
        aspect_ratio_limit=ds.aspect_ratio_limit,
        area_tolerance=ds.area_tolerance,
        no_upscale=ds.bucket_no_upscale,
    )
    items = expand_items(records, ds.sources, ds, bm) if records else []
    counts: dict[tuple[int, int], int] = {}
    for it in items:
        counts[it.bucket.key] = counts.get(it.bucket.key, 0) + 1
    batches = sum(math.ceil(n / ds.batch_size) for n in counts.values())
    steps_per_epoch = math.ceil(batches / cfg.loop.grad_accum) if batches else 0
    by_epochs = (cfg.loop.epochs or 10**9) * steps_per_epoch
    total_steps = min(by_epochs, cfg.loop.max_steps or 10**9) if steps_per_epoch else 0
    out.update(
        {
            "images": len(records),
            "items": len(items),
            "captioned": sum(1 for r in records if r.caption_path),
            "buckets": [
                {"w": w, "h": h, "items": n, "batches": math.ceil(n / ds.batch_size)}
                for (w, h), n in sorted(counts.items())
            ],
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "epochs": cfg.loop.epochs,
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
    if any(n < ds.batch_size for n in counts.values()):
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
                aset = inject(
                    backbone, cfg.adapter, presets[cfg.adapter.preset], prefix=family.spec.adapter_prefix
                )
                params = {
                    "base": base_params,
                    "trainable": aset.num_params(),
                    "adapted_layers": len(aset.layers),
                    "by_algo": aset.summary()["by_algo"],
                }
                base_bytes = DTYPE_BYTES[
                    cfg.memory.base_precision if cfg.memory.base_precision != "auto" else cfg.model.dtype
                ]
                weights_mb = base_params * base_bytes / 2**20
                adapter_mb = aset.num_params() * (4 if cfg.adapter.param_dtype == "fp32" else 2) / 2**20
                optimizer_mb = aset.num_params() * 4 * (0.5 if "8bit" in cfg.optimizer.type else 2) / 2**20
                layout = (
                    family.memory_layout_meta(backbone) if hasattr(family, "memory_layout_meta") else None
                )
                act_by_bucket = []
                for (w, h), _n in sorted(counts.items()) or [((r, r), 0) for r in ds.resolutions]:
                    tokens = (w // family.spec.latent.align) * (h // family.spec.latent.align)
                    hidden = getattr(backbone, "dim", None) or getattr(backbone, "model_channels", 2048)
                    n_blocks = len(layout.blocks) if layout else 1
                    ckpt = cfg.memory.activation_checkpointing != "none"
                    per_block = tokens * hidden * 2 * (2 if ckpt else 14)
                    act = per_block * n_blocks * ds.batch_size / 2**20
                    act_by_bucket.append({"w": w, "h": h, "mb": round(act)})
                swapped_mb = 0.0
                if layout and cfg.memory.blocks_to_swap and layout.blocks:
                    swapped_mb = (
                        weights_mb * min(cfg.memory.blocks_to_swap, len(layout.blocks)) / len(layout.blocks)
                    )
                text_mode = ds.text_encoding
                if text_mode == "auto":
                    text_mode = "online" if "online_text" in family.spec.capabilities else "cached"
                text_encoder_mb = 0.0
                if text_mode == "online" and not cfg.memory.offload_text_encoder:
                    text_encoder_mb = family.spec.text.encoder_params * DTYPE_BYTES[cfg.model.dtype] / 2**20
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
                    "suggestions": [],
                }
                if gpu_total_mb and peak > gpu_total_mb * 0.9:
                    if cfg.memory.activation_checkpointing == "none":
                        memory["suggestions"].append("set memory.activation_checkpointing = 'block'")
                    if "block_swap" in family.spec.capabilities and not cfg.memory.blocks_to_swap:
                        memory["suggestions"].append("enable memory.blocks_to_swap")
                    if "8bit" not in cfg.optimizer.type:
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
            out["warnings"].append(
                {"code": "plan.params_failed", "msg": f"could not estimate parameters: {e}"}
            )
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
