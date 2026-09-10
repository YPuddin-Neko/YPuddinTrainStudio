"""Training / validation datasets built from the config, with cached or online latents."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ypuddin.config import CaptionConfig, DatasetConfig, DatasetSourceConfig, TrainConfig
from ypuddin.models import LatentSpec

from .buckets import Bucket, BucketManager
from .cache import LatentCache, build_latent_cache
from .captions import read_caption, transform_caption
from .images import load_mask, load_rgb, pil_to_tensor, to_bucket
from .index import ImageRecord, IndexDB, dataset_fingerprint, scan_sources

log = logging.getLogger(__name__)


@dataclass
class Item:
    record: ImageRecord
    bucket: Bucket
    source: DatasetSourceConfig
    caption_cfg: CaptionConfig
    is_reg: bool
    weight: float


@dataclass
class DataPlan:
    images: int
    items: int
    buckets: list[dict[str, Any]] = field(default_factory=list)
    fingerprint: str = ""
    captioned: int = 0
    validation_images: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "images": self.images,
            "items": self.items,
            "captioned": self.captioned,
            "validation_images": self.validation_images,
            "buckets": self.buckets,
            "fingerprint": self.fingerprint,
        }


def _split_by_hash(records: list[ImageRecord], ratio: float) -> tuple[list[ImageRecord], list[ImageRecord]]:
    if ratio <= 0:
        return records, []
    train, val = [], []
    for r in records:
        frac = int(r.content_hash[:8], 16) / 0xFFFFFFFF
        (val if frac < ratio else train).append(r)
    return train, val


def expand_items(records: list[ImageRecord], sources: list[DatasetSourceConfig], ds: DatasetConfig, bm: BucketManager) -> list[Item]:
    items: list[Item] = []
    for r in records:
        src = sources[r.source_index]
        resolutions = src.resolutions or ds.resolutions
        cap_cfg = src.caption or ds.caption
        for base in resolutions:
            bucket = bm.assign(r.width, r.height, base)
            for _ in range(src.repeats):
                items.append(Item(r, bucket, src, cap_cfg, src.is_reg, src.prior_weight if src.is_reg else 1.0))
    return items


class TrainDataset(Dataset):
    """Returns per-sample dicts; latents come from the cache when present, else raw pixels."""

    def __init__(
        self,
        items: list[Item],
        *,
        latent_spec: LatentSpec,
        latent_cache: LatentCache | None,
        flip: bool,
        masked_loss: bool,
        seed: int,
        deterministic_captions: bool = False,
    ) -> None:
        self.items = items
        self.latent_spec = latent_spec
        self.cache = latent_cache
        self.flip = flip
        self.masked_loss = masked_loss
        self.seed = seed
        self.deterministic = deterministic_captions
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.items)

    def bucket_keys(self) -> list[tuple[int, int]]:
        return [it.bucket.key for it in self.items]

    def _rng(self, index: int) -> random.Random:
        return random.Random((self.seed * 7_919 + self.epoch) * 1_000_003 + index)

    def cache_key(self, item: Item, flip: bool) -> str:
        return LatentCache.key(item.record.content_hash, item.bucket.width, item.bucket.height, self.latent_spec.fingerprint, flip)

    def load_pixels(self, item: Item, flip: bool) -> tuple[Tensor, Tensor | None]:
        im, alpha = load_rgb(item.record.path)
        px = pil_to_tensor(to_bucket(im, item.bucket.width, item.bucket.height, flip=flip))
        mask = None
        if self.masked_loss:
            mask = load_mask(item.record.mask_path, alpha, item.bucket.width, item.bucket.height, flip=flip)
        return px, mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        rng = self._rng(index)
        flip = bool(self.flip and not self.deterministic and rng.random() < 0.5)
        out: dict[str, Any] = {
            "index": index,
            "bucket": item.bucket.key,
            "is_reg": item.is_reg,
            "weight": item.weight,
            "path": item.record.path,
        }
        raw = read_caption(item.record.caption_path, item.source.class_prompt)
        if self.deterministic:
            cap: str | None = raw
        else:
            cap = transform_caption(raw, item.caption_cfg, rng)
        out["caption"] = "" if cap is None else cap
        out["uncond"] = cap is None
        key = self.cache_key(item, flip)
        if self.cache is not None and self.cache.has(key):
            entry = self.cache.get(key)
            out["latents"] = entry["latents"]
            if self.masked_loss and "mask" in entry:
                out["mask"] = entry["mask"].float()
        else:
            px, mask = self.load_pixels(item, flip)
            out["pixels"] = px
            if mask is not None:
                out["mask"] = mask
        return out


def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "index": torch.tensor([s["index"] for s in samples]),
        "caption": [s["caption"] for s in samples],
        "uncond": torch.tensor([s["uncond"] for s in samples]),
        "is_reg": torch.tensor([s["is_reg"] for s in samples]),
        "weight": torch.tensor([s["weight"] for s in samples], dtype=torch.float32),
        "bucket": samples[0]["bucket"],
        "paths": [s["path"] for s in samples],
    }
    if "latents" in samples[0]:
        batch["latents"] = torch.stack([s["latents"] for s in samples])
    else:
        batch["pixels"] = torch.stack([s["pixels"] for s in samples])
    if all("mask" in s for s in samples):
        batch["mask"] = torch.stack([s["mask"] for s in samples])
    return batch


@dataclass
class DataBundle:
    train: TrainDataset
    validation: TrainDataset | None
    plan: DataPlan
    records: list[ImageRecord]
    bucket_manager: BucketManager
    latent_cache: LatentCache | None


def build_data(cfg: TrainConfig, latent_spec: LatentSpec, *, cache_root: str | Path, progress: Callable[[str, int, int], None] | None = None) -> DataBundle:
    ds = cfg.dataset
    index_db = IndexDB(Path(cache_root) / "index.sqlite")
    try:
        records = scan_sources(ds.sources, index_db=index_db, progress=(lambda d, t: progress("index", d, t)) if progress else None)
    finally:
        index_db.close()
    if not records:
        raise ValueError("no images found in dataset sources")
    val_records: list[ImageRecord] = []
    if cfg.validation.enabled:
        if cfg.validation.split_ratio > 0:
            records, val_records = _split_by_hash(records, cfg.validation.split_ratio)
        if cfg.validation.sources:
            extra = scan_sources(cfg.validation.sources)
            val_records.extend(extra)
    bm = BucketManager(
        ds.resolutions,
        align=latent_spec.align,
        step=ds.bucket_step,
        aspect_ratio_limit=ds.aspect_ratio_limit,
        area_tolerance=ds.area_tolerance,
        no_upscale=ds.bucket_no_upscale,
    )
    items = expand_items(records, ds.sources, ds, bm)
    cache = LatentCache(Path(cache_root) / "latents") if ds.cache_latents else None
    train = TrainDataset(items, latent_spec=latent_spec, latent_cache=cache, flip=ds.flip, masked_loss=ds.masked_loss, seed=cfg.loop.seed)
    val = None
    if val_records:
        vsources = list(ds.sources) + list(cfg.validation.sources)
        vitems = expand_items(val_records, vsources, ds, bm)
        # validation: one item per image, first resolution only, no repeats
        seen: set[str] = set()
        uniq = []
        for it in vitems:
            if it.record.content_hash in seen:
                continue
            seen.add(it.record.content_hash)
            uniq.append(it)
        if cfg.validation.max_images:
            uniq = uniq[: cfg.validation.max_images]
        val = TrainDataset(uniq, latent_spec=latent_spec, latent_cache=cache, flip=False, masked_loss=ds.masked_loss, seed=cfg.validation.seed, deterministic_captions=True)
    counts: dict[tuple[int, int, int], int] = {}
    for it in items:
        k = (it.bucket.base, it.bucket.width, it.bucket.height)
        counts[k] = counts.get(k, 0) + 1
    plan = DataPlan(
        images=len(records),
        items=len(items),
        captioned=sum(1 for r in records if r.caption_path),
        validation_images=len(val.items) if val else 0,
        buckets=[{"base": b, "w": w, "h": h, "items": n} for (b, w, h), n in sorted(counts.items())],
        fingerprint=dataset_fingerprint(records, ds.sources),
    )
    return DataBundle(train, val, plan, records, bm, cache)


def cache_latents(bundle: DataBundle, encode: Callable[[Tensor], Tensor], *, device: torch.device | str, batch_size: int = 4, dtype: torch.dtype = torch.bfloat16, progress: Callable[[int, int], None] | None = None) -> int:
    """Encode every (item, flip) combination missing from the cache."""
    assert bundle.latent_cache is not None
    datasets = [bundle.train] + ([bundle.validation] if bundle.validation else [])

    def jobs() -> Iterator[tuple[str, dict[str, Tensor]]]:
        seen: set[str] = set()
        for ds in datasets:
            for item in ds.items:
                for flip in ((False, True) if ds.flip else (False,)):
                    key = ds.cache_key(item, flip)
                    if key in seen:
                        continue
                    seen.add(key)
                    if bundle.latent_cache.has(key):
                        yield key, {}
                        continue
                    px, mask = ds.load_pixels(item, flip)
                    yield key, {"pixels": px, "mask": mask}

    total = sum(len(ds.items) * (2 if ds.flip else 1) for ds in datasets)
    return build_latent_cache(jobs(), bundle.latent_cache, encode, batch_size=batch_size, device=device, dtype=dtype, progress=progress, total=total)
