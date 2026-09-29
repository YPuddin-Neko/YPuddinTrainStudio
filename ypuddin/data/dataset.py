"""Training / validation datasets built from the config, with cached or online latents."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ypuddin.config import CaptionConfig, DatasetConfig, DatasetSourceConfig, TrainConfig
from ypuddin.config.schema import MAX_SIDE
from ypuddin.models import LatentSpec

from .buckets import BUCKET_POLICY, Bucket, BucketManager, crop_offset, fit_crop, fit_pad
from .cache import CacheKey, LatentCache, build_latent_cache, cache_names
from .caption_formats import effective_caption_extension, family_caption_formats, require_caption_format
from .caption_json import StructuredCaption
from .captions import (
    caption_variants_for_cache,
    read_training_caption,
    transform_caption,
    transform_caption_deterministic,
)
from .images import (
    load_alpha,
    load_mask,
    load_rgb,
    pil_to_tensor,
    to_bucket,
    to_native,
    to_padded,
    valid_image_mask,
)
from .index import (
    ImageRecord,
    IndexDB,
    dataset_fingerprint,
    mask_for,
    probe_image,
    record_content_key,
    scan_sources,
)
from .native import native_size

log = logging.getLogger(__name__)


@dataclass
class Item:
    record: ImageRecord
    bucket: Bucket
    source: DatasetSourceConfig
    caption_cfg: CaptionConfig
    is_reg: bool
    weight: float
    native_scale: float | None = None
    image_fit: str = "crop"
    no_upscale: bool = False
    crop_anchor: str = "center"
    # The image's name in the caches: its file name, with a content hash when another image shares it.
    cache_name: str = ""

    @property
    def max_scale(self) -> float | None:
        return self.native_scale if self.native_scale is not None else 1.0 if self.no_upscale else None


def item_latent_key(item: Item, fingerprint: str, flip: bool) -> CacheKey:
    if item.record.color_key_transparency:
        # Color-keyed RGB/grayscale PNGs are composited onto white; never reuse
        # latents cached without color-key compositing.
        fingerprint += "|color-key-white-v1"
    if item.image_fit == "pad":
        rw, rh, *_ = fit_pad(
            item.record.width,
            item.record.height,
            item.bucket.width,
            item.bucket.height,
            max_scale=item.max_scale,
        )
        fingerprint += f"|whole-image-pad-v1:{rw}x{rh}:edge"
    elif item.native_scale is not None:
        rw = round(item.record.width * item.native_scale)
        rh = round(item.record.height * item.native_scale)
        fingerprint += f"|native-crop-v1:{rw}x{rh}"
    if item.image_fit == "crop" and item.crop_anchor != "center":
        fingerprint += f"|crop-anchor-v1:{item.crop_anchor}"
    name = item.cache_name or Path(item.record.path).stem
    return LatentCache.key(
        name, item.record.content_hash, item.bucket.width, item.bucket.height, fingerprint, flip
    )


def item_geometry(item: Item) -> dict[str, Any]:
    """Actual shared pixel geometry for the plan; coordinates are on the resized canvas."""
    w, h = item.bucket.key
    sw, sh = item.record.width, item.record.height
    padding = cropped = 0
    if item.image_fit == "pad":
        rw, rh, left, top, right, bottom = fit_pad(sw, sh, w, h, max_scale=item.max_scale)
        padding = w * h - rw * rh
    elif item.native_scale is not None:
        rw, rh = max(w, round(sw * item.native_scale)), max(h, round(sh * item.native_scale))
        left, top = crop_offset(rw, rh, w, h, item.crop_anchor)
        right, bottom = left + w, top + h
        cropped = rw * rh - w * h
    else:
        rw, rh, left, top, right, bottom = fit_crop(sw, sh, w, h, anchor=item.crop_anchor)
        cropped = rw * rh - w * h
    return {
        "path": item.record.path,
        "source_width": sw,
        "source_height": sh,
        "width": w,
        "height": h,
        "resized_width": rw,
        "resized_height": rh,
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "padding_pixels": padding,
        "cropped_pixels": cropped,
    }


def item_conditioning_geometry(item: Item) -> dict[str, tuple[int, int]]:
    """Size conditioning in (height, width) / (top, left) order.

    Original dimensions are EXIF-oriented, before resizing. Crop coordinates refer to
    the resized image, as in the actual pixel transform; mirroring happens after the
    crop and does not move its source rectangle. Whole-image padding has no crop, so
    its crop origin is (0, 0), not the image's placement offset inside the padding.
    Target dimensions always describe the final canvas, including any padding.
    """
    geometry = item_geometry(item)
    return {
        "original_size": (geometry["source_height"], geometry["source_width"]),
        "crop_top_left": (0, 0) if item.image_fit == "pad" else (geometry["top"], geometry["left"]),
        "target_size": (geometry["height"], geometry["width"]),
    }


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


def training_layout_line(resolution_mode: str, buckets: list[dict[str, Any]]) -> str:
    """How training sizes the images: native layouts or buckets, each size with its item count."""
    ordered = sorted(buckets, key=lambda bucket: (-bucket["items"], bucket["w"], bucket["h"]))
    sizes = ", ".join(f"{bucket['w']}x{bucket['h']} ({bucket['items']})" for bucket in ordered)
    if resolution_mode == "native":
        return f"training layout: native resolution, {len(ordered)} image layouts: {sizes}"
    return f"training layout: buckets, {len(ordered)} buckets: {sizes}"


def _split_by_hash(records: list[ImageRecord], ratio: float) -> tuple[list[ImageRecord], list[ImageRecord]]:
    if ratio <= 0:
        return records, []
    train, val = [], []
    for r in records:
        frac = int(r.content_hash[:8], 16) / 0xFFFFFFFF
        (val if frac < ratio else train).append(r)
    return train, val


def expand_items(
    records: list[ImageRecord], sources: list[DatasetSourceConfig], ds: DatasetConfig, bm: BucketManager
) -> list[Item]:
    items: list[Item] = []
    names = cache_names(records)
    for r in records:
        src = sources[r.source_index]
        resolutions = src.resolutions or ds.resolutions
        # Class-prior captions describe the base class, without the training trigger.
        # An explicit source caption configuration still takes precedence.
        cap_cfg = src.caption if src.caption is not None else CaptionConfig() if src.is_reg else ds.caption
        if (ds.resolution_mode == "native" or ds.image_fit == "pad") and ds.masked_loss and r.mask_path:
            try:
                mask_width, mask_height, _ = probe_image(Path(r.mask_path))
                if (mask_width, mask_height) != (r.width, r.height):
                    raise ValueError(
                        f"mask must match the oriented image dimensions {r.width}x{r.height}; "
                        f"received {mask_width}x{mask_height}"
                    )
            except (OSError, ValueError) as error:
                raise DataConfigError("dataset.masked_loss", f"{Path(r.path).name}: {error}") from error
        for base in [0] if ds.resolution_mode == "native" else resolutions:
            try:
                size = (
                    native_size(
                        r.width,
                        r.height,
                        align=bm.align,
                        max_pixels=ds.native_max_pixels,
                        max_side=ds.native_max_side,
                        overflow=ds.native_overflow,
                        image_fit=ds.image_fit,
                    )
                    if ds.resolution_mode == "native"
                    else None
                )
                bucket = (
                    Bucket(size.width, size.height, 0)
                    if size
                    else bm.assign(r.width, r.height, base, image_fit=ds.image_fit)
                )
            except ValueError as error:
                field = "native_max_pixels" if ds.resolution_mode == "native" else "bucket_no_upscale"
                raise DataConfigError(f"dataset.{field}", f"{Path(r.path).name}: {error}") from error
            for _ in range(src.repeats):
                items.append(
                    Item(
                        r,
                        bucket,
                        src,
                        cap_cfg,
                        src.is_reg,
                        src.prior_weight if src.is_reg else 1.0,
                        size.scale if size else None,
                        ds.image_fit,
                        ds.bucket_no_upscale,
                        ds.crop_anchor,
                        names[r.path],
                    )
                )
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
        # per-item caption variants when text encodings are pre-cached (None = transform online)
        self.caption_variants: list[list[str]] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def raw_caption(self, item: Item) -> str | StructuredCaption:
        return read_training_caption(
            item.record.caption_path, item.source.class_prompt, require_known_format=True
        )

    def use_cached_captions(self) -> list[tuple[str, str]]:
        """Restrict every item to a bounded, deterministic set of caption variants; return ``(cache name, caption)``.

        Needed when text encodings are cached up front: shuffle / tag dropout / wildcards would
        otherwise produce captions that were never encoded. The variant set only depends on the
        seed, the item index and its raw caption, so a resumed run sees exactly the same captions.
        """
        variants: list[list[str]] = []
        for index, item in enumerate(self.items):
            raw = self.raw_caption(item)
            if self.deterministic:
                variants.append([transform_caption_deterministic(raw, item.caption_cfg)])
            else:
                variants.append(
                    caption_variants_for_cache(
                        raw,
                        item.caption_cfg,
                        item.caption_cfg.cache_variants,
                        seed=self.seed * 1_000_003 + index,
                    )
                )
        self.caption_variants = variants
        return sorted(
            {
                (item.cache_name, caption)
                for item, captions in zip(self.items, variants, strict=True)
                for caption in captions
            }
        )

    def _caption(self, index: int, item: Item, rng: random.Random) -> str | None:
        if self.caption_variants is not None:
            if (
                not self.deterministic
                and item.caption_cfg.caption_dropout > 0
                and rng.random() < item.caption_cfg.caption_dropout
            ):
                return None
            return rng.choice(self.caption_variants[index])
        raw = self.raw_caption(item)
        if self.deterministic:
            return transform_caption_deterministic(raw, item.caption_cfg)
        return transform_caption(raw, item.caption_cfg, rng)

    def __len__(self) -> int:
        return len(self.items)

    def bucket_keys(self) -> list[tuple[int, int]]:
        return [it.bucket.key for it in self.items]

    def _rng(self, index: int) -> random.Random:
        return random.Random((self.seed * 7_919 + self.epoch) * 1_000_003 + index)

    def cache_key(self, item: Item, flip: bool) -> CacheKey:
        return item_latent_key(item, self.latent_spec.fingerprint, flip)

    def load_pixels(
        self, item: Item, flip: bool, *, include_mask: bool = True
    ) -> tuple[Tensor, Tensor | None]:
        im, alpha = load_rgb(item.record.path)
        if item.image_fit == "pad":
            fitted = to_padded(im, item.bucket.width, item.bucket.height, max_scale=item.max_scale, flip=flip)
        else:
            fitted = (
                to_bucket(im, item.bucket.width, item.bucket.height, flip=flip, crop_anchor=item.crop_anchor)
                if item.native_scale is None
                else to_native(
                    im, item.bucket.width, item.bucket.height,
                    scale=item.native_scale, flip=flip, crop_anchor=item.crop_anchor,
                )
            )
        px = pil_to_tensor(fitted)
        mask = None
        if include_mask:
            mask = self._image_mask(item, flip, alpha)
        return px, mask

    def _image_mask(self, item: Item, flip: bool, alpha) -> Tensor | None:
        mask = None
        if self.masked_loss:
            mask = load_mask(
                mask_for(Path(item.record.path)),
                alpha,
                item.bucket.width,
                item.bucket.height,
                flip=flip,
                native_scale=item.native_scale,
                image_fit=item.image_fit,
                max_scale=item.max_scale,
                source_size=(item.record.width, item.record.height),
                crop_anchor=item.crop_anchor,
            )
        if item.image_fit == "pad":
            valid = valid_image_mask(
                item.record.width,
                item.record.height,
                item.bucket.width,
                item.bucket.height,
                max_scale=item.max_scale,
                flip=flip,
            )
            mask = valid if mask is None else mask * valid
        return mask

    def current_mask(self, item: Item, flip: bool) -> Tensor | None:
        path = mask_for(Path(item.record.path)) if self.masked_loss else None
        alpha = (
            load_alpha(item.record.path)
            if self.masked_loss and path is None and item.record.has_alpha
            else None
        )
        return self._image_mask(item, flip, alpha)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        rng = self._rng(index)
        flip = bool(self.flip and not self.deterministic and rng.random() < 0.5)
        out: dict[str, Any] = {
            "index": index,
            "bucket": item.bucket.key,
            "geometry": item_conditioning_geometry(item),
            "is_reg": item.is_reg,
            "weight": item.weight,
            "path": item.record.path,
            "cache_name": item.cache_name,
        }
        cap = self._caption(index, item, rng)
        out["caption"] = "" if cap is None else cap
        out["uncond"] = cap is None
        latents = self.cache.load(self.cache_key(item, flip)) if self.cache is not None else None
        if latents is not None:
            out["latents"] = latents
            # Ignore masks in pre-v2 cache entries. Sidecars can change independently of the
            # image/VAE key, including being added after an unmasked pre-cache job.
            if self.masked_loss or item.image_fit == "pad":
                mask = self.current_mask(item, flip)
                if mask is not None:
                    out["mask"] = mask
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
        "geometry": {
            key: torch.tensor([s["geometry"][key] for s in samples], dtype=torch.int64)
            for key in ("original_size", "crop_top_left", "target_size")
        },
        "paths": [s["path"] for s in samples],
        "cache_names": [s.get("cache_name", "") for s in samples],
    }
    cached = ["latents" in s for s in samples]
    if all(cached):
        batch["latents"] = torch.stack([s["latents"] for s in samples])
    elif not any(cached):
        batch["pixels"] = torch.stack([s["pixels"] for s in samples])
    else:
        # A cache file another run rewrote while this one trained: the missing images are encoded again.
        batch["partial_latents"] = [s.get("latents") for s in samples]
        batch["pixels"] = {index: s["pixels"] for index, s in enumerate(samples) if "pixels" in s}
    if any("mask" in s for s in samples):
        reference = next(s["mask"] for s in samples if "mask" in s)
        batch["mask"] = torch.stack([s.get("mask", torch.ones_like(reference)) for s in samples])
    return batch


@dataclass
class DataBundle:
    train: TrainDataset
    validation: TrainDataset | None
    plan: DataPlan
    records: list[ImageRecord]
    bucket_manager: BucketManager
    latent_cache: LatentCache | None


class DataConfigError(ValueError):
    """A data preflight failure that the service can associate with a configuration field."""

    def __init__(self, loc: str, message: str):
        super().__init__(message)
        self.loc = loc


@dataclass
class DataLayout:
    records: list[ImageRecord]
    validation_records: list[ImageRecord]
    items: list[Item]
    validation_items: list[Item]
    sources: list[DatasetSourceConfig]
    bucket_manager: BucketManager


def prepare_data_layout(
    cfg: TrainConfig,
    latent_spec: LatentSpec,
    *,
    index_db: IndexDB | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> DataLayout:
    """Shared preflight and train data selection; does not load models or create tensor caches."""
    ds = cfg.dataset
    caption_formats = family_caption_formats(cfg.model.family)
    if not ds.sources:
        raise DataConfigError("dataset.sources", "at least one training dataset source is required")
    extra_sources = list(cfg.validation.sources) if cfg.validation.enabled else []
    sources = list(ds.sources) + extra_sources
    resolutions = set(ds.resolutions)
    for i, src in enumerate(sources):
        prefix = (
            f"dataset.sources.{i}" if i < len(ds.sources) else f"validation.sources.{i - len(ds.sources)}"
        )
        if src.resolutions is not None:
            if not src.resolutions or any(r < 32 or r > MAX_SIDE for r in src.resolutions):
                raise DataConfigError(f"{prefix}.resolutions", f"resolutions must be within [32, {MAX_SIDE}]")
            resolutions.update(src.resolutions)
        if not Path(src.path).expanduser().is_dir():
            raise DataConfigError(f"{prefix}.path", f"dataset source not found: {src.path}")
    try:
        bm = BucketManager(
            [] if ds.resolution_mode == "native" else sorted(resolutions, reverse=True),
            align=latent_spec.align,
            step=latent_spec.align if ds.resolution_mode == "native" else ds.bucket_step,
            aspect_ratio_limit=ds.aspect_ratio_limit,
            area_tolerance=ds.area_tolerance,
            no_upscale=ds.bucket_no_upscale,
        )
    except ValueError as e:
        raise DataConfigError("dataset.bucket_step", str(e)) from e

    def scan(group: list[DatasetSourceConfig], prefix: str, offset: int = 0) -> list[ImageRecord]:
        try:
            records = scan_sources(
                [
                    source.model_copy(
                        update={
                            "caption_ext": effective_caption_extension(source.caption_ext, caption_formats)
                        }
                    )
                    for source in group
                ],
                index_db=index_db,
                progress=(lambda d, t: progress("index", d, t)) if progress else None,
            )
            for record in records:
                if record.caption_path:
                    try:
                        require_caption_format(record.caption_path, caption_formats, cfg.model.family)
                        if Path(record.caption_path).suffix.lower() == ".json":
                            read_training_caption(record.caption_path, require_known_format=True)
                    except (ValueError, OSError) as error:
                        raise DataConfigError(
                            f"{prefix}.{record.source_index}.caption_ext", str(error)
                        ) from error
            for index, source in enumerate(group):
                if source.caption_ext.lower() != "auto":
                    try:
                        require_caption_format(
                            Path(source.path) / f"*{source.caption_ext}", caption_formats, cfg.model.family
                        )
                    except ValueError as error:
                        raise DataConfigError(f"{prefix}.{index}.caption_ext", str(error)) from error
            return sorted(
                (replace(record, source_index=record.source_index + offset) for record in records),
                key=record_content_key,
            )
        except OSError as e:
            raise DataConfigError(prefix, str(e)) from e

    records = scan(ds.sources, "dataset.sources")
    if not records:
        raise DataConfigError("dataset.sources", "no readable images found in dataset sources")
    val_records: list[ImageRecord] = []
    if cfg.validation.enabled:
        priors = [record for record in records if sources[record.source_index].is_reg]
        examples = [record for record in records if not sources[record.source_index].is_reg]
        examples, held_out = _split_by_hash(examples, cfg.validation.split_ratio)
        records = sorted(examples + priors, key=record_content_key)
        explicit = scan(extra_sources, "validation.sources", len(ds.sources)) if extra_sources else []
        # Explicit validation source configuration wins when the same image is also in a split.
        val_records = explicit + held_out
        validation_hashes = {record.content_hash for record in val_records}
        records = [record for record in records if record.content_hash not in validation_hashes]
    if not records:
        raise DataConfigError("dataset.sources", "no training images remain after validation exclusion/split")
    items = expand_items(records, sources, ds, bm)
    if not items:
        raise DataConfigError("dataset.sources", "training dataset has no items")
    validation_items: list[Item] = []
    seen: set[str] = set()
    for item in expand_items(val_records, sources, ds, bm):
        if item.record.content_hash not in seen:
            seen.add(item.record.content_hash)
            validation_items.append(item)
    if cfg.validation.max_images:
        validation_items = validation_items[: cfg.validation.max_images]
    return DataLayout(records, val_records, items, validation_items, sources, bm)


def build_data(
    cfg: TrainConfig,
    latent_spec: LatentSpec,
    *,
    cache_root: str | Path,
    progress: Callable[[str, int, int], None] | None = None,
) -> DataBundle:
    ds = cfg.dataset
    index_db = IndexDB(Path(cache_root) / "index.sqlite")
    try:
        layout = prepare_data_layout(
            cfg,
            latent_spec,
            index_db=index_db,
            progress=progress,
        )
    finally:
        index_db.close()
    records, items, bm = layout.records, layout.items, layout.bucket_manager
    cache = LatentCache(Path(cache_root) / "latents") if ds.cache_latents else None
    train = TrainDataset(
        items,
        latent_spec=latent_spec,
        latent_cache=cache,
        flip=ds.flip,
        masked_loss=ds.masked_loss,
        seed=cfg.loop.seed,
    )
    val = None
    if layout.validation_items:
        val = TrainDataset(
            layout.validation_items,
            latent_spec=latent_spec,
            latent_cache=cache,
            flip=False,
            masked_loss=ds.masked_loss,
            seed=cfg.validation.seed,
            deterministic_captions=True,
        )
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
        fingerprint=dataset_fingerprint(
            records + [item.record for item in layout.validation_items],
            layout.sources,
            settings={
                **({"bucket_policy": BUCKET_POLICY} if ds.resolution_mode == "bucket" else {}),
                "dataset": ds.model_dump(
                    mode="json",
                    exclude={"sources", "cache_dir", "num_workers"}
                    | ({"image_fit"} if ds.image_fit == "crop" else set())
                    | ({"crop_anchor"} if ds.crop_anchor == "center" or ds.image_fit == "pad" else set())
                    | (
                        {"resolution_mode", "native_max_pixels", "native_max_side", "native_overflow"}
                        if ds.resolution_mode == "bucket"
                        else set()
                    ),
                ),
                "validation": cfg.validation.model_dump(mode="json", exclude={"sources"}),
                "validation_content": [item.record.content_hash for item in layout.validation_items],
            },
        ),
    )
    return DataBundle(train, val, plan, records, bm, cache)


def cache_latents(
    bundle: DataBundle,
    encode: Callable[[Tensor], Tensor],
    *,
    device: torch.device | str,
    batch_size: int = 4,
    dtype: torch.dtype = torch.bfloat16,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Encode every (item, flip) combination missing from the cache."""
    assert bundle.latent_cache is not None
    datasets = [bundle.train] + ([bundle.validation] if bundle.validation else [])

    def jobs() -> Iterator[tuple[CacheKey, dict[str, Tensor]]]:
        seen: set[CacheKey] = set()
        for ds in datasets:
            for item in ds.items:
                for flip in (False, True) if ds.flip else (False,):
                    key = ds.cache_key(item, flip)
                    if key in seen:
                        continue
                    seen.add(key)
                    if bundle.latent_cache.has(key):
                        yield key, {}
                        continue
                    px, _mask = ds.load_pixels(item, flip, include_mask=False)
                    yield key, {"pixels": px}

    total = sum(len(ds.items) * (2 if ds.flip else 1) for ds in datasets)
    return build_latent_cache(
        jobs(),
        bundle.latent_cache,
        encode,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        progress=progress,
        total=total,
    )
