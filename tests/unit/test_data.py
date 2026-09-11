import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest
import torch
from PIL import Image

from ypuddin.config import CaptionConfig, DatasetSourceConfig, TrainConfig
from ypuddin.data import (
    BucketBatchSampler,
    BucketManager,
    IndexDB,
    LatentCache,
    build_data,
    cache_latents,
    collate,
    scan_sources,
    transform_caption,
)
from ypuddin.models import get_family


def test_bucket_manager_generates_reasonable_buckets():
    bm = BucketManager([1024], align=16, aspect_ratio_limit=2.0, area_tolerance=0.1)
    bs = bm.buckets[1024]
    assert any(b.key == (1024, 1024) for b in bs)
    assert all(b.width % 16 == 0 and b.height % 16 == 0 for b in bs)
    assert all(0.9 * 1024**2 <= b.area <= 1.1 * 1024**2 or b.key == (1024, 1024) for b in bs)
    assert all(max(b.aspect, 1 / b.aspect) <= 2.0 + 1e-9 for b in bs)
    assert 20 <= len(bs) <= 60
    tall = bm.assign(600, 1200, 1024)
    assert tall.height > tall.width
    wide = bm.assign(1920, 1080, 1024)
    assert abs(wide.aspect - 16 / 9) < 0.15


def test_bucket_no_upscale_shrinks():
    bm = BucketManager([1024], align=16, no_upscale=True)
    b = bm.assign(256, 256, 1024)
    assert b.key == (256, 256)


def test_scan_sources_and_index_cache(image_dataset, tmp_path):
    src = DatasetSourceConfig(path=str(image_dataset))
    db = IndexDB(tmp_path / "idx.sqlite")
    recs = scan_sources([src], index_db=db)
    assert len(recs) == 12
    assert all(r.caption_path for r in recs)
    assert sum(r.has_alpha for r in recs) == 2
    recs2 = scan_sources([src], index_db=db)
    assert [r.content_hash for r in recs] == [r.content_hash for r in recs2]
    db.close()


def test_transform_caption_rules():
    cfg = CaptionConfig(
        trigger_word="ypd", keep_tokens=1, shuffle=True, tag_dropout=0.5, prefix="masterpiece"
    )
    rng = random.Random(0)
    out = transform_caption("a, b, c, d, e, f", cfg, rng)
    tags = [t.strip() for t in out.split(",")]
    assert tags[0] == "masterpiece" and tags[1] == "ypd" and tags[2] == "a"
    assert set(tags[3:]) <= {"b", "c", "d", "e", "f"}
    drop = CaptionConfig(caption_dropout=1.0)
    assert transform_caption("x, y", drop, rng) is None
    wc = CaptionConfig(wildcard=True)
    assert transform_caption("{red|blue} hair", wc, random.Random(1)) in ("red hair", "blue hair")


def test_caption_variants_for_cache_are_bounded_and_deterministic():
    from ypuddin.data.captions import (
        caption_variants_for_cache,
        is_stochastic,
        transform_caption_deterministic,
    )

    fixed = CaptionConfig(trigger_word="ypd", prefix="masterpiece", caption_dropout=0.5)
    assert not is_stochastic(fixed)
    assert caption_variants_for_cache("a, b, c", fixed, 8, seed=1) == ["masterpiece, ypd, a, b, c"]
    assert (
        transform_caption_deterministic("a, b, c", fixed) == "masterpiece, ypd, a, b, c"
    )  # dropout never applies here
    stoch = CaptionConfig(trigger_word="ypd", shuffle=True, tag_dropout=0.3, caption_dropout=0.9)
    assert is_stochastic(stoch)
    v1 = caption_variants_for_cache("a, b, c, d, e", stoch, 16, seed=5)
    v2 = caption_variants_for_cache("a, b, c, d, e", stoch, 16, seed=5)
    assert v1 == v2 and 1 < len(v1) <= 16
    assert all(v.startswith("ypd") for v in v1)  # caption dropout is excluded: no "" variant
    assert caption_variants_for_cache("a, b, c, d, e", stoch, 16, seed=6) != v1


def test_sampler_deterministic_resumable_no_double_train():
    buckets = [(64, 64)] * 7 + [(96, 64)] * 5 + [(64, 96)] * 3
    s = BucketBatchSampler(buckets, batch_size=2, seed=3)
    plan = list(s)
    flat = sorted(i for b in plan for i in b)
    assert flat == list(range(15))  # every index exactly once, drop_last=False keeps tails
    assert all(len({buckets[i] for i in b}) == 1 for b in plan)
    s2 = BucketBatchSampler(buckets, batch_size=2, seed=3)
    assert list(s2) == plan
    s2.set_position(3)
    assert list(s2) == plan[3:]
    s2.set_epoch(1)
    assert list(s2) != plan
    state = s2.state_dict()
    s3 = BucketBatchSampler(buckets, batch_size=2, seed=99)
    s3.load_state_dict(state)
    assert s3.seed == 3 and s3.epoch == 1
    assert list(s3) == list(s2)


def test_build_data_and_cache_latents(image_dataset, tmp_path):
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "toy"},
            "dataset": {
                "sources": [{"path": str(image_dataset), "repeats": 2}],
                "resolutions": [64],
                "bucket_step": 16,
                "flip": True,
                "masked_loss": True,
            },
            "validation": {"enabled": True, "split_ratio": 0.25},
        }
    )
    fam = get_family("toy")
    bundle = build_data(cfg, fam.spec.latent, cache_root=tmp_path / "cache")
    assert bundle.plan.images + bundle.plan.validation_images == 12
    assert bundle.plan.items == bundle.plan.images * 2
    assert all(b["w"] % 16 == 0 for b in bundle.plan.buckets)
    loaded = fam.load(cfg.model, cfg.memory, device="cpu", dtype=torch.float32)
    n = cache_latents(bundle, loaded.latent.encode, device="cpu", batch_size=3, dtype=torch.float32)
    assert n == bundle.plan.images * 2 + bundle.plan.validation_images  # flip doubles train only
    assert cache_latents(bundle, loaded.latent.encode, device="cpu") == 0  # idempotent
    bundle.train.set_epoch(0)
    sample = bundle.train[0]
    assert "latents" in sample and sample["latents"].shape[0] == 4
    batch = collate(
        [
            bundle.train[i]
            for i in range(2)
            if bundle.train.items[i].bucket.key == bundle.train.items[0].bucket.key
        ][:1]
        * 2
    )
    assert batch["latents"].shape[0] == 2 and len(batch["caption"]) == 2
    # masks: alpha images produce a mask; sidecar-less RGB images have none unless alpha
    keys = [bundle.train.cache_key(it, False) for it in bundle.train.items]
    assert all(LatentCache(tmp_path / "cache" / "latents").has(k) for k in keys)


def test_dataset_cached_captions_cover_everything_it_emits(image_dataset, tmp_path):
    cfg = TrainConfig.model_validate(
        {
            "model": {"family": "toy"},
            "dataset": {
                "sources": [{"path": str(image_dataset)}],
                "resolutions": [64],
                "bucket_step": 16,
                "cache_latents": False,
                "caption": {
                    "trigger_word": "ypd",
                    "shuffle": True,
                    "tag_dropout": 0.25,
                    "caption_dropout": 0.3,
                    "cache_variants": 6,
                },
            },
            "validation": {"enabled": True, "split_ratio": 0.25},
        }
    )
    bundle = build_data(cfg, get_family("toy").spec.latent, cache_root=tmp_path / "cache")
    train, val = bundle.train, bundle.validation
    # online mode: unbounded variants, trigger word always first
    train.set_epoch(0)
    online = train[0]["caption"]
    assert online == "" or online.startswith("ypd")
    cached_train = set(train.use_cached_captions())
    cached_val = set(val.use_cached_captions())
    assert cached_train and all(
        c.startswith("ypd") for c in cached_train
    )  # "" is cached by the trainer separately
    assert len(cached_train) <= 6 * len(train.items)
    seen: set[str] = set()
    n_uncond = 0
    for epoch in range(6):
        train.set_epoch(epoch)
        for i in range(len(train)):
            s = train[i]
            if s["uncond"]:
                n_uncond += 1
                assert s["caption"] == ""
            else:
                seen.add(s["caption"])
    assert (
        seen <= cached_train and n_uncond > 0
    )  # every emitted caption was pre-cached; dropout still happens
    # validation: deterministic transform (trigger word applied, no shuffle / dropout), one variant per item
    for epoch in range(3):
        val.set_epoch(epoch)
        caps = [val[i]["caption"] for i in range(len(val))]
        assert set(caps) <= cached_val and all(c.startswith("ypd") for c in caps)
        assert not any(val[i]["uncond"] for i in range(len(val)))
    assert caps == [val[i]["caption"] for i in range(len(val))]  # identical across epochs


def _data_config(path, **dataset):
    return TrainConfig.model_validate(
        {
            "model": {"family": "toy"},
            "dataset": {"sources": [{"path": str(path)}], "resolutions": [64], "bucket_step": 16, **dataset},
        }
    )


def test_masks_are_current_and_independent_of_latent_cache(tmp_path):
    source = tmp_path / "images"
    source.mkdir()
    Image.new("RGB", (64, 64), "blue").save(source / "image.png")
    cfg = _data_config(source)
    spec = get_family("toy").spec.latent
    bundle = build_data(cfg, spec, cache_root=tmp_path / "cache")
    calls = []

    def encode(pixels):
        calls.append(len(pixels))
        return torch.full((len(pixels), 4, 8, 8), 0.5)

    assert cache_latents(bundle, encode, device="cpu") == 1
    key = bundle.train.cache_key(bundle.train.items[0], False)
    assert set(bundle.latent_cache.get(key)) == {"latents"}
    # Enable masks after pre-caching; add a new sidecar without changing the latent key.
    cfg.dataset.masked_loss = True
    masked = build_data(cfg, spec, cache_root=tmp_path / "cache")
    Image.new("L", (64, 64), 255).save(source / "image.mask.png")
    assert cache_latents(masked, encode, device="cpu") == 0
    sample = masked.train[0]
    assert sample["mask"].eq(1).all() and "pixels" not in sample
    # Old caches may contain a stale mask; it is never reused.
    masked.latent_cache.put(key, {"latents": sample["latents"], "mask": torch.zeros(64, 64)})
    assert masked.train[0]["mask"].eq(1).all()
    Image.new("L", (64, 64), 0).save(source / "image.mask.png")
    assert masked.train[0]["mask"].eq(0).all()
    (source / "image.mask.png").unlink()
    assert "mask" not in masked.train[0]
    assert calls == [1]


def test_collate_preserves_masks_in_mixed_batch(tmp_path):
    source = tmp_path / "images"
    source.mkdir()
    for name, color in (("a", "red"), ("b", "blue")):
        Image.new("RGB", (64, 64), color).save(source / f"{name}.png")
    Image.new("L", (64, 64), 0).save(source / "a.mask.png")
    bundle = build_data(
        _data_config(source, masked_loss=True, cache_latents=False),
        get_family("toy").spec.latent,
        cache_root=tmp_path / "cache",
    )
    samples = [bundle.train[index] for index in range(2)]
    batch = collate(samples)
    for index, sample in enumerate(samples):
        assert batch["mask"][index].eq(0 if "mask" in sample else 1).all()


def test_source_resolution_union_and_explicit_validation_are_disjoint(image_dataset, tmp_path):
    validation = tmp_path / "validation"
    validation.mkdir()
    shutil.copy(image_dataset / "img_000.jpg", validation / "held.jpg")
    (validation / "held.txt").write_text("held out")
    cfg = _data_config(image_dataset)
    cfg.dataset.sources[0].resolutions = [80]
    cfg.validation.enabled = True
    cfg.validation.split_ratio = 0
    cfg.validation.sources = [
        DatasetSourceConfig(
            path=str(validation), resolutions=[96], repeats=7, caption=CaptionConfig(prefix="validation")
        )
    ]
    bundle = build_data(cfg, get_family("toy").spec.latent, cache_root=tmp_path / "cache")
    assert set(bundle.bucket_manager.resolutions) == {64, 80, 96}
    assert len(bundle.train) == 11 and len(bundle.validation) == 1
    assert all(item.bucket.base == 80 for item in bundle.train.items)
    held = bundle.validation.items[0]
    assert held.record.source_index == 1 and held.source.path == str(validation)
    assert held.bucket.base == 96
    assert bundle.validation[0]["caption"] == "validation, held out"
    assert held.record.content_hash not in {item.record.content_hash for item in bundle.train.items}


def test_data_fingerprint_tracks_sidecars_but_not_locations(image_dataset, tmp_path):
    cfg = _data_config(image_dataset)
    spec = get_family("toy").spec.latent
    original = build_data(cfg, spec, cache_root=tmp_path / "cache")
    moved = tmp_path / "moved"
    shutil.copytree(image_dataset, moved)
    (moved / "img_000.jpg").rename(moved / "renamed.jpg")
    (moved / "img_000.txt").rename(moved / "renamed.txt")
    moved_cfg = _data_config(moved)
    copied = build_data(moved_cfg, spec, cache_root=tmp_path / "cache")
    assert copied.plan.fingerprint == original.plan.fingerprint
    assert [item.record.content_hash for item in copied.train.items] == [
        item.record.content_hash for item in original.train.items
    ]
    (moved / "renamed.txt").write_text("changed caption")
    caption_changed = build_data(moved_cfg, spec, cache_root=tmp_path / "cache")
    assert caption_changed.plan.fingerprint != copied.plan.fingerprint
    Image.new("L", (96, 64), 0).save(moved / "renamed.mask.png")
    mask_added = build_data(moved_cfg, spec, cache_root=tmp_path / "cache")
    assert mask_added.plan.fingerprint != caption_changed.plan.fingerprint
    Image.new("L", (96, 64), 255).save(moved / "renamed.mask.png")
    assert (
        build_data(moved_cfg, spec, cache_root=tmp_path / "cache").plan.fingerprint
        != mask_added.plan.fingerprint
    )


def test_actual_vae_fingerprint_invalidates_only_latents(image_dataset, tmp_path):
    cfg = _data_config(image_dataset)
    spec = get_family("toy").spec.latent
    first = build_data(cfg, replace(spec, fingerprint="vae-one"), cache_root=tmp_path / "cache")
    second = build_data(cfg, replace(spec, fingerprint="vae-two"), cache_root=tmp_path / "cache")
    assert first.plan.fingerprint == second.plan.fingerprint
    assert first.train.cache_key(first.train.items[0], False) != second.train.cache_key(
        second.train.items[0], False
    )


def test_tensor_cache_concurrent_same_key_is_atomic(tmp_path, monkeypatch):
    import ypuddin.data.cache as cache_module

    cache = LatentCache(tmp_path / "latents")
    original_save = cache_module.save_file
    barrier = Barrier(4)

    def synchronized_save(tensors, filename):
        original_save(tensors, filename)
        barrier.wait(timeout=10)

    monkeypatch.setattr(cache_module, "save_file", synchronized_save)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(
            executor.map(
                lambda value: cache.put("same-key", {"latents": torch.full((4, 8, 8), float(value))}),
                range(4),
            )
        )
    values = cache.get("same-key")["latents"].unique().tolist()
    assert len(values) == 1 and values[0] in range(4)
    assert not list(cache.root.rglob("*.tmp"))


def test_tensor_cache_failed_write_preserves_previous_entry(tmp_path, monkeypatch):
    import ypuddin.data.cache as cache_module

    cache = LatentCache(tmp_path / "latents")
    cache.put("entry", {"latents": torch.ones(2)})

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cache_module, "save_file", fail)
    with pytest.raises(OSError, match="disk full"):
        cache.put("entry", {"latents": torch.zeros(2)})
    assert cache.get("entry")["latents"].eq(1).all()
    assert not list(cache.root.rglob("*.tmp"))


def test_image_index_detects_replacement_even_with_same_size_and_mtime(tmp_path):
    image = tmp_path / "image.bmp"
    Image.new("RGB", (64, 64), "red").save(image)
    source = DatasetSourceConfig(path=str(tmp_path))
    db = IndexDB(tmp_path / "index.sqlite")
    try:
        first = scan_sources([source], index_db=db)[0]
        original = image.stat()
        Image.new("RGB", (64, 64), "blue").save(image)
        assert image.stat().st_size == original.st_size
        os.utime(image, ns=(original.st_atime_ns, original.st_mtime_ns))
        second = scan_sources([source], index_db=db)[0]
        assert first.content_hash != second.content_hash
    finally:
        db.close()
