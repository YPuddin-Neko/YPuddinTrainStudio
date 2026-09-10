import random

import torch

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
