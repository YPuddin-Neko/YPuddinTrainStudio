"""Latent-cache progress advances after cache hits or successful batch writes."""

import pytest
import torch

from ypuddin.data.cache import LatentCache, build_latent_cache


def pixels(height=8, width=8):
    return {"pixels": torch.ones(3, height, width)}


@pytest.mark.parametrize("batch_size", [1, 2, 16])
def test_progress_waits_for_written_latents_including_final_partial_batch(tmp_path, batch_size):
    cache = LatentCache(tmp_path)
    updates, batches = [], []
    keys = list("abcde")

    def progress(done, total):
        assert done == sum(cache.has(key) for key in keys)
        for key in keys:
            if cache.has(key):
                latent = cache.get(key)["latents"]
                assert latent.dtype == torch.bfloat16 and latent.shape == (3, 8, 8)
        updates.append((done, total))

    def encode(batch):
        assert updates[-1][0] == sum(batches)
        assert updates[-1][0] < len(keys)
        batches.append(len(batch))
        return batch

    assert build_latent_cache(
        ((key, pixels()) for key in keys),
        cache,
        encode,
        batch_size=batch_size,
        progress=progress,
        total=len(keys),
    ) == len(keys)
    assert batches == [min(batch_size, len(keys) - offset) for offset in range(0, len(keys), batch_size)]
    assert updates == [(done, len(keys)) for done in range(len(keys) + 1)]


def test_each_shape_batch_completes_only_after_its_own_flush(tmp_path):
    cache = LatentCache(tmp_path)
    updates, shapes = [], []

    def encode(batch):
        shapes.append(tuple(batch.shape))
        assert updates[-1][0] == (0 if len(shapes) == 1 else 2)
        return batch

    jobs = [("a", pixels(8, 8)), ("b", pixels(8, 16)), ("c", pixels(8, 8))]
    assert (
        build_latent_cache(
            jobs,
            cache,
            encode,
            batch_size=2,
            progress=lambda done, total: updates.append((done, total)),
            total=3,
        )
        == 3
    )
    assert shapes == [(2, 3, 8, 8), (1, 3, 8, 16)]
    assert updates == [(0, 3), (1, 3), (2, 3), (3, 3)]
    assert cache.get("b")["latents"].shape == (3, 8, 16)


def test_cache_hit_finishes_while_another_shape_remains_pending(tmp_path):
    cache = LatentCache(tmp_path)
    cache.put("hit", {"latents": torch.ones(3, 8, 8)})
    updates = []

    def encode(batch):
        assert updates == [(0, 3), (1, 3)]
        assert len(batch) == 2
        return batch

    assert (
        build_latent_cache(
            [("a", pixels()), ("hit", {}), ("b", pixels())],
            cache,
            encode,
            progress=lambda done, total: updates.append((done, total)),
            total=3,
        )
        == 2
    )
    assert updates == [(0, 3), (1, 3), (2, 3), (3, 3)]


def test_all_hits_and_duplicate_hits_skip_encoder(tmp_path):
    cache = LatentCache(tmp_path)
    cache.put("hit", {"latents": torch.ones(3, 8, 8)})
    updates = []

    def encode(_):
        pytest.fail("Existing latents must not be encoded again")

    assert (
        build_latent_cache(
            [("hit", {}), ("hit", {})],
            cache,
            encode,
            progress=lambda done, total: updates.append((done, total)),
            total=2,
        )
        == 0
    )
    assert updates == [(0, 2), (1, 2), (2, 2)]


def test_pending_duplicate_jobs_keep_existing_batch_behavior(tmp_path):
    cache = LatentCache(tmp_path)
    updates = []

    def encode(batch):
        assert len(batch) == 2
        assert updates == [(0, 2)]
        return batch

    assert (
        build_latent_cache(
            [("same", pixels()), ("same", pixels())],
            cache,
            encode,
            progress=lambda done, total: updates.append((done, total)),
            total=2,
        )
        == 2
    )
    assert cache.count() == 1
    assert updates == [(0, 2), (1, 2), (2, 2)]


def test_encoder_failure_keeps_completed_hit_but_not_pending_images(tmp_path):
    cache = LatentCache(tmp_path)
    cache.put("hit", {"latents": torch.ones(3, 8, 8)})
    updates = []

    def encode(_):
        raise RuntimeError("VAE failed")

    with pytest.raises(RuntimeError, match="VAE failed"):
        build_latent_cache(
            [("a", pixels()), ("hit", {})],
            cache,
            encode,
            progress=lambda done, total: updates.append((done, total)),
            total=2,
        )
    assert updates == [(0, 2), (1, 2)]
    assert not cache.has("a")


@pytest.mark.parametrize("returned", [1, 3])
def test_bad_batch_size_does_not_publish_or_report_completion(tmp_path, returned):
    cache = LatentCache(tmp_path)
    updates = []
    with pytest.raises(ValueError, match=f"returned {returned} entries for 2 images"):
        build_latent_cache(
            [("a", pixels()), ("b", pixels())],
            cache,
            lambda _: torch.ones(returned, 3, 8, 8),
            progress=lambda done, total: updates.append((done, total)),
            total=2,
        )
    assert updates == [(0, 2)]
    assert cache.count() == 0


def test_write_failure_reports_only_successfully_published_latents(tmp_path, monkeypatch):
    cache = LatentCache(tmp_path)
    original_put = cache.put
    updates = []

    def put(key, value):
        if key == "b":
            raise OSError("disk full")
        original_put(key, value)

    monkeypatch.setattr(cache, "put", put)
    with pytest.raises(OSError, match="disk full"):
        build_latent_cache(
            [("a", pixels()), ("b", pixels())],
            cache,
            lambda batch: batch,
            progress=lambda done, total: updates.append((done, total)),
            total=2,
        )
    assert updates == [(0, 2), (1, 2)]
    assert cache.has("a") and not cache.has("b")
