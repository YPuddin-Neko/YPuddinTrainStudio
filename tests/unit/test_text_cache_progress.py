"""Text-cache progress represents readable entries, including duplicate input captions."""

import pytest
import torch

from ypuddin.data.cache import TextCache, build_text_cache

FINGERPRINT = "progress-test"


def entry():
    return {"embeds": torch.ones(2, 3)}


def key(caption):
    return TextCache.key(caption, FINGERPRINT)


@pytest.mark.parametrize("batch_size", [1, 2, 16])
def test_progress_waits_for_actual_batch_writes(tmp_path, batch_size):
    cache = TextCache(tmp_path)
    captions = ["a", "b", "c", "d", "e"]
    updates, batches = [], []

    def progress(done, total):
        # Even the final callback must run after all advertised files can be read.
        assert done == sum(cache.has(key(caption)) for caption in captions)
        for caption in captions:
            if cache.has(key(caption)):
                assert cache.get(key(caption))["embeds"].shape == (2, 3)
        updates.append((done, total))

    def encode(batch):
        assert updates[-1][0] == sum(len(previous) for previous in batches)
        assert updates[-1][0] < len(captions)
        batches.append(list(batch))
        return [entry() for _ in batch]

    written = build_text_cache(
        iter(captions),
        cache,
        encode,
        FINGERPRINT,
        batch_size=batch_size,
        progress=progress,
        total=len(captions),
    )
    assert written == len(captions)
    assert [caption for batch in batches for caption in batch] == captions
    assert [len(batch) for batch in batches] == [
        min(batch_size, len(captions) - offset) for offset in range(0, len(captions), batch_size)
    ]
    assert updates == [(done, len(captions)) for done in range(len(captions) + 1)]


def test_cache_hits_and_completed_duplicates_need_no_encoding(tmp_path):
    cache = TextCache(tmp_path)
    for caption in ("a", "b"):
        cache.put(key(caption), entry())
    updates = []

    def encode(_):
        pytest.fail("Existing entries must not be encoded again")

    assert (
        build_text_cache(
            ["a", "a", "b"],
            cache,
            encode,
            FINGERPRINT,
            progress=lambda done, total: updates.append((done, total)),
            total=3,
        )
        == 0
    )
    assert updates == [(0, 3), (1, 3), (2, 3), (3, 3)]


def test_pending_duplicates_finish_only_when_the_shared_entry_is_written(tmp_path):
    cache = TextCache(tmp_path)
    updates, batches = [], []

    def encode(batch):
        batches.append(list(batch))
        assert updates[-1][0] == (0 if len(batches) == 1 else 4)
        return [entry() for _ in batch]

    assert (
        build_text_cache(
            ["a", "a", "b", "a", "c", "c"],
            cache,
            encode,
            FINGERPRINT,
            batch_size=2,
            progress=lambda done, total: updates.append((done, total)),
            total=6,
        )
        == 3
    )
    assert batches == [["a", "b"], ["c"]]
    assert updates == [(0, 6), (2, 6), (3, 6), (4, 6), (6, 6)]


def test_hits_can_finish_while_another_caption_is_still_pending(tmp_path):
    cache = TextCache(tmp_path)
    cache.put(key("hit"), entry())
    updates = []

    def encode(batch):
        assert batch == ["new", "last"]
        assert updates == [(0, 4), (1, 4)]
        return [entry(), entry()]

    assert (
        build_text_cache(
            ["new", "hit", "new", "last"],
            cache,
            encode,
            FINGERPRINT,
            progress=lambda done, total: updates.append((done, total)),
            total=4,
        )
        == 2
    )
    assert updates == [(0, 4), (1, 4), (3, 4), (4, 4)]


def test_encoder_failure_does_not_finish_pending_entries(tmp_path):
    cache = TextCache(tmp_path)
    cache.put(key("hit"), entry())
    updates = []

    def encode(_):
        raise RuntimeError("encoder failed")

    with pytest.raises(RuntimeError, match="encoder failed"):
        build_text_cache(
            ["new", "hit", "new"],
            cache,
            encode,
            FINGERPRINT,
            progress=lambda done, total: updates.append((done, total)),
            total=3,
        )
    assert updates == [(0, 3), (1, 3)]
    assert not cache.has(key("new"))


@pytest.mark.parametrize("returned", [1, 3])
def test_invalid_encoder_result_count_cannot_publish_false_completion(tmp_path, returned):
    cache = TextCache(tmp_path)
    updates = []
    with pytest.raises(ValueError, match=f"returned {returned} entries for 2 captions"):
        build_text_cache(
            ["a", "b"],
            cache,
            lambda _: [entry() for _ in range(returned)],
            FINGERPRINT,
            progress=lambda done, total: updates.append((done, total)),
            total=2,
        )
    assert updates == [(0, 2)]
    assert cache.count() == 0


def test_write_failure_reports_only_the_entries_already_published(tmp_path, monkeypatch):
    cache = TextCache(tmp_path)
    original_put = cache.put
    updates = []

    def put(cache_key, value):
        if cache_key == key("b"):
            raise OSError("disk full")
        original_put(cache_key, value)

    monkeypatch.setattr(cache, "put", put)
    with pytest.raises(OSError, match="disk full"):
        build_text_cache(
            ["a", "b"],
            cache,
            lambda _: [entry(), entry()],
            FINGERPRINT,
            progress=lambda done, total: updates.append((done, total)),
            total=2,
        )
    assert updates == [(0, 2), (1, 2)]
    assert cache.has(key("a")) and not cache.has(key("b"))
