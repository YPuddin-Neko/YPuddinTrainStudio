import os
import shutil

import pytest

from ypuddin.models import fingerprints
from ypuddin.models.fingerprints import content_fingerprint, fingerprint_cache


def test_file_fingerprint_tracks_bytes_with_stat_cached_reuse(tmp_path, monkeypatch):
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"weight-A")
    cache = tmp_path / "hash-cache"
    reads = []
    real_read = fingerprints._read_digest

    def track(path):
        reads.append(path)
        return real_read(path)

    monkeypatch.setattr(fingerprints, "_read_digest", track)
    first = content_fingerprint([weights], cache_dir=cache)
    fingerprints._MEMORY.clear()  # Simulate a new task/process using the persistent index.
    assert content_fingerprint([weights], cache_dir=cache) == first
    assert len(reads) == 1
    old_stat = weights.stat()
    weights.write_bytes(b"weight-B")  # Same length and restored mtime must not reuse old contents.
    os.utime(weights, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    second = content_fingerprint([weights], cache_dir=cache)
    assert second != first and len(reads) == 2
    renamed = tmp_path / "renamed.safetensors"
    weights.rename(renamed)
    assert content_fingerprint([renamed], cache_dir=cache) == second
    assert content_fingerprint([renamed], cache_dir=cache, namespace="other-preprocessing") != second


def test_hf_directory_and_selected_tokenizer_contents_are_fingerprinted(tmp_path):
    hf = tmp_path / "hf"
    tokenizer = tmp_path / "bundled-tokenizer"
    hf.mkdir()
    tokenizer.mkdir()
    for name, content in (
        ("model-00001.safetensors", b"weights"),
        ("config.json", b"config"),
        ("model.safetensors.index.json", b"index"),
    ):
        (hf / name).write_bytes(content)
    (tokenizer / "tokenizer.json").write_bytes(b"tokenizer")
    original = content_fingerprint([hf, tokenizer])
    moved = tmp_path / "moved-hf"
    shutil.copytree(hf, moved)
    assert content_fingerprint([moved, tokenizer]) == original
    # Runtime/download metadata does not alter the model identity.
    (hf / ".cache").mkdir()
    (hf / ".cache" / "download.lock").write_text("metadata")
    assert content_fingerprint([hf, tokenizer]) == original
    for asset in (
        hf / "model-00001.safetensors",
        hf / "config.json",
        hf / "model.safetensors.index.json",
        tokenizer / "tokenizer.json",
    ):
        before = content_fingerprint([hf, tokenizer])
        asset.write_bytes(asset.read_bytes() + b"changed")
        assert content_fingerprint([hf, tokenizer]) != before


def test_fingerprint_cache_context_and_explicit_override(tmp_path, monkeypatch):
    asset = tmp_path / "weights"
    asset.write_text("weights")
    monkeypatch.setenv("YPUDDIN_FINGERPRINT_CACHE", str(tmp_path / "environment"))
    with fingerprint_cache(tmp_path / "outer"):
        content_fingerprint([asset])
        with fingerprint_cache(tmp_path / "inner"):
            content_fingerprint([asset], cache_dir=tmp_path / "explicit")
        content_fingerprint([asset])
    content_fingerprint([asset])
    assert all(
        (tmp_path / name / "file-hashes.sqlite").exists() for name in ("outer", "explicit", "environment")
    )
    assert not (tmp_path / "inner").exists()


def test_fingerprint_retries_file_modified_during_hash(tmp_path, monkeypatch):
    path = tmp_path / "weights"
    path.write_bytes(b"before")
    real_read = fingerprints._read_digest
    calls = []

    def modified_once(asset):
        calls.append(asset)
        value = real_read(asset)
        if len(calls) == 1:
            asset.write_bytes(b"after!")
        return value

    monkeypatch.setattr(fingerprints, "_read_digest", modified_once)
    result = content_fingerprint([path])
    assert len(calls) == 2
    assert result == content_fingerprint([path])


def test_missing_or_empty_encoder_assets_are_not_valid_identities(tmp_path):
    with pytest.raises(FileNotFoundError):
        content_fingerprint([tmp_path / "missing"])
    with pytest.raises(ValueError, match="empty"):
        content_fingerprint([tmp_path])
    with pytest.raises(ValueError, match="at least one"):
        content_fingerprint([])
