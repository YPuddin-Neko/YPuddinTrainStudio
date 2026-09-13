"""Content-addressed caches for latents and text encodings (one safetensors file per entry)."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

log = logging.getLogger(__name__)


def _key(*parts: object) -> str:
    h = hashlib.blake2b(digest_size=12)
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


class TensorCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.safetensors"

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def get(self, key: str) -> dict[str, Tensor]:
        return load_file(str(self._path(key)))

    def put(self, key: str, tensors: dict[str, Tensor]) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Jobs can populate the same content-addressed entry concurrently. Each writer owns a
        # temporary file; replace exposes only a complete safetensors file to readers.
        fd, name = tempfile.mkstemp(prefix=f".{p.stem}-", suffix=".tmp", dir=p.parent)
        os.close(fd)
        tmp = Path(name)
        try:
            save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(tmp))
            tmp.replace(p)
        finally:
            tmp.unlink(missing_ok=True)

    def count(self) -> int:
        return sum(1 for _ in self.root.rglob("*.safetensors"))


class LatentCache(TensorCache):
    """Key = (content hash, bucket w/h, latent fingerprint, flip)."""

    @staticmethod
    def key(content_hash: str, width: int, height: int, fingerprint: str, flip: bool) -> str:
        return _key("latent", content_hash, width, height, fingerprint, int(flip))


class TextCache(TensorCache):
    """Key = (caption text, text fingerprint). Entries are trimmed to real length."""

    @staticmethod
    def key(caption: str, fingerprint: str) -> str:
        return _key("text", caption, fingerprint)


def build_latent_cache(
    jobs: Iterable[tuple[str, dict[str, Tensor]]],
    cache: LatentCache,
    encode: Callable[[Tensor], Tensor],
    *,
    batch_size: int = 4,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    progress: Callable[[int, int], None] | None = None,
    total: int | None = None,
) -> int:
    """Encode ``(key, {"pixels": (3,H,W)})`` items grouped by shape; returns #written.

    Masks are deliberately not cached with VAE outputs: changing a sidecar or enabling masked
    loss must not require another VAE pass, nor reuse a mask from an earlier training run.
    """
    pending: dict[tuple[int, int], list[tuple[str, dict[str, Tensor]]]] = {}
    written = 0
    done = 0

    def report() -> None:
        if progress and total:
            progress(done, total)

    def flush(shape: tuple[int, int]) -> None:
        nonlocal written, done
        items = pending.pop(shape, [])
        if not items:
            return
        pixels = torch.stack([it[1]["pixels"] for it in items]).to(device)
        with torch.no_grad():
            latents = encode(pixels).to(dtype).cpu()
        if len(latents) != len(items):
            raise ValueError(f"Latent encoder returned {len(latents)} entries for {len(items)} images")
        for (key, _extra), lat in zip(items, latents, strict=True):
            cache.put(key, {"latents": lat})
            written += 1
            done += 1
            report()

    report()
    for key, item in jobs:
        if cache.has(key):
            done += 1
            report()
            continue
        shape = tuple(item["pixels"].shape[-2:])
        pending.setdefault(shape, []).append((key, item))
        if len(pending[shape]) >= batch_size:
            flush(shape)
    for shape in list(pending):
        flush(shape)
    return written


def build_text_cache(
    captions: Iterable[str],
    cache: TextCache,
    encode_for_cache: Callable[[list[str]], list[dict[str, Tensor]]],
    fingerprint: str,
    *,
    batch_size: int = 16,
    progress: Callable[[int, int], None] | None = None,
    total: int | None = None,
) -> int:
    """Encode unique missing captions, reporting input occurrences only once their entry is ready."""
    written = 0
    done = 0
    pending: dict[str, int] = {}
    completed: set[str] = set()

    def report() -> None:
        if progress and total:
            progress(done, total)

    def flush() -> None:
        nonlocal written, done
        if not pending:
            return
        batch = list(pending)
        entries = encode_for_cache(batch)
        # Validate before publishing any entries: an extra encoder result must not
        # turn a failed batch into a reported 100% completion.
        if len(entries) != len(batch):
            raise ValueError(f"Text encoder returned {len(entries)} entries for {len(batch)} captions")
        for cap, entry in zip(batch, entries, strict=True):
            cache.put(TextCache.key(cap, fingerprint), entry)
            written += 1
            done += pending.pop(cap)
            completed.add(cap)
            report()

    report()
    for cap in captions:
        if cap in pending:
            # Repeated captions share one encoding, but none of their occurrences
            # is complete until that pending entry has actually been written.
            pending[cap] += 1
            continue
        if cap in completed or cache.has(TextCache.key(cap, fingerprint)):
            completed.add(cap)
            done += 1
            report()
            continue
        pending[cap] = 1
        if len(pending) >= batch_size:
            flush()
    flush()
    return written
