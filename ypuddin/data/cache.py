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

    def flush(shape: tuple[int, int]) -> None:
        nonlocal written
        items = pending.pop(shape, [])
        if not items:
            return
        pixels = torch.stack([it[1]["pixels"] for it in items]).to(device)
        with torch.no_grad():
            latents = encode(pixels).to(dtype).cpu()
        for (key, _extra), lat in zip(items, latents, strict=True):
            cache.put(key, {"latents": lat})
            written += 1

    for key, item in jobs:
        done += 1
        if cache.has(key):
            if progress and total:
                progress(done, total)
            continue
        shape = tuple(item["pixels"].shape[-2:])
        pending.setdefault(shape, []).append((key, item))
        if len(pending[shape]) >= batch_size:
            flush(shape)
        if progress and total:
            progress(done, total)
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
    written = 0
    batch: list[str] = []
    done = 0

    def flush() -> None:
        nonlocal written
        if not batch:
            return
        for cap, entry in zip(batch, encode_for_cache(batch), strict=True):
            cache.put(TextCache.key(cap, fingerprint), entry)
            written += 1
        batch.clear()

    seen: set[str] = set()
    for cap in captions:
        done += 1
        if cap in seen or cache.has(TextCache.key(cap, fingerprint)):
            if progress and total:
                progress(done, total)
            continue
        seen.add(cap)
        batch.append(cap)
        if len(batch) >= batch_size:
            flush()
        if progress and total:
            progress(done, total)
    flush()
    return written
