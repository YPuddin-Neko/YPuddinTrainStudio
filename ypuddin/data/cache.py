"""Latent and text-encoding caches: one safetensors file per image, named after it.

``latents/<image>_<W>x<H>.safetensors`` holds an image's latents at one bucket size; ``text/<image>.safetensors``
holds its caption encodings, and ``text/_prompts.safetensors`` those of sample prompts and the empty caption.
A file keeps each variant under a short tag of what made it (image content, encoder, fit and flip, or the caption),
so flips, other models and other settings share the file without overwriting each other. Once a file holds its
limit of variants, the oldest go first.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file
from torch import Tensor

log = logging.getLogger(__name__)

# Entries of the earlier layout: <2 hex>/<24 hex>.safetensors, keyed by a hash alone.
_LEGACY_FOLDER = re.compile(r"^[0-9a-f]{2}$")
_LEGACY_ENTRY = re.compile(r"^[0-9a-f]{24}\.safetensors$")
_LOCK_STALE_SECONDS = 60.0
# A file that is missing, being replaced or damaged reads as holding nothing; the next write rebuilds it.
_UNREADABLE = (OSError, ValueError, RuntimeError, SafetensorError)


def _key(*parts: object) -> str:
    h = hashlib.blake2b(digest_size=12)
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def cache_names(records: Iterable) -> dict[str, str]:
    """Each image's cache name by path: its file name without the extension.

    Images that share a name but not their content (``a/001.png`` and ``b/001.png``) get a short content hash
    after it, as does an image named like a shared file.
    """
    groups: dict[str, list] = {}
    for record in records:
        groups.setdefault(Path(record.path).stem.casefold(), []).append(record)
    names = {}
    for folded, members in groups.items():
        shared = len({record.content_hash for record in members}) > 1 or folded == TextCache.PROMPTS
        for record in members:
            stem = Path(record.path).stem
            names[record.path] = f"{stem}_{record.content_hash[:8]}" if shared else stem
    return names


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Serializes rewrites of one file across processes; a lock older than a minute was left by a crash."""
    lock = path.with_name(path.name + ".lock")
    while True:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > _LOCK_STALE_SECONDS:
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            time.sleep(0.01)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _replace(source: Path, target: Path) -> None:
    # Windows refuses to replace a file another process is reading at that moment.
    for attempt in range(50):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(0.05)


class VariantFiles:
    """Safetensors files holding variants as ``<tag>.<tensor>``; the metadata lists the tags, oldest first."""

    def __init__(self, root: str | Path, *, limit: int):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self._drop_legacy_entries()

    def _drop_legacy_entries(self) -> None:
        # The earlier layout cannot be read under the new names, so its entries would only take up space.
        removed = 0
        for folder in self.root.iterdir():
            if not folder.is_dir() or not _LEGACY_FOLDER.match(folder.name):
                continue
            files = list(folder.iterdir())
            if files and all(file.is_file() and _LEGACY_ENTRY.match(file.name) for file in files):
                removed += len(files)
                shutil.rmtree(folder, ignore_errors=True)
        if removed:
            log.info("removed %d cache entries of the earlier layout from %s", removed, self.root)

    def path(self, file: str) -> Path:
        return self.root / file

    def holds(self, file: str, tag: str) -> bool:
        try:
            with safe_open(str(self.path(file)), framework="pt", device="cpu") as handle:
                return any(key.startswith(tag + ".") for key in handle.keys())
        except _UNREADABLE:
            return False

    def read(self, file: str, tag: str) -> dict[str, Tensor] | None:
        prefix = tag + "."
        try:
            with safe_open(str(self.path(file)), framework="pt", device="cpu") as handle:
                keys = [key for key in handle.keys() if key.startswith(prefix)]
                return {key[len(prefix) :]: handle.get_tensor(key) for key in keys} if keys else None
        except _UNREADABLE:
            return None

    def write(self, file: str, variants: dict[str, dict[str, Tensor]]) -> None:
        """Add or replace ``variants`` (by tag), keeping the file's other variants up to the limit."""
        path = self.path(file)
        with _locked(path):
            kept: dict[str, dict[str, Tensor]] = {}
            order: list[str] = []
            try:
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    order = json.loads((handle.metadata() or {}).get("order", "[]"))
                    for key in handle.keys():
                        tag, _, name = key.partition(".")
                        kept.setdefault(tag, {})[name] = handle.get_tensor(key)
            except _UNREADABLE:
                kept, order = {}, []
            order = [tag for tag in order if tag in kept] + [tag for tag in kept if tag not in order]
            for tag, tensors in variants.items():
                kept[tag] = {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}
                order = [other for other in order if other != tag] + [tag]
            while len(order) > self.limit:
                kept.pop(order.pop(0), None)
            fd, name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=self.root)
            os.close(fd)
            temporary = Path(name)
            try:
                save_file(
                    {f"{tag}.{key}": tensor for tag in order for key, tensor in kept[tag].items()},
                    str(temporary),
                    metadata={"order": json.dumps(order)},
                )
                _replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

    def count(self) -> int:
        return sum(1 for _ in self.root.glob("*.safetensors"))


@dataclass(frozen=True)
class CacheKey:
    """Where one variant lives: its file and its tag inside."""

    file: str
    tag: str


class LatentCache(VariantFiles):
    """``<image>_<W>x<H>.safetensors``; a variant per image content, latent encoder and fit, and flip."""

    def __init__(self, root: str | Path):
        super().__init__(root, limit=8)

    @staticmethod
    def key(name: str, content_hash: str, width: int, height: int, fingerprint: str, flip: bool) -> CacheKey:
        return CacheKey(
            f"{name}_{width}x{height}.safetensors", _key("latent", content_hash, fingerprint, int(flip))[:16]
        )

    def has(self, key: CacheKey) -> bool:
        return self.holds(key.file, key.tag)

    def load(self, key: CacheKey) -> Tensor | None:
        entry = self.read(key.file, key.tag)
        return None if entry is None else entry["latents"]

    def put(self, key: CacheKey, latents: Tensor) -> None:
        self.write(key.file, {key.tag: {"latents": latents}})


class TextCache(VariantFiles):
    """``<image>.safetensors`` with a variant per caption and text encoder; prompts share ``_prompts``."""

    PROMPTS = "_prompts"

    def __init__(self, root: str | Path):
        super().__init__(root, limit=256)

    @staticmethod
    def key(name: str, caption: str, fingerprint: str) -> CacheKey:
        # The empty caption (dropout, unconditional sampling) is the same for every image.
        file = TextCache.PROMPTS if caption == "" else name
        return CacheKey(f"{file}.safetensors", _key("text", caption, fingerprint)[:16])

    def has(self, key: CacheKey) -> bool:
        return self.holds(key.file, key.tag)

    def get(self, key: CacheKey) -> dict[str, Tensor] | None:
        return self.read(key.file, key.tag)

    def put(self, key: CacheKey, entry: dict[str, Tensor]) -> None:
        self.write(key.file, {key.tag: entry})


def build_latent_cache(
    jobs: Iterable[tuple[CacheKey, dict[str, Tensor]]],
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
    pending: dict[tuple[int, int], list[tuple[CacheKey, dict[str, Tensor]]]] = {}
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
        # An image's flips and settings at one size share a file, written once per batch.
        files: dict[str, dict[str, dict[str, Tensor]]] = {}
        counts: dict[str, int] = {}
        for (key, _extra), lat in zip(items, latents, strict=True):
            files.setdefault(key.file, {})[key.tag] = {"latents": lat}
            counts[key.file] = counts.get(key.file, 0) + 1
        for file, variants in files.items():
            cache.write(file, variants)
            written += counts[file]
            done += counts[file]
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
    captions: Iterable[tuple[str, str]],
    cache: TextCache,
    encode_for_cache: Callable[[list[str]], list[dict[str, Tensor]]],
    fingerprint: str,
    *,
    batch_size: int = 16,
    progress: Callable[[int, int], None] | None = None,
    total: int | None = None,
) -> int:
    """Encode each ``(image name, caption)`` missing from the cache; returns #written.

    A caption several images share is encoded once. Progress counts input pairs, each once its entry is written.
    """
    written = 0
    done = 0
    # caption -> the keys waiting for it, with how often each was asked for
    pending: dict[str, dict[CacheKey, int]] = {}
    completed: set[CacheKey] = set()

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
        for caption, entry in zip(batch, entries, strict=True):
            for key, occurrences in pending.pop(caption).items():
                cache.put(key, entry)
                written += 1
                done += occurrences
                completed.add(key)
                report()

    report()
    for name, caption in captions:
        key = TextCache.key(name, caption, fingerprint)
        waiting = pending.get(caption, {})
        if key in waiting:
            # A repeated pair is complete only once its pending entry has actually been written.
            waiting[key] += 1
            continue
        if key in completed or cache.has(key):
            completed.add(key)
            done += 1
            report()
            continue
        pending.setdefault(caption, {})[key] = 1
        if len(pending) >= batch_size:
            flush()
    flush()
    return written
