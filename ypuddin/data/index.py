"""Dataset indexing: scan source directories, content-hash files, read dimensions, find sidecars."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from ypuddin.config import DatasetSourceConfig

log = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".avif", ".jxl"}
MASK_SUFFIXES = (".mask.png", ".mask")


@dataclass(frozen=True)
class ImageRecord:
    path: str
    source_index: int
    content_hash: str
    width: int
    height: int
    caption_path: str | None
    mask_path: str | None
    has_alpha: bool

    @property
    def stem(self) -> str:
        return Path(self.path).stem

    def to_dict(self) -> dict:
        return asdict(self)


def content_hash(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.blake2b(digest_size=8)
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sidecar(path: Path, ext: str) -> str | None:
    p = path.with_suffix(ext) if ext.startswith(".") else path.with_name(path.stem + ext)
    return str(p) if p.exists() else None


def _mask_for(path: Path) -> str | None:
    for suf in MASK_SUFFIXES:
        p = path.with_name(path.stem + suf)
        if p.exists():
            return str(p)
    return None


def iter_images(root: str | Path) -> Iterator[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset source not found: {root}")
    for dirpath, _dirs, files in os.walk(root):
        for f in sorted(files):
            p = Path(dirpath) / f
            if p.suffix.lower() in IMAGE_EXTS and not p.name.endswith(".mask.png"):
                yield p


class IndexDB:
    """SQLite cache of (path, mtime, size) -> (hash, width, height, has_alpha) so rescans are cheap."""

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, mtime REAL, size INTEGER, hash TEXT, width INTEGER, height INTEGER, has_alpha INTEGER)"
        )
        self.conn.commit()

    def lookup(self, path: str, mtime: float, size: int) -> tuple[str, int, int, bool] | None:
        row = self.conn.execute("SELECT hash, width, height, has_alpha, mtime, size FROM files WHERE path=?", (path,)).fetchone()
        if row and abs(row[4] - mtime) < 1e-6 and row[5] == size:
            return row[0], row[1], row[2], bool(row[3])
        return None

    def store(self, path: str, mtime: float, size: int, h: str, w: int, ht: int, alpha: bool) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?)", (path, mtime, size, h, w, ht, int(alpha))
        )

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


def probe_image(path: Path) -> tuple[int, int, bool]:
    with Image.open(path) as im:
        return im.width, im.height, im.mode in ("RGBA", "LA", "P") and ("transparency" in im.info or im.mode != "P")


def scan_sources(
    sources: list[DatasetSourceConfig],
    *,
    index_db: IndexDB | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    paths: list[tuple[int, Path, DatasetSourceConfig]] = []
    for si, src in enumerate(sources):
        paths.extend((si, p, src) for p in iter_images(src.path))
    total = len(paths)
    for i, (si, p, src) in enumerate(paths):
        st = p.stat()
        cached = index_db.lookup(str(p), st.st_mtime, st.st_size) if index_db else None
        if cached is None:
            try:
                w, h, alpha = probe_image(p)
            except Exception as e:  # noqa: BLE001
                log.warning("skipping unreadable image %s (%s)", p, e)
                continue
            digest = content_hash(p)
            if index_db:
                index_db.store(str(p), st.st_mtime, st.st_size, digest, w, h, alpha)
        else:
            digest, w, h, alpha = cached
        records.append(
            ImageRecord(
                path=str(p),
                source_index=si,
                content_hash=digest,
                width=w,
                height=h,
                caption_path=_sidecar(p, src.caption_ext),
                mask_path=_mask_for(p),
                has_alpha=alpha,
            )
        )
        if progress and (i % 50 == 0 or i == total - 1):
            progress(i + 1, total)
    if index_db:
        index_db.commit()
    return records


def dataset_fingerprint(records: list[ImageRecord], sources: list[DatasetSourceConfig]) -> str:
    h = hashlib.blake2b(digest_size=8)
    h.update(json.dumps([s.model_dump(mode="json") for s in sources], sort_keys=True).encode())
    for r in sorted(records, key=lambda r: r.content_hash):
        h.update(r.content_hash.encode())
    return h.hexdigest()
