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

from .image_metadata import has_alpha, has_color_key, has_transparent_pixels

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
    color_key_transparency: bool = False
    # An alpha channel or transparency key with pixels below full opacity; what masked loss can use.
    has_transparency: bool = False

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


def caption_for(
    path: str | Path,
    ext: str = "auto",
    *,
    directory_cache: dict[Path, dict[str, dict[str, Path]]] | None = None,
) -> str | None:
    """Select one caption sidecar. Auto prefers JSON; explicit suffixes never fall back."""
    path = Path(path)
    siblings = None
    if directory_cache is not None:
        if path.parent not in directory_cache:
            directory_cache[path.parent] = _caption_siblings(path.parent)
        siblings = directory_cache[path.parent]
    return _caption_for(path, ext, siblings)


def _caption_for(path: Path, ext: str, siblings: dict[str, dict[str, Path]] | None = None) -> str | None:
    for suffix in (".json", ".txt") if ext.lower() == "auto" else (ext,):
        candidate = path.with_suffix(suffix) if suffix.startswith(".") else path.with_name(path.stem + suffix)
        # Preserve exact image stem matching on case-sensitive systems; only suffix case varies.
        if siblings is None:
            siblings = _caption_siblings(path.parent)
        matches = siblings.get(path.stem + suffix.lower())
        if matches:
            return str(matches.get(candidate.name) or matches[min(matches)])
        if candidate.is_file():  # Existing custom suffixes keep their original lookup behavior.
            return str(candidate)
    return None


def _caption_siblings(directory: Path) -> dict[str, dict[str, Path]]:
    # One directory enumeration per scan, including the common auto+TXT-only case.
    entries: dict[str, dict[str, Path]] = {}
    try:
        with os.scandir(directory) as listing:
            for item in listing:
                # The listing's file type answers this without a stat for ordinary files.
                if item.is_file():
                    entry = Path(directory) / item.name
                    entries.setdefault(entry.stem + entry.suffix.lower(), {})[entry.name] = entry
    except (FileNotFoundError, NotADirectoryError):
        pass
    return entries


class FolderFiles:
    """One listing of a folder: its files by caption key, and every name for exact lookups."""

    __slots__ = ("folded", "siblings")

    def __init__(self, siblings: dict[str, dict[str, Path]]):
        self.siblings = siblings
        self.folded = {name.casefold() for group in siblings.values() for name in group}

    def has(self, path: Path) -> bool:
        """``path.is_file()`` as of the listing; only a name differing in case is checked on disk."""
        group = self.siblings.get(path.stem + path.suffix.lower())
        if group and path.name in group:
            return True
        return path.name.casefold() in self.folded and path.is_file()

    def caption(self, image: Path, ext: str = "auto") -> str | None:
        """``caption_for`` of an image in this folder."""
        for suffix in (".json", ".txt") if ext.lower() == "auto" else (ext,):
            candidate = (
                image.with_suffix(suffix) if suffix.startswith(".") else image.with_name(image.stem + suffix)
            )
            matches = self.siblings.get(image.stem + suffix.lower())
            if matches:
                return str(matches.get(candidate.name) or matches[min(matches)])
            if self.has(candidate):
                return str(candidate)
        return None

    def caption_target(self, image: Path, ext: str = "auto") -> Path:
        """``caption_target`` of an image in this folder."""
        existing = self.caption(image, ext)
        if existing:
            return Path(existing)
        suffix = ".txt" if ext.lower() == "auto" else ext
        return image.with_suffix(suffix) if suffix.startswith(".") else image.with_name(image.stem + suffix)

    def mask(self, image: Path) -> str | None:
        """``mask_for`` of an image in this folder."""
        for suffix in MASK_SUFFIXES:
            candidate = image.with_name(image.stem + suffix)
            if self.has(candidate):
                return str(candidate)
        return None


@dataclass
class SourceListing:
    """One walk of a source folder, shared by its change signature, its scan and its inspection."""

    root: Path
    images: list[Path]
    folders: dict[Path, FolderFiles]

    def folder(self, image: Path) -> FolderFiles:
        return self.folders[image.parent]


def list_source(root: str | Path, folders: dict[Path, FolderFiles] | None = None) -> SourceListing:
    """Images in ``iter_images`` order with one listing of each folder that holds them.

    ``folders`` shares listings between sources of one pass, so a folder is listed once.
    """
    images = list(iter_images(root))
    shared = {} if folders is None else folders
    for image in images:
        if image.parent not in shared:
            shared[image.parent] = FolderFiles(_caption_siblings(image.parent))
    return SourceListing(Path(root).expanduser(), images, shared)


def caption_target(
    path: str | Path,
    ext: str = "auto",
    *,
    directory_cache: dict[Path, dict[str, dict[str, Path]]] | None = None,
) -> Path:
    """Use the selected existing caption; a missing auto caption is created as TXT."""
    path = Path(path)
    existing = caption_for(path, ext, directory_cache=directory_cache)
    if existing:
        return Path(existing)
    suffix = ".txt" if ext.lower() == "auto" else ext
    return path.with_suffix(suffix) if suffix.startswith(".") else path.with_name(path.stem + suffix)


def mask_for(path: Path) -> str | None:
    for suf in MASK_SUFFIXES:
        p = path.with_name(path.stem + suf)
        if p.is_file():
            return str(p)
    return None


def iter_images(root: str | Path) -> Iterator[Path]:
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset source not found: {root}")
    for dirpath, dirs, files in os.walk(root):
        dirs.sort()
        for f in sorted(files):
            p = Path(dirpath) / f
            if p.suffix.lower() in IMAGE_EXTS and not p.name.endswith(".mask.png"):
                yield p


class IndexDB:
    """SQLite image probe cache with nanosecond file identity checks so rescans are cheap."""

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS files_v2 (path TEXT PRIMARY KEY, signature TEXT, hash TEXT, width INTEGER, height INTEGER, has_alpha INTEGER)"
        )
        # Rows from before carry NULL and are measured on their next scan, without rehashing the file.
        if "transparent" not in {row[1] for row in self.conn.execute("PRAGMA table_info(files_v2)")}:
            self.conn.execute("ALTER TABLE files_v2 ADD COLUMN transparent INTEGER")
        # Whole-file digests of images and sidecars, kept across restarts and keyed by file identity.
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS file_digests (path TEXT PRIMARY KEY, signature TEXT NOT NULL,"
            " digest TEXT NOT NULL)"
        )
        self.conn.commit()

    def digests(self, files: dict[str, str]) -> dict[str, str]:
        """Stored digests of ``{path: stat signature}`` whose file is unchanged since it was read."""
        found: dict[str, str] = {}
        paths = list(files)
        for start in range(0, len(paths), 500):
            chunk = paths[start : start + 500]
            marks = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT path, signature, digest FROM file_digests WHERE path IN ({marks})", chunk
            )
            found.update({path: digest for path, signature, digest in rows if files[path] == signature})
        return found

    def store_digests(self, rows: list[tuple[str, str, str]]) -> None:
        """Remember ``(path, stat signature, digest)`` rows."""
        self.conn.executemany(
            "INSERT OR REPLACE INTO file_digests (path, signature, digest) VALUES (?,?,?)", rows
        )
        self.conn.commit()

    def lookup(
        self, path: str, mtime: float, size: int, *, stat_signature: str | None = None
    ) -> tuple[str, int, int, bool, bool | None] | None:
        row = self.conn.execute(
            "SELECT hash, width, height, has_alpha, signature, transparent FROM files_v2 WHERE path=?",
            (path,),
        ).fetchone()
        if row and row[4] == (stat_signature or json.dumps((mtime, size))):
            return row[0], row[1], row[2], bool(row[3]), None if row[5] is None else bool(row[5])
        return None

    def store(
        self,
        path: str,
        mtime: float,
        size: int,
        h: str,
        w: int,
        ht: int,
        alpha: bool,
        *,
        transparent: bool | None = None,
        stat_signature: str | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO files_v2 (path, signature, hash, width, height, has_alpha, transparent)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                path,
                stat_signature or json.dumps((mtime, size)),
                h,
                w,
                ht,
                int(alpha),
                None if transparent is None else int(transparent),
            ),
        )

    def store_transparency(self, path: str, transparent: bool) -> None:
        self.conn.execute("UPDATE files_v2 SET transparent=? WHERE path=?", (int(transparent), path))

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


def probe_transparency(path: Path) -> bool:
    """Decode the alpha of an image that has one; opaque channels and keys do not count."""
    with Image.open(path) as im:
        return has_alpha(im) and has_transparent_pixels(im)


def probe_image(path: Path) -> tuple[int, int, bool]:
    with Image.open(path) as im:
        width, height = im.size
        if im.getexif().get(274) in (5, 6, 7, 8):
            width, height = height, width
        return (
            width,
            height,
            has_alpha(im),
        )


def stat_signature(st: os.stat_result) -> str:
    """The file identity that ``IndexDB`` keys a stored probe with."""
    return json.dumps(("exif-size-alpha-v2", st.st_mtime_ns, st.st_ctime_ns, st.st_size, st.st_ino, st.st_dev))


def scan_sources(
    sources: list[DatasetSourceConfig],
    *,
    index_db: IndexDB | None = None,
    progress: Callable[[int, int], None] | None = None,
    listings: dict[str, SourceListing] | None = None,
) -> list[ImageRecord]:
    """``listings`` holds walks already made of these source paths, such as for a change signature."""
    records: list[ImageRecord] = []
    paths: list[tuple[int, Path, DatasetSourceConfig, FolderFiles]] = []
    folders: dict[Path, FolderFiles] = {}
    for si, src in enumerate(sources):
        excluded = set(src.excluded_files)
        prefixes = tuple(directory + "/" for directory in src.excluded_dirs)
        listing = (listings or {}).get(src.path) or list_source(src.path, folders)
        for p in listing.images:
            relative = p.relative_to(Path(src.path).expanduser()).as_posix()
            if relative not in excluded and not relative.startswith(prefixes):
                paths.append((si, p, src, listing.folder(p)))
    total = len(paths)
    for i, (si, p, src, folder) in enumerate(paths):
        st = p.stat()
        signature = stat_signature(st)
        cached = (
            index_db.lookup(str(p), st.st_mtime, st.st_size, stat_signature=signature) if index_db else None
        )
        if cached is None:
            try:
                w, h, alpha = probe_image(p)
            except Exception as e:  # noqa: BLE001
                log.warning("skipping unreadable image %s (%s)", p, e)
                continue
            digest = content_hash(p)
            transparent = alpha and probe_transparency(p)
            if index_db:
                index_db.store(
                    str(p),
                    st.st_mtime,
                    st.st_size,
                    digest,
                    w,
                    h,
                    alpha,
                    transparent=transparent,
                    stat_signature=signature,
                )
        else:
            digest, w, h, alpha, transparent = cached
            if transparent is None:
                transparent = alpha and probe_transparency(p)
                if index_db:
                    index_db.store_transparency(str(p), transparent)
        color_key = False
        if alpha:
            with Image.open(p) as image:
                color_key = has_color_key(image)
        records.append(
            ImageRecord(
                path=str(p),
                source_index=si,
                content_hash=digest,
                width=w,
                height=h,
                caption_path=folder.caption(p, src.caption_ext),
                mask_path=folder.mask(p),
                has_alpha=alpha,
                color_key_transparency=color_key,
                has_transparency=transparent,
            )
        )
        if progress and (i % 50 == 0 or i == total - 1):
            progress(i + 1, total)
    if index_db:
        index_db.commit()
    return records


def record_content_key(record: ImageRecord) -> tuple[int, str, str, str]:
    """Semantic ordering: renaming/moving an image and its sidecars keeps its sampler position."""
    caption_hash = content_hash(record.caption_path) if record.caption_path else ""
    if record.caption_path and Path(record.caption_path).suffix.lower() == ".json":
        # TXT hashes stay unprefixed to keep their fingerprints and resume ordering. The
        # prefix keeps checkpoints that trained JSON as raw file text from resuming with
        # structured semantics.
        caption_hash = "structured-json-v1:" + caption_hash
    return (
        record.source_index,
        ("color-key-white-v1:" if record.color_key_transparency else "") + record.content_hash,
        caption_hash,
        content_hash(record.mask_path) if record.mask_path else "",
    )


def dataset_fingerprint(
    records: list[ImageRecord],
    sources: list[DatasetSourceConfig],
    *,
    settings: dict | None = None,
) -> str:
    """Identity of training content and its interpretation, independent of source locations.

    Caption/mask bytes and source association matter; directory names and image names do not.
    Versioning intentionally invalidates old checkpoints whose fingerprints ignored sidecars.
    """
    h = hashlib.blake2b(digest_size=8)
    # Selection is already represented by the records; keep names out of resume identity.
    semantic_sources = [s.model_dump(mode="json", exclude={"path", "excluded_files", "excluded_dirs"}) for s in sources]
    payload = {
        "version": 2,
        "sources": semantic_sources,
        "settings": settings or {},
        "records": sorted(record_content_key(r) for r in records),
    }
    h.update(json.dumps(payload, sort_keys=True).encode())
    return h.hexdigest()
