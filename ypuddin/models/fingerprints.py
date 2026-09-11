"""Content identities for actual encoder weights, configuration and tokenizer assets.

Callers pass the files/directories actually used by the loader. A single safetensors input
identifies its bytes; include its adjacent config separately if the loader reads one. An HF
directory includes all non-hidden files (weight shards, indexes, config, tokenizer assets),
and an explicitly selected or bundled tokenizer directory must be supplied as another input
when it is outside that directory. Namespace the identity by preprocessing/version choices.

The digest never includes absolute paths. Directory entries retain relative names because
renaming a weight shard/tokenizer configuration changes an HF directory's interpretation.
Hash memoization validates size, nanosecond mtime/ctime, inode and device before reusing a
digest. It is an optimization for ordinary local filesystems, not a defense against a writer
that can forge filesystem metadata.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

log = logging.getLogger(__name__)
_CACHE_DIR: ContextVar[Path | None] = ContextVar("encoder_fingerprint_cache", default=None)
_MEMORY: OrderedDict[str, tuple[str, str]] = OrderedDict()
_LOCK = threading.Lock()
_MEMORY_LIMIT = 8192


@contextmanager
def fingerprint_cache(cache_dir: str | Path) -> Iterator[None]:
    """Use a project-local persistent hash index while loading its model components."""
    token = _CACHE_DIR.set(Path(cache_dir))
    try:
        yield
    finally:
        _CACHE_DIR.reset(token)


def _signature(path: Path) -> str:
    stat = path.stat()
    return json.dumps((stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_dev))


def _read_digest(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=32)
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_digest(path: Path, db: sqlite3.Connection | None) -> str:
    identity = str(path.resolve())
    for _ in range(3):
        signature = _signature(path)
        with _LOCK:
            cached = _MEMORY.get(identity)
            if cached and cached[0] == signature:
                _MEMORY.move_to_end(identity)
        row = (
            db.execute("SELECT signature, digest FROM file_hashes WHERE path=?", (identity,)).fetchone()
            if db
            else None
        )
        if cached and cached[0] == signature:
            digest = cached[1]
        elif row and row[0] == signature:
            digest = row[1]
        else:
            digest = _read_digest(path)
        if _signature(path) != signature:
            continue  # Never commit a digest assembled while a file was being replaced/modified.
        with _LOCK:
            _MEMORY[identity] = (signature, digest)
            _MEMORY.move_to_end(identity)
            if len(_MEMORY) > _MEMORY_LIMIT:
                _MEMORY.popitem(last=False)
        if db is not None:
            db.execute("INSERT OR REPLACE INTO file_hashes VALUES (?,?,?)", (identity, signature, digest))
            db.commit()
        return digest
    raise RuntimeError(f"encoder asset changed repeatedly while fingerprinting: {path}")


def _open_cache(cache_dir: str | Path | None) -> sqlite3.Connection | None:
    directory = cache_dir if cache_dir is not None else _CACHE_DIR.get()
    if directory is None:
        directory = os.environ.get("YPUDDIN_FINGERPRINT_CACHE")
    if not directory:
        return None
    db = None
    try:
        root = Path(directory).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(root / "file-hashes.sqlite", timeout=30)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute(
            "CREATE TABLE IF NOT EXISTS file_hashes (path TEXT PRIMARY KEY, signature TEXT NOT NULL, digest TEXT NOT NULL)"
        )
        db.commit()
        return db
    except (OSError, sqlite3.Error) as error:
        if db is not None:
            db.close()
        log.warning("fingerprint index unavailable; hashing without persistence: %s", error)
        return None


def content_fingerprint(
    paths: Iterable[str | Path],
    *,
    cache_dir: str | Path | None = None,
    namespace: str = "encoder",
) -> str:
    """Hash actual asset contents with optional stat-validated persistent memoization.

    Cache precedence: explicit directory, fingerprint_cache context, environment variable,
    then process memory only. Changing namespace or any used asset invalidates the identity.
    """
    inputs = list(paths)
    if not inputs:
        raise ValueError("at least one encoder asset is required for a content fingerprint")
    db = _open_cache(cache_dir)
    manifests = []
    try:
        for value in inputs:
            path = Path(value).expanduser()
            if path.is_file():
                manifests.append({"file": _file_digest(path, db)})
            elif path.is_dir():
                files = sorted(
                    file
                    for file in path.rglob("*")
                    if file.is_file()
                    and not any(part.startswith(".") for part in file.relative_to(path).parts)
                )
                if not files:
                    raise ValueError(f"encoder asset directory is empty: {path}")
                manifests.append(
                    {
                        "directory": [
                            (file.relative_to(path).as_posix(), _file_digest(file, db)) for file in files
                        ]
                    }
                )
            else:
                raise FileNotFoundError(f"encoder asset not found: {path}")
    finally:
        if db is not None:
            db.close()
    payload = json.dumps({"version": 1, "namespace": namespace, "assets": manifests}, sort_keys=True).encode()
    return hashlib.blake2b(payload, digest_size=24).hexdigest()
