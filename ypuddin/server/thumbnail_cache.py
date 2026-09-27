"""Bounded, disposable image previews, separate from training caches."""

from __future__ import annotations

import io
import os
import re
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path

from PIL import Image

_NAME = re.compile(r"[a-f0-9]{16}_[0-9]+\.jpg")


class ThumbnailCache:
    def __init__(self, context):
        self.context = context
        self.lock = threading.RLock()
        self._roots: tuple[Path, ...] = ()
        self._files: OrderedDict[Path, tuple[int, float]] = OrderedDict()
        self._bytes = 0
        self._stamps: tuple[int | None, ...] = ()
        self._generation = 0

    def _directory_stamps(self, roots):
        return tuple(root.stat().st_mtime_ns if root.exists() else None for root in roots)

    def _locations(self):
        active = self.context.service_cache_dir("thumbnails")
        roots = tuple(dict.fromkeys((self.context.data_root / "thumbs", active)))
        for root in roots:
            # Configured roots are resolved by settings; never follow a link added below them.
            if root.resolve() != root.absolute():
                raise OSError("缩略图缓存目录不能是符号链接")
        return active, roots

    def _scan(self, roots):
        files = []
        for root in roots:
            if not root.exists():
                continue
            with os.scandir(root) as entries:
                for entry in entries:
                    if not _NAME.fullmatch(entry.name) or not entry.is_file(follow_symlinks=False):
                        continue
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    files.append((Path(entry.path), (info.st_size, info.st_mtime)))
        self._files = OrderedDict(sorted(files, key=lambda item: item[1][1]))
        self._bytes = sum(info[0] for info in self._files.values())
        self._roots = roots
        self._stamps = self._directory_stamps(roots)

    def _sync(self, *, rescan=False):
        active, roots = self._locations()
        if rescan or roots != self._roots or self._directory_stamps(roots) != self._stamps:
            self._scan(roots)
        return active

    def _limit(self):
        return self.context.settings()["cache"]["thumbnail_max_mb"] * 1024**2

    def _forget(self, path):
        item = self._files.pop(path, None)
        if item:
            self._bytes -= item[0]

    def _prune(self, limit):
        removed = failed = 0
        if self._bytes <= limit:
            self._stamps = self._directory_stamps(self._roots)
            return removed, failed
        for path, (size, _) in list(self._files.items()):
            if self._bytes <= limit:
                break
            try:
                if not path.is_symlink():
                    path.unlink(missing_ok=True)
                    removed += size
                self._forget(path)
            except OSError:
                failed += 1
        self._stamps = self._directory_stamps(self._roots)
        return removed, failed

    def manage(self, *, clear=False):
        with self.lock:
            if clear:
                self._generation += 1
            self._sync(rescan=True)
            limit = self._limit()
            removed, failed = self._prune(0 if clear else limit)
            return {
                "used_bytes": self._bytes,
                "file_count": len(self._files),
                "max_bytes": limit,
                "cleared_bytes": removed,
                "failed_files": failed,
            }

    def image(self, source: str | Path, content_hash: str, size: int) -> bytes:
        name = f"{content_hash}_{size}.jpg"
        if not _NAME.fullmatch(name):
            raise ValueError("Invalid thumbnail identity")
        # Return bytes so clearing or evicting a file cannot race FileResponse's later open.
        with self.lock:
            root = self._sync()
            limit = self._limit()
            path = root / name
            if path in self._files and not path.is_symlink():
                try:
                    data = path.read_bytes()
                except FileNotFoundError:
                    self._forget(path)
                else:
                    length, accessed = self._files[path]
                    self._bytes += len(data) - length
                    self._files[path] = (len(data), accessed)
                    stamp = time.time()
                    self._files.move_to_end(path)
                    if stamp - accessed >= 60:
                        os.utime(path, (stamp, stamp))
                        self._files[path] = (len(data), stamp)
                    self._prune(limit)
                    return data
            generation = self._generation
        # Decoding a large original must not hold up other thumbnails or a manual clear.
        with Image.open(source) as original:
            image = original.convert("RGB")
            image.thumbnail((size, size))
            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=85)
            data = buffer.getvalue()
        with self.lock:
            # A clear during decoding must not be undone by this in-flight request.
            if generation != self._generation:
                return data
            root = self._sync()
            path = root / name
            limit = self._limit()
            if len(data) <= limit and not path.is_symlink():
                root.mkdir(parents=True, exist_ok=True)
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=root, suffix=".tmp", delete=False) as stream:
                        temporary = Path(stream.name)
                        stream.write(data)
                    temporary.replace(path)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
                self._forget(path)
                self._files[path] = (len(data), time.time())
                self._bytes += len(data)
            self._prune(limit)
            return data
