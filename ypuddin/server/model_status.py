"""Inspection results of registered model files, kept while the files are unchanged.

The model list is read every few seconds while its page is open. Each read would otherwise reopen
every VAE and Klein component to parse its header (on network storage, one round trip per file).
A result is reused while the stat signature of everything the inspection reads is the same: the file
with the ``config.json`` and ``.ypuddin.json`` sidecar beside it, or every file of a model folder.
"""

from __future__ import annotations

import os
import stat
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

LIMIT = 1024  # remembered results
TREE_LIMIT = 4096  # files of a model folder; a larger folder is inspected on every read

_lock = threading.Lock()
_results: OrderedDict[tuple[Any, ...], tuple[Any, Any]] = OrderedDict()


def _stat(path: Path) -> tuple[Any, ...]:
    try:
        info = path.stat()
    except OSError:
        return (path.name, None)
    return (path.name, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino, info.st_dev)


def signature(path: Path) -> tuple[Any, ...] | None:
    """What an inspection of ``path`` reads, by size and modification time; None when unreadable."""
    try:
        info = path.stat()
    except OSError:
        return None
    if not stat.S_ISDIR(info.st_mode):
        sidecars = (_stat(path.parent / "config.json"), _stat(path.with_name(path.name + ".ypuddin.json")))
        return ("file", info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino, info.st_dev, sidecars)
    entries = []
    try:
        # Components may be links to shared folders; a loop ends at the limit.
        for folder, directories, files in os.walk(path, followlinks=True):
            directories.sort()
            for name in sorted(files):
                entries.append(
                    (os.path.relpath(os.path.join(folder, name), path), *_stat(Path(folder, name))[1:])
                )
                if len(entries) > TREE_LIMIT:
                    return None
    except OSError:
        return None
    return ("folder", info.st_mtime_ns, tuple(entries))


def remembered(
    purpose: str, path: Path, compute: Callable[[], tuple[Any, bool]], *, scope: tuple[Any, ...] = ()
) -> Any:
    """``compute()`` returns the result and whether it may be kept (a passing read error may not)."""
    key = (purpose, str(path), scope)
    before = signature(path)
    if before is not None:
        with _lock:
            entry = _results.get(key)
            if entry is not None and entry[0] == before:
                _results.move_to_end(key)
                return entry[1]
    value, keep = compute()
    # A file that changed during the inspection is read again next time.
    if keep and before is not None and signature(path) == before:
        with _lock:
            _results[key] = (before, value)
            _results.move_to_end(key)
            while len(_results) > LIMIT:
                _results.popitem(last=False)
    return value


def clear() -> None:
    with _lock:
        _results.clear()
