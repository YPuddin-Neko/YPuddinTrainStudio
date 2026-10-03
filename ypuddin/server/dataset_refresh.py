"""Dataset indexes follow files changed outside the studio without making readers wait.

A page that shows a dataset asks for a check at most every ``CHECK_INTERVAL`` seconds. The check
lists the folder and compares image sizes and change times, and which captions and masks exist,
with the signature stored by the last index. Only a difference re-indexes, in the
background and listed in the task center; edits are never refused because of it.

The signature adds up one hash per image, so the studio's own edits, which keep the index
records current themselves, update it for just the images they touched.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ypuddin.data.index import FolderFiles, SourceListing, _caption_siblings, list_source

log = logging.getLogger(__name__)

# A dataset is checked again only after this many seconds; the refresh button checks at once.
CHECK_INTERVAL = 10.0
SIGNATURE_PREFIX = "v2:"
_MODULUS = 1 << 128
_create_lock = threading.Lock()


def _hash(value: Any) -> int:
    payload = json.dumps(value, ensure_ascii=False).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), "big")


def _entry(root: Path, image: Path, stat: Any, caption: str | None, mask: str | None) -> int:
    """One image's part of the signature: what its index record depends on."""
    return _hash([
        image.relative_to(root).as_posix(),
        stat.st_size,
        stat.st_mtime_ns,
        # Free with the same stat: a replacement that kept the old time and size still shows.
        stat.st_ctime_ns,
        Path(caption).name if caption else None,
        Path(mask).name if mask else None,
    ])


def _header(row: dict) -> int:
    return _hash(["dataset", row["path"], row["caption_ext"]])


def _encode(value: int) -> str:
    return f"{SIGNATURE_PREFIX}{value % _MODULUS:032x}"


def listing_signature(row: dict, listing: SourceListing) -> str:
    total = _header(row)
    root = Path(row["path"]).expanduser()
    for image in listing.images:
        folder = listing.folder(image)
        total += _entry(root, image, image.stat(), folder.caption(image, row["caption_ext"]), folder.mask(image))
    return _encode(total)


def dataset_signature(row: dict) -> tuple[str, SourceListing | None]:
    """The folder's current signature and the listing it came from; a missing folder has no listing."""
    try:
        listing = list_source(row["path"])
        return listing_signature(row, listing), listing
    except OSError as exc:
        # Unavailable sources keep a stable value between reads and retry when their files return.
        return _encode(_hash(["unavailable", row["path"], type(exc).__name__, exc.errno, exc.filename])), None


def image_entry(row: dict, image: Path, folders: dict[Path, FolderFiles] | None = None) -> int | None:
    """The signature part of one image as it is on disk now; None when it cannot be read.

    ``folders`` shares one listing per folder between the images of one edit.
    """
    try:
        folders = {} if folders is None else folders
        if image.parent not in folders:
            folders[image.parent] = FolderFiles(_caption_siblings(image.parent))
        folder = folders[image.parent]
        return _entry(
            Path(row["path"]).expanduser(), image, image.stat(),
            folder.caption(image, row["caption_ext"]), folder.mask(image),
        )
    except (OSError, ValueError):
        return None


def note_own_edit(c: Any, did: str, before: list[int | None], after: list[int | None]) -> None:
    """Move the stored signature past a studio edit that already updated the index records.

    ``before`` and ``after`` are ``image_entry`` values of the touched images around the edit. A
    signature that was already out of date stays out of date, so the next check still re-indexes.
    """
    if None in before or None in after or sorted(before) == sorted(after):
        return
    with c.db.lock:
        row = c.db.fetchone("SELECT stats_json FROM datasets WHERE id=?", (did,))
        if row is None:
            return
        stats = json.loads(row["stats_json"] or "{}")
        stored = stats.get("_source_signature")
        if not isinstance(stored, str) or not stored.startswith(SIGNATURE_PREFIX):
            return
        try:
            value = int(stored[len(SIGNATURE_PREFIX):], 16)
        except ValueError:
            return
        stats["_source_signature"] = _encode(value - sum(before) + sum(after))
        c.db.update("datasets", did, {"stats_json": json.dumps(stats)})


def data_page(row: dict) -> str | None:
    if not row.get("project_id"):
        return None
    version = f"/v/{quote(row['version_id'], safe='')}" if row.get("version_id") else ""
    return f"/projects/{quote(row['project_id'], safe='')}{version}?step=data&data_step=datasets"


class DatasetRefresher:
    def __init__(self, context: Any):
        self.c = context
        self._lock = threading.Lock()
        self._checked: dict[str, float] = {}
        self._pending: set[str] = set()
        self._refreshing: set[str] = set()
        self._index_locks: dict[str, threading.Lock] = {}
        self._idle = threading.Condition(self._lock)
        self._closed = False
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dataset-refresh")

    def close(self) -> None:
        with self._lock:
            self._closed = True
        # A re-index already running finishes; queued checks are dropped.
        self.executor.shutdown(wait=True, cancel_futures=True)
        with self._idle:
            self._pending.clear()
            self._idle.notify_all()

    def wait(self, timeout: float = 30.0) -> bool:
        """Wait until no check or refresh is queued or running."""
        deadline = time.monotonic() + timeout
        with self._idle:
            while self._pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle.wait(remaining)
        return True

    def checked(self, did: str) -> None:
        """The folder was just listed, by a check or an index run."""
        with self._lock:
            self._checked[did] = time.monotonic()

    def refreshing(self, did: str) -> bool:
        with self._lock:
            return did in self._refreshing

    @contextmanager
    def index_lock(self, did: str) -> Iterator[None]:
        """One index run of a dataset at a time, whoever started it."""
        with self._lock:
            lock = self._index_locks.setdefault(did, threading.Lock())
        with lock:
            yield

    def request(self, row: dict, *, force: bool = False) -> None:
        """Check a dataset that a page is about to show, in the background."""
        did = row["id"]
        if row["index_status"] == "indexing":
            return
        with self._lock:
            if self._closed or did in self._pending:
                return
            last = self._checked.get(did)
            if not force and last is not None and time.monotonic() - last < CHECK_INTERVAL:
                return
            self._pending.add(did)
        try:
            self.executor.submit(self._check, did)
        except RuntimeError:  # shutting down
            self._done(did)

    def _done(self, did: str) -> None:
        with self._idle:
            self._pending.discard(did)
            self._refreshing.discard(did)
            self._idle.notify_all()

    def _check(self, did: str) -> None:
        from .routes_work import _index_dataset, _records_path

        try:
            row = self.c.db.fetchone("SELECT * FROM datasets WHERE id=?", (did,))
            if row is None or row["index_status"] == "indexing" or not self._may_refresh(row):
                return
            signature, listing = dataset_signature(row)
            self.checked(did)
            stats = json.loads(row["stats_json"] or "{}")
            if stats.get("_source_signature") == signature and (
                row["index_status"] == "failed"
                or (row["index_status"] == "ready" and _records_path(self.c, did).is_file())
            ):
                return
            with self._lock:
                self._refreshing.add(did)
            tasks = getattr(self.c, "background_tasks", None)
            task = (
                tasks.start(
                    "dataset_refresh", Path(row["path"]).name or row["path"], unit="images", link=data_page(row)
                )
                if tasks is not None
                else None
            )
            self.c.bus.publish("dataset.changed", {
                "dataset_id": did, "project_id": row["project_id"], "version_id": row.get("version_id"),
                "reason": "refreshing",
            })
            error = None
            try:
                _index_dataset(self.c, did, listing=listing, signature=signature, refresh=True, task=task)
                current = self.c.db.fetchone("SELECT index_status,stats_json FROM datasets WHERE id=?", (did,))
                if current is not None and current["index_status"] == "failed":
                    error = json.loads(current["stats_json"] or "{}").get("error") or "索引失败"
            except Exception as exc:  # noqa: BLE001 - a failed refresh keeps the previous index
                log.exception("dataset refresh failed")
                error = str(exc)
            finally:
                if task is not None:
                    tasks.finish(task, error=error)
        except Exception:  # noqa: BLE001 - checks must never break the reading page
            log.exception("dataset check failed")
        finally:
            self._done(did)

    def _may_refresh(self, row: dict) -> bool:
        """Leave datasets alone while their version's files are being changed or deleted."""
        if row.get("version_id"):
            version = self.c.db.fetchone(
                "SELECT busy,status FROM project_versions WHERE id=?", (row["version_id"],)
            )
            if version is None or version["busy"] or version["status"] != "ready":
                return False
        if row.get("project_id"):
            from .project_deletion import deleting

            if deleting(self.c, row["project_id"]):
                return False
        return True


def refresher(c: Any) -> DatasetRefresher:
    service = getattr(c, "dataset_refresh", None)
    if service is None:
        with _create_lock:
            service = getattr(c, "dataset_refresh", None)
            if service is None:
                service = c.dataset_refresh = DatasetRefresher(c)
    return service
