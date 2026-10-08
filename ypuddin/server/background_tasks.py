"""Work that keeps running outside any page, listed by the task center.

Services register long work here (server-side upload imports, project deletion,
dataset refreshes). Services that already keep their own status, such as model
downloads or environment and trainer updates, are read through sources.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .bus import EventBus

log = logging.getLogger(__name__)

EVENT = "background.changed"
# Finished work stays listed for a while; failures stay until dismissed.
KEEP_FINISHED_SECONDS = 600.0
UPDATE_INTERVAL = 0.5
# How often services that keep their own status are read for changes to publish.
SOURCE_INTERVAL = 1.0

State = Literal["running", "completed", "failed", "cancelled"]


class BackgroundTask(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    # What the work is: dataset_upload, model_download, project_delete, dataset_refresh, environment, trainer_update.
    kind: str
    # What it works on, such as a project name or a file name; the page words the title from kind and subject.
    subject: str | None = None
    state: State = "running"
    done: float | None = None
    total: float | None = None
    # bytes, files or items; None when only the state is known.
    unit: str | None = None
    detail: str | None = None
    # In-app route that shows the work.
    link: str | None = None
    cancellable: bool = False
    started_at: float
    finished_at: float | None = None
    error: str | None = None
    # Units per second while it runs, when known.
    rate: float | None = None


class BackgroundTasks:
    def __init__(self, bus: EventBus | None = None, clock: Callable[[], float] = time.time) -> None:
        self._bus = bus
        self._clock = clock
        # Failures a service recorded before this start are not listed again.
        self.started_at = clock()
        self._lock = threading.RLock()
        self._tasks: dict[str, BackgroundTask] = {}
        self._cancel: dict[str, Callable[[], None]] = {}
        self._published: dict[str, float] = {}
        self._samples: dict[str, tuple[float, float]] = {}
        self._sources: list[tuple[Callable[[], list[dict[str, Any]]], Callable[[str], Any] | None]] = []
        self._source_of: dict[str, Callable[[str], Any] | None] = {}
        self._seen: dict[str, tuple[Any, ...]] = {}
        self._dismissed: set[str] = set()

    def start(
        self,
        kind: str,
        subject: str | None = None,
        *,
        total: float | None = None,
        unit: str | None = None,
        link: str | None = None,
        detail: str | None = None,
        cancel: Callable[[], None] | None = None,
        task_id: str | None = None,
    ) -> str:
        """Register running work; ``cancel`` asks it to stop, and the owner then calls ``finish``."""
        task = BackgroundTask(
            id=task_id or f"bg_{uuid.uuid4().hex[:12]}",
            kind=kind,
            subject=subject,
            done=0 if total is not None else None,
            total=total,
            unit=unit,
            link=link,
            detail=detail,
            cancellable=cancel is not None,
            started_at=self._clock(),
        )
        with self._lock:
            self._prune()
            self._tasks[task.id] = task
            if cancel is not None:
                self._cancel[task.id] = cancel
        self._publish(task)
        return task.id

    def update(
        self,
        task_id: str,
        *,
        done: float | None = None,
        total: float | None = None,
        detail: str | None = None,
        force: bool = False,
    ) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.state != "running":
                return
            changes: dict[str, Any] = {}
            now = self._clock()
            if done is not None:
                changes["done"] = done
                sampled, before = self._samples.get(task_id, (task.started_at, task.done or 0))
                if done > before and now - sampled >= UPDATE_INTERVAL:
                    measured = (done - before) / (now - sampled)
                    changes["rate"] = measured if task.rate is None else 0.3 * measured + 0.7 * task.rate
                    self._samples[task_id] = (now, done)
            if total is not None:
                changes["total"] = total
            if detail is not None:
                changes["detail"] = detail
            task = self._tasks[task_id] = task.model_copy(update=changes)
            if not force and now - self._published.get(task_id, 0.0) < UPDATE_INTERVAL:
                return
        self._publish(task)

    def finish(self, task_id: str, *, error: str | None = None, cancelled: bool = False) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.state != "running":
                return
            state: State = "cancelled" if cancelled else "failed" if error else "completed"
            changes: dict[str, Any] = {"state": state, "finished_at": self._clock(), "error": error, "rate": None}
            if state == "completed" and task.total is not None:
                changes["done"] = task.total
            task = self._tasks[task_id] = task.model_copy(update=changes)
            self._cancel.pop(task_id, None)
            self._samples.pop(task_id, None)
        self._publish(task)

    def cancel(self, task_id: str) -> bool:
        """Ask running work to stop; False when it cannot be cancelled."""
        with self._lock:
            owned = task_id in self._tasks
            task = self._tasks.get(task_id)
            request = self._cancel.get(task_id)
        if not owned:
            task = next((item for item in self._from_sources() if item.id == task_id), None)
            stop = self._source_of.get(task_id)
            if task is None or task.state != "running" or not task.cancellable or stop is None:
                return False
            stop(task_id)
            return True
        if task is None or task.state != "running" or request is None:
            return False
        request()
        return True

    def dismiss(self, task_id: str) -> bool:
        """Remove finished work from the list; running work cannot be dismissed."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                if task.state == "running":
                    return False
                del self._tasks[task_id]
                self._published.pop(task_id, None)
        if task is None:
            # A service's own record stays; it is only no longer listed here.
            task = next((item for item in self._from_sources() if item.id == task_id), None)
            if task is None or task.state == "running":
                return False
            with self._lock:
                self._dismissed.add(task_id)
                self._seen.pop(task_id, None)
        if self._bus is not None:
            self._bus.publish(EVENT, {"id": task_id, "dismissed": True})
        return True

    def get(self, task_id: str) -> BackgroundTask | None:
        with self._lock:
            task = self._tasks.get(task_id)
        if task is not None:
            return task
        return next((item for item in self._from_sources() if item.id == task_id), None)

    def add_source(
        self, source: Callable[[], list[dict[str, Any]]], *, cancel: Callable[[str], Any] | None = None
    ) -> None:
        """A callable returning the current work of a service that keeps its own status.

        It is read every ``SOURCE_INTERVAL`` while the service runs, so it must only copy in-memory
        status. ``cancel`` receives the id of a task the source marked cancellable.
        """
        with self._lock:
            self._sources.append((source, cancel))

    def list(self) -> list[BackgroundTask]:
        with self._lock:
            self._prune()
            tasks = list(self._tasks.values())
        tasks += self._from_sources()
        return sorted(tasks, key=lambda task: (task.state != "running", -task.started_at))

    def _from_sources(self) -> list[BackgroundTask]:
        with self._lock:
            sources = list(self._sources)
            dismissed = set(self._dismissed)
        tasks = []
        for source, cancel in sources:
            try:
                found = [BackgroundTask.model_validate(item) for item in source()]
            except Exception:  # noqa: BLE001 - one failing service must not hide the others
                log.exception("background task source failed")
                continue
            for task in found:
                if task.id in dismissed:
                    continue
                with self._lock:
                    self._source_of[task.id] = cancel
                tasks.append(task)
        return tasks

    def poll_sources(self) -> None:
        """Publish what changed in services that keep their own status, as owned work does."""
        current = {task.id: task for task in self._from_sources()}
        signatures = {
            task_id: (task.state, task.done, task.total, task.detail, task.error, task.subject, task.cancellable)
            for task_id, task in current.items()
        }
        with self._lock:
            previous, self._seen = self._seen, signatures
            self._source_of = {task_id: self._source_of.get(task_id) for task_id in current}
        for task_id, task in current.items():
            if previous.get(task_id) != signatures[task_id]:
                self._publish(task)
        if self._bus is not None:
            for task_id in previous.keys() - current.keys():
                self._bus.publish(EVENT, {"id": task_id, "dismissed": True})

    async def watch(self, interval: float = SOURCE_INTERVAL) -> None:
        while True:
            await asyncio.sleep(interval)
            await asyncio.to_thread(self.poll_sources)

    def _prune(self) -> None:
        cutoff = self._clock() - KEEP_FINISHED_SECONDS
        for task_id, task in list(self._tasks.items()):
            if task.state in ("completed", "cancelled") and (task.finished_at or 0) < cutoff:
                del self._tasks[task_id]
                self._published.pop(task_id, None)
                self._samples.pop(task_id, None)

    def _publish(self, task: BackgroundTask) -> None:
        with self._lock:
            self._published[task.id] = self._clock()
        if self._bus is not None:
            self._bus.publish(EVENT, task.model_dump())


def _recent(row: dict[str, Any], since: float, now: float) -> bool:
    """Completed or cancelled work shows for a while; failures since the service started stay."""
    finished = row.get("finished_at") or 0
    if row["state"] == "failed":
        return finished >= since
    return now - finished < KEEP_FINISHED_SECONDS


def model_download_source(downloads: Any, since: float) -> Callable[[], list[dict[str, Any]]]:
    """Base model downloads (``ModelDownloads.list()``) as task center work."""

    def tasks() -> list[dict[str, Any]]:
        now, found = time.time(), []
        for row in downloads.list():
            active = row["status"] in ("queued", "downloading")
            task = {
                "id": row["id"],
                "kind": "model_download",
                "subject": row["filename"],
                "state": "running" if active else row["status"],
                "done": row.get("downloaded_bytes") or 0,
                "total": row.get("total_bytes") or row.get("expected_size"),
                "unit": "bytes",
                "rate": (row.get("bytes_per_second") or None) if row["status"] == "downloading" else None,
                "detail": "queued" if row["status"] == "queued" else None,
                "link": "/settings/environment?tab=models",
                "cancellable": active,
                "started_at": row["created_at"],
                "finished_at": row.get("finished_at"),
                "error": row.get("error"),
            }
            if active or _recent(task, since, now):
                found.append(task)
        return found

    return tasks


def tts_model_download_source(downloads: Any, since: float) -> Callable[[], list[dict[str, Any]]]:
    """Complete speech model packages retain their own persistent download records."""

    def tasks() -> list[dict[str, Any]]:
        current, found = time.time(), []
        for download in downloads.list():
            row = download.model_dump()
            active = row["status"] in ("queued", "downloading", "verifying")
            task = {
                "id": row["id"],
                "kind": "model_download",
                "subject": row["name"],
                "state": "running" if active else row["status"],
                "done": row["downloaded_bytes"],
                "total": row["total_bytes"],
                "unit": "bytes",
                "rate": row["bytes_per_second"] if row["phase"] == "download" else None,
                "detail": row["status"] if row["status"] in ("queued", "verifying") else None,
                "link": "/settings/environment?tab=models&type=tts",
                "cancellable": active,
                "started_at": row["created_at"],
                "finished_at": row["finished_at"],
                "error": row["error"],
            }
            if active or _recent(task, since, current):
                found.append(task)
        return found

    return tasks


def vision_download_source(vision: Any, since: float) -> Callable[[], list[dict[str, Any]]]:
    """Tagging and mask detection model downloads (``VisionModels.tasks``) as task center work."""
    from .model_catalog import VISION_MODELS

    def tasks() -> list[dict[str, Any]]:
        with vision.lock:
            rows = [(model_id, dict(row)) for model_id, row in vision.tasks.items()]
        now, found = time.time(), []
        for model_id, row in rows:
            status = row.get("status")
            active = status in ("queued", "downloading", "verifying")
            task = {
                "id": f"vision-{model_id}",
                "kind": "model_download",
                "subject": VISION_MODELS.get(model_id, {}).get("label", model_id),
                "state": "running" if active else status,
                "done": row.get("downloaded_bytes") or 0,
                "total": row.get("total_bytes"),
                "unit": "bytes",
                "rate": (row.get("bytes_per_second") or None) if status == "downloading" else None,
                "detail": status if status in ("queued", "verifying") else None,
                "link": "/settings/environment?tab=tagging",
                "cancellable": active,
                "started_at": row.get("started_at") or now,
                "finished_at": row.get("finished_at"),
                "error": row.get("error"),
            }
            if active or _recent(task, since, now):
                found.append(task)
        return found

    return tasks
