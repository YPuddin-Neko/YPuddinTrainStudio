"""In-process event bus with a replay ring for SSE ``Last-Event-ID`` resumption."""

from __future__ import annotations

import asyncio
import collections
import json
import threading
import time
from typing import Any


class EventBus:
    def __init__(self, history: int = 5000):
        self._history: collections.deque[dict[str, Any]] = collections.deque(maxlen=history)
        self._subs: set[asyncio.Queue] = set()
        self._seq = 0
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def close(self) -> None:
        """Wake event streams before the HTTP server waits for connections to finish."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            subs = list(self._subs)
        for queue in subs:
            if self._loop is not None:
                self._loop.call_soon_threadsafe(_close_subscription, queue)
            else:
                _close_subscription(queue)

    def publish(self, type_: str, data: dict[str, Any]) -> dict[str, Any]:
        """Thread-safe; may be called from supervisor threads."""
        with self._lock:
            self._seq += 1
            event = {"id": self._seq, "type": type_, "ts": time.time(), "data": data}
            self._history.append(event)
            subs = [] if self._closed else list(self._subs)
        loop = self._loop
        if loop is not None:
            for q in subs:
                loop.call_soon_threadsafe(_safe_put, q, event)
        return event

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2048)
        with self._lock:
            if self._closed:
                q.put_nowait(None)
            else:
                self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subs.discard(q)

    def replay(self, after_id: int) -> list[dict[str, Any]]:
        with self._lock:
            return [e for e in self._history if e["id"] > after_id]

    @staticmethod
    def format_sse(event: dict[str, Any]) -> str:
        return f"id: {event['id']}\nevent: {event['type']}\ndata: {json.dumps(event['data'], ensure_ascii=False, default=str)}\n\n"


def _safe_put(q: asyncio.Queue, event: dict[str, Any]) -> None:
    try:
        q.put_nowait(event)
    except asyncio.QueueFull:
        pass


def _close_subscription(queue: asyncio.Queue) -> None:
    # A slow browser can fill its queue. Shutdown must not be dropped behind
    # that backlog; the client will fetch current state after reconnecting.
    while not queue.empty():
        queue.get_nowait()
    queue.put_nowait(None)
