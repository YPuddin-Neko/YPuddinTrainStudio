"""Service-log warnings for API requests that are slow to answer.

A request counts until its response starts, so an open event stream is not slow. One still waiting
after the threshold is logged then, so a request that never finishes shows up too; it is logged again
with its total time when it is answered. Only the method and path are logged.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger("ypuddin.server.requests")

SLOW_SECONDS = 3.0


class SlowRequestLog:
    """ASGI middleware timing every ``/api`` request on the event loop (no locks, no extra tasks)."""

    def __init__(self, app: Any, threshold: float | None = None) -> None:
        self.app = app
        self.threshold = SLOW_SECONDS if threshold is None else threshold
        self.waiting = 0  # requests whose response has not started

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or not scope["path"].startswith("/api/"):
            await self.app(scope, receive, send)
            return
        method, path = scope["method"], scope["path"]
        loop = asyncio.get_running_loop()
        started = loop.time()
        answered = False
        self.waiting += 1

        def still_waiting() -> None:
            log.warning(
                "%s %s has been waiting %.1f s for its answer (%d requests waiting)",
                method,
                path,
                loop.time() - started,
                self.waiting,
            )

        timer = loop.call_later(self.threshold, still_waiting)

        def answer() -> None:
            nonlocal answered
            if answered:
                return
            answered = True
            timer.cancel()
            self.waiting -= 1
            elapsed = loop.time() - started
            if elapsed >= self.threshold:
                log.warning(
                    "%s %s took %.1f s to answer (%d requests waiting)", method, path, elapsed, self.waiting
                )

        async def timed_send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                answer()
            await send(message)

        try:
            await self.app(scope, receive, timed_send)
        finally:
            answer()
