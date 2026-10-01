"""HTTP lifecycle integration for finite shutdown of browser event streams."""

from __future__ import annotations

import socket

import uvicorn

from .bus import EventBus


class StudioServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, *, event_bus: EventBus):
        super().__init__(config)
        self.event_bus = event_bus
        self.on_started = None

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        if self.started and self.on_started:
            self.on_started()

    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        # Uvicorn drains HTTP connections before sending lifespan.shutdown.
        # Close SSE here for both requested restarts and console/OS shutdown.
        self.event_bus.close()
        await super().shutdown(sockets=sockets)
