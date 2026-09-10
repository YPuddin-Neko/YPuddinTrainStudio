"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import ypuddin

from . import errors, routes_core, routes_work
from .bus import EventBus
from .context import ServiceContext
from .db import Database
from .supervisor import JobSupervisor

log = logging.getLogger(__name__)


def create_app(
    data_root: str | Path = "studio_data",
    *,
    frontend_dist: str | Path | None = None,
    poll_interval: float = 0.5,
) -> FastAPI:
    root = Path(data_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    db = Database(root / "studio.db")
    bus = EventBus()
    supervisor = JobSupervisor(db, bus, root, poll_interval=poll_interval)
    context = ServiceContext(data_root=root, db=db, bus=bus, supervisor=supervisor)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        bus.attach_loop(asyncio.get_running_loop())
        await supervisor.start()
        stats_task = asyncio.create_task(routes_core.stats_publisher(context))
        try:
            yield
        finally:
            stats_task.cancel()
            await supervisor.stop()
            db.close()

    app = FastAPI(
        title="YPuddin Train Studio",
        version=ypuddin.__version__,
        lifespan=lifespan,
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
    )
    app.state.ctx = context
    errors.install(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Trace-Id"],
    )
    app.include_router(routes_core.router, prefix="/api")
    app.include_router(routes_work.router, prefix="/api")

    dist = Path(frontend_dist) if frontend_dist else Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")
    return app
