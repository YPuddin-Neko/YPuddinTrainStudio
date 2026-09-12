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

from . import (
    errors,
    routes_core,
    routes_credentials,
    routes_dataset_masks,
    routes_dataset_paint,
    routes_dataset_pipeline,
    routes_environment,
    routes_model_downloads,
    routes_model_recommendations,
    routes_regularization,
    routes_work,
)
from .bus import EventBus
from .context import ServiceContext
from .dataset_pipeline import DatasetPipeline
from .db import Database
from .environment import EnvironmentManager
from .model_downloads import ModelDownloads
from .regularization import RegularizationManager
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
    model_downloads = ModelDownloads(context)
    environment = EnvironmentManager(context)
    dataset_pipeline = DatasetPipeline(context)
    regularization = RegularizationManager(context, credentials=model_downloads.credentials)

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
            await asyncio.to_thread(regularization.close)
            await asyncio.to_thread(model_downloads.close)
            await asyncio.to_thread(environment.close)
            await asyncio.to_thread(dataset_pipeline.close)
            await asyncio.to_thread(context.versions.close)
            db.close()

    app = FastAPI(
        title="YPuddin Train Studio",
        version=ypuddin.__version__,
        lifespan=lifespan,
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
    )
    app.state.ctx = context
    app.state.model_downloads = model_downloads
    app.state.environment = environment
    app.state.dataset_pipeline = dataset_pipeline
    app.state.regularization = regularization
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
    app.include_router(routes_model_downloads.router, prefix="/api")
    app.include_router(routes_model_recommendations.router, prefix="/api")
    app.include_router(routes_dataset_masks.router, prefix="/api")
    app.include_router(routes_dataset_paint.router, prefix="/api")
    app.include_router(routes_dataset_pipeline.router, prefix="/api")
    app.include_router(routes_environment.router, prefix="/api")
    app.include_router(routes_regularization.router, prefix="/api")
    app.include_router(routes_credentials.router, prefix="/api")

    dist = Path(frontend_dist) if frontend_dist else Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if (dist / "index.html").exists():
        _mount_spa(app, dist)
    return app


def _mount_spa(app: FastAPI, dist: Path) -> None:
    """Serve the built frontend with history-API fallback: real files as-is, every other non-API
    path gets ``index.html`` so deep links (``/jobs/j_123``) survive a refresh."""
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    index = dist / "index.html"
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=str(dist / "assets")), name="frontend-assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str):  # noqa: ANN202
        if path.startswith("api/"):
            raise HTTPException(status_code=404)
        candidate = (dist / path).resolve() if path else index
        if path and candidate.is_file() and dist.resolve() in candidate.parents:
            return FileResponse(str(candidate))
        return FileResponse(str(index))
