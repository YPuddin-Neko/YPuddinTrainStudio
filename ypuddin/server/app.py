"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

import ypuddin

from . import (
    errors,
    routes_background,
    routes_core,
    routes_credentials,
    routes_dataset_management,
    routes_dataset_masks,
    routes_dataset_overview,
    routes_dataset_paint,
    routes_dataset_pipeline,
    routes_environment,
    routes_model_downloads,
    routes_model_recommendations,
    routes_regularization,
    routes_site_downloads,
    routes_updates,
    routes_uploads,
    routes_vision,
    routes_vlm,
    routes_work,
    routes_xyz,
)
from .background_tasks import BackgroundTasks, model_download_source, vision_download_source
from .bus import EventBus
from .config_migration import migrate_dora_axis
from .context import ServiceContext
from .dataset_pipeline import DatasetPipeline
from .dataset_refresh import DatasetRefresher
from .db import Database
from .environment import EnvironmentManager
from .family_geometry import FamilyGeometry
from .job_layout import migrate_job_files
from .lifecycle import ServiceLifecycle
from .model_downloads import ModelDownloads
from .project_deletion import ProjectDeletions
from .regularization import RegularizationManager
from .request_timing import SlowRequestLog
from .site_downloads import SiteDownloadManager
from .supervisor import JobSupervisor
from .torch_environments import TorchEnvironments
from .trainer_install import TrainerInstaller
from .trainer_updates import TrainerUpdates
from .vision_downloads import VisionModels

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
    context.background_tasks = BackgroundTasks(bus)
    trainer_updates = TrainerUpdates(context)
    model_downloads = ModelDownloads(context)
    environment = EnvironmentManager(context)
    torch_environments = TorchEnvironments(context, environment)
    lifecycle = ServiceLifecycle(context, environment, torch_environments)
    lifecycle.model_downloads = model_downloads
    trainer_installer = TrainerInstaller(context, trainer_updates, lifecycle)
    # Before the pipeline, whose recovery may re-index datasets.
    context.dataset_refresh = DatasetRefresher(context)
    dataset_pipeline = DatasetPipeline(context)
    vision_models = VisionModels(context, credentials=model_downloads.credentials)
    lifecycle.vision_models = vision_models
    dataset_pipeline.vision = vision_models
    dataset_pipeline.credentials = model_downloads.credentials
    regularization = RegularizationManager(context, credentials=model_downloads.credentials)
    site_downloads = SiteDownloadManager(
        context, credentials=model_downloads.credentials, regularization=regularization
    )
    # After the services whose records a deletion removes; continues one a stop interrupted.
    context.project_deletions = ProjectDeletions(context)
    family_geometry = FamilyGeometry(db)
    tasks = context.background_tasks
    tasks.add_source(context.upload_sessions.background_tasks)
    tasks.add_source(model_download_source(model_downloads, tasks.started_at), cancel=model_downloads.cancel)
    tasks.add_source(
        vision_download_source(vision_models, tasks.started_at),
        cancel=lambda task_id: vision_models.cancel(task_id.removeprefix("vision-")),
    )

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        bus.attach_loop(asyncio.get_running_loop())
        # Move files of jobs saved before the jobs/ folder existed, before any of them can run.
        await asyncio.to_thread(migrate_job_files, context)
        # Historical DoRA configs keep their calculation axis before schema defaults apply.
        await asyncio.to_thread(migrate_dora_axis, context)
        await supervisor.start()
        stats_task = asyncio.create_task(routes_core.stats_publisher(context))

        async def expire_uploads() -> None:
            while True:
                await asyncio.sleep(60)
                await asyncio.to_thread(context.upload_sessions.prune)

        uploads_task = asyncio.create_task(expire_uploads())
        # Services with their own status reach the task center through these reads.
        sources_task = asyncio.create_task(context.background_tasks.watch())
        try:
            yield
        finally:
            await asyncio.to_thread(trainer_installer.close)
            bus.close()
            stats_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stats_task
            uploads_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await uploads_task
            sources_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sources_task
            await asyncio.to_thread(context.upload_sessions.close)
            await supervisor.stop()
            await asyncio.to_thread(site_downloads.close)
            await asyncio.to_thread(regularization.close)
            await asyncio.to_thread(model_downloads.close)
            await asyncio.to_thread(vision_models.close)
            await asyncio.to_thread(torch_environments.close)
            await asyncio.to_thread(environment.close)
            await asyncio.to_thread(context.project_deletions.close)
            await asyncio.to_thread(context.dataset_refresh.close)
            await asyncio.to_thread(dataset_pipeline.close)
            await asyncio.to_thread(context.versions.close)
            await asyncio.to_thread(family_geometry.close)
            db.close()

    app = FastAPI(
        title="YPuddin Train Studio",
        version=ypuddin.__version__,
        lifespan=lifespan,
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
    )
    app.state.ctx = context
    app.state.background_tasks = context.background_tasks
    app.state.model_downloads = model_downloads
    app.state.environment = environment
    app.state.torch_environments = torch_environments
    app.state.lifecycle = lifecycle
    app.state.dataset_pipeline = dataset_pipeline
    app.state.vision_models = vision_models
    app.state.regularization = regularization
    app.state.site_downloads = site_downloads
    app.state.trainer_updates = trainer_updates
    app.state.trainer_installer = trainer_installer
    app.state.family_geometry = family_geometry
    errors.install(app)

    # The counter's own arithmetic; an admission also holds the database lock, which orders it
    # against an update start reading the counter under that lock.
    counting = threading.Lock()

    @app.middleware("http")
    async def update_admission(request, call_next):
        mutation = request.url.path.startswith("/api/") and request.method not in {"GET", "HEAD", "OPTIONS"}
        mutation = mutation and request.url.path not in {"/api/updates/install", "/api/updates/check"}
        if not mutation:
            return await call_next(request)
        admitted = []

        def admit() -> bool:
            with db.lock:
                if db.get_kv("environment.maintenance", {}).get("trainer_update"):
                    return False
                with counting:
                    context._update_requests = getattr(context, "_update_requests", 0) + 1
                admitted.append(True)
                return True

        try:
            # Other work may hold the database lock for a while. A worker thread waits for it, so the
            # event loop keeps answering every other request; the wait is not abandoned on cancel.
            if not await run_in_threadpool(admit):
                return JSONResponse(status_code=409, content=errors.envelope("updates.in_progress", "训练器正在更新，请稍后再试。"))
            return await call_next(request)
        finally:
            if admitted:
                with counting:
                    context._update_requests -= 1
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Trace-Id"],
    )
    # Outermost, so the time includes every middleware's waiting.
    app.add_middleware(SlowRequestLog)
    app.include_router(routes_core.router, prefix="/api")
    app.include_router(routes_background.router, prefix="/api")
    app.include_router(routes_xyz.router, prefix="/api")
    app.include_router(routes_work.router, prefix="/api")
    app.include_router(routes_model_downloads.router, prefix="/api")
    app.include_router(routes_model_recommendations.router, prefix="/api")
    app.include_router(routes_dataset_masks.router, prefix="/api")
    app.include_router(routes_dataset_management.router, prefix="/api")
    app.include_router(routes_dataset_overview.router, prefix="/api")
    app.include_router(routes_dataset_paint.router, prefix="/api")
    app.include_router(routes_dataset_pipeline.router, prefix="/api")
    app.include_router(routes_environment.router, prefix="/api")
    app.include_router(routes_regularization.router, prefix="/api")
    app.include_router(routes_site_downloads.router, prefix="/api")
    app.include_router(routes_credentials.router, prefix="/api")
    app.include_router(routes_vision.router, prefix="/api")
    app.include_router(routes_vlm.router, prefix="/api")
    app.include_router(routes_updates.router, prefix="/api")
    app.include_router(routes_uploads.router, prefix="/api")

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
