"""Projects / datasets / jobs / artifacts endpoints."""

from __future__ import annotations

import asyncio
import io
import json
import math
import os
import re
import shutil
import tempfile
import unicodedata
import zipfile
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import psutil
from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from starlette.background import BackgroundTask

from ypuddin.config import DatasetSourceConfig, TrainConfig, deep_merge
from ypuddin.config.intervals import normalize_legacy_intervals
from ypuddin.config.io import absolute_paths
from ypuddin.data import IndexDB, scan_sources

from . import models as m
from .context import ServiceContext
from .dataset_uploads import UploadBatch, read_upload, staged_upload
from .db import new_id, now
from .environment import maintenance_reason
from .errors import ApiError, NotFound
from .gpu_metrics import device_metric_series
from .gpu_selection import GpuSelection, planning_devices, planning_memory, selection_error
from .hardware import gpu_info
from .import_progress import ImportProgress
from .job_logs import failure_record_bytes, missing_failure_record, read_log
from .job_paths import event_file, log_file, owned_job_directories
from .project_covers import cover_path, cover_url, read_cover_upload, remove_cover, replace_cover, thumbnail
from .sample_events import read_events, samples_with_loss
from .upload_sessions import CHUNK_BYTES, UploadSession, UploadSessionBody
from .versions import ACTIVE_JOBS, assert_version_writable, version_row

router = APIRouter()


def ctx(request: Request) -> ServiceContext:
    return request.app.state.ctx


def _page(items: list[Any], page: int, page_size: int) -> dict[str, Any]:
    start = (page - 1) * page_size
    return {
        "items": items[start : start + page_size],
        "total": len(items),
        "page": page,
        "page_size": page_size,
    }


# --------------------------------------------------------------------------- projects
class ProjectBody(BaseModel):
    id: str | None = Field(None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_]+$")
    name: str
    note: str = ""
    category: str | None = Field(None, max_length=64)
    family: Literal["anima", "krea2", "sdxl", "flux2", "toy"] = "anima"

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, value):
        return _category(value)

    @field_validator("id")
    @classmethod
    def safe_project_id(cls, value):
        if value is not None and re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])", value):
            raise ValueError("project id cannot be a Windows device name")
        return value


class ProjectPatch(BaseModel):
    name: str | None = None
    note: str | None = None
    archived: bool | None = None
    active_version_id: str | None = None
    category: str | None = Field(None, max_length=64)

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, value):
        return _category(value)


def _category(value: Any) -> Any:
    if value is None or not isinstance(value, str):
        return value
    value = unicodedata.normalize("NFC", value).strip()
    if any(unicodedata.category(char).startswith("C") for char in value):
        raise ValueError("category cannot contain control characters")
    return value or None


def _training_images(datasets: list[dict[str, Any]]) -> int | None:
    """Training images of the active version, or None while any source is not indexed."""
    total = 0
    for row in datasets:
        if row["is_reg"]:
            continue
        stats = json.loads(row["stats_json"] or "{}")
        if row["index_status"] != "ready" or not isinstance(stats.get("images"), int):
            return None
        total += stats["images"]
    return total


def _latest_training(c: ServiceContext, project_id: str) -> dict[str, Any] | None:
    """The run a project card reports: an active one first, then waiting, then the newest."""
    job = c.db.fetchone(
        "SELECT id,name,status,progress_json,created_at,finished_at,error FROM jobs"
        " WHERE project_id=? AND type='train' ORDER BY CASE"
        " WHEN status IN ('running','pausing','cancelling') THEN 0 WHEN status='paused' THEN 1"
        " WHEN status IN ('queued','scheduled') THEN 2 ELSE 3 END, created_at DESC, id DESC LIMIT 1",
        (project_id,),
    )
    if not job:
        return None
    progress = json.loads(job["progress_json"] or "{}")
    return {
        "id": job["id"],
        "name": job["name"],
        "status": job["status"],
        "step": progress.get("step"),
        "total_steps": progress.get("total_steps"),
        "created_at": job["created_at"],
        "finished_at": job["finished_at"],
        "error": job["error"],
    }


def _project_row(c: ServiceContext, r: dict[str, Any]) -> dict[str, Any]:
    from .artifact_inventory import artifact_count
    from .family_config import version_family
    from .project_deletion import public_state

    ds = c.db.fetchall(
        "SELECT id,is_reg,stats_json,index_status FROM datasets WHERE version_id=?", (r["active_version_id"],)
    )
    jobs = c.db.fetchone("SELECT COUNT(*) AS n FROM jobs WHERE project_id=?", (r["id"],))["n"]
    arts = artifact_count(c.db, project_id=r["id"])
    version = (
        c.db.fetchone("SELECT name,number FROM project_versions WHERE id=?", (r["active_version_id"],))
        if r["active_version_id"]
        else None
    )
    return {
        **{key: value for key, value in r.items() if key != "cover_key"},
        "category": r.get("category"),
        "cover_url": cover_url(c, r),
        "active_family": version_family(c, c.resolve_version(r["id"], r["active_version_id"])),
        "active_version_name": version["name"] if version else None,
        "active_version_number": version["number"] if version else None,
        "archived": bool(r["archived"]),
        "dataset_ids": [d["id"] for d in ds],
        "image_count": _training_images(ds),
        "version_count": c.db.fetchone(
            "SELECT count(*) n FROM project_versions WHERE project_id=?", (r["id"],)
        )["n"],
        "stats": {"jobs": jobs, "artifacts": arts},
        "latest_job": _latest_training(c, r["id"]),
        # A deletion running in the background, or one that stopped and can be retried.
        "deletion": public_state(c, r["id"]),
    }


@router.get("/projects", response_model=list[m.Project] | m.ProjectPage, response_model_exclude_unset=True)
def list_projects(
    include_archived: bool = False,
    q: str = Query("", max_length=500),
    category: str | None = Query(None, max_length=64),
    uncategorized: bool = False,
    archived: bool | None = None,
    page: int | None = Query(None, ge=1),
    page_size: int = Query(24, ge=1, le=200),
    c: ServiceContext = Depends(ctx),
) -> list[dict[str, Any]] | dict[str, Any]:
    where, values = [], []
    if archived is not None or not include_archived:
        where.append("archived=?")
        values.append(int(bool(archived)))
    if q.strip():
        where.append("(name LIKE ? ESCAPE '\\' OR id LIKE ? ESCAPE '\\' OR note LIKE ? ESCAPE '\\')")
        pattern = "%" + q.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        values.extend([pattern] * 3)
    try:
        normalized = _category(category)
    except ValueError as exc:
        raise ApiError(str(exc), code="project.category") from exc
    if normalized and uncategorized:
        raise ApiError("choose one category filter", code="project.category")
    if normalized:
        where.append("category=?")
        values.append(normalized)
    elif uncategorized:
        where.append("category IS NULL")
    sql = " FROM projects" + (" WHERE " + " AND ".join(where) if where else "")
    with c.db.lock:
        total = c.db.fetchone("SELECT count(*) n" + sql, tuple(values))["n"]
        ordered = "SELECT *" + sql + " ORDER BY created_at DESC,id"
        rows = c.db.fetchall(
            ordered + (" LIMIT ? OFFSET ?" if page is not None else ""),
            tuple(values) + ((page_size, (page - 1) * page_size) if page is not None else ()),
        )
        items = [_project_row(c, row) for row in rows]
        return (
            {"items": items, "total": total, "page": page, "page_size": page_size}
            if page is not None
            else items
        )


@router.get("/project-categories", response_model=m.ProjectCategories)
def project_categories(c: ServiceContext = Depends(ctx)) -> dict:
    with c.db.lock:
        return {
            "items": c.db.fetchall(
                "SELECT category name,count(*) count FROM projects WHERE category IS NOT NULL GROUP BY category ORDER BY lower(category),category"
            ),
            "uncategorized": c.db.fetchone("SELECT count(*) n FROM projects WHERE category IS NULL")["n"],
            "total": c.db.fetchone("SELECT count(*) n FROM projects")["n"],
        }


@router.post("/projects", status_code=201, response_model=m.Project, response_model_exclude_unset=True)
def create_project(body: ProjectBody, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    with c.db.lock:
        pid = body.id or new_id("p")
        container = c.data_root / "project"
        target = container / pid
        if c.db.fetchone("SELECT id FROM projects WHERE lower(id)=lower(?)", (pid,)) or (
            container.exists() and any(path.name.casefold() == pid.casefold() for path in container.iterdir())
        ):
            raise ApiError("project id or directory already exists", code="project.duplicate", status=409)
        if container.is_symlink():
            raise ApiError(
                "project directory cannot be redirected with symbolic links", code="project.path", status=409
            )
        t = now()
        created = False
        c.db.execute("SAVEPOINT create_project")
        try:
            c.db.insert(
                "projects",
                {
                    "id": pid,
                    "name": body.name,
                    "note": body.note,
                    "category": body.category,
                    "archived": 0,
                    "layout_version": 2,
                    "created_at": t,
                    "updated_at": t,
                },
            )
            vid = new_id("v")
            c.db.insert(
                "project_versions",
                {
                    "id": vid,
                    "project_id": pid,
                    "number": 1,
                    "name": "v1",
                    "note": "",
                    "created_at": t,
                    "updated_at": t,
                },
            )
            c.db.update("projects", pid, {"active_version_id": vid})
            target.mkdir(parents=True, exist_ok=False)
            created = True
            root = c.version_dir(pid, vid)
            for name in ("traindata", "reg", "samples", "output", "jobs", "cache"):
                (root / name).mkdir(parents=True, exist_ok=True)
            from .family_config import initial_family_config

            initial = initial_family_config(c, body.family)
            initial = deep_merge(
                initial,
                {
                    "checkpoint": {"output_dir": str(c.default_runs_dir(pid, vid))},
                    "dataset": {"cache_dir": str(c.version_dir(pid, vid) / "cache")},
                },
            )
            _write_project_config(c, pid, initial, vid)
            c.db.execute("RELEASE SAVEPOINT create_project")
        except BaseException:
            c.db.execute("ROLLBACK TO SAVEPOINT create_project")
            c.db.execute("RELEASE SAVEPOINT create_project")
            if created:
                shutil.rmtree(target, ignore_errors=True)
            raise
        return _project_row(c, c.db.fetchone("SELECT * FROM projects WHERE id=?", (pid,)))


def _get_project(c: ServiceContext, pid: str) -> dict[str, Any]:
    r = c.db.fetchone("SELECT * FROM projects WHERE id=?", (pid,))
    if not r:
        raise NotFound(f"project {pid} not found", code="project.not_found")
    return r


def _assert_not_deleting(c: ServiceContext, pid: str) -> None:
    from .project_deletion import deleting

    if deleting(c, pid):
        raise ApiError("这个项目正在删除。", code="project.deleting", status=409)


@router.get("/projects/{pid}", response_model=m.Project, response_model_exclude_unset=True)
def get_project(pid: str, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    return _project_row(c, _get_project(c, pid))


@router.patch("/projects/{pid}", response_model=m.Project, response_model_exclude_unset=True)
def patch_project(pid: str, body: ProjectPatch, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    with c.db.lock:
        _get_project(c, pid)
        _assert_not_deleting(c, pid)
        fields = {
            k: (int(v) if isinstance(v, bool) else v) for k, v in body.model_dump().items() if v is not None
        }
        if "category" in body.model_fields_set:
            fields["category"] = body.category
        if body.active_version_id:
            assert_version_writable(c, pid, body.active_version_id)
        fields["updated_at"] = now()
        c.db.update("projects", pid, fields)
        return _project_row(c, _get_project(c, pid))


@router.post(
    "/projects/{pid}/cover",
    response_model=m.Project,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["file"],
                        "properties": {
                            "file": {"type": "string", "format": "binary"},
                            "crop": {
                                "type": "string",
                                "description": (
                                    "Optional JSON object with normalized x, y, width and height (0–1), "
                                    "relative to the EXIF-oriented image."
                                ),
                                "example": '{"x":0,"y":0.1,"width":1,"height":0.8}',
                            },
                        },
                    }
                }
            },
        }
    },
)
async def upload_project_cover(pid: str, request: Request, c: ServiceContext = Depends(ctx)) -> dict:
    original = _get_project(c, pid)
    data, crop = await read_cover_upload(request)
    encoded = await run_in_threadpool(thumbnail, data, crop)

    def save():
        with c.db.lock:
            row = _get_project(c, pid)
            _assert_not_deleting(c, pid)
            if row["created_at"] != original["created_at"]:
                raise ApiError(
                    "project was replaced during upload; retry", code="project.cover_conflict", status=409
                )
            try:
                replace_cover(c, row, encoded)
            except OSError as exc:
                raise ApiError(
                    "could not save the cover; the existing cover is preserved",
                    code="project.cover_write",
                    status=500,
                ) from exc
            return _project_row(c, _get_project(c, pid))

    return await run_in_threadpool(save)


@router.get(
    "/projects/{pid}/cover",
    response_class=Response,
    responses={200: {"content": {"image/webp": {"schema": {"type": "string", "format": "binary"}}}}},
)
def get_project_cover(pid: str, c: ServiceContext = Depends(ctx)) -> Response:
    with c.db.lock:
        path = cover_path(c, _get_project(c, pid))
        if path is None:
            raise NotFound("project has no cover", code="project.cover_not_found")
        data = path.read_bytes()
    return Response(
        data,
        media_type="image/webp",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


@router.delete("/projects/{pid}/cover", response_model=m.Project)
def delete_project_cover(pid: str, c: ServiceContext = Depends(ctx)) -> dict:
    with c.db.lock:
        _assert_not_deleting(c, pid)
        remove_cover(c, _get_project(c, pid))
        return _project_row(c, _get_project(c, pid))


@router.get("/projects/{pid}/storage", response_model=m.ProjectStorage, response_model_exclude_unset=True)
def project_storage(pid: str, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    """Every folder deleting the project removes, with sizes and what keeps any of them."""
    from .project_deletion import deletions

    return deletions(c).storage(pid)


@router.delete(
    "/projects/{pid}", response_model=m.ProjectDeletionStarted, response_model_exclude_unset=True
)
def delete_project(
    pid: str,
    delete_files: bool = False,
    skip: list[str] = Query([], description="Listed folders the user chose to keep, such as an unplugged drive."),
    c: ServiceContext = Depends(ctx),
) -> dict[str, Any]:
    from .project_deletion import _project_busy, deletions

    service = deletions(c)
    if delete_files:
        # Files go in the background; the project is marked as deleting until its records go too.
        return {"ok": True, "task_id": service.start(pid, skip)}
    with c.db.lock:
        project = _get_project(c, pid)
        if error := _project_busy(c, pid, project):
            raise error
        records = [_records_path(c, row["id"]) for row in c.db.fetchall("SELECT id FROM datasets WHERE project_id=?", (pid,))]
        c.db.execute("DELETE FROM jobs WHERE project_id=?", (pid,))
        c.db.execute("DELETE FROM artifacts WHERE project_id=?", (pid,))
        c.db.execute("DELETE FROM project_deletions WHERE project_id=?", (pid,))
        c.db.delete("projects", pid)
    for record in records:
        record.unlink(missing_ok=True)
    return {"ok": True}


@router.get("/projects/{pid}/config")
def get_project_config(
    pid: str, c: ServiceContext = Depends(ctx), version_id: str | None = None
) -> dict[str, Any]:
    version = c.resolve_version(pid, version_id)
    if version["status"] != "ready":
        raise ApiError("version configuration is not ready", code="version.not_ready", status=409)
    f = c.config_path(pid, version_id)
    if f.exists():
        from .source_roles import normalize_source_roles

        raw = normalize_legacy_intervals(json.loads(f.read_text(encoding="utf-8")))
        return normalize_source_roles(c, pid, raw, version["id"])
    from .environment import environment_attention_default

    cfg = TrainConfig()
    cfg.model.attention = environment_attention_default(c)
    return deep_merge(
        cfg.to_dict(),
        {
            "checkpoint": {"output_dir": str(c.default_runs_dir(pid, version_id))},
            "dataset": {"cache_dir": str(c.version_dir(pid, version_id) / "cache")},
        },
    )


@router.put("/projects/{pid}/config")
def put_project_config(
    pid: str, body: dict[str, Any], c: ServiceContext = Depends(ctx), version_id: str | None = None
) -> dict[str, Any]:
    from ypuddin.train.native_resolution import clear_native_vram_resolution

    body = normalize_legacy_intervals(clear_native_vram_resolution(body))
    reindex = []
    with c.db.lock:
        version = assert_version_writable(c, pid, version_id)
        from .source_roles import normalize_source_roles

        body = normalize_source_roles(c, pid, body, version["id"])
        configured = {}
        purposes = {}
        for section in ("dataset", "validation"):
            value = body.get(section)
            for source in value.get("sources", []) if isinstance(value, dict) else []:
                if isinstance(source, dict) and source.get("path"):
                    path = str(Path(source["path"]).expanduser().resolve())
                    extension = _validate_caption_extension(source.get("caption_ext", "auto"))
                    if path in configured and configured[path] != extension:
                        raise ApiError(
                            "one source folder cannot have conflicting training/validation caption extensions",
                            code="dataset.caption_ext",
                        )
                    configured[path] = extension
                    if "is_reg" in source:
                        purpose = bool(source["is_reg"])
                        if path in purposes and purposes[path] != purpose:
                            raise ApiError(
                                "one source folder cannot have conflicting training/validation purposes",
                                code="dataset.purpose",
                            )
                        purposes[path] = purpose
        for dataset in c.db.fetchall(
            "SELECT id,path,caption_ext FROM datasets WHERE project_id=? AND version_id=?",
            (pid, version["id"]),
        ):
            extension = configured.get(str(Path(dataset["path"]).expanduser().resolve()))
            if extension is not None and extension != dataset["caption_ext"]:
                reindex.append((dataset["id"], extension))
        if reindex:
            assert_version_writable(c, pid, version["id"], data=True)
        _write_project_config(c, pid, body, version["id"])
        for dataset in c.db.fetchall(
            "SELECT id,path FROM datasets WHERE project_id=? AND version_id=?", (pid, version["id"])
        ):
            purpose = purposes.get(str(Path(dataset["path"]).expanduser().resolve()))
            if purpose is not None:
                c.db.update("datasets", dataset["id"], {"is_reg": int(purpose)})
        for did, extension in reindex:
            c.db.update("datasets", did, {"caption_ext": extension, "index_status": "indexing"})
            _records_path(c, did).unlink(missing_ok=True)
        c.db.update("project_versions", version["id"], {"updated_at": now()})
        c.db.update("projects", pid, {"updated_at": now()})
    for did, _ in reindex:
        _index_dataset(c, did)
    return body


class SourceRoleBody(BaseModel):
    config: dict[str, Any]


class SourceRole(BaseModel):
    path: str
    section: str
    is_reg: bool
    managed: bool
    root: str | None
    origin: str
    images: int | None


@router.post("/projects/{pid}/source-roles", response_model=list[SourceRole])
def project_source_roles(
    pid: str, body: SourceRoleBody, c: ServiceContext = Depends(ctx), version_id: str | None = None
) -> list[dict]:
    from .source_roles import describe_source_roles

    return describe_source_roles(c, pid, body.config, version_id)


class OutputBinding(BaseModel):
    directory_template: str
    name: str
    automatic_name: bool
    inherits_output_dir: bool


@router.post("/projects/{pid}/output-binding", response_model=OutputBinding)
def project_output_binding(
    pid: str, body: SourceRoleBody, c: ServiceContext = Depends(ctx), version_id: str | None = None
) -> dict[str, Any]:
    from .output_binding import output_binding

    return output_binding(c, pid, body.config, version_id)


def _validate_caption_extension(value: str) -> str:
    from ypuddin.data.index import IMAGE_EXTS

    if value == "auto":
        return value
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"\.[A-Za-z0-9][A-Za-z0-9._-]{0,31}", value)
        or value.lower() in IMAGE_EXTS
        or value.lower() in {".mask", ".mask.png"}
    ):
        raise ApiError(
            "caption_ext must be auto or a sidecar suffix such as .txt or .json, not an image, mask or path",
            code="dataset.caption_ext",
        )
    return value


def _write_project_config(
    c: ServiceContext, pid: str, body: dict[str, Any], version_id: str | None = None
) -> None:
    """Caller holds the DB lock to serialize config edits with dataset source registration."""
    body = normalize_legacy_intervals(body)
    d = c.config_path(pid, version_id).parent
    d.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=d, suffix=".tmp", delete=False) as fp:
        temporary = Path(fp.name)
        try:
            json.dump(body, fp, indent=2, ensure_ascii=False)
        except BaseException:
            fp.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(d / "config.json")
    finally:
        temporary.unlink(missing_ok=True)


# --------------------------------------------------------------------------- datasets
class VersionBody(BaseModel):
    name: str = Field(min_length=1, max_length=100, pattern=r".*\S.*")
    note: str = Field("", max_length=4000)
    source_version_id: str | None = None
    data_mode: Literal["copy", "empty"] = "copy"
    copy_config: bool = True
    family: Literal["anima", "krea2", "sdxl", "flux2", "toy"] | None = None


class VersionPatch(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=100, pattern=r".*\S.*")
    note: str | None = Field(None, max_length=4000)
    archived: bool | None = None


@router.get("/projects/{pid}/versions", response_model=list[m.ProjectVersion])
def list_versions(pid: str, include_archived: bool = True, c: ServiceContext = Depends(ctx)) -> list[dict]:
    _get_project(c, pid)
    sql = "SELECT * FROM project_versions WHERE project_id=?" + (
        "" if include_archived else " AND archived=0"
    )
    return [version_row(c, r) for r in c.db.fetchall(sql + " ORDER BY created_at", (pid,))]


@router.post("/projects/{pid}/versions", status_code=202, response_model=m.ProjectVersion)
def create_version(pid: str, body: VersionBody, c: ServiceContext = Depends(ctx)) -> dict:
    return c.versions.create(
        pid,
        body.name.strip(),
        body.note,
        body.source_version_id,
        body.data_mode,
        body.copy_config,
        family=body.family,
    )


@router.get("/projects/{pid}/versions/{vid}", response_model=m.ProjectVersion)
def get_version(pid: str, vid: str, c: ServiceContext = Depends(ctx)) -> dict:
    return version_row(c, c.resolve_version(pid, vid))


@router.patch("/projects/{pid}/versions/{vid}", response_model=m.ProjectVersion)
def patch_version(pid: str, vid: str, body: VersionPatch, c: ServiceContext = Depends(ctx)) -> dict:
    with c.db.lock:
        row = c.resolve_version(pid, vid)
        if row["busy"] or row["status"] == "copying":
            raise ApiError("wait for version copy to finish", code="version.busy", status=409)
        fields = {
            k: (int(v) if isinstance(v, bool) else v.strip() if k == "name" else v)
            for k, v in body.model_dump().items()
            if v is not None
        }
        if fields.get("name") and c.db.fetchone(
            "SELECT id FROM project_versions WHERE project_id=? AND name=? AND id<>?",
            (pid, fields["name"], vid),
        ):
            raise ApiError("a version with this name already exists", code="version.duplicate", status=409)
        next_active = None
        if body.archived:
            if c.db.fetchone(f"SELECT id FROM jobs WHERE version_id=? AND status IN {ACTIVE_JOBS}", (vid,)):
                raise ApiError("version has queued or running jobs", code="version.jobs_busy", status=409)
            if _get_project(c, pid)["active_version_id"] == vid:
                fallback = c.db.fetchone(
                    "SELECT id FROM project_versions WHERE project_id=? AND id<>? AND archived=0 "
                    "AND status='ready' AND busy IS NULL ORDER BY created_at DESC LIMIT 1",
                    (pid, vid),
                )
                if not fallback:
                    raise ApiError(
                        "keep at least one ready, unarchived version", code="version.active", status=409
                    )
                next_active = fallback["id"]
        fields["updated_at"] = now()
        c.db.execute("BEGIN IMMEDIATE")
        try:
            if next_active:
                c.db.update("projects", pid, {"active_version_id": next_active, "updated_at": now()})
            c.db.update("project_versions", vid, fields)
            c.db.execute("COMMIT")
        except BaseException:
            c.db.execute("ROLLBACK")
            raise
    return version_row(c, c.resolve_version(pid, vid))


class DatasetBody(BaseModel):
    version_id: str | None = None
    path: str
    repeats: int = Field(1, ge=1, le=1_000_000)
    caption_ext: str = "auto"
    is_reg: bool = False
    prior_weight: float = Field(1.0, ge=0)
    class_prompt: str | None = None


def _dataset_config(
    c: ServiceContext, pid: str, version_id: str | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = get_project_config(pid, c, version_id)
    dataset = cfg.setdefault("dataset", {})
    if not isinstance(dataset, dict):
        raise ApiError("project dataset config must be an object", code="config.invalid")
    sources = dataset.setdefault("sources", [])
    if not isinstance(sources, list) or any(not isinstance(item, dict) for item in sources):
        raise ApiError("project dataset.sources must be a list of objects", code="config.invalid")
    return cfg, sources


def _same_source(path: Any, expected: str) -> bool:
    return isinstance(path, str) and Path(path).expanduser().resolve() == Path(expected).resolve()


def _register_dataset(
    c: ServiceContext,
    pid: str,
    body: DatasetBody,
    *,
    did: str | None = None,
    version_id: str | None = None,
    internal: bool = False,
    origin_path: str | None = None,
    caption: dict[str, Any] | None = None,
) -> str:
    """Register the source and add it to the project's training draft under one lock."""
    p = Path(body.path).expanduser().resolve()
    if not p.is_dir():
        raise NotFound(f"directory not found: {p}", code="fs.not_found")
    source = body.model_dump(exclude={"version_id"}) | {"path": str(p)}
    did = did or new_id("d")
    with c.db.lock:
        version = c.resolve_version(pid, version_id or body.version_id)
        if not internal:
            assert_version_writable(c, pid, version["id"], data=True)
        if any(
            _same_source(row["path"], str(p))
            for row in c.db.fetchall("SELECT path FROM datasets WHERE version_id=?", (version["id"],))
        ):
            raise ApiError(
                "this dataset directory is already registered", code="dataset.duplicate", status=409
            )
        config, sources = _dataset_config(c, pid, version["id"])
        previous = json.loads(json.dumps(config))
        matching = [
            item
            for item in [*sources, *config.get("validation", {}).get("sources", [])]
            if _same_source(item.get("path"), str(p))
            or (origin_path is not None and _same_source(item.get("path"), origin_path))
        ]
        if not matching:
            sources.append(source)
        else:
            explicit = body.model_dump(exclude={"version_id"}, exclude_unset=True) | {"path": str(p)}
            for item in matching:
                item.update(explicit)  # Preserve each source's role and all unedited advanced settings.
            source = {key: matching[0].get(key, value) for key, value in source.items()}
        _validate_caption_extension(source["caption_ext"])
        from .source_roles import managed_source_role

        role = managed_source_role(c, pid, version["id"], str(p))
        if role is not None:
            source["is_reg"] = role[0]
            for item in [*sources, *config.get("validation", {}).get("sources", [])]:
                if _same_source(item.get("path"), str(p)):
                    item["is_reg"] = role[0]
        if caption is not None:
            from ypuddin.config.schema import CaptionConfig

            source["caption"] = CaptionConfig.model_validate(caption).model_dump()
            for item in matching:
                item["caption"] = source["caption"]
        c.db.execute("BEGIN IMMEDIATE")
        written = False
        try:
            c.db.insert(
                "datasets",
                {
                    "id": did,
                    "project_id": pid,
                    "version_id": version["id"],
                    "origin_path": origin_path,
                    **{key: value for key, value in source.items() if key != "caption"},
                    "is_reg": int(source["is_reg"]),
                    "created_at": now(),
                    "index_status": "indexing",
                    "stats_json": "{}",
                },
            )
            _write_project_config(c, pid, config, version["id"])
            written = True
            c.db.update("projects", pid, {"updated_at": now()})
            c.db.execute("COMMIT")
        except BaseException:
            c.db.execute("ROLLBACK")
            if written:
                _write_project_config(c, pid, previous, version["id"])
            raise
    c.bus.publish(
        "dataset.changed", {"dataset_id": did, "project_id": pid, "version_id": version["id"], "reason": "added"}
    )
    return did


def _register_upload(
    c: ServiceContext,
    pid: str,
    batch: UploadBatch,
    version_id: str | None = None,
    progress: ImportProgress | None = None,
) -> list[str]:
    with c.versions.mutation(pid, version_id) as version:
        with staged_upload(
            c.version_dir(pid, version["id"]),
            batch,
            dataset_root=c.dataset_dir(pid, version["id"], is_reg=batch.is_reg),
            progress=progress,
        ) as directories:
            if progress:
                progress.set_phase("registering")
            with c.db.lock:
                managed_root = c.dataset_dir(pid, version["id"], is_reg=batch.is_reg).resolve()
                config, sources = _dataset_config(c, pid, version["id"])
                previous = json.loads(json.dumps(config))
                rows = c.db.fetchall("SELECT * FROM datasets WHERE version_id=?", (version["id"],))
                configured = [*sources, *config.get("validation", {}).get("sources", [])]
                selected: dict[str, dict] = {}
                additions = []
                for directory in directories:
                    directory = directory.resolve()
                    # An existing source already recurses into this directory. Reuse
                    # it and keep its caption/repeat/regularization options unchanged.
                    ancestors = [row for row in rows if directory.is_relative_to(Path(row["path"]).resolve())]
                    if len(ancestors) > 1:
                        raise ApiError(
                            f"overlapping dataset sources already cover {directory.name}",
                            code="dataset.overlap",
                            status=409,
                        )
                    if ancestors:
                        if not Path(ancestors[0]["path"]).resolve().is_relative_to(managed_root):
                            raise ApiError(
                                "an existing source spans the training and regularization directories",
                                code="dataset.overlap",
                                status=409,
                            )
                        selected[ancestors[0]["id"]] = ancestors[0]
                        continue
                    if any(Path(row["path"]).resolve().is_relative_to(directory) for row in rows):
                        raise ApiError(
                            f"a child of {directory.name} is already a dataset; import that child folder instead",
                            code="dataset.overlap",
                            status=409,
                        )
                    matching = [
                        source
                        for source in configured
                        if isinstance(source.get("path"), str)
                        and source["path"].strip()
                        and directory.is_relative_to(Path(source["path"]).expanduser().resolve())
                    ]
                    # A draft may already reference the managed root without an index
                    # record. Index that same source instead of adding a duplicate child.
                    source_path = directory
                    if matching:
                        paths = {Path(source["path"]).expanduser().resolve() for source in matching}
                        if len(paths) != 1:
                            raise ApiError(
                                f"overlapping configured sources cover {directory.name}",
                                code="dataset.overlap",
                                status=409,
                            )
                        source_path = paths.pop()
                        if not source_path.is_relative_to(managed_root):
                            raise ApiError(
                                "a configured source spans the training and regularization directories",
                                code="dataset.overlap",
                                status=409,
                            )
                        if any(Path(row["path"]).resolve().is_relative_to(source_path) for row in rows):
                            raise ApiError(
                                f"configured source {source_path.name} overlaps an indexed child dataset",
                                code="dataset.overlap",
                                status=409,
                            )
                        existing = next(
                            (row for row in selected.values() if Path(row["path"]) == source_path), None
                        )
                        if existing:
                            continue
                    elif any(
                        isinstance(source.get("path"), str)
                        and source["path"].strip()
                        and Path(source["path"]).expanduser().resolve().is_relative_to(directory)
                        for source in configured
                    ):
                        raise ApiError(
                            f"a child of {directory.name} is already configured; import that child folder instead",
                            code="dataset.overlap",
                            status=409,
                        )
                    source = DatasetBody(
                        path=str(source_path),
                        repeats=batch.repeats,
                        is_reg=batch.is_reg,
                        prior_weight=batch.prior_weight,
                        class_prompt=batch.class_prompt,
                        caption_ext=batch.caption_ext,
                    ).model_dump(exclude={"version_id"})
                    if matching:
                        source.update({key: matching[0][key] for key in source if key in matching[0]})
                        source["path"] = str(source_path)
                        source["is_reg"] = batch.is_reg
                    else:
                        sources.append(source)
                    _validate_caption_extension(source["caption_ext"])
                    row = {
                        "id": new_id("d"),
                        "project_id": pid,
                        "version_id": version["id"],
                        "origin_path": None,
                        **source,
                        "is_reg": int(source["is_reg"]),
                        "created_at": now(),
                        "index_status": "indexing",
                        "stats_json": "{}",
                    }
                    additions.append(row)
                    selected[row["id"]] = row
                # One transaction covers every concept and the draft. The staging
                # context restores only this upload's new files if registration fails.
                c.db.execute("BEGIN IMMEDIATE")
                written = False
                try:
                    for row in additions:
                        c.db.insert("datasets", row)
                    for did in selected:
                        c.db.update("datasets", did, {"index_status": "indexing"})
                    _write_project_config(c, pid, config, version["id"])
                    written = True
                    c.db.update("projects", pid, {"updated_at": now()})
                    c.db.execute("COMMIT")
                except BaseException:
                    c.db.execute("ROLLBACK")
                    if written:
                        _write_project_config(c, pid, previous, version["id"])
                    raise
    for did in selected:
        c.bus.publish(
            "dataset.changed",
            {"dataset_id": did, "project_id": pid, "version_id": version["id"], "reason": "imported"},
        )
    return list(selected)


def _records_path(c: ServiceContext, did: str) -> Path:
    return c.data_root / "datasets" / f"{did}.json"


def _refresh_dataset(c: ServiceContext, row: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    """Return the dataset as indexed now; files changed elsewhere are re-indexed in the background."""
    from .dataset_refresh import refresher

    refresher(c).request(row, force=force)
    return row


def _index_error(error: Exception) -> str:
    if isinstance(error, (FileNotFoundError, NotADirectoryError)):
        return "数据集目录不存在或无法访问。"
    if isinstance(error, PermissionError):
        return "没有权限读取数据集目录。"
    return str(error)


def _index_dataset(
    c: ServiceContext,
    did: str,
    *,
    listing: Any = None,
    signature: str | None = None,
    refresh: bool = False,
    task: str | None = None,
) -> None:
    """Rebuild a dataset's index records.

    ``refresh`` marks a re-index started by a read: it leaves the dataset editable and gives way to
    an index someone requested meanwhile. ``listing`` and ``signature`` reuse the walk that found the
    change, so the folder is listed once.
    """
    from .dataset_refresh import dataset_signature, refresher

    service = refresher(c)
    with service.index_lock(did):
        row = c.db.fetchone("SELECT * FROM datasets WHERE id=?", (did,))
        if not row or refresh and row["index_status"] == "indexing":
            return
        src = DatasetSourceConfig(
            path=row["path"],
            repeats=row["repeats"],
            caption_ext=row["caption_ext"],
            is_reg=bool(row["is_reg"]),
            prior_weight=row["prior_weight"],
            class_prompt=row["class_prompt"],
        )
        if signature is None:
            signature, listing = dataset_signature(row)
        service.checked(did)

        def progress(done: int, total: int) -> None:
            c.bus.publish("job.cache_progress", {"job_id": did, "kind": "index", "done": done, "total": total})
            if task is not None and c.background_tasks is not None:
                c.background_tasks.update(task, done=done, total=total)

        def superseded(current: dict[str, Any] | None) -> bool:
            return (
                current is None
                or any(current[key] != row[key] for key in ("path", "caption_ext"))
                or refresh and current["index_status"] == "indexing"
            )

        temporary = None
        try:
            db = IndexDB(c.service_cache_dir("index") / "index.sqlite")
            try:
                records = scan_sources(
                    [src],
                    index_db=db,
                    progress=progress,
                    listings={row["path"]: listing} if listing is not None else None,
                )
            finally:
                db.close()
            res: dict[tuple[int, int], int] = {}
            ars: dict[str, int] = {}
            for r in records:
                res[(r.width, r.height)] = res.get((r.width, r.height), 0) + 1
                ar = round(r.width / r.height, 1)
                ars[str(ar)] = ars.get(str(ar), 0) + 1
            stats = {
                "images": len(records),
                "captioned": sum(1 for r in records if r.caption_path),
                "resolutions": [
                    {"w": w, "h": h, "count": n} for (w, h), n in sorted(res.items(), key=lambda x: -x[1])[:50]
                ],
                "ar_hist": [{"ar": k, "count": v} for k, v in sorted(ars.items(), key=lambda x: float(x[0]))],
                "masks": sum(1 for r in records if r.mask_path),
                # The pre-scan signature leaves concurrent filesystem changes detectable on the next check.
                "_source_signature": signature,
            }
            path = _records_path(c, did)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                json.dump([r.to_dict() for r in records], stream)
            with c.db.lock:
                if superseded(c.db.fetchone("SELECT * FROM datasets WHERE id=?", (did,))):
                    return
                temporary.replace(path)
                c.db.update("datasets", did, {"index_status": "ready", "stats_json": json.dumps(stats)})
        except Exception as e:  # noqa: BLE001
            with c.db.lock:
                current = c.db.fetchone("SELECT * FROM datasets WHERE id=?", (did,))
                if superseded(current):
                    return
                stats = json.loads(current["stats_json"] or "{}")
                stats.update(error=_index_error(e), _source_signature=signature)
                c.db.update("datasets", did, {"index_status": "failed", "stats_json": json.dumps(stats)})
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    c.bus.publish("dataset.changed", {
        "dataset_id": did, "project_id": row["project_id"], "version_id": row.get("version_id"),
        "reason": "indexed",
    })


def _dataset_row(c: ServiceContext, r: dict[str, Any], *, include_cache: bool = True) -> dict[str, Any]:
    from .routes_dataset_management import can_rename, included, source_states
    from .source_roles import managed_source_role
    states = source_states(c, r)
    exact = next((state[3] for state in states if state[0] == Path(r["path"]).resolve()), None)
    stats = json.loads(r["stats_json"] or "{}")
    stats.pop("_source_signature", None)
    if r["index_status"] == "ready":
        records = _records(c, r["id"])
        stats["training_images"] = sum(included(record["path"], states) for record in records)
        stats["held_out_images"] = len(records) - stats["training_images"]
    role = managed_source_role(c, r["project_id"], r.get("version_id"), r["path"])
    source = {
        "id": r["id"],
        "project_id": r["project_id"],
        "version_id": r.get("version_id"),
        "path": r["path"],
        "repeats": exact.get("repeats", r["repeats"]) if exact else r["repeats"],
        "can_rename": can_rename(c, r),
        "can_append": bool(role and Path(r["path"]).absolute() == Path(r["path"]).resolve()),
        "caption_ext": r["caption_ext"],
        "is_reg": role[0] if role is not None else bool(r["is_reg"]),
        "prior_weight": r["prior_weight"],
        "class_prompt": r["class_prompt"],
        "created_at": r["created_at"],
    }
    from .dataset_refresh import refresher

    return {
        "source": source,
        "masked_loss": bool(get_project_config(r["project_id"], c, r.get("version_id")).get("dataset", {}).get("masked_loss", False)) if r.get("project_id") else False,
        "stats": stats,
        "index_status": r["index_status"],
        # Files changed outside the studio are being indexed; the dataset stays editable meanwhile.
        "refreshing": refresher(c).refreshing(r["id"]),
        "cache": _cache_stats(c, r) if include_cache else {},
    }


def _cache_stats(c: ServiceContext, r: dict[str, Any]) -> dict[str, Any]:
    """Latent-cache coverage of this dataset under the project's current draft config.

    Keys depend on the training resolutions, bucketing and the family's latent fingerprint, so the
    project draft (or defaults) decides what counts as cached. Text encodings are keyed by the
    transformed caption variants and are only knowable at train time, so they are not reported.
    """
    if not r.get("project_id"):
        return {}
    try:
        from ypuddin.data import BucketManager, LatentCache
        from ypuddin.data.dataset import expand_items
        from ypuddin.data.index import ImageRecord
        from ypuddin.models import get_family

        raw = get_project_config(r["project_id"], c, r.get("version_id"))
        source_path = Path(r["path"]).expanduser().resolve()
        matching = [
            src
            for src in raw.get("dataset", {}).get("sources", [])
            if Path(src["path"]).expanduser().resolve() == source_path
        ]
        raw.setdefault("dataset", {})["sources"] = matching or [{"path": r["path"], "repeats": r["repeats"]}]
        cfg = TrainConfig.model_validate(raw)
        family = get_family(cfg.model.family)
        recs = [
            ImageRecord(**{k: v for k, v in rec.items() if k in ImageRecord.__dataclass_fields__})
            for rec in _records(c, r["id"])
        ]
        from .routes_dataset_management import included, source_states

        # Exclusions belong to this dataset's configured source. A dataset outside the
        # configured sources is measured whole, like the path fallback above.
        states = [state for state in source_states(c, r) if state[0] == source_path]
        if states:
            recs = [record for record in recs if included(record.path, states)]
        if not recs:
            return {}
        ds = cfg.dataset
        from dataclasses import replace

        recs = [replace(rec, source_index=i) for i in range(len(ds.sources)) for rec in recs]
        bm = BucketManager(
            []
            if ds.resolution_mode == "native"
            else sorted(
                {resolution for src in ds.sources for resolution in (src.resolutions or ds.resolutions)}
            ),
            align=family.spec.latent.align,
            step=family.spec.latent.align if ds.resolution_mode == "native" else ds.bucket_step,
            aspect_ratio_limit=ds.aspect_ratio_limit,
            area_tolerance=ds.area_tolerance,
            no_upscale=ds.bucket_no_upscale,
        )
        items = expand_items(recs, ds.sources, ds, bm)
        cache_root = c.cache_dir(r["project_id"], r.get("version_id"))
        import torch

        from ypuddin.models.fingerprints import fingerprint_cache

        devices = gpu_info()
        from ypuddin.models.precision import model_load_precision

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[model_load_precision(cfg.model)]
        if cfg.memory.no_half_vae or not devices or devices[0]["kind"] == "mps":
            dtype = torch.float32
        with fingerprint_cache(cache_root / "fingerprints"):
            latent_identity = family.latent_fingerprint(cfg.model, dtype=dtype)
            if cfg.model.family in {"anima", "krea2"}:
                from ypuddin.models.vae_tiling import tiled_latent_fingerprint

                latent_identity = tiled_latent_fingerprint(
                    latent_identity,
                    vae_tiling=cfg.memory.vae_tiling,
                    cache_encode_tiled=cfg.memory.cache_encode_tiled,
                )
        lc = LatentCache(cache_root / "latents")
        from ypuddin.data.dataset import item_latent_key

        keys = {
            item_latent_key(it, latent_identity, flip)
            for it in items
            for flip in ((False, True) if ds.flip else (False,))
        }
        cached = sum(1 for k in keys if lc.has(k))
        return {"latents": {"cached": cached, "total": len(keys)}, "cache_dir": str(cache_root)}
    except Exception as e:  # noqa: BLE001 - statistics must never break the dataset endpoint
        return {"error": str(e)}


@router.get("/projects/{pid}/datasets", response_model=list[m.DatasetInfo], response_model_exclude_unset=True)
def list_datasets(
    pid: str,
    c: ServiceContext = Depends(ctx),
    version_id: str | None = None,
    include_cache: bool = True,
    refresh: bool = Query(False, description="Check the folders for changes now instead of when next due."),
) -> list[dict[str, Any]]:
    version = c.resolve_version(pid, version_id)
    return [
        _dataset_row(c, _refresh_dataset(c, r, force=refresh), include_cache=include_cache)
        for r in c.db.fetchall(
            "SELECT * FROM datasets WHERE version_id=? ORDER BY created_at", (version["id"],)
        )
    ]


@router.post(
    "/projects/{pid}/datasets",
    status_code=201,
    response_model=m.DatasetInfo,
    response_model_exclude_unset=True,
)
def add_dataset(
    pid: str,
    body: DatasetBody,
    background_tasks: BackgroundTasks,
    c: ServiceContext = Depends(ctx),
    version_id: str | None = None,
    progress_id: str | None = None,
) -> dict[str, Any]:
    with c.import_admission(), c.import_progress.track(pid, progress_id, "validating") as progress:
        version = c.resolve_version(pid, version_id or body.version_id)
        did = c.versions.import_directory(pid, version["id"], body, progress=progress)
    background_tasks.add_task(_index_dataset, c, did)
    return _dataset_row(c, c.db.fetchone("SELECT * FROM datasets WHERE id=?", (did,)))


@router.get("/projects/{pid}/datasets/import-progress/{progress_id}", response_model=m.DatasetImportProgress)
async def dataset_import_progress(pid: str, progress_id: str, request: Request) -> dict[str, Any]:
    # This route intentionally performs no database lookup or thread-pool dependency:
    # callers can poll while a dataset registration holds the database lock.
    return request.app.state.ctx.import_progress.get(pid, progress_id)


@router.post(
    "/projects/{pid}/datasets/upload",
    response_model=m.DatasetUploadInfo,
    response_model_exclude_unset=True,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["files"],
                        "properties": {
                            "files": {"type": "array", "items": {"type": "string", "format": "binary"}},
                            "name": {"type": "string", "maxLength": 100},
                            "repeats": {"type": "integer", "minimum": 1, "maximum": 1_000_000, "default": 1},
                            "is_reg": {"type": "boolean", "default": False},
                            "prior_weight": {"type": "number", "minimum": 0, "default": 1},
                            "class_prompt": {"type": "string"},
                            "caption_ext": {"type": "string", "default": "auto"},
                        },
                    }
                }
            },
        }
    },
)
async def upload_dataset(
    pid: str,
    request: Request,
    background_tasks: BackgroundTasks,
    c: ServiceContext = Depends(ctx),
    version_id: str | None = None,
    progress_id: str | None = None,
) -> dict[str, Any]:
    with c.import_admission(), c.import_progress.track(pid, progress_id, "receiving") as progress:
        version = await run_in_threadpool(assert_version_writable, c, pid, version_id, data=True)
        async with read_upload(request, progress) as batch:
            ids = await run_in_threadpool(_register_upload, c, pid, batch, version["id"], progress)
    datasets = []
    for did in ids:
        background_tasks.add_task(_index_dataset, c, did)
        datasets.append(_dataset_row(c, _get_dataset(c, did)))
    return {**datasets[0], "datasets": datasets}


def _get_dataset(c: ServiceContext, did: str) -> dict[str, Any]:
    r = c.db.fetchone("SELECT * FROM datasets WHERE id=?", (did,))
    if not r:
        raise NotFound(f"dataset {did} not found", code="dataset.not_found")
    return r


def _assert_upload_session_target(c: ServiceContext, session: UploadSession) -> None:
    assert_version_writable(c, session.pid, session.vid, data=True)
    if session.body.target_dataset_id:
        row = _get_dataset(c, session.body.target_dataset_id)
        if row["project_id"] != session.pid or row["version_id"] != session.vid:
            raise NotFound("上传目标不存在。", code="dataset.not_found")


@router.post("/projects/{pid}/datasets/upload-sessions", response_model=m.DatasetUploadSession)
def create_upload_session(pid: str, body: UploadSessionBody, c: ServiceContext = Depends(ctx)) -> dict:
    with c.import_admission():
        vid = body.version_id
        if body.target_dataset_id:
            row = _get_dataset(c, body.target_dataset_id)
            if row["project_id"] != pid or (vid and vid != row["version_id"]):
                raise NotFound("上传目标不存在。", code="dataset.not_found")
            vid = row["version_id"]
            body = body.model_copy(update={"caption_ext": row["caption_ext"]})
        version = assert_version_writable(c, pid, vid, data=True)
        session = c.upload_sessions.create(pid, version["id"], body)
    return {"id": session.id, "chunk_bytes": CHUNK_BYTES}


@router.put(
    "/projects/{pid}/datasets/upload-sessions/{sid}/files/{index}",
    response_model=m.DatasetUploadChunk,
    openapi_extra={"requestBody": {"required": True, "content": {
        "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
    }}},
)
async def put_upload_chunk(
    pid: str, sid: str, index: int, request: Request, offset: int = Query(ge=0),
    c: ServiceContext = Depends(ctx),
) -> dict:
    from .upload_sessions import assert_receivable

    with c.import_admission(), c.upload_sessions.use(pid, sid) as session:
        # Other imports and index runs may hold the version meanwhile; only "complete" needs it.
        await run_in_threadpool(assert_receivable, c, session)
        if not 0 <= index < len(session.manifest):
            raise ApiError("上传文件编号无效。", code="upload.file_index", status=404)
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/octet-stream":
            raise ApiError("上传分片需要使用 application/octet-stream。", code="upload.content_type", status=415)
        length = request.headers.get("content-length")
        if length is not None:
            try:
                declared = int(length)
            except ValueError as exc:
                raise ApiError("上传分片长度无效。", code="upload.invalid") from exc
            if declared < 0 or declared > CHUNK_BYTES:
                raise ApiError("上传分片超过大小限制。", code="upload.chunk_size", status=413)
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > CHUNK_BYTES:
                raise ApiError("上传分片超过大小限制。", code="upload.chunk_size", status=413)
            data.extend(chunk)
        received = await run_in_threadpool(session.put, index, offset, bytes(data))
        return {"received": received}


@router.post(
    "/projects/{pid}/datasets/upload-sessions/{sid}/complete",
    response_model=m.DatasetUploadInfo, response_model_exclude_unset=True,
)
def complete_upload_session(
    pid: str, sid: str, background_tasks: BackgroundTasks, c: ServiceContext = Depends(ctx),
) -> dict:
    with c.import_admission(), c.upload_sessions.use(pid, sid) as session:
        def publish(batch: UploadBatch, progress: ImportProgress) -> dict:
            _assert_upload_session_target(c, session)
            if session.body.target_dataset_id:
                result = _append_upload(c, session.body.target_dataset_id, batch, progress)
                return {**result, "datasets": [result]}
            ids = _register_upload(c, pid, batch, session.vid, progress)
            datasets = [_dataset_row(c, _get_dataset(c, did), include_cache=False) for did in ids]
            for did in ids:
                background_tasks.add_task(_index_dataset, c, did)
            return {**datasets[0], "datasets": datasets}

        return session.finish(publish)


@router.delete("/projects/{pid}/datasets/upload-sessions/{sid}", response_model=m.Ok)
def delete_upload_session(pid: str, sid: str, c: ServiceContext = Depends(ctx)) -> dict:
    with c.import_admission():
        # Only this session's staging files/receipt are removed, even if training has since started.
        c.upload_sessions.delete(pid, sid)
    return {"ok": True}


def _append_upload(c: ServiceContext, did: str, batch: UploadBatch, progress: ImportProgress | None) -> dict:
    from .source_roles import managed_source_role

    original = _get_dataset(c, did)
    with c.versions.mutation(original["project_id"], original["version_id"]) as version:
        row = _get_dataset(c, did)
        root = Path(row["path"])
        role = managed_source_role(c, row["project_id"], version["id"], str(root))
        if not role or root.absolute() != root.resolve() or not root.is_dir():
            raise ApiError("只能向当前版本内的数据集添加图片。", code="dataset.unmanaged", status=409)
        batch.caption_ext = row["caption_ext"]
        protected = tuple(
            Path(item["path"]).resolve()
            for item in c.db.fetchall(
                "SELECT path FROM datasets WHERE version_id=? AND id<>?", (version["id"], did)
            )
            if Path(item["path"]).resolve().is_relative_to(root.resolve())
        )
        with staged_upload(
            c.version_dir(row["project_id"], version["id"]), batch,
            dataset_root=root, append=True, protected_roots=protected, progress=progress,
        ):
            with c.db.lock:
                c.db.update("datasets", did, {"index_status": "indexing"})
                c.db.update("projects", row["project_id"], {"updated_at": now()})
        _index_dataset(c, did)
        return _dataset_row(c, _get_dataset(c, did), include_cache=False)


@router.post(
    "/datasets/{did}/upload", response_model=m.DatasetInfo, response_model_exclude_unset=True,
    openapi_extra={"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
        "type": "object", "required": ["files"], "properties": {
            "files": {"type": "array", "items": {"type": "string", "format": "binary"}},
            "caption_ext": {"type": "string", "default": "auto"},
        },
    }}}}},
)
async def append_dataset_images(
    did: str, request: Request, c: ServiceContext = Depends(ctx), progress_id: str | None = None
) -> dict[str, Any]:
    row = _get_dataset(c, did)
    with c.import_admission(), c.import_progress.track(row["project_id"], progress_id, "receiving") as progress:
        await run_in_threadpool(assert_version_writable, c, row["project_id"], row["version_id"], data=True)
        async with read_upload(request, progress) as batch:
            return await run_in_threadpool(_append_upload, c, did, batch, progress)


@router.get("/datasets/{did}", response_model=m.DatasetInfo, response_model_exclude_unset=True)
def get_dataset(did: str, c: ServiceContext = Depends(ctx), include_cache: bool = True) -> dict[str, Any]:
    return _dataset_row(c, _refresh_dataset(c, _get_dataset(c, did)), include_cache=include_cache)


@router.post("/datasets/{did}/rescan", response_model=m.Ok, response_model_exclude_unset=True)
async def rescan_dataset(did: str, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    with c.db.lock:
        row = _get_dataset(c, did)
        assert_version_writable(c, row["project_id"], row["version_id"])
        if row["index_status"] == "indexing":
            return {"ok": True}
        assert_version_writable(c, row["project_id"], row["version_id"], data=True)
        c.db.update("datasets", did, {"index_status": "indexing"})
    asyncio.get_running_loop().run_in_executor(None, _index_dataset, c, did)
    return {"ok": True}


@router.delete("/datasets/{did}", response_model=m.Ok, response_model_exclude_unset=True)
def delete_dataset(did: str, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    with c.db.lock:
        row = _get_dataset(c, did)
        pid = row["project_id"]
        assert_version_writable(c, pid, row["version_id"], data=True)
        config, sources = _dataset_config(c, pid, row["version_id"]) if pid else (None, [])
        previous = json.loads(json.dumps(config)) if config is not None else None
        if config is not None:
            config["dataset"]["sources"] = [
                item for item in sources if not _same_source(item.get("path"), row["path"])
            ]
            if isinstance(config.get("validation"), dict):
                config["validation"]["sources"] = [
                    item
                    for item in config["validation"].get("sources", [])
                    if not _same_source(item.get("path"), row["path"])
                ]
        written = False
        c.db.execute("BEGIN IMMEDIATE")
        try:
            c.db.delete("datasets", did)
            if config is not None:
                _write_project_config(c, pid, config, row["version_id"])
                written = True
                c.db.update("projects", pid, {"updated_at": now()})
            c.db.execute("COMMIT")
        except BaseException:
            c.db.execute("ROLLBACK")
            if written:
                _write_project_config(c, pid, previous, row["version_id"])
            raise
    _records_path(c, did).unlink(missing_ok=True)
    c.bus.publish(
        "dataset.changed",
        {"dataset_id": did, "project_id": pid, "version_id": row["version_id"], "reason": "deleted"},
    )
    return {"ok": True}


def _records(c: ServiceContext, did: str) -> list[dict[str, Any]]:
    p = _records_path(c, did)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def _image_in_source(row: dict, record: dict) -> bool:
    root, image = Path(row["path"]), Path(record["path"])
    try:
        return (
            not root.is_symlink()
            and image.is_relative_to(root)
            and image.resolve().is_relative_to(root.resolve())
            and image.is_file()
        )
    except (OSError, RuntimeError):
        return False


def _current_caption_path(
    row: dict, record: dict, directory_cache: dict, *, create: bool = False
) -> Path | None:
    from ypuddin.data.index import caption_for, caption_target

    if not _image_in_source(row, record):
        raise ApiError("image is outside its dataset or no longer exists", code="dataset.image_path")
    image, root = Path(record["path"]), Path(row["path"]).resolve()
    selected = (
        caption_target(image, row["caption_ext"], directory_cache=directory_cache)
        if create
        else caption_for(image, row["caption_ext"], directory_cache=directory_cache)
    )
    if selected is None:
        return None
    path = Path(selected)
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ApiError("caption must remain inside its dataset", code="dataset.caption_path")
    return path


def _dataset_image_items(c: ServiceContext, row: dict) -> list[dict]:
    from ypuddin.data.caption_json import StructuredCaption, load_caption_structure, render, unique
    from ypuddin.data.captions import read_training_caption

    from .routes_dataset_management import included, source_states
    states = source_states(c, row)
    root = Path(row["path"])
    items = []
    directory_cache = {}
    seen = set()
    for r in _records(c, row["id"]):
        # Index files are source-local snapshots. Never follow a stale or redirected
        # record into another source/version, and count each image path once.
        if r["path"] in seen or not _image_in_source(row, r):
            continue
        seen.add(r["path"])
        error = None
        caption_path = None
        structure = None
        try:
            # Rediscover sidecars once per directory so additions/removals and auto
            # JSON preference are visible immediately, without a metadata rescan.
            caption_path = _current_caption_path(row, r, directory_cache)
            structure = load_caption_structure(caption_path)
            # Retain the source document for the read-only editor, but classify
            # unsupported JSON exactly as training does instead of calling it empty.
            raw = read_training_caption(caption_path, require_known_format=True)
            cap = raw.text() if isinstance(raw, StructuredCaption) else raw
            tokens = (
                unique((raw.trigger, *raw.fixed, *raw.appearance, *raw.tags, *raw.environment))
                if isinstance(raw, StructuredCaption)
                else unique(raw.split(","))
            )
            editable = render(tokens, "") if isinstance(raw, StructuredCaption) else raw
            description = raw.nl if isinstance(raw, StructuredCaption) else ""
            status = "captioned" if cap.strip() else "missing"
            if not cap:
                cap = row["class_prompt"] or ""
        except (ValueError, OSError, ApiError) as exc:
            cap, editable, description, tokens, status = "", "", "", (), "invalid"
            error = str(exc) if isinstance(exc, ValueError) else "Could not read this caption safely"
        items.append(
            {
                "hash": r["content_hash"],
                "rel_path": str(Path(r["path"]).relative_to(root)),
                "width": r["width"],
                "height": r["height"],
                "caption": cap,
                "caption_tags": editable,
                "caption_description": description,
                "caption_structure": structure,
                "caption_format": caption_path.suffix.lower().lstrip(".") if caption_path else None,
                "caption_error": error,
                "caption_status": status,
                "_tokens": tokens,
                "has_mask": bool(r["mask_path"]),
                "training_enabled": included(r["path"], states),
            }
        )
    return items


@router.get("/datasets/{did}/caption-stats", response_model=m.DatasetCaptionStats)
def caption_stats(did: str, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    items = _dataset_image_items(c, _get_dataset(c, did))
    formats: dict[str, int] = {}
    counts: dict[str, dict] = {}
    result = {"images": len(items), "captioned": 0, "missing": 0, "invalid": 0}
    for item in items:
        result[item["caption_status"]] += 1
        if fmt := item["caption_format"]:
            formats[fmt] = formats.get(fmt, 0) + 1
        for token in item["_tokens"]:
            key = token.casefold()
            entry = counts.setdefault(key, {"tag": token, "count": 0})
            entry["count"] += 1
    return {
        **result,
        "formats": formats,
        "unique_tags": len(counts),
        "tags": sorted(
            counts.values(), key=lambda item: (-item["count"], item["tag"].casefold(), item["tag"])
        ),
    }


@router.get("/datasets/{did}/images", response_model=m.ImagePage, response_model_exclude_unset=True)
def list_images(
    did: str,
    page: int = 1,
    page_size: int = 50,
    q: str = "",
    c: ServiceContext = Depends(ctx),
    tag: str | None = None,
    caption_status: Literal["captioned", "missing", "invalid"] | None = None,
    membership: Literal["all", "training", "unused"] = "all",
    sort: Literal["filename", "folder", "modified"] = "filename",
) -> dict[str, Any]:
    from .dataset_sort import sort_images

    row = _get_dataset(c, did)
    items = []
    query, exact_tag = q.casefold(), tag.strip().casefold() if tag else ""
    for item in _dataset_image_items(c, row):
        if membership != "all" and item["training_enabled"] != (membership == "training"):
            continue
        if query and query not in item["caption"].casefold() and query not in item["rel_path"].casefold():
            continue
        if caption_status and item["caption_status"] != caption_status:
            continue
        if exact_tag and exact_tag not in {token.casefold() for token in item["_tokens"]}:
            continue
        item.pop("_tokens")
        items.append(item)
    return _page(sort_images(items, row["path"], sort), page, page_size)


def _record_by_hash(c: ServiceContext, did: str, h: str, rel_path: str | None = None) -> dict[str, Any]:
    root = Path(_get_dataset(c, did)["path"]) if rel_path is not None else None
    for r in _records(c, did):
        if r["content_hash"] == h and (root is None or Path(r["path"]) == root / rel_path):
            return r
    raise NotFound("image not found", code="image.not_found")


@router.get("/datasets/{did}/images/{h}/thumb")
def thumb(
    did: str, h: str, size: int = Query(256, ge=16, le=2048), c: ServiceContext = Depends(ctx)
) -> Response:
    r = _record_by_hash(c, did, h)
    return Response(
        c.thumbnails.image(r["path"], h, size),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/datasets/{did}/images/{h}/file")
def image_file(did: str, h: str, c: ServiceContext = Depends(ctx)) -> Response:
    r = _record_by_hash(c, did, h)
    return FileResponse(r["path"])


class CaptionFieldEdit(BaseModel):
    path: list[str] = Field(min_length=1)
    value: str | list[str]


class CaptionBody(BaseModel):
    caption: str | None = None
    description: str | None = None
    caption_fields: list[CaptionFieldEdit] | None = None
    caption_revision: str | None = None


@router.get("/datasets/{did}/images/{h}/caption", response_model=m.Caption, response_model_exclude_unset=True)
def get_caption(
    did: str, h: str, c: ServiceContext = Depends(ctx), rel_path: str | None = None
) -> dict[str, Any]:
    from ypuddin.data import read_caption
    from ypuddin.data.caption_json import load_caption_structure

    row = _get_dataset(c, did)
    r = _record_by_hash(c, did, h, rel_path)
    try:
        path = _current_caption_path(row, r, {})
        return {
            "caption": read_caption(path, row["class_prompt"]),
            "caption_structure": load_caption_structure(path),
        }
    except (ValueError, OSError) as exc:
        raise ApiError(str(exc), code="dataset.caption_invalid", status=422) from exc


@router.put("/datasets/{did}/images/{h}/caption", response_model=m.Caption, response_model_exclude_unset=True)
def put_caption(
    did: str, h: str, body: CaptionBody, c: ServiceContext = Depends(ctx), rel_path: str | None = None
) -> dict[str, Any]:
    from ypuddin.data.caption_json import CaptionConflictError, load_caption_structure
    from ypuddin.data.captions import read_caption, write_caption

    from .dataset_refresh import image_entry, note_own_edit

    row = _get_dataset(c, did)
    if body.caption is None and body.description is None and body.caption_fields is None:
        raise ApiError(
            "caption, description or caption_fields is required", code="dataset.caption_invalid", status=422
        )
    with c.versions.mutation(row["project_id"], row["version_id"]):
        row = _get_dataset(c, did)
        r = _record_by_hash(c, did, h, rel_path)
        cap_path = _current_caption_path(row, r, {}, create=True)
        if body.description is not None and cap_path.suffix.lower() != ".json":
            raise ApiError(
                "separate descriptions require a JSON caption", code="dataset.caption_format", status=422
            )
        # Only a newly selected caption file changes what the index records about the image.
        selected = r["caption_path"] != str(cap_path)
        before = image_entry(row, Path(r["path"])) if selected else None
        try:
            write_caption(
                cap_path,
                body.caption,
                description=body.description,
                fields=[field.model_dump() for field in body.caption_fields]
                if body.caption_fields is not None
                else None,
                revision=body.caption_revision,
            )
        except CaptionConflictError as exc:
            raise ApiError(str(exc), code="dataset.caption_conflict", status=409) from exc
        except (ValueError, OSError) as exc:
            raise ApiError(str(exc), code="dataset.caption_invalid", status=422) from exc
        if selected:
            recs = _records(c, did)
            for rec in recs:
                if rec["path"] == r["path"]:
                    rec["caption_path"] = str(cap_path)
            _records_path(c, did).write_text(json.dumps(recs), encoding="utf-8")
            note_own_edit(c, did, [before], [image_entry(row, Path(r["path"]))])
        c.bus.publish("dataset.changed", {
            "dataset_id": did, "project_id": row["project_id"], "version_id": row["version_id"], "reason": "caption",
        })
        return {"caption": read_caption(cap_path), "caption_structure": load_caption_structure(cap_path)}


class TagBatch(BaseModel):
    hashes: list[str]
    rel_paths: list[str] | None = None
    add: list[str] = Field(default_factory=list)
    remove: list[str] = Field(default_factory=list)


@router.post("/datasets/{did}/tags/batch", response_model=m.TagBatchResult, response_model_exclude_unset=True)
def tags_batch(did: str, body: TagBatch, c: ServiceContext = Depends(ctx)) -> dict[str, int]:
    import os

    from ypuddin.data.captions import caption_content, read_editable_caption
    from ypuddin.data.index import caption_target

    from .dataset_refresh import image_entry, note_own_edit

    row = _get_dataset(c, did)
    with c.versions.mutation(row["project_id"], row["version_id"]):
        row = _get_dataset(c, did)
        changed = 0
        created = 0
        # Images whose record gains a caption file; the stored folder signature follows them.
        selected: list[Path] = []
        wanted = set(body.hashes)
        recs = _records(c, did)
        updates: dict[Path, dict[str, Any]] = {}
        caption_directories = {}
        staged_files: set[Path] = set()
        applied: list[dict[str, Any]] = []

        def stage(path: Path, content: bytes) -> Path:
            # Same-directory staging keeps both publication and rollback atomic per file,
            # including legacy data that lives on another filesystem.
            with tempfile.NamedTemporaryFile(
                prefix=f".{path.name}.batch-", suffix=".tmp", dir=path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                staged_files.add(temporary)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            return temporary

        try:
            # Parse and serialize every selected caption before changing even the first file.
            for r in recs:
                if r["content_hash"] not in wanted or (body.rel_paths is not None and Path(r["path"]).relative_to(row["path"]).as_posix() not in body.rel_paths):
                    continue
                cap_path = (
                    Path(r["caption_path"])
                    if r["caption_path"]
                    else caption_target(
                        Path(r["path"]), row["caption_ext"], directory_cache=caption_directories
                    )
                )
                if cap_path not in updates:
                    original = cap_path.read_bytes() if cap_path.exists() else None
                    tags = [
                        t.strip()
                        for t in read_editable_caption(cap_path, row["class_prompt"]).split(",")
                        if t.strip()
                    ]
                    tags = [t for t in tags if t not in body.remove]
                    for tag in body.add:
                        if tag not in tags:
                            tags.append(tag)
                    updates[cap_path] = {
                        "path": cap_path,
                        "before": original,
                        "after": caption_content(cap_path, ", ".join(tags)).encode("utf-8"),
                    }
                if not r["caption_path"]:
                    r["caption_path"] = str(cap_path)
                    created += 1
                    selected.append(Path(r["path"]))
                changed += 1
            before_folders: dict = {}
            before = [image_entry(row, image, before_folders) for image in selected]
            if created:
                index_path = _records_path(c, did)
                updates[index_path] = {
                    "path": index_path,
                    "before": index_path.read_bytes() if index_path.exists() else None,
                    "after": json.dumps(recs).encode("utf-8"),
                }
            # Prepare rollback bytes before publication; an I/O error here leaves data intact.
            for update in updates.values():
                update["staged"] = stage(update["path"], update["after"])
                if update["before"] is not None:
                    update["backup"] = stage(update["path"], update["before"])
            for update in updates.values():
                path = update["path"]
                current = path.read_bytes() if path.exists() else None
                if current != update["before"]:
                    raise ApiError(
                        "A caption changed outside the editor; reload and retry",
                        code="dataset.caption_conflict",
                        status=409,
                    )
                os.replace(update["staged"], path)
                applied.append(update)
        except (OSError, ValueError, ApiError) as exc:
            recovery = {}
            for update in reversed(applied):
                path = update["path"]
                try:
                    if path.read_bytes() != update["after"]:
                        raise OSError("file changed after publication")
                    if update["before"] is None:
                        path.unlink()
                    else:
                        os.replace(update["backup"], path)
                except OSError:
                    # Do not erase recovery copies or overwrite a concurrent external edit.
                    backup = update.get("backup")
                    if backup:
                        staged_files.discard(backup)
                    recovery[str(path)] = (
                        str(backup) if backup else "remove the newly created caption after reviewing it"
                    )
            if recovery:
                raise ApiError(
                    "Caption update failed and some files need recovery from the retained backups",
                    code="dataset.caption_recovery",
                    status=500,
                    details={"recovery": recovery},
                ) from exc
            if isinstance(exc, ApiError):
                raise
            raise ApiError(
                f"Could not update captions: {exc}",
                code="dataset.caption_invalid" if isinstance(exc, ValueError) else "dataset.caption_io",
                status=422,
            ) from exc
        finally:
            for temporary in staged_files:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass  # Cleanup must not hide the original error; these are isolated temp files.
        if selected:
            after_folders: dict = {}
            note_own_edit(c, did, before, [image_entry(row, image, after_folders) for image in selected])
        if changed:
            c.bus.publish("dataset.changed", {
                "dataset_id": did, "project_id": row["project_id"], "version_id": row["version_id"], "reason": "tags",
            })
        return {"changed": changed}


# --------------------------------------------------------------------------- jobs
class JobBody(GpuSelection):
    type: str = "train"
    name: str
    project_id: str | None = None
    version_id: str | None = None
    config: dict[str, Any] | None = None
    priority: int = 0
    scheduled_at: float | None = None
    dora_precision_confirmed: bool = False


class JobPatch(GpuSelection):
    priority: int | None = None
    name: str | None = None
    # True moves a finished job to the archive and keeps its files; False brings it back.
    archived: bool | None = None


FINISHED = ("completed", "failed", "cancelled")


def _job_training_mode(job: dict[str, Any]) -> Literal["adapter", "full"] | None:
    if job.get("type") != "train":
        return None
    try:
        config = json.loads(job.get("config_json") or "null")
    except (TypeError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    training = config.get("training", {})
    if not isinstance(training, dict):
        return None
    mode = training.get("mode", "adapter")
    return mode if mode in ("adapter", "full") else None


_JOB_TRAINING_MODE_SQL = """CASE
    WHEN j.type='train' AND json_valid(j.config_json) THEN CASE
        WHEN json_type(j.config_json)='object' THEN CASE
            WHEN json_type(j.config_json, '$.training') IS NULL THEN 'adapter'
            WHEN json_type(j.config_json, '$.training')='object' THEN CASE
                WHEN json_type(j.config_json, '$.training.mode') IS NULL THEN 'adapter'
                WHEN json_extract(j.config_json, '$.training.mode') IN ('adapter', 'full')
                    THEN json_extract(j.config_json, '$.training.mode')
            END
        END
    END
END"""


def _job_row(r: dict[str, Any]) -> dict[str, Any]:
    out = dict(r)
    out["training_mode"] = _job_training_mode(r)
    out["progress"] = json.loads(r.get("progress_json") or "{}")
    out["latest"] = json.loads(r.get("latest_json") or "{}")
    out["gpu_devices"] = json.loads(out.pop("gpu_devices_json", None) or "[]")
    out.pop("progress_json", None)
    out.pop("latest_json", None)
    out.pop("config_json", None)
    return out


@router.get("/jobs", response_model=m.JobPage, response_model_exclude_unset=True)
def list_jobs(
    status: str | None = None,
    project_id: str | None = None,
    version_id: str | None = None,
    page: int = 1,
    page_size: int = 50,
    c: ServiceContext = Depends(ctx),
    group: Literal["active", "waiting", "history", "archive"] | None = None,
    type: Literal["train", "cache", "xyz"] | None = None,
    training_mode: Literal["adapter", "full"] | None = None,
    q: str | None = None,
) -> dict[str, Any]:
    sql = " FROM jobs j LEFT JOIN projects p ON p.id=j.project_id LEFT JOIN project_versions v ON v.id=j.version_id"
    params = []
    conds = []
    if status:
        conds.append("j.status IN ({})".format(",".join("?" for _ in status.split(","))))
        params += status.split(",")
    # Archived jobs are listed only in the archive, as if deleted everywhere else.
    conds.append("j.archived_at IS NOT NULL" if group == "archive" else "j.archived_at IS NULL")
    if group and group != "archive":
        groups = {
            "active": ("running", "pausing", "cancelling", "paused"),
            "waiting": ("queued", "scheduled"),
            "history": FINISHED,
        }
        conds.append("j.status IN ({})".format(",".join("?" for _ in groups[group])))
        params += groups[group]
    if group in {"history", "archive"}:
        conds.append("j.type!='xyz'")
    if type:
        conds.append("j.type=?")
        params.append(type)
    if training_mode:
        conds.append(f"({_JOB_TRAINING_MODE_SQL})=?")
        params.append(training_mode)
    if project_id:
        conds.append("j.project_id=?")
        params.append(project_id)
    if version_id:
        conds.append("j.version_id=?")
        params.append(version_id)
    if q and q.strip():
        # instr treats user input literally, including SQL LIKE wildcard characters.
        conds.append(
            "instr(lower(j.name || ' ' || j.id || ' ' || coalesce(p.name,'') || ' ' || coalesce(j.project_id,'') || ' ' || coalesce(v.name,'')), lower(?)) > 0"
        )
        params.append(q.strip())
    if conds:
        sql += " WHERE " + " AND ".join(conds)
    page, page_size = max(1, page), max(1, min(200, page_size))
    total = c.db.fetchone("SELECT count(*) AS n" + sql, tuple(params))["n"]
    order = (
        "j.archived_at DESC, j.id DESC"
        if group == "archive"
        else "j.created_at DESC, j.id DESC"
        if group == "history"
        else "CASE j.status WHEN 'running' THEN 0 WHEN 'pausing' THEN 0 WHEN 'cancelling' THEN 0 WHEN 'queued' THEN 1 WHEN 'scheduled' THEN 2 ELSE 3 END, j.priority DESC, CASE WHEN j.status IN ('queued','scheduled') THEN j.created_at END ASC, j.created_at DESC, j.id ASC"
    )
    rows = c.db.fetchall(
        "SELECT j.*, p.name AS project_name, v.name AS version_name, v.number AS version_number"
        + sql
        + " ORDER BY "
        + order
        + " LIMIT ? OFFSET ?",
        (*params, page_size, (page - 1) * page_size),
    )
    return {"items": [_job_row(r) for r in rows], "total": total, "page": page, "page_size": page_size}


@router.post("/jobs", status_code=201, response_model=m.Job, response_model_exclude_unset=True)
def create_job(body: JobBody, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    if body.type not in ("train", "cache"):
        raise ApiError(f"unsupported job type {body.type}", code="job.bad_type")
    version = assert_version_writable(c, body.project_id, body.version_id) if body.project_id else None
    if body.version_id and not body.project_id:
        raise ApiError("version_id requires project_id", code="version.project_required")
    vid = version["id"] if version else None
    config = body.config
    if config is None and body.project_id:
        config = get_project_config(body.project_id, c, vid)
    if config is None:
        raise ApiError("config is required", code="job.no_config")
    from ypuddin.train.native_resolution import clear_native_vram_resolution, resolve_native_vram_config

    checkpoint = config.get("checkpoint")
    if not isinstance(checkpoint, dict) or not checkpoint.get("resume"):
        config = clear_native_vram_resolution(config)
    if body.project_id:
        from .output_binding import bind_output_name
        from .source_roles import normalize_source_roles

        config = normalize_source_roles(c, body.project_id, config, vid)
        config = bind_output_name(c, body.project_id, config, vid)
    for section, key in (
        ("checkpoint", "state_dir"),
        ("checkpoint", "output_dir"),
        ("sampling", "output_dir"),
        ("logging", "output_dir"),
        ("logging", "events_path"),
        ("dataset", "cache_dir"),
    ):
        group = config.get(section, {})
        if not isinstance(group, dict) or group.get(key) is not None and not isinstance(group[key], str):
            raise ApiError(
                "保存路径必须是字符串。",
                code="config.invalid",
                status=422,
                details={"errors": [{"loc": f"{section}.{key}", "msg": "path must be a string"}]},
            )
    jid = new_id("j")
    from .job_layout import makes_products

    # Products go to output/<job>; the job's own records, logs and resume points to jobs/<job>.
    # A job outside any project keeps everything in one folder.
    products = c.job_output_dir(body.project_id, vid, jid, config.get("checkpoint", {}).get("output_dir"))
    run_dir = c.job_records_dir(body.project_id, vid, jid) if body.project_id else products
    output_dir = products if makes_products(body.type) else run_dir
    samples_dir = c.job_storage_dir(
        body.project_id, vid, jid, "samples_dir", run_dir, config.get("sampling", {}).get("output_dir")
    )
    state_dir = c.job_storage_dir(
        body.project_id, vid, jid, "state_dir", run_dir, config.get("checkpoint", {}).get("state_dir")
    )
    requested_events = config.get("logging", {}).get("events_path")
    logs_dir = c.job_storage_dir(
        body.project_id,
        vid,
        jid,
        "logs_dir",
        run_dir,
        config.get("logging", {}).get("output_dir")
        or (str(Path(requested_events).parent) if requested_events else None),
    )
    events_name = Path(requested_events).name if requested_events else "events.jsonl"
    if events_name.casefold() == "run.log":
        raise ApiError(
            "事件日志文件不能命名为 run.log，该文件用于控制台日志。", code="config.invalid", status=422
        )

    config = deep_merge(
        config,
        {
            "checkpoint": {"output_dir": str(output_dir), "state_dir": str(state_dir)},
            "logging": {"events_path": str(logs_dir / events_name), "output_dir": str(logs_dir), "level": "debug"},
            "sampling": {"output_dir": str(samples_dir)},
        },
    )
    training_loop = config.get("loop", {})
    if body.type == "cache":
        # Cache preparation has its own single-device worker, even when the
        # project's following training run is configured for several GPUs.
        config = deep_merge(config, {"loop": {"gpu_count": 1, "distributed_strategy": "ddp"}})
    config = deep_merge(
        config,
        {
            "dataset": {
                "cache_dir": str(
                    c.training_cache_dir(body.project_id, vid, (config.get("dataset") or {}).get("cache_dir"))
                )
            }
        },
    )
    from pydantic import ValidationError

    from ypuddin.config.issues import validation_issues

    try:
        cfg = absolute_paths(TrainConfig.model_validate(config))
    except ValidationError as e:
        raise ApiError(
            "invalid config",
            code="config.invalid",
            details={"errors": validation_issues(e)},
        ) from e
    from ypuddin.train.plan import plan

    planning_cfg = cfg
    if body.type == "cache" and cfg.dataset.resolution_mode == "native" and cfg.dataset.native_max_pixels_mode == "auto_vram":
        # Cache the exact canvases the later training run uses, including its multi-GPU budget.
        try:
            planning_cfg = TrainConfig.model_validate({**cfg.to_dict(), "loop": training_loop})
        except ValidationError as e:
            raise ApiError("invalid config", code="config.invalid", details={"errors": validation_issues(e)}) from e
    devices = gpu_info()
    from .supervisor import training_device_error

    if error := training_device_error(planning_cfg.loop.gpu_count, devices, strategy=planning_cfg.loop.distributed_strategy):
        raise ApiError(
            error,
            code="config.invalid",
            details={"errors": [{"loc": "loop.gpu_count", "msg": error}]},
        )
    if error := selection_error(body.gpu_devices, planning_cfg.loop.gpu_count, devices):
        raise ApiError(error, code="job.gpu_selection", status=422)
    selected = planning_devices(
        devices, planning_cfg.loop.gpu_count, body.gpu_devices,
        prefer_available=(
            planning_cfg.dataset.resolution_mode == "native"
            and planning_cfg.dataset.native_max_pixels_mode == "auto_vram"
            and planning_cfg.dataset.native_max_pixels_resolved is None
            and not planning_cfg.checkpoint.resume
        ),
    )
    memory_target = planning_memory(selected)
    # The shared image index lets an unchanged dataset skip re-reading and hashing every image,
    # as the parameter check's plan already does.
    preflight = plan(
        planning_cfg,
        check_compile=body.type == "train",
        index_db_path=c.service_cache_dir("index") / "index.sqlite",
        **memory_target,
    )
    memory = preflight.get("memory") or {}
    estimated_peak_mb = memory.get("peak_mb_estimate")
    estimated_host_mb = memory.get("host_memory_mb_estimate") if body.type == "train" else None
    if body.type == "cache":
        # This worker only caches text/latent features; it never materializes
        # the trainable backbone, gradients or optimizer. Missing/partial phase
        # estimates stay unknown instead of inheriting the training peak.
        phases = memory.get("cache_phase_peak_mb_estimates")
        values = list(phases.values()) if isinstance(phases, dict) else []
        estimated_peak_mb = (
            max(values)
            if values
            and all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in values)
            else None
        )
    from .memory_fit import capacity_shortfall, host_memory_error, shortfall_error

    if c.db.get_kv("queue.settings", {}).get("memory_admission", True) and (
        shortfall := capacity_shortfall(estimated_peak_mb, selected or devices, cfg.loop.gpu_count)
    ):
        preflight["errors"].append(shortfall_error(shortfall))
        preflight["ok"] = False
    if c.db.get_kv("queue.settings", {}).get("memory_admission", True) and (
        issue := host_memory_error(estimated_host_mb, psutil.virtual_memory().available / 2**20)
    ):
        preflight["errors"].append(issue)
        preflight["ok"] = False
    if not preflight["ok"]:
        raise ApiError(
            "training preflight failed", code="config.invalid", details={"errors": preflight["errors"]}
        )
    from ypuddin.data.dataset import DataConfigError

    try:
        cfg = resolve_native_vram_config(
            cfg, device=memory_target["device"], plan_result=preflight,
        )
    except DataConfigError as error:
        raise ApiError(
            str(error), code="config.invalid",
            details={"errors": [{"loc": error.loc, "msg": str(error)}]},
        ) from error
    dora = preflight.get("dora")
    if body.type == "train" and dora and dora["confirmation_required"] and not body.dora_precision_confirmed:
        raise ApiError(
            dora["confirmation_message"],
            code="dora.confirmation_required",
            status=409,
            details={"dora": dora},
        )
    if cfg.checkpoint.resume:
        state = Path(cfg.checkpoint.resume).expanduser()
        if not (state / "state.json").is_file():
            raise ApiError(
                "resume must point to a complete state directory",
                code="config.invalid",
                details={"errors": [{"loc": "checkpoint.resume", "msg": "complete checkpoint not found"}]},
            )
    with c.db.lock:
        if version:
            assert_version_writable(c, body.project_id, vid)
        status = "scheduled" if body.scheduled_at and body.scheduled_at > now() else "queued"
        c.db.insert(
            "jobs",
            {
                "id": jid,
                "type": body.type,
                "name": body.name,
                "project_id": body.project_id,
                "version_id": vid,
                "status": status,
                "priority": body.priority,
                "gpu_devices_json": json.dumps(body.gpu_devices[:1] if body.type == "cache" else body.gpu_devices),
                "scheduled_at": body.scheduled_at,
                "created_at": now(),
                "run_dir": str(run_dir),
                "samples_dir": str(samples_dir),
                "config_json": json.dumps(cfg.to_dict()),
                "progress_json": json.dumps({"estimated_peak_mb": estimated_peak_mb, "estimated_host_mb": estimated_host_mb}),
                "latest_json": "{}",
            },
        )
    c.bus.publish("queue.changed", {})
    row = c.db.fetchone("SELECT * FROM jobs WHERE id=?", (jid,))
    c.bus.publish("job.state", {"job_id": jid, "status": status})
    return _job_row(row)


def _get_job(c: ServiceContext, jid: str) -> dict[str, Any]:
    r = c.db.fetchone("SELECT * FROM jobs WHERE id=?", (jid,))
    if not r:
        raise NotFound(f"job {jid} not found", code="job.not_found")
    return r


@router.get("/jobs/{jid}", response_model=m.Job, response_model_exclude_unset=True)
def get_job(jid: str, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    row = c.db.fetchone(
        "SELECT j.*, p.name AS project_name, v.name AS version_name, v.number AS version_number"
        " FROM jobs j LEFT JOIN projects p ON p.id=j.project_id"
        " LEFT JOIN project_versions v ON v.id=j.version_id"
        " WHERE j.id=?",
        (jid,),
    )
    if not row:
        raise NotFound(f"job {jid} not found", code="job.not_found")
    return _job_row(row)


@router.patch("/jobs/{jid}", response_model=m.Job, response_model_exclude_unset=True)
def patch_job(jid: str, body: JobPatch, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    from .job_paths import deleting_jobs

    with c.db.lock:
        job = _get_job(c, jid)
        if jid in deleting_jobs:
            raise ApiError("这个任务正在删除。", code="job.deleting", status=409)
        patch = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None}
        if "gpu_devices" in patch:
            if job["status"] not in {
                "queued",
                "scheduled",
                "paused",
                "failed",
                "cancelled",
            } or c.supervisor.is_running(jid):
                raise ApiError("请先暂停或取消任务，再更换显卡。", code="job.running", status=409)
            if error := selection_error(body.gpu_devices, c.supervisor._gpu_count(job), gpu_info()):
                raise ApiError(error, code="job.gpu_selection", status=422)
            patch["gpu_devices_json"] = json.dumps(patch.pop("gpu_devices"))
        if "archived" in patch:
            archived = patch.pop("archived")
            if archived and (job["status"] not in FINISHED or c.supervisor.is_running(jid)):
                raise ApiError(
                    "只能归档已结束的任务，请先取消或等待任务结束。", code="job.not_finished", status=409
                )
            patch["archived_at"] = (job.get("archived_at") or now()) if archived else None
        c.db.update("jobs", jid, patch)
    c.bus.publish("queue.changed", {})
    return _job_row(_get_job(c, jid))


def _jobs_using(
    c: ServiceContext,
    owner_id: str,
    folders: list[Path],
    *,
    owner_type: Literal["job", "project"] = "job",
) -> list[dict[str, Any]]:
    """Unfinished jobs whose saved paths point into these folders, such as a resume from them."""
    roots = [str(folder) for folder in folders]

    def uses(value: Any) -> bool:
        if isinstance(value, str):
            return any(value == root or value.startswith(root + os.sep) for root in roots)
        if isinstance(value, dict):
            return any(uses(item) for item in value.values())
        return isinstance(value, list) and any(uses(item) for item in value)

    column = "project_id" if owner_type == "project" else "id"
    rows = c.db.fetchall(
        "SELECT id, name, config_json, resume_from FROM jobs"
        f" WHERE ({column} IS NULL OR {column}!=?) AND status IN {ACTIVE_JOBS[:-1]},'paused')",
        (owner_id,),
    )
    return [row for row in rows if uses(row["resume_from"]) or uses(json.loads(row["config_json"] or "{}"))]


def _folder_size(folder: Path) -> tuple[int, int]:
    total = files = 0
    pending = [folder]
    while pending:
        try:
            entries = list(os.scandir(pending.pop()))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
                    files += 1
            except OSError:
                continue
    return total, files


@router.get("/jobs/{jid}/storage", response_model=m.JobStorage, response_model_exclude_unset=True)
def job_storage(jid: str, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    """The folders deleting this job removes, with their sizes, so the archive can say so first."""
    from .artifact_inventory import artifact_count
    from .job_paths import output_directory, state_directory

    r = _get_job(c, jid)
    roles = [
        ("products", output_directory(r)),
        ("records", Path(r["run_dir"])),
        ("resume", state_directory(r)),
        ("logs", log_file(r).parent),
        ("samples", Path(r["samples_dir"]) if r.get("samples_dir") else None),
    ]
    folders = []
    for folder in owned_job_directories(r):
        kinds = [role for role, path in roles if path is not None and path == folder]
        # Resume points and logs saved inside the records folder go with it.
        kinds += [
            role for role, path in roles if path is not None and role not in kinds and folder in path.parents
        ]
        size, count = _folder_size(folder) if folder.is_dir() else (0, 0)
        folders.append(
            {"path": str(folder), "kinds": kinds, "exists": folder.is_dir(), "bytes": size, "files": count}
        )
    artifacts = artifact_count(c.db, job_id=jid)
    return {"folders": folders, "total_bytes": sum(f["bytes"] for f in folders), "artifacts": artifacts}


@router.delete("/jobs/{jid}", response_model=m.Ok, response_model_exclude_unset=True)
def delete_job(jid: str, delete_files: bool = False, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    from .job_paths import deleting_jobs, removal_problem
    from .project_deletion import deleting
    from .xyz import dependent_tests, preserve_source

    folders: list[Path] = []
    if delete_files:
        # The disk is checked before taking the lock; the folders a job records never change.
        folders = owned_job_directories(_get_job(c, jid))
        problem = removal_problem(c, jid, folders)
        if problem == "outside":
            raise ApiError("任务目录不在允许访问的范围内。", code="job.path", status=403)
        if problem == "shared":
            raise ApiError("目录中包含其他任务的文件，无法删除。", code="job.files_in_use", status=409)
    with c.db.lock:
        r = _get_job(c, jid)
        if jid in deleting_jobs:
            raise ApiError("这个任务正在删除。", code="job.deleting", status=409)
        if r["status"] in ("running", "pausing", "cancelling") or c.supervisor.is_running(jid):
            raise ApiError("请先取消任务，再删除。", code="job.running", status=409)
        if not r.get("archived_at"):
            raise ApiError("请先归档任务，再永久删除。", code="job.archive_required", status=409)
        if r.get("project_id") and deleting(c, r["project_id"]):
            raise ApiError("这个任务所在的项目正在删除。", code="project.deleting", status=409)
        tests = dependent_tests(c, r)
        if not delete_files:
            # Written only once nothing can refuse the deletion.
            preserve_source(c, r, tests)
            c.db.delete("jobs", jid)
        else:
            if users := _jobs_using(c, jid, folders):
                raise ApiError(
                    f"任务“{users[0]['name']}”还要用到这个任务的文件，请等它结束或取消后再删除。",
                    code="job.files_in_use",
                    status=409,
                    details={"jobs": [row["id"] for row in users]},
                )
            # Its products go with its folders; a comparison started from now on no longer finds them.
            c.db.execute("DELETE FROM artifacts WHERE job_id=?", (jid,))
            deleting_jobs.add(jid)
    if delete_files:
        # Removed outside the lock, so other pages and live updates keep working meanwhile.
        try:
            for directory in folders:
                if directory.is_dir():
                    shutil.rmtree(directory)
        except OSError as exc:
            with c.db.lock:
                deleting_jobs.discard(jid)
            raise ApiError(
                "无法删除这个任务的全部文件，已保留归档的任务，可以重试。",
                code="job.delete_files_failed",
                status=500,
            ) from exc
        with c.db.lock:
            deleting_jobs.discard(jid)
            current = c.db.fetchone("SELECT * FROM jobs WHERE id=?", (jid,))
            if current is not None:
                if current["status"] not in FINISHED or c.supervisor.is_running(jid):
                    raise ApiError(
                        "删除文件时任务又开始运行了，请取消任务后再删除它的记录。", code="job.running", status=409
                    )
                # Written only once nothing can refuse the deletion.
                preserve_source(c, current, tests)
                c.db.delete("jobs", jid)
    c.bus.publish("queue.changed", {})
    return {"ok": True}


@router.post("/jobs/{jid}/{command}", response_model=m.Job, response_model_exclude_unset=True)
def job_command(jid: str, command: str, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    if command not in ("pause", "resume", "cancel", "save", "retry", "force"):
        raise NotFound("unknown command", code="job.bad_command")
    try:
        return _job_row(c.supervisor.request(jid, command))
    except KeyError as e:
        raise NotFound(f"job {jid} not found", code="job.not_found") from e
    except ValueError as e:
        raise ApiError(str(e), code="job.bad_state", status=409) from e


@router.get("/jobs/{jid}/config")
def job_config(jid: str, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    return json.loads(_get_job(c, jid)["config_json"] or "{}")


def _events_file(c: ServiceContext, jid: str) -> list[dict[str, Any]]:
    r = _get_job(c, jid)
    return read_events(event_file(r))


@router.get("/jobs/{jid}/metrics", response_model=m.JobMetrics, response_model_exclude_unset=True)
def job_metrics(jid: str, since_step: int = 0, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    steps, loss, loss_ema, grad, vram, its = [], [], [], [], [], []
    power, temperature, load = [], [], []
    lr: dict[str, list[float]] = {}
    validation = []
    vram_metric = None
    events = _events_file(c, jid)
    for ev in events:
        if ev.get("type") == "step" and ev["step"] > since_step:
            vram_metric = ev.get("vram_metric") or vram_metric
            steps.append(ev["step"])
            loss.append(ev.get("loss"))
            loss_ema.append(ev.get("loss_ema"))
            grad.append(ev.get("grad_norm"))
            vram.append(ev.get("vram_mb"))
            its.append(ev.get("it_s"))
            power.append(ev.get("gpu_power_w"))
            temperature.append(ev.get("gpu_temp_c"))
            load.append(ev.get("gpu_util_pct"))
            for g, v in (ev.get("lr") or {}).items():
                lr.setdefault(g, []).append(v)
        elif ev.get("type") == "validation":
            validation.append({"step": ev["step"], "per_t": ev["per_t"], "mean": ev["mean"]})
    return {
        "steps": steps,
        "loss": loss,
        "loss_ema": loss_ema,
        "lr": lr,
        "grad_norm": grad,
        "vram_mb": vram,
        "vram_metric": vram_metric,
        "it_s": its,
        "validation": validation,
        # Older runs recorded no driver readings.
        "gpu_power_w": power if any(value is not None for value in power) else [],
        "gpu_temp_c": temperature if any(value is not None for value in temperature) else [],
        "gpu_util_pct": load if any(value is not None for value in load) else [],
        "gpu_devices": device_metric_series(events, since_step),
    }


# Recorded with each preview since sampling details are shown beside it.
SAMPLE_DETAILS = ("epoch", "negative", "sampler", "scheduler", "steps", "cfg", "shift", "guidance")


@router.get("/jobs/{jid}/samples", response_model=list[m.JobSample], response_model_exclude_unset=True)
def job_samples(jid: str, c: ServiceContext = Depends(ctx)) -> list[dict[str, Any]]:
    out = []
    for ev, loss in samples_with_loss(_events_file(c, jid)):
        name = Path(ev["path"]).name
        out.append(
            {
                "step": ev["step"],
                "prompt_index": ev["prompt_index"],
                "prompt": ev["prompt"],
                "seed": ev["seed"],
                "url": f"/api/jobs/{jid}/files?path={name}&kind=sample",
                "width": ev["width"],
                "height": ev["height"],
                "created_at": ev["ts"],
                "loss": loss,
                **{key: ev[key] for key in SAMPLE_DETAILS if ev.get(key) is not None},
            }
        )
    return out


@router.get(
    "/jobs/{jid}/checkpoints", response_model=list[m.JobCheckpoint], response_model_exclude_unset=True
)
def job_checkpoints(jid: str, c: ServiceContext = Depends(ctx)) -> list[dict[str, Any]]:
    from .artifact_types import export_type

    out = []
    job = _get_job(c, jid)
    arts = {
        a["path"]: a["id"] for a in c.db.fetchall("SELECT id, path FROM artifacts WHERE job_id=?", (jid,))
    }
    events = list(_events_file(c, jid))
    # Each artifact shows the first preview of its step and, for older runs, that step's logged loss.
    previews: dict[int, str] = {}
    for ev, _loss in samples_with_loss(events):
        previews.setdefault(ev["step"], f"/api/jobs/{jid}/files?path={Path(ev['path']).name}&kind=sample")
    step_losses = {
        ev["step"]: ev.get("loss")
        for ev in events
        if ev.get("type") == "step" and isinstance(ev.get("step"), int)
    }
    for ev in events:
        if ev.get("type") == "checkpoint.saved":
            p = Path(ev["path"])
            if not p.exists():
                continue  # Retention has removed this checkpoint; do not offer a dead download/resume.
            size = (
                p.stat().st_size
                if p.is_file()
                else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                if p.is_dir()
                else 0
            )
            out.append(
                {
                    "step": ev["step"],
                    "kind": ev["kind"],
                    "path": str(p),
                    "size": size,
                    "created_at": ev["ts"],
                    "artifact_id": arts.get(str(p)),
                    "export_type": export_type(ev["kind"], p, config_json=job.get("config_json")),
                    "ema": bool(ev.get("ema")),
                    "epoch": ev.get("epoch"),
                    "loss": ev["loss"] if ev.get("loss") is not None else step_losses.get(ev["step"]),
                    "sample_url": previews.get(ev["step"]),
                }
            )
    return out


@router.delete("/jobs/{jid}/checkpoints", response_model=m.Ok, response_model_exclude_unset=True)
def delete_job_checkpoint(jid: str, path: str, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    """Delete a saved output or resume point of this job from disk."""
    job = _get_job(c, jid)
    saved = next(
        (
            ev
            for ev in _events_file(c, jid)
            if ev.get("type") == "checkpoint.saved" and str(Path(ev["path"])) == str(Path(path))
        ),
        None,
    )
    target = Path(path)
    if saved is None or not target.exists():
        raise NotFound("checkpoint not found", code="checkpoint.not_found")
    if saved.get("kind") != "full":
        artifact = c.db.fetchone("SELECT id FROM artifacts WHERE job_id=? AND path=?", (jid, str(target)))
        if artifact:
            return delete_artifact(artifact["id"], delete_file=True, c=c)
        if job["project_id"]:
            assert_version_writable(c, job["project_id"], job.get("version_id"))
        if target.is_file():
            target.unlink()
        return {"ok": True}
    if job["project_id"]:
        assert_version_writable(c, job["project_id"], job.get("version_id"))
    # A queued, running or paused job may still resume from this state.
    waiting = c.db.fetchall(
        f"SELECT id FROM jobs WHERE status IN {ACTIVE_JOBS[:-1]},'paused') "
        "AND (resume_from=? OR json_extract(config_json,'$.checkpoint.resume')=?)",
        (str(target), str(target)),
    )
    if waiting:
        raise ApiError(
            "这个恢复点正被排队、运行或暂停中的训练使用，任务结束或取消后才能删除。",
            code="checkpoint.in_use",
            status=409,
        )
    if target.is_symlink() or not target.is_dir() or not (target / "state.json").is_file():
        raise ApiError("not a resume point directory", code="checkpoint.invalid", status=409)
    shutil.rmtree(target)
    c.bus.publish("job.checkpoint", {"job_id": jid, "deleted": str(target)})
    return {"ok": True}


@router.get("/jobs/{jid}/files")
def job_file(jid: str, path: str, kind: str = "sample", c: ServiceContext = Depends(ctx)) -> Response:
    r = _get_job(c, jid)
    base = (
        (Path(r["samples_dir"]) if r.get("samples_dir") else Path(r["run_dir"]) / "samples")
        if kind == "sample"
        else Path(r["run_dir"])
    )
    target = (base / Path(path).name).resolve()
    if not target.exists() or base.resolve() not in target.parents:
        raise NotFound("file not found", code="file.not_found")
    return FileResponse(str(target))


@router.get("/jobs/{jid}/log", response_model=m.JobLog, response_model_exclude_unset=True)
def job_log(
    jid: str,
    offset: int = 0,
    limit: int = 1000,
    c: ServiceContext = Depends(ctx),
    tail: bool = False,
    before: int | None = None,
) -> dict[str, Any]:
    # The exit record is appended under this lock with the outcome it records, so a snapshot that
    # sees a stopped worker sees its final file; the page itself is read without holding the lock.
    with c.db.lock:
        r = _get_job(c, jid)
        # A live worker may be mid-line; a stopped one has written its final output.
        live = c.supervisor.is_running(jid) and r.get("exit_code") is None
        path = log_file(r)
        terminal = None if live else missing_failure_record(path, r)
    page = read_log(path, offset=offset, limit=limit, tail=tail, before=before, complete_only=live)
    # This separate record must never advance the worker file's byte cursor.
    page["terminal"] = terminal
    return page


@router.get(
    "/jobs/{jid}/log/raw",
    response_class=FileResponse,
    responses={200: {"content": {"text/plain": {}}, "description": "The complete job log, including saved supervisor failures"}},
)
def job_log_raw(jid: str, c: ServiceContext = Depends(ctx)) -> Response:
    with c.db.lock:
        r = _get_job(c, jid)
        path = log_file(r)
        live = c.supervisor.is_running(jid) and r.get("exit_code") is None
        terminal = None if live else missing_failure_record(path, r)
    if terminal is not None:
        def contents():
            last = b""
            remaining = terminal["offset"]
            try:
                with path.open("rb") as stream:
                    while remaining and (chunk := stream.read(min(remaining, 64 * 1024))):
                        remaining -= len(chunk)
                        last = chunk[-1:]
                        yield chunk
            except FileNotFoundError:
                pass
            if last and last != b"\n":
                yield b"\n"
            yield failure_record_bytes(terminal)

        return StreamingResponse(
            contents(),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{jid}-run.log"'},
        )
    if not path.is_file():
        raise NotFound("log file not found", code="file.not_found")
    return FileResponse(str(path), media_type="text/plain; charset=utf-8", filename=f"{jid}-run.log")


# --------------------------------------------------------------------------- queue settings
@router.get("/queue/settings", response_model=m.QueueSettings, response_model_exclude_unset=True)
def queue_settings(c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    return c.db.get_kv("queue.settings", {"held": False, "max_concurrent": None})


@router.get("/queue/devices", response_model=m.QueueDevices)
def queue_devices(c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    inventory = gpu_info()
    with c.db.lock:
        ownership = {}
        for jid, allocation in list(c.supervisor._devices.items()):
            job = c.db.fetchone("SELECT name,status FROM jobs WHERE id=?", (jid,))
            for device in c.supervisor._allocation(allocation):
                ownership[device] = {
                    "job_id": jid,
                    "job_name": job["name"] if job else jid,
                    "status": job["status"] if job else "running",
                }
        return {
            "devices": [gpu | ownership.get(gpu["device"], {}) for gpu in inventory],
            "max_concurrent": queue_settings(c).get("max_concurrent"),
            "blocked_reason": maintenance_reason(c.db),
            "restart_required": bool(c.db.get_kv("environment.maintenance", {}).get("restart_required")),
        }


@router.put("/queue/settings", response_model=m.QueueSettings, response_model_exclude_unset=True)
def put_queue_settings(body: m.QueueSettings, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    cur = queue_settings(c)
    cur.update(body.model_dump(exclude_unset=True))
    c.db.set_kv("queue.settings", cur)
    c.bus.publish("queue.changed", {})
    return cur


# --------------------------------------------------------------------------- artifacts
ADAPTER_ARTIFACT_KINDS = {"weights", "comfyui", "kohya"}


@lru_cache(maxsize=1024)
def _adapter_header(path: str, mtime_ns: int, size: int) -> tuple[tuple[str, ...], dict[str, str]]:
    """Tensor names and metadata of a safetensors file, read from its header alone."""
    del mtime_ns, size  # they key the cache, so a rewritten file is read again
    from safetensors import safe_open

    with safe_open(path, framework="pt") as file:
        return tuple(file.keys()), dict(file.metadata() or {})


def _artifact_row(r: dict[str, Any], c: ServiceContext | None = None) -> dict[str, Any]:
    from .artifact_types import export_type

    out = dict(r)
    meta = json.loads(r.get("meta_json") or "{}")
    out.pop("meta_json", None)
    path = Path(r["path"])
    if r["kind"] == "model" and path.is_dir():
        try:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("format") == "ypuddin-full-model-v1":
                meta = {
                    "training_mode": "full",
                    "components": list(manifest.get("components", {})),
                    "frozen_assets": manifest.get("frozen_assets"),
                }
                out["family"] = manifest.get("family")
        except (OSError, ValueError):
            pass
    elif path.is_file():
        from ypuddin.adapters.convert import has_legacy_adapter_keys

        try:
            stat = path.stat()
            keys, header = _adapter_header(str(path), stat.st_mtime_ns, stat.st_size)
        except Exception:  # noqa: BLE001 - an unreadable file still lists; its download shows what it is
            keys, header = (), None
        if header is not None and r["kind"] in ADAPTER_ARTIFACT_KINDS:
            out["legacy_text_keys"] = has_legacy_adapter_keys(keys, header)
        if not meta and header is not None:
            meta = header
            args = {}
            for field in ("ss_network_args", "ypuddin.adapter"):
                try:
                    decoded = json.loads(header.get(field, "{}"))
                except (TypeError, json.JSONDecodeError):
                    continue
                if isinstance(decoded, dict):
                    args.update(decoded)
            rank = header.get("ss_network_dim", args.get("rank"))
            if isinstance(rank, str) and rank.isdecimal():
                rank = int(rank)
            alpha = header.get("ss_network_alpha", args.get("alpha"))
            if alpha != "full":
                try:
                    alpha = float(alpha)
                except (TypeError, ValueError):
                    alpha = None
                if alpha is not None and not math.isfinite(alpha):
                    alpha = None
            out.update(
                {
                    "algo": args.get("algo"),
                    "rank": rank,
                    "alpha": alpha,
                    "factor": args.get("factor"),
                    "family": header.get("ypuddin.family") or header.get("ss_base_model_version") or args.get("model_family"),
                }
            )
    out["metadata"] = {k: v for k, v in meta.items() if k != "ypuddin.targets"}
    out["export_type"] = export_type(r["kind"], path, family=out.get("family") or meta.get("ypuddin.family"))
    if out["export_type"] is None and r["kind"] == "model" and c and r.get("job_id"):
        job = c.db.fetchone("SELECT config_json FROM jobs WHERE id=?", (r["job_id"],))
        if job:
            out["export_type"] = export_type(r["kind"], path, config_json=job["config_json"])
    return out


@router.get("/artifacts", response_model=list[m.Artifact], response_model_exclude_unset=True)
def list_artifacts(
    project_id: str | None = None,
    c: ServiceContext = Depends(ctx),
    version_id: str | None = None,
    job_id: str | None = None,
) -> list[dict[str, Any]]:
    from .artifact_inventory import artifact_exists

    conditions, params = [], []
    for key, value in (("project_id", project_id), ("version_id", version_id), ("job_id", job_id)):
        if value:
            conditions.append(key + "=?")
            params.append(value)
    sql = "SELECT * FROM artifacts" + (" WHERE " + " AND ".join(conditions) if conditions else "")
    return [
        _artifact_row(r, c)
        for r in c.db.fetchall(sql + " ORDER BY created_at DESC", tuple(params))
        if artifact_exists(r)
    ]


def _get_artifact(c: ServiceContext, aid: str) -> dict[str, Any]:
    r = c.db.fetchone("SELECT * FROM artifacts WHERE id=?", (aid,))
    if not r:
        raise NotFound(f"artifact {aid} not found", code="artifact.not_found")
    return r


@router.get("/artifacts/{aid}", response_model=m.Artifact, response_model_exclude_unset=True)
def get_artifact(aid: str, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    return _artifact_row(_get_artifact(c, aid), c)


@router.delete("/artifacts/{aid}", response_model=m.Ok, response_model_exclude_unset=True)
def delete_artifact(aid: str, delete_file: bool = False, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    with c.db.lock:
        r = _get_artifact(c, aid)
        if r["project_id"]:
            assert_version_writable(c, r["project_id"], r.get("version_id"))
        references = c.db.fetchall(
            f"SELECT jobs.id, jobs.status IN {ACTIVE_JOBS} AS active FROM jobs "
            "JOIN json_each(jobs.config_json,'$.xyz.checkpoints') AS checkpoints "
            "WHERE jobs.type='xyz' AND checkpoints.key=?",
            (aid,),
        )
        if any(row["active"] or c.supervisor.is_running(row["id"]) for row in references):
            raise ApiError(
                "排队或运行中的模型测试正在使用此检查点，请先取消该测试并等待进程退出。",
                code="artifact.xyz_dependencies",
                status=409,
            )
        if delete_file:
            path = Path(r["path"])
            if r["kind"] == "model" and path.is_dir() and not path.is_symlink():
                try:
                    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    manifest = {}
                if manifest.get("format") != "ypuddin-full-model-v1":
                    raise ApiError(
                        "Model artifact manifest is missing or invalid", code="artifact.manifest", status=409
                    )
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        c.db.delete("artifacts", aid)
    return {"ok": True}


@router.get("/artifacts/{aid}/download")
def download_artifact(aid: str, c: ServiceContext = Depends(ctx)) -> Response:
    r = _get_artifact(c, aid)
    path = Path(r["path"])
    if r["kind"] == "model" and path.is_dir():
        if path.is_symlink() or not (path / "manifest.json").is_file():
            raise ApiError("Invalid model artifact directory", code="artifact.manifest", status=409)
        folder = Path(tempfile.mkdtemp(prefix="ypuddin-model-export-"))
        target = folder / (path.name + ".zip")
        try:
            with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for file in sorted(path.rglob("*")):
                    if file.is_symlink() or not file.resolve().is_relative_to(path.resolve()):
                        raise ApiError(
                            "Model artifact contains an external link", code="artifact.path", status=409
                        )
                    if file.is_file():
                        archive.write(file, str(Path(path.name) / file.relative_to(path)))
            return FileResponse(
                target,
                filename=target.name,
                background=BackgroundTask(shutil.rmtree, folder, ignore_errors=True),
            )
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
    return FileResponse(path, filename=path.name)


@router.post("/artifacts/{aid}/fix-text-keys", response_model=m.Artifact, response_model_exclude_unset=True)
def fix_artifact_text_keys(aid: str, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    """Rename an older adapter's keys to the names ComfyUI reads; weights stay the same."""
    r = _get_artifact(c, aid)
    with (
        c.versions.mutation(r["project_id"], r.get("version_id"), data=False)
        if r["project_id"]
        else nullcontext()
    ):
        return _fix_text_keys(c, r)


def _fix_text_keys(c: ServiceContext, r: dict) -> dict[str, Any]:
    if r["kind"] not in ADAPTER_ARTIFACT_KINDS:
        raise ApiError(
            "仅适配器权重支持修复键名。", code="artifact.kind", status=422
        )
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    from ypuddin.adapters.convert import modernize_adapter_keys

    path = Path(r["path"])
    if not path.is_file():
        raise NotFound("artifact file is missing", code="artifact.missing")
    # Compare file names directly: the training loader canonicalizes Klein aliases on read.
    tensors = load_file(str(path))
    with safe_open(str(path), framework="pt") as file:
        meta = dict(file.metadata() or {})
    try:
        renamed, renamed_meta = modernize_adapter_keys(tensors, meta)
    except ValueError as exc:
        raise ApiError(str(exc), code="artifact.duplicate_keys", status=422) from exc
    if renamed.keys() == tensors.keys() and renamed_meta == meta:
        return _artifact_row(r)
    staged = path.with_name(f".{path.name}.{new_id('k')}.tmp")
    try:
        save_file({k: v.contiguous() for k, v in renamed.items()}, str(staged), metadata=renamed_meta)
        # Whoever could read the file before still can, e.g. an inference tool under another account.
        os.chmod(staged, path.stat().st_mode & 0o7777)
        # The file is replaced in one step, so a reader sees either the old names or the new ones.
        os.replace(staged, path)
    except PermissionError as exc:
        raise ApiError(
            "the file is in use; try again when nothing reads it", code="artifact.busy", status=409
        ) from exc
    finally:
        staged.unlink(missing_ok=True)
    c.db.update("artifacts", r["id"], {"size": path.stat().st_size})
    return _artifact_row(c.db.fetchone("SELECT * FROM artifacts WHERE id=?", (r["id"],)))


_ = io
