"""Projects / datasets / jobs / artifacts endpoints."""

from __future__ import annotations

import asyncio
import io
import json
import math
import re
import shutil
import tempfile
import unicodedata
import zipfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, field_validator
from starlette.background import BackgroundTask

from ypuddin.config import DatasetSourceConfig, TrainConfig, deep_merge
from ypuddin.config.io import absolute_paths
from ypuddin.data import IndexDB, scan_sources

from . import models as m
from .context import ServiceContext
from .dataset_uploads import UploadBatch, read_upload, staged_upload
from .db import new_id, now
from .errors import ApiError, NotFound
from .gpu_selection import GpuSelection, selection_error
from .hardware import gpu_info
from .import_progress import ImportProgress
from .project_covers import cover_path, cover_url, read_cover_upload, remove_cover, replace_cover, thumbnail
from .sample_events import read_events, samples_with_loss
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


def _project_row(c: ServiceContext, r: dict[str, Any]) -> dict[str, Any]:
    from .family_config import version_family

    ds = c.db.fetchall("SELECT id FROM datasets WHERE version_id=?", (r["active_version_id"],))
    jobs = c.db.fetchone("SELECT COUNT(*) AS n FROM jobs WHERE project_id=?", (r["id"],))["n"]
    arts = c.db.fetchone("SELECT COUNT(*) AS n FROM artifacts WHERE project_id=?", (r["id"],))["n"]
    return {
        **{key: value for key, value in r.items() if key != "cover_key"},
        "category": r.get("category"),
        "cover_url": cover_url(c, r),
        "active_family": version_family(c, c.resolve_version(r["id"], r["active_version_id"])),
        "archived": bool(r["archived"]),
        "dataset_ids": [d["id"] for d in ds],
        "version_count": c.db.fetchone(
            "SELECT count(*) n FROM project_versions WHERE project_id=?", (r["id"],)
        )["n"],
        "stats": {"jobs": jobs, "artifacts": arts},
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
            for name in ("traindata", "reg", "samples", "output", "cache"):
                (root / name).mkdir(parents=True, exist_ok=True)
            from .family_config import initial_family_config

            initial = initial_family_config(c, body.family)
            initial = deep_merge(
                initial,
                {
                    "checkpoint": {"output_dir": str(c.default_runs_dir(pid, vid))},
                    "dataset": {"cache_dir": str(c.cache_dir(pid, vid))},
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


@router.get("/projects/{pid}", response_model=m.Project, response_model_exclude_unset=True)
def get_project(pid: str, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    return _project_row(c, _get_project(c, pid))


@router.patch("/projects/{pid}", response_model=m.Project, response_model_exclude_unset=True)
def patch_project(pid: str, body: ProjectPatch, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    with c.db.lock:
        _get_project(c, pid)
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
        remove_cover(c, _get_project(c, pid))
        return _project_row(c, _get_project(c, pid))


@router.delete("/projects/{pid}", response_model=m.Ok, response_model_exclude_unset=True)
def delete_project(pid: str, delete_files: bool = False, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    with c.db.lock:
        project = _get_project(c, pid)
        project_root = c.project_dir(pid)
        if c.db.fetchone(f"SELECT id FROM jobs WHERE project_id=? AND status IN {ACTIVE_JOBS}", (pid,)):
            raise ApiError("project has running jobs", code="project.busy", status=409)
        if c.db.fetchone(
            "SELECT id FROM project_versions WHERE project_id=? AND (status='copying' OR busy IS NOT NULL)",
            (pid,),
        ):
            raise ApiError("project has an active data copy", code="project.busy", status=409)
        if any(
            c.supervisor.is_running(row["id"])
            for row in c.db.fetchall("SELECT id FROM jobs WHERE project_id=?", (pid,))
        ):
            raise ApiError("wait for the project's worker processes to exit", code="project.busy", status=409)
        if not project["archived"]:
            raise ApiError(
                "archive the project before permanently deleting it",
                code="project.archive_required",
                status=409,
            )
        if delete_files:
            try:
                for job in c.db.fetchall(
                    "SELECT id, run_dir, samples_dir FROM jobs WHERE project_id=?", (pid,)
                ):
                    run = Path(job["run_dir"])
                    if run.name == job["id"] and run.is_dir() and not run.is_symlink():
                        shutil.rmtree(run)
                    samples = Path(job["samples_dir"]) if job["samples_dir"] else None
                    if (
                        samples
                        and samples.name == job["id"]
                        and samples.is_dir()
                        and not samples.is_symlink()
                    ):
                        shutil.rmtree(samples)
                if project_root.exists():
                    shutil.rmtree(project_root)
            except OSError as exc:
                raise ApiError(
                    "could not remove all project files; the archived project is retained so you can retry",
                    code="project.delete_files_failed",
                    status=500,
                ) from exc
        c.db.execute("DELETE FROM jobs WHERE project_id=?", (pid,))
        c.db.execute("DELETE FROM artifacts WHERE project_id=?", (pid,))
        c.db.delete("projects", pid)
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

        return normalize_source_roles(c, pid, json.loads(f.read_text(encoding="utf-8")), version["id"])
    from .environment import environment_attention_default

    cfg = TrainConfig()
    cfg.model.attention = environment_attention_default(c)
    return deep_merge(
        cfg.to_dict(),
        {
            "checkpoint": {"output_dir": str(c.default_runs_dir(pid, version_id))},
            "dataset": {"cache_dir": str(c.cache_dir(pid, version_id))},
        },
    )


@router.put("/projects/{pid}/config")
def put_project_config(
    pid: str, body: dict[str, Any], c: ServiceContext = Depends(ctx), version_id: str | None = None
) -> dict[str, Any]:
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
    c.bus.publish("dataset.changed", {"dataset_id": did, "project_id": pid, "reason": "added"})
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
        c.bus.publish("dataset.changed", {"dataset_id": did, "project_id": pid, "reason": "imported"})
    return list(selected)


def _records_path(c: ServiceContext, did: str) -> Path:
    return c.data_root / "datasets" / f"{did}.json"


def _index_dataset(c: ServiceContext, did: str) -> None:
    row = c.db.fetchone("SELECT * FROM datasets WHERE id=?", (did,))
    if not row:
        return
    src = DatasetSourceConfig(
        path=row["path"],
        repeats=row["repeats"],
        caption_ext=row["caption_ext"],
        is_reg=bool(row["is_reg"]),
        prior_weight=row["prior_weight"],
        class_prompt=row["class_prompt"],
    )
    try:
        db = IndexDB(c.data_root / "cache" / "index.sqlite")
        try:
            records = scan_sources(
                [src],
                index_db=db,
                progress=lambda d, t: c.bus.publish(
                    "job.cache_progress", {"job_id": did, "kind": "index", "done": d, "total": t}
                ),
            )
        finally:
            db.close()
    except Exception as e:  # noqa: BLE001
        c.db.update("datasets", did, {"index_status": "failed", "stats_json": json.dumps({"error": str(e)})})
        c.bus.publish("dataset.changed", {"dataset_id": did})
        return
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
    }
    with c.db.lock:
        if not c.db.fetchone("SELECT id FROM datasets WHERE id=?", (did,)):
            return  # A source removed during indexing must not recreate its index file.
        _records_path(c, did).parent.mkdir(parents=True, exist_ok=True)
        _records_path(c, did).write_text(json.dumps([r.to_dict() for r in records]), encoding="utf-8")
        c.db.update("datasets", did, {"index_status": "ready", "stats_json": json.dumps(stats)})
    c.bus.publish("dataset.changed", {"dataset_id": did})


def _dataset_row(c: ServiceContext, r: dict[str, Any]) -> dict[str, Any]:
    from .source_roles import managed_source_role

    role = managed_source_role(c, r["project_id"], r.get("version_id"), r["path"])
    source = {
        "id": r["id"],
        "project_id": r["project_id"],
        "version_id": r.get("version_id"),
        "path": r["path"],
        "repeats": r["repeats"],
        "caption_ext": r["caption_ext"],
        "is_reg": role[0] if role is not None else bool(r["is_reg"]),
        "prior_weight": r["prior_weight"],
        "class_prompt": r["class_prompt"],
        "created_at": r["created_at"],
    }
    return {
        "source": source,
        "stats": json.loads(r["stats_json"] or "{}"),
        "index_status": r["index_status"],
        "cache": _cache_stats(c, r),
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
        if not recs:
            return {}
        ds = cfg.dataset
        from dataclasses import replace

        recs = [replace(rec, source_index=i) for i in range(len(ds.sources)) for rec in recs]
        bm = BucketManager(
            sorted({resolution for src in ds.sources for resolution in (src.resolutions or ds.resolutions)}),
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
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.model.dtype]
        if not devices or devices[0]["kind"] == "mps":
            dtype = torch.float32
        with fingerprint_cache(cache_root / "fingerprints"):
            latent_identity = family.latent_fingerprint(cfg.model, dtype=dtype)
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
    pid: str, c: ServiceContext = Depends(ctx), version_id: str | None = None
) -> list[dict[str, Any]]:
    version = c.resolve_version(pid, version_id)
    return [
        _dataset_row(c, r)
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


@router.get("/datasets/{did}", response_model=m.DatasetInfo, response_model_exclude_unset=True)
def get_dataset(did: str, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    return _dataset_row(c, _get_dataset(c, did))


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
    c.bus.publish("dataset.changed", {"dataset_id": did, "reason": "deleted"})
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
) -> dict[str, Any]:
    items = []
    query, exact_tag = q.casefold(), tag.strip().casefold() if tag else ""
    for item in _dataset_image_items(c, _get_dataset(c, did)):
        if query and query not in item["caption"].casefold() and query not in item["rel_path"].casefold():
            continue
        if caption_status and item["caption_status"] != caption_status:
            continue
        if exact_tag and exact_tag not in {token.casefold() for token in item["_tokens"]}:
            continue
        item.pop("_tokens")
        items.append(item)
    return _page(items, page, page_size)


def _record_by_hash(c: ServiceContext, did: str, h: str, rel_path: str | None = None) -> dict[str, Any]:
    root = Path(_get_dataset(c, did)["path"]) if rel_path is not None else None
    for r in _records(c, did):
        if r["content_hash"] == h and (root is None or Path(r["path"]) == root / rel_path):
            return r
    raise NotFound("image not found", code="image.not_found")


@router.get("/datasets/{did}/images/{h}/thumb")
def thumb(did: str, h: str, size: int = 256, c: ServiceContext = Depends(ctx)) -> Response:
    from PIL import Image

    r = _record_by_hash(c, did, h)
    cache = c.data_root / "thumbs" / f"{h}_{size}.jpg"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(r["path"]) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size))
            im.save(cache, "JPEG", quality=85)
    return FileResponse(
        str(cache), media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"}
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
        if r["caption_path"] != str(cap_path):
            recs = _records(c, did)
            for rec in recs:
                if rec["path"] == r["path"]:
                    rec["caption_path"] = str(cap_path)
            _records_path(c, did).write_text(json.dumps(recs), encoding="utf-8")
        c.bus.publish("dataset.changed", {"dataset_id": did, "reason": "caption"})
        return {"caption": read_caption(cap_path), "caption_structure": load_caption_structure(cap_path)}


class TagBatch(BaseModel):
    hashes: list[str]
    add: list[str] = Field(default_factory=list)
    remove: list[str] = Field(default_factory=list)


@router.post("/datasets/{did}/tags/batch", response_model=m.TagBatchResult, response_model_exclude_unset=True)
def tags_batch(did: str, body: TagBatch, c: ServiceContext = Depends(ctx)) -> dict[str, int]:
    import os

    from ypuddin.data.captions import caption_content, read_editable_caption
    from ypuddin.data.index import caption_target

    row = _get_dataset(c, did)
    with c.versions.mutation(row["project_id"], row["version_id"]):
        row = _get_dataset(c, did)
        changed = 0
        created = 0
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
                if r["content_hash"] not in wanted:
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
                changed += 1
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
        if changed:
            c.bus.publish("dataset.changed", {"dataset_id": did, "reason": "tags"})
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


class JobPatch(GpuSelection):
    priority: int | None = None
    name: str | None = None


def _job_row(r: dict[str, Any]) -> dict[str, Any]:
    out = dict(r)
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
    group: Literal["active", "waiting", "history"] | None = None,
    type: Literal["train", "cache", "xyz"] | None = None,
    q: str | None = None,
) -> dict[str, Any]:
    sql = " FROM jobs j LEFT JOIN projects p ON p.id=j.project_id LEFT JOIN project_versions v ON v.id=j.version_id"
    params = []
    conds = []
    if status:
        conds.append("j.status IN ({})".format(",".join("?" for _ in status.split(","))))
        params += status.split(",")
    if group:
        groups = {
            "active": ("running", "pausing", "cancelling", "paused"),
            "waiting": ("queued", "scheduled"),
            "history": ("completed", "failed", "cancelled"),
        }
        conds.append("j.status IN ({})".format(",".join("?" for _ in groups[group])))
        params += groups[group]
    if type:
        conds.append("j.type=?")
        params.append(type)
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
        "j.created_at DESC, j.id DESC"
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
    if body.project_id:
        from .output_binding import bind_output_name
        from .source_roles import normalize_source_roles

        config = normalize_source_roles(c, body.project_id, config, vid)
        config = bind_output_name(c, body.project_id, config, vid)
    jid = new_id("j")
    run_dir = c.job_output_dir(body.project_id, vid, jid, config.get("checkpoint", {}).get("output_dir"))
    samples_dir = c.samples_dir(body.project_id, vid) / jid if body.project_id else run_dir / "samples"
    config = deep_merge(
        config,
        {
            "checkpoint": {"output_dir": str(run_dir)},
            "logging": {"events_path": str(run_dir / "events.jsonl")},
            "sampling": {"output_dir": str(samples_dir)},
        },
    )
    if body.type == "cache":
        # Cache preparation has its own single-device worker, even when the
        # project's following training run is configured for several GPUs.
        config = deep_merge(config, {"loop": {"gpu_count": 1, "distributed_strategy": "ddp"}})
    if body.project_id or not (config.get("dataset") or {}).get("cache_dir"):
        # a pre-cache job and the training jobs after it must hit the same cache
        config = deep_merge(config, {"dataset": {"cache_dir": str(c.cache_dir(body.project_id, vid))}})
    from pydantic import ValidationError

    try:
        cfg = absolute_paths(TrainConfig.model_validate(config))
    except ValidationError as e:
        raise ApiError(
            "invalid config",
            code="config.invalid",
            details={
                "errors": [
                    {"loc": ".".join(str(x) for x in err["loc"]), "msg": err["msg"]} for err in e.errors()
                ]
            },
        ) from e
    from ypuddin.train.plan import plan

    devices = gpu_info()
    from .supervisor import training_device_error

    if error := training_device_error(cfg.loop.gpu_count, devices):
        raise ApiError(
            error,
            code="config.invalid",
            details={"errors": [{"loc": "loop.gpu_count", "msg": error}]},
        )
    if error := selection_error(body.gpu_devices, cfg.loop.gpu_count, devices):
        raise ApiError(error, code="job.gpu_selection", status=422)
    selected = [gpu for gpu in devices if gpu["device"] in body.gpu_devices]
    preflight = plan(
        cfg,
        gpu_total_mb=min((g["mem_total_mb"] for g in selected), default=None)
        if selected
        else max((g["mem_total_mb"] for g in devices), default=None),
        device=(selected or devices)[0]["device"] if devices else "cpu",
    )
    if not preflight["ok"]:
        raise ApiError(
            "training preflight failed", code="config.invalid", details={"errors": preflight["errors"]}
        )
    if cfg.checkpoint.resume:
        state = Path(cfg.checkpoint.resume).expanduser()
        if not (state / "state.json").is_file():
            raise ApiError(
                "resume must point to a complete state directory",
                code="config.invalid",
                details={"errors": [{"loc": "checkpoint.resume", "msg": "complete checkpoint not found"}]},
            )
    memory = preflight.get("memory") or {}
    estimated_peak_mb = memory.get("peak_mb_estimate")
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
                "gpu_devices_json": json.dumps(body.gpu_devices),
                "scheduled_at": body.scheduled_at,
                "created_at": now(),
                "run_dir": str(run_dir),
                "samples_dir": str(samples_dir),
                "config_json": json.dumps(cfg.to_dict()),
                "progress_json": json.dumps({"estimated_peak_mb": estimated_peak_mb}),
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
    return _job_row(_get_job(c, jid))


@router.patch("/jobs/{jid}", response_model=m.Job, response_model_exclude_unset=True)
def patch_job(jid: str, body: JobPatch, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    with c.db.lock:
        job = _get_job(c, jid)
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
        c.db.update("jobs", jid, patch)
    c.bus.publish("queue.changed", {})
    return _job_row(_get_job(c, jid))


@router.delete("/jobs/{jid}", response_model=m.Ok, response_model_exclude_unset=True)
def delete_job(jid: str, delete_files: bool = False, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    with c.db.lock:
        r = _get_job(c, jid)
        if r["status"] in ("running", "pausing", "cancelling") or c.supervisor.is_running(jid):
            raise ApiError("cancel the job first", code="job.running", status=409)
        if c.db.fetchone(
            "SELECT id FROM jobs WHERE type='xyz' AND json_extract(config_json,'$.xyz.source_job_id')=?",
            (jid,),
        ):
            raise ApiError(
                "请先删除使用此训练任务的模型测试记录，这些记录仍依赖该任务。",
                code="job.xyz_dependencies",
                status=409,
            )
        c.db.delete("jobs", jid)
        if delete_files and r["run_dir"] and Path(r["run_dir"]).exists():
            shutil.rmtree(r["run_dir"])
        if delete_files and r.get("samples_dir"):
            samples = Path(r["samples_dir"])
            if samples.name == jid and samples.is_dir() and not samples.is_symlink():
                shutil.rmtree(samples)
        c.bus.publish("queue.changed", {})
        return {"ok": True}


@router.post("/jobs/{jid}/{command}", response_model=m.Job, response_model_exclude_unset=True)
def job_command(jid: str, command: str, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    if command not in ("pause", "resume", "cancel", "save", "retry"):
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
    return read_events(Path(r["run_dir"]) / "events.jsonl")


@router.get("/jobs/{jid}/metrics", response_model=m.JobMetrics, response_model_exclude_unset=True)
def job_metrics(jid: str, since_step: int = 0, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    steps, loss, loss_ema, grad, vram, its = [], [], [], [], [], []
    lr: dict[str, list[float]] = {}
    validation = []
    vram_metric = None
    for ev in _events_file(c, jid):
        if ev.get("type") == "step" and ev["step"] > since_step:
            vram_metric = ev.get("vram_metric") or vram_metric
            steps.append(ev["step"])
            loss.append(ev.get("loss"))
            loss_ema.append(ev.get("loss_ema"))
            grad.append(ev.get("grad_norm"))
            vram.append(ev.get("vram_mb"))
            its.append(ev.get("it_s"))
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
    }


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
            }
        )
    return out


@router.get(
    "/jobs/{jid}/checkpoints", response_model=list[m.JobCheckpoint], response_model_exclude_unset=True
)
def job_checkpoints(jid: str, c: ServiceContext = Depends(ctx)) -> list[dict[str, Any]]:
    out = []
    arts = {
        a["path"]: a["id"] for a in c.db.fetchall("SELECT id, path FROM artifacts WHERE job_id=?", (jid,))
    }
    for ev in _events_file(c, jid):
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
                    "ema": bool(ev.get("ema")),
                }
            )
    return out


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
    jid: str, offset: int = 0, limit: int = 2000, c: ServiceContext = Depends(ctx), tail: bool = False
) -> dict[str, Any]:
    r = _get_job(c, jid)
    p = Path(r["run_dir"]) / "run.log"
    if not p.exists():
        return {"lines": [], "next_offset": 0, "has_more": False}
    limit = max(1, min(2000, limit))
    with p.open("rb") as stream:
        size = p.stat().st_size
        if tail:
            # Latest view is bounded even when a run has produced gigabytes of logs.
            stream.seek(max(0, size - 512 * 1024))
            if stream.tell():
                stream.readline(512 * 1024)
            lines = stream.read(512 * 1024).decode("utf-8", errors="replace").splitlines()[-limit:]
        else:
            stream.seek(min(size, max(0, offset)))
            lines = []
            start = stream.tell()
            for _ in range(limit):
                line = stream.readline(512 * 1024)
                if not line:
                    break
                lines.append(line.decode("utf-8", errors="replace").rstrip("\r\n"))
                if stream.tell() - start >= 512 * 1024:
                    break
        next_offset = stream.tell()
        has_more = bool(stream.read(1))
    out = []
    for ln in lines:
        level = "info"
        low = ln.lower()
        if " error" in low or "traceback" in low or "exception" in low:
            level = "error"
        elif "warn" in low:
            level = "warn"
        out.append({"ts": None, "level": level, "msg": ln})
    return {"lines": out, "next_offset": next_offset, "has_more": has_more}


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
        }


@router.put("/queue/settings", response_model=m.QueueSettings, response_model_exclude_unset=True)
def put_queue_settings(body: m.QueueSettings, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    cur = queue_settings(c)
    cur.update(body.model_dump(exclude_unset=True))
    c.db.set_kv("queue.settings", cur)
    c.bus.publish("queue.changed", {})
    return cur


# --------------------------------------------------------------------------- artifacts
def _artifact_row(r: dict[str, Any]) -> dict[str, Any]:
    out = dict(r)
    meta = json.loads(r.get("meta_json") or "{}")
    out.pop("meta_json", None)
    if r["kind"] == "model" and Path(r["path"]).is_dir():
        try:
            manifest = json.loads((Path(r["path"]) / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("format") == "ypuddin-full-model-v1":
                meta = {
                    "training_mode": "full",
                    "components": list(manifest.get("components", {})),
                    "frozen_assets": manifest.get("frozen_assets"),
                }
                out["family"] = manifest.get("family")
        except (OSError, ValueError):
            pass
    elif not meta and Path(r["path"]).exists():
        try:
            from ypuddin.adapters import load_adapter_file

            _, m = load_adapter_file(r["path"])
            meta = m
            args = json.loads(m.get("ypuddin.adapter", "{}"))
            out.update(
                {
                    "algo": args.get("algo"),
                    "rank": args.get("rank"),
                    "alpha": args.get("alpha"),
                    "factor": args.get("factor"),
                    "family": m.get("ypuddin.family"),
                }
            )
        except Exception:  # noqa: BLE001
            pass
    out["metadata"] = {k: v for k, v in meta.items() if k != "ypuddin.targets"}
    return out


@router.get("/artifacts", response_model=list[m.Artifact], response_model_exclude_unset=True)
def list_artifacts(
    project_id: str | None = None,
    c: ServiceContext = Depends(ctx),
    version_id: str | None = None,
    job_id: str | None = None,
) -> list[dict[str, Any]]:
    conditions, params = [], []
    for key, value in (("project_id", project_id), ("version_id", version_id), ("job_id", job_id)):
        if value:
            conditions.append(key + "=?")
            params.append(value)
    sql = "SELECT * FROM artifacts" + (" WHERE " + " AND ".join(conditions) if conditions else "")
    return [
        _artifact_row(r)
        for r in c.db.fetchall(sql + " ORDER BY created_at DESC", tuple(params))
        if Path(r["path"]).is_file() or (r["kind"] == "model" and Path(r["path"]).is_dir())
    ]


def _get_artifact(c: ServiceContext, aid: str) -> dict[str, Any]:
    r = c.db.fetchone("SELECT * FROM artifacts WHERE id=?", (aid,))
    if not r:
        raise NotFound(f"artifact {aid} not found", code="artifact.not_found")
    return r


@router.get("/artifacts/{aid}", response_model=m.Artifact, response_model_exclude_unset=True)
def get_artifact(aid: str, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    return _artifact_row(_get_artifact(c, aid))


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


class ConvertBody(BaseModel):
    format: str


@router.post("/artifacts/{aid}/convert", response_model=m.Artifact, response_model_exclude_unset=True)
def convert_artifact(aid: str, body: ConvertBody, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    r = _get_artifact(c, aid)
    with (
        c.versions.mutation(r["project_id"], r.get("version_id"), data=False)
        if r["project_id"]
        else nullcontext()
    ):
        return _convert_artifact(c, r, body)


def _convert_artifact(c: ServiceContext, r: dict, body: ConvertBody) -> dict[str, Any]:
    if r["kind"] not in {"weights", "comfyui", "kohya"}:
        raise ApiError(
            "Only adapter weight artifacts support format conversion", code="artifact.kind", status=422
        )
    from safetensors.torch import save_file

    from ypuddin.adapters import load_adapter_file
    from ypuddin.adapters.convert import comfy_to_kohya, kohya_to_comfy, lycoris_to_kohya
    from ypuddin.models import get_family

    tensors, meta = load_adapter_file(r["path"])
    family = meta.get("ypuddin.family", "anima")
    if body.format == "comfyui":
        fam = get_family(family)
        names = fam.linear_module_names() if hasattr(fam, "linear_module_names") else []
        out = kohya_to_comfy(tensors, names)
    elif body.format == "kohya":
        out = comfy_to_kohya(lycoris_to_kohya(tensors))
    else:
        raise ApiError(f"unknown format {body.format}", code="convert.bad_format")
    dst = Path(r["path"]).with_name(Path(r["path"]).stem + f"-{body.format}.safetensors")
    save_file({k: v.contiguous() for k, v in out.items()}, str(dst), metadata=meta)
    nid = new_id("a")
    c.db.insert(
        "artifacts",
        {
            "id": nid,
            "project_id": r["project_id"],
            "version_id": r.get("version_id"),
            "job_id": r["job_id"],
            "name": dst.name,
            "path": str(dst),
            "size": dst.stat().st_size,
            "kind": body.format,
            "step": r["step"],
            "created_at": now(),
            "meta_json": "{}",
        },
    )
    return _artifact_row(c.db.fetchone("SELECT * FROM artifacts WHERE id=?", (nid,)))


_ = io
