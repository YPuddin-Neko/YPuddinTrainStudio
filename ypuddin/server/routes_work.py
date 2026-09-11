"""Projects / datasets / jobs / artifacts endpoints."""

from __future__ import annotations

import asyncio
import io
import json
import re
import shutil
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, field_validator

from ypuddin.config import DatasetSourceConfig, TrainConfig, deep_merge
from ypuddin.config.io import absolute_paths
from ypuddin.data import IndexDB, scan_sources

from . import models as m
from .context import ServiceContext
from .dataset_uploads import UploadBatch, read_upload, staged_upload
from .db import new_id, now
from .errors import ApiError, NotFound
from .hardware import gpu_info
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


def _project_row(c: ServiceContext, r: dict[str, Any]) -> dict[str, Any]:
    ds = c.db.fetchall("SELECT id FROM datasets WHERE version_id=?", (r["active_version_id"],))
    jobs = c.db.fetchone("SELECT COUNT(*) AS n FROM jobs WHERE project_id=?", (r["id"],))["n"]
    arts = c.db.fetchone("SELECT COUNT(*) AS n FROM artifacts WHERE project_id=?", (r["id"],))["n"]
    return {
        **r,
        "archived": bool(r["archived"]),
        "dataset_ids": [d["id"] for d in ds],
        "version_count": c.db.fetchone(
            "SELECT count(*) n FROM project_versions WHERE project_id=?", (r["id"],)
        )["n"],
        "stats": {"jobs": jobs, "artifacts": arts},
    }


@router.get("/projects", response_model=list[m.Project], response_model_exclude_unset=True)
def list_projects(include_archived: bool = False, c: ServiceContext = Depends(ctx)) -> list[dict[str, Any]]:
    rows = c.db.fetchall(
        "SELECT * FROM projects"
        + ("" if include_archived else " WHERE archived=0")
        + " ORDER BY created_at DESC"
    )
    return [_project_row(c, r) for r in rows]


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
            _write_project_config(c, pid, get_project_config(pid, c, vid), vid)
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
        if body.active_version_id:
            assert_version_writable(c, pid, body.active_version_id)
        fields["updated_at"] = now()
        c.db.update("projects", pid, fields)
        return _project_row(c, _get_project(c, pid))


@router.delete("/projects/{pid}", response_model=m.Ok, response_model_exclude_unset=True)
def delete_project(pid: str, delete_files: bool = False, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    with c.db.lock:
        _get_project(c, pid)
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
        if delete_files:
            for job in c.db.fetchall("SELECT id, run_dir, samples_dir FROM jobs WHERE project_id=?", (pid,)):
                run = Path(job["run_dir"])
                if run.name == job["id"] and run.is_dir() and not run.is_symlink():
                    shutil.rmtree(run)
                samples = Path(job["samples_dir"]) if job["samples_dir"] else None
                if samples and samples.name == job["id"] and samples.is_dir() and not samples.is_symlink():
                    shutil.rmtree(samples)
        c.db.execute("DELETE FROM jobs WHERE project_id=?", (pid,))
        c.db.execute("DELETE FROM artifacts WHERE project_id=?", (pid,))
        c.db.delete("projects", pid)
        if delete_files and project_root.exists():
            shutil.rmtree(project_root)
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
        return json.loads(f.read_text(encoding="utf-8"))
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
        configured = {}
        for section in ("dataset", "validation"):
            value = body.get(section)
            for source in value.get("sources", []) if isinstance(value, dict) else []:
                if isinstance(source, dict) and source.get("path"):
                    path = str(Path(source["path"]).expanduser().resolve())
                    extension = _validate_caption_extension(source.get("caption_ext", ".txt"))
                    if path in configured and configured[path] != extension:
                        raise ApiError(
                            "one source folder cannot have conflicting training/validation caption extensions",
                            code="dataset.caption_ext",
                        )
                    configured[path] = extension
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
        for did, extension in reindex:
            c.db.update("datasets", did, {"caption_ext": extension, "index_status": "indexing"})
            _records_path(c, did).unlink(missing_ok=True)
        c.db.update("project_versions", version["id"], {"updated_at": now()})
        c.db.update("projects", pid, {"updated_at": now()})
    for did, _ in reindex:
        _index_dataset(c, did)
    return body


def _validate_caption_extension(value: str) -> str:
    from ypuddin.data.index import IMAGE_EXTS

    if (
        not isinstance(value, str)
        or not re.fullmatch(r"\.[A-Za-z0-9][A-Za-z0-9._-]{0,31}", value)
        or value.lower() in IMAGE_EXTS
        or value.lower() in {".mask", ".mask.png"}
    ):
        raise ApiError(
            "caption_ext must be a text sidecar suffix such as .txt or .caption, not an image, mask or path",
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
        pid, body.name.strip(), body.note, body.source_version_id, body.data_mode, body.copy_config
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
    caption_ext: str = ".txt"
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
            if _same_source(item.get("path"), origin_path or str(p))
        ]
        if not matching:
            sources.append(source)
        else:
            explicit = body.model_dump(exclude={"version_id"}, exclude_unset=True) | {"path": str(p)}
            for item in matching:
                item.update(explicit)  # Preserve each source's role and all unedited advanced settings.
            source = {key: matching[0].get(key, value) for key, value in source.items()}
        _validate_caption_extension(source["caption_ext"])
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


def _register_upload(c: ServiceContext, pid: str, batch: UploadBatch, version_id: str | None = None) -> str:
    did = new_id("d")
    with c.versions.mutation(pid, version_id) as version:
        with staged_upload(
            c.version_dir(pid, version["id"]),
            did,
            batch,
            dataset_root=c.dataset_dir(pid, version["id"], is_reg=batch.is_reg),
        ) as directory:
            _register_dataset(
                c,
                pid,
                DatasetBody(
                    path=str(directory),
                    repeats=batch.repeats,
                    is_reg=batch.is_reg,
                    prior_weight=batch.prior_weight,
                    class_prompt=batch.class_prompt,
                    caption_ext=batch.caption_ext,
                ),
                did=did,
                version_id=version["id"],
                internal=True,
            )
    return did


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
    source = {
        "id": r["id"],
        "project_id": r["project_id"],
        "version_id": r.get("version_id"),
        "path": r["path"],
        "repeats": r["repeats"],
        "caption_ext": r["caption_ext"],
        "is_reg": bool(r["is_reg"]),
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
) -> dict[str, Any]:
    version = c.resolve_version(pid, version_id or body.version_id)
    did = c.versions.import_directory(pid, version["id"], body)
    background_tasks.add_task(_index_dataset, c, did)
    return _dataset_row(c, c.db.fetchone("SELECT * FROM datasets WHERE id=?", (did,)))


@router.post(
    "/projects/{pid}/datasets/upload",
    response_model=m.DatasetInfo,
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
                            "caption_ext": {"type": "string", "default": ".txt"},
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
) -> dict[str, Any]:
    version = assert_version_writable(c, pid, version_id, data=True)
    async with read_upload(request) as batch:
        did = await run_in_threadpool(_register_upload, c, pid, batch, version["id"])
    background_tasks.add_task(_index_dataset, c, did)
    return _dataset_row(c, _get_dataset(c, did))


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


@router.get("/datasets/{did}/images", response_model=m.ImagePage, response_model_exclude_unset=True)
def list_images(
    did: str, page: int = 1, page_size: int = 50, q: str = "", c: ServiceContext = Depends(ctx)
) -> dict[str, Any]:
    from ypuddin.data import read_caption

    row = _get_dataset(c, did)
    root = Path(row["path"])
    items = []
    for r in _records(c, did):
        cap = read_caption(r["caption_path"], row["class_prompt"])
        if q and q.lower() not in cap.lower() and q.lower() not in r["path"].lower():
            continue
        items.append(
            {
                "hash": r["content_hash"],
                "rel_path": str(Path(r["path"]).relative_to(root))
                if r["path"].startswith(str(root))
                else r["path"],
                "width": r["width"],
                "height": r["height"],
                "caption": cap,
                "has_mask": bool(r["mask_path"]),
            }
        )
    return _page(items, page, page_size)


def _record_by_hash(c: ServiceContext, did: str, h: str) -> dict[str, Any]:
    for r in _records(c, did):
        if r["content_hash"] == h:
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


class CaptionBody(BaseModel):
    caption: str


@router.get("/datasets/{did}/images/{h}/caption", response_model=m.Caption, response_model_exclude_unset=True)
def get_caption(did: str, h: str, c: ServiceContext = Depends(ctx)) -> dict[str, str]:
    from ypuddin.data import read_caption

    row = _get_dataset(c, did)
    r = _record_by_hash(c, did, h)
    return {"caption": read_caption(r["caption_path"], row["class_prompt"])}


@router.put("/datasets/{did}/images/{h}/caption", response_model=m.Caption, response_model_exclude_unset=True)
def put_caption(did: str, h: str, body: CaptionBody, c: ServiceContext = Depends(ctx)) -> dict[str, str]:
    row = _get_dataset(c, did)
    with c.versions.mutation(row["project_id"], row["version_id"]):
        row = _get_dataset(c, did)
        r = _record_by_hash(c, did, h)
        cap_path = (
            Path(r["caption_path"]) if r["caption_path"] else Path(r["path"]).with_suffix(row["caption_ext"])
        )
        cap_path.write_text(body.caption.strip() + "\n", encoding="utf-8")
        if not r["caption_path"]:
            recs = _records(c, did)
            for rec in recs:
                if rec["content_hash"] == h:
                    rec["caption_path"] = str(cap_path)
            _records_path(c, did).write_text(json.dumps(recs), encoding="utf-8")
        c.bus.publish("dataset.changed", {"dataset_id": did, "reason": "caption"})
        return {"caption": body.caption.strip()}


class TagBatch(BaseModel):
    hashes: list[str]
    add: list[str] = Field(default_factory=list)
    remove: list[str] = Field(default_factory=list)


@router.post("/datasets/{did}/tags/batch", response_model=m.TagBatchResult, response_model_exclude_unset=True)
def tags_batch(did: str, body: TagBatch, c: ServiceContext = Depends(ctx)) -> dict[str, int]:
    from ypuddin.data import read_caption

    row = _get_dataset(c, did)
    with c.versions.mutation(row["project_id"], row["version_id"]):
        row = _get_dataset(c, did)
        changed = 0
        created = 0
        wanted = set(body.hashes)
        recs = _records(c, did)
        for r in recs:
            if r["content_hash"] not in wanted:
                continue
            cap_path = (
                Path(r["caption_path"])
                if r["caption_path"]
                else Path(r["path"]).with_suffix(row["caption_ext"])
            )
            tags = [
                t.strip()
                for t in read_caption(r["caption_path"], row["class_prompt"]).split(",")
                if t.strip()
            ]
            tags = [t for t in tags if t not in body.remove]
            for t in body.add:
                if t not in tags:
                    tags.append(t)
            cap_path.write_text(", ".join(tags) + "\n", encoding="utf-8")
            if not r["caption_path"]:
                r["caption_path"] = str(
                    cap_path
                )  # a freshly created caption file must be found on the next read
                created += 1
            changed += 1
        if created:
            _records_path(c, did).write_text(json.dumps(recs), encoding="utf-8")
        if changed:
            c.bus.publish("dataset.changed", {"dataset_id": did, "reason": "tags"})
        return {"changed": changed}


# --------------------------------------------------------------------------- jobs
class JobBody(BaseModel):
    type: str = "train"
    name: str
    project_id: str | None = None
    version_id: str | None = None
    config: dict[str, Any] | None = None
    priority: int = 0
    scheduled_at: float | None = None


class JobPatch(BaseModel):
    priority: int | None = None
    name: str | None = None


def _job_row(r: dict[str, Any]) -> dict[str, Any]:
    out = dict(r)
    out["progress"] = json.loads(r.get("progress_json") or "{}")
    out["latest"] = json.loads(r.get("latest_json") or "{}")
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
) -> dict[str, Any]:
    sql, params = "SELECT * FROM jobs", []
    conds = []
    if status:
        conds.append("status IN ({})".format(",".join("?" for _ in status.split(","))))
        params += status.split(",")
    if project_id:
        conds.append("project_id=?")
        params.append(project_id)
    if version_id:
        conds.append("version_id=?")
        params.append(version_id)
    if conds:
        sql += " WHERE " + " AND ".join(conds)
    sql += " ORDER BY CASE status WHEN 'running' THEN 0 WHEN 'pausing' THEN 0 WHEN 'queued' THEN 1 WHEN 'scheduled' THEN 2 ELSE 3 END, priority DESC, created_at DESC"
    rows = [_job_row(r) for r in c.db.fetchall(sql, tuple(params))]
    return _page(rows, page, page_size)


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
    preflight = plan(
        cfg,
        gpu_total_mb=max((g["mem_total_mb"] for g in devices), default=None),
        device=devices[0]["device"] if devices else "cpu",
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
                "scheduled_at": body.scheduled_at,
                "created_at": now(),
                "run_dir": str(run_dir),
                "samples_dir": str(samples_dir),
                "config_json": json.dumps(cfg.to_dict()),
                "progress_json": json.dumps(
                    {"estimated_peak_mb": preflight.get("memory", {}).get("peak_mb_estimate")}
                ),
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
    _get_job(c, jid)
    c.db.update("jobs", jid, {k: v for k, v in body.model_dump().items() if v is not None})
    c.bus.publish("queue.changed", {})
    return _job_row(_get_job(c, jid))


@router.delete("/jobs/{jid}", response_model=m.Ok, response_model_exclude_unset=True)
def delete_job(jid: str, delete_files: bool = False, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    with c.db.lock:
        r = _get_job(c, jid)
        if r["status"] in ("running", "pausing", "cancelling") or c.supervisor.is_running(jid):
            raise ApiError("cancel the job first", code="job.running", status=409)
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
    p = Path(r["run_dir"]) / "events.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


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
    for ev in _events_file(c, jid):
        if ev.get("type") == "sample.saved":
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
def job_log(jid: str, offset: int = 0, limit: int = 2000, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    r = _get_job(c, jid)
    p = Path(r["run_dir"]) / "run.log"
    if not p.exists():
        return {"lines": [], "next_offset": 0}
    data = p.read_bytes()
    chunk = data[offset:]
    text = chunk.decode("utf-8", errors="replace")
    lines = text.splitlines()
    lines = lines[:limit]
    consumed = len("\n".join(lines).encode("utf-8")) + (1 if lines else 0)
    out = []
    for ln in lines:
        level = "info"
        low = ln.lower()
        if " error" in low or "traceback" in low or "exception" in low:
            level = "error"
        elif "warn" in low:
            level = "warning"
        out.append({"ts": None, "level": level, "msg": ln})
    return {"lines": out, "next_offset": min(len(data), offset + consumed)}


# --------------------------------------------------------------------------- queue settings
@router.get("/queue/settings", response_model=m.QueueSettings, response_model_exclude_unset=True)
def queue_settings(c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    return c.db.get_kv("queue.settings", {"held": False, "max_concurrent": 1})


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
    if not meta and Path(r["path"]).exists():
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
        if Path(r["path"]).is_file()
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
        c.db.delete("artifacts", aid)
        if delete_file:
            Path(r["path"]).unlink(missing_ok=True)
    return {"ok": True}


@router.get("/artifacts/{aid}/download")
def download_artifact(aid: str, c: ServiceContext = Depends(ctx)) -> Response:
    r = _get_artifact(c, aid)
    return FileResponse(r["path"], filename=Path(r["path"]).name)


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
