"""Projects / datasets / jobs / artifacts endpoints."""

from __future__ import annotations

import asyncio
import io
import json
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from ypuddin.config import DatasetSourceConfig, TrainConfig, deep_merge
from ypuddin.data import IndexDB, scan_sources

from . import models as m
from .context import ServiceContext
from .db import new_id, now
from .errors import ApiError, NotFound

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
    name: str
    note: str = ""


class ProjectPatch(BaseModel):
    name: str | None = None
    note: str | None = None
    archived: bool | None = None


def _project_row(c: ServiceContext, r: dict[str, Any]) -> dict[str, Any]:
    ds = c.db.fetchall("SELECT id FROM datasets WHERE project_id=?", (r["id"],))
    jobs = c.db.fetchone("SELECT COUNT(*) AS n FROM jobs WHERE project_id=?", (r["id"],))["n"]
    arts = c.db.fetchone("SELECT COUNT(*) AS n FROM artifacts WHERE project_id=?", (r["id"],))["n"]
    return {
        **r,
        "archived": bool(r["archived"]),
        "dataset_ids": [d["id"] for d in ds],
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
    pid = new_id("p")
    t = now()
    c.db.insert(
        "projects",
        {"id": pid, "name": body.name, "note": body.note, "archived": 0, "created_at": t, "updated_at": t},
    )
    c.project_dir(pid).mkdir(parents=True, exist_ok=True)
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
    _get_project(c, pid)
    fields = {
        k: (int(v) if isinstance(v, bool) else v) for k, v in body.model_dump().items() if v is not None
    }
    fields["updated_at"] = now()
    c.db.update("projects", pid, fields)
    return _project_row(c, _get_project(c, pid))


@router.delete("/projects/{pid}", response_model=m.Ok, response_model_exclude_unset=True)
def delete_project(pid: str, delete_files: bool = False, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    _get_project(c, pid)
    if c.db.fetchone(
        "SELECT id FROM jobs WHERE project_id=? AND status IN ('running','pausing','cancelling')", (pid,)
    ):
        raise ApiError("project has running jobs", code="project.busy", status=409)
    c.db.execute("DELETE FROM jobs WHERE project_id=?", (pid,))
    c.db.execute("DELETE FROM artifacts WHERE project_id=?", (pid,))
    c.db.delete("projects", pid)
    if delete_files and c.project_dir(pid).exists():
        shutil.rmtree(c.project_dir(pid))
    return {"ok": True}


@router.get("/projects/{pid}/config")
def get_project_config(pid: str, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    _get_project(c, pid)
    f = c.project_dir(pid) / "config.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    cfg = TrainConfig()
    return deep_merge(cfg.to_dict(), {"checkpoint": {"output_dir": str(c.runs_dir(pid))}})


@router.put("/projects/{pid}/config")
def put_project_config(pid: str, body: dict[str, Any], c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    _get_project(c, pid)
    d = c.project_dir(pid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    c.db.update("projects", pid, {"updated_at": now()})
    return body


# --------------------------------------------------------------------------- datasets
class DatasetBody(BaseModel):
    path: str
    repeats: int = 1
    caption_ext: str = ".txt"
    is_reg: bool = False
    prior_weight: float = 1.0
    class_prompt: str | None = None


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
    _records_path(c, did).parent.mkdir(parents=True, exist_ok=True)
    _records_path(c, did).write_text(json.dumps([r.to_dict() for r in records]), encoding="utf-8")
    c.db.update("datasets", did, {"index_status": "ready", "stats_json": json.dumps(stats)})
    c.bus.publish("dataset.changed", {"dataset_id": did})


def _dataset_row(c: ServiceContext, r: dict[str, Any]) -> dict[str, Any]:
    source = {
        "id": r["id"],
        "project_id": r["project_id"],
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

        raw = get_project_config(r["project_id"], c)
        raw.setdefault("dataset", {})["sources"] = [{"path": r["path"]}]
        cfg = TrainConfig.model_validate(raw)
        family = get_family(cfg.model.family)
        recs = [
            ImageRecord(**{k: v for k, v in rec.items() if k in ImageRecord.__dataclass_fields__})
            for rec in _records(c, r["id"])
        ]
        if not recs:
            return {}
        ds = cfg.dataset
        bm = BucketManager(
            ds.resolutions,
            align=family.spec.latent.align,
            step=ds.bucket_step,
            aspect_ratio_limit=ds.aspect_ratio_limit,
            area_tolerance=ds.area_tolerance,
            no_upscale=ds.bucket_no_upscale,
        )
        items = expand_items(recs, ds.sources, ds, bm)
        cache_root = Path(ds.cache_dir) if ds.cache_dir else c.cache_dir(r["project_id"])
        lc = LatentCache(cache_root / "latents")
        keys = {
            LatentCache.key(
                it.record.content_hash,
                it.bucket.width,
                it.bucket.height,
                family.spec.latent.fingerprint,
                flip,
            )
            for it in items
            for flip in ((False, True) if ds.flip else (False,))
        }
        cached = sum(1 for k in keys if lc.has(k))
        return {"latents": {"cached": cached, "total": len(keys)}, "cache_dir": str(cache_root)}
    except Exception as e:  # noqa: BLE001 - statistics must never break the dataset endpoint
        return {"error": str(e)}


@router.get("/projects/{pid}/datasets", response_model=list[m.DatasetInfo], response_model_exclude_unset=True)
def list_datasets(pid: str, c: ServiceContext = Depends(ctx)) -> list[dict[str, Any]]:
    _get_project(c, pid)
    return [
        _dataset_row(c, r)
        for r in c.db.fetchall("SELECT * FROM datasets WHERE project_id=? ORDER BY created_at", (pid,))
    ]


@router.post(
    "/projects/{pid}/datasets",
    status_code=201,
    response_model=m.DatasetInfo,
    response_model_exclude_unset=True,
)
async def add_dataset(pid: str, body: DatasetBody, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    _get_project(c, pid)
    p = Path(body.path).expanduser()
    if not p.is_dir():
        raise NotFound(f"directory not found: {p}", code="fs.not_found")
    did = new_id("d")
    c.db.insert(
        "datasets",
        {
            "id": did,
            "project_id": pid,
            "path": str(p),
            "repeats": body.repeats,
            "caption_ext": body.caption_ext,
            "is_reg": int(body.is_reg),
            "prior_weight": body.prior_weight,
            "class_prompt": body.class_prompt,
            "created_at": now(),
            "index_status": "indexing",
            "stats_json": "{}",
        },
    )
    asyncio.get_running_loop().run_in_executor(None, _index_dataset, c, did)
    return _dataset_row(c, c.db.fetchone("SELECT * FROM datasets WHERE id=?", (did,)))


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
    _get_dataset(c, did)
    c.db.update("datasets", did, {"index_status": "indexing"})
    asyncio.get_running_loop().run_in_executor(None, _index_dataset, c, did)
    return {"ok": True}


@router.delete("/datasets/{did}", response_model=m.Ok, response_model_exclude_unset=True)
def delete_dataset(did: str, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    _get_dataset(c, did)
    c.db.delete("datasets", did)
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
    changed = 0
    created = 0
    wanted = set(body.hashes)
    recs = _records(c, did)
    for r in recs:
        if r["content_hash"] not in wanted:
            continue
        cap_path = (
            Path(r["caption_path"]) if r["caption_path"] else Path(r["path"]).with_suffix(row["caption_ext"])
        )
        tags = [
            t.strip() for t in read_caption(r["caption_path"], row["class_prompt"]).split(",") if t.strip()
        ]
        tags = [t for t in tags if t not in body.remove]
        for t in body.add:
            if t not in tags:
                tags.append(t)
        cap_path.write_text(", ".join(tags) + "\n", encoding="utf-8")
        if not r["caption_path"]:
            r["caption_path"] = str(cap_path)  # a freshly created caption file must be found on the next read
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


@router.get("/jobs", response_model=list[m.Job], response_model_exclude_unset=True)
def list_jobs(
    status: str | None = None,
    project_id: str | None = None,
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
    if conds:
        sql += " WHERE " + " AND ".join(conds)
    sql += " ORDER BY CASE status WHEN 'running' THEN 0 WHEN 'pausing' THEN 0 WHEN 'queued' THEN 1 WHEN 'scheduled' THEN 2 ELSE 3 END, priority DESC, created_at DESC"
    rows = [_job_row(r) for r in c.db.fetchall(sql, tuple(params))]
    return _page(rows, page, page_size)


@router.post("/jobs", status_code=201, response_model=m.Job, response_model_exclude_unset=True)
def create_job(body: JobBody, c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    if body.type not in ("train", "cache"):
        raise ApiError(f"unsupported job type {body.type}", code="job.bad_type")
    config = body.config
    if config is None and body.project_id:
        config = get_project_config(body.project_id, c)
    if config is None:
        raise ApiError("config is required", code="job.no_config")
    jid = new_id("j")
    run_dir = c.runs_dir(body.project_id) / jid
    config = deep_merge(config, {"checkpoint": {"output_dir": str(run_dir)}})
    if not (config.get("dataset") or {}).get("cache_dir"):
        # a pre-cache job and the training jobs after it must hit the same cache
        config = deep_merge(config, {"dataset": {"cache_dir": str(c.cache_dir(body.project_id))}})
    from pydantic import ValidationError

    try:
        cfg = TrainConfig.model_validate(config)
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
    status = "scheduled" if body.scheduled_at and body.scheduled_at > now() else "queued"
    c.db.insert(
        "jobs",
        {
            "id": jid,
            "type": body.type,
            "name": body.name,
            "project_id": body.project_id,
            "status": status,
            "priority": body.priority,
            "scheduled_at": body.scheduled_at,
            "created_at": now(),
            "run_dir": str(run_dir),
            "config_json": json.dumps(cfg.to_dict()),
            "progress_json": "{}",
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
    r = _get_job(c, jid)
    if r["status"] in ("running", "pausing", "cancelling"):
        raise ApiError("cancel the job first", code="job.running", status=409)
    c.db.delete("jobs", jid)
    if delete_files and r["run_dir"] and Path(r["run_dir"]).exists():
        shutil.rmtree(r["run_dir"])
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
    for ev in _events_file(c, jid):
        if ev.get("type") == "step" and ev["step"] > since_step:
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
                }
            )
    return out


@router.get("/jobs/{jid}/files")
def job_file(jid: str, path: str, kind: str = "sample", c: ServiceContext = Depends(ctx)) -> Response:
    r = _get_job(c, jid)
    base = Path(r["run_dir"]) / ("samples" if kind == "sample" else "")
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
def put_queue_settings(body: dict[str, Any], c: ServiceContext = Depends(ctx)) -> dict[str, Any]:
    cur = queue_settings(c)
    cur.update({k: v for k, v in body.items() if k in ("held", "max_concurrent")})
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
def list_artifacts(project_id: str | None = None, c: ServiceContext = Depends(ctx)) -> list[dict[str, Any]]:
    sql, params = "SELECT * FROM artifacts", ()
    if project_id:
        sql, params = sql + " WHERE project_id=?", (project_id,)
    return [_artifact_row(r) for r in c.db.fetchall(sql + " ORDER BY created_at DESC", params)]


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
    r = _get_artifact(c, aid)
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
    from safetensors.torch import save_file

    from ypuddin.adapters import load_adapter_file
    from ypuddin.adapters.convert import comfy_to_kohya, kohya_to_comfy, lycoris_to_kohya
    from ypuddin.models import get_family

    r = _get_artifact(c, aid)
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
