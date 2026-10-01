"""Saved-training-checkpoint XYZ comparisons."""

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from . import xyz
from .errors import ApiError, NotFound
from .routes_core import ctx

router = APIRouter()


@router.get("/xyz/sources", response_model=xyz.XyzSourcePage)
def sources(project_id: str | None = None, version_id: str | None = None, q: str = "", page: int = 1, page_size: int = 50, context=Depends(ctx)):
    return xyz.sources(context, project_id=project_id, version_id=version_id, q=q, page=page, page_size=page_size)


@router.get("/xyz/sources/{source}", response_model=xyz.XyzSource)
def source_details(source: str, context=Depends(ctx)):
    return xyz.source_details(context, source)


@router.get("/jobs/{source}/xyz/options", response_model=xyz.XyzOptions)
def options(source: str, context=Depends(ctx)):
    return xyz.options(context, source)


@router.post("/jobs/{source}/xyz", response_model=xyz.XyzTask, status_code=202)
def create(source: str, body: xyz.XyzRequest, context=Depends(ctx)):
    return xyz.start(context, source, body)


@router.get("/jobs/{source}/xyz", response_model=list[xyz.XyzTask])
def history(source: str, context=Depends(ctx)):
    return xyz.history(context, source)


@router.get("/xyz/models", response_model=xyz.KeptModels)
def kept_models(context=Depends(ctx)):
    return {"models": context.supervisor.loaded_models()}


@router.post("/xyz/models/release", response_model=xyz.KeptModels)
def release_models(context=Depends(ctx)):
    """Unload the base models idle model-test workers keep; one that is drawing keeps its model."""
    context.supervisor.release_models(wait=10)
    return {"models": context.supervisor.loaded_models()}


@router.get("/xyz/{jid}", response_model=xyz.XyzTask)
def task(jid: str, context=Depends(ctx)):
    return xyz.task(context, jid)


@router.delete("/xyz/{jid}")
def delete(jid: str, context=Depends(ctx)):
    from .job_paths import owned_job_directories, removal_problem

    with context.db.lock:
        row = xyz._row(context, jid)
        if row["status"] not in {"completed", "failed", "cancelled"} or context.supervisor.is_running(jid):
            raise ApiError("请先取消生成，等待任务结束后再删除。", code="xyz.running", status=409)
        roots = owned_job_directories(row)
        result = xyz.result_root(context, row)
        if not roots or not any(result == root.resolve() or result.is_relative_to(root.resolve()) for root in roots):
            raise ApiError("模型测试文件不在该任务的独立目录内。", code="xyz.path", status=403)
        problem = removal_problem(context, jid, roots)
        if problem == "outside":
            raise ApiError("模型测试目录不在允许访问的范围内。", code="xyz.path", status=403)
        if problem == "shared":
            raise ApiError("目录中包含其他任务的文件，无法删除。", code="xyz.path", status=409)
        try:
            for root in roots:
                if root.is_dir():
                    shutil.rmtree(root)
        except OSError as exc:
            raise ApiError("部分模型测试文件未能删除，请重试。", code="xyz.delete_failed", status=500) from exc
        context.db.execute("DELETE FROM artifacts WHERE job_id=?", (jid,))
        context.db.delete("jobs", jid)
    context.bus.publish("queue.changed", {})
    return {"ok": True}


@router.post("/xyz/{jid}/cancel", response_model=xyz.XyzTask)
def cancel(jid: str, context=Depends(ctx)):
    row = xyz.task(context, jid)
    if row["can_cancel"]:
        context.supervisor.request(jid, "cancel")
    return xyz.task(context, jid)


@router.get(
    "/xyz/{jid}/file",
    response_class=FileResponse,
    responses={200: {"content": {"image/png": {"schema": {"type": "string", "format": "binary"}}}}},
)
def file(jid: str, name: str, context=Depends(ctx)):
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ApiError("Invalid result filename", code="xyz.path", status=403)
    task = xyz.task(context, jid)
    published = {entry["file"] for entry in task["manifest"]["cells"] + task["manifest"]["grids"]}
    if name not in published:
        raise NotFound("模型测试图片不存在。", code="xyz.file")
    root = xyz.result_root(context, xyz._row(context, jid))
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root) or not context.is_allowed(path.resolve()):
        raise ApiError("模型测试图片不在结果目录内。", code="xyz.path", status=403)
    if not path.is_file():
        raise NotFound("模型测试图片不存在。", code="xyz.file")
    return FileResponse(
        path,
        media_type="image/png",
        filename=f"{task['id']}-{name}",
        content_disposition_type="inline",
    )
