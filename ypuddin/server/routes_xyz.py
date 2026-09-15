"""Saved-training-checkpoint XYZ comparisons."""

from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from . import xyz
from .errors import ApiError, NotFound
from .routes_core import ctx

router = APIRouter()


@router.get("/jobs/{source}/xyz/options", response_model=xyz.XyzOptions)
def options(source: str, context=Depends(ctx)):
    return xyz.options(context, source)


@router.post("/jobs/{source}/xyz", response_model=xyz.XyzTask, status_code=202)
def create(source: str, body: xyz.XyzRequest, context=Depends(ctx)):
    return xyz.start(context, source, body)


@router.get("/jobs/{source}/xyz", response_model=list[xyz.XyzTask])
def history(source: str, context=Depends(ctx)):
    return xyz.history(context, source)


@router.get("/xyz/{jid}", response_model=xyz.XyzTask)
def task(jid: str, context=Depends(ctx)):
    return xyz.task(context, jid)


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
