"""The task center's list of background work."""

import time

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from . import models as m
from .background_tasks import BackgroundTask, BackgroundTasks
from .errors import ApiError, NotFound

router = APIRouter()


class BackgroundTaskList(BaseModel):
    tasks: list[BackgroundTask]
    # The service clock, so pages measure ages of started_at/finished_at without clock skew.
    now: float


def registry(request: Request) -> BackgroundTasks:
    return request.app.state.background_tasks


@router.get("/background-tasks", response_model=BackgroundTaskList)
def tasks(service: BackgroundTasks = Depends(registry)):
    return {"tasks": service.list(), "now": time.time()}


@router.post("/background-tasks/{tid}/cancel", response_model=BackgroundTask)
def cancel(tid: str, service: BackgroundTasks = Depends(registry)):
    task = service.get(tid)
    if task is None:
        raise NotFound("后台任务不存在。", code="background.not_found")
    if not service.cancel(tid):
        raise ApiError("这个后台任务现在不能取消。", code="background.not_cancellable", status=409)
    return service.get(tid) or task


@router.delete("/background-tasks/{tid}", response_model=m.Ok, response_model_exclude_unset=True)
def dismiss(tid: str, service: BackgroundTasks = Depends(registry)):
    task = service.get(tid)
    if task is None:
        raise NotFound("后台任务不存在。", code="background.not_found")
    if not service.dismiss(tid):
        raise ApiError("正在进行的后台任务不能移除。", code="background.running", status=409)
    return {"ok": True}
