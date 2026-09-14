"""Environment status, reviewed dependency operations, and compatible local wheel uploads."""

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, Request, UploadFile

from .db import new_id
from .environment import (
    MAX_WHEEL_BYTES,
    EnvironmentError,
    EnvironmentManager,
    EnvironmentOperation,
    EnvironmentRequest,
    EnvironmentSettings,
    EnvironmentSnapshot,
    EnvironmentWheel,
)
from .lifecycle import RestartRequest, RestartResult, ServiceRuntime
from .torch_environments import TorchOperation, TorchRequest, TorchSnapshot

router = APIRouter()


def environment(request: Request) -> EnvironmentManager:
    return request.app.state.environment


@router.get("/environment", response_model=EnvironmentSnapshot)
def status(refresh: bool = False, service: EnvironmentManager = Depends(environment)):
    return service.status(refresh=refresh)


@router.put("/environment/settings", response_model=EnvironmentSettings)
def settings(body: EnvironmentSettings, service: EnvironmentManager = Depends(environment)):
    return service.save_settings(body)


@router.get("/environment/operations", response_model=list[EnvironmentOperation])
def operations(service: EnvironmentManager = Depends(environment)):
    return service.list()


@router.post("/environment/operations", response_model=EnvironmentOperation, status_code=202)
def start(body: EnvironmentRequest, service: EnvironmentManager = Depends(environment)):
    return service.start(body)


@router.get("/environment/operations/{id_}", response_model=EnvironmentOperation)
def operation(id_: str, service: EnvironmentManager = Depends(environment)):
    return service.get(id_)


@router.post("/environment/operations/{id_}/apply", response_model=EnvironmentOperation, status_code=202)
def apply(id_: str, service: EnvironmentManager = Depends(environment)):
    return service.apply(id_)


@router.post("/environment/operations/{id_}/cancel", response_model=EnvironmentOperation)
def cancel(id_: str, service: EnvironmentManager = Depends(environment)):
    return service.cancel(id_)


@router.post("/environment/operations/{id_}/dismiss", response_model=EnvironmentOperation)
def dismiss(id_: str, service: EnvironmentManager = Depends(environment)):
    return service.dismiss(id_)


@router.post("/environment/wheels", response_model=EnvironmentWheel, status_code=201)
def wheel(file: UploadFile, service: EnvironmentManager = Depends(environment)):
    filename = file.filename or ""
    if (
        Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
        or not filename.endswith(".whl")
    ):
        raise EnvironmentError(422, "Choose a .whl file with a valid wheel filename")
    folder = service.root / "uploads" / new_id("upload")
    folder.mkdir(parents=True)
    target = folder / filename
    try:
        size = 0
        with target.open("xb") as stream:
            while chunk := file.file.read(1024**2):
                size += len(chunk)
                if size > MAX_WHEEL_BYTES:
                    raise EnvironmentError(413, "Wheel exceeds the 2 GiB upload limit")
                stream.write(chunk)
        return service.register_wheel(target)
    except EnvironmentError:
        shutil.rmtree(folder)
        raise
    except Exception as exc:
        shutil.rmtree(folder)
        raise EnvironmentError(422, str(exc)) from exc
    finally:
        file.file.close()


@router.get("/environment/torch", response_model=TorchSnapshot)
def torch_status(request: Request):
    return request.app.state.torch_environments.status()


@router.post("/environment/torch/operations", response_model=TorchOperation, status_code=201)
def torch_plan(body: TorchRequest, request: Request):
    return request.app.state.torch_environments.start(body)


@router.get("/environment/torch/operations/{id_}", response_model=TorchOperation)
def torch_operation(id_: str, request: Request):
    return request.app.state.torch_environments.get(id_)


@router.post("/environment/torch/operations/{id_}/apply", response_model=TorchOperation, status_code=202)
def torch_apply(id_: str, request: Request):
    return request.app.state.torch_environments.apply(id_)


@router.post("/environment/torch/operations/{id_}/cancel", response_model=TorchOperation)
def torch_cancel(id_: str, request: Request):
    return request.app.state.torch_environments.cancel(id_)


@router.post("/environment/torch/operations/{id_}/dismiss", response_model=TorchOperation)
def torch_dismiss(id_: str, request: Request):
    return request.app.state.torch_environments.dismiss(id_)


@router.get("/service/runtime", response_model=ServiceRuntime)
def service_runtime(request: Request):
    return request.app.state.lifecycle.status()


@router.post("/service/restart", response_model=RestartResult, status_code=202)
def service_restart(body: RestartRequest, request: Request):
    return request.app.state.lifecycle.restart(body)
