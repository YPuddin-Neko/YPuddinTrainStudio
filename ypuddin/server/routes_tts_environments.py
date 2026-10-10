"""Local environment inventory and explicit speech preparation operations."""

from fastapi import APIRouter, Depends, Request

from ypuddin.tts.environment_models import (
    Engine,
    TtsEnvironmentCheckRequest,
    TtsEnvironmentDefaultRequest,
    TtsEnvironmentOperation,
    TtsEnvironmentPrepareRequest,
    TtsEnvironmentSnapshot,
)

from .models import ApiErrorResponse
from .tts_environments import TtsEnvironments

router = APIRouter(responses={code: {"model": ApiErrorResponse} for code in (400, 403, 404, 409, 422)})


def environments(request: Request) -> TtsEnvironments:
    return request.app.state.tts_environments


@router.get("/tts/environments", response_model=TtsEnvironmentSnapshot)
def list_environments(service: TtsEnvironments = Depends(environments)):
    return service.snapshot()


@router.post("/tts/environments/checks", response_model=TtsEnvironmentOperation, status_code=202)
def check_environment(body: TtsEnvironmentCheckRequest, service: TtsEnvironments = Depends(environments)):
    return service.start("check", body.engine, body.candidate_id)


@router.post("/tts/environments/preparations", response_model=TtsEnvironmentOperation, status_code=202)
def prepare_environment(body: TtsEnvironmentPrepareRequest, service: TtsEnvironments = Depends(environments)):
    return service.start("prepare", body.engine, body.candidate_id, make_default=body.make_default)


@router.get("/tts/environments/operations/{operation_id}", response_model=TtsEnvironmentOperation)
def get_operation(operation_id: str, service: TtsEnvironments = Depends(environments)):
    return service.get(operation_id)


@router.post("/tts/environments/operations/{operation_id}/cancel", response_model=TtsEnvironmentOperation)
def cancel_operation(operation_id: str, service: TtsEnvironments = Depends(environments)):
    return service.cancel(operation_id)


@router.post("/tts/environments/operations/{operation_id}/retry", response_model=TtsEnvironmentOperation, status_code=202)
def retry_operation(operation_id: str, service: TtsEnvironments = Depends(environments)):
    return service.retry(operation_id)


@router.put("/tts/environments/defaults/{engine}", response_model=TtsEnvironmentSnapshot)
def select_default(engine: Engine, body: TtsEnvironmentDefaultRequest,
                   service: TtsEnvironments = Depends(environments)):
    return service.set_default(engine, body.environment_id)
