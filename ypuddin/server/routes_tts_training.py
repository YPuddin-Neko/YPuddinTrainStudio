"""Speech capabilities and validation/launch of an explicitly saved project version."""

from typing import Literal

from fastapi import APIRouter, Depends, Header, Response

from ypuddin.tts.capabilities import TtsCapabilities, TtsTrainSchema, get_capabilities, get_train_schema
from ypuddin.tts.validation_models import TtsValidationBody, TtsValidationReport

from . import models as m
from .context import ServiceContext
from .routes_tts_projects import ERRORS, TtsProjectRoute, ctx
from .tts_training import TtsTrainingBody, create_training
from .tts_validation import validate_version

router = APIRouter(route_class=TtsProjectRoute)


@router.get("/tts/capabilities", response_model=TtsCapabilities, responses=ERRORS)
def capabilities():
    return get_capabilities()


@router.get("/tts/schema/train", response_model=TtsTrainSchema, responses=ERRORS)
def training_schema(engine: Literal["voxcpm1.5"] = "voxcpm1.5"):
    return get_train_schema(engine)


@router.post(
    "/tts/projects/{pid}/versions/{vid}/validate", response_model=TtsValidationReport, responses=ERRORS
)
def validate(pid: str, vid: str, body: TtsValidationBody, c: ServiceContext = Depends(ctx)):
    return validate_version(c, pid, vid, body.revision, body.data_revision, gpu_devices=body.gpu_devices)


@router.post(
    "/tts/projects/{pid}/versions/{vid}/jobs",
    status_code=201,
    response_model=m.Job,
    response_model_exclude_unset=True,
    responses={**ERRORS, 200: {"model": m.Job}},
)
def launch(
    pid: str,
    vid: str,
    body: TtsTrainingBody,
    response: Response,
    idempotency_key: str = Header(alias="Idempotency-Key"),
    c: ServiceContext = Depends(ctx),
):
    job, response.status_code = create_training(c, pid, vid, body, idempotency_key)
    return job
