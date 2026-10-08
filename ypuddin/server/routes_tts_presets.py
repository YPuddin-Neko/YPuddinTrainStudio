"""Portable speech preset management and non-persistent version previews."""

from fastapi import APIRouter, Depends, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from ypuddin.tts.issues import TtsIssue
from ypuddin.tts.preset_models import (
    TtsPreset,
    TtsPresetConfig,
    TtsPresetCreateBody,
    TtsPresetDeleted,
    TtsPresetDocument,
    TtsPresetResolveBody,
    TtsPresetResolveResponse,
    TtsPresetUpdateBody,
)
from ypuddin.tts.version_config import TtsEngine

from . import tts_presets
from .context import ServiceContext
from .errors import ApiError
from .routes_tts_projects import ERRORS


class TtsPresetRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def preset_handler(request: Request):
            try:
                return await handler(request)
            except RequestValidationError as exc:
                raise ApiError(
                    "请检查预设字段。", code="tts.preset_invalid", status=422,
                    details={"issues": tts_presets.validation_issues(exc.errors())},
                ) from exc
            except ApiError as exc:
                location = {
                    "tts.preset_conflict": ["expected_revision"],
                    "tts.preset_duplicate": ["name"],
                    "tts.preset_engine_mismatch": ["config", "engine"],
                }.get(exc.code, [])
                details = dict(exc.details)
                details.setdefault("issues", [TtsIssue(
                    code=exc.code, loc=location, message=exc.message,
                ).model_dump()])
                raise ApiError(exc.message, code=exc.code, status=exc.status, details=details) from exc

        return preset_handler


router = APIRouter(route_class=TtsPresetRoute)


def ctx(request: Request) -> ServiceContext:
    return request.app.state.ctx


@router.get("/tts/presets", response_model=list[TtsPreset], responses=ERRORS)
def list_presets(engine: TtsEngine | None = None, c: ServiceContext = Depends(ctx)):
    return tts_presets.list_presets(c, engine)


@router.get("/tts/presets/defaults", response_model=TtsPresetConfig, responses=ERRORS)
def defaults(engine: TtsEngine, c: ServiceContext = Depends(ctx)):
    return tts_presets.defaults(engine)


@router.post("/tts/presets", response_model=TtsPreset, status_code=201, responses=ERRORS)
def create_preset(body: TtsPresetCreateBody, c: ServiceContext = Depends(ctx)):
    return tts_presets.create_preset(c, body)


@router.post("/tts/presets/import", response_model=TtsPreset, status_code=201, responses=ERRORS)
def import_preset(body: TtsPresetDocument, c: ServiceContext = Depends(ctx)):
    return tts_presets.import_preset(c, body)


@router.get("/tts/presets/{preset_id}", response_model=TtsPreset, responses=ERRORS)
def get_preset(preset_id: str, c: ServiceContext = Depends(ctx)):
    return tts_presets.get_preset(c, preset_id)


@router.put("/tts/presets/{preset_id}", response_model=TtsPreset, responses=ERRORS)
def update_preset(preset_id: str, body: TtsPresetUpdateBody, c: ServiceContext = Depends(ctx)):
    return tts_presets.update_preset(c, preset_id, body)


@router.delete("/tts/presets/{preset_id}", response_model=TtsPresetDeleted, responses=ERRORS)
def delete_preset(preset_id: str, expected_revision: int = Query(ge=1), c: ServiceContext = Depends(ctx)):
    tts_presets.delete_preset(c, preset_id, expected_revision)
    return TtsPresetDeleted()


@router.get("/tts/presets/{preset_id}/export", response_model=TtsPresetDocument, responses=ERRORS)
def export_preset(preset_id: str, c: ServiceContext = Depends(ctx)):
    return tts_presets.export_preset(c, preset_id)


@router.post("/tts/presets/{preset_id}/resolve", response_model=TtsPresetResolveResponse, responses=ERRORS)
def resolve_preset(preset_id: str, body: TtsPresetResolveBody, c: ServiceContext = Depends(ctx)):
    return tts_presets.resolve_preset(c, preset_id, body)
