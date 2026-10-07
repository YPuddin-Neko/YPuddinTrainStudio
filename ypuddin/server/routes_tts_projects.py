"""Typed configuration endpoints for a speech project version."""

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict

from ypuddin.tts.issues import TtsIssue
from ypuddin.tts.version_config import TtsConfigResponse, TtsConfigSaveBody

from .context import ServiceContext
from .errors import ApiError
from .tts_projects import get_config, save_config


class TtsErrorDetails(BaseModel):
    model_config = ConfigDict(extra="allow")
    issues: list[TtsIssue]
    current_revision: int | None = None
    current_data_revision: int | None = None


class TtsError(BaseModel):
    code: str
    message: str
    trace_id: str
    details: TtsErrorDetails


class TtsErrorResponse(BaseModel):
    error: TtsError


class TtsProjectRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def scoped_handler(request: Request):
            try:
                return await handler(request)
            except RequestValidationError as exc:
                issues = [
                    TtsIssue(
                        code="tts.validation." + item["type"],
                        loc=list(item["loc"])[1:]
                        if item["loc"] and item["loc"][0] == "body"
                        else list(item["loc"]),
                        message=item["msg"],
                    ).model_dump()
                    for item in exc.errors()
                ]
                raise ApiError(
                    "请检查语音配置字段。", code="tts.invalid", status=422, details={"issues": issues}
                ) from exc
            except ApiError as exc:
                loc = {
                    "tts.config_conflict": ["expected_revision"],
                    "tts.data_conflict": ["expected_data_revision"],
                }.get(exc.code, [])
                details = dict(exc.details)
                details.setdefault(
                    "issues", [TtsIssue(code=exc.code, loc=loc, message=exc.message).model_dump()]
                )
                raise ApiError(exc.message, code=exc.code, status=exc.status, details=details) from exc

        return scoped_handler


router = APIRouter(route_class=TtsProjectRoute)
ERRORS = {code: {"model": TtsErrorResponse} for code in (400, 403, 404, 409, 410, 416, 422, 500)}


def ctx(request: Request) -> ServiceContext:
    return request.app.state.ctx


@router.get("/tts/projects/{pid}/versions/{vid}/config", response_model=TtsConfigResponse, responses=ERRORS)
def read_version_config(pid: str, vid: str, c: ServiceContext = Depends(ctx)) -> TtsConfigResponse:
    return get_config(c, pid, vid)


@router.put("/tts/projects/{pid}/versions/{vid}/config", response_model=TtsConfigResponse, responses=ERRORS)
def save_version_config(
    pid: str, vid: str, body: TtsConfigSaveBody, c: ServiceContext = Depends(ctx)
) -> TtsConfigResponse:
    return save_config(c, pid, vid, body)
