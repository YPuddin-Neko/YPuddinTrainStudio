"""Error envelope + trace ids."""

from __future__ import annotations

import contextvars
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ypuddin.config.issues import plain_context

trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")


class ApiError(Exception):
    status = 400
    code = "api.error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if status:
            self.status = status
        self.details = details or {}


class NotFound(ApiError):
    status = 404
    code = "not_found"


class Conflict(ApiError):
    status = 409
    code = "conflict"


def envelope(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "error": {"code": code, "message": message, "trace_id": trace_id_var.get(), "details": details or {}}
    }


class TraceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        tid = request.headers.get("X-Trace-Id") or uuid.uuid4().hex[:16]
        token = trace_id_var.set(tid)
        try:
            response = await call_next(request)
        finally:
            trace_id_var.reset(token)
        response.headers["X-Trace-Id"] = tid
        return response


def install(app: FastAPI) -> None:
    app.add_middleware(TraceMiddleware)

    @app.exception_handler(ApiError)
    async def _api_error(_req: Request, exc: ApiError):
        return JSONResponse(status_code=exc.status, content=envelope(exc.code, exc.message, exc.details))

    @app.exception_handler(RequestValidationError)
    async def _validation(_req: Request, exc: RequestValidationError):
        # The submitted value is never echoed: it may be a pasted key or a private source URL.
        sensitive = _req.url.path.startswith(("/api/models", "/api/credentials"))
        errors = []
        for error in exc.errors():
            issue = {
                "loc": list(error.get("loc", [])),
                "msg": error.get("msg", ""),
                "type": error.get("type"),
            }
            if not sensitive and (ctx := plain_context(error.get("ctx"))):
                issue["ctx"] = ctx
            errors.append(issue)
        if _req.url.path.startswith("/api/credentials"):
            # Even an unexpected JSON property name can be a mistakenly pasted key.
            fields = {"body", "path", "query", "provider", "token", "username", "user_id", "api_key"}
            for error in errors:
                error["loc"] = [
                    part if isinstance(part, int) or part in fields else "[field]"
                    for part in error.get("loc", [])
                ]
        return JSONResponse(
            status_code=422,
            content=envelope(
                "validation",
                "request validation failed",
                {"errors": jsonable_encoder(errors)},
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled(_req: Request, exc: Exception):
        return JSONResponse(status_code=500, content=envelope("internal", f"{type(exc).__name__}: {exc}"))
