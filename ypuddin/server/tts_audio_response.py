"""Serve a checked recording through one open file handle, including byte ranges."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path
from urllib.parse import quote

from fastapi import Request
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, StreamingResponse

from ypuddin.tts.source_scan import SourceFileError, open_source

from .errors import ApiError, envelope


def audio_response(
    request: Request, asset: dict, *, media_type: str = "audio/wav", attachment: bool = False
) -> StreamingResponse | JSONResponse:
    path = asset["path"]
    handle = None
    try:
        with open_source(Path(path), request.app.state.ctx.is_allowed) as checked:
            fd = os.dup(checked.fileno())
            handle = os.fdopen(fd, "rb")
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size != asset["size"]:
            raise ApiError("录音已改变，请重新检查数据。", code="tts.source_stale", status=409)
        digest = hashlib.sha256()
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
        if digest.hexdigest() != asset["sha256"]:
            raise ApiError("录音已改变，请重新检查数据。", code="tts.source_stale", status=409)
        size = info.st_size
        start, end, status = 0, size - 1, 200
        etag = '"' + asset["sha256"] + '"'
        headers = {"Accept-Ranges": "bytes", "ETag": etag, "Cache-Control": "no-store"}

        def invalid_range():
            message = "请求的录音范围无效。"
            return JSONResponse(
                envelope(
                    "tts.audio_range",
                    message,
                    {
                        "issues": [
                            {
                                "code": "tts.audio_range",
                                "loc": ["header", "range"],
                                "message": message,
                                "severity": "error",
                                "details": {},
                            }
                        ]
                    },
                ),
                status_code=416,
                headers={**headers, "Content-Range": f"bytes */{size}"},
            )

        requested = request.headers.get("range")
        if requested and request.headers.get("if-range", etag) == etag:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip())
            if match is None or not any(match.groups()):
                return invalid_range()
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            else:
                start = max(0, size - int(last))
            if start > end or start >= size:
                return invalid_range()
            status = 206
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        headers["Content-Disposition"] = ("attachment" if attachment else "inline") + "; filename*=UTF-8''" + quote(Path(path).name, safe="")
        headers["Content-Length"] = str(end - start + 1)
        handle.seek(start)
        stream = handle

        def chunks():
            remaining = end - start + 1
            try:
                while remaining:
                    chunk = stream.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk
            finally:
                stream.close()

        response = StreamingResponse(
            chunks(),
            status_code=status,
            media_type=media_type,
            headers=headers,
            background=BackgroundTask(stream.close),
        )
        handle = None
        return response
    except FileNotFoundError as exc:
        raise ApiError("录音文件不存在。", code="tts.audio_not_found", status=404) from exc
    except SourceFileError as exc:
        raise ApiError(str(exc), code=exc.code, status=403 if exc.code == "tts.path_denied" else 409) from exc
    except OSError as exc:
        raise ApiError("无法安全读取录音。", code="tts.path_denied", status=403) from exc
    finally:
        if handle is not None:
            handle.close()
