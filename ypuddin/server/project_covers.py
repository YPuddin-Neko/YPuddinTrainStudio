"""Bounded project-cover uploads, isolated from every training data source."""

from __future__ import annotations

import io
import logging
import re
import tempfile
import uuid
import warnings
from pathlib import Path
from typing import Any

from fastapi import Request
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from .errors import ApiError

log = logging.getLogger(__name__)
MAX_BYTES = 8 * 1024**2
MAX_REQUEST_BYTES = MAX_BYTES + 64 * 1024
MAX_PIXELS = 16_000_000
MAX_SIDE = 8192
COVER_SIDE = 640
_KEY = re.compile(r"^[a-f0-9]{32}\.webp$")


class _TooLarge(MultiPartException):
    pass


class _CoverParser(MultiPartParser):
    spool_max_size = 64 * 1024
    max_file_size = spool_max_size

    def on_part_begin(self) -> None:
        super().on_part_begin()
        self.part_bytes = 0

    def on_headers_finished(self) -> None:
        super().on_headers_finished()
        if self._current_part.field_name != "file" or self._current_part.file is None:
            raise MultiPartException("use exactly one uploaded file in the file field")

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        self.part_bytes += end - start
        if self.part_bytes > MAX_BYTES:
            raise _TooLarge("project cover exceeds the 8 MiB file limit")
        super().on_part_data(data, start, end)


async def read_cover_upload(request: Request) -> bytes:
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
        raise ApiError("cover upload requires multipart/form-data", code="project.cover_type", status=415)
    length = request.headers.get("content-length")
    if length:
        try:
            declared = int(length)
        except ValueError as exc:
            raise ApiError("invalid Content-Length", code="project.cover_invalid") from exc
        if declared < 0 or declared > MAX_REQUEST_BYTES:
            raise ApiError("cover upload exceeds the 8 MiB limit", code="project.cover_size", status=413)

    async def bounded_stream():
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > MAX_REQUEST_BYTES:
                raise _TooLarge("cover upload exceeds the request limit")
            yield chunk

    parser = _CoverParser(request.headers, bounded_stream(), max_files=1, max_fields=0)
    form = None
    try:
        try:
            form = await parser.parse()
        except MultiPartException as exc:
            raise ApiError(
                str(exc),
                code="project.cover_size" if isinstance(exc, _TooLarge) else "project.cover_invalid",
                status=413 if isinstance(exc, _TooLarge) else 400,
            ) from exc
        except Exception as exc:
            raise ApiError("invalid or interrupted cover upload", code="project.cover_invalid") from exc
        files = form.getlist("file")
        if len(form.multi_items()) != 1 or len(files) != 1 or not isinstance(files[0], UploadFile):
            raise ApiError("upload exactly one file", code="project.cover_invalid")
        data = await files[0].read(MAX_BYTES + 1)
        if not data or len(data) > MAX_BYTES:
            raise ApiError("cover must be nonempty and at most 8 MiB", code="project.cover_size", status=413)
        return data
    finally:
        if form is not None:
            await form.close()
        else:
            for handle in parser._files_to_close_on_error:
                handle.close()


def thumbnail(data: bytes) -> bytes:
    """Decode an allowed raster format and encode a static WebP without input metadata."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in {"JPEG", "PNG", "WEBP"} or getattr(source, "n_frames", 1) != 1:
                    raise ApiError(
                        "cover must be a static JPEG, PNG or WebP image",
                        code="project.cover_type",
                        status=415,
                    )
                if max(source.size) > MAX_SIDE or source.width * source.height > MAX_PIXELS:
                    raise ApiError(
                        "cover exceeds 16 megapixels or 8192 pixels per side",
                        code="project.cover_dimensions",
                        status=413,
                    )
                source.verify()
            with Image.open(io.BytesIO(data)) as source:
                image = ImageOps.exif_transpose(source).convert("RGBA")
                image.thumbnail((COVER_SIDE, COVER_SIDE), Image.Resampling.LANCZOS)
                output = io.BytesIO()
                image.save(output, "WEBP", quality=88, method=4)
                return output.getvalue()
    except ApiError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ApiError(
            "cover image is invalid or exceeds image safety limits", code="project.cover_invalid"
        ) from exc


def _directory(c: Any, pid: str) -> Path:
    root = c.project_dir(pid)
    directory = root / ".studio"
    if any(path.is_symlink() for path in (root.parent, root, directory)):
        raise ApiError(
            "project cover directory cannot use symbolic links", code="project.cover_path", status=409
        )
    if root.resolve().parent not in {
        (c.data_root / "project").resolve(),
        (c.data_root / "projects").resolve(),
    }:
        raise ApiError("invalid project cover directory", code="project.cover_path", status=409)
    return directory


def cover_path(c: Any, row: dict) -> Path | None:
    key = row.get("cover_key")
    if not isinstance(key, str) or not _KEY.fullmatch(key):
        return None
    path = _directory(c, row["id"]) / key
    return path if path.is_file() and not path.is_symlink() else None


def cover_url(c: Any, row: dict) -> str | None:
    try:
        path = cover_path(c, row)
    except ApiError:
        return None
    return f"/api/projects/{row['id']}/cover?v={row['cover_key']}" if path else None


def _cleanup(path: Path | None) -> None:
    if path is not None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            log.warning("Could not remove an obsolete project-cover file")


def replace_cover(c: Any, row: dict, data: bytes) -> None:
    """Caller holds DB lock; a failed write/DB update keeps the previous cover selected."""
    from .db import now

    old = cover_path(c, row)
    directory = _directory(c, row["id"])
    directory.mkdir(parents=True, exist_ok=True)
    key = uuid.uuid4().hex + ".webp"
    target = directory / key
    if target.exists() or target.is_symlink():
        raise ApiError(
            "cover destination already exists; retry upload", code="project.cover_conflict", status=409
        )
    temporary = None
    published = False
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        temporary.replace(target)
        published = True
        c.db.update("projects", row["id"], {"cover_key": key, "updated_at": now()})
    except BaseException:
        if published:
            _cleanup(target)
        raise
    finally:
        _cleanup(temporary)
    _cleanup(old)


def remove_cover(c: Any, row: dict) -> None:
    from .db import now

    old = cover_path(c, row)
    c.db.update("projects", row["id"], {"cover_key": None, "updated_at": now()})
    _cleanup(old)
