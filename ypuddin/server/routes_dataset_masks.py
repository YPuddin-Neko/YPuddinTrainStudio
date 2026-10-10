"""Image-bound, bounded PNG mask editing. White trains; black is ignored."""

from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from ypuddin.data.image_metadata import alpha_channel

from .context import ServiceContext
from .errors import ApiError, NotFound
from .routes_work import _get_dataset, _records, _records_path, ctx

router = APIRouter()
MAX_REQUEST_BYTES = 32 * 1024**2
MAX_PIXELS = 16_777_216
MAX_EDGE = 8192


class MaskInfo(BaseModel):
    source: Literal["sidecar", "alpha", "full"]
    width: int
    height: int
    coverage: float
    has_mask: bool
    filename: str
    revision: str
    resized: bool = False


def _resolve(c: ServiceContext, did: str, h: str, rel_path: str | None = None):
    row = _get_dataset(c, did)
    root = Path(row["path"]).expanduser().resolve()
    candidates = [r for r in _records(c, did) if r["content_hash"] == h]
    if any(
        not Path(r["path"]).resolve().is_relative_to(root) or not c.is_allowed(Path(r["path"]))
        for r in candidates
    ):
        raise ApiError("image is outside the allowed dataset storage", code="mask.path", status=403)
    if rel_path is not None:
        # Windows lists native separators; painting returns POSIX paths. On POSIX,
        # backslashes remain literal filename characters.
        rel_path = rel_path.replace(os.sep, "/")
        candidates = [r for r in candidates if Path(r["path"]).relative_to(root).as_posix() == rel_path]
    if not candidates:
        raise NotFound("image not found in this dataset", code="image.not_found")
    if len({r["path"] for r in candidates}) != 1:
        raise ApiError(
            "image hash matches multiple files; provide rel_path", code="mask.ambiguous", status=409
        )
    image = Path(candidates[0]["path"]).resolve()
    if not image.is_relative_to(root) or not c.is_allowed(image):
        raise ApiError("image is outside the allowed dataset storage", code="mask.path", status=403)
    if not image.is_file():
        raise NotFound("source image no longer exists; rescan the dataset", code="image.not_found")
    target = image.with_name(image.stem + ".mask.png")
    # Do not follow a sidecar symlink, including one pointing to another allowed dataset.
    for path in (target, image.with_name(image.stem + ".mask")):
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not c.is_allowed(path):
            raise ApiError(
                "mask sidecar is outside the dataset or is a symbolic link", code="mask.path", status=403
            )
    return row, image, target


def _check_size(size: tuple[int, int]) -> None:
    if min(size) < 1 or max(size) > MAX_EDGE or size[0] * size[1] > MAX_PIXELS:
        raise ApiError(
            "mask editing supports up to 16 megapixels and 8192 pixels per side",
            code="mask.dimensions",
            status=413,
        )


def _load(image: Path, target: Path) -> tuple[Image.Image, MaskInfo]:
    try:
        with Image.open(image) as original:
            _check_size(original.size)
            oriented = ImageOps.exif_transpose(original)
            size = oriented.size
            legacy = image.with_name(image.stem + ".mask")
            sidecar = target if target.is_file() else legacy if legacy.is_file() else None
            resized = False
            if sidecar:
                with Image.open(sidecar) as saved:
                    _check_size(saved.size)
                    gray = ImageOps.exif_transpose(saved).convert("L")
                resized = gray.size != size
                if resized:
                    gray = ImageOps.fit(gray, size, method=Image.Resampling.BILINEAR)
                source = "sidecar"
                version_path = sidecar
            elif (alpha := alpha_channel(oriented)) is not None:
                gray = alpha
                source = "alpha"
                version_path = image
            else:
                gray = Image.new("L", size, 255)
                source = "full"
                version_path = image
        stat = version_path.stat()
        image_stat = image.stat()
        histogram = gray.histogram()
        coverage = sum(value * count for value, count in enumerate(histogram)) / (255 * size[0] * size[1])
        return gray, MaskInfo(
            source=source,
            width=size[0],
            height=size[1],
            coverage=coverage,
            has_mask=sidecar is not None,
            filename=target.name,
            revision=f"{source}:{stat.st_mtime_ns}:{stat.st_size}:{image_stat.st_mtime_ns}:{image_stat.st_size}",
            resized=resized,
        )
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ApiError(
            "cannot read source image or mask; check the files and rescan", code="mask.image"
        ) from exc


@router.get("/datasets/{did}/images/{h}/mask/info", response_model=MaskInfo)
def mask_info(did: str, h: str, rel_path: str | None = None, c: ServiceContext = Depends(ctx)) -> MaskInfo:
    _, image, target = _resolve(c, did, h, rel_path)
    return _load(image, target)[1]


@router.get(
    "/datasets/{did}/images/{h}/mask",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}, "description": "Current loss weights as grayscale PNG"}},
)
def mask_file(did: str, h: str, rel_path: str | None = None, c: ServiceContext = Depends(ctx)) -> Response:
    _, image, target = _resolve(c, did, h, rel_path)
    mask, _ = _load(image, target)
    output = io.BytesIO()
    mask.save(output, format="PNG")
    return Response(output.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})


@router.get(
    "/datasets/{did}/images/{h}/mask/source",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}, "description": "Source image in training orientation"}},
)
def mask_source(did: str, h: str, rel_path: str | None = None, c: ServiceContext = Depends(ctx)) -> Response:
    _, image, _ = _resolve(c, did, h, rel_path)
    try:
        with Image.open(image) as original:
            _check_size(original.size)
            oriented = ImageOps.exif_transpose(original).convert("RGBA")
            background = Image.new("RGBA", oriented.size, (255, 255, 255, 255))
            rgb = Image.alpha_composite(background, oriented).convert("RGB")
        output = io.BytesIO()
        rgb.save(output, format="PNG")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ApiError("cannot read source image", code="mask.image") from exc
    return Response(output.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})


def _save(c: ServiceContext, did: str, h: str, rel_path: str | None, data: bytes, revision: str) -> MaskInfo:
    try:
        with Image.open(io.BytesIO(data)) as uploaded:
            if uploaded.format != "PNG" or getattr(uploaded, "n_frames", 1) != 1:
                raise ApiError("mask must be a single-frame PNG", code="mask.format")
            _check_size(uploaded.size)
            uploaded.load()
            gray = uploaded.convert("L")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ApiError("mask is not a valid PNG image", code="mask.format") from exc
    with c.db.lock:
        row, image, target = _resolve(c, did, h, rel_path)
        from .versions import assert_version_writable

        assert_version_writable(c, row["project_id"], row["version_id"], data=True)
        if row["index_status"] == "indexing":
            raise ApiError(
                "dataset is being indexed; retry after indexing finishes", code="mask.indexing", status=409
            )
        if c.db.fetchone(
            "SELECT id FROM jobs WHERE version_id=? AND status IN ('running','pausing','cancelling')",
            (row["version_id"],),
        ):
            raise ApiError(
                "stop the project's active job before editing training masks", code="mask.busy", status=409
            )
        _, previous = _load(image, target)
        if revision != previous.revision:
            raise ApiError(
                "mask changed since it was opened; reload before saving", code="mask.conflict", status=409
            )
        if gray.size != (previous.width, previous.height):
            raise ApiError(f"mask dimensions must be {previous.width}×{previous.height}", code="mask.size")
        from .dataset_refresh import image_entry, note_own_edit

        records = _records(c, did)
        # The image as the index lists it; a newly created mask changes its part of the folder signature.
        listed = next((Path(record["path"]) for record in records if Path(record["path"]).resolve() == image), None)
        before = image_entry(row, listed) if listed else None
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                gray.save(stream, format="PNG")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        # Masks are read fresh by Dataset.current_mask, including when latents are cached.
        for record in records:
            if Path(record["path"]).resolve() == image:
                record["mask_path"] = str(target)
        records_path = _records_path(c, did)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=records_path.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            try:
                json.dump(records, stream)
                stream.close()
                temporary.replace(records_path)
            finally:
                temporary.unlink(missing_ok=True)
        stats = json.loads(row["stats_json"] or "{}")
        stats["masks"] = sum(bool(record["mask_path"]) for record in records)
        c.db.update("datasets", did, {"stats_json": json.dumps(stats)})
        # The records are current again, so the next check of the folder finds nothing to re-index.
        if listed:
            note_own_edit(c, did, [before], [image_entry(row, listed)])
    c.bus.publish("dataset.changed", {
        "dataset_id": did, "project_id": row["project_id"], "version_id": row["version_id"], "reason": "mask",
    })
    return _load(image, target)[1]


@router.put(
    "/datasets/{did}/images/{h}/mask",
    response_model=MaskInfo,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["file", "revision"],
                        "properties": {
                            "file": {"type": "string", "format": "binary"},
                            "revision": {"type": "string"},
                        },
                    }
                }
            },
        }
    },
)
async def save_mask(
    did: str, h: str, request: Request, rel_path: str | None = None, c: ServiceContext = Depends(ctx)
) -> MaskInfo:
    _resolve(c, did, h, rel_path)
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
        raise ApiError(
            "expected multipart/form-data with file and revision", code="mask.content_type", status=415
        )

    async def bounded_stream():
        count = 0
        async for chunk in request.stream():
            count += len(chunk)
            if count > MAX_REQUEST_BYTES:
                raise MultiPartException("mask upload exceeds 32 MiB")
            yield chunk

    parser = MultiPartParser(request.headers, bounded_stream(), max_files=1, max_fields=1)
    form = None
    try:
        try:
            form = await parser.parse()
        except MultiPartException as exc:
            raise ApiError(str(exc), code="mask.upload", status=413) from exc
        except Exception as exc:
            raise ApiError("invalid or interrupted multipart upload", code="mask.upload") from exc
        upload, revision = form.get("file"), form.get("revision")
        if (
            set(form) != {"file", "revision"}
            or not isinstance(upload, UploadFile)
            or not isinstance(revision, str)
        ):
            raise ApiError("provide one file and its revision", code="mask.upload")
        data = await upload.read(MAX_REQUEST_BYTES + 1)
        if len(data) > MAX_REQUEST_BYTES:
            raise ApiError("mask upload exceeds 32 MiB", code="mask.upload", status=413)
        return await run_in_threadpool(_save, c, did, h, rel_path, data, revision)
    finally:
        if form is not None:
            await form.close()
        else:
            for stream in parser._files_to_close_on_error:
                stream.close()
