"""Version-owned image painting with one durable image/mask transaction."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from ypuddin.data.index import IMAGE_EXTS, content_hash

from .dataset_pipeline import DatasetPipeline, _digest, _dump
from .db import new_id, now
from .errors import ApiError
from .routes_dataset_masks import _check_size, _load
from .routes_dataset_masks import _resolve as _resolve_mask
from .routes_dataset_pipeline import pipeline
from .routes_work import _index_dataset, _records

router = APIRouter()
MAX_REQUEST_BYTES = 64 * 1024**2


class PaintInfo(BaseModel):
    width: int
    height: int
    revision: str
    image_id: str
    rel_path: str
    has_mask: bool
    mask_source: Literal["sidecar", "alpha", "full"]
    can_restore: bool
    last_operation_id: str | None = None


class PaintSaved(PaintInfo):
    operation_id: str


class PaintRestore(BaseModel):
    revision: str = Field(min_length=1, max_length=256)


def _resolve(c, did: str, h: str, rel_path: str | None):
    row, image, mask = _resolve_mask(c, did, h, rel_path)
    root = Path(row["path"])
    # The mask resolver normalizes paths; retain the indexed lexical path here
    # so resolving a symlink cannot silently turn painting A into editing B.
    for record in _records(c, did):
        candidate = Path(record["path"])
        if record["content_hash"] == h and candidate.resolve() == image:
            if candidate.is_symlink() or any(
                parent.is_symlink() for parent in candidate.parents if parent.is_relative_to(root)
            ):
                raise ApiError("image links cannot be painted", code="paint.path", status=403)
    if content_hash(image) != h:
        raise ApiError(
            "source image changed; rescan the dataset before editing", code="paint.conflict", status=409
        )
    return row, image, mask


def _revision(image: Path, mask: Path) -> str:
    # Include both sidecars: creating/removing the preferred PNG can change which
    # mask is effective, even if the previously selected file did not change.
    paths = (image, mask, image.with_name(image.stem + ".mask"))
    return hashlib.sha256(_dump([_digest(p) if p.is_file() else None for p in paths]).encode()).hexdigest()


def _original(image: Path) -> tuple[Image.Image, str, dict]:
    try:
        with Image.open(image) as source:
            _check_size(source.size)
            if getattr(source, "n_frames", 1) != 1:
                raise ApiError("painting supports single-frame images only", code="paint.format")
            if source.mode not in {"RGB", "RGBA", "L", "LA", "P", "1"}:
                raise ApiError("painting supports 8-bit images only", code="paint.format")
            oriented = ImageOps.exif_transpose(source)
            return oriented.convert("RGBA"), source.format or "", dict(oriented.info)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ApiError("cannot decode source image", code="paint.image") from exc


def _last(manager: DatasetPipeline, row: dict, image: Path) -> dict | None:
    rows = manager.c.db.fetchall(
        "SELECT id,result_json FROM dataset_pipeline_operations WHERE version_id=? AND action='paint' "
        "AND status='completed' ORDER BY created_at DESC",
        (row["version_id"],),
    )
    for item in rows:
        result = json.loads(item["result_json"])
        if result.get("image_path") == str(image) and row["id"] in result.get("dataset_ids", []):
            if not result.get("undone_by"):
                return {"id": item["id"], "result": result}
    return None


def _info(manager: DatasetPipeline, row: dict, image: Path, mask: Path) -> PaintInfo:
    _, info = _load(image, mask)
    original = _last(manager, row, image)
    can_restore = (
        bool(original)
        and original["result"].get("after_revision") == _revision(image, mask)
        and all(
            (_digest(Path(change["target"])) if Path(change["target"]).exists() else None) == change["after"]
            for change in original["result"]["changes"]
        )
    )
    return PaintInfo(
        width=info.width,
        height=info.height,
        revision=_revision(image, mask),
        image_id=content_hash(image),
        rel_path=image.relative_to(Path(row["path"]).resolve()).as_posix(),
        has_mask=info.has_mask,
        mask_source=info.source,
        can_restore=can_restore,
        last_operation_id=original["id"] if original else None,
    )


@router.get("/datasets/{did}/images/{h}/paint/info", response_model=PaintInfo)
def paint_info(
    did: str, h: str, rel_path: str | None = None, manager: DatasetPipeline = Depends(pipeline)
) -> PaintInfo:
    with manager.c.db.lock:
        row, image, mask = _resolve(manager.c, did, h, rel_path)
        return _info(manager, row, image, mask)


@router.get(
    "/datasets/{did}/images/{h}/paint/source",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}, "description": "Oriented source RGBA pixels"}},
)
def paint_source(
    did: str, h: str, rel_path: str | None = None, manager: DatasetPipeline = Depends(pipeline)
) -> Response:
    with manager.c.db.lock:
        _, image, _ = _resolve(manager.c, did, h, rel_path)
        rgba, _, _ = _original(image)
        output = io.BytesIO()
        rgba.save(output, format="PNG")
    return Response(output.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})


def _decode(data: bytes | None, *, mask: bool = False) -> Image.Image | None:
    if data is None:
        return None
    try:
        with Image.open(io.BytesIO(data)) as image:
            _check_size(image.size)
            if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                raise ApiError("upload a single-frame PNG", code="paint.format")
            if mask and image.mode not in {"1", "L", "RGB", "RGBA"}:
                raise ApiError("mask must be a grayscale PNG", code="paint.format")
            if mask and image.mode in {"RGB", "RGBA"}:
                # Canvas exports RGBA even for a grayscale mask. Accept exactly
                # opaque gray pixels; do not silently turn color/alpha into loss.
                channels = image.split()
                if (
                    channels[0].tobytes() != channels[1].tobytes()
                    or channels[0].tobytes() != channels[2].tobytes()
                    or (len(channels) == 4 and channels[3].getextrema() != (255, 255))
                ):
                    raise ApiError("mask pixels must be opaque grayscale", code="paint.format")
            if not mask and image.mode not in {"RGB", "RGBA"}:
                raise ApiError("painted image must be an RGB or RGBA PNG", code="paint.format")
            image.load()
            return image.convert("L" if mask else "RGBA")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ApiError("upload is not a valid PNG", code="paint.format") from exc


def _encode(rgba: Image.Image, fmt: str, metadata: dict, staged: Path) -> None:
    if fmt not in {"PNG", "JPEG", "WEBP", "TIFF", "BMP"}:
        raise ApiError(
            "this image format cannot be safely painted; use PNG, JPEG, WebP, TIFF or BMP",
            code="paint.format",
        )
    options = {}
    if fmt in {"JPEG", "BMP"}:
        if rgba.getchannel("A").getextrema() != (255, 255):
            raise ApiError("this source format cannot store transparency", code="paint.alpha")
        output = rgba.convert("RGB")
    else:
        output = rgba
    if fmt == "JPEG":
        options.update(quality=95, subsampling=0)
    if fmt == "WEBP":
        options.update(lossless=True, exact=True)
    if fmt == "TIFF":
        options.update(compression="tiff_deflate")
    if fmt in {"PNG", "JPEG", "WEBP"}:
        for key in ("icc_profile", "exif"):
            if metadata.get(key):
                options[key] = metadata[key]
    output.save(staged, format=fmt, **options)
    with Image.open(staged) as check:
        check.load()
        if check.format != fmt or check.size != rgba.size:
            raise ApiError("saved image format or dimensions changed", code="paint.encode")
        if (
            fmt not in {"JPEG", "BMP"}
            and check.convert("RGBA").getchannel("A").tobytes() != rgba.getchannel("A").tobytes()
        ):
            raise ApiError("saved image did not preserve alpha", code="paint.encode")


def _restore_guards(manager: DatasetPipeline, original: dict) -> None:
    result = original["result"]
    image = Path(result["image_path"])
    mask = image.with_name(image.stem + ".mask.png")
    if result.get("after_revision") != _revision(image, mask):
        raise ApiError(
            "image or mask was edited after painting; restore stopped", code="paint.conflict", status=409
        )
    if any(Path(change["target"]) != image for change in result["changes"]) and any(
        p != image and p.stem == image.stem and p.suffix.lower() in IMAGE_EXTS for p in image.parent.iterdir()
    ):
        raise ApiError(
            "another image now shares this mask filename; restore stopped",
            code="paint.shared_mask",
            status=409,
        )


def _reindex(manager: DatasetPipeline, did: str) -> None:
    _index_dataset(manager.c, did)
    row = manager.c.db.fetchone("SELECT index_status FROM datasets WHERE id=?", (did,))
    if not row or row["index_status"] != "ready":
        raise ApiError("image indexing failed; edits were rolled back", code="paint.index", status=422)


def _save(
    manager: DatasetPipeline,
    did: str,
    h: str,
    rel_path: str | None,
    revision: str,
    image_data: bytes | None = None,
    mask_data: bytes | None = None,
    restore: bool = False,
) -> PaintSaved:
    rgba, gray = _decode(image_data), _decode(mask_data, mask=True)
    c = manager.c
    # A lease spans staging, commit, index refresh and operation publication.
    # Register the durable operation under the same lock used by job admission.
    with c.db.lock:
        row, image, mask = _resolve(c, did, h, rel_path)
        pid, vid = row["project_id"], row["version_id"]
        lease = c.versions.mutation(pid, vid)
        lease.__enter__()
        oid = new_id("dp")
        try:
            relative = image.relative_to(Path(row["path"]).resolve()).as_posix()
            manager._resolve_images(pid, vid, [{"dataset_id": did, "rel_path": relative}])
            if _revision(image, mask) != revision:
                raise ApiError(
                    "image or mask changed; reload before saving", code="paint.conflict", status=409
                )
            original = _last(manager, row, image) if restore else None
            if restore and not original:
                raise ApiError("there is no painting operation to restore", code="paint.restore", status=409)
            c.db.insert(
                "dataset_pipeline_operations",
                {
                    "id": oid,
                    "project_id": pid,
                    "version_id": vid,
                    "action": "restore" if restore else "paint",
                    "status": "running",
                    "phase": "staging",
                    "created_at": now(),
                    "updated_at": now(),
                    "request_json": _dump(
                        {
                            "action": "restore" if restore else "paint",
                            "images": [{"dataset_id": did, "rel_path": relative}],
                            "restore_operation_id": original["id"] if original else None,
                        }
                    ),
                },
            )
        except BaseException:
            lease.__exit__(None, None, None)
            raise
    changes = []
    result = {"dataset_ids": [did], "image_path": str(image)}
    try:
        work = manager.root(pid, vid) / oid
        work.mkdir(parents=True, exist_ok=False)
        staged_dir = work / "staged"
        staged_dir.mkdir()
        if restore:
            _restore_guards(manager, original)
            for index, change in enumerate(original["result"]["changes"]):
                target = Path(change["target"])
                if target not in {image, mask, image.with_name(image.stem + ".mask")} or target.is_symlink():
                    raise ApiError(
                        "restore target is outside this image transaction", code="paint.path", status=409
                    )
                if (_digest(target) if target.exists() else None) != change["after"]:
                    raise ApiError(
                        "image or mask was edited after painting; restore stopped",
                        code="paint.conflict",
                        status=409,
                    )
                staged = None
                if change["backup"]:
                    backup = Path(change["backup"])
                    if (
                        backup.is_symlink()
                        or not backup.resolve().is_relative_to(manager.root(pid, vid).resolve())
                        or _digest(backup) != change["before"]
                    ):
                        raise ApiError(
                            "original backup changed; restore stopped", code="paint.backup", status=409
                        )
                    staged = staged_dir / str(index)
                    shutil.copy2(backup, staged)
                    if _digest(staged) != change["before"]:
                        raise ApiError("backup changed while restoring", code="paint.backup", status=409)
                changes.append(manager._change(work, target, staged))
        else:
            pixels, fmt, metadata = _original(image)
            for upload in (rgba, gray):
                if upload is not None and upload.size != pixels.size:
                    raise ApiError(
                        f"painted image and mask must remain {pixels.width}×{pixels.height}",
                        code="paint.size",
                    )
            if rgba is not None:
                staged = staged_dir / "image"
                _encode(rgba, fmt, metadata, staged)
                changes.append(manager._change(work, image, staged))
            if gray is not None:
                if any(
                    p != image and p.stem == image.stem and p.suffix.lower() in IMAGE_EXTS
                    for p in image.parent.iterdir()
                ):
                    raise ApiError(
                        "another image shares this mask filename; rename it before editing the mask",
                        code="paint.shared_mask",
                        status=409,
                    )
                staged = staged_dir / "mask"
                gray.save(staged, format="PNG")
                changes.append(manager._change(work, mask, staged))
        if not changes:
            raise ApiError("provide an image or mask to save", code="paint.upload")
        if any(Path(change["target"]) != image for change in changes) and any(
            p != image and p.stem == image.stem and p.suffix.lower() in IMAGE_EXTS
            for p in image.parent.iterdir()
        ):
            raise ApiError(
                "another image shares this mask filename; rename it before editing the mask",
                code="paint.shared_mask",
                status=409,
            )
        # Detect changes made outside the application while uploads were encoded.
        if _revision(image, mask) != revision:
            raise ApiError(
                "image or mask changed while preparing; reload before saving",
                code="paint.conflict",
                status=409,
            )
        manager._commit(oid, work, changes)
        _reindex(manager, did)
        result.update(
            changes=[] if restore else changes,
            changed_files=len(changes),
            after_revision=_revision(image, mask),
        )
        with c.db.lock:
            # Undo's completion and original-operation marker are one SQLite
            # transaction; restart must never roll back files but retain undone_by.
            c.db.execute("BEGIN IMMEDIATE")
            try:
                if restore:
                    c.db.update(
                        "dataset_pipeline_operations",
                        original["id"],
                        {"result_json": _dump(original["result"] | {"undone_by": oid})},
                    )
                c.db.update(
                    "dataset_pipeline_operations",
                    oid,
                    {
                        "status": "completed",
                        "phase": "completed",
                        "result_json": _dump(result),
                        "updated_at": now(),
                        "finished_at": now(),
                    },
                )
                info = _info(manager, row, image, mask)
                c.db.execute("COMMIT")
            except BaseException:
                c.db.execute("ROLLBACK")
                raise
        return PaintSaved(**info.model_dump(), operation_id=oid)
    except Exception as exc:
        error = str(exc)
        try:
            manager._rollback(changes)
            if changes:
                _reindex(manager, did)
            result.update(changes=[], rolled_back=True)
        except Exception as recovery_error:
            error += f". Recovery needs attention; backups retained: {recovery_error}"
            result.update(changes=changes, recovery_needed=True)
        c.db.update(
            "dataset_pipeline_operations",
            oid,
            {
                "status": "failed",
                "phase": "failed",
                "error": error,
                "result_json": _dump(result),
                "updated_at": now(),
                "finished_at": now(),
            },
        )
        if isinstance(exc, ApiError) and not result.get("recovery_needed"):
            raise
        raise ApiError(error, code="paint.save_failed", status=422) from exc
    finally:
        lease.__exit__(None, None, None)
        c.bus.publish("dataset.pipeline", {"operation_id": oid, "project_id": pid, "version_id": vid})
        c.bus.publish("dataset.changed", {"dataset_id": did, "project_id": pid, "version_id": vid})


@router.put("/datasets/{did}/images/{h}/paint", response_model=PaintSaved)
async def save_paint(
    did: str,
    h: str,
    request: Request,
    rel_path: str | None = None,
    manager: DatasetPipeline = Depends(pipeline),
) -> PaintSaved:
    _resolve(manager.c, did, h, rel_path)
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
        raise ApiError("expected multipart image/mask and revision", code="paint.upload", status=415)

    async def bounded_stream():
        count = 0
        async for chunk in request.stream():
            count += len(chunk)
            if count > MAX_REQUEST_BYTES:
                raise MultiPartException("combined paint upload exceeds 64 MiB")
            yield chunk

    parser = MultiPartParser(request.headers, bounded_stream(), max_files=2, max_fields=1)
    form = None
    try:
        try:
            form = await parser.parse()
        except MultiPartException as exc:
            raise ApiError(str(exc), code="paint.upload", status=413) from exc
        except Exception as exc:
            raise ApiError("invalid or interrupted upload", code="paint.upload") from exc
        if (
            len(form.multi_items()) != len(form)
            or set(form) - {"file", "mask", "revision"}
            or not isinstance(form.get("revision"), str)
            or not form["revision"]
            or not any(key in form for key in ("file", "mask"))
        ):
            raise ApiError("provide an image and/or mask with its revision", code="paint.upload")
        payloads = []
        for key in ("file", "mask"):
            upload = form.get(key)
            if upload is not None and not isinstance(upload, UploadFile):
                raise ApiError("image and mask must be file uploads", code="paint.upload")
            payloads.append(await upload.read(MAX_REQUEST_BYTES + 1) if upload else None)
        return await run_in_threadpool(_save, manager, did, h, rel_path, form["revision"], *payloads)
    finally:
        if form is not None:
            await form.close()
        else:
            for stream in parser._files_to_close_on_error:
                stream.close()


@router.post("/datasets/{did}/images/{h}/paint/restore", response_model=PaintSaved)
def restore_paint(
    did: str,
    h: str,
    body: PaintRestore,
    rel_path: str | None = None,
    manager: DatasetPipeline = Depends(pipeline),
) -> PaintSaved:
    return _save(manager, did, h, rel_path, body.revision, restore=True)
