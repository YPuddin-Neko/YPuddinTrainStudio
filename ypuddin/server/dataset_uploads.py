"""Bounded multipart ingestion into isolated, project-owned dataset directories."""

from __future__ import annotations

import re
import shutil
import stat
import tempfile
import unicodedata
import warnings
import zipfile
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import BinaryIO

from fastapi import Request
from PIL import Image
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from ypuddin.data.index import IMAGE_EXTS

from .errors import ApiError

MAX_UPLOAD_BYTES = 2 * 1024**3
MAX_EXPANDED_BYTES = 4 * 1024**3
MAX_FILE_BYTES = 256 * 1024**2
MAX_CAPTION_BYTES = 1024**2
MAX_FILES = 5000
MAX_IMAGE_PIXELS = 100_000_000
MAX_ZIP_RATIO = 200
CHUNK = 1024**2


class _MultipartLimit(MultiPartException):
    pass


class _UploadParser(MultiPartParser):
    """Bound form text even with Starlette versions predating max_part_size."""

    spool_max_size = 64 * 1024
    max_file_size = spool_max_size  # Starlette < 0.46 uses this name for its spool threshold.

    def on_part_begin(self) -> None:
        super().on_part_begin()
        self._part_bytes = 0
        self._part_limit = 1024

    def on_headers_finished(self) -> None:
        super().on_headers_finished()
        part = self._current_part
        if part.file is not None:
            if part.field_name != "files":
                raise MultiPartException("uploaded files must use the files field")
            path = relative_upload_path(part.file.filename or "")
            self._part_limit = (
                MAX_UPLOAD_BYTES
                if path.suffix.lower() == ".zip"
                else MAX_CAPTION_BYTES
                if _ignored(path)
                else MAX_FILE_BYTES
                if path.suffix.lower() in IMAGE_EXTS or path.name.endswith(".mask")
                else MAX_CAPTION_BYTES
            )

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        self._part_bytes += end - start
        if self._part_bytes > self._part_limit:
            raise _MultipartLimit("upload part exceeds its size limit")
        super().on_part_data(data, start, end)


@dataclass
class UploadBatch:
    files: list[UploadFile]
    name: str
    repeats: int
    is_reg: bool = False
    prior_weight: float = 1.0
    class_prompt: str | None = None
    caption_ext: str = ".txt"


@asynccontextmanager
async def read_upload(request: Request) -> AsyncIterator[UploadBatch]:
    """Parse incrementally; spooled upload files close on every success/error path."""
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
        raise ApiError("expected multipart/form-data", code="upload.content_type", status=415)
    length = request.headers.get("content-length")
    if length:
        try:
            declared = int(length)
        except ValueError as exc:
            raise ApiError("invalid Content-Length", code="upload.invalid") from exc
        if declared < 0 or declared > MAX_UPLOAD_BYTES:
            raise ApiError("upload exceeds the 2 GiB request limit", code="upload.too_large", status=413)

    async def bounded_stream():
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise _MultipartLimit("upload exceeds the 2 GiB request limit")
            yield chunk

    parser = _UploadParser(request.headers, bounded_stream(), max_files=MAX_FILES, max_fields=6)
    form = None
    try:
        try:
            form = await parser.parse()
        except ApiError:
            raise
        except MultiPartException as exc:
            raise ApiError(
                str(exc), code="upload.invalid", status=413 if isinstance(exc, _MultipartLimit) else 400
            ) from exc
        except Exception as exc:
            raise ApiError("invalid or interrupted multipart upload", code="upload.invalid") from exc
        entries = form.multi_items()
        fields = {"name", "repeats", "is_reg", "prior_weight", "class_prompt", "caption_ext"}
        if any(key not in {"files", *fields} for key, _value in entries):
            raise ApiError("unknown upload field", code="upload.invalid")
        files = form.getlist("files")
        if not files or any(not isinstance(item, UploadFile) for item in files):
            raise ApiError("files must contain uploaded images or one ZIP", code="upload.no_files")
        if any(len(form.getlist(key)) > 1 for key in fields):
            raise ApiError("upload option fields must occur at most once", code="upload.invalid")
        if any(not isinstance(form.get(key, ""), str) for key in fields):
            raise ApiError("upload options must be text fields", code="upload.invalid")
        name = form.get("name", "upload")
        repeats = form.get("repeats", "1")
        if not isinstance(name, str) or not isinstance(repeats, str):
            raise ApiError("name and repeats must be text fields", code="upload.invalid")
        if len(name) > 100:
            raise ApiError("dataset name must not exceed 100 characters", code="upload.invalid")
        try:
            repeat_count = int(repeats)
        except ValueError as exc:
            raise ApiError("repeats must be a positive integer", code="upload.invalid") from exc
        if not 1 <= repeat_count <= 1_000_000:
            raise ApiError("repeats must be between 1 and 1000000", code="upload.invalid")
        from pydantic import ValidationError

        from .routes_work import DatasetBody, _validate_caption_extension

        try:
            options = DatasetBody.model_validate(
                {
                    "path": ".",
                    "is_reg": form.get("is_reg", "false"),
                    "prior_weight": form.get("prior_weight", "1"),
                    "class_prompt": form.get("class_prompt") or None,
                    "caption_ext": form.get("caption_ext") or ".txt",
                }
            )
            _validate_caption_extension(options.caption_ext)
        except (ValidationError, ValueError) as exc:
            raise ApiError(f"invalid dataset options: {exc}", code="upload.invalid") from exc
        yield UploadBatch(
            files=files,
            name=name,
            repeats=repeat_count,
            is_reg=options.is_reg,
            prior_weight=options.prior_weight,
            class_prompt=options.class_prompt,
            caption_ext=options.caption_ext,
        )
    finally:
        if form is not None:
            await form.close()
        else:
            # Older Starlette only cleans up MultiPartException, not disconnect/cancellation.
            for handle in parser._files_to_close_on_error:
                handle.close()


def relative_upload_path(name: str) -> Path:
    """Accept browser relative paths, rejecting ambiguous or unsafe Windows/POSIX names."""
    name = unicodedata.normalize("NFC", name.replace("\\", "/"))
    win = PureWindowsPath(name)
    parts = name.split("/")
    if not name or len(name) > 240 or len(parts) > 16 or win.drive or win.root:
        raise ApiError(f"invalid upload path: {name!r}", code="upload.path")
    for part in parts:
        if (
            not part
            or part in {".", ".."}
            or part[-1] in {".", " "}
            or any(ord(char) < 32 or char in '<>:"|?*' for char in part)
            or re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³]", part.split(".")[0].rstrip(" "), re.I)
        ):
            raise ApiError(f"invalid upload path: {name!r}", code="upload.path")
    path = Path(*parts)
    if path.name.lower().endswith(".mask.png"):
        path = path.with_name(path.name[:-9] + ".mask.png")
    elif path.suffix.lower() in {".txt", ".mask"}:
        path = path.with_suffix(path.suffix.lower())
    return path


def _ignored(path: Path) -> bool:
    return any(part == "__MACOSX" or part == ".DS_Store" or part.startswith("._") for part in path.parts)


def _file_limit(path: Path, caption_ext: str = ".txt") -> int:
    if path.name.endswith(caption_ext):
        return MAX_CAPTION_BYTES
    if path.suffix.lower() in IMAGE_EXTS or path.name.endswith(".mask"):
        return MAX_FILE_BYTES
    raise ApiError(f"unsupported dataset file: {path}", code="upload.file_type")


def _copy_file(source: BinaryIO, destination: Path, limit: int, budget: list[int]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with destination.open("xb") as output:
        while chunk := source.read(CHUNK):
            size += len(chunk)
            budget[0] += len(chunk)
            if size > limit or budget[0] > MAX_EXPANDED_BYTES:
                raise ApiError(
                    "dataset file or expanded upload is too large", code="upload.too_large", status=413
                )
            output.write(chunk)


def _validate_files(paths: list[Path], caption_ext: str = ".txt") -> None:
    images: dict[Path, Path] = {}
    sidecars = []
    for path in paths:
        if path.name.endswith(caption_ext):
            try:
                path.read_text(encoding="utf-8-sig")
            except UnicodeError as exc:
                raise ApiError(f"caption must be UTF-8: {path.name}", code="upload.caption") from exc
            sidecars.append((path, path.with_name(path.name[: -len(caption_ext)])))
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as im:
                    if im.width * im.height > MAX_IMAGE_PIXELS:
                        raise ValueError("image exceeds the pixel limit")
                    im.verify()
                with Image.open(path) as im:
                    im.load()  # Header-only probes can accept truncated JPEGs that fail in training.
        except Exception as exc:
            raise ApiError(f"invalid or oversized image: {path.name}", code="upload.image") from exc
        if path.name.endswith(".mask.png"):
            sidecars.append((path, path.with_name(path.name[:-9])))
        elif path.name.endswith(".mask"):
            sidecars.append((path, path.with_suffix("")))
        else:
            stem = path.with_suffix("")
            if stem in images:
                raise ApiError(f"image names share a caption stem: {path.name}", code="upload.duplicate")
            images[stem] = path
    if not images:
        raise ApiError("upload contains no training images", code="upload.no_images")
    for path, stem in sidecars:
        if stem not in images:
            raise ApiError(
                f"caption or mask has no matching image: {path.name}", code="upload.orphan_sidecar"
            )


@contextmanager
def staged_upload(
    project_dir: Path, dataset_id: str, batch: UploadBatch, *, dataset_root: Path | None = None
) -> Iterator[Path]:
    """Promote a validated batch once; remove only this batch if registration fails."""
    root = dataset_root or project_dir / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.resolve().is_relative_to(project_dir.resolve()):
        raise ApiError("managed dataset directory must remain inside its project", code="upload.path")
    slug = re.sub(r"[^\w-]+", "-", batch.name, flags=re.UNICODE).strip("-_")[:60] or "upload"
    destination = root / f"{dataset_id}-{slug}"
    temporary = Path(tempfile.mkdtemp(prefix=".upload-", dir=root))
    promoted = False
    completed = False
    try:
        names = [relative_upload_path(item.filename or "") for item in batch.files]
        zip_inputs = [i for i, path in enumerate(names) if path.suffix.lower() == ".zip"]
        budget = [0]
        paths = []
        seen = set()

        def target(path: Path) -> Path:
            key = path.as_posix().casefold()
            if key in seen:
                raise ApiError(f"duplicate upload path: {path}", code="upload.duplicate")
            seen.add(key)
            result = temporary / path
            paths.append(result)
            return result

        if zip_inputs:
            if len(batch.files) != 1:
                raise ApiError("upload one ZIP or separate files, not both", code="upload.mixed_zip")
            batch.files[0].file.seek(0)
            with zipfile.ZipFile(batch.files[0].file) as archive:
                entries = archive.infolist()
                if len(entries) > MAX_FILES:
                    raise ApiError("ZIP contains too many entries", code="upload.too_many", status=413)
                if sum(item.file_size for item in entries) > MAX_EXPANDED_BYTES:
                    raise ApiError("ZIP expands beyond 4 GiB", code="upload.too_large", status=413)
                for item in entries:
                    path = relative_upload_path(item.filename.rstrip("/"))
                    mode = (item.external_attr >> 16) & 0xFFFF
                    if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}):
                        raise ApiError(
                            "ZIP links and special files are not supported", code="upload.zip_entry"
                        )
                    if item.flag_bits & 1 or item.compress_type not in {
                        zipfile.ZIP_STORED,
                        zipfile.ZIP_DEFLATED,
                    }:
                        raise ApiError("encrypted or unsupported ZIP compression", code="upload.zip_entry")
                    if item.is_dir() or _ignored(path):
                        continue
                    limit = _file_limit(path, batch.caption_ext)
                    if item.file_size > limit or (
                        item.file_size > CHUNK and item.file_size / max(item.compress_size, 1) > MAX_ZIP_RATIO
                    ):
                        raise ApiError(
                            "ZIP entry exceeds size or compression-ratio limit",
                            code="upload.too_large",
                            status=413,
                        )
                    with archive.open(item) as source:
                        _copy_file(source, target(path), limit, budget)
        else:
            if len(batch.files) > MAX_FILES:
                raise ApiError("too many uploaded files", code="upload.too_many", status=413)
            for upload, path in zip(batch.files, names, strict=True):
                if _ignored(path):
                    continue
                limit = _file_limit(path, batch.caption_ext)
                if upload.size is not None and upload.size > limit:
                    raise ApiError(f"uploaded file is too large: {path}", code="upload.too_large", status=413)
                upload.file.seek(0)
                _copy_file(upload.file, target(path), limit, budget)
        _validate_files(paths, batch.caption_ext)
        if destination.exists():
            raise ApiError("dataset destination already exists", code="upload.duplicate", status=409)
        temporary.rename(destination)
        promoted = True
        yield destination
        completed = True
    except ApiError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ApiError(f"could not import upload: {exc}", code="upload.invalid") from exc
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
        if promoted and not completed:
            shutil.rmtree(destination, ignore_errors=True)
