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
from .import_progress import ImportProgress

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
    progress: ImportProgress | None = None

    def on_part_end(self) -> None:
        super().on_part_end()
        if self.progress and self._current_part.file is not None:
            self.progress.advance(files_done=1)

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
    caption_ext: str = "auto"


@asynccontextmanager
async def read_upload(request: Request, progress: ImportProgress | None = None) -> AsyncIterator[UploadBatch]:
    """Parse incrementally; spooled upload files close on every success/error path."""
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
        raise ApiError("expected multipart/form-data", code="upload.content_type", status=415)
    length = request.headers.get("content-length")
    declared = None
    if length:
        try:
            declared = int(length)
        except ValueError as exc:
            raise ApiError("invalid Content-Length", code="upload.invalid") from exc
        if declared < 0 or declared > MAX_UPLOAD_BYTES:
            raise ApiError("upload exceeds the 2 GiB request limit", code="upload.too_large", status=413)
    if progress:
        progress.set_phase("receiving", bytes_total=declared)

    async def bounded_stream():
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if progress:
                progress.advance(bytes_done=len(chunk))
            if total > MAX_UPLOAD_BYTES:
                raise _MultipartLimit("upload exceeds the 2 GiB request limit")
            yield chunk

    parser = _UploadParser(request.headers, bounded_stream(), max_files=MAX_FILES, max_fields=6)
    parser.progress = progress
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
        name = form.get("name", "")
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
                    "caption_ext": form.get("caption_ext") or "auto",
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
    elif path.suffix.lower() in {".txt", ".json", ".mask"}:
        path = path.with_suffix(path.suffix.lower())
    return path


def _ignored(path: Path) -> bool:
    return any(part == "__MACOSX" or part == ".DS_Store" or part.startswith("._") for part in path.parts)


def _file_limit(path: Path, caption_ext: str = "auto") -> int:
    if path.suffix.lower() in {".txt", ".json"} or (
        caption_ext != "auto" and path.name.endswith(caption_ext)
    ):
        return MAX_CAPTION_BYTES
    if path.suffix.lower() in IMAGE_EXTS or path.name.endswith(".mask"):
        return MAX_FILE_BYTES
    raise ApiError(f"unsupported dataset file: {path}", code="upload.file_type")


def _copy_file(
    source: BinaryIO, destination: Path, limit: int, budget: list[int], progress: ImportProgress | None = None
) -> None:
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
            if progress:
                progress.advance(bytes_done=len(chunk))
    if progress:
        progress.advance(files_done=1)


def _validate_files(
    paths: list[Path], caption_ext: str = "auto", progress: ImportProgress | None = None
) -> None:
    if progress:
        progress.set_phase("validating", files_total=len(paths))
    images: dict[Path, Path] = {}
    sidecars = []
    for path in paths:
        if path.suffix.lower() in {".txt", ".json"} or (
            caption_ext != "auto" and path.name.endswith(caption_ext)
        ):
            try:
                path.read_text(encoding="utf-8-sig")
                if path.suffix.lower() == ".json":
                    from ypuddin.data.captions import read_caption

                    read_caption(str(path), None)
            except (UnicodeError, ValueError) as exc:
                raise ApiError(f"invalid caption {path.name}: {exc}", code="upload.caption") from exc
            suffix = path.suffix if path.suffix.lower() in {".txt", ".json"} else caption_ext
            sidecars.append((path, path.with_name(path.name[: -len(suffix)])))
            if progress:
                progress.advance(files_done=1)
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
        if progress:
            progress.advance(files_done=1)
    if not images:
        raise ApiError("upload contains no training images", code="upload.no_images")
    for path, stem in sidecars:
        if stem not in images:
            raise ApiError(
                f"caption or mask has no matching image: {path.name}", code="upload.orphan_sidecar"
            )


def _same_bytes(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as a, right.open("rb") as b:
        while chunk := a.read(CHUNK):
            if chunk != b.read(CHUNK):
                return False
        return not b.read(1)


def _destination_path(
    root: Path, relative: Path, children: dict[Path, dict[str, str]], *, directory: bool = False
) -> Path:
    """Reject links, file/directory clashes and case aliases before publishing any files."""
    parent = root
    for index, name in enumerate(relative.parts):
        candidate = parent / name
        if parent.is_dir():
            if parent not in children:
                children[parent] = {
                    unicodedata.normalize("NFC", p.name).casefold(): p.name for p in parent.iterdir()
                }
            alias = children[parent].get(name.casefold())
            if alias is not None and alias != name:
                raise ApiError(
                    f"upload path conflicts with existing spelling: {relative}",
                    code="upload.conflict",
                    status=409,
                )
        if candidate.is_symlink():
            raise ApiError(f"upload target contains a symbolic link: {relative}", code="upload.path")
        if candidate.exists() and candidate.is_dir() != (index < len(relative.parts) - 1 or directory):
            raise ApiError(
                f"upload file and directory names conflict: {relative}",
                code="upload.conflict",
                status=409,
            )
        parent = candidate
    return parent


@contextmanager
def merge_dataset_files(
    root: Path,
    temporary: Path,
    paths: list[Path],
    *,
    loose_name: str = "",
    group_loose: bool = True,
    protected_roots: tuple[Path, ...] = (),
    progress: ImportProgress | None = None,
) -> Iterator[list[Path]]:
    """Merge a verified snapshot; the caller registers sources before this context commits."""
    created_files: list[Path] = []
    created_dirs: list[Path] = []
    completed = False
    try:
        # Only loose images need a new container. Real directory names, including
        # those inside an archive, are kept exactly as supplied by the user.
        loose = [path for path in paths if path.parent == temporary]
        if loose and group_loose:
            base = re.sub(r"[^\w-]+", "-", loose_name, flags=re.UNICODE).strip("-_")[:60] or "images"
            occupied = {p.name.casefold() for p in [*root.iterdir(), *temporary.iterdir()]}
            name, number = base, 2
            while name.casefold() in occupied:
                name, number = f"{base}-{number}", number + 1
            container = temporary / name
            container.mkdir()
            for path in loose:
                replacement = container / path.name
                path.rename(replacement)
                paths[paths.index(path)] = replacement

        directories = sorted(root / path.name for path in temporary.iterdir() if path.is_dir())
        pending: list[tuple[Path, Path]] = []
        children: dict[Path, dict[str, str]] = {}
        caption_stems: dict[Path, dict[str, set[str]]] = {}
        if progress:
            progress.set_phase("validating", files_total=len(paths))
        for directory in directories:
            _destination_path(root, directory.relative_to(root), children, directory=True)
        for source in paths:
            relative = source.relative_to(temporary)
            destination = _destination_path(root, relative, children)
            if any(destination.is_relative_to(protected) for protected in protected_roots):
                raise ApiError(
                    "上传路径属于另一个数据集，请打开该数据集后添加。", code="upload.conflict", status=409
                )
            if source.suffix.lower() in IMAGE_EXTS and not source.name.endswith(".mask.png"):
                if destination.parent not in caption_stems:
                    siblings: dict[str, set[str]] = {}
                    for name in children.get(destination.parent, {}).values():
                        existing = Path(name)
                        if existing.suffix.lower() in IMAGE_EXTS and not name.lower().endswith(".mask.png"):
                            siblings.setdefault(existing.stem.casefold(), set()).add(name)
                    caption_stems[destination.parent] = siblings
                matches = caption_stems[destination.parent].get(destination.stem.casefold(), set())
                if matches - {destination.name}:
                    raise ApiError(
                        f"an existing image already uses this caption name: {relative}",
                        code="upload.conflict",
                        status=409,
                    )
            if destination.exists():
                if not destination.is_file() or not _same_bytes(source, destination):
                    raise ApiError(
                        f"existing file has different content: {relative}; rename it before importing",
                        code="upload.conflict",
                        status=409,
                    )
                if progress:
                    progress.advance(files_done=1)
                continue
            pending.append((source, destination))
            if progress:
                progress.advance(files_done=1)

        if progress:
            progress.set_phase(
                "copying",
                bytes_total=sum(source.stat().st_size for source, _ in pending),
                files_total=len(pending),
            )
        for directory in directories:
            if not directory.exists():
                directory.mkdir()
                created_dirs.append(directory)
        for source, destination in pending:
            missing = []
            parent = destination.parent
            while parent != root and not parent.exists():
                missing.append(parent)
                parent = parent.parent
            for directory in reversed(missing):
                directory.mkdir()
                created_dirs.append(directory)
            # Exclusive creation also protects against a target appearing after preflight.
            with destination.open("xb") as output:
                created_files.append(destination)
                with source.open("rb") as stream:
                    while chunk := stream.read(CHUNK):
                        output.write(chunk)
                        if progress:
                            progress.advance(bytes_done=len(chunk))
            shutil.copystat(source, destination)
            if progress:
                progress.advance(files_done=1)
        yield directories
        completed = True
    finally:
        if not completed:
            for path in reversed(created_files):
                path.unlink(missing_ok=True)
            for path in reversed(created_dirs):
                try:
                    path.rmdir()
                except OSError:
                    pass  # Never remove preexisting or concurrently added user files.


@contextmanager
def staged_upload(
    project_dir: Path,
    batch: UploadBatch,
    *,
    dataset_root: Path | None = None,
    append: bool = False,
    protected_roots: tuple[Path, ...] = (),
    progress: ImportProgress | None = None,
) -> Iterator[list[Path]]:
    """Merge validated folders without wrappers; roll back only newly published data."""
    root = dataset_root or project_dir / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.resolve().is_relative_to(project_dir.resolve()):
        raise ApiError("managed dataset directory must remain inside its project", code="upload.path")
    temporary = Path(tempfile.mkdtemp(prefix=".upload-", dir=root))
    staging_root = temporary
    try:
        names = [relative_upload_path(item.filename or "") for item in batch.files]
        zip_inputs = [i for i, path in enumerate(names) if path.suffix.lower() == ".zip"]
        budget = [0]
        paths = []
        seen = set()
        spellings: dict[str, str] = {}

        def target(path: Path) -> Path:
            key = path.as_posix().casefold()
            if key in seen:
                raise ApiError(f"duplicate upload path: {path}", code="upload.duplicate")
            seen.add(key)
            for count in range(1, len(path.parts) + 1):
                prefix = Path(*path.parts[:count]).as_posix()
                previous = spellings.setdefault(prefix.casefold(), prefix)
                if previous != prefix:
                    raise ApiError(f"upload paths differ only by case: {path}", code="upload.duplicate")
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
                if progress:
                    included = [
                        item
                        for item in entries
                        if not item.is_dir() and not _ignored(relative_upload_path(item.filename))
                    ]
                    progress.set_phase(
                        "extracting",
                        bytes_total=sum(item.file_size for item in included),
                        files_total=len(included),
                    )
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
                        _copy_file(source, target(path), limit, budget, progress)
        else:
            if len(batch.files) > MAX_FILES:
                raise ApiError("too many uploaded files", code="upload.too_many", status=413)
            if progress:
                included_uploads = [
                    upload for upload, path in zip(batch.files, names, strict=True) if not _ignored(path)
                ]
                sizes = [upload.size for upload in included_uploads]
                progress.set_phase(
                    "copying",
                    bytes_total=sum(sizes) if all(size is not None for size in sizes) else None,
                    files_total=len(included_uploads),
                )
            for upload, path in zip(batch.files, names, strict=True):
                if _ignored(path):
                    continue
                limit = _file_limit(path, batch.caption_ext)
                if upload.size is not None and upload.size > limit:
                    raise ApiError(f"uploaded file is too large: {path}", code="upload.too_large", status=413)
                upload.file.seek(0)
                _copy_file(upload.file, target(path), limit, budget, progress)
        _validate_files(paths, batch.caption_ext, progress)
        if append and paths:
            # A folder/ZIP's single outer folder is the upload container, not a new dataset.
            roots = {path.relative_to(temporary).parts[0] for path in paths}
            if len(roots) == 1 and all(len(path.relative_to(temporary).parts) > 1 for path in paths):
                container = temporary / next(iter(roots))
                temporary = container

        with merge_dataset_files(
            root,
            temporary,
            paths,
            loose_name=batch.name,
            group_loose=not append,
            protected_roots=protected_roots,
            progress=progress,
        ) as directories:
            yield directories
    except ApiError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ApiError(f"could not import upload: {exc}", code="upload.invalid") from exc
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
