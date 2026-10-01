"""Bounded upload staging with replay-safe chunks and one dataset publication."""

from __future__ import annotations

import io
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from starlette.datastructures import UploadFile

from . import dataset_uploads as uploads
from .errors import ApiError
from .import_progress import ImportProgress, ImportProgressStore

CHUNK_BYTES = 8 * 1024**2


class UploadManifestFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=240)
    size: int = Field(ge=0, strict=True)


class UploadSessionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version_id: str | None = None
    target_dataset_id: str | None = None
    progress_id: str | None = None
    files: list[UploadManifestFile] = Field(min_length=1, max_length=5000)
    name: str = Field("", max_length=100)
    repeats: int = Field(1, ge=1, le=1_000_000)
    is_reg: bool = False
    prior_weight: float = Field(1, ge=0, allow_inf_nan=False)
    class_prompt: str | None = Field(None, max_length=1024)
    caption_ext: str = Field("auto", max_length=32)


class _StagedReader(io.RawIOBase):
    """Seekable staged file without retaining one descriptor per selected image."""

    def __init__(self, path: Path):
        self.path = path
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._checkClosed()
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.path.stat().st_size + offset
        else:
            raise ValueError("invalid seek origin")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        self._checkClosed()
        with self.path.open("rb") as stream:
            stream.seek(self.position)
            data = stream.read(size)
        self.position += len(data)
        return data


def _manifest(body: UploadSessionBody) -> list[tuple[str, int]]:
    from .routes_work import _validate_caption_extension

    _validate_caption_extension(body.caption_ext)
    if len(body.files) > uploads.MAX_FILES:
        raise ApiError("单次最多上传 5,000 个文件。", code="upload.too_many", status=413)
    paths = [uploads.relative_upload_path(item.name) for item in body.files]
    zipped = [path for path in paths if path.suffix.lower() == ".zip"]
    if zipped and len(paths) != 1:
        raise ApiError("请单独上传一个 ZIP，或选择图片及标签文件。", code="upload.mixed_zip")
    spellings: dict[str, str] = {}
    files: set[str] = set()
    directories: set[str] = set()
    for item, path in zip(body.files, paths, strict=True):
        key = path.as_posix().casefold()
        if key in files or key in directories:
            raise ApiError("上传文件名重复或与目录冲突。", code="upload.duplicate")
        for count in range(1, len(path.parts) + 1):
            prefix = Path(*path.parts[:count]).as_posix()
            folded = prefix.casefold()
            if spellings.setdefault(folded, prefix) != prefix or folded in files:
                raise ApiError("上传路径重复或仅大小写不同。", code="upload.duplicate")
            if count < len(path.parts):
                directories.add(folded)
        files.add(key)
        limit = (
            uploads.MAX_UPLOAD_BYTES if zipped else
            uploads.MAX_CAPTION_BYTES if uploads._ignored(path) else
            uploads._file_limit(path, body.caption_ext)
        )
        if item.size > limit:
            raise ApiError(f"文件超过大小限制：{path.name}", code="upload.too_large", status=413)
    if sum(item.size for item in body.files) > uploads.MAX_UPLOAD_BYTES:
        raise ApiError("单次上传不能超过 2 GiB。", code="upload.too_large", status=413)
    return [(path.as_posix(), item.size) for path, item in zip(paths, body.files, strict=True)]


@dataclass
class UploadSession:
    id: str
    pid: str
    vid: str
    directory: Path
    body: UploadSessionBody
    manifest: list[tuple[str, int]]
    progress: ImportProgress
    touched: float
    received: list[int]
    active: int = 0
    result: dict[str, Any] | None = None
    failure: ApiError | None = None
    cancelled: bool = False
    lock: Any = field(default_factory=threading.RLock)

    @property
    def reserved_bytes(self) -> int:
        return sum(size for _, size in self.manifest) if self.result is None else 0

    def put(self, index: int, offset: int, data: bytes) -> int:
        with self.lock:
            if self.cancelled or self.result is not None or self.failure is not None:
                raise ApiError("上传已结束，请重新选择文件导入。", code="upload.finished", status=409)
            if not 0 <= index < len(self.manifest):
                raise ApiError("上传文件编号无效。", code="upload.file_index", status=404)
            if len(data) > CHUNK_BYTES:
                raise ApiError("上传分片超过大小限制。", code="upload.chunk_size", status=413)
            expected = self.received[index]
            size = self.manifest[index][1]
            if offset < 0 or offset > expected or offset + len(data) > size:
                raise ApiError("上传分片位置不匹配。", code="upload.offset", status=409,
                               details={"received": expected})
            path = self.directory / str(index)
            if offset < expected:
                if offset + len(data) > expected:
                    raise ApiError("上传分片与已接收内容重叠。", code="upload.offset", status=409,
                                   details={"received": expected})
                with path.open("rb") as stream:
                    stream.seek(offset)
                    if stream.read(len(data)) != data:
                        raise ApiError("重传分片与已接收内容不同。", code="upload.chunk_conflict", status=409)
                return expected
            if not data and size != expected:
                raise ApiError("上传分片不能为空。", code="upload.chunk_empty")
            with path.open("r+b" if path.exists() else "xb") as stream:
                stream.seek(expected)
                stream.truncate(expected)
                try:
                    stream.write(data)
                    stream.flush()
                except BaseException:
                    stream.truncate(expected)
                    raise
            self.received[index] += len(data)
            self.progress.advance(bytes_done=len(data), files_done=int(expected < size == self.received[index]))
            return self.received[index]

    def finish(self, publish: Callable[[uploads.UploadBatch, ImportProgress], dict[str, Any]]) -> dict[str, Any]:
        with self.lock:
            if self.cancelled:
                raise ApiError("上传已取消。", code="upload.cancelled", status=409)
            if self.result is not None:
                return self.result
            if self.failure is not None:
                raise self.failure
            if any(received != size for received, (_, size) in zip(self.received, self.manifest, strict=True)):
                raise ApiError("文件尚未上传完成。", code="upload.incomplete", status=409)
            try:
                with ExitStack() as stack:
                    files = []
                    for index, (name, size) in enumerate(self.manifest):
                        path = self.directory / str(index)
                        if size == 0 and not path.exists():
                            path.touch()
                        stream = stack.enter_context(_StagedReader(path))
                        files.append(UploadFile(stream, size=size, filename=name))
                    batch = uploads.UploadBatch(
                        files=files,
                        **self.body.model_dump(exclude={"version_id", "target_dataset_id", "progress_id", "files"}),
                    )
                    self.result = publish(batch, self.progress)
            except BaseException as exc:
                self.failure = exc if isinstance(exc, ApiError) else ApiError(
                    f"导入失败：{exc}", code="upload.failed", status=500
                )
                self.progress.fail(self.failure.code)
                raise
            self.progress.complete()
            # Keep the result for lost-response retries, not another copy of all images.
            shutil.rmtree(self.directory, ignore_errors=True)
            return self.result


class UploadSessionStore:
    def __init__(
        self, root: Path, progress: ImportProgressStore, *, capacity: int = 16,
        max_bytes: int = 8 * 1024**3, ttl_seconds: float = 3600,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.root, self.progress = root, progress
        self.capacity, self.max_bytes, self.ttl_seconds, self.clock = capacity, max_bytes, ttl_seconds, clock
        self.entries: dict[str, UploadSession] = {}
        self.lock = threading.RLock()
        self.closed = False
        self.worker: Path | None = None
        self.owner: Any = None

    @staticmethod
    def _owner_bytes(directory: Path) -> bytes:
        return f"YPuddin dataset upload staging v1\n{directory.name}\n".encode()

    @staticmethod
    def _lock_owner(stream: Any) -> bool:
        try:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def _ensure_worker(self) -> Path:
        if self.worker is not None:
            return self.worker
        if self.root.is_symlink():
            raise ApiError("上传临时目录不能是符号链接。", code="upload.path")
        self.root.mkdir(parents=True, exist_ok=True)
        worker = self.root / f"worker-{uuid.uuid4().hex}"
        worker.mkdir(mode=0o700)
        stream = (worker / "owner.lock").open("x+b")
        try:
            # Publish recognizable ownership only after acquiring the live-process lock.
            stream.write(b"\0")
            stream.flush()
            if not self._lock_owner(stream):
                raise ApiError("无法锁定上传临时目录。", code="upload.staging_busy", status=503)
            stream.seek(0)
            stream.write(self._owner_bytes(worker))
            stream.truncate()
            stream.flush()
        except BaseException:
            stream.close()
            shutil.rmtree(worker, ignore_errors=True)
            raise
        self.worker, self.owner = worker, stream
        return worker

    def _prune_orphans(self) -> None:
        if not self.root.is_dir() or self.root.is_symlink():
            return
        for directory in self.root.iterdir():
            if (directory == self.worker or not re.fullmatch(r"worker-[a-f0-9]{32}", directory.name)
                    or directory.is_symlink() or not directory.is_dir()):
                continue
            marker = directory / "owner.lock"
            if marker.is_symlink():
                continue
            try:
                with marker.open("r+b") as stream:
                    if stream.read(256) != self._owner_bytes(directory) or not self._lock_owner(stream):
                        continue
                # Owners only create fresh UUID directories. After this lock is released,
                # another cleaner may race with removal, but no live owner can adopt it.
                shutil.rmtree(directory, ignore_errors=True)
            except OSError:
                continue

    def _close_worker(self) -> None:
        if self.closed and not self.entries and self.worker is not None:
            self.owner.close()
            shutil.rmtree(self.worker, ignore_errors=True)
            self.worker = self.owner = None

    def _remove(self, entry: UploadSession, error: str) -> None:
        self.entries.pop(entry.id, None)
        entry.progress.fail(error)
        # Entries are created by this process; never scan or remove unrelated directories.
        shutil.rmtree(entry.directory, ignore_errors=True)

    def prune(self) -> None:
        with self.lock:
            self._prune_orphans()
            cutoff = self.clock() - self.ttl_seconds
            for entry in list(self.entries.values()):
                if not entry.active and entry.touched <= cutoff:
                    self._remove(entry, "upload.expired")

    def create(self, pid: str, vid: str, body: UploadSessionBody) -> UploadSession:
        manifest = _manifest(body)
        with self.lock:
            self.prune()
            if self.closed:
                raise ApiError("服务正在关闭，请稍后重试。", code="service.restarting", status=409)
            active = [entry for entry in self.entries.values() if entry.result is None]
            if len(active) >= self.capacity or len(self.entries) >= self.progress.capacity:
                raise ApiError("同时上传的任务过多，请稍后重试。", code="upload.capacity", status=503)
            if sum(entry.reserved_bytes for entry in active) + sum(size for _, size in manifest) > self.max_bytes:
                raise ApiError("待导入文件总量过大，请等待当前上传完成。", code="upload.capacity", status=503)
            directory = Path(tempfile.mkdtemp(prefix="session-", dir=self._ensure_worker()))
            sid = uuid.uuid4().hex
            try:
                progress = self.progress.start(pid, body.progress_id or sid, "receiving")
                progress.set_phase("receiving", bytes_total=sum(size for _, size in manifest), files_total=len(manifest))
                progress.advance(files_done=sum(size == 0 for _, size in manifest))
                entry = UploadSession(sid, pid, vid, directory, body, manifest, progress, self.clock(), [0] * len(manifest))
                self.entries[sid] = entry
            except BaseException:
                shutil.rmtree(directory, ignore_errors=True)
                raise
            return entry

    @contextmanager
    def use(self, pid: str, sid: str) -> Iterator[UploadSession]:
        with self.lock:
            if self.closed:
                raise ApiError("服务正在关闭，请稍后重试。", code="service.restarting", status=409)
            self.prune()
            entry = self.entries.get(sid)
            if entry is None or entry.pid != pid or entry.cancelled:
                raise ApiError("上传会话不存在或已过期，请重新导入。", code="upload.session_not_found", status=404)
            entry.active += 1
            entry.touched = self.clock()
        try:
            yield entry
        finally:
            with self.lock:
                entry.active -= 1
                entry.touched = self.clock()
                if (self.closed or entry.cancelled) and not entry.active:
                    self._remove(entry, "upload.cancelled")
                self._close_worker()

    def delete(self, pid: str, sid: str) -> None:
        with self.lock:
            entry = self.entries.get(sid)
            if entry is None or entry.pid != pid:
                raise ApiError("上传会话不存在或已过期。", code="upload.session_not_found", status=404)
            entry.cancelled = True
            entry.progress.fail("upload.cancelled")
            if not entry.active:
                self._remove(entry, "upload.cancelled")

    def close(self) -> None:
        with self.lock:
            self.closed = True
            for entry in list(self.entries.values()):
                entry.cancelled = True
                if not entry.active:
                    self._remove(entry, "upload.cancelled")
            self._close_worker()
