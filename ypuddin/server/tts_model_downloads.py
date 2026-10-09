"""Stream complete speech model packages into independently verified directories."""

from __future__ import annotations

import copy
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from ypuddin.tts.model_download_models import (
    TtsInstalledModel,
    TtsModelBindings,
    TtsModelDownload,
    TtsModelFile,
    TtsModelIssue,
    TtsModelPackage,
)

from .db import new_id, now
from .download_errors import download_auth_error, download_http_error
from .errors import ApiError, Conflict, NotFound
from .model_downloads import _Redirect
from .network import ProxyPolicy

ACTIVE = {"queued", "downloading", "verifying"}
STATE_KEY = "tts_model_downloads_v1"
CATALOG_PATH = Path(__file__).resolve().parents[1] / "tts" / "model_assets.json"
CHUNK_SIZE = 256 * 1024
MARKER = ".ypuddin-tts-package.json"
STAGE_MARKER = ".owner.json"


class _Cancelled(Exception):
    pass


def _failure(code: str, message: str) -> ApiError:
    return ApiError(message, code=f"tts.model_download.{code}")


def _relative(value: str) -> str:
    if (
        not value or "\\" in value or any(c in value for c in ':<>"|?*')
        or any(ord(c) < 32 for c in value)
        or PurePosixPath(value).is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or any(part.endswith((".", " ")) or part.split(".")[0].upper() in {
            "CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(10)], *[f"LPT{i}" for i in range(10)]
        } for part in value.split("/"))
    ):
        raise _failure("path", "模型包包含无效文件路径。")
    return value


def _package(spec: dict) -> TtsModelPackage:
    value = copy.deepcopy(spec)
    files = [TtsModelFile.model_validate(item) for item in value["files"]]
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,79}", value["id"]):
        raise _failure("catalog", "模型包标识无效。")
    occupied = {MARKER.casefold()}
    for item in files:
        if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", item.repo_id) or ".." in item.repo_id:
            raise _failure("catalog", "模型包来源无效。")
        _relative(item.filename)
        for path in [item.path, *[member.path for member in item.extract]]:
            folded = _relative(path).casefold()
            if folded in occupied or any(
                folded.startswith(previous + "/") or previous.startswith(folded + "/")
                for previous in occupied
            ):
                raise _failure("catalog", "模型包包含重复文件路径。")
            occupied.add(folded)
        members = [_relative(member.member).casefold() for member in item.extract]
        if len(members) != len(set(members)):
            raise _failure("catalog", "模型压缩包包含重复成员。")
    if not files:
        raise _failure("catalog", "模型包没有文件。")
    normalized = {key: value[key] for key in ("id", "engine", "variant", "name", "license", "url")}
    normalized["files"] = [item.model_dump(exclude_none=True) for item in files]
    revision = hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":"),
                                          ensure_ascii=False).encode()).hexdigest()
    return TtsModelPackage(**normalized, revision=revision, size=sum(item.size for item in files))


def _spec(package: TtsModelPackage) -> dict:
    return package.model_dump(exclude={"revision", "size", "providers"}, exclude_none=True) | {
        "variant": package.variant,
    }


def _stat(path: Path) -> list[int]:
    value = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(value.st_mode):
        raise _failure("path", "模型文件不是普通文件。")
    return [value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_ino, value.st_dev]


def _redirected(path: Path) -> bool:
    try:
        value = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(value.st_mode) or bool(getattr(value, "st_file_attributes", 0) & 0x400)


def _directory_identity(path: Path) -> tuple[int, int]:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or _redirected(path):
        raise _failure("path", "临时模型目录不是普通目录。")
    return info.st_dev, info.st_ino


def _rename_new(source: Path, destination: Path) -> None:
    """A directory that appears during publication must never be replaced, even when empty."""
    if os.name == "nt":
        os.rename(source, destination)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = rename(os.fsencode(source), os.fsencode(destination), 4)  # RENAME_EXCL
    else:
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise OSError(errno.ENOTSUP, "Atomic exclusive directory publication is unavailable")
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        result = rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1)  # RENAME_NOREPLACE
    if result:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


class TtsModelDownloads:
    def __init__(self, context, credentials, *, opener=None, catalog_path=None):
        self.context = context
        self.credentials = credentials
        self.opener = opener
        self.catalog_path = Path(catalog_path) if catalog_path else CATALOG_PATH
        self.lock = threading.RLock()
        self._admission = threading.Lock()
        self._saving = threading.Lock()
        self._revision = self._saved = 0
        self.closed = False
        state = context.db.get_kv(STATE_KEY, {})
        self.tasks = {row["id"]: row for row in state.get("tasks", [])}
        self.models = {row["id"]: row for row in state.get("installed", [])}
        self.cancelled: dict[str, threading.Event] = {}
        self._publishing: set[str] = set()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-model-download")
        for id_, record in list(self.models.items()):
            task = self.tasks.get(record.get("_download_id"))
            recovering_task = task is not None and task["status"] in ACTIVE
            if not record.get("_pending") and not recovering_task:
                continue
            root, target = Path(record["_root"]), Path(record["path"])
            try:
                marker_path = target / MARKER
                self._safe_path(root, marker_path)
                if _stat(marker_path) != record["_files"][MARKER]["stat"]:
                    raise ValueError("publication marker changed")
                with marker_path.open("rb") as stream:
                    marker_bytes = stream.read(4097)
                if len(marker_bytes) > 4096 or hashlib.sha256(marker_bytes).hexdigest() != record["_files"][MARKER]["sha256"]:
                    raise ValueError("publication marker identity mismatch")
                marker = json.loads(marker_bytes)
                identity = {"download_id": record["_download_id"], "installed_id": id_,
                            "package_id": record["package_id"], "package_revision": record["package_revision"]}
                if marker != identity:
                    raise ValueError("publication identity mismatch")
                record["_pending"] = False
                if recovering_task and all((
                    task["package_id"] == record["package_id"],
                    task["package_revision"] == record["package_revision"],
                    task["target_path"] == record["path"],
                    task["_root"] == record["_root"],
                )):
                    task["installed_id"] = id_
            except (OSError, ApiError, ValueError):
                if not record.get("_pending"):
                    continue
                if root.is_dir():
                    self.models.pop(id_, None)
                else:
                    # Preserve the frozen location when its disk is temporarily disconnected.
                    record["_pending"] = False
        for row in self.tasks.values():
            if row["status"] in ACTIVE:
                installed = self.models.get(row.get("installed_id"))
                message = "服务停止前下载未完成，请重试。"
                if installed is not None:
                    message = (
                        "服务停止前下载任务未完成；完整模型包已保存，可在已安装模型中选择。"
                        if self._installation(installed).ready
                        else "服务停止前下载任务未完成，请检查已安装模型的文件及目录。"
                    )
                row.update(status="failed", phase="failed", error_code="tts.model_download.interrupted",
                           error=message, finished_at=now(),
                           bytes_per_second=0, eta_seconds=None)
                self._cleanup(row)
        self._persist()

    def _persist(self):
        with self.lock:
            self._revision += 1
            revision = self._revision
            state = {"schema_version": 1, "tasks": copy.deepcopy(list(self.tasks.values())),
                     "installed": copy.deepcopy(list(self.models.values()))}
        with self._saving:
            if revision > self._saved:
                self.context.db.set_kv(STATE_KEY, state)
                self._saved = revision

    @staticmethod
    def _public(row):
        return TtsModelDownload.model_validate({key: row[key] for key in TtsModelDownload.model_fields if key in row})

    def _update(self, id_, **updates):
        with self.lock:
            self.tasks[id_].update(updates)
            row = self._public(self.tasks[id_])
        self._persist()
        self.context.bus.publish("tts.model_download", row.model_dump())
        return row

    def catalog(self) -> list[TtsModelPackage]:
        document = json.loads(self.catalog_path.read_text("utf-8"))
        if document.get("schema_version") != 1:
            raise _failure("catalog", "模型目录版本不受支持。")
        result = [_package(value) for value in document["packages"]]
        if len({item.id for item in result}) != len(result):
            raise _failure("catalog", "模型目录包含重复标识。")
        return result

    def list(self) -> list[TtsModelDownload]:
        with self.lock:
            return [self._public(row) for row in sorted(self.tasks.values(), key=lambda row: row["created_at"],
                                                        reverse=True)]

    def get(self, id_: str) -> TtsModelDownload:
        with self.lock:
            if id_ not in self.tasks:
                raise NotFound("找不到下载任务。", code="tts.model_download.not_found")
            return self._public(self.tasks[id_])

    def _safe_path(self, root: Path, path: Path):
        if _redirected(root) or root.resolve() != root or not self.context.is_allowed(root):
            raise _failure("path", "模型目录不可访问或包含文件重定向。")
        if not path.is_relative_to(root):
            raise _failure("path", "模型路径超出配置目录。")
        current = root
        for part in path.relative_to(root).parts:
            current = current / part
            if _redirected(current):
                raise _failure("path", "模型路径包含文件重定向。")
        if path.resolve() != path:
            raise _failure("path", "模型路径包含文件重定向。")

    def _root(self):
        configured = Path(self.context.settings()["paths"]["models_dir"]).expanduser().absolute()
        if _redirected(configured):
            raise _failure("path", "模型目录不能是符号链接。")
        root = configured.resolve()
        self._safe_path(root, root)
        return root

    def _installation(self, record) -> TtsInstalledModel:
        root, target = Path(record["_root"]), Path(record["path"])
        status, issues = "ready", []
        try:
            self._safe_path(root, target)
            if not target.is_dir():
                raise FileNotFoundError()
            for relative, expected in record["_files"].items():
                path = target / _relative(relative)
                self._safe_path(root, path)
                if _stat(path) != expected["stat"]:
                    status = "changed"
                    issues = [TtsModelIssue(code="tts.model.changed", message="模型文件已改变，请检查文件，或更换模型目录后重新下载。")]
                    break
        except FileNotFoundError:
            status = "missing"
            issues = [TtsModelIssue(code="tts.model.missing", message="模型文件已移走或丢失。")]
        except (OSError, ApiError, ValueError):
            status = "unavailable"
            issues = [TtsModelIssue(code="tts.model.unavailable", message="无法访问模型文件，请检查目录及权限。")]
        public = {key: record[key] for key in TtsInstalledModel.model_fields if key in record}
        return TtsInstalledModel(**public, status=status, ready=status == "ready", issues=issues,
                                 bindings=TtsModelBindings(model_path=str(target)))

    def installed(self) -> list[TtsInstalledModel]:
        with self.lock:
            records = copy.deepcopy([value for value in self.models.values() if not value.get("_pending")])
        return [self._installation(record) for record in records]

    def start(self, package_id: str, provider="huggingface") -> TtsModelDownload:
        if provider != "huggingface":
            raise _failure("provider", "该模型包仅提供 Hugging Face 下载来源。")
        package = next((item for item in self.catalog() if item.id == package_id), None)
        if package is None:
            raise NotFound("找不到模型包。", code="tts.model_download.package_not_found")
        root = self._root()
        target = root / "tts" / package.id / package.revision[:16]
        return self._start(package, provider, root, target)

    def _start(self, package, provider, root, target):
        self._safe_path(root, target)
        with self._admission:
            with self.context.db.lock, self.lock:
                if self.closed:
                    raise Conflict("下载服务正在停止。", code="tts.model_download.closed")
                active = next((row for row in self.tasks.values() if row["package_id"] == package.id
                               and row["_root"] == str(root)
                               and row["status"] in ACTIVE), None)
                if active:
                    return self._public(active)
                if self.context.db.get_kv("environment.maintenance", {}).get("restarting"):
                    raise Conflict("服务正在重启，请稍后重试。", code="service.restarting")
                if target.exists() or target.is_symlink():
                    raise Conflict("目标目录已存在，请选择已有模型或使用其他模型目录。",
                                   code="tts.model_download.exists", details={"path": str(target)})
                id_ = new_id("tdl")
                dto = TtsModelDownload(id=id_, package_id=package.id, package_revision=package.revision,
                                       name=package.name, engine=package.engine, variant=package.variant,
                                       provider=provider, target_path=str(target), status="queued", phase="queued",
                                       total_bytes=package.size, created_at=now())
                self.tasks[id_] = dto.model_dump() | {"_root": str(root), "_spec": _spec(package)}
                self.cancelled[id_] = threading.Event()
                self.context._active_tts_downloads += 1
            try:
                self._persist()
                self.pool.submit(self._run, id_)
            except Exception:
                with self.context.db.lock, self.lock:
                    self.context._active_tts_downloads -= 1
                    self.tasks.pop(id_, None)
                    self.cancelled.pop(id_, None)
                raise
        return dto

    def retry(self, id_: str) -> TtsModelDownload:
        with self.lock:
            row = copy.deepcopy(self.tasks.get(id_))
        if row is None:
            raise NotFound("找不到下载任务。", code="tts.model_download.not_found")
        if row["status"] not in {"failed", "cancelled"}:
            raise Conflict("仅失败或已取消的下载可以重试。", code="tts.model_download.retry")
        package = _package(row["_spec"])
        if package.revision != row["package_revision"]:
            raise Conflict("原下载的模型清单已改变，请从模型目录重新创建任务。", code="tts.model_download.spec")
        root = Path(row["_root"])
        target = root / "tts" / package.id / package.revision[:16]
        if str(target) != row["target_path"]:
            raise _failure("path", "原下载的目标目录无效。")
        return self._start(package, row["provider"], root, target)

    def cancel(self, id_: str) -> TtsModelDownload:
        with self.lock:
            if id_ not in self.tasks:
                raise NotFound("找不到下载任务。", code="tts.model_download.not_found")
            if self.tasks[id_]["status"] in ACTIVE and id_ not in self._publishing:
                self.cancelled[id_].set()
            return self._public(self.tasks[id_])

    def _check(self, id_):
        if self.cancelled[id_].is_set() or self.closed:
            raise _Cancelled()

    def _cleanup(self, row, *, identity=None):
        if not re.fullmatch(r"tdl_[0-9a-f]{12}", row["id"]):
            return
        expected = identity if identity is not None else row.get("_stage_identity")
        if expected is None:
            return
        root = Path(row["_root"])
        stage = root / ".tts-downloads" / row["id"]
        try:
            self._safe_path(root, stage)
            if stage.exists():
                if _directory_identity(stage) != tuple(expected):
                    return
                marker = stage / STAGE_MARKER
                if marker.exists():
                    self._safe_path(root, marker)
                    with marker.open("rb") as stream:
                        content = stream.read(4097)
                    if len(content) > 4096 or json.loads(content) != {"id": row["id"], "target_path": row["target_path"]}:
                        return
                if _directory_identity(stage) != tuple(expected):
                    return
                shutil.rmtree(stage)
        except (OSError, ApiError, ValueError):
            pass

    def _progress(self, id_, done, *, rate=0):
        total = self.tasks[id_]["total_bytes"]
        self._update(id_, downloaded_bytes=done, bytes_per_second=rate,
                     eta_seconds=max(0, total - done) / rate if rate else None, progress_at=now())

    def _reuse(self, id_, item, destination, done):
        with self.lock:
            records = copy.deepcopy([value for value in self.models.values() if not value.get("_pending")])
        for record in records:
            if not self._installation(record).ready:
                continue
            for relative, value in record["_files"].items():
                if value["sha256"] != item.sha256 or value["stat"][0] != item.size:
                    continue
                source = Path(record["path"]) / relative
                self._update(id_, status="downloading", phase="reuse", current_file=item.path,
                             bytes_per_second=0, eta_seconds=None)
                try:
                    with source.open("rb") as reader, destination.open("xb") as writer:
                        self._stream(id_, reader, writer, item, done, network=False)
                    return True
                except _Cancelled:
                    raise
                except (OSError, ApiError):
                    destination.unlink(missing_ok=True)
                    self._progress(id_, done)
        return False

    def _stream(self, id_, reader, writer, item, done, *, network, report_progress=True):
        received, digest = 0, hashlib.sha256()
        last, last_bytes = time.monotonic(), 0
        while True:
            self._check(id_)
            chunk = reader.read(min(CHUNK_SIZE, item.size - received + 1))
            if not chunk:
                break
            received += len(chunk)
            if received > item.size:
                raise _failure("size", "模型文件大小与固定版本不符。")
            writer.write(chunk)
            digest.update(chunk)
            moment = time.monotonic()
            if report_progress and moment - last >= 0.4:
                rate = (received - last_bytes) / (moment - last) if network else 0
                self._progress(id_, done + received, rate=rate)
                last, last_bytes = moment, received
        self._check(id_)
        if received != item.size:
            raise _failure("size", "模型文件下载不完整，请重试。")
        if item.sha256 and digest.hexdigest() != item.sha256:
            raise _failure("checksum", "模型文件内容校验失败，请重试。")
        writer.flush()
        os.fsync(writer.fileno())
        return digest.hexdigest()

    def _extract(self, id_, item, payload):
        self._update(id_, status="verifying", phase="extract", current_file=item.path,
                     bytes_per_second=0, eta_seconds=None)
        result = {}
        with zipfile.ZipFile(payload / item.path) as archive:
            entries = archive.infolist()
            if len(entries) > 10000 or sum(entry.file_size for entry in entries) > (
                sum(entry.size for entry in item.extract) + item.size
            ):
                raise _failure("archive", "模型压缩包超出允许大小。")
            names = set()
            for entry in entries:
                name = _relative(entry.filename.rstrip("/") if entry.is_dir() else entry.filename)
                if name.casefold() in names or stat.S_ISLNK(entry.external_attr >> 16):
                    raise _failure("archive", "模型压缩包包含重复成员或文件重定向。")
                names.add(name.casefold())
            for member in item.extract:
                self._check(id_)
                try:
                    entry = archive.getinfo(member.member)
                except KeyError:
                    raise _failure("archive", "模型压缩包缺少必要文件。") from None
                if entry.is_dir() or entry.file_size != member.size:
                    raise _failure("archive", "模型压缩包成员大小不符。")
                destination = payload / member.path
                self._safe_path(Path(self.tasks[id_]["_root"]), destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as reader, destination.open("xb") as writer:
                    digest = self._stream(id_, reader, writer, member, 0, network=False, report_progress=False)
                result[member.path] = {"sha256": digest, "size": member.size}
        return result

    def _verify(self, id_, payload, expected):
        verified = {}
        for relative, record in expected.items():
            self._check(id_)
            self._update(id_, status="verifying", phase="verify", current_file=relative,
                         bytes_per_second=0, eta_seconds=None)
            path = payload / relative
            self._safe_path(Path(self.tasks[id_]["_root"]), path)
            before = _stat(path)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(CHUNK_SIZE):
                    self._check(id_)
                    digest.update(chunk)
            if before != _stat(path) or before[0] != record["size"] or digest.hexdigest() != record["sha256"]:
                raise _failure("checksum", "模型文件在检查时改变，请重试。")
            verified[relative] = {"sha256": digest.hexdigest(), "stat": before}
        return verified

    def _run(self, id_):
        row = self.tasks[id_]
        root, target = Path(row["_root"]), Path(row["target_path"])
        stage = root / ".tts-downloads" / id_
        payload = stage / "package"
        token, policy = None, ProxyPolicy()
        published = False
        stage_identity = None
        published_identity = None
        installed_id = None
        try:
            self._check(id_)
            package = _package(row["_spec"])
            if package.revision != row["package_revision"]:
                raise _failure("spec", "模型包清单与创建任务时不符。")
            self._safe_path(root, stage)
            stage.parent.mkdir(parents=True, exist_ok=True)
            stage.mkdir()
            stage_identity = _directory_identity(stage)
            with self.lock:
                row["_stage_identity"] = list(stage_identity)
            self._persist()
            if _directory_identity(stage) != stage_identity:
                raise _failure("path", "临时模型目录在创建时改变。")
            (stage / STAGE_MARKER).write_text(json.dumps({"id": id_, "target_path": row["target_path"]}),
                                             encoding="utf-8")
            payload.mkdir()
            done, expected = 0, {}
            credentials_resolved = False
            for item in package.files:
                self._check(id_)
                destination = payload / item.path
                self._safe_path(root, destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not self._reuse(id_, item, destination, done):
                    self._update(id_, status="downloading", phase="download", current_file=item.path,
                                 bytes_per_second=0, eta_seconds=None)
                    policy = ProxyPolicy.from_context(self.context)
                    if not credentials_resolved:
                        token = self.credentials.token("huggingface", policy=policy)
                        credentials_resolved = True
                    headers = {"User-Agent": "YPuddinTrainStudio", "Accept-Encoding": "identity"}
                    if token:
                        headers["Authorization"] = f"Bearer {token}"
                    url = (f"https://huggingface.co/{item.repo_id}/resolve/{item.revision}/"
                           f"{urllib.parse.quote(item.filename, safe='/')}")
                    request = urllib.request.Request(url, headers=headers)
                    opener = self.opener or policy.opener(_Redirect())
                    with opener.open(request, timeout=15) as response, destination.open("xb") as writer:
                        if response.status != 200:
                            raise _failure("http", f"下载源返回意外状态（HTTP {response.status}）。")
                        size = response.headers.get("Content-Length")
                        if size is not None and (not size.isdecimal() or int(size) != item.size):
                            raise _failure("size", "下载源提供的文件大小与固定版本不符。")
                        self._stream(id_, response, writer, item, done, network=True)
                done += item.size
                self._progress(id_, done)
                expected[item.path] = {"size": item.size, "sha256": item.sha256}
                if item.extract:
                    expected.update(self._extract(id_, item, payload))
                    self._progress(id_, done)
            files = self._verify(id_, payload, expected)
            installed_id = new_id("tmi")
            marker = json.dumps({"download_id": id_, "installed_id": installed_id,
                                 "package_id": package.id, "package_revision": package.revision}, sort_keys=True)
            (payload / MARKER).write_text(marker, encoding="utf-8")
            files[MARKER] = {"sha256": hashlib.sha256(marker.encode()).hexdigest(), "stat": _stat(payload / MARKER)}
            self._update(id_, phase="publish", current_file=None)
            with self.lock:
                self.models[installed_id] = {
                    "id": installed_id, "package_id": package.id, "package_revision": package.revision,
                    "name": package.name, "engine": package.engine, "variant": package.variant,
                    "path": str(target), "_root": str(root), "_files": files,
                    "_pending": True, "_download_id": id_,
                }
            # Persist ownership before rename so startup can recover either side of publication.
            self._persist()
            with self.lock:
                self._check(id_)
                self._safe_path(root, target)
                target.parent.mkdir(parents=True, exist_ok=True)
                self._safe_path(root, target)
                for relative, value in files.items():
                    source = payload / relative
                    self._safe_path(root, source)
                    if _stat(source) != value["stat"]:
                        raise _failure("checksum", "模型文件在发布前改变，请重试。")
                if target.exists() or target.is_symlink():
                    raise _failure("exists", "目标目录已存在，未覆盖其中的文件。")
                self._publishing.add(id_)
                published_identity = _directory_identity(payload)
                _rename_new(payload, target)
                published = True
                # Directory publication preserves file metadata used for subsequent cheap polling.
                self.models[installed_id]["_pending"] = False
            self._persist()
            self._cleanup(row, identity=stage_identity)
            stage_identity = None
            self._update(id_, status="completed", phase="completed", current_file=None,
                         downloaded_bytes=package.size, bytes_per_second=0, eta_seconds=None,
                         installed_id=installed_id, finished_at=now())
        except Exception as error:
            if installed_id:
                with self.lock:
                    self.models.pop(installed_id, None)
            if published:
                try:
                    self._safe_path(root, target)
                    if (_directory_identity(target) == published_identity
                            and json.loads((target / MARKER).read_text("utf-8"))["download_id"] == id_
                            and _directory_identity(target) == published_identity):
                        shutil.rmtree(target)
                except (OSError, ApiError, ValueError, KeyError):
                    pass
            cancelled = isinstance(error, _Cancelled) or self.cancelled[id_].is_set()
            message = policy.redact(error)
            if token:
                message = message.replace(token, "[redacted]").replace(urllib.parse.quote(token, safe=""), "[redacted]")
            message = re.sub(r"https?://[^\s<>]+", lambda match: match[0].split("?", 1)[0].split("#", 1)[0], message)
            if isinstance(error, urllib.error.HTTPError):
                message = download_auth_error(error, provider="huggingface", authenticated=bool(token)) or download_http_error(error.code)
            code = error.code if isinstance(error, ApiError) else (
                "tts.model_download.archive" if isinstance(error, zipfile.BadZipFile)
                else "tts.model_download.network" if isinstance(error, (urllib.error.HTTPError, urllib.error.URLError))
                else "tts.model_download.failed"
            )
            self._cleanup(row, identity=stage_identity)
            stage_identity = None
            self._update(id_, status="cancelled" if cancelled else "failed", phase="cancelled" if cancelled else "failed",
                         error_code=None if cancelled else code, error=None if cancelled else message,
                         bytes_per_second=0, eta_seconds=None, finished_at=now())
        finally:
            with self.lock:
                self._publishing.discard(id_)
            if stage_identity is not None:
                self._cleanup(row, identity=stage_identity)
            with self.context.db.lock:
                self.context._active_tts_downloads -= 1

    def close(self):
        with self._admission:
            with self.lock:
                self.closed = True
                for id_, event in self.cancelled.items():
                    if id_ not in self._publishing:
                        event.set()
        self.pool.shutdown(wait=True)
