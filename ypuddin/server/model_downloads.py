"""Managed HF/ModelScope single-file downloads with durable status and atomic registration."""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .context import ServiceContext
from .db import new_id, now
from .errors import ApiError, Conflict, NotFound
from .model_catalog import TAGGER_FILES
from .model_credentials import ModelCredentials, Provider

ACTIVE = {"queued", "downloading"}


class DownloadVerification(BaseModel):
    """Immutable checks copied into each attempt, independent of later catalog edits."""

    id: str
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ModelDownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    family: Literal["anima", "krea2", "sdxl", "flux", "flux2"]
    kind: Literal["dit", "text_encoder", "text_encoder_2", "vae"]
    provider: Provider = "huggingface"
    mirror: Literal["official", "hf-mirror"] = "official"
    url: str | None = None
    repo_id: str | None = None
    filename: str | None = None
    revision: str | None = None
    dtype: Literal["bf16", "fp16", "fp32", "fp8"] | None = None
    is_default: bool = True

    @model_validator(mode="after")
    def source(self):
        if self.family == "flux2" and self.kind == "text_encoder":
            raise ValueError(
                "FLUX.2 text encoders require a complete local HF directory with tokenizer assets; register that directory instead of downloading a single weight file"
            )
        resolve_source(self)
        return self


class ModelDownload(BaseModel):
    id: str
    family: str
    kind: str
    provider: Provider = "huggingface"
    mirror: Literal["official", "hf-mirror"] = "official"
    source_url: str
    filename: str
    target_path: str
    status: Literal["queued", "downloading", "completed", "failed", "cancelled"]
    downloaded_bytes: int = 0
    total_bytes: int | None = None
    error: str | None = None
    model_id: str | None = None
    created_at: float
    finished_at: float | None = None
    dtype: str | None = None
    is_default: bool = True
    recommendation_id: str | None = None
    expected_size: int | None = None
    sha256: str | None = None


def resolve_source(body: ModelDownloadRequest) -> tuple[str, str]:
    """Canonicalize explicitly selected, allowlisted providers; never accept URL credentials.

    ModelScope's official file endpoint is /api/v1/models/{repo}/repo with Revision
    and FilePath query parameters (modelscope.hub.file_download.get_file_download_url).
    """
    repo, filename = body.repo_id, body.filename
    revision = body.revision or ("master" if body.provider == "modelscope" else "main")
    if body.provider == "modelscope" and body.mirror != "official":
        raise ValueError("HF-Mirror is only available for Hugging Face")
    if body.url:
        if repo or filename:
            raise ValueError("use a file URL or repo_id + filename, not both")
        parsed = urllib.parse.urlsplit(body.url.strip())
        hosts = (
            {"modelscope.cn", "www.modelscope.cn"}
            if body.provider == "modelscope"
            else {"huggingface.co", "hf-mirror.com"}
        )
        if parsed.scheme != "https" or parsed.netloc.lower() not in hosts:
            raise ValueError("file URL must use the selected provider's official HTTPS domain (or HF-Mirror)")
        query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        parts = urllib.parse.unquote(parsed.path).strip("/").split("/")
        if body.provider == "modelscope":
            if len(parts) != 6 or parts[:3] != ["api", "v1", "models"] or parts[5] != "repo":
                raise ValueError(
                    "use a ModelScope /api/v1/models/owner/repo/repo?Revision=master&FilePath=... URL, or repository + filename"
                )
            if (
                set(query) - {"Revision", "FilePath"}
                or any(len(v) != 1 for v in query.values())
                or "FilePath" not in query
            ):
                raise ValueError(
                    "ModelScope file URL needs FilePath and optional Revision; tokens belong in credentials settings"
                )
            repo = "/".join(parts[3:5])
            revision = query.get("Revision", [revision])[0]
            filename = query["FilePath"][0]
        else:
            if set(query) - {"download"}:
                raise ValueError(
                    "URL query credentials are not supported; save tokens in credentials settings"
                )
            if len(parts) < 5 or parts[2] not in {"resolve", "blob"}:
                raise ValueError("paste a Hugging Face file URL containing /resolve/ or /blob/")
            repo, revision, filename = "/".join(parts[:2]), parts[3], "/".join(parts[4:])
            if parsed.netloc.lower() == "hf-mirror.com" and body.mirror != "hf-mirror":
                raise ValueError("select HF-Mirror before using its URL")
    if not repo or not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", repo) or ".." in repo:
        raise ValueError("repo_id must be owner/repository")
    if not revision or not re.fullmatch(r"[A-Za-z0-9_.-]+", revision) or revision in {".", ".."}:
        raise ValueError("revision must be a branch, tag or commit without slashes")
    if (
        not filename
        or "\\" in filename
        or any(c in filename for c in ':<>"|?*')
        or any(ord(c) < 32 for c in filename)
    ):
        raise ValueError("filename must be a safe relative repository path")
    path = PurePosixPath(filename)
    if path.is_absolute() or any(p in {".", "..", ""} for p in filename.split("/")):
        raise ValueError("filename cannot contain absolute paths or traversal")
    if any(
        p.endswith((".", " "))
        or p.split(".")[0].upper()
        in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(10)], *[f"LPT{i}" for i in range(10)]}
        for p in path.parts
    ):
        raise ValueError("filename is not portable to Windows")
    if path.suffix.lower() != ".safetensors":
        raise ValueError("download a complete .safetensors file; register local HF directories separately")
    if re.search(r"-\d{5}-of-\d{5}\.safetensors$", path.name):
        raise ValueError("this is one weight shard; register the complete local HF directory instead")
    if body.provider == "modelscope":
        url = f"https://modelscope.cn/api/v1/models/{repo}/repo?{urllib.parse.urlencode({'Revision': revision, 'FilePath': filename})}"
    else:
        host = "hf-mirror.com" if body.mirror == "hf-mirror" else "huggingface.co"
        url = f"https://{host}/{repo}/resolve/{revision}/{urllib.parse.quote(filename, safe='/')}"
    return url, path.name


class _Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        destination = urllib.parse.urlsplit(newurl)
        if destination.scheme != "https" or destination.username or destination.password:
            raise ValueError("download redirected to a non-HTTPS destination")
        if redirected and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            for name in list(redirected.headers) + list(redirected.unredirected_hdrs):
                if name.lower() in {"authorization", "cookie", "proxy-authorization"}:
                    redirected.remove_header(name)
        return redirected


class _Cancelled(Exception):
    pass


def check_component(weights: Any, family: str, kind: str) -> None:
    """Reject recognisable mismatches; complete architectural validation belongs to the loader."""
    if family == "flux2" and kind in ("text_encoder", "text_encoder_2"):
        raise ValueError(
            "FLUX.2 文本编码器需要包含配置、权重和 tokenizer/processor 的完整本地 HF 目录，不支持单文件下载。"
        )
    names = {}
    for key in weights.keys():
        clean = key
        for prefix in ("model.diffusion_model.", "diffusion_model.", "net."):
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
                break
        names[clean] = key
    from .model_inspection import flux_component, sdxl_component

    shapes = {key: weights.get_slice(original).get_shape() for key, original in names.items()}
    sdxl_kind = sdxl_component(shapes)
    flux = flux_component(shapes)
    found_family, found_kind = None, None
    candidates = []
    if sdxl_kind == "dit":
        found_family, found_kind = "sdxl", "dit"
    elif flux is not None:
        found_family, found_kind, candidates = flux["family"], flux["kind"], flux["candidates"]
    elif sdxl_kind:
        found_family, found_kind = ("sdxl" if sdxl_kind == "dit" else None), sdxl_kind
        candidates = ["sdxl", "flux"] if sdxl_kind == "text_encoder" else ["sdxl"]
    elif "x_embedder.proj.1.weight" in names:
        found_family, found_kind = "anima", "dit"
    elif "first.weight" in names and "txtfusion.projector.weight" in names:
        found_family, found_kind = "krea2", "dit"
    elif any(k.startswith("encoder.") for k in names) and any(k.startswith("decoder.") for k in names):
        found_kind = "vae"
    elif any(k.endswith("embed_tokens.weight") for k in names):
        found_kind = "text_encoder"
        key = next(k for k in names if k.endswith("embed_tokens.weight"))
        shape = weights.get_slice(names[key]).get_shape()
        if len(shape) == 2:
            found_family = {1024: "anima", 2560: "krea2"}.get(shape[1])
    if found_kind is not None and (
        found_kind != kind or found_family not in (None, family) or candidates and family not in candidates
    ):
        raise ValueError(f"file looks like {found_family or 'shared'} {found_kind}, not {family} {kind}")


class ModelDownloads:
    def __init__(self, context: ServiceContext):
        self.context = context
        self.credentials = ModelCredentials(context.data_root)
        self.lock = threading.RLock()
        self.tasks: dict[str, dict[str, Any]] = {
            row["id"]: row for row in context.db.get_kv("model_downloads", [])
        }
        self.cancelled: dict[str, threading.Event] = {}
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="model-download")
        self.opener = urllib.request.build_opener(_Redirect())
        self.closed = False
        for row in self.tasks.values():
            row.setdefault("provider", "huggingface")
            row.setdefault("mirror", "official")
            if row["status"] in ACTIVE:
                row.update(
                    status="failed",
                    error="Service restarted before download completed. Start a new download.",
                    finished_at=now(),
                )
                # Only this task's own staging file is removed; registered/user files are untouched.
                target = Path(row["target_path"])
                if len(target.parents) >= 4 and re.fullmatch(r"dl_[0-9a-f]{12}", row["id"]):
                    root = target.parents[3]
                    stage = root / ".downloads" / row["id"]
                    partial = stage / target.name
                    if partial.resolve().is_relative_to(root.resolve()):
                        try:
                            partial.unlink(missing_ok=True)
                            if row["kind"] == "tagger":
                                (stage / "selected_tags.csv").unlink(missing_ok=True)
                            if stage.exists():
                                stage.rmdir()
                        except OSError:
                            pass  # An unavailable old drive should not prevent service startup.
        self._persist()

    def _persist(self) -> None:
        self.context.db.set_kv("model_downloads", list(self.tasks.values()))

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            return [
                dict(row)
                for row in sorted(self.tasks.values(), key=lambda row: row["created_at"], reverse=True)
            ]

    def _update(self, id_: str, **updates: Any) -> None:
        with self.lock:
            self.tasks[id_].update(updates)
            self._persist()
            self.context.bus.publish("model.download", dict(self.tasks[id_]))

    def start(self, body: ModelDownloadRequest, *, recommendation=None) -> dict[str, Any]:
        if recommendation is not None:
            recommendation = DownloadVerification(
                id=recommendation.id, size=recommendation.size, sha256=recommendation.sha256
            )
        source, filename = resolve_source(body)
        root = Path(self.context.settings()["paths"]["models_dir"]).resolve()
        if not self.context.is_allowed(root):
            raise ApiError("model directory is outside allowed storage roots", status=403)
        identity = recommendation.sha256 if recommendation else hashlib.sha256(source.encode()).hexdigest()
        folder = root / body.family / body.kind / identity[:12]
        target = folder / filename
        with self.lock:
            if self.closed:
                raise Conflict("download service is stopping")
            if target.exists() or target.is_symlink() or folder.exists() or folder.is_symlink():
                raise Conflict(
                    "target file already exists; register the existing path instead",
                    code="download.exists",
                    details={"path": str(target)},
                )
            if any(
                Path(row["target_path"]).parent == folder and row["status"] in ACTIVE
                for row in self.tasks.values()
            ):
                raise Conflict("this model file is already downloading", code="download.active")
            if not target.resolve().is_relative_to(root):
                raise ApiError("model target escapes the configured model directory", code="download.path")
            id_ = new_id("dl")
            row = ModelDownload(
                id=id_,
                family=body.family,
                kind=body.kind,
                provider=body.provider,
                mirror=body.mirror,
                source_url=source,
                filename=filename,
                target_path=str(target),
                status="queued",
                created_at=now(),
                dtype=body.dtype,
                is_default=body.is_default,
                recommendation_id=recommendation.id if recommendation else None,
                expected_size=recommendation.size if recommendation else None,
                sha256=recommendation.sha256 if recommendation else None,
            ).model_dump()
            self.tasks[id_] = row
            self.cancelled[id_] = threading.Event()
            self._persist()
            self.pool.submit(self._run, id_, root)
            return dict(row)

    def catalog(self) -> list[dict[str, Any]]:
        # Kept empty for old clients; automatic tagging is no longer offered.
        return []

    def cancel(self, id_: str) -> dict[str, Any]:
        with self.lock:
            if id_ not in self.tasks:
                raise NotFound("download not found", code="download.not_found")
            if self.tasks[id_]["status"] in ACTIVE:
                self.cancelled[id_].set()
            return dict(self.tasks[id_])

    def retry(self, id_: str) -> dict[str, Any]:
        with self.lock:
            row = self.tasks.get(id_)
            if row is None:
                raise NotFound("download not found", code="download.not_found")
            if row["kind"] == "tagger":
                raise ApiError(
                    "automatic tagging is no longer available", status=410, code="download.retired"
                )
            if row["status"] not in {"failed", "cancelled"}:
                raise Conflict("only failed or cancelled downloads can be retried", code="download.retry")
            body = ModelDownloadRequest(
                family=row["family"],
                kind=row["kind"],
                provider=row["provider"],
                mirror=row["mirror"],
                url=row["source_url"],
                dtype=row["dtype"],
                is_default=row["is_default"],
            )
            recommendation = None
            if row.get("recommendation_id") or row.get("sha256") or row.get("expected_size"):
                try:
                    recommendation = DownloadVerification(
                        id=row.get("recommendation_id"),
                        size=row.get("expected_size"),
                        sha256=row.get("sha256"),
                    )
                except ValidationError:
                    raise Conflict(
                        "saved recommendation verification is incomplete; start from the catalog again",
                        code="download.verification",
                    ) from None
            return self.start(body, recommendation=recommendation)

    def _run(self, id_: str, root: Path) -> None:
        row = self.tasks[id_]
        event = self.cancelled[id_]
        partial = root / ".downloads" / id_ / row["filename"]
        target = Path(row["target_path"])
        published = False
        token = None
        try:
            if event.is_set():
                raise _Cancelled()
            if not partial.resolve().is_relative_to(root):
                raise ValueError("temporary download directory escapes model storage")
            partial.parent.mkdir(parents=True, exist_ok=True)
            self._update(id_, status="downloading")
            headers = {"User-Agent": "YPuddinTrainStudio", "Accept-Encoding": "identity"}
            # Third-party mirrors are anonymous. Sensitive headers are also stripped on redirects.
            if row["mirror"] == "official":
                token = self.credentials.token(row["provider"])
                if token:
                    headers["Cookie" if row["provider"] == "modelscope" else "Authorization"] = (
                        f"m_session_id={token}" if row["provider"] == "modelscope" else f"Bearer {token}"
                    )
            done, last = 0, time.monotonic()
            bundle = row["kind"] == "tagger"
            verified = bool(row.get("sha256"))
            files = (
                TAGGER_FILES
                if bundle
                else {row["filename"]: {"size": row.get("expected_size"), "sha256": row.get("sha256")}}
            )
            if bundle:
                self._update(id_, total_bytes=sum(meta["size"] for meta in files.values()))
            for filename, meta in files.items():
                source = row["source_url"].rsplit("/", 1)[0] + "/" + filename if bundle else row["source_url"]
                request = urllib.request.Request(source, headers=headers)
                received, digest = 0, hashlib.sha256()
                with (
                    self.opener.open(request, timeout=15) as response,
                    (partial.parent / filename).open("xb") as file,
                ):
                    if response.status != 200:
                        raise ValueError(f"unexpected download HTTP status {response.status}")
                    total = (
                        int(response.headers["Content-Length"])
                        if response.headers.get("Content-Length")
                        else None
                    )
                    if not bundle:
                        self._update(id_, total_bytes=total)
                    while True:
                        if event.is_set():
                            raise _Cancelled()
                        chunk = response.read(256 * 1024)
                        if not chunk:
                            break
                        file.write(chunk)
                        received += len(chunk)
                        done += len(chunk)
                        if bundle or verified:
                            digest.update(chunk)
                            if received > meta["size"]:
                                raise ValueError("catalog file exceeds its verified size")
                        if time.monotonic() - last >= 0.4:
                            self._update(id_, downloaded_bytes=done)
                            last = time.monotonic()
                    if not received or total is not None and received != total:
                        raise ValueError(f"incomplete download: received {received} of {total} bytes")
                    if (bundle or verified) and (
                        received != meta["size"] or digest.hexdigest() != meta["sha256"]
                    ):
                        raise ValueError(f"catalog integrity check failed for {filename}")
                    file.flush()
                    os.fsync(file.fileno())
            self._update(id_, downloaded_bytes=done)
            from safetensors import safe_open

            # Header/offset validation mmaps the file; no GPU allocation or full tensor loading.
            if not bundle:
                with safe_open(partial, framework="pt", device="cpu") as weights:
                    keys = weights.keys()
                    if not keys or all(k.startswith(("lora_", "lycoris_")) for k in keys):
                        raise ValueError("expected a base model component, not empty or adapter weights")
                    check_component(weights, row["family"], row["kind"])
            with self.lock:
                if event.is_set() or self.closed:
                    raise _Cancelled()
                target.parent.parent.mkdir(parents=True, exist_ok=True)
                if not target.resolve().is_relative_to(root):
                    raise ValueError("model target escapes the configured model directory")
                # Publish the complete one-file directory atomically. Directory rename works on
                # Windows/NTFS and exFAT without hardlinks; an occupied target is never replaced.
                if target.parent.exists():
                    raise FileExistsError(f"download destination already exists: {target.parent}")
                partial.parent.rename(target.parent)
                published = True
                if verified:
                    from .model_recommendations import remember_verified_file

                    remember_verified_file(self.context, target, row["sha256"])
                from .routes_core import ModelBody, add_model

                asset = add_model(
                    ModelBody(
                        family=row["family"],
                        kind=row["kind"],
                        path=str(target.parent if bundle else target),
                        dtype=row["dtype"],
                        is_default=row["is_default"],
                    ),
                    self.context,
                )
                self._update(id_, status="completed", model_id=asset["id"], finished_at=now())
        except _Cancelled:
            self._update(id_, status="cancelled", finished_at=now())
        except Exception as error:
            if published:
                target.unlink(missing_ok=True)
                if row["kind"] == "tagger":
                    (target.parent / "selected_tags.csv").unlink(missing_ok=True)
                target.parent.rmdir()
            message = str(error)
            if token:
                message = message.replace(token, "[redacted]")
            if isinstance(error, urllib.error.HTTPError):
                message = f"HTTP {error.code}: check the repository/file and your network access."
                if error.code in (401, 403):
                    provider = "ModelScope" if row["provider"] == "modelscope" else "Hugging Face"
                    message += f" For gated/private files, accept the repository license on {provider}, then save its access token in Settings → Access keys and retry."
                    if row["mirror"] != "official":
                        message += " Mirrors are anonymous; switch to the official source for authenticated downloads."
                elif error.code == 429:
                    message += " Rate limited: save the official source token or wait before retrying."
            self._update(
                id_,
                status="cancelled" if event.is_set() else "failed",
                error=None if event.is_set() else message,
                finished_at=now(),
            )
        finally:
            partial.unlink(missing_ok=True)
            if row["kind"] == "tagger":
                (partial.parent / "selected_tags.csv").unlink(missing_ok=True)
            if partial.parent.exists():
                partial.parent.rmdir()

    def close(self) -> None:
        with self.lock:
            self.closed = True
            for event in self.cancelled.values():
                event.set()
        self.pool.shutdown(wait=True)
