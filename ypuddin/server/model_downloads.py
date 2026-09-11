"""Managed single-file Hugging Face downloads, with durable status and atomic registration."""

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

from pydantic import BaseModel, ConfigDict, model_validator

from .context import ServiceContext
from .db import new_id, now
from .errors import ApiError, Conflict, NotFound

ACTIVE = {"queued", "downloading"}


class ModelDownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    family: Literal["anima", "krea2"]
    kind: Literal["dit", "text_encoder", "vae"]
    url: str | None = None
    repo_id: str | None = None
    filename: str | None = None
    revision: str = "main"
    dtype: Literal["bf16", "fp16", "fp32", "fp8"] | None = None
    is_default: bool = True

    @model_validator(mode="after")
    def source(self):
        resolve_source(self)
        return self


class ModelDownload(BaseModel):
    id: str
    family: str
    kind: str
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


def resolve_source(body: ModelDownloadRequest) -> tuple[str, str]:
    """Accept only HF file URLs, not arbitrary servers, user credentials or query tokens."""
    repo, filename, revision = body.repo_id, body.filename, body.revision
    if body.url:
        if repo or filename:
            raise ValueError("use a Hugging Face URL or repo_id + filename, not both")
        parsed = urllib.parse.urlsplit(body.url.strip())
        if parsed.scheme != "https" or parsed.netloc.lower() != "huggingface.co":
            raise ValueError("only https://huggingface.co file URLs are supported")
        parts = urllib.parse.unquote(parsed.path).strip("/").split("/")
        if len(parts) < 5 or parts[2] not in {"resolve", "blob"}:
            raise ValueError("paste a Hugging Face file URL containing /resolve/ or /blob/")
        repo, revision, filename = "/".join(parts[:2]), parts[3], "/".join(parts[4:])
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
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(filename, safe='/')}"
    return url, path.name


class _Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise ValueError("download redirected to a non-HTTPS destination")
        if redirected and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            redirected.remove_header("Authorization")
        return redirected


class _Cancelled(Exception):
    pass


def check_component(weights: Any, family: str, kind: str) -> None:
    """Reject recognisable mismatches; complete architectural validation belongs to the loader."""
    names = {}
    for key in weights.keys():
        clean = key
        for prefix in ("model.diffusion_model.", "diffusion_model.", "net."):
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
                break
        names[clean] = key
    found_family, found_kind = None, None
    if "x_embedder.proj.1.weight" in names:
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
    if found_kind is not None and (found_kind != kind or found_family not in (None, family)):
        raise ValueError(f"file looks like {found_family or 'shared'} {found_kind}, not {family} {kind}")


class ModelDownloads:
    def __init__(self, context: ServiceContext):
        self.context = context
        self.lock = threading.RLock()
        self.tasks: dict[str, dict[str, Any]] = {
            row["id"]: row for row in context.db.get_kv("model_downloads", [])
        }
        self.cancelled: dict[str, threading.Event] = {}
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="model-download")
        self.opener = urllib.request.build_opener(_Redirect())
        self.closed = False
        for row in self.tasks.values():
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

    def start(self, body: ModelDownloadRequest) -> dict[str, Any]:
        source, filename = resolve_source(body)
        root = Path(self.context.settings()["paths"]["models_dir"]).resolve()
        if not self.context.is_allowed(root):
            raise ApiError("model directory is outside allowed storage roots", status=403)
        folder = root / body.family / body.kind / hashlib.sha256(source.encode()).hexdigest()[:12]
        target = folder / filename
        with self.lock:
            if self.closed:
                raise Conflict("download service is stopping")
            if target.exists() or target.is_symlink():
                raise Conflict(
                    "target file already exists; register the existing path instead",
                    code="download.exists",
                    details={"path": str(target)},
                )
            if any(
                row["target_path"] == str(target) and row["status"] in ACTIVE for row in self.tasks.values()
            ):
                raise Conflict("this model file is already downloading", code="download.active")
            if not target.resolve().is_relative_to(root):
                raise ApiError("model target escapes the configured model directory", code="download.path")
            id_ = new_id("dl")
            row = ModelDownload(
                id=id_,
                family=body.family,
                kind=body.kind,
                source_url=source,
                filename=filename,
                target_path=str(target),
                status="queued",
                created_at=now(),
                dtype=body.dtype,
                is_default=body.is_default,
            ).model_dump()
            self.tasks[id_] = row
            self.cancelled[id_] = threading.Event()
            self._persist()
            self.pool.submit(self._run, id_, root)
            return dict(row)

    def cancel(self, id_: str) -> dict[str, Any]:
        with self.lock:
            if id_ not in self.tasks:
                raise NotFound("download not found", code="download.not_found")
            if self.tasks[id_]["status"] in ACTIVE:
                self.cancelled[id_].set()
            return dict(self.tasks[id_])

    def _run(self, id_: str, root: Path) -> None:
        row = self.tasks[id_]
        event = self.cancelled[id_]
        partial = root / ".downloads" / id_ / row["filename"]
        target = Path(row["target_path"])
        published = False
        try:
            if event.is_set():
                raise _Cancelled()
            if not partial.resolve().is_relative_to(root):
                raise ValueError("temporary download directory escapes model storage")
            partial.parent.mkdir(parents=True, exist_ok=True)
            self._update(id_, status="downloading")
            headers = {"User-Agent": "YPuddinTrainStudio/0.1", "Accept-Encoding": "identity"}
            # Tokens stay in the HF credential store/environment; never persist in status or URLs.
            try:
                from huggingface_hub import get_token

                token = get_token()
                if token:
                    headers["Authorization"] = f"Bearer {token}"
            except ImportError:
                pass
            request = urllib.request.Request(row["source_url"], headers=headers)
            done, last = 0, time.monotonic()
            with self.opener.open(request, timeout=15) as response, partial.open("xb") as file:
                if response.status != 200:
                    raise ValueError(f"unexpected download HTTP status {response.status}")
                total = (
                    int(response.headers["Content-Length"])
                    if response.headers.get("Content-Length")
                    else None
                )
                self._update(id_, total_bytes=total)
                while True:
                    if event.is_set():
                        raise _Cancelled()
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    file.write(chunk)
                    done += len(chunk)
                    if time.monotonic() - last >= 0.4:
                        self._update(id_, downloaded_bytes=done)
                        last = time.monotonic()
                if not done or total is not None and done != total:
                    raise ValueError(f"incomplete download: received {done} of {total} bytes")
                file.flush()
                os.fsync(file.fileno())
            self._update(id_, downloaded_bytes=done)
            from safetensors import safe_open

            # Header/offset validation mmaps the file; no GPU allocation or full tensor loading.
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
                from .routes_core import ModelBody, add_model

                asset = add_model(
                    ModelBody(
                        family=row["family"],
                        kind=row["kind"],
                        path=str(target),
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
                target.parent.rmdir()
            message = str(error)
            if isinstance(error, urllib.error.HTTPError):
                message = f"HTTP {error.code}: check the repository/file and your network access."
                if error.code in (401, 403):
                    message += " For gated/private files, accept the license and sign in with hf auth login on the server."
            self._update(
                id_,
                status="cancelled" if event.is_set() else "failed",
                error=None if event.is_set() else message,
                finished_at=now(),
            )
        finally:
            partial.unlink(missing_ok=True)
            if partial.parent.exists():
                partial.parent.rmdir()

    def close(self) -> None:
        with self.lock:
            self.closed = True
            for event in self.cancelled.values():
                event.set()
        self.pool.shutdown(wait=True)
