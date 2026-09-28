"""Downloads of the reviewed tagging and mask-detection models (``model_catalog.VISION_MODELS``).

Each model lives in ``<models_dir>/<folder>/`` (``tagger/<family>/<version>`` or
``mask/<kind>/<version>``) as the files its catalog entry names. A file is accepted only at its
pinned size and SHA-256, whichever source served it, so a mirror can never hand over different
bytes. ``.verified.json`` records the hashes checked at download time. Models downloaded before this
layout, in ``<models_dir>/vision/<id>/``, move to their folder the first time the catalog is read.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from .db import now
from .errors import ApiError, NotFound
from .model_catalog import VISION_MODELS

log = logging.getLogger(__name__)

Source = Literal["huggingface", "modelscope"]
ACTIVE = ("queued", "downloading", "verifying")
MARKER = ".verified.json"


def file_url(entry: dict, repo_path: str, source: Source) -> str:
    if source == "modelscope":
        if not entry.get("modelscope"):
            raise ApiError("this model has no ModelScope copy; use Hugging Face", code="vision.source")
        query = urllib.parse.urlencode({"Revision": "master", "FilePath": repo_path})
        return f"https://modelscope.cn/api/v1/models/{entry['modelscope']}/repo?{query}"
    return f"https://huggingface.co/{entry['repo']}/resolve/{entry['revision']}/{urllib.parse.quote(repo_path, safe='/')}"


class VisionModels:
    def __init__(self, context: Any, credentials: Any = None):
        self.context = context
        self.credentials = credentials
        self.lock = threading.RLock()
        self.tasks: dict[str, dict[str, Any]] = {}
        self.cancelled: dict[str, threading.Event] = {}
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision-download")
        self.opener = None  # test transport; live requests follow the current proxy policy

    # ----------------------------------------------------------------- storage
    def root(self) -> Path:
        return Path(self.context.settings()["paths"]["models_dir"]).expanduser().resolve()

    def directory(self, model_id: str) -> Path:
        if model_id not in VISION_MODELS:
            raise NotFound("unknown model", code="vision.model")
        return self.root() / VISION_MODELS[model_id]["folder"]

    def migrate(self) -> None:
        """Move models from the former ``vision/<id>`` folders into their own folders."""
        legacy = self.root() / "vision"
        if not legacy.is_dir():
            return
        with self.lock:
            for model_id in VISION_MODELS:
                old, new = legacy / model_id, self.directory(model_id)
                if old.is_dir() and not new.exists():
                    new.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        old.rename(new)
                    except OSError as error:
                        log.warning("could not move %s to %s: %s", old, new, error)
            for leftover in legacy.glob(".*.partial"):
                shutil.rmtree(leftover, ignore_errors=True)
            try:
                legacy.rmdir()
            except OSError:
                pass  # other files remain; leave them to the user

    def ready(self, model_id: str) -> bool:
        folder = self.directory(model_id)
        try:
            verified = json.loads((folder / MARKER).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        for name, size, sha256 in VISION_MODELS[model_id]["files"].values():
            path = folder / name
            if not path.is_file() or path.stat().st_size != size or verified.get(name) != sha256:
                return False
        return True

    def files(self, model_id: str) -> dict[str, Path]:
        """Local paths by name (model.onnx and the label table); raises when the model is not downloaded."""
        if not self.ready(model_id):
            label = VISION_MODELS[model_id]["label"]
            raise ApiError(f"{label} is not downloaded", code="vision.model_missing", status=409)
        folder = self.directory(model_id)
        return {name: folder / name for name, _, _ in VISION_MODELS[model_id]["files"].values()}

    # ----------------------------------------------------------------- catalog
    def catalog(self) -> list[dict[str, Any]]:
        self.migrate()
        token = bool(self.credentials and self.credentials.token("huggingface"))
        rows = []
        for model_id, entry in VISION_MODELS.items():
            task = self.tasks.get(model_id) or {}
            rows.append(
                {
                    "id": model_id,
                    "role": entry["role"],
                    "family": entry["family"],
                    "label": entry["label"],
                    "repo": entry["repo"],
                    "revision": entry["revision"],
                    "license": entry["license"],
                    "size": sum(size for _, size, _ in entry["files"].values()),
                    "recommended": bool(entry.get("recommended")),
                    "categories": list(entry.get("categories", ())),
                    "thresholds": entry.get("thresholds"),
                    "token_required": bool(entry.get("token_required")),
                    "token_configured": token,
                    "sources": ["huggingface", *(["modelscope"] if entry.get("modelscope") else [])],
                    "ready": self.ready(model_id),
                    "path": str(self.directory(model_id)),
                    "download": {
                        key: task.get(key)
                        for key in (
                            "status",
                            "source",
                            "downloaded_bytes",
                            "total_bytes",
                            "bytes_per_second",
                            "error",
                        )
                    }
                    if task
                    else None,
                }
            )
        return rows

    # ----------------------------------------------------------------- downloads
    def start(self, model_id: str, source: Source) -> dict[str, Any]:
        entry = VISION_MODELS.get(model_id)
        if entry is None:
            raise NotFound("unknown model", code="vision.model")
        if source == "modelscope" and not entry.get("modelscope"):
            raise ApiError("this model has no ModelScope copy; use Hugging Face", code="vision.source")
        with self.lock:
            current = self.tasks.get(model_id)
            if current and current["status"] in ACTIVE:
                raise ApiError("this model is already downloading", code="vision.busy", status=409)
            if self.ready(model_id):
                raise ApiError("this model is already downloaded", code="vision.ready", status=409)
            if entry.get("token_required") and not (
                self.credentials and self.credentials.token("huggingface")
            ):
                raise ApiError(
                    f"{entry['label']} needs a Hugging Face access token; save one under Settings → Access keys",
                    code="vision.token_required",
                    status=409,
                )
            self.tasks[model_id] = {
                "status": "queued",
                "source": source,
                "downloaded_bytes": 0,
                "total_bytes": sum(size for _, size, _ in entry["files"].values()),
                "bytes_per_second": None,
                "error": None,
                "started_at": now(),
            }
            self.cancelled[model_id] = threading.Event()
        self.pool.submit(self._run, model_id)
        return next(row for row in self.catalog() if row["id"] == model_id)

    def cancel(self, model_id: str) -> dict[str, Any]:
        with self.lock:
            event = self.cancelled.get(model_id)
            if event is None or (self.tasks.get(model_id) or {}).get("status") not in ACTIVE:
                raise ApiError("this model is not downloading", code="vision.idle", status=409)
            event.set()
        return next(row for row in self.catalog() if row["id"] == model_id)

    def remove(self, model_id: str) -> dict[str, Any]:
        with self.lock:
            if (self.tasks.get(model_id) or {}).get("status") in ACTIVE:
                raise ApiError("cancel the download first", code="vision.busy", status=409)
            folder = self.directory(model_id)
            if folder.exists():
                shutil.rmtree(folder)
            self.tasks.pop(model_id, None)
        return next(row for row in self.catalog() if row["id"] == model_id)

    def _update(self, model_id: str, **values: Any) -> None:
        with self.lock:
            self.tasks[model_id].update(values)

    def _run(self, model_id: str) -> None:
        from .model_downloads import _Redirect
        from .network import ProxyPolicy

        entry = VISION_MODELS[model_id]
        task = self.tasks[model_id]
        event = self.cancelled[model_id]
        folder = self.directory(model_id)
        stage = folder.with_name(f".{folder.name}.partial")
        policy = ProxyPolicy.from_context(self.context)
        done, last, last_bytes, rate = 0, time.monotonic(), 0, 0.0
        try:
            shutil.rmtree(stage, ignore_errors=True)
            stage.mkdir(parents=True)
            self._update(model_id, status="downloading")
            verified = {}
            for repo_path, (name, size, sha256) in entry["files"].items():
                headers = {"User-Agent": "YPuddinTrainStudio", "Accept-Encoding": "identity"}
                provider = "modelscope" if task["source"] == "modelscope" else "huggingface"
                token = self.credentials.token(provider) if self.credentials else None
                if token:
                    headers["Cookie" if provider == "modelscope" else "Authorization"] = (
                        f"m_session_id={token}" if provider == "modelscope" else f"Bearer {token}"
                    )
                request = urllib.request.Request(file_url(entry, repo_path, task["source"]), headers=headers)
                opener = self.opener or policy.opener(_Redirect())
                digest, received = hashlib.sha256(), 0
                with opener.open(request, timeout=20) as response, (stage / name).open("xb") as out:
                    if response.status != 200:
                        raise ValueError(f"unexpected HTTP status {response.status}")
                    while True:
                        if event.is_set():
                            raise InterruptedError()
                        chunk = response.read(256 * 1024)
                        if not chunk:
                            break
                        received += len(chunk)
                        if received > size:
                            raise ValueError(f"{name} is larger than the reviewed file")
                        out.write(chunk)
                        digest.update(chunk)
                        done += len(chunk)
                        current = time.monotonic()
                        if current - last >= 0.4:
                            measured = (done - last_bytes) / max(current - last, 0.001)
                            rate = measured if not rate else 0.3 * measured + 0.7 * rate
                            self._update(model_id, downloaded_bytes=done, bytes_per_second=rate)
                            last, last_bytes = current, done
                if received != size or digest.hexdigest() != sha256:
                    raise ValueError(
                        f"{name} does not match the reviewed file (size {received}, expected {size}); "
                        "try another source"
                    )
                verified[name] = sha256
            self._update(model_id, status="verifying", downloaded_bytes=done)
            (stage / MARKER).write_text(json.dumps(verified, indent=2), encoding="utf-8")
            if folder.exists():
                shutil.rmtree(folder)
            stage.rename(folder)
            self._update(model_id, status="completed", bytes_per_second=None, finished_at=now())
        except InterruptedError:
            self._update(model_id, status="cancelled", bytes_per_second=None, finished_at=now())
        except Exception as error:  # noqa: BLE001 - the task records every failure
            message = policy.redact(error)
            if isinstance(error, urllib.error.HTTPError):
                message = (
                    f"HTTP {error.code}: Hugging Face refused the download; accept the model's terms on its page "
                    "with the account of the saved access token"
                    if error.code in (401, 403) and entry.get("token_required")
                    else f"HTTP {error.code}: the file could not be downloaded from this source"
                )
            log.warning("vision model download %s failed: %s", model_id, message)
            self._update(model_id, status="failed", error=message, bytes_per_second=None, finished_at=now())
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def close(self) -> None:
        with self.lock:
            for event in self.cancelled.values():
                event.set()
        self.pool.shutdown(wait=True)
