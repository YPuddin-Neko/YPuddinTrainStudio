"""Transactional class-prior batches: isolated base-model inference or bounded booru collection."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

from ypuddin.config import CaptionConfig, ModelConfig
from ypuddin.data.index import iter_images

from .booru import (
    MEDIA_EXTS,
    SITES,
    BooruClient,
    Cancelled,
    Downloads,
    Post,
    Redirect,
    normalize,
    user_agent,
)
from .db import new_id, now
from .environment import maintenance_blocked
from .errors import ApiError, NotFound
from .hardware import gpu_info
from .model_credentials import ModelCredentials

TERMINAL = {"completed", "failed", "cancelled"}
RESERVATION = "regularization.reservation"
MAX_FILE_BYTES = 40 * 1024 * 1024
MAX_BATCH_BYTES = 1024 * 1024 * 1024
MAX_PIXELS = 16_777_216
OWNER_FILE = ".regularization-owner"


def image_identity(image: Image.Image) -> str:
    return hashlib.sha256(f"{image.width}x{image.height}:".encode() + image.tobytes()).hexdigest()


def decoded_image(data: bytes | Path) -> Image.Image:
    with Image.open(io.BytesIO(data) if isinstance(data, bytes) else data) as image:
        if (
            image.width * image.height > MAX_PIXELS
            or max(image.size) > 8192
            or image.width < 16
            or image.height < 16
        ):
            raise ValueError("Image dimensions exceed the regularization limits")
        image.seek(0)
        oriented = ImageOps.exif_transpose(image).convert("RGBA")
        background = Image.new("RGBA", oriented.size, "white")
        return Image.alpha_composite(background, oriented).convert("RGB")


class _Batch(Downloads):
    """Posts downloaded in parallel into one batch, keeping only new, safe, distinct images."""

    over_limit = ("Regularization download exceeded its byte limit", "regularization.too_large")

    def __init__(self, manager, oid, client, output, count, excluded, taken, cancelled):
        super().__init__(client, count, max_bytes=MAX_BATCH_BYTES, cancelled=cancelled)
        self.manager, self.oid, self.output = manager, oid, output
        self.excluded, self.taken = set(excluded), taken
        self.identities: set[str] = set()
        self.manifest: list[dict] = []

    def wanted(self, post: Post) -> bool:
        return (
            post.safe
            and post.ext in MEDIA_EXTS
            and bool(post.tags)
            and post.id not in self.seen
            and (self.source, post.id) not in self.taken
            and (None, post.id) not in self.taken
            and not self.excluded & post.all_tags
        )

    def read(self, data):
        return decoded_image(data)

    def progress(self):
        self.manager._update(self.oid, phase="collecting", done=self.done)

    def keep(self, post, image, query):
        digest = image_identity(image)
        if digest in self.identities:
            self.manager._update(self.oid, duplicates=self.manager._row(self.oid)["duplicates"] + 1)
            return False
        self.identities.add(digest)
        name = f"{self.source}_{post.id}.png"
        image.save(self.output / name)
        (self.output / name).with_suffix(".txt").write_text(
            ", ".join(tag.replace("_", " ") for tag in post.tags), encoding="utf-8"
        )
        self.manifest.append(
            {
                "file": name,
                "provider": self.source,
                "post_id": post.id,
                "page": post.page,
                "source": post.source,
                "query": query,
                "width": image.width,
                "height": image.height,
                "pixel_sha256": digest,
            }
        )
        return True

    def finish(self):
        if self.skipped:
            self.manager._update(
                self.oid, message=f"Skipped {self.skipped} files that were not usable images"
            )
        if not self.done:
            raise ApiError(
                "No new matching safe images were found. Change the search tags or excluded tags; no batch was added.",
                code="regularization.insufficient",
            )
        if self.done < self.count:
            # What was found is kept: asking again would only download the same pictures.
            self.manager._update(
                self.oid, message=f"Found {self.done} of {self.count} matching images; adding them"
            )
        (self.output / "manifest.json").write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )


class RegularizationManager:
    def __init__(self, context, *, opener=None, credentials=None):
        self.c = context
        self.credentials = credentials or ModelCredentials(context.data_root)
        self.opener = opener  # Test seam; production always uses the provider-restricted opener.
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="regularization")
        self.cancel_events: dict[str, threading.Event] = {}
        self.preview_clients: dict[tuple[str, str, str], BooruClient] = {}
        self.preview_lock = threading.Lock()
        self.processes: dict[str, subprocess.Popen] = {}
        self.closed = False
        self.c.db.execute("""CREATE TABLE IF NOT EXISTS regularization_operations (
            id TEXT PRIMARY KEY, project_id TEXT NOT NULL, version_id TEXT NOT NULL,
            source TEXT NOT NULL, status TEXT NOT NULL, phase TEXT NOT NULL,
            done INTEGER NOT NULL DEFAULT 0, total INTEGER NOT NULL, request_json TEXT NOT NULL,
            logs_json TEXT NOT NULL DEFAULT '[]', error TEXT, dataset_id TEXT, path TEXT,
            images INTEGER NOT NULL DEFAULT 0, duplicates INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL, finished_at REAL, worker_pid INTEGER
        )""")
        self._recover()

    def _paths(self, row):
        root = self.c.reg_dir(row["project_id"], row["version_id"])
        if root.is_symlink():
            raise ApiError("Regularization storage cannot be a symbolic link", code="regularization.path")
        return root, root / f".staging-{row['id']}", root / row["id"]

    def _recover(self):
        reservation = self.c.db.get_kv(RESERVATION, {})
        rows = self.c.db.fetchall(
            "SELECT * FROM regularization_operations WHERE status NOT IN ('completed','failed','cancelled') OR id=?",
            (reservation.get("id", ""),),
        )
        blocked = None
        for row in rows:
            # Completion commits only after the child exited and the source indexed.
            # A crash before finally clears its reservation must never undo that batch.
            if row["status"] == "completed":
                continue
            try:
                _, staging, _ = self._paths(row)
                if row["worker_pid"]:
                    import psutil

                    try:
                        process = psutil.Process(row["worker_pid"])
                        command = process.cmdline()
                        if (
                            "ypuddin.server.regularization_worker" in command
                            and str(staging / "request.json") in command
                        ):
                            process.terminate()
                            try:
                                process.wait(timeout=3)
                            except psutil.TimeoutExpired:
                                process.kill()
                                process.wait(timeout=3)
                    except psutil.NoSuchProcess:
                        pass
                self._rollback(row)
                error = "Studio stopped during regularization. The unpublished batch was discarded; start a new task."
            except Exception:
                error = "Studio stopped during regularization. Recovery requires attention; original data was preserved."
                if row["source"] == "ai":
                    blocked = {"id": row["id"], "recovery_required": True}
            self.c.db.update(
                "regularization_operations",
                row["id"],
                {"status": "failed", "phase": "failed", "error": error, "finished_at": now()},
            )
        self.c.db.set_kv(RESERVATION, blocked or {})

    def close(self):
        with self.c.db.lock:
            self.closed = True
            for event in self.cancel_events.values():
                event.set()
        self.pool.shutdown(wait=True)

    def _row(self, oid):
        row = self.c.db.fetchone("SELECT * FROM regularization_operations WHERE id=?", (oid,))
        if not row:
            raise NotFound("Regularization task not found", code="regularization.not_found")
        return row

    def get(self, oid):
        row = self._row(oid)
        self.c.resolve_version(row["project_id"], row["version_id"])
        return {
            **{
                key: value
                for key, value in row.items()
                if key not in {"request_json", "logs_json", "worker_pid"}
            },
            "method": "ai" if row["source"] == "ai" else "web",
            "provider": None if row["source"] == "ai" else row["source"],
            "logs": json.loads(row["logs_json"]),
            "can_cancel": row["status"] not in TERMINAL and row["phase"] != "publishing",
        }

    def snapshot(self, pid, vid):
        self.c.resolve_version(pid, vid)
        root = self.c.reg_dir(pid, vid)
        operations = [
            self.get(row["id"])
            for row in self.c.db.fetchall(
                "SELECT id FROM regularization_operations WHERE project_id=? AND version_id=? ORDER BY created_at DESC LIMIT 50",
                (pid, vid),
            )
        ]
        count = (
            sum(
                1
                for image in iter_images(root)
                if not any(part.startswith(".") for part in image.relative_to(root).parts)
            )
            if root.exists()
            else 0
        )
        return {"path": str(root), "images": count, "operations": operations}

    def _update(self, oid, *, message=None, **fields):
        with self.c.db.lock:
            if message:
                fields["logs_json"] = json.dumps([*json.loads(self._row(oid)["logs_json"]), message][-100:])
            self.c.db.update("regularization_operations", oid, fields)
        self.c.bus.publish("regularization.changed", self.get(oid))

    def plan(self, pid, vid, request):
        from .regularization_plan import training_plan
        from .routes_work import get_project_config

        if request.source != "ai" or request.prompt_source != "training_tags":
            raise ApiError("请先选择按训练标签生成", status=422, code="regularization.source")
        return training_plan(self.c, pid, vid, request, get_project_config(pid, self.c, vid))[0]

    @staticmethod
    def _search_tags(request):
        tags = request.prompt.split()
        if not tags or len(tags) > 12 or any(not re.fullmatch(r"[\w()\-]+", tag) for tag in tags):
            raise ApiError(
                "Use up to 12 plain search tags; ratings and other query operators are fixed by Studio",
                status=422,
                code="regularization.query",
            )
        if any(not re.fullmatch(r"[\w()\- ]{1,100}", tag) for tag in request.excluded_tags):
            raise ApiError("Excluded search tags must be plain tags", status=422, code="regularization.query")
        return tags

    def _site_client(self, source):
        """A long-lived client per site and account for previews, so their requests share one pace."""
        credentials = self.credentials.site(source)
        key = (source, *credentials)
        with self.preview_lock:
            client = self.preview_clients.get(key)
            if client is None:
                self.preview_clients = {
                    other: value for other, value in self.preview_clients.items() if other[0] != source
                }
                client = self.preview_clients[key] = self.client(source, credentials)
        return client

    def estimate(self, pid, vid, request):
        """How many safe posts a manual search finds, and how it is sent within the account's limit."""
        self.c.resolve_version(pid, vid)
        if request.source == "ai":
            raise ApiError("Only site searches can be estimated", status=422, code="regularization.source")
        tags = self._search_tags(request)
        client = self._site_client(request.source)
        excluded = list(dict.fromkeys(tag for tag in map(normalize, request.excluded_tags) if tag))
        terms, local = client.query(tags, excluded)
        return {
            "source": request.source,
            "count": client.count(terms),
            "terms": terms,
            "local_exclusions": local,
            "tag_limit": client.tag_limit(),
        }

    def match_plan(self, pid, vid, request):
        """What a batch that follows the training set would search for, from the captions alone;
        only the Danbooru account level is read from the site."""
        from .regularization_match import build_profile, plan
        from .routes_work import get_project_config

        if request.source == "ai":
            raise ApiError(
                "Only site collection follows the training tags", status=422, code="regularization.source"
            )
        profile = build_profile(
            self.c, pid, vid, get_project_config(pid, self.c, vid), request.source_ids, request.excluded_tags
        )
        limit, known = None, True
        if request.source == "danbooru":
            try:
                limit = self._site_client(request.source).tag_limit()
            except ApiError:
                limit, known = 2, False
        return plan(profile, tag_limit=limit, tag_limit_known=known, source=request.source)

    def start(self, pid, vid, request):
        from .routes_work import get_project_config

        prompts = [line.strip() for line in request.prompt.splitlines() if line.strip()]
        excluded = {tag.strip().replace("_", " ").lower() for tag in request.excluded_tags}
        prompts = [
            ", ".join(
                tag.strip()
                for tag in line.split(",")
                if tag.strip().replace("_", " ").lower() not in excluded
            )
            for line in prompts
        ]
        prompts = [line for line in prompts if line]
        if not prompts and request.prompt_source == "manual":
            raise ApiError(
                "Enter a non-empty class prompt or search query", status=422, code="regularization.prompt"
            )
        credentials = ("", "")
        if request.source != "ai":
            if request.prompt_source == "manual":
                self._search_tags(request)
            # Legacy API clients may still supply one-use credentials. Never mix a
            # partial override with an account/key taken from the central store.
            if request.user_id or request.username or request.api_key.get_secret_value():
                credentials = (request.user_id or request.username, request.api_key.get_secret_value())
                if not all(credentials):
                    raise ApiError(
                        "Supply both site account and API key, or configure access keys in Settings",
                        status=422,
                        code="regularization.credentials",
                    )
            else:
                credentials = self.credentials.site(request.source)
            if request.source in {"gelbooru", "rule34"} and not (credentials[0].isdigit() and credentials[1]):
                raise ApiError(
                    f"Configure the {SITES[request.source].label} user ID and API key in Settings → Access keys",
                    status=422,
                    code="regularization.credentials",
                )
        oid = new_id("reg")
        lease = self.c.versions.mutation(pid, vid)
        with self.c.db.lock:
            if self.closed:
                raise ApiError("Regularization manager is stopping", status=503)
            if request.source == "ai":
                running = self.c.db.fetchone(
                    "SELECT id FROM jobs WHERE status IN ('running','pausing','cancelling')"
                )
                if (
                    maintenance_blocked(self.c.db)
                    or running
                    or any(p.poll() is None for p in self.c.supervisor._procs.values())
                ):
                    raise ApiError(
                        "Wait for training, another AI task or environment maintenance to finish",
                        status=409,
                        code="regularization.busy",
                    )
            version = lease.__enter__()
            try:
                config = get_project_config(pid, self.c, version["id"])
                _, staging, final = self._paths({"id": oid, "project_id": pid, "version_id": version["id"]})
                if staging.exists() or final.exists():
                    raise ApiError(
                        "This batch directory already exists; start a new task",
                        status=409,
                        code="regularization.directory_exists",
                    )
                payload = request.model_dump(mode="json") | {
                    "prompts": prompts,
                    "ownership_token": uuid.uuid4().hex,
                }
                if request.prompt_source == "training_tags" and request.source == "ai":
                    from .regularization_plan import training_plan

                    plan, entries = training_plan(self.c, pid, version["id"], request, config)
                    if request.plan_signature and request.plan_signature != plan["signature"]:
                        raise ApiError(
                            "训练来源或已有正则图已变化，请重新预览计划",
                            status=409,
                            code="regularization.plan_stale",
                        )
                    if not entries:
                        raise ApiError(
                            "当前范围没有待生成图片，请检查已有来源、标签与排除词",
                            status=422,
                            code="regularization.empty_plan",
                        )
                    payload.update(
                        prompts=[entry["prompt"] for entry in entries], entries=entries, count=len(entries)
                    )
                if request.source == "ai":
                    from ypuddin.models import get_family

                    model = ModelConfig.model_validate(config.get("model", {}))
                    # Prior generation uses the base model's SDPA path, independent of training adapters.
                    model.attention = "auto"
                    family = get_family(model.family)
                    issues = family.validate_config(model)
                    if issues:
                        raise ApiError("; ".join(issues), status=422, code="regularization.models")
                    if request.width % family.spec.latent.align or request.height % family.spec.latent.align:
                        raise ApiError(
                            f"Width and height must be multiples of {family.spec.latent.align}",
                            status=422,
                            code="regularization.dimensions",
                        )
                    for field in (
                        "dit_path",
                        "text_encoder_path",
                        "text_encoder_2_path",
                        "vae_path",
                        "tokenizer_path",
                    ):
                        path = getattr(model, field)
                        if path and not self.c.is_allowed(Path(path)):
                            raise ApiError(
                                "Model path is outside permitted roots",
                                status=403,
                                code="regularization.path",
                            )
                    devices = gpu_info() if model.family != "toy" else []
                    payload.update(
                        model=model.model_dump(mode="json"),
                        sampling=config.get("sampling", {}),
                        device=devices[0]["device"] if devices else "cpu",
                        fingerprint_cache=str(self.c.cache_dir(pid, version["id"]) / "fingerprints"),
                    )
                    self.c.db.set_kv(RESERVATION, {"id": oid})
                self.c.db.insert(
                    "regularization_operations",
                    {
                        "id": oid,
                        "project_id": pid,
                        "version_id": version["id"],
                        "source": request.source,
                        "status": "queued",
                        "phase": "queued",
                        "total": payload["count"],
                        "request_json": json.dumps(payload),
                        "created_at": now(),
                    },
                )
                event = threading.Event()
                self.cancel_events[oid] = event
                self.pool.submit(self._run, oid, payload, credentials, event, lease)
            except BaseException:
                self.cancel_events.pop(oid, None)
                if self.c.db.fetchone("SELECT id FROM regularization_operations WHERE id=?", (oid,)):
                    self.c.db.update(
                        "regularization_operations",
                        oid,
                        {
                            "status": "failed",
                            "phase": "failed",
                            "error": "Could not start the regularization worker",
                            "finished_at": now(),
                        },
                    )
                if self.c.db.get_kv(RESERVATION, {}).get("id") == oid:
                    self.c.db.set_kv(RESERVATION, {})
                lease.__exit__(None, None, None)
                raise
        return self.get(oid)

    def cancel(self, oid):
        with self.c.db.lock:
            row = self._row(oid)
            if row["status"] in TERMINAL:
                return self.get(oid)
            if row["phase"] == "publishing":
                raise ApiError(
                    "The completed batch is being published; wait for it to finish",
                    status=409,
                    code="regularization.publishing",
                )
            event = self.cancel_events.get(oid)
            if event:
                event.set()
            self._update(oid, status="cancelling", message="Cancellation requested")
        return self.get(oid)

    def _rollback(self, row):
        from .routes_work import _records_path, _write_project_config, get_project_config

        _, staging, final = self._paths(row)
        token = json.loads(row["request_json"]).get("ownership_token")

        def owned(path):
            marker = path / OWNER_FILE
            return bool(
                token
                and path.is_dir()
                and not path.is_symlink()
                and marker.is_file()
                and not marker.is_symlink()
                and marker.read_text(encoding="utf-8") == token
            )

        with self.c.db.lock:
            dataset = (
                self.c.db.fetchone(
                    "SELECT id FROM datasets WHERE version_id=? AND path=?", (row["version_id"], str(final))
                )
                if owned(final)
                else None
            )
            if dataset:
                config = get_project_config(row["project_id"], self.c, row["version_id"])
                config.setdefault("dataset", {})["sources"] = [
                    source
                    for source in config.get("dataset", {}).get("sources", [])
                    if source.get("path") != str(final)
                ]
                _write_project_config(self.c, row["project_id"], config, row["version_id"])
                self.c.db.delete("datasets", dataset["id"])
                _records_path(self.c, dataset["id"]).unlink(missing_ok=True)
        for path in (staging, final):
            if owned(path):
                shutil.rmtree(path)

    def _run(self, oid, payload, credentials, cancelled, lease):
        row = self._row(oid)
        try:
            root, staging, final = self._paths(row)
            root.mkdir(parents=True, exist_ok=True)
            staging.mkdir()
            (staging / OWNER_FILE).write_text(payload["ownership_token"], encoding="utf-8")
            images = staging / "images"
            images.mkdir()
            self._check(cancelled)
            self._update(
                oid,
                status="running",
                phase="preparing",
                message="Preparing a new isolated regularization batch",
            )
            if row["source"] == "ai":
                self._generate(oid, payload, staging, images, cancelled)
            else:
                self._collect(oid, payload, credentials, images, cancelled)
            self._check(cancelled)
            self._deduplicate(oid, images, cancelled)
            count = len(list(iter_images(images)))
            if not count:
                raise ApiError("No new non-duplicate images were produced", code="regularization.empty")
            from .routes_work import DatasetBody, _index_dataset, _register_dataset

            # The version lease owns all writes. Cancel and publish are serialized at this boundary.
            with self.c.db.lock:
                self._check(cancelled)
                self._update(
                    oid, phase="publishing", message=f"Publishing {count} verified image/caption pairs"
                )
                # mkdir reserves the final path without replacing even an empty existing
                # directory. The source is registered only after all verified files move.
                final.mkdir()
                (final / OWNER_FILE).write_text(payload["ownership_token"], encoding="utf-8")
                for entry in images.iterdir():
                    entry.rename(final / entry.name)
                did = _register_dataset(
                    self.c,
                    row["project_id"],
                    DatasetBody(
                        path=str(final),
                        repeats=payload["repeats"],
                        prior_weight=payload["prior_weight"],
                        is_reg=True,
                    ),
                    version_id=row["version_id"],
                    internal=True,
                    caption=CaptionConfig().model_dump(),
                )
                self._update(oid, dataset_id=did, path=str(final))
            _index_dataset(self.c, did)
            indexed = self.c.db.fetchone("SELECT index_status FROM datasets WHERE id=?", (did,))
            if not indexed or indexed["index_status"] != "ready":
                raise ApiError(
                    "The new regularization batch could not be indexed", code="regularization.index"
                )
            self._update(
                oid,
                status="completed",
                phase="completed",
                images=count,
                done=payload["count"],
                finished_at=now(),
                message=f"Added {count} regularization images to this version",
            )
            shutil.rmtree(staging, ignore_errors=True)
        except Exception as exc:
            stopped = isinstance(exc, Cancelled) or cancelled.is_set()
            try:
                self._rollback(row)
            except Exception:
                self._update(oid, message="Cleanup requires attention; existing datasets were preserved")
            message = (
                None
                if stopped
                else (
                    exc.message
                    if isinstance(exc, ApiError)
                    else f"Regularization failed ({type(exc).__name__}); no batch was published"
                )
            )
            self._update(
                oid,
                status="cancelled" if stopped else "failed",
                phase="cancelled" if stopped else "failed",
                error=message,
                dataset_id=None,
                path=None,
                images=0,
                finished_at=now(),
            )
        finally:
            with self.c.db.lock:
                self.cancel_events.pop(oid, None)
                if self.c.db.get_kv(RESERVATION, {}).get("id") == oid:
                    self.c.db.set_kv(RESERVATION, {})
                lease.__exit__(None, None, None)
            self.c.bus.publish(
                "version.changed", {"project_id": row["project_id"], "version_id": row["version_id"]}
            )

    @staticmethod
    def _check(cancelled):
        if cancelled.is_set():
            raise Cancelled()

    def _generate(self, oid, payload, staging, images, cancelled):
        import psutil

        request_path, events, stop = staging / "request.json", staging / "events.jsonl", staging / "stop"
        request_path.write_text(
            json.dumps(
                payload | {"parent_pid": os.getpid(), "parent_created": psutil.Process().create_time()}
            ),
            encoding="utf-8",
        )
        command = [
            self.c.supervisor.python,
            "-m",
            "ypuddin.server.regularization_worker",
            str(request_path),
            str(images),
            str(events),
            str(stop),
        ]
        with (staging / "worker.log").open("wb") as log:
            proc = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=staging,
                env=dict(os.environ, PYTHONUNBUFFERED="1"),
            )
            self.processes[oid] = proc
            self.c.db.update("regularization_operations", oid, {"worker_pid": proc.pid})
            seen = 0
            cancel_time = None
            failure = None

            def pump():
                nonlocal seen, failure
                if not events.exists():
                    return
                lines = events.read_text(encoding="utf-8").splitlines()
                for line in lines[seen:]:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        break
                    seen += 1
                    failure = event.get("error") or failure
                    self._update(
                        oid,
                        phase=event.get("phase", "generating"),
                        done=event.get("done", self._row(oid)["done"]),
                        message=event.get("message"),
                    )

            try:
                while True:
                    if cancelled.is_set():
                        stop.touch(exist_ok=True)
                        cancel_time = cancel_time or time.monotonic()
                        if time.monotonic() - cancel_time > 5 and proc.poll() is None:
                            proc.terminate()
                    pump()
                    if proc.poll() is not None:
                        pump()  # Process exit can race the preceding read of its final error/progress.
                        break
                    if cancel_time and time.monotonic() - cancel_time > 8:
                        proc.kill()
                    time.sleep(0.05)
                self._check(cancelled)
                if proc.returncode:
                    raise ApiError(
                        failure
                        or "Base-model inference failed; check the configured model files and available memory",
                        code="regularization.inference",
                    )
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=3)
                self.processes.pop(oid, None)

    def _open_request(self, request, source, media):
        from .network import ProxyPolicy

        # The policy is read per request, so saved proxy settings apply to the next one.
        opener = self.opener or ProxyPolicy.from_context(self.c).opener(Redirect(source, media))
        return opener.open(request, timeout=15)

    def _open(self, url, source, *, media=False, headers=None):
        request = urllib.request.Request(url, headers={"User-Agent": user_agent(), **(headers or {})})
        return self._open_request(request, source, media)

    def client(self, source, credentials, *, cancelled=None, notify=None):
        return BooruClient(
            source,
            *credentials,
            opener=lambda request, media: self._open_request(request, source, media),
            cancelled=cancelled,
            notify=notify,
            max_file_bytes=MAX_FILE_BYTES,
        )

    def collected_posts(self, pid, vid):
        """Site posts earlier batches of this version took, including images deleted since then,
        so a new batch neither repeats them nor brings back what was removed."""
        root = self.c.reg_dir(pid, vid).resolve()
        seen = set()
        for row in self.c.db.fetchall(
            "SELECT path FROM regularization_operations WHERE project_id=? AND version_id=? AND status='completed' AND source!='ai'",
            (pid, vid),
        ):
            batch = Path(row["path"] or "").resolve()
            manifest = batch / "manifest.json"
            try:
                if not batch.is_relative_to(root) or manifest.stat().st_size > 16 * 1024 * 1024:
                    continue
                entries = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for entry in entries if isinstance(entries, list) else []:
                if isinstance(entry, dict) and isinstance(entry.get("post_id"), str):
                    seen.add((entry.get("provider"), entry["post_id"]))
        return seen

    def _collect(self, oid, payload, credentials, output, cancelled):
        source = payload["source"]
        manual = payload.get("prompt_source") != "training_tags"
        if manual and any(not re.fullmatch(r"[\w()\- ]{1,100}", tag) for tag in payload["excluded_tags"]):
            raise ApiError("Excluded search tags must be plain tags", code="regularization.query")
        row = self._row(oid)
        client = self.client(
            source,
            credentials,
            cancelled=cancelled,
            notify=lambda message: self._update(oid, message=message),
        )
        excluded = list(dict.fromkeys(tag for tag in map(normalize, payload["excluded_tags"]) if tag))
        taken = self.collected_posts(row["project_id"], row["version_id"])
        if not manual:
            from .regularization_match import build_profile, collect
            from .routes_work import get_project_config

            self._update(oid, phase="preparing", message="Reading the training captions and image sizes")
            profile = build_profile(
                self.c,
                row["project_id"],
                row["version_id"],
                get_project_config(row["project_id"], self.c, row["version_id"]),
                payload["source_ids"],
                payload["excluded_tags"],
            )
            if not profile.weights:
                raise ApiError(
                    "The training captions have no searchable tags left; check the captions and excluded tags",
                    code="regularization.empty_plan",
                )
            batch = _Batch(
                self, oid, client, output, payload["count"], excluded, taken | profile.post_ids, cancelled
            )
            collect(
                profile, client, batch, payload["count"], lambda message: self._update(oid, message=message)
            )
            batch.finish()
            return
        batch = _Batch(self, oid, client, output, payload["count"], excluded, taken, cancelled)
        tags = payload["prompt"].split()
        terms, local = client.query(tags, excluded)
        if local:
            self._update(
                oid, message=f"Checking {len(local)} excluded tags locally beyond the site's tag limit"
            )
        for page in range(1, 21):
            self._check(cancelled)
            self._update(oid, phase="searching", done=batch.done, message=f"Searching {source}, page {page}")
            posts = client.search(terms, page, min(client.site.page_size, max(20, payload["count"] * 2)))
            if not posts:
                break
            batch.take(posts, query=" ".join(tags))
            if batch.full:
                break
        batch.finish()

    def _deduplicate(self, oid, output, cancelled):
        from .routes_work import get_project_config

        row = self._row(oid)
        config = get_project_config(row["project_id"], self.c, row["version_id"])
        known = set()
        roots = {
            source["path"]
            for source in [
                *config.get("dataset", {}).get("sources", []),
                *config.get("validation", {}).get("sources", []),
            ]
        }
        for root in roots:
            if not self.c.is_allowed(Path(root)):
                continue
            for path in iter_images(root):
                self._check(cancelled)
                try:
                    known.add(image_identity(decoded_image(path)))
                except (OSError, ValueError):
                    pass
        duplicates = row["duplicates"]
        retained = set()
        for path in list(iter_images(output)):
            self._check(cancelled)
            if not path.with_suffix(".txt").is_file():
                raise ApiError("Generated image is missing its caption", code="regularization.caption")
            digest = image_identity(decoded_image(path))
            if digest in known:
                path.unlink()
                path.with_suffix(".txt").unlink()
                duplicates += 1
            else:
                known.add(digest)
                retained.add(path.name)
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_path.write_text(
            json.dumps(
                [entry for entry in manifest if entry["file"] in retained], ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
        self._update(oid, phase="verifying", duplicates=duplicates)
