"""Job supervisor: queue scheduling, subprocess launch, event-file tailing, control commands."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from ypuddin.config import TrainConfig, write_config
from ypuddin.config.io import absolute_paths
from ypuddin.runtime_profiles import current_profile

from .bus import EventBus
from .db import Database, new_id, now
from .environment import maintenance_blocked
from .gpu_selection import selection_error
from .hardware import gpu_info
from .sample_events import sample_event_loss

log = logging.getLogger(__name__)

ACTIVE = ("queued", "scheduled", "running", "pausing", "cancelling")
TERMINAL = ("completed", "failed", "cancelled", "paused")
EVENT_BATCH_SIZE = 128
EVENT_BATCH_BYTES = 256 * 1024


def training_device_error(
    count: int, inventory: list[dict[str, Any]], *, strategy: str = "ddp"
) -> str | None:
    """Validate requested parallelism against the running service, never a CPU fallback."""
    if count <= 1:
        return None
    import torch

    if current_profile().endswith("-cpu") or sys.platform not in {"linux", "win32"}:
        return "多卡训练需要 Linux CUDA/DTK 或 Windows CUDA DDP 环境；当前环境请使用 1 张显卡。"
    if sys.platform == "win32" and strategy != "ddp":
        return (
            "原生 Windows 暂不支持显存分片（FSDP）；可改用 DDP，或在 Linux 环境运行分片训练。"
        )
    devices = [g for g in inventory if str(g.get("device", "")).startswith("cuda:")]
    if len(devices) < count:
        return f"请求使用 {count} 张显卡，当前环境仅有 {len(devices)} 张可用显卡。"
    if sys.platform == "win32":
        if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
            return "Windows 多卡 DDP 需要可用的 CUDA PyTorch 环境。"
        if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
            return "当前 PyTorch 未提供 Gloo，无法启动 Windows 多卡 DDP。"
        # CUDA collectives are verified on the selected cards in each worker,
        # before constructing Trainer or loading/writing any training assets.
        return None
    if not torch.distributed.is_available() or not torch.distributed.is_nccl_available():
        return "当前 PyTorch 未提供 GPU 集体通信后端（NCCL 兼容接口），无法启动多卡训练。"
    return None


def worker_device_environment(devices: tuple[str, ...], env: dict[str, str]) -> dict[str, str]:
    """Translate service-local indices through inherited visibility masks.

    HIP visibility is applied after ROCR visibility. Keep the inherited ROCR mask
    intact, then narrow HIP's indices within it; resetting both masks to the
    service's logical indices could select somebody else's physical devices.
    """
    import torch

    indices = [int(device.split(":", 1)[1]) for device in devices]
    hip = bool(getattr(torch.version, "hip", None)) or current_profile() == "linux-dtk"
    key = "HIP_VISIBLE_DEVICES" if hip and "HIP_VISIBLE_DEVICES" in env else "CUDA_VISIBLE_DEVICES"
    inherited = env.get(key)
    visible = [part.strip() for part in inherited.split(",")] if inherited is not None else None
    if visible is not None and any(index >= len(visible) or not visible[index] for index in indices):
        raise ValueError(
            "GPU visibility changed since device discovery; refresh the environment before retrying"
        )
    selected = ",".join(visible[index] if visible is not None else str(index) for index in indices)
    result = dict(env, CUDA_VISIBLE_DEVICES=selected)
    if hip:
        result["HIP_VISIBLE_DEVICES"] = selected
    return result


class JobSupervisor:
    """One scheduler loop; jobs run as ``ypuddin train`` subprocesses writing ``events.jsonl``.

    The supervisor never parses stdout: it tails the structured event file, mirrors events onto the
    bus, and derives job status from the terminal ``run.*`` event (or the exit code if none arrived).
    """

    def __init__(
        self,
        db: Database,
        bus: EventBus,
        data_root: Path,
        *,
        poll_interval: float = 0.5,
        max_concurrent: int | None = None,
        python: str | None = None,
    ):
        self.db = db
        self.bus = bus
        self.data_root = data_root
        self.poll = poll_interval
        self.max_concurrent = max_concurrent
        self.python = python or sys.executable
        self._procs: dict[str, subprocess.Popen] = {}
        self._devices: dict[str, str | tuple[str, ...]] = {}
        self._offsets: dict[str, int] = {}
        # jobs whose current process already reported its outcome through the event stream; the
        # later process-exit notification must not touch their status (the user may have resumed)
        self._outcome_seen: set[str] = set()
        self._task: asyncio.Task | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._stopping = False
        self._pending_events: ContextVar[list[tuple[str, dict[str, Any]]] | None] = ContextVar(
            "supervisor_pending_events", default=None
        )

    # ----------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        self._event_loop = asyncio.get_running_loop()
        # Jobs that were running when the service died cannot be trusted; mark them failed.
        for job in self.db.fetchall("SELECT id FROM jobs WHERE status IN ('running','pausing','cancelling')"):
            self._set_status(job["id"], "failed", error="service restarted while the job was running")
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stopping = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Request a recoverable pause before closing SQLite. Preparation/cache pauses
        # retain their cache and can restart without a training checkpoint.
        for job_id, proc in list(self._procs.items()):
            if proc.poll() is None:
                self._pump_events(job_id)
                job = self.db.fetchone("SELECT * FROM jobs WHERE id=?", (job_id,))
                if job and job["status"] in ("running", "pausing") and job_id not in self._outcome_seen:
                    self._control_file(job, "pause")
                    self._set_status(job_id, "pausing")
        deadline = asyncio.get_running_loop().time() + 20
        while self._procs and asyncio.get_running_loop().time() < deadline:
            for job_id, proc in list(self._procs.items()):
                drained = self._pump_events(job_id)
                if proc.poll() is not None and drained and self._pump_events(job_id):
                    self._on_exit(job_id, proc.returncode)
                    del self._procs[job_id]
                    self._devices.pop(job_id, None)
            if self._procs:
                await asyncio.sleep(0.1)
        for job_id, proc in list(self._procs.items()):
            self._kill_process_tree(proc)
            await asyncio.to_thread(proc.wait, 5)
            while not self._pump_events(job_id):
                await asyncio.sleep(0)
            self._on_exit(job_id, proc.returncode)
        self._procs.clear()
        self._devices.clear()

    async def _loop(self) -> None:
        while not self._stopping:
            backlog = False
            try:
                backlog = self._tick()
            except Exception:  # noqa: BLE001
                log.exception("supervisor tick failed")
            # Drain bursts promptly without starving HTTP or limiting ingestion
            # to one small batch per polling interval.
            await asyncio.sleep(0 if backlog else self.poll)

    # ----------------------------------------------------------------- scheduling
    def _tick(self) -> bool:
        backlog = False
        for job_id in list(self._procs):
            drained = self._pump_events(job_id)
            backlog |= not drained
            proc = self._procs[job_id]
            if proc.poll() is not None:
                # An exited worker may leave several batches behind. Its terminal
                # event and artifacts must arrive before exit-code reconciliation.
                if not drained:
                    continue
                if not self._pump_events(job_id):
                    backlog = True
                    continue
                self._on_exit(job_id, proc.returncode)
                del self._procs[job_id]
                self._devices.pop(job_id, None)
        settings = self.db.get_kv("queue.settings", {"held": False, "max_concurrent": self.max_concurrent})
        if settings.get("held") or maintenance_blocked(self.db):
            return backlog
        t = now()
        self.db.execute(
            "UPDATE jobs SET status='queued' WHERE status='scheduled' AND scheduled_at IS NOT NULL AND scheduled_at <= ?",
            (t,),
        )
        limit = settings.get("max_concurrent", self.max_concurrent)
        slots = max(1, min(64, int(limit))) if limit is not None else max(1, len(gpu_info()))
        for nxt in self.db.fetchall(
            "SELECT * FROM jobs WHERE status='queued' ORDER BY priority DESC, created_at ASC"
        ):
            if len(self._procs) >= slots:
                break
            if nxt["id"] in self._procs:
                continue
            device = self._choose_device(nxt, check_memory=settings.get("memory_admission", True))
            if device is None:
                continue
            try:
                # Environment Apply takes this same lock before checking running jobs and
                # setting maintenance. A queued worker cannot slip through that boundary.
                with self.db.lock:
                    if maintenance_blocked(self.db):
                        break
                    current = self.db.fetchone("SELECT * FROM jobs WHERE id=?", (nxt["id"],))
                    if not current or current["status"] != "queued":
                        continue  # It may have been paused/deleted while hardware was inspected.
                    if (current.get("gpu_devices_json") or "[]") != (nxt.get("gpu_devices_json") or "[]"):
                        continue  # A changed request must be allocated from a fresh snapshot.
                    if current.get("version_id"):
                        version = self.db.fetchone(
                            "SELECT * FROM project_versions WHERE id=?", (current["version_id"],)
                        )
                        if version and version["busy"]:
                            continue
                        self._check_job_version(current)
                    self._launch(current, device=device)
            except Exception as exc:
                log.exception("could not launch job %s", nxt["id"])
                self._set_status(nxt["id"], "failed", error=str(exc), finished_at=now())
        return backlog

    def _choose_device(
        self, job: dict[str, Any], *, check_memory: bool = True
    ) -> str | tuple[str, ...] | None:
        count = self._gpu_count(job)
        requested = json.loads(job.get("gpu_devices_json") or "[]")
        inventory = [] if current_profile().endswith("-cpu") else gpu_info()
        if error := selection_error(requested, count, inventory):
            self._publish_admission(job, error=error)
            return None
        if current_profile().endswith("-cpu"):
            if count > 1:
                self._publish_admission(job, error="CPU 环境不能启动多卡训练")
                return None
            return "cpu"
        strategy = (
            json.loads(job.get("config_json") or "{}").get("loop", {}).get("distributed_strategy", "ddp")
        )
        if count > 1 and (error := training_device_error(count, inventory, strategy=strategy)):
            self._publish_admission(job, error=error)
            return None
        if not inventory:
            return "cpu"
        used = {device for allocation in self._devices.values() for device in self._allocation(allocation)}
        estimate = json.loads(job.get("progress_json") or "{}").get("estimated_peak_mb") or 0
        selected = []
        for gpu in sorted(inventory, key=lambda g: g.get("mem_free_mb") or 0, reverse=True):
            device = gpu["device"]
            if requested and device not in requested:
                continue
            if count > 1 and not device.startswith("cuda:"):
                continue
            if device in used:
                continue  # exclusive accelerator ownership; max_concurrent is an upper bound
            available = gpu.get("mem_free_mb")
            if check_memory and estimate and available is not None and estimate > available * 0.95:
                continue
            selected.append(device)
            if len(selected) == count:
                allocation = requested or selected
                return allocation[0] if count == 1 else tuple(allocation)
        patch = {
            "phase": "waiting_for_device",
            "wait_reason": f"等待所选显卡 {', '.join(requested)} 空闲且显存充足"
            if requested
            else f"等待 {count} 张空闲且显存充足的显卡",
        }
        self._publish_admission(job, progress=patch)
        return None

    def _publish_admission(
        self, job: dict[str, Any], *, error: str | None = None, progress: dict[str, Any] | None = None
    ) -> None:
        # Hardware inspection is outside this lock. Its result may only update
        # the same pending request, never a changed selection or a paused task.
        with self.db.lock:
            current = self.db.fetchone("SELECT * FROM jobs WHERE id=?", (job["id"],))
            if (
                not current
                or current["status"] != "queued"
                or job["id"] in self._procs
                or (current.get("gpu_devices_json") or "[]") != (job.get("gpu_devices_json") or "[]")
            ):
                return
            if error:
                self._set_status(job["id"], "failed", error=error, finished_at=now())
            elif progress:
                latest = json.loads(current.get("progress_json") or "{}")
                if any(latest.get(key) != value for key, value in progress.items()):
                    self._merge_progress(job["id"], progress)
                    self._publish("job.phase", {"job_id": job["id"], **progress})

    @staticmethod
    def _gpu_count(job: dict[str, Any]) -> int:
        if job.get("type") != "train":
            return 1
        return int(json.loads(job.get("config_json") or "{}").get("loop", {}).get("gpu_count", 1))

    @staticmethod
    def _allocation(device: str | tuple[str, ...]) -> tuple[str, ...]:
        return (device,) if isinstance(device, str) else device

    def _launch(self, job: dict[str, Any], *, device: str | tuple[str, ...] | None = None) -> None:
        job_id = job["id"]
        devices = self._allocation(device or "cpu")
        count = self._gpu_count(job)
        requested = json.loads(job.get("gpu_devices_json") or "[]")
        if requested and tuple(requested) != devices:
            raise ValueError("所选显卡已改变，请重新调度此任务。")
        if len(devices) != count or (count > 1 and any(not d.startswith("cuda:") for d in devices)):
            raise ValueError("所需显卡尚未全部分配，训练未启动。")
        # Every CUDA/HIP worker sees only its assigned cards. Single-card jobs
        # address their own card as cuda:0 even when the service assigned cuda:1.
        # This also isolates libraries that allocate on the default device.
        masked = all(d.startswith("cuda:") for d in devices)
        worker_device = "cuda:0" if masked else devices[0]
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        env.pop("YPUDDIN_LEGACY_CUDA_RNG_INDEX", None)
        if masked:
            env = worker_device_environment(devices, env)
        run_dir = Path(job["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        for command in ("pause", "stop", "save"):
            (run_dir / "control" / command).unlink(missing_ok=True)
        resume_from = None
        if job["type"] == "xyz":
            import psutil

            payload = json.loads(job["config_json"])
            source = self.db.fetchone(
                "SELECT config_json FROM jobs WHERE id=?", (payload["xyz"]["source_job_id"],)
            )
            if (
                source
                and json.loads(source["config_json"]).get("training", {}).get("mode") == "full"
                and payload.get("training", {}).get("mode") != "full"
            ):
                raise ValueError("这个旧的模型测试未固定完整模型权重，请选择已导出的检查点重新创建测试。")
            payload.update(
                device=worker_device,
                fingerprint_cache=str(self.data_root / "cache" / "xyz-fingerprints"),
                parent_pid=os.getpid(),
                parent_created=psutil.Process().create_time(),
            )
            cfg_path = run_dir / "xyz-request.json"
            cfg_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            cmd = [self.python, "-m", "ypuddin.server.xyz_worker", str(cfg_path)]
        else:
            payload = json.loads(job["config_json"])
            # New queue snapshots include the switch. Existing immutable jobs
            # predate it and must resume with their original calculation mode.
            payload.setdefault("loop", {}).setdefault("deterministic", False)
            if job["type"] == "cache":
                payload.setdefault("loop", {}).update(gpu_count=1, distributed_strategy="ddp")
            cfg = absolute_paths(TrainConfig.model_validate(payload))
            if job.get("resume_from"):
                cfg = cfg.model_copy(
                    update={"checkpoint": cfg.checkpoint.model_copy(update={"resume": job["resume_from"]})}
                )
            cfg.logging.events_path = str(run_dir / "events.jsonl")
            cfg_path = run_dir / "job-config.toml"
            write_config(cfg, cfg_path)
            sub = {"train": "train", "cache": "cache"}[job["type"]]
            cmd = [self.python, "-m", "ypuddin.cli", sub, str(cfg_path)]
            if count > 1:
                if sys.platform == "win32":
                    env.setdefault("YPUDDIN_DISTRIBUTED_BACKEND", "gloo")
                    env.setdefault("USE_LIBUV", "0")
                cmd = [
                    self.python,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc_per_node={count}",
                    "-m",
                    "ypuddin.cli",
                    sub,
                    str(cfg_path),
                    "--device",
                    "cuda",
                ]
            elif device:
                cmd += ["--device", worker_device]
            resume_from = cfg.checkpoint.resume
        legacy_binding = None
        if masked and count == 1 and resume_from:
            progress = json.loads(job.get("progress_json") or "{}")
            checkpoint = str(Path(resume_from).expanduser().resolve())
            saved = progress.get("legacy_cuda_rng_source") or {}
            if saved.get("checkpoint") == checkpoint:
                legacy_binding = saved
            else:
                previous = progress.get("devices") or [progress.get("device", "")]
                if len(previous) == 1 and str(previous[0]).startswith("cuda:"):
                    index = previous[0].split(":", 1)[1]
                    if index.isdecimal():
                        legacy_binding = {"checkpoint": checkpoint, "device_index": int(index)}
            if legacy_binding is not None:
                env["YPUDDIN_LEGACY_CUDA_RNG_INDEX"] = str(legacy_binding["device_index"])
        events_path = run_dir / "events.jsonl"
        self._offsets[job_id] = events_path.stat().st_size if events_path.exists() else 0
        kwargs: dict[str, Any] = {}
        if os.name == "nt":  # pragma: no cover
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        with open(run_dir / "run.log", "ab") as log_fp:
            proc = subprocess.Popen(
                cmd, stdout=log_fp, stderr=subprocess.STDOUT, cwd=str(run_dir), env=env, **kwargs
            )
        self._procs[job_id] = proc
        self._devices[job_id] = device or "cpu"
        phase = "checking_communication" if count > 1 and sys.platform == "win32" else "starting"
        self._merge_progress(
            job_id,
            {
                "device": ", ".join(devices),
                "devices": list(devices),
                "gpu_count": count,
                "phase": phase,
                "wait_reason": "",
                **({"legacy_cuda_rng_source": legacy_binding} if legacy_binding is not None else {}),
            },
        )
        self._outcome_seen.discard(job_id)
        self._set_status(
            job_id, "running", started_at=now(), pid=proc.pid, exit_code=None, resume_from=resume_from
        )
        self._publish("job.phase", {"job_id": job_id, "phase": phase})

    # ----------------------------------------------------------------- events
    def _publish(self, type_: str, data: dict[str, Any]) -> None:
        pending = self._pending_events.get()
        if pending is None:
            self.bus.publish(type_, data)
        else:
            pending.append((type_, data))

    def _pump_events(self, job_id: str) -> bool:
        """Consume one bounded batch; return whether no complete backlog remains.

        Workers can emit much faster than a network-backed SQLite file commits.
        One transaction per batch avoids a durable write per progress event, and
        a bounded batch gives HTTP requests and other jobs a turn between polls.
        """
        job = self.db.fetchone("SELECT run_dir, status FROM jobs WHERE id=?", (job_id,))
        if not job:
            return True
        path = Path(job["run_dir"]) / "events.jsonl"
        if not path.exists():
            return True
        offset = self._offsets.get(job_id, 0)
        end = offset
        events = []
        with open(path, "rb") as f:
            f.seek(offset)
            drained = True
            for _ in range(EVENT_BATCH_SIZE):
                line = f.readline()
                if not line.endswith(b"\n"):
                    break  # Leave a partially written line for the next poll.
                end = f.tell()
                events.append(line)
                if end - offset >= EVENT_BATCH_BYTES:
                    drained = not f.read(1)
                    break
            else:
                drained = not f.read(1)
        if not events:
            return drained
        # Neither SSE nor the read cursor can get ahead of committed state. A
        # ContextVar isolates HTTP-control publications from this synchronous batch.
        pending: list[tuple[str, dict[str, Any]]] = []
        with self.db.lock:
            outcome_seen = job_id in self._outcome_seen
            token = self._pending_events.set(pending)
            try:
                with self.db.transaction():
                    for line in events:
                        try:
                            ev = json.loads(line)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue
                        if isinstance(ev, dict):
                            self._handle_event(job_id, ev)
            except BaseException:
                if not outcome_seen:
                    self._outcome_seen.discard(job_id)
                raise
            finally:
                self._pending_events.reset(token)
            self._offsets[job_id] = end
            for type_, data in pending:
                self.bus.publish(type_, data)
        return drained

    def _handle_event(self, job_id: str, ev: dict[str, Any]) -> None:
        t = ev.get("type")
        data = {k: v for k, v in ev.items() if k not in ("seq", "ts", "type")}
        data["job_id"] = job_id
        if t == "step":
            progress = json.loads(
                self.db.fetchone("SELECT progress_json FROM jobs WHERE id=?", (job_id,))["progress_json"]
                or "{}"
            )
            progress.update(
                {
                    "phase": "training",
                    "preparing": False,
                    "step": ev["step"],
                    "epoch": ev["epoch"],
                    "eta_s": ev.get("eta_s"),
                    "it_s": ev.get("it_s"),
                    "vram_peak_mb": ev.get("vram_mb"),
                    "vram_metric": ev.get("vram_metric"),
                }
            )
            latest = {
                key: ev.get(key)
                for key in ("loss", "loss_ema", "lr", "loss_mean", "loss_count", "loss_mean_scope")
            }
            self.db.update(
                "jobs", job_id, {"progress_json": json.dumps(progress), "latest_json": json.dumps(latest)}
            )
            self._publish("job.step", data)
        elif t == "run.prepared":
            self.db.update("jobs", job_id, {"resume_from": None})
            self._merge_progress(
                job_id,
                {
                    "total_steps": ev.get("total_steps"),
                    "steps_per_epoch": ev.get("steps_per_epoch"),
                    "preparing": False,
                },
            )
            self._publish(
                "job.phase",
                {"job_id": job_id, "phase": "prepared", **{k: v for k, v in data.items() if k != "job_id"}},
            )
        elif t == "phase.changed":
            self._merge_progress(job_id, {"phase": ev.get("phase")})
            self._publish("job.phase", data)
        elif t == "cache.progress":
            self._merge_progress(
                job_id,
                {"cache_done": ev.get("done"), "cache_total": ev.get("total"), "cache_kind": ev.get("kind")},
            )
            self._publish("job.cache_progress", data)
        elif t == "sample.progress":
            self._publish("job.sample_progress", data)
        elif t == "xyz.progress":
            self._merge_progress(job_id, {k: v for k, v in data.items() if k != "job_id"})
            self._publish("job.xyz_progress", data)
        elif t == "validation":
            self._publish("job.validation", data)
        elif t == "sample.saved":
            row = self.db.fetchone("SELECT run_dir FROM jobs WHERE id=?", (job_id,))
            events_path = Path(row["run_dir"]) / "events.jsonl" if row and row["run_dir"] else None
            self._publish(
                "job.sample",
                {
                    **data,
                    "url": f"/api/jobs/{job_id}/files?path={Path(ev['path']).name}&kind=sample",
                    "created_at": ev.get("ts"),
                    "loss": sample_event_loss(ev, events_path),
                },
            )
        elif t == "checkpoint.saved":
            if ev.get("kind") in {"weights", "model"}:
                self._register_artifact(job_id, ev)
            self._publish("job.checkpoint", data)
        elif t == "warning":
            self._publish("job.warning", data)
        elif t in ("run.finished", "run.paused", "run.stopped", "run.failed"):
            self._set_terminal(job_id, t, ev)
        else:
            self._publish("job.event", data)

    def _merge_progress(self, job_id: str, patch: dict[str, Any]) -> None:
        row = self.db.fetchone("SELECT progress_json FROM jobs WHERE id=?", (job_id,))
        progress = json.loads(row["progress_json"] or "{}") if row else {}
        progress.update({k: v for k, v in patch.items() if v is not None})
        self.db.update("jobs", job_id, {"progress_json": json.dumps(progress)})

    def _register_artifact(self, job_id: str, ev: dict[str, Any]) -> None:
        job = self.db.fetchone("SELECT project_id, version_id, name FROM jobs WHERE id=?", (job_id,))
        path = Path(ev["path"])
        if not path.exists():
            return
        aid = new_id("a")
        self.db.insert(
            "artifacts",
            {
                "id": aid,
                "project_id": job["project_id"] if job else None,
                "version_id": job["version_id"] if job else None,
                "job_id": job_id,
                "name": path.name,
                "path": str(path),
                "size": sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())
                if path.is_dir()
                else path.stat().st_size,
                "kind": ev.get("kind", "weights"),
                "step": ev.get("step"),
                "created_at": now(),
                "meta_json": "{}",
            },
        )
        self._publish("artifact.created", {"job_id": job_id, "artifact_id": aid, "path": str(path)})

    def _set_terminal(self, job_id: str, event_type: str, event: dict[str, Any] | None = None) -> None:
        status = {
            "run.finished": "completed",
            "run.paused": "paused",
            "run.stopped": "cancelled",
            "run.failed": "failed",
        }[event_type]
        fields: dict[str, Any] = {"finished_at": now()}
        event = event or {}
        if status == "failed":
            fields["error"] = str(
                event.get("error") or event.get("message") or "training failed; see run.log"
            )
        self._outcome_seen.add(job_id)
        if status == "paused":
            job = self.db.fetchone("SELECT run_dir, resume_from FROM jobs WHERE id=?", (job_id,))
            run_dir = Path(job["run_dir"])
            state = run_dir / "state-paused"
            fields["resume_from"] = (
                job["resume_from"]
                if event.get("preparing")
                else (str(state) if (state / "state.json").exists() else None)
            )
            self._merge_progress(job_id, {"preparing": bool(event.get("preparing"))})
        self._set_status(job_id, status, **fields)

    def _on_exit(self, job_id: str, code: int | None) -> None:
        job = self.db.fetchone("SELECT status, run_dir FROM jobs WHERE id=?", (job_id,))
        if not job:
            return
        failed_after_completion = job["status"] == "completed" and code is not None and code != 0
        if (job["status"] in TERMINAL or job_id in self._outcome_seen) and not failed_after_completion:
            self._outcome_seen.discard(job_id)
            self.db.update("jobs", job_id, {"exit_code": code})
            return
        self._outcome_seen.discard(job_id)
        error = (
            f"worker process exited with code {code} after reporting completion"
            if failed_after_completion
            else f"process exited with code {code}"
        )
        log_path = Path(job["run_dir"]) / "run.log"
        if log_path.exists():
            tail = log_path.read_bytes()[-4000:].decode("utf-8", errors="replace").strip().splitlines()
            if tail:
                error += ": " + tail[-1][:500]
        status = "cancelled" if job["status"] == "cancelling" else "failed"
        self._set_status(
            job_id, status, finished_at=now(), exit_code=code, error=error if status == "failed" else None
        )

    def _set_status(self, job_id: str, status: str, **fields: Any) -> None:
        self.db.update("jobs", job_id, {"status": status, **fields})
        row = self.db.fetchone("SELECT progress_json, error FROM jobs WHERE id=?", (job_id,))
        self._publish(
            "job.state",
            {
                "job_id": job_id,
                "status": status,
                "progress": json.loads(row["progress_json"] or "{}") if row else {},
                "error": row["error"] if row else None,
            },
        )
        self._publish("queue.changed", {})

    # ----------------------------------------------------------------- control
    def request(self, job_id: str, command: str) -> dict[str, Any]:
        with self.db.lock:
            return self._request(job_id, command)

    def _request(self, job_id: str, command: str) -> dict[str, Any]:
        job = self.db.fetchone("SELECT * FROM jobs WHERE id=?", (job_id,))
        if not job:
            raise KeyError(job_id)
        if job["type"] == "xyz" and command not in {"cancel", "retry"}:
            raise ValueError("模型测试支持取消和重试，不能执行训练任务的暂停、恢复或保存操作。")
        if command in {"resume", "retry"}:
            self._check_job_version(job)
        status = job["status"]
        if command == "pause":
            if status in ("queued", "scheduled"):
                self._set_status(job_id, "paused")
            elif status == "running":
                self._control_file(job, "pause")
                self._set_status(job_id, "pausing")
            else:
                raise ValueError(f"cannot pause a job in status {status}")
        elif command == "resume":
            if status not in ("paused", "failed", "cancelled"):
                raise ValueError(f"cannot resume a job in status {status}")
            progress = json.loads(job.get("progress_json") or "{}")
            resume_from = job.get("resume_from")
            if not progress.get("preparing"):
                resume_from = resume_from or self._latest_state(Path(job["run_dir"]))
            if status != "paused" and job["type"] != "cache" and not resume_from:
                raise ValueError("no checkpoint to resume from")
            self._set_status(job_id, "queued", resume_from=resume_from, error=None, finished_at=None)
        elif command == "cancel":
            if status in ("queued", "scheduled", "paused"):
                self._set_status(job_id, "cancelled", finished_at=now())
            elif status in ("running", "pausing"):
                self._control_file(job, "stop")
                self._set_status(job_id, "cancelling")
                proc = self._procs.get(job_id)
                if proc is not None and self._event_loop is not None:
                    self._event_loop.call_soon_threadsafe(
                        self._event_loop.call_later, 30, self._force_kill, job_id, proc
                    )
            else:
                raise ValueError(f"cannot cancel a job in status {status}")
        elif command == "save":
            if status != "running":
                raise ValueError("job is not running")
            self._control_file(job, "save")
        elif command == "retry":
            return self.clone(job)
        else:
            raise ValueError(f"unknown command {command}")
        return self.db.fetchone("SELECT * FROM jobs WHERE id=?", (job_id,))

    def _latest_state(self, run_dir: Path) -> str | None:
        states = sorted(
            (p for p in run_dir.glob("state-*") if (p / "state.json").is_file()),
            key=lambda p: p.stat().st_mtime,
        )
        return str(states[-1]) if states else None

    def _control_file(self, job: dict[str, Any], name: str) -> None:
        ctl = Path(job["run_dir"]) / "control"
        ctl.mkdir(parents=True, exist_ok=True)
        (ctl / name).touch()
        # File control works on Windows and cannot SIGINT a child before its
        # handlers are installed. The trainer polls it during preparation too.

    def _force_kill(self, job_id: str, expected: subprocess.Popen | None = None) -> None:
        proc = self._procs.get(job_id)
        if proc is not None and (expected is None or proc is expected) and proc.poll() is None:
            self._kill_process_tree(proc)

    @staticmethod
    def _kill_process_tree(proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        if os.name == "nt":  # pragma: no cover
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=10, check=False
            )
        else:
            import psutil

            # torchrun gives every rank its own session/process group. Killing
            # only the launcher's group leaves those ranks and DataLoader
            # workers alive. Freeze each owned parent before discovering its
            # children so elastic cannot restart a rank while we tear it down.
            # psutil retains each process identity and rejects reused PIDs.
            owned = []
            try:
                pending = [psutil.Process(proc.pid)]
                while pending:
                    process = pending.pop()
                    try:
                        process.suspend()
                        owned.append(process)
                        pending.extend(process.children())
                    except psutil.NoSuchProcess:
                        continue
            except psutil.NoSuchProcess:
                pass
            finally:
                for process in reversed(owned):
                    try:
                        process.kill()
                    except psutil.NoSuchProcess:
                        pass

    def clone(self, job: dict[str, Any]) -> dict[str, Any]:
        self._check_job_version(job)
        new = new_id("j")
        run_dir = Path(job["run_dir"]).parent / new
        samples_dir = (
            Path(job["samples_dir"]).parent / new
            if job.get("samples_dir") and Path(job["samples_dir"]).name == job["id"]
            else run_dir / "samples"
        )
        cfg = json.loads(job["config_json"])
        cfg.setdefault("checkpoint", {})["output_dir"] = str(run_dir)
        cfg["checkpoint"]["resume"] = None
        cfg.setdefault("logging", {})["events_path"] = str(run_dir / "events.jsonl")
        cfg.setdefault("sampling", {})["output_dir"] = str(samples_dir)
        self.db.insert(
            "jobs",
            {
                "id": new,
                "type": job["type"],
                "name": job["name"] + " (retry)",
                "project_id": job["project_id"],
                "version_id": job.get("version_id"),
                "status": "queued",
                "priority": job["priority"],
                "gpu_devices_json": job.get("gpu_devices_json") or "[]",
                "created_at": now(),
                "run_dir": str(run_dir),
                "samples_dir": str(samples_dir),
                "config_json": json.dumps(cfg),
                "progress_json": json.dumps(
                    {
                        "estimated_peak_mb": json.loads(job.get("progress_json") or "{}").get(
                            "estimated_peak_mb"
                        )
                    }
                ),
                "latest_json": "{}",
            },
        )
        self._publish("queue.changed", {})
        return self.db.fetchone("SELECT * FROM jobs WHERE id=?", (new,))

    def is_running(self, job_id: str) -> bool:
        return job_id in self._procs

    def _check_job_version(self, job: dict[str, Any]) -> None:
        from .errors import ApiError

        if job.get("project_id"):
            project = self.db.fetchone("SELECT archived FROM projects WHERE id=?", (job["project_id"],))
            if not project or project["archived"]:
                raise ApiError(
                    "项目已归档或不可用，请先恢复项目再创建或恢复任务。", code="project.archived", status=409
                )
        if job.get("version_id"):
            version = self.db.fetchone("SELECT * FROM project_versions WHERE id=?", (job["version_id"],))
            if not version or version["status"] != "ready" or version["busy"] or version["archived"]:
                raise ApiError(
                    "任务所属版本不可用、已归档或正在处理，请检查版本状态。", code="version.busy", status=409
                )
