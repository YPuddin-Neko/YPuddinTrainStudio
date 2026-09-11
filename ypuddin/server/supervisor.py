"""Job supervisor: queue scheduling, subprocess launch, event-file tailing, control commands."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ypuddin.config import TrainConfig, write_config
from ypuddin.config.io import absolute_paths

from .bus import EventBus
from .db import Database, new_id, now
from .environment import maintenance_blocked
from .hardware import gpu_info

log = logging.getLogger(__name__)

ACTIVE = ("queued", "scheduled", "running", "pausing", "cancelling")
TERMINAL = ("completed", "failed", "cancelled", "paused")


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
        max_concurrent: int = 1,
        python: str | None = None,
    ):
        self.db = db
        self.bus = bus
        self.data_root = data_root
        self.poll = poll_interval
        self.max_concurrent = max_concurrent
        self.python = python or sys.executable
        self._procs: dict[str, subprocess.Popen] = {}
        self._devices: dict[str, str] = {}
        self._offsets: dict[str, int] = {}
        # jobs whose current process already reported its outcome through the event stream; the
        # later process-exit notification must not touch their status (the user may have resumed)
        self._outcome_seen: set[str] = set()
        self._task: asyncio.Task | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._stopping = False

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
                self._pump_events(job_id)
                if proc.poll() is not None:
                    self._pump_events(job_id)
                    self._on_exit(job_id, proc.returncode)
                    del self._procs[job_id]
                    self._devices.pop(job_id, None)
            if self._procs:
                await asyncio.sleep(0.1)
        for job_id, proc in list(self._procs.items()):
            proc.kill()
            await asyncio.to_thread(proc.wait, 5)
            self._pump_events(job_id)
            self._on_exit(job_id, proc.returncode)
        self._procs.clear()
        self._devices.clear()

    async def _loop(self) -> None:
        while not self._stopping:
            try:
                self._tick()
            except Exception:  # noqa: BLE001
                log.exception("supervisor tick failed")
            await asyncio.sleep(self.poll)

    # ----------------------------------------------------------------- scheduling
    def _tick(self) -> None:
        for job_id in list(self._procs):
            self._pump_events(job_id)
            proc = self._procs[job_id]
            if proc.poll() is not None:
                self._pump_events(job_id)
                self._on_exit(job_id, proc.returncode)
                del self._procs[job_id]
                self._devices.pop(job_id, None)
        settings = self.db.get_kv("queue.settings", {"held": False, "max_concurrent": self.max_concurrent})
        if settings.get("held") or maintenance_blocked(self.db):
            return
        t = now()
        self.db.execute(
            "UPDATE jobs SET status='queued' WHERE status='scheduled' AND scheduled_at IS NOT NULL AND scheduled_at <= ?",
            (t,),
        )
        slots = max(1, min(64, int(settings.get("max_concurrent", 1))))
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

    def _choose_device(self, job: dict[str, Any], *, check_memory: bool = True) -> str | None:
        inventory = gpu_info()
        if not inventory:
            return "cpu"
        used = set(self._devices.values())
        estimate = json.loads(job.get("progress_json") or "{}").get("estimated_peak_mb") or 0
        for gpu in sorted(inventory, key=lambda g: g.get("mem_free_mb", 0), reverse=True):
            device = gpu["device"]
            if device in used:
                continue  # exclusive accelerator ownership; max_concurrent is an upper bound
            available = gpu.get("mem_free_mb")
            if check_memory and estimate and available is not None and estimate > available * 0.95:
                continue
            return device
        patch = {
            "phase": "waiting_for_device",
            "wait_reason": "waiting for a free accelerator with enough memory",
        }
        current = json.loads(job.get("progress_json") or "{}")
        if any(current.get(k) != v for k, v in patch.items()):
            self._merge_progress(job["id"], patch)
            self.bus.publish("job.phase", {"job_id": job["id"], **patch})
        return None

    def _launch(self, job: dict[str, Any], *, device: str | None = None) -> None:
        job_id = job["id"]
        run_dir = Path(job["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = absolute_paths(TrainConfig.model_validate(json.loads(job["config_json"])))
        if job.get("resume_from"):
            cfg = cfg.model_copy(
                update={"checkpoint": cfg.checkpoint.model_copy(update={"resume": job["resume_from"]})}
            )
        cfg.logging.events_path = str(run_dir / "events.jsonl")
        for command in ("pause", "stop", "save"):
            (run_dir / "control" / command).unlink(missing_ok=True)
        cfg_path = run_dir / "job-config.toml"
        write_config(cfg, cfg_path)
        events_path = run_dir / "events.jsonl"
        self._offsets[job_id] = events_path.stat().st_size if events_path.exists() else 0
        sub = {"train": "train", "cache": "cache"}[job["type"]]
        cmd = [self.python, "-m", "ypuddin.cli", sub, str(cfg_path)]
        if device:
            cmd += ["--device", device]
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        kwargs: dict[str, Any] = {}
        if os.name == "nt":  # pragma: no cover
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        with open(run_dir / "run.log", "ab") as log_fp:
            proc = subprocess.Popen(
                cmd, stdout=log_fp, stderr=subprocess.STDOUT, cwd=str(run_dir), env=env, **kwargs
            )
        self._procs[job_id] = proc
        self._devices[job_id] = device or "cpu"
        self._merge_progress(job_id, {"device": device or "cpu", "phase": "starting", "wait_reason": ""})
        self._outcome_seen.discard(job_id)
        self._set_status(job_id, "running", started_at=now(), pid=proc.pid, resume_from=cfg.checkpoint.resume)
        self.bus.publish("job.phase", {"job_id": job_id, "phase": "starting"})

    # ----------------------------------------------------------------- events
    def _pump_events(self, job_id: str) -> None:
        job = self.db.fetchone("SELECT run_dir, status FROM jobs WHERE id=?", (job_id,))
        if not job:
            return
        path = Path(job["run_dir"]) / "events.jsonl"
        if not path.exists():
            return
        offset = self._offsets.get(job_id, 0)
        with open(path, "rb") as f:
            f.seek(offset)
            chunk = f.read()
        if not chunk:
            return
        # only consume complete lines
        cut = chunk.rfind(b"\n")
        if cut < 0:
            return
        self._offsets[job_id] = offset + cut + 1
        for line in chunk[: cut + 1].splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._handle_event(job_id, ev)

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
            latest = {"loss": ev.get("loss"), "loss_ema": ev.get("loss_ema"), "lr": ev.get("lr")}
            self.db.update(
                "jobs", job_id, {"progress_json": json.dumps(progress), "latest_json": json.dumps(latest)}
            )
            self.bus.publish("job.step", data)
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
            self.bus.publish(
                "job.phase",
                {"job_id": job_id, "phase": "prepared", **{k: v for k, v in data.items() if k != "job_id"}},
            )
        elif t == "phase.changed":
            self._merge_progress(job_id, {"phase": ev.get("phase")})
            self.bus.publish("job.phase", data)
        elif t == "cache.progress":
            self.bus.publish("job.cache_progress", data)
        elif t == "sample.progress":
            self.bus.publish("job.sample_progress", data)
        elif t == "validation":
            self.bus.publish("job.validation", data)
        elif t == "sample.saved":
            self.bus.publish(
                "job.sample",
                {**data, "url": f"/api/jobs/{job_id}/files?path={Path(ev['path']).name}&kind=sample"},
            )
        elif t == "checkpoint.saved":
            if ev.get("kind") == "weights":
                self._register_artifact(job_id, ev)
            self.bus.publish("job.checkpoint", data)
        elif t == "warning":
            self.bus.publish("job.warning", data)
        elif t in ("run.finished", "run.paused", "run.stopped", "run.failed"):
            self._set_terminal(job_id, t, ev)
        else:
            self.bus.publish("job.event", data)

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
                "size": path.stat().st_size,
                "kind": "weights",
                "step": ev.get("step"),
                "created_at": now(),
                "meta_json": "{}",
            },
        )
        self.bus.publish("artifact.created", {"job_id": job_id, "artifact_id": aid, "path": str(path)})

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
        if job["status"] in TERMINAL or job_id in self._outcome_seen:
            self._outcome_seen.discard(job_id)
            self.db.update("jobs", job_id, {"exit_code": code})
            return
        error = f"process exited with code {code}"
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
        self.bus.publish(
            "job.state",
            {
                "job_id": job_id,
                "status": status,
                "progress": json.loads(row["progress_json"] or "{}") if row else {},
                "error": row["error"] if row else None,
            },
        )
        self.bus.publish("queue.changed", {})

    # ----------------------------------------------------------------- control
    def request(self, job_id: str, command: str) -> dict[str, Any]:
        with self.db.lock:
            return self._request(job_id, command)

    def _request(self, job_id: str, command: str) -> dict[str, Any]:
        job = self.db.fetchone("SELECT * FROM jobs WHERE id=?", (job_id,))
        if not job:
            raise KeyError(job_id)
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
            proc.kill()

    def clone(self, job: dict[str, Any]) -> dict[str, Any]:
        self._check_job_version(job)
        new = new_id("j")
        run_dir = Path(job["run_dir"]).parent / new
        cfg = json.loads(job["config_json"])
        cfg.setdefault("checkpoint", {})["output_dir"] = str(run_dir)
        cfg["checkpoint"]["resume"] = None
        cfg.setdefault("logging", {})["events_path"] = str(run_dir / "events.jsonl")
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
                "created_at": now(),
                "run_dir": str(run_dir),
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
        self.bus.publish("queue.changed", {})
        return self.db.fetchone("SELECT * FROM jobs WHERE id=?", (new,))

    def is_running(self, job_id: str) -> bool:
        return job_id in self._procs

    def _check_job_version(self, job: dict[str, Any]) -> None:
        if job.get("version_id"):
            version = self.db.fetchone("SELECT * FROM project_versions WHERE id=?", (job["version_id"],))
            if not version or version["status"] != "ready" or version["busy"] or version["archived"]:
                raise ValueError("job version is unavailable, archived or busy")
