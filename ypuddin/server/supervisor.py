"""Job supervisor: queue scheduling, subprocess launch, event-file tailing, control commands."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ypuddin.config import TrainConfig, write_config

from .bus import EventBus
from .db import Database, new_id, now

log = logging.getLogger(__name__)

ACTIVE = ("queued", "scheduled", "running", "pausing", "cancelling")
TERMINAL = ("completed", "failed", "cancelled", "paused")


class JobSupervisor:
    """One scheduler loop; jobs run as ``ypuddin train`` subprocesses writing ``events.jsonl``.

    The supervisor never parses stdout: it tails the structured event file, mirrors events onto the
    bus, and derives job status from the terminal ``run.*`` event (or the exit code if none arrived).
    """

    def __init__(self, db: Database, bus: EventBus, data_root: Path, *, poll_interval: float = 0.5, max_concurrent: int = 1, python: str | None = None):
        self.db = db
        self.bus = bus
        self.data_root = data_root
        self.poll = poll_interval
        self.max_concurrent = max_concurrent
        self.python = python or sys.executable
        self._procs: dict[str, subprocess.Popen] = {}
        self._offsets: dict[str, int] = {}
        self._task: asyncio.Task | None = None
        self._stopping = False

    # ----------------------------------------------------------------- lifecycle
    async def start(self) -> None:
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
        for job_id, proc in list(self._procs.items()):
            if proc.poll() is None:
                proc.terminate()
                self._set_status(job_id, "failed", error="service shutdown")

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
        settings = self.db.get_kv("queue.settings", {"held": False, "max_concurrent": self.max_concurrent})
        if settings.get("held"):
            return
        t = now()
        self.db.execute("UPDATE jobs SET status='queued' WHERE status='scheduled' AND scheduled_at IS NOT NULL AND scheduled_at <= ?", (t,))
        while len(self._procs) < int(settings.get("max_concurrent", 1)):
            nxt = self.db.fetchone("SELECT * FROM jobs WHERE status='queued' ORDER BY priority DESC, created_at ASC LIMIT 1")
            if not nxt:
                break
            self._launch(nxt)

    def _launch(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        run_dir = Path(job["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = TrainConfig.model_validate(json.loads(job["config_json"]))
        if job.get("resume_from"):
            cfg = cfg.model_copy(update={"checkpoint": cfg.checkpoint.model_copy(update={"resume": job["resume_from"]})})
        cfg_path = run_dir / "job-config.toml"
        write_config(cfg, cfg_path)
        events_path = run_dir / "events.jsonl"
        self._offsets[job_id] = events_path.stat().st_size if events_path.exists() else 0
        sub = {"train": "train", "cache": "cache"}[job["type"]]
        cmd = [self.python, "-m", "ypuddin.cli", sub, str(cfg_path)]
        log_fp = open(run_dir / "run.log", "ab")  # noqa: SIM115
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        kwargs: dict[str, Any] = {}
        if os.name == "nt":  # pragma: no cover
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT, cwd=str(run_dir), env=env, **kwargs)
        self._procs[job_id] = proc
        self._set_status(job_id, "running", started_at=now(), pid=proc.pid, resume_from=None)
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
            progress = json.loads(self.db.fetchone("SELECT progress_json FROM jobs WHERE id=?", (job_id,))["progress_json"] or "{}")
            progress.update({"phase": "training", "step": ev["step"], "epoch": ev["epoch"], "eta_s": ev.get("eta_s"), "it_s": ev.get("it_s"), "vram_peak_mb": ev.get("vram_mb")})
            latest = {"loss": ev.get("loss"), "loss_ema": ev.get("loss_ema"), "lr": ev.get("lr")}
            self.db.update("jobs", job_id, {"progress_json": json.dumps(progress), "latest_json": json.dumps(latest)})
            self.bus.publish("job.step", data)
        elif t == "run.prepared":
            self._merge_progress(job_id, {"total_steps": ev.get("total_steps"), "steps_per_epoch": ev.get("steps_per_epoch")})
            self.bus.publish("job.phase", {"job_id": job_id, "phase": "prepared", **{k: v for k, v in data.items() if k != "job_id"}})
        elif t == "phase.changed":
            self._merge_progress(job_id, {"phase": ev.get("phase")})
            self.bus.publish("job.phase", data)
        elif t == "cache.progress":
            self.bus.publish("job.cache_progress", data)
        elif t == "validation":
            self.bus.publish("job.validation", data)
        elif t == "sample.saved":
            self.bus.publish("job.sample", {**data, "url": f"/api/jobs/{job_id}/files?path={Path(ev['path']).name}&kind=sample"})
        elif t == "checkpoint.saved":
            if ev.get("kind") == "weights":
                self._register_artifact(job_id, ev)
            self.bus.publish("job.checkpoint", data)
        elif t == "warning":
            self.bus.publish("job.warning", data)
        elif t in ("run.finished", "run.paused", "run.stopped", "run.failed"):
            self._set_terminal(job_id, t)
        else:
            self.bus.publish("job.event", data)

    def _merge_progress(self, job_id: str, patch: dict[str, Any]) -> None:
        row = self.db.fetchone("SELECT progress_json FROM jobs WHERE id=?", (job_id,))
        progress = json.loads(row["progress_json"] or "{}") if row else {}
        progress.update({k: v for k, v in patch.items() if v is not None})
        self.db.update("jobs", job_id, {"progress_json": json.dumps(progress)})

    def _register_artifact(self, job_id: str, ev: dict[str, Any]) -> None:
        job = self.db.fetchone("SELECT project_id, name FROM jobs WHERE id=?", (job_id,))
        path = Path(ev["path"])
        if not path.exists():
            return
        aid = new_id("a")
        self.db.insert(
            "artifacts",
            {"id": aid, "project_id": job["project_id"] if job else None, "job_id": job_id, "name": path.name, "path": str(path), "size": path.stat().st_size, "kind": "weights", "step": ev.get("step"), "created_at": now(), "meta_json": "{}"},
        )
        self.bus.publish("artifact.created", {"job_id": job_id, "artifact_id": aid, "path": str(path)})

    def _set_terminal(self, job_id: str, event_type: str) -> None:
        status = {"run.finished": "completed", "run.paused": "paused", "run.stopped": "cancelled", "run.failed": "failed"}[event_type]
        fields: dict[str, Any] = {"finished_at": now()}
        if status == "paused":
            run_dir = Path(self.db.fetchone("SELECT run_dir FROM jobs WHERE id=?", (job_id,))["run_dir"])
            state = run_dir / "state-paused"
            fields["resume_from"] = str(state) if state.exists() else None
        self._set_status(job_id, status, **fields)

    def _on_exit(self, job_id: str, code: int | None) -> None:
        job = self.db.fetchone("SELECT status, run_dir FROM jobs WHERE id=?", (job_id,))
        if not job:
            return
        if job["status"] in TERMINAL:
            self.db.update("jobs", job_id, {"exit_code": code})
            return
        error = f"process exited with code {code}"
        log_path = Path(job["run_dir"]) / "run.log"
        if log_path.exists():
            tail = log_path.read_bytes()[-4000:].decode("utf-8", errors="replace").strip().splitlines()
            if tail:
                error += ": " + tail[-1][:500]
        status = "cancelled" if job["status"] == "cancelling" else "failed"
        self._set_status(job_id, status, finished_at=now(), exit_code=code, error=error if status == "failed" else None)

    def _set_status(self, job_id: str, status: str, **fields: Any) -> None:
        self.db.update("jobs", job_id, {"status": status, **fields})
        row = self.db.fetchone("SELECT progress_json, error FROM jobs WHERE id=?", (job_id,))
        self.bus.publish("job.state", {"job_id": job_id, "status": status, "progress": json.loads(row["progress_json"] or "{}") if row else {}, "error": row["error"] if row else None})
        self.bus.publish("queue.changed", {})

    # ----------------------------------------------------------------- control
    def request(self, job_id: str, command: str) -> dict[str, Any]:
        job = self.db.fetchone("SELECT * FROM jobs WHERE id=?", (job_id,))
        if not job:
            raise KeyError(job_id)
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
            resume_from = job.get("resume_from") or self._latest_state(Path(job["run_dir"]))
            if status != "paused" and not resume_from:
                raise ValueError("no checkpoint to resume from")
            self._set_status(job_id, "queued", resume_from=resume_from, error=None, finished_at=None)
        elif command == "cancel":
            if status in ("queued", "scheduled", "paused"):
                self._set_status(job_id, "cancelled", finished_at=now())
            elif status in ("running", "pausing"):
                self._control_file(job, "stop")
                self._set_status(job_id, "cancelling")
                proc = self._procs.get(job_id)
                if proc is not None:
                    asyncio.get_event_loop().call_later(30, self._force_kill, job_id)
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
        states = sorted(run_dir.glob("state-*"), key=lambda p: p.stat().st_mtime)
        return str(states[-1]) if states else None

    def _control_file(self, job: dict[str, Any], name: str) -> None:
        ctl = Path(job["run_dir"]) / "control"
        ctl.mkdir(parents=True, exist_ok=True)
        (ctl / name).touch()
        proc = self._procs.get(job["id"])
        if proc is not None and proc.poll() is None and os.name != "nt" and name in ("pause", "stop"):
            try:
                proc.send_signal(signal.SIGINT if name == "pause" else signal.SIGTERM)
            except OSError:
                pass

    def _force_kill(self, job_id: str) -> None:
        proc = self._procs.get(job_id)
        if proc is not None and proc.poll() is None:
            proc.kill()

    def clone(self, job: dict[str, Any]) -> dict[str, Any]:
        new = new_id("j")
        run_dir = Path(job["run_dir"]).parent / new
        cfg = json.loads(job["config_json"])
        cfg.setdefault("checkpoint", {})["output_dir"] = str(run_dir)
        self.db.insert(
            "jobs",
            {"id": new, "type": job["type"], "name": job["name"] + " (retry)", "project_id": job["project_id"], "status": "queued", "priority": job["priority"], "created_at": now(), "run_dir": str(run_dir), "config_json": json.dumps(cfg), "progress_json": "{}", "latest_json": "{}"},
        )
        self.bus.publish("queue.changed", {})
        return self.db.fetchone("SELECT * FROM jobs WHERE id=?", (new,))

    def is_running(self, job_id: str) -> bool:
        return job_id in self._procs
