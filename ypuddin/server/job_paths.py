"""Paths frozen with each queued job, independent of later service settings."""

from __future__ import annotations

import json
from pathlib import Path


def job_config(job: dict) -> dict:
    return json.loads(job.get("config_json") or "{}")


def state_directory(job: dict) -> Path:
    return Path(job_config(job).get("checkpoint", {}).get("state_dir") or job["run_dir"])


def event_file(job: dict) -> Path:
    path = job_config(job).get("logging", {}).get("events_path")
    if path:
        return Path(path)
    return Path(job["run_dir"]) / "events.jsonl"


def log_file(job: dict) -> Path:
    return event_file(job).parent / "run.log"


def owned_job_directories(job: dict) -> list[Path]:
    candidates = [Path(job["run_dir"]), state_directory(job), log_file(job).parent]
    if job.get("samples_dir"):
        candidates.append(Path(job["samples_dir"]))
    return list(
        dict.fromkeys(path for path in candidates if path.name == job["id"] and not path.is_symlink())
    )
