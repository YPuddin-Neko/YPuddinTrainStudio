"""Paths frozen with each queued job, independent of later service settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal


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


def output_directory(job: dict) -> Path:
    """The folder the job's products go to; the same as its records for jobs saved before jobs/."""
    return Path(job_config(job).get("checkpoint", {}).get("output_dir") or job["run_dir"])


def job_directories(job: dict) -> list[Path]:
    """Every recorded file directory, including legacy and custom locations."""
    config = job_config(job)
    checkpoint = config.get("checkpoint") or {}
    logs = config.get("logging") or {}
    candidates = [Path(value) for value in (
        job.get("run_dir"), job.get("samples_dir"), checkpoint.get("output_dir"),
        checkpoint.get("state_dir"), logs.get("output_dir"), (config.get("sampling") or {}).get("output_dir"),
    ) if value]
    if logs.get("events_path"):
        candidates.append(Path(logs["events_path"]).parent)
    return list(dict.fromkeys(candidates))


def owned_job_directories(job: dict) -> list[Path]:
    """Folders that belong to this job alone: each is named after it (resume/ lives inside jobs/<job>)."""
    return list(
        dict.fromkeys(path for path in job_directories(job) if path.name == job["id"] and not path.is_symlink())
    )


def removal_problem(context: Any, job_id: str, folders: list[Path]) -> Literal["outside", "shared"] | None:
    """Why the job's folders cannot be removed: reached through a link or outside the allowed
    roots, or holding a folder another job records (an old or edited path may nest one there)."""
    targets = []
    for folder in folders:
        if any(part.is_symlink() for part in (folder, *folder.parents)) or not context.is_allowed(folder.resolve()):
            return "outside"
        targets.append(folder.resolve())
    if not targets:
        return None
    others = {
        path.resolve()
        for row in context.db.fetchall("SELECT run_dir, samples_dir, config_json FROM jobs WHERE id!=?", (job_id,))
        for path in job_directories(row)
    }
    return "shared" if any(path.is_relative_to(target) for path in others for target in targets) else None
