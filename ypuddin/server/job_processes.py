"""Identify workers left behind by a previous service process without signalling them."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import psutil


def capture_created(pid: int) -> float | None:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    try:
        return psutil.Process(pid).create_time()
    except (psutil.Error, OSError):
        return None


def _path(value: str | Path) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(value)))


def _alive(process: psutil.Process) -> bool | None:
    try:
        return process.is_running() and process.status() not in {psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD}
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, OSError):
        return None


def _matches(process: psutil.Process, directory: str, configs: set[str]) -> bool | None:
    unreadable = False
    try:
        cwd = process.cwd()
        if cwd and _path(cwd) == directory:
            return True
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, OSError):
        unreadable = True
    try:
        for argument in process.cmdline():
            value = argument.split("=", 1)[1] if argument.startswith(("--config=", "--request=")) else argument
            if os.path.isabs(value) and _path(value) in configs:
                return True
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, OSError):
        unreadable = True
    return None if unreadable else False


def remaining_workers(job: dict[str, Any]) -> bool:
    """Return true for a live worker or an unreadable recorded process identity.

    A PID from a legacy row is checked against the job's paths. Other processes
    must match a path: an inaccessible unrelated process cannot block every job.
    Call on restart or when rechecking an interrupted job, not for every queued row.
    """
    pid, created = job.get("pid"), job.get("pid_created")
    directory = _path(job["run_dir"]) if job.get("run_dir") else None
    configs = ({_path(Path(directory) / name) for name in
                ("job-config.toml", "xyz-request.json", "tts-request.json")} if directory else set())
    recorded_pid = pid if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0 else None
    if recorded_pid is not None:
        try:
            process = psutil.Process(recorded_pid)
            expected = (isinstance(created, (int, float)) and not isinstance(created, bool)
                        and math.isfinite(created) and created > 0)
            if expected:
                if process.create_time() == created and _alive(process) is not False:
                    return True
            elif _alive(process) is not False:
                if directory is None or _matches(process, directory, configs) is not False:
                    return True
        except psutil.NoSuchProcess:
            pass
        except (psutil.AccessDenied, OSError):
            return True
    if directory is None:
        return False
    # A torchrun rank can have its own session and outlive the launcher. Its
    # inherited working directory or exact config argument still identifies it.
    try:
        for process in psutil.process_iter():
            if process.pid == recorded_pid:
                continue
            if _alive(process) is False:
                continue
            if _matches(process, directory, configs) is True:
                return True
    except (psutil.AccessDenied, OSError):
        return True
    return False
