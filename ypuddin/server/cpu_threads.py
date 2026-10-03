"""Bound automatic training CPU threads by the process's affinity and cgroup budget."""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path, PurePosixPath

_PROC_SELF = Path("/proc/self")
_MAX_THREADS_PER_RANK = 4


def _mount_path(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def _quota(directory: Path, *, unified: bool) -> float | None:
    if unified:
        amount, period = (directory / "cpu.max").read_text().split()
    else:
        amount = (directory / "cpu.cfs_quota_us").read_text().strip()
        period = (directory / "cpu.cfs_period_us").read_text().strip()
    interval = int(period)
    if interval <= 0:
        raise ValueError("invalid CPU quota period")
    if amount == ("max" if unified else "-1"):
        return None
    budget = int(amount)
    if budget <= 0:
        raise ValueError("invalid CPU quota")
    return budget / interval


def _linux_cpu_quota() -> float | None:
    memberships: dict[str, PurePosixPath] = {}
    for line in (_PROC_SELF / "cgroup").read_text().splitlines():
        _, controllers, path = line.split(":", 2)
        if not path.startswith("/") or ".." in PurePosixPath(path).parts:
            raise ValueError("invalid cgroup membership")
        for controller in controllers.split(","):
            memberships[controller] = PurePosixPath(path)
    # A hybrid host's CPU controller can remain on v1 while other controllers use v2.
    unified = "cpu" not in memberships
    member = memberships.get("" if unified else "cpu")
    if member is None:
        raise ValueError("CPU cgroup membership unavailable")
    budgets = []
    found = False
    for line in (_PROC_SELF / "mountinfo").read_text().splitlines():
        before, after = line.split(" - ", 1)
        fields, filesystem = before.split(), after.split()
        if unified:
            matches = filesystem[0] == "cgroup2"
        else:
            matches = filesystem[0] == "cgroup" and "cpu" in filesystem[2].split(",")
        if not matches:
            continue
        root = PurePosixPath(_mount_path(fields[3]))
        try:
            relative = member.relative_to(root)
        except ValueError:
            continue
        mount = Path(_mount_path(fields[4]))
        if not mount.is_absolute() or ".." in root.parts:
            raise ValueError("invalid cgroup mount")
        directory = mount.joinpath(*relative.parts)
        found = True
        while True:
            try:
                budget = _quota(directory, unified=unified)
            except FileNotFoundError:
                # A v2 cgroup without cpu.max has no CPU controller and so no limit
                # at that level: the hierarchy root, or a systemd user session
                # without CPU delegation. Its parents can still set one. A missing
                # cgroup directory means the membership did not resolve.
                if not (unified and directory.is_dir()):
                    raise
                budget = None
            if budget is not None:
                budgets.append(budget)
            if directory == mount:
                break
            directory = directory.parent
    if not found:
        raise ValueError("CPU cgroup mount unavailable")
    return min(budgets) if budgets else None


def effective_cpu_budget() -> float:
    """Return a conservative budget; unknown container or platform limits keep one thread."""
    if sys.platform != "linux":
        return 1
    try:
        affinity = len(os.sched_getaffinity(0))
        if affinity <= 0:
            return 1
        available = float(min(affinity, os.cpu_count() or affinity))
        quota = _linux_cpu_quota()
        return min(available, quota) if quota is not None else available
    except (OSError, ValueError, IndexError, AttributeError, OverflowError):
        return 1


def training_worker_environment(env: dict[str, str], world_size: int) -> dict[str, str]:
    result = dict(env)
    if world_size > 1 and "OMP_NUM_THREADS" not in result:
        threads = max(1, min(_MAX_THREADS_PER_RANK, math.floor(effective_cpu_budget() / world_size)))
        result["OMP_NUM_THREADS"] = str(threads)
    return result
