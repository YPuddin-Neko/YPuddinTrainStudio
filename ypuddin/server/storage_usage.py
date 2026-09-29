"""How much disk the studio's own files take: the trainer, each project version, model weights and other data.

Folders in the data root are the studio's. Outside it only what the studio wrote or registered is counted, so a
shared folder, such as a models folder another program also uses, adds only the studio's files. Every file is
counted once, in the most specific part holding it. Scans are kept for a few minutes.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .db import now
from .job_paths import owned_job_directories

SOURCE_ROOT = Path(__file__).resolve().parents[2]
KEEP_SECONDS = 300
GROUPS = ("trainer", "projects", "models", "other")


@dataclass
class Part:
    group: str  # one of GROUPS
    paths: list[Path]
    version: dict[str, str] | None = None  # a project version's id, name and project name
    bytes: int = 0


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _files_size(path: Path, skip: set[str]) -> int:
    """Bytes of the files under ``path`` without following links, leaving out the paths in ``skip``."""
    try:
        if path.is_file() and not path.is_symlink():
            return path.stat().st_size
    except OSError:
        return 0
    total, pending = 0, [str(path)]
    while pending:
        folder = pending.pop()
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    if entry.path in skip:
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def _parts(context) -> list[Part]:
    from ypuddin.runtime_profiles import PROFILES

    settings = context.settings()["paths"]
    data_root = context.data_root.resolve()
    cache_root = _resolved(settings["cache_dir"])
    custom_cache = cache_root != (data_root / "cache").resolve()
    jobs: dict[str | None, list[dict]] = defaultdict(list)
    for job in context.db.fetchall("SELECT * FROM jobs"):
        jobs[job["version_id"]].append(job)

    # The trainer: its program files and the Python environments and package caches it runs from.
    trainer = [
        SOURCE_ROOT,
        data_root / "environment",
        _resolved(settings.get("bootstrap_env_dir") or SOURCE_ROOT / "environment"),
    ]
    if sys.prefix != sys.base_prefix:
        trainer.append(_resolved(sys.prefix))
    if custom_cache:
        trainer += [cache_root / "packages" / profile for profile in PROFILES]
    parts = [Part("trainer", trainer)]

    # Projects: each version's folder, its cache in a custom cache folder and its jobs' folders anywhere.
    for project in context.db.fetchall("SELECT id, name FROM projects ORDER BY created_at"):
        pid = project["id"]
        parts.append(Part("projects", [context.project_dir(pid).resolve()]))
        for version in context.db.fetchall(
            "SELECT id, name, number FROM project_versions WHERE project_id=? ORDER BY number, created_at",
            (pid,),
        ):
            vid = version["id"]
            try:
                paths = [context.version_dir(pid, vid).resolve()]
            except Exception:  # noqa: BLE001 - a version whose folder is unusable is counted with its project
                continue
            if custom_cache:
                paths.append(cache_root / pid / vid)
            paths += [_resolved(path) for job in jobs[vid] for path in owned_job_directories(job)]
            name = f"v{version['number']}" if version["number"] else version["name"]
            parts.append(
                Part("projects", paths, {"version_id": vid, "name": name, "project_name": project["name"]})
            )

    # Model weights: registered models wherever they are, the tagging and mask models, and all of the models
    # folder when it is the studio's own.
    models_root = _resolved(settings["models_dir"])
    models = [_resolved(row["path"]) for row in context.db.fetchall("SELECT DISTINCT path FROM models")]
    models += [models_root / "tagger", models_root / "mask"]
    if data_root in models_root.parents:
        models.append(models_root)
    parts.append(Part("models", models))

    # Everything else the studio keeps: the rest of the data root, the shared and service caches in a custom
    # cache folder, and the folders of jobs outside any project.
    other = [data_root] + ([cache_root / "shared", cache_root / "service"] if custom_cache else [])
    other += [_resolved(path) for job in jobs[None] for path in owned_job_directories(job)]
    parts.append(Part("other", other))
    return parts


def _measure(parts: list[Part]) -> None:
    # A path belongs to the first part naming it; a folder leaves out the paths other parts hold inside it.
    owner: dict[Path, Part] = {}
    for part in parts:
        for path in part.paths:
            owner.setdefault(path, part)
    nested: dict[Path, set[str]] = defaultdict(set)
    for path in owner:
        for parent in path.parents:
            if parent in owner:
                nested[parent].add(str(path))
    for path, part in owner.items():
        if path.exists():
            part.bytes += _files_size(path, nested[path])


def _usage(context) -> dict[str, Any]:
    parts = _parts(context)
    _measure(parts)
    groups = dict.fromkeys(GROUPS, 0)
    for part in parts:
        groups[part.group] += part.bytes
    return {
        "scanned_at": now(),
        "groups": groups,
        "versions": [{**part.version, "bytes": part.bytes} for part in parts if part.version and part.bytes],
    }


_lock = threading.Lock()
_kept: dict[int, tuple[float, dict[str, Any]]] = {}


def storage_usage(context, *, refresh: bool = False) -> dict[str, Any]:
    """The latest scan, or a new one when asked or when it is older than a few minutes."""
    with _lock:
        kept = _kept.get(id(context))
        if kept and not refresh and time.monotonic() - kept[0] < KEEP_SECONDS:
            return kept[1]
        result = _usage(context)
        _kept[id(context)] = (time.monotonic(), result)
        return result
