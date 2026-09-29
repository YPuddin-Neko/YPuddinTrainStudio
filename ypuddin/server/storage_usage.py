"""How much disk the studio's parts take: the trainer, each project version, model weights and other data.

Every file is counted once, in the most specific part whose folder holds it: a version's products are not also
counted in its version folder, nor that in the data root. Scans are kept for a few minutes.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .db import now
from .job_layout import RESUME
from .job_paths import job_config

SOURCE_ROOT = Path(__file__).resolve().parents[2]
KEEP_SECONDS = 300


@dataclass
class Part:
    group: str  # trainer | projects | models | other
    key: str
    paths: list[Path]
    name: str = ""  # the data's own name: a project, a version, a model file or folder
    project_id: str | None = None
    project_name: str | None = None
    version_id: str | None = None
    kind: str | None = None  # a version's: products | resume | data | cache | other
    by_volume: dict[str, int] = field(default_factory=dict)


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _files_size(path: Path, skip: set[Path]) -> int:
    """Bytes of the files under ``path`` without following links, leaving out the folders in ``skip``."""
    try:
        if path.is_file() and not path.is_symlink():
            return path.stat().st_size
    except OSError:
        return 0
    total, pending = 0, [path]
    while pending:
        folder = pending.pop()
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            child = Path(entry.path)
                            if child not in skip:
                                pending.append(child)
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def _volume(path: Path) -> Path | None:
    """The mount point (a drive on Windows) holding ``path``, from its nearest existing folder."""
    probe = path
    while not probe.exists():
        if probe.parent == probe:
            return None
        probe = probe.parent
    try:
        device = probe.stat().st_dev
        while probe.parent != probe and probe.parent.stat().st_dev == device:
            probe = probe.parent
    except OSError:
        return None
    return probe


def _parts(context) -> list[Part]:
    from ypuddin.runtime_profiles import PROFILES

    settings = context.settings()["paths"]
    data_root = context.data_root.resolve()
    parts: list[Part] = []

    # The trainer: its program files and the Python environments and package caches it runs from.
    environments = [
        data_root / "environment",
        _resolved(settings.get("bootstrap_env_dir") or SOURCE_ROOT / "environment"),
    ]
    if sys.prefix != sys.base_prefix:
        environments.append(_resolved(sys.prefix))
    cache_root = _resolved(settings["cache_dir"])
    if cache_root != (data_root / "cache").resolve():
        environments += [cache_root / "packages" / profile for profile in PROFILES]
    parts.append(Part("trainer", "program", [SOURCE_ROOT]))
    parts.append(Part("trainer", "environment", environments))

    # Projects: each version's folder, split by what its files are, plus files moved out by the settings.
    output_mode = settings["output_mode"]
    output_root = _resolved(settings["output_dir"])
    custom = {
        kind: _resolved(settings[kind])
        for kind in ("state_dir", "samples_dir", "logs_dir")
        if settings.get(kind)
    }
    for project in context.db.fetchall("SELECT id, name FROM projects ORDER BY created_at"):
        pid = project["id"]
        project_dir = context.project_dir(pid).resolve()
        parts.append(
            Part(
                "projects",
                f"{pid}:project",
                [project_dir],
                project["name"],
                pid,
                project["name"],
                None,
                "other",
            )
        )
        for version in context.db.fetchall(
            "SELECT id, name, number FROM project_versions WHERE project_id=? ORDER BY number, created_at",
            (pid,),
        ):
            vid = version["id"]
            try:
                root = context.version_dir(pid, vid).resolve()
                label = context.version_label(pid, vid)
            except Exception:  # noqa: BLE001 - a version whose folder is unusable still shows its other files
                continue
            name = f"v{version['number']}" if version["number"] else version["name"]
            owner = {"name": name, "project_id": pid, "project_name": project["name"], "version_id": vid}
            jobs = context.db.fetchall("SELECT * FROM jobs WHERE project_id=? AND version_id=?", (pid, vid))
            products = [root / "output", root / "runs"]
            if output_mode == "custom":
                products.append(output_root / pid / label)
            resume = [path for path in (root / "jobs").glob(f"*/{RESUME}")]
            if "state_dir" in custom:
                resume.append(custom["state_dir"] / pid / label)
            for job in jobs:
                checkpoint = job_config(job).get("checkpoint", {})
                if checkpoint.get("output_dir") and job["type"] == "train":
                    products.append(_resolved(checkpoint["output_dir"]))
                if checkpoint.get("state_dir"):
                    resume.append(_resolved(checkpoint["state_dir"]))
            caches = [root / "cache"]
            if cache_root != (data_root / "cache").resolve():
                caches.append(cache_root / pid / vid)
            others = [root] + [
                custom[kind] / pid / label for kind in ("samples_dir", "logs_dir") if kind in custom
            ]
            for kind, paths in (
                ("products", products),
                ("resume", resume),
                ("data", [root / "traindata", root / "reg", root / "datasets"]),
                ("cache", caches),
                ("other", others),
            ):
                parts.append(Part("projects", f"{vid}:{kind}", paths, kind=kind, **owner))

    # Model weights: each folder or file of the models folder, and registered models kept elsewhere.
    models_root = _resolved(settings["models_dir"])
    if models_root.is_dir():
        for entry in sorted(models_root.iterdir(), key=lambda item: item.name.casefold()):
            if not entry.name.startswith(".") and not entry.is_symlink():
                parts.append(Part("models", f"models:{entry.name}", [entry.resolve()], entry.name))
    for row in context.db.fetchall("SELECT DISTINCT path FROM models"):
        path = _resolved(row["path"])
        if models_root not in path.parents and path != models_root:
            parts.append(Part("models", f"registered:{path}", [path], path.name, kind="external"))

    # Everything else the studio keeps.
    parts += [
        Part("other", "cache", [data_root / "cache", cache_root]),
        Part(
            "other", "thumbnails", [data_root / "thumbs", context.service_cache_dir("thumbnails").resolve()]
        ),
        Part("other", "runs", [data_root / "runs"] + ([output_root] if output_mode == "custom" else [])),
        Part(
            "other",
            "database",
            [data_root / name for name in ("studio.db", "studio.db-wal", "studio.db-shm")],
        ),
        Part("other", "rest", [data_root]),
    ]
    return parts


def _measure(parts: list[Part]) -> list[dict[str, Any]]:
    # A path belongs to the first part naming it; a folder skips the paths other parts hold inside it.
    owner: dict[Path, Part] = {}
    for part in parts:
        for path in dict.fromkeys(part.paths):
            owner.setdefault(path, part)
    claimed = list(owner)
    volumes: dict[Path, str] = {}
    for path, part in owner.items():
        if not path.exists():
            continue
        nested = {other for other in claimed if other != path and path in other.parents}
        size = _files_size(path, nested)
        if not size:
            continue
        mount = _volume(path)
        if mount is None:
            continue
        volume = volumes.setdefault(mount, str(len(volumes)))
        part.by_volume[volume] = part.by_volume.get(volume, 0) + size
    return [{"id": volume, "path": str(mount)} for mount, volume in volumes.items()]


def _usage(context) -> dict[str, Any]:
    parts = _parts(context)
    volumes = _measure(parts)
    for volume in volumes:
        disk = shutil.disk_usage(volume["path"])
        volume.update(total=disk.total, used=disk.used, free=disk.free)
    return {
        "scanned_at": now(),
        "volumes": volumes,
        "parts": [
            {
                "group": part.group,
                "key": part.key,
                "name": part.name,
                "project_id": part.project_id,
                "project_name": part.project_name,
                "version_id": part.version_id,
                "kind": part.kind,
                "bytes": sum(part.by_volume.values()),
                "by_volume": part.by_volume,
            }
            for part in parts
            if part.by_volume
        ],
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
