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


# Jobs and model tests whose folders are being removed outside the database lock; their records go last.
deleting_jobs: set[str] = set()


def deletion_roots(context: Any, extra: list[Path] | tuple[Path, ...] = ()) -> list[Path]:
    """Folders in which deletions may remove the subfolders the studio made for a project or job:
    the data folder, the custom folders set in settings, explicitly allowed roots and ``extra``."""
    paths = context.settings()["paths"]
    configured = [paths.get(key) for key in ("cache_dir", "output_dir", "state_dir", "samples_dir", "logs_dir")]
    roots = []
    for root in (context.data_root, *context.allowed_roots, *(Path(value) for value in configured if value), *extra):
        try:
            roots.append(Path(root).expanduser().resolve())
        except (OSError, RuntimeError):
            continue
    return list(dict.fromkeys(roots))


def studio_root(context: Any, job: dict, folder: Path) -> Path | None:
    """The folder a job folder was made in: ``<root>/<project>/<version>/<job>``, or ``<root>/<job>`` for a
    job outside any project. Settings may have changed since, so the folder's own shape says where it is."""
    if folder.name != job["id"]:
        return None
    if not job.get("project_id"):
        return folder.parent
    try:
        label = context.version_label(job["project_id"], job.get("version_id"))
    except Exception:  # noqa: BLE001 - a missing version leaves only the known roots
        return None
    if folder.parent.name == label and folder.parent.parent.name == job["project_id"]:
        return folder.parent.parent.parent
    return None


def owning_root(folder: Path, roots: list[Path]) -> Path | None:
    """The innermost root that holds the resolved folder."""
    resolved = folder.resolve()
    return max((root for root in roots if resolved.is_relative_to(root)), key=lambda root: len(root.parts), default=None)


def linked_below(folder: Path, root: Path) -> bool:
    """Whether reaching ``folder`` passes a symbolic link at or below ``root``.

    Links above the root, such as macOS's /tmp, only lead to where the root itself is.
    """
    path = Path(folder).absolute()
    while path.resolve() != root:
        if path.is_symlink() or path.parent == path:
            return True
        path = path.parent
    return False


def folder_problem(folder: Path, roots: list[Path]) -> Literal["outside", "linked", "unavailable"] | None:
    """Why a folder cannot be removed safely: it lies outside every root or is a root itself, it is
    reached through a link below its root, or its root cannot be reached (an unplugged drive)."""
    root = owning_root(folder, roots)
    if root is None or folder.resolve() == root:
        return "outside"
    try:
        if not root.is_dir():
            return "unavailable"
    except OSError:
        return "unavailable"
    return "linked" if linked_below(folder, root) else None


def removal_problem(
    context: Any,
    owner_id: str,
    folders: list[Path],
    *,
    owner_type: Literal["job", "project"] = "job",
    roots: list[Path] | None = None,
) -> Literal["outside", "shared"] | None:
    """Reject linked/out-of-bounds folders and files owned by jobs outside this deletion.

    A folder whose root cannot be reached is left for the caller to skip.
    """
    if roots is None:
        job = context.db.fetchone("SELECT * FROM jobs WHERE id=?", (owner_id,)) if owner_type == "job" else None
        extra = [root for folder in folders if job and (root := studio_root(context, job, folder))]
        roots = deletion_roots(context, extra)
    targets = []
    for folder in folders:
        problem = folder_problem(folder, roots)
        if problem in ("outside", "linked"):
            return "outside"
        if problem is None:
            targets.append(folder.resolve())
    if not targets:
        return None
    column = "project_id" if owner_type == "project" else "id"
    others = {
        path.resolve()
        for row in context.db.fetchall(
            f"SELECT run_dir, samples_dir, config_json FROM jobs WHERE {column} IS NULL OR {column}!=?",
            (owner_id,),
        )
        for path in job_directories(row)
    }
    return "shared" if any(path.is_relative_to(target) for path in others for target in targets) else None
