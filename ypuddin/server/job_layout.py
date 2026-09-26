"""Where a project job keeps its files, and moving jobs saved before this layout into it.

Inside a project version directory:

    output/<job>/        LoRA and model products, nothing else
    jobs/<job>/          records: job-config.toml, config.toml, run.log, events.jsonl, control files
    jobs/<job>/resume/   resume points
    samples/<job>/       training previews

Cache preparation and model tests make no products, so all of their files stay in jobs/<job>/.
Settings and job-specific paths can move products, resume points, previews or logs elsewhere. Job
creation, the settings page and the parameter form all take their defaults from here.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

RECORDS = "jobs"
RESUME = "resume"
SAMPLES = "samples"
# What a job wrote next to its products before this layout; anything else found there stays put.
RECORD_NAMES = frozenset(
    {
        "job-config.toml",
        "config.toml",
        "run.log",
        "events.jsonl",
        "control",
        "xyz-request.json",
        "tensorboard",
        "wandb",
    }
)
MIGRATED = "layout.job_records"


def records_dir(version: Path, job_id: str) -> Path:
    return version / RECORDS / job_id


def resume_dir(version: Path, job_id: str) -> Path:
    return records_dir(version, job_id) / RESUME


def samples_dir(version: Path, job_id: str) -> Path:
    return version / SAMPLES / job_id


def makes_products(job_type: str) -> bool:
    return job_type == "train"


def renamed_for_job(path: Path, old_id: str, new_id: str) -> Path | None:
    """The same place for another job: the last path part naming the old job names the new one."""
    parts = path.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == old_id:
            return Path(*parts[:index], new_id, *parts[index + 1 :])
    return None


class PathMap:
    """Rewrites paths saved in job records after the files they name moved."""

    def __init__(self) -> None:
        self.moves: dict[str, str] = {}

    def __bool__(self) -> bool:
        return bool(self.moves)

    def add(self, old: Path, new: Path) -> None:
        self.moves[str(old)] = str(new)

    def path(self, value: str) -> str:
        # The longest moved directory that contains the path wins; only whole path parts match.
        candidate = value
        while candidate:
            if candidate in self.moves:
                return self.moves[candidate] + value[len(candidate) :]
            cut = candidate.rfind(os.sep)
            if cut <= 0:
                break
            candidate = candidate[:cut]
        return value

    def apply(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.path(value)
        if isinstance(value, list):
            return [self.apply(item) for item in value]
        if isinstance(value, dict):
            return {key: self.apply(item) for key, item in value.items()}
        return value


def migrate_job_files(c: Any) -> None:
    """Move the records and resume points of jobs saved next to their products into jobs/<job>/.

    Runs at startup before the queue does, so no worker holds these files. Files only move by rename
    inside their version directory, and every saved path to them (job records, event logs, resume
    requests, project configs) is rewritten. A job whose files cannot all move keeps its old paths
    and is tried again at the next start.
    """
    if c.db.get_kv(MIGRATED, 0) >= 1:
        return
    moved = PathMap()
    patches: dict[str, dict[str, Any]] = {}
    pending = False
    for job in c.db.fetchall("SELECT * FROM jobs WHERE project_id IS NOT NULL AND run_dir IS NOT NULL"):
        try:
            patch = _move_job(c, job, moved)
        except Exception:  # noqa: BLE001 - one job must not stop the others from moving
            log.exception("could not move the files of job %s into the jobs folder", job["id"])
            patch = False
        if patch is False:
            pending = True
        elif patch:
            patches[job["id"]] = patch
    if moved:
        for jid, patch in patches.items():
            _rewrite_events(Path(patch["events"]), moved, jid)
        _rewrite_project_configs(c, moved)
    if patches or moved:
        _rewrite_jobs(c, moved, patches)
    if patches:
        log.info("moved the records of %d jobs into their version's jobs folder", len(patches))
    if not pending:
        c.db.set_kv(MIGRATED, 1)


def _children(path: Path) -> list[Path]:
    try:
        return sorted(path.iterdir())
    except OSError:
        return []


def _alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    import psutil

    try:
        return "ypuddin" in " ".join(psutil.Process(pid).cmdline())
    except (psutil.Error, OSError):
        return False


def _move_job(c: Any, job: dict[str, Any], moved: PathMap) -> dict[str, Any] | bool | None:
    """Move one job's files; returns what its record needs, None when it needs nothing, False to retry."""
    from .job_paths import event_file

    jid = job["id"]
    try:
        version = c.version_dir(job["project_id"], job.get("version_id"))
        products = c.default_runs_dir(job["project_id"], job.get("version_id"))
    except Exception:  # noqa: BLE001 - the project or version is gone; nothing to reorganize
        return None
    old = Path(job["run_dir"])
    if old != products / jid or old.is_symlink():
        return None  # Already in this layout, or kept under a custom output root.
    if job["status"] in ("running", "pausing", "cancelling") and _alive(job.get("pid")):
        return False
    new = records_dir(version, jid)
    config = json.loads(job.get("config_json") or "{}")
    events = event_file(job)
    records = RECORD_NAMES | {events.name}
    products_job = makes_products(job["type"])
    plan: list[tuple[Path, Path]] = []
    for source in _children(old):
        state = source.name.startswith("state-") and source.is_dir() and not source.is_symlink()
        if products_job and state:
            target = new / RESUME / source.name
        elif not products_job or source.name in records:
            target = new / source.name
        else:
            continue  # Products, and anything not written by the job itself, stay in output/.
        if target.exists() or target.is_symlink():
            log.warning("not moving %s: %s already exists", source, target)
            return False
        plan.append((source, target))
    done: list[tuple[Path, Path]] = []
    try:
        for source, target in plan:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.rename(source, target)  # Same version directory: a rename, never a copy.
            done.append((source, target))
    except OSError:
        for source, target in reversed(done):
            with contextlib.suppress(OSError):
                os.rename(target, source)
        log.exception("could not move the files of job %s; it keeps its old paths", jid)
        return False
    # Map what earlier, interrupted attempts already moved as well as what moved now.
    if products_job:
        for state in _children(new / RESUME):
            if state.name.startswith("state-"):
                moved.add(old / state.name, state)
        # Records a queued job has not written yet move too, so it never writes them into output/.
        for name in records:
            moved.add(old / name, new / name)
    else:
        moved.add(old, new)
    with contextlib.suppress(OSError):
        old.rmdir()  # Only when nothing is left: a job without products has no output/ folder.
    fields: dict[str, Any] = {"run_dir": str(new)}
    settings: dict[tuple[str, str], str] = {}
    if products_job:
        # The folder itself stays as the products folder, so settings naming it follow the files.
        if (config.get("checkpoint") or {}).get("state_dir") in (None, "", str(old)):
            settings[("checkpoint", "state_dir")] = str(new / RESUME)
        if (config.get("logging") or {}).get("output_dir") == str(old):
            settings[("logging", "output_dir")] = str(new)
    return {"fields": fields, "settings": settings, "events": str(moved.path(str(events)))}


def _write_atomic(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as fp:
        temporary = Path(fp.name)
        try:
            fp.write(text)
        except BaseException:
            fp.close()
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _rewrite_events(path: Path, moved: PathMap, jid: str) -> None:
    """Point saved-state and other file events at their new places; other lines stay byte for byte."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError:
        return
    changed = False
    for index, line in enumerate(lines):
        if jid not in line:
            continue  # Every moved path contains the job id.
        try:
            event = json.loads(line)
        except ValueError:
            continue
        updated = moved.apply(event)
        if updated != event:
            # The same serialization as the trainer's event writer.
            lines[index] = json.dumps(updated, ensure_ascii=False) + ("\n" if line.endswith("\n") else "")
            changed = True
    if changed:
        _write_atomic(path, "".join(lines))


def _rewrite_project_configs(c: Any, moved: PathMap) -> None:
    """A project config may continue from a resume point that moved."""
    for version in c.db.fetchall("SELECT id, project_id FROM project_versions"):
        try:
            path = c.config_path(version["project_id"], version["id"])
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - no config, or one the user must fix; leave it alone
            continue
        updated = moved.apply(data)
        if updated != data:
            _write_atomic(path, json.dumps(updated, indent=2, ensure_ascii=False))


def _rewrite_jobs(c: Any, moved: PathMap, patches: dict[str, dict[str, Any]]) -> None:
    with c.db.transaction():
        for job in c.db.fetchall(
            "SELECT id, run_dir, samples_dir, resume_from, config_json, progress_json, latest_json FROM jobs"
        ):
            patch = patches.get(job["id"], {})
            config = json.loads(job["config_json"] or "{}")
            for (section, key), value in patch.get("settings", {}).items():
                config.setdefault(section, {})[key] = value
            fields: dict[str, Any] = dict(patch.get("fields", {}))
            config = moved.apply(config)
            if config != json.loads(job["config_json"] or "{}"):
                fields["config_json"] = json.dumps(config)
            for name in ("samples_dir", "resume_from"):
                if job[name] and moved.path(job[name]) != job[name]:
                    fields[name] = moved.path(job[name])
            for name in ("progress_json", "latest_json"):
                data = json.loads(job[name] or "{}")
                if (updated := moved.apply(data)) != data:
                    fields[name] = json.dumps(updated)
            if fields:
                c.db.update("jobs", job["id"], fields)
