"""Shared service state."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bus import EventBus
from .db import Database
from .import_progress import ImportProgressStore
from .models import Settings, SettingsTagging
from .supervisor import JobSupervisor

DEFAULT_SETTINGS: dict[str, Any] = {
    "cache": {"thumbnail_max_gb": 1.0},
    "downloads": {"pypi": "auto", "pytorch": "auto", "fallback": True},
    "tagging": SettingsTagging().model_dump(),
    "paths": {
        "bootstrap_env_dir": "",
        "data_root": "",
        "cache_dir": "",
        "models_dir": "",
        "output_dir": "",
        "state_dir": "",
        "samples_dir": "",
        "logs_dir": "",
    },
    "server": {"host": "127.0.0.1", "port": 8123, "open_browser": True},
    "ui": {"language": "zh-CN", "theme": "system", "telemetry_interval": 2.5, "onboarding_completed": False},
    "network": {
        "proxy_mode": "system",
        "proxy_url": "",
        "proxy_username": "",
        "proxy_password_configured": False,
    },
}


def _cache_settings(section: dict[str, Any]) -> dict[str, Any]:
    """Earlier releases stored the thumbnail limit in MiB as ``thumbnail_max_mb``."""
    section = dict(section)
    mib = section.pop("thumbnail_max_mb", None)
    if "thumbnail_max_gb" not in section and isinstance(mib, int) and not isinstance(mib, bool):
        section["thumbnail_max_gb"] = min(1024.0, max(0.1, round(mib / 1024, 2)))
    return section


@dataclass
class ServiceContext:
    data_root: Path
    db: Database
    bus: EventBus
    supervisor: JobSupervisor
    allowed_roots: list[Path] = field(default_factory=list)
    _settings_lock: Any = field(default_factory=threading.RLock, init=False, repr=False)
    versions: Any = field(default=None, init=False, repr=False)
    import_progress: ImportProgressStore = field(default_factory=ImportProgressStore, init=False, repr=False)
    upload_sessions: Any = field(default=None, init=False, repr=False)
    _active_imports: int = field(default=0, init=False, repr=False)

    @contextmanager
    def import_admission(self):
        """Protect receiving bodies too, including callers without a progress identifier.

        Restart claims maintenance under the same lock. A request is either admitted
        before that claim and blocks restart, or rejected before reading any files.
        """
        from .errors import ApiError

        with self.db.lock:
            if self.db.get_kv("environment.maintenance", {}).get("restarting"):
                raise ApiError(
                    "Service is restarting; retry after reconnecting", code="service.restarting", status=409
                )
            self._active_imports += 1
        try:
            yield
        finally:
            with self.db.lock:
                self._active_imports -= 1

    def __post_init__(self) -> None:
        from .thumbnail_cache import ThumbnailCache
        from .upload_sessions import UploadSessionStore
        from .versions import VersionManager

        self.thumbnails = ThumbnailCache(self)
        self.versions = VersionManager(self)
        self.upload_sessions = UploadSessionStore(self.data_root / ".upload-sessions", self.import_progress)

    @property
    def settings_path(self) -> Path:
        return self.data_root / "settings.json"

    def pending_data_root(self) -> str | None:
        """Return a data root saved for the next launcher restart, if any."""
        try:
            value = json.loads(self.settings_path.read_text(encoding="utf-8")).get("_pending_data_root")
        except (OSError, ValueError, TypeError):
            return None
        return str(value) if isinstance(value, str) and value.strip() else None

    def settings(self) -> dict[str, Any]:
        with self._settings_lock:
            return self._settings()

    def _settings(self) -> dict[str, Any]:
        from .network import PASSWORD_REVISION, ProxyCredentials

        base = json.loads(json.dumps(DEFAULT_SETTINGS))
        base["paths"]["data_root"] = str(self.data_root)
        base["paths"]["cache_dir"] = str(self.data_root / "cache")
        base["paths"]["models_dir"] = str(self.data_root / "models")
        base["paths"]["output_dir"] = str(self.data_root / "runs")
        saved = {}
        if self.settings_path.exists():
            saved = json.loads(self.settings_path.read_text(encoding="utf-8"))
            for k, v in saved.items():
                if k == "cache" and isinstance(v, dict):
                    v = _cache_settings(v)
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k].update(v)
                else:
                    base[k] = v
        if "onboarding_completed" not in saved.get("ui", {}):
            existing = self.db.fetchone(
                "SELECT EXISTS(SELECT 1 FROM projects) OR EXISTS(SELECT 1 FROM jobs) AS established"
            )
            base["ui"]["onboarding_completed"] = bool(saved.get("ui")) or bool(existing["established"])
        if "output_mode" not in base["paths"]:
            base["paths"]["output_mode"] = (
                "project"
                if Path(base["paths"]["output_dir"]).expanduser().resolve()
                == (self.data_root / "runs").resolve()
                else "custom"
            )
        base["paths"]["data_root"] = str(self.data_root)
        for key in ("cache_dir", "models_dir", "output_dir"):
            base["paths"][key] = str(Path(base["paths"][key]).expanduser().resolve())
        for key in ("state_dir", "samples_dir", "logs_dir"):
            value = base["paths"].get(key, "")
            base["paths"][key] = str(Path(value).expanduser().resolve()) if value else ""
        base["network"].pop("proxy_password", None)
        revision = base.pop(PASSWORD_REVISION, "")
        base["network"]["proxy_password_configured"] = bool(
            ProxyCredentials(self.data_root).password(revision)
        )
        return base

    def service_cache_dir(self, kind: str) -> Path:
        """Preserve default locations; custom roots apply to future cache writes."""
        if kind not in {"index", "thumbnails"}:
            raise ValueError("Unknown service cache kind")
        root = Path(self.settings()["paths"]["cache_dir"])
        if root.resolve() == (self.data_root / "cache").resolve():
            return self.data_root / ("thumbs" if kind == "thumbnails" else "cache")
        return root / "service" / kind

    def package_cache_dir(self, profile: str) -> Path:
        from ypuddin.runtime_profiles import profile_root

        root = Path(self.settings()["paths"]["cache_dir"])
        if root.resolve() == (self.data_root / "cache").resolve():
            return profile_root(self.data_root, profile) / "cache"
        return root / "packages" / profile

    def save_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        with self._settings_lock:
            return self._save_settings(patch)

    def _save_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        from .network import PASSWORD_REVISION, ProxyCredentials, validate_proxy_settings

        patch = dict(patch)
        if isinstance(patch.get("cache"), dict):
            patch["cache"] = _cache_settings(patch["cache"])
        if PASSWORD_REVISION in patch:
            raise ValueError("Internal settings revisions cannot be changed through the API")
        password = None
        if "network" in patch:
            if not isinstance(patch["network"], dict):
                raise ValueError("network settings must be an object")
            patch["network"] = dict(patch["network"])
            if "proxy_password" in patch["network"]:
                password = patch["network"].pop("proxy_password")
                if (
                    not isinstance(password, str)
                    or len(password) > 4096
                    or any(ord(c) < 32 for c in password)
                ):
                    raise ValueError("Invalid proxy password")
        cur = self.settings()
        for k, v in patch.items():
            if isinstance(v, dict) and isinstance(cur.get(k), dict):
                cur[k].update(v)
            else:
                cur[k] = v
        paths_patch = patch.get("paths", {})
        if "output_dir" in paths_patch and "output_mode" not in paths_patch:
            cur["paths"]["output_mode"] = (
                "project"
                if Path(paths_patch["output_dir"]).expanduser().resolve()
                == (self.data_root / "runs").resolve()
                else "custom"
            )
        if "data_root" not in paths_patch and (pending := self.pending_data_root()):
            cur["paths"]["data_root"] = pending
        requested_data_root = Path(cur["paths"]["data_root"]).expanduser().resolve()
        pending_data_root = None if requested_data_root == self.data_root.resolve() else str(requested_data_root)
        # Keep the running context bound to its current root. The launcher reads
        # this pending value after restart, so the UI can save it without changing
        # the database and model paths underneath the live service.
        cur["paths"]["data_root"] = str(self.data_root)
        for key in ("cache_dir", "output_dir", "models_dir"):
            value = cur["paths"][key]
            if not value or not str(value).strip():
                raise ValueError(f"paths.{key} must not be empty")
            cur["paths"][key] = str(Path(value).expanduser().resolve())
        for key in ("state_dir", "samples_dir", "logs_dir"):
            value = cur["paths"].get(key, "")
            if not isinstance(value, str):
                raise ValueError(f"paths.{key} must be a path string")
            cur["paths"][key] = str(Path(value.strip()).expanduser().resolve()) if value.strip() else ""
        env_root = cur["paths"].get("bootstrap_env_dir", "").strip()
        cur["paths"]["bootstrap_env_dir"] = str(Path(env_root).expanduser().absolute()) if env_root else ""
        cur["network"] = validate_proxy_settings(cur["network"])
        cur = Settings.model_validate(cur).model_dump()
        revision = (
            json.loads(self.settings_path.read_text("utf-8")).get(PASSWORD_REVISION, "")
            if self.settings_path.exists()
            else ""
        )
        if password is not None:
            next_revision = uuid.uuid4().hex
            ProxyCredentials(self.data_root).prepare(password, revision, next_revision)
            revision = next_revision
            cur["network"]["proxy_password_configured"] = bool(password)
        else:
            cur["network"]["proxy_password_configured"] = bool(
                ProxyCredentials(self.data_root).password(revision)
            )
        # This replacement is the single commit point for both the public policy
        # and the private password. Failed/uncommitted preparations remain inactive.
        stored = {**cur, PASSWORD_REVISION: revision}
        if pending_data_root:
            stored["_pending_data_root"] = pending_data_root
        else:
            stored.pop("_pending_data_root", None)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=self.data_root, suffix=".tmp", delete=False, encoding="utf-8"
        ) as f:
            temporary = Path(f.name)
            try:
                f.write(json.dumps(stored, indent=2, ensure_ascii=False))
                f.flush()
                os.fsync(f.fileno())
                f.close()
                temporary.replace(self.settings_path)
            finally:
                temporary.unlink(missing_ok=True)
        try:
            ProxyCredentials(self.data_root).finish(revision)
        except OSError:
            # The commit already succeeded. An inactive old credential can safely
            # remain in the private file until the next save retries this cleanup.
            pass
        return cur

    def project_dir(self, project_id: str) -> Path:
        project = self.db.fetchone("SELECT layout_version FROM projects WHERE id=?", (project_id,))
        return (
            self.data_root
            / ("project" if project and project["layout_version"] >= 2 else "projects")
            / project_id
        )

    def project_layout(self, project_id: str) -> int:
        project = self.db.fetchone("SELECT layout_version FROM projects WHERE id=?", (project_id,))
        return project["layout_version"] if project else 1

    def resolve_version(self, project_id: str, version_id: str | None = None) -> dict[str, Any]:
        from .errors import NotFound

        project = self.db.fetchone("SELECT * FROM projects WHERE id=?", (project_id,))
        if not project:
            raise NotFound(f"project {project_id} not found", code="project.not_found")
        row = self.db.fetchone(
            "SELECT * FROM project_versions WHERE id=? AND project_id=?",
            (version_id or project["active_version_id"], project_id),
        )
        if not row:
            raise NotFound("project version not found", code="version.not_found")
        return row

    def version_dir(self, project_id: str, version_id: str | None = None) -> Path:
        from .errors import ApiError

        root = self.project_dir(project_id)
        version = self.resolve_version(project_id, version_id)
        path = (
            root / f"v{version['number']}"
            if self.project_layout(project_id) >= 2
            else root / "versions" / version["id"]
        )
        if any(p.is_symlink() for p in (root.parent, root, path.parent, path)):
            raise ApiError(
                "managed version directory cannot be redirected with symbolic links", code="version.path"
            )
        return path

    def dataset_dir(self, project_id: str, version_id: str | None = None, *, is_reg: bool = False) -> Path:
        name = "reg" if is_reg else "traindata" if self.project_layout(project_id) >= 2 else "datasets"
        return self.version_dir(project_id, version_id) / name

    def reg_dir(self, project_id: str, version_id: str | None = None) -> Path:
        return self.dataset_dir(project_id, version_id, is_reg=True)

    def version_label(self, project_id: str, version_id: str | None = None) -> str:
        version = self.resolve_version(project_id, version_id)
        return f"v{version['number']}" if self.project_layout(project_id) >= 2 else version["id"]

    def samples_dir(self, project_id: str, version_id: str | None = None) -> Path:
        from .job_layout import SAMPLES

        return self.version_dir(project_id, version_id) / SAMPLES

    def records_root(self, project_id: str, version_id: str | None = None) -> Path:
        from .job_layout import RECORDS

        return self.version_dir(project_id, version_id) / RECORDS

    def default_runs_dir(self, project_id: str, version_id: str | None = None) -> Path:
        return self.version_dir(project_id, version_id) / (
            "output" if self.project_layout(project_id) >= 2 else "runs"
        )

    def inherits_output_dir(
        self, project_id: str | None, version_id: str | None, requested: str | None
    ) -> bool:
        if not requested or requested == "outputs/run":
            return True
        path = Path(requested).expanduser().resolve()
        defaults = [self.runs_dir(project_id, version_id)]
        if project_id:
            defaults.append(self.default_runs_dir(project_id, version_id))
        return any(path == root.resolve() for root in defaults)

    def job_output_dir(
        self, project_id: str | None, version_id: str | None, job_id: str, requested: str | None = None
    ) -> Path:
        """Where the job's products go: output/<job> in its version unless settings or the job say otherwise."""
        if self.inherits_output_dir(project_id, version_id, requested):
            return self.runs_dir(project_id, version_id) / job_id
        root = Path(requested).expanduser().resolve()
        if (sibling := self._reused_job_path(root, project_id, version_id, job_id, "output_dir")) is not None:
            return sibling
        if project_id:
            root = root / project_id / self.version_label(project_id, version_id)
        return root / job_id

    def job_records_dir(self, project_id: str, version_id: str | None, job_id: str) -> Path:
        """The job's own folder: its config, logs, events and resume points, apart from its products."""
        from .job_layout import records_dir

        return records_dir(self.version_dir(project_id, version_id), job_id)

    def _reused_job_path(
        self, root: Path, project_id: str | None, version_id: str | None, job_id: str, kind: str
    ) -> Path | None:
        """A path copied from an earlier job's saved config gets this job's sibling, never a folder
        inside the earlier job's, which would be removed with that job's files."""
        from .job_layout import renamed_for_job
        from .job_paths import event_file, job_config, state_directory

        for part in reversed(root.parts):
            previous = self.db.fetchone(
                "SELECT * FROM jobs WHERE id=? AND project_id IS ? AND version_id IS ?",
                (part, project_id, version_id),
            )
            if not previous:
                continue
            products = job_config(previous).get("checkpoint", {}).get("output_dir") or previous["run_dir"]
            owned = {
                "output_dir": [Path(products), Path(previous["run_dir"])],
                "state_dir": [state_directory(previous)],
                "logs_dir": [event_file(previous).parent],
                "samples_dir": [Path(previous["samples_dir"])] if previous.get("samples_dir") else [],
            }[kind]
            if root in owned:
                return renamed_for_job(root, part, job_id)
        return None

    def job_storage_dir(
        self,
        project_id: str | None,
        version_id: str | None,
        job_id: str,
        kind: str,
        run_dir: Path,
        requested: str | None = None,
    ) -> Path:
        """Where the job keeps previews, resume points or logs; `run_dir` is its records folder."""
        configured = requested or self.settings()["paths"].get(kind, "")
        if not configured:
            if not project_id:
                return run_dir / "samples" if kind == "samples_dir" else run_dir
            from .job_layout import RESUME, samples_dir

            if kind == "samples_dir":
                return samples_dir(self.version_dir(project_id, version_id), job_id)
            return run_dir / RESUME if kind == "state_dir" else run_dir
        root = Path(configured).expanduser().resolve()
        if (sibling := self._reused_job_path(root, project_id, version_id, job_id, kind)) is not None:
            return sibling
        if project_id:
            root = root / project_id / self.version_label(project_id, version_id)
        return root / job_id

    def training_cache_dir(
        self,
        project_id: str | None,
        version_id: str | None,
        requested: str | None,
    ) -> Path:
        defaults = [self.cache_dir(project_id, version_id)]
        if project_id:
            defaults.append(self.version_dir(project_id, version_id) / "cache")
        if requested and Path(requested).expanduser().resolve() not in [path.resolve() for path in defaults]:
            return Path(requested).expanduser().resolve()
        return self.cache_dir(project_id, version_id)

    def config_path(self, project_id: str, version_id: str | None = None) -> Path:
        version = self.resolve_version(project_id, version_id)
        root = (
            self.project_dir(project_id)
            if version["legacy_layout"]
            else self.version_dir(project_id, version["id"])
        )
        return root / "config.json"

    def runs_dir(self, project_id: str | None, version_id: str | None = None) -> Path:
        paths = self.settings()["paths"]
        configured = Path(paths["output_dir"])
        if project_id:
            vid = self.resolve_version(project_id, version_id)["id"]
            if paths["output_mode"] == "custom":
                return configured / project_id / self.version_label(project_id, vid)
            return self.default_runs_dir(project_id, vid)
        if paths["output_mode"] == "custom":
            return configured
        return self.data_root / "runs"

    def cache_dir(self, project_id: str | None, version_id: str | None = None) -> Path:
        """Jobs share caches only within their bound project version."""
        configured = Path(self.settings()["paths"]["cache_dir"])
        if project_id:
            vid = self.resolve_version(project_id, version_id)["id"]
            if configured.resolve() != (self.data_root / "cache").resolve():
                return configured / project_id / vid
            return self.version_dir(project_id, vid) / "cache"
        if configured.resolve() != (self.data_root / "cache").resolve():
            return configured / "shared"
        return self.data_root / "cache" / "shared"

    def is_allowed(self, path: Path) -> bool:
        p = path.resolve()
        roots = [self.data_root.resolve(), *[r.resolve() for r in self.allowed_roots]]
        return any(p == r or r in p.parents for r in roots) or not self.allowed_roots and True
