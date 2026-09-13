"""Shared service state."""

from __future__ import annotations

import json
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bus import EventBus
from .db import Database
from .import_progress import ImportProgressStore
from .models import Settings
from .supervisor import JobSupervisor

DEFAULT_SETTINGS: dict[str, Any] = {
    "paths": {"data_root": "", "cache_dir": "", "models_dir": "", "output_dir": ""},
    "server": {"host": "127.0.0.1", "port": 8765},
    "ui": {"language": "zh-CN", "theme": "system"},
}


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

    def __post_init__(self) -> None:
        from .versions import VersionManager

        self.versions = VersionManager(self)

    @property
    def settings_path(self) -> Path:
        return self.data_root / "settings.json"

    def settings(self) -> dict[str, Any]:
        base = json.loads(json.dumps(DEFAULT_SETTINGS))
        base["paths"]["data_root"] = str(self.data_root)
        base["paths"]["cache_dir"] = str(self.data_root / "cache")
        base["paths"]["models_dir"] = str(self.data_root / "models")
        base["paths"]["output_dir"] = str(self.data_root / "runs")
        if self.settings_path.exists():
            saved = json.loads(self.settings_path.read_text(encoding="utf-8"))
            for k, v in saved.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k].update(v)
                else:
                    base[k] = v
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
        return base

    def save_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        with self._settings_lock:
            return self._save_settings(patch)

    def _save_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
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
        if Path(cur["paths"]["data_root"]).expanduser().resolve() != self.data_root.resolve():
            raise ValueError(
                "data_root is controlled by --data-root; changing it requires a separate data migration"
            )
        for key in ("cache_dir", "output_dir", "models_dir"):
            value = cur["paths"][key]
            if not value or not str(value).strip():
                raise ValueError(f"paths.{key} must not be empty")
            cur["paths"][key] = str(Path(value).expanduser().resolve())
        cur = Settings.model_validate(cur).model_dump()
        with tempfile.NamedTemporaryFile(
            mode="w", dir=self.data_root, suffix=".tmp", delete=False, encoding="utf-8"
        ) as f:
            temporary = Path(f.name)
            try:
                f.write(json.dumps(cur, indent=2, ensure_ascii=False))
                f.close()
                temporary.replace(self.settings_path)
            finally:
                temporary.unlink(missing_ok=True)
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
        return self.version_dir(project_id, version_id) / "samples"

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
        if self.inherits_output_dir(project_id, version_id, requested):
            return self.runs_dir(project_id, version_id) / job_id
        root = Path(requested).expanduser().resolve()
        # Reusing a saved job config creates a sibling, not a child directory
        # that would be removed when deleting the original job's files.
        if self.db.fetchone(
            "SELECT id FROM jobs WHERE run_dir=? AND project_id IS ? AND version_id IS ?",
            (str(root), project_id, version_id),
        ):
            return root.parent / job_id
        if project_id:
            root = root / project_id / self.version_label(project_id, version_id)
        return root / job_id

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
