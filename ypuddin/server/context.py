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
        return self.data_root / "projects" / project_id

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
        path = root / "versions" / self.resolve_version(project_id, version_id)["id"]
        if any(p.is_symlink() for p in (root, root / "versions", path)):
            raise ApiError(
                "managed version directory cannot be redirected with symbolic links", code="version.path"
            )
        return path

    def config_path(self, project_id: str, version_id: str | None = None) -> Path:
        version = self.resolve_version(project_id, version_id)
        root = (
            self.project_dir(project_id)
            if version["legacy_layout"]
            else self.version_dir(project_id, version["id"])
        )
        return root / "config.json"

    def runs_dir(self, project_id: str | None, version_id: str | None = None) -> Path:
        configured = Path(self.settings()["paths"]["output_dir"])
        if project_id:
            vid = self.resolve_version(project_id, version_id)["id"]
            if configured.resolve() != (self.data_root / "runs").resolve():
                return configured / project_id / vid
            return self.version_dir(project_id, vid) / "runs"
        if configured.resolve() != (self.data_root / "runs").resolve():
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
