"""Shared service state."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bus import EventBus
from .db import Database
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
        return base

    def save_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        cur = self.settings()
        for k, v in patch.items():
            if isinstance(v, dict) and isinstance(cur.get(k), dict):
                cur[k].update(v)
            else:
                cur[k] = v
        self.settings_path.write_text(json.dumps(cur, indent=2, ensure_ascii=False), encoding="utf-8")
        return cur

    def project_dir(self, project_id: str) -> Path:
        return self.data_root / "projects" / project_id

    def runs_dir(self, project_id: str | None) -> Path:
        return self.project_dir(project_id) / "runs" if project_id else self.data_root / "runs"

    def is_allowed(self, path: Path) -> bool:
        p = path.resolve()
        roots = [self.data_root.resolve(), *[r.resolve() for r in self.allowed_roots]]
        return any(p == r or r in p.parents for r in roots) or not self.allowed_roots and True
