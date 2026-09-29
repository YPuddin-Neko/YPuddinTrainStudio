"""One-time rewrites of saved project configs and presets after a default they stored changes."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DORA_INPUT = "config.dora_axis_input"


def _rewrite(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def migrate_dora_axis(c: Any) -> None:
    """Saved configs that name DoRA's output axis switch to the input axis, now the default, once.

    They name it because it was the default, and DoRA files trained on it render off in ComfyUI,
    Forge and A1111. Jobs keep their own config snapshots, so queued runs and resume points are
    unaffected.
    """
    if c.db.get_kv(DORA_INPUT, 0) >= 1:
        return
    paths: list[Path] = []
    for version in c.db.fetchall("SELECT id, project_id FROM project_versions"):
        try:
            paths.append(c.config_path(version["project_id"], version["id"]))
        except Exception:  # noqa: BLE001 - a version without a usable folder has no config to change
            continue
    paths += sorted((c.data_root / "presets").glob("*.json"))
    changed = 0
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        adapter = data.get("adapter") if isinstance(data, dict) else None
        if isinstance(adapter, dict) and adapter.get("dora_axis") == "output":
            adapter["dora_axis"] = "input"
            _rewrite(path, data)
            changed += 1
    if changed:
        log.info("set the DoRA axis of %d saved configs to input", changed)
    c.db.set_kv(DORA_INPUT, 1)
