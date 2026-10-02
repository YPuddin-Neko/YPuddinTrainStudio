"""Preserve historical calculation settings when schema defaults change."""

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
    """Keep legacy DoRA configs on output; never overwrite an explicit axis.

    Version 1 rewrote explicit output settings and missed presets' config wrapper.
    Version 2 only fills absent axes. Existing input settings and job snapshots stay intact.
    """
    if c.db.get_kv(DORA_INPUT, 0) >= 2:
        return
    paths: list[tuple[Path, bool]] = []
    for version in c.db.fetchall("SELECT id, project_id FROM project_versions"):
        try:
            paths.append((c.config_path(version["project_id"], version["id"]), False))
        except Exception:  # noqa: BLE001 - a version without a usable folder has no config to change
            continue
    paths += [(path, True) for path in sorted((c.data_root / "presets").glob("*.json"))]
    changed = 0
    for path, preset in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        config = data.get("config") if preset and isinstance(data, dict) else data
        adapter = config.get("adapter") if isinstance(config, dict) else None
        if isinstance(adapter, dict) and adapter.get("dora") is True and "dora_axis" not in adapter:
            adapter["dora_axis"] = "output"
            _rewrite(path, data)
            changed += 1
    if changed:
        log.info("preserved the output DoRA axis of %d legacy configs", changed)
    c.db.set_kv(DORA_INPUT, 2)
