"""Version-owned dataset purposes, determined by real directory ancestry."""

from __future__ import annotations

import copy
import json
from pathlib import Path


def _resolved(path: str) -> Path | None:
    try:
        return Path(path).expanduser().resolve()
    except (OSError, ValueError, RuntimeError):
        return None  # Config/Plan validation owns malformed-path field errors.


def managed_source_role(c, pid: str, vid: str, path: str) -> tuple[bool, str] | None:
    if not path or not str(path).strip():
        return None
    source = _resolved(path)
    if source is None:
        return None
    roots = [(True, c.reg_dir(pid, vid))]
    if c.project_layout(pid) >= 2:
        roots.append((False, c.dataset_dir(pid, vid)))
    for is_reg, root in roots:
        # Managed roots themselves may not be redirected. A source alias pointing into
        # a real managed root is recognized; an alias escaping it stays external.
        if root.is_symlink():
            continue
        real_root = root.resolve()
        if source == real_root or real_root in source.parents:
            return is_reg, str(real_root)
    return None


def _registered(c, pid: str, vid: str) -> dict[Path, dict]:
    return {
        _resolved(row["path"]): row
        for row in c.db.fetchall(
            "SELECT id,path,is_reg,index_status,stats_json FROM datasets WHERE project_id=? AND version_id=?",
            (pid, vid),
        )
    }


def normalize_source_roles(c, pid: str, config: dict, version_id: str | None = None) -> dict:
    """Normalize only known managed roots; external explicit compatibility metadata survives."""
    vid = c.resolve_version(pid, version_id)["id"]
    result = copy.deepcopy(config)
    registered = _registered(c, pid, vid)
    for section in ("dataset", "validation"):
        group = result.get(section)
        if not isinstance(group, dict) or not isinstance(group.get("sources"), list):
            continue
        for item in group["sources"]:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("path"), str)
                or not item["path"].strip()
            ):
                continue
            role = managed_source_role(c, pid, vid, item["path"])
            if role is not None:
                item["is_reg"] = role[0]
            elif "is_reg" not in item:
                existing = registered.get(_resolved(item["path"]))
                if existing is not None:
                    item["is_reg"] = bool(existing["is_reg"])
    return result


def describe_source_roles(c, pid: str, config: dict, version_id: str | None = None) -> list[dict]:
    vid = c.resolve_version(pid, version_id)["id"]
    normalized = normalize_source_roles(c, pid, config, vid)
    registered = _registered(c, pid, vid)
    result = []
    for section in ("dataset", "validation"):
        group = normalized.get(section)
        if not isinstance(group, dict) or not isinstance(group.get("sources"), list):
            continue
        for item in group["sources"]:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("path"), str)
                or not item["path"].strip()
            ):
                continue
            path = item["path"]
            role = managed_source_role(c, pid, vid, path)
            row = registered.get(_resolved(path))
            images = None
            if row and row["index_status"] == "ready":
                try:
                    images = json.loads(row["stats_json"]).get("images")
                except (ValueError, TypeError):
                    pass
                if item.get("excluded_files") or item.get("excluded_dirs"):
                    from .routes_dataset_management import included
                    from .routes_work import _records
                    states = [(Path(path).expanduser().resolve(), set(item.get("excluded_files", [])),
                               tuple(p + "/" for p in item.get("excluded_dirs", [])), item)]
                    images = sum(included(record["path"], states) for record in _records(c, row["id"]))
            result.append(
                {
                    "path": path,
                    "section": section,
                    "is_reg": bool(item.get("is_reg", False)),
                    "managed": role is not None,
                    "root": role[1] if role else None,
                    "images": images if isinstance(images, int) and images >= 0 else None,
                    "origin": "version"
                    if role
                    else "registered"
                    if _resolved(path) in registered
                    else "external",
                }
            )
    return result
