"""Saved image/mask edits projected from registration and operation receipts."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from ypuddin.data.index import iter_images

from .db import new_id, now

BASELINE_PREFIX = "dataset.edit_baseline."
EDIT_ACTIONS = ("paint", "preprocess", "automask", "mask", "restore")


def image_paths(root: str | Path) -> set[str]:
    root = Path(root)
    return {image.relative_to(root).as_posix() for image in iter_images(root)} if root.is_dir() else set()


def _layers(image: Path) -> dict[str, Path]:
    return {"image": image, "mask_png": image.with_name(image.stem + ".mask.png"),
            "mask_legacy": image.with_name(image.stem + ".mask")}


def _stat(path: Path):
    try:
        value = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(value.st_mode):
        raise OSError("image or mask is not a regular file")
    return [value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_ino, value.st_dev]


def _stamp(image: Path) -> dict:
    return {kind: _stat(path) for kind, path in _layers(image).items()}


def register_baseline(c, did: str, *, rel_paths: set[str] | None = None) -> None:
    """Confirm newly registered files; never rebase an existing image on a read/rescan."""
    row = c.db.fetchone("SELECT path FROM datasets WHERE id=?", (did,))
    if row is None:
        return
    key = BASELINE_PREFIX + did
    baseline = c.db.get_kv(key, {"version": 1, "images": {}})
    if not isinstance(baseline, dict) or baseline.get("version") != 1 or not isinstance(baseline.get("images"), dict):
        return
    root, registered_at = Path(row["path"]), now()
    for relative in image_paths(root) if rel_paths is None else rel_paths:
        if relative not in baseline["images"]:
            baseline["images"][relative] = {"registered_at": registered_at, "stat": _stamp(root / relative)}
    c.db.set_kv(key, baseline)


def operation_images(c, dataset_ids: list[str], changes: list[dict]) -> list[dict]:
    """Bind existing transaction changes to stable dataset-relative image names."""
    from .routes_work import _records

    by_path = {change["target"]: change for change in changes}
    result = []
    for did in dataset_ids:
        row = c.db.fetchone("SELECT path FROM datasets WHERE id=?", (did,))
        if row is None:
            continue
        root = Path(row["path"])
        for record in _records(c, did):
            image = Path(record["path"])
            if not image.is_relative_to(root):
                continue
            layers = {kind: {key: by_path[str(path)][key] for key in ("before", "after", "before_stat")
                             if key in by_path[str(path)]}
                      for kind, path in _layers(image).items() if str(path) in by_path}
            if layers:
                result.append({"dataset_id": did, "rel_path": image.relative_to(root).as_posix(),
                               "stat": _stamp(image), "layers": layers})
    return result


def record_mask_save(c, row: dict, image: Path, *, changed: bool, before_stat) -> None:
    """Record legacy mask saves without granting paint/pipeline undo capabilities."""
    relative = image.relative_to(Path(row["path"]).resolve()).as_posix()
    timestamp = now()
    c.db.insert("dataset_pipeline_operations", {
        "id": new_id("dp"), "project_id": row["project_id"], "version_id": row["version_id"],
        "action": "mask", "status": "completed", "phase": "completed",
        "created_at": timestamp, "updated_at": timestamp, "finished_at": timestamp,
        "request_json": json.dumps({"images": [{"dataset_id": row["id"], "rel_path": relative}]}),
        "result_json": json.dumps({"changes": [], "dataset_ids": [row["id"]], "edit_images": [{
            "dataset_id": row["id"], "rel_path": relative, "stat": _stamp(image),
            "layers": {"mask_png": {"changed": changed, "before_stat": before_stat}},
        }]}),
    })


def saved_statuses(c, row: dict, rel_paths: list[str]) -> dict[str, str]:
    """Read metadata only; missing history and changed files remain unknown."""
    normalized = {name: name.replace(os.sep, "/") for name in rel_paths}
    wanted = set(normalized.values())
    if not wanted:
        return {}
    try:
        baseline = c.db.get_kv(BASELINE_PREFIX + row["id"], None)
        if baseline is not None and (
            not isinstance(baseline, dict) or baseline.get("version") != 1
            or not isinstance(baseline.get("images"), dict)
        ):
            raise ValueError("invalid baseline")
        baselines = baseline["images"] if baseline is not None else {}
        if any(not isinstance(value, dict) or not isinstance(value.get("stat"), dict)
               or not isinstance(value.get("registered_at"), (int, float))
               for name, value in baselines.items() if name in wanted):
            raise ValueError("invalid image baseline")
    except (AttributeError, TypeError, ValueError):
        return {name: "unknown" for name in rel_paths}
    states = {name: {"baseline": baselines.get(name), "stamp": baselines.get(name, {}).get("stat"),
                     "edited": False, "unknown": False} for name in wanted}
    # Read only edit fields, not inspection reports or backups. No history-page limit applies.
    operations = c.db.fetchall(
        "SELECT id,action,status,created_at,request_json,"
        "CASE WHEN json_valid(result_json) THEN json_object("
        "'edit_images',json_extract(result_json,'$.edit_images'),"
        "'changes',json_extract(result_json,'$.changes'),"
        "'dataset_ids',json_extract(result_json,'$.dataset_ids'),"
        "'undone_by',json_extract(result_json,'$.undone_by'),"
        "'rolled_back',json_extract(result_json,'$.rolled_back'),"
        "'recovery_needed',json_extract(result_json,'$.recovery_needed')) ELSE NULL END AS edit_result "
        "FROM dataset_pipeline_operations WHERE version_id=? "
        "AND action IN ('paint','preprocess','automask','mask','restore') ORDER BY created_at,id",
        (row["version_id"],),
    )
    scopes = {}
    for operation in operations:
        affected = set()
        try:
            request = json.loads(operation["request_json"])
            affected = {item["rel_path"].replace(os.sep, "/") for item in request.get("images", [])
                        if item.get("dataset_id") == row["id"]} & wanted
            if row["id"] in request.get("dataset_ids", []):
                affected = wanted.copy()
            if operation["action"] == "restore":
                affected |= scopes.get(request.get("restore_operation_id"), set())
            result = json.loads(operation["edit_result"]) if operation["edit_result"] else None
            if result is None:
                raise ValueError("invalid operation receipt")
            entries = result.get("edit_images")
            if entries is not None and not isinstance(entries, list):
                raise ValueError("invalid edit image receipt")
            if entries is None and not affected and row["id"] in (result.get("dataset_ids") or []):
                affected = wanted.copy()
            relevant = [entry for entry in entries or [] if entry["dataset_id"] == row["id"]
                        and entry["rel_path"] in wanted]
            affected |= {entry["rel_path"] for entry in relevant}
            scopes[operation["id"]] = affected
            if result.get("recovery_needed"):
                raise ValueError("incomplete recovery")
            if operation["status"] != "completed" and not result.get("rolled_back"):
                continue
            for name in affected:
                registered = states[name]["baseline"]
                if registered and operation["created_at"] < registered["registered_at"]:
                    continue
                if entries is None and result.get("changes"):
                    states[name]["unknown"] = True
            for entry in relevant:
                state = states[entry["rel_path"]]
                registered = state["baseline"]
                if registered and operation["created_at"] < registered["registered_at"]:
                    continue
                for layer, change in entry["layers"].items():
                    if layer not in {"image", "mask_png", "mask_legacy"}:
                        raise ValueError("invalid edited layer")
                    # Only a continuous edit can advance this layer's known identity.
                    # Saving a mask must not adopt an unrelated external image edit.
                    if (not isinstance(state["stamp"], dict) or layer not in state["stamp"]
                            or "before_stat" not in change or state["stamp"][layer] != change["before_stat"]):
                        state["unknown"] = True
                    if state["stamp"] is None:
                        state["stamp"] = {}
                    state["stamp"][layer] = entry["stat"][layer]
                    if (operation["status"] != "completed" or result.get("undone_by")
                            or operation["action"] == "restore"):
                        continue
                    if operation["action"] == "mask":
                        if not isinstance(change["changed"], bool):
                            raise ValueError("invalid mask edit")
                        state["edited"] |= change["changed"]
                    elif change["before"] != change["after"]:
                        state["edited"] = True
        except (AttributeError, KeyError, TypeError, ValueError):
            for name in affected or wanted:
                states[name]["unknown"] = True
    result = {}
    for name, state in states.items():
        try:
            unchanged_stamp = state["stamp"] is not None and _stamp(Path(row["path"]) / name) == state["stamp"]
        except OSError:
            unchanged_stamp = False
        result[name] = ("unknown" if state["unknown"] or not unchanged_stamp else "modified" if state["edited"]
                        else "unmodified" if state["baseline"] else "unknown")
    return {name: result[relative] for name, relative in normalized.items()}
