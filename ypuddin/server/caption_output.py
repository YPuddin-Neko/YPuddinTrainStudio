"""Explicit tag-output destinations and JSON layouts; file writes remain journaled by the pipeline."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from ypuddin.data.caption_json import document_categories, parse_caption

from .errors import ApiError


def destination(record: dict, output_format: str | None) -> Path:
    original = record["caption"]
    if output_format is None:
        return original
    suffix = ".txt" if output_format == "txt" else ".json"
    target = original if original.suffix.lower() == suffix else record["path"].with_suffix(suffix)
    if target.is_symlink():
        raise ApiError("sidecar links cannot be changed", code="pipeline.path")
    return target


def json_layout(content: str, output_format: str | None) -> str:
    if output_format not in {"json", "json_simplified"}:
        return content
    data = json.loads(content)
    fields = document_categories(data)
    full = any(key in data for key in ("ai_output", "fixed", "from_path"))
    override = data.get("_ypuddin_caption_edit") == 1 or full and isinstance(data.get("tags"), list)
    nested = isinstance(data.get("tags"), dict)
    if not override and not nested and full == (output_format == "json"):
        return content

    # Move active caption fields; preserve unknown metadata, including fields inside old containers.
    result = copy.deepcopy(data)
    known = ("quality", "count", "character", "series", "artist", "appearance", "tags", "environment", "nl")
    extras = {}
    if isinstance(data.get("character"), dict):
        extras["character"] = copy.deepcopy(data["character"])
    if "_ypuddin_original_tags" in data:
        extras["_ypuddin_original_tags"] = copy.deepcopy(data["_ypuddin_original_tags"])
    if nested:
        remainder = {key: value for key, value in result.pop("tags").items() if key not in known}
        if remainder:
            extras["tags"] = remainder
    for owner in ("ai_output", "fixed", "from_path"):
        if owner in result:
            values = result.pop(owner) or {}
            remainder = {
                key: value
                for key, value in values.items()
                if key not in (*known, "extra_tags", "extra_appearance")
            }
            if remainder:
                extras[owner] = remainder
    for key in (*known, "_ypuddin_caption_edit", "_ypuddin_original_tags"):
        result.pop(key, None)
    if (extras or fields["trigger"]) and result.get("meta") is None:
        result["meta"] = {}
    if extras:
        meta = result.setdefault("meta", {})
        existing = meta.get("previous_caption_fields", [])
        meta["previous_caption_fields"] = (existing if isinstance(existing, list) else [existing]) + [extras]
    if fields["trigger"]:
        result.setdefault("meta", {})["trigger"] = fields["trigger"]
    variables = {
        "count": ", ".join(fields["count"]),
        **{key: fields[key] for key in ("appearance", "tags", "environment")},
        "nl": fields["nl"],
    }
    fixed = {key: ", ".join(fields[key]) for key in ("quality", "series", "artist")}
    if output_format == "json":
        result.update(fixed=fixed, character=", ".join(fields["character"]), ai_output=variables)
    else:
        result.update(**fixed, character=", ".join(fields["character"]), **variables)
    parse_caption(result)
    return json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
