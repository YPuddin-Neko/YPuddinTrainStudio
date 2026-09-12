"""Bind new project jobs to a stable output name without changing saved recipes."""

from __future__ import annotations

import copy
import re
import unicodedata
from pathlib import PureWindowsPath
from typing import Any


def automatic_name(value: Any) -> bool:
    return value is None or isinstance(value, str) and value.strip() in ("", "lora")


def project_weight_name(name: str, project_id: str, number: int) -> str:
    """Keep Unicode names, reserving room for tags/extensions on common filesystems."""
    stem = unicodedata.normalize("NFC", name)
    stem = re.sub(r'[/\\:*?"<>|\x00-\x1f\x7f]', "_", stem)
    stem = "".join(char for char in stem if not unicodedata.category(char).startswith("C"))
    stem = re.sub(r"\s+", " ", stem).strip(" .")
    if not stem or PureWindowsPath(stem).is_reserved():
        stem = project_id
    suffix = f"_v{number}"
    # The trainer adds epoch/step/final tags and .safetensors. Bound UTF-8 bytes
    # as well as characters, since non-ASCII names can exceed a 255-byte limit.
    while len(stem + suffix) > 120 or len((stem + suffix).encode("utf-8")) > 180:
        stem = stem[:-1]
    stem = stem.rstrip(" .") or project_id
    return stem + suffix


def bind_output_name(c, pid: str, config: dict[str, Any], vid: str | None = None) -> dict[str, Any]:
    result = copy.deepcopy(config)
    checkpoint = result.get("checkpoint")
    if "checkpoint" in result and not isinstance(checkpoint, dict):
        return result  # Leave malformed configs to the schema's field errors.
    checkpoint = result.setdefault("checkpoint", {})
    if automatic_name(checkpoint.get("name")):
        version = c.resolve_version(pid, vid)
        project = c.db.fetchone("SELECT name FROM projects WHERE id=?", (pid,))
        checkpoint["name"] = project_weight_name(project["name"], pid, version["number"])
    return result


def output_binding(c, pid: str, config: dict[str, Any], vid: str | None = None) -> dict[str, Any]:
    version = c.resolve_version(pid, vid)
    original = config.get("checkpoint", {})
    if not isinstance(original, dict):
        from .errors import ApiError

        raise ApiError("checkpoint must be an object", code="config.invalid", status=422)
    requested = original.get("output_dir")
    if requested is not None and not isinstance(requested, str):
        from .errors import ApiError

        raise ApiError("checkpoint.output_dir must be a path string", code="config.invalid", status=422)
    bound = bind_output_name(c, pid, config, version["id"])
    from pydantic import ValidationError

    from ypuddin.config.schema import CheckpointConfig

    try:
        CheckpointConfig(name=bound["checkpoint"].get("name"))
    except ValidationError as error:
        from .errors import ApiError

        raise ApiError("invalid checkpoint filename prefix", code="config.invalid", status=422) from error
    try:
        directory = c.job_output_dir(pid, version["id"], "{job_id}", requested)
        inherited = c.inherits_output_dir(pid, version["id"], requested)
    except (ValueError, OSError, RuntimeError) as error:
        from .errors import ApiError

        raise ApiError("invalid checkpoint output path", code="config.invalid", status=422) from error
    return {
        "directory_template": str(directory),
        "name": bound["checkpoint"].get("name"),
        "automatic_name": automatic_name(original.get("name")),
        "inherits_output_dir": inherited,
    }
