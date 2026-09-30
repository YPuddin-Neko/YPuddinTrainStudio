"""Stable ordering for image browsers; training plans keep their scan order."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any, Literal

ImageSort = Literal["filename", "folder", "modified"]


def natural_key(value: str) -> tuple:
    return tuple(
        (1, int(part)) if part.isdecimal() else (0, part.casefold()) for part in re.split(r"(\d+)", value)
    )


def sort_images(
    items: list[dict[str, Any]], root: str | Path, order: ImageSort = "filename"
) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> tuple:
        relative = str(item["rel_path"]).replace("\\", "/")
        path = PurePosixPath(relative)
        stable = (natural_key(path.name), natural_key(relative), relative, str(item.get("hash", "")))
        if order == "folder":
            return (natural_key(str(path.parent)), *stable)
        if order == "modified":
            try:
                modified = (Path(root) / relative).stat().st_mtime_ns
            except OSError:
                modified = -1
            return (-modified, *stable)
        return stable

    return sorted(items, key=key)
