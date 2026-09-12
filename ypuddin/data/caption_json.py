"""Structured caption data and parsing, implemented from the documented JSON formats.

Classification is kept until the training transform; editor overrides are authoritative.
Unknown metadata is retained by the writer, but never interpreted as training text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EDIT_MARKER = "_ypuddin_caption_edit"
ORIGINAL_TAGS = "_ypuddin_original_tags"
_FIXED = ("quality", "count", "character", "series", "artist")
_VARIABLE = ("appearance", "tags", "environment")


@dataclass(frozen=True)
class StructuredCaption:
    fixed: tuple[str, ...] = ()
    appearance: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    environment: tuple[str, ...] = ()
    nl: str = ""
    trigger: str = ""

    def text(self, separator: str = ",") -> str:
        tokens = unique((self.trigger, *self.fixed, *self.appearance, *self.tags, *self.environment))
        return render(tokens, self.nl, separator)


def unique(values) -> tuple[str, ...]:
    result = []
    seen = set()
    for value in values:
        value = value.strip()
        if value and value.casefold() not in seen:
            result.append(value)
            seen.add(value.casefold())
    return tuple(result)


def render(tokens, nl: str, separator: str = ",") -> str:
    joined = (separator + " ").join(tokens)
    return f"{joined}. {nl}" if joined and nl else joined or nl


def _object(value: Any, field: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _text(value: Any, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value.strip()


def _tokens(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return unique(value.split(","))
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return unique(value)
    raise ValueError(f"{field} must be a string or an array of strings")


def _character(value: Any) -> tuple[str, ...]:
    if isinstance(value, dict):
        if value.get("full"):
            return _tokens(value["full"], "character.full")
        return unique(
            (
                *_tokens(value.get("name"), "character.name"),
                *_tokens(value.get("variant"), "character.variant"),
            )
        )
    return _tokens(value, "character")


def _parse(data: dict) -> StructuredCaption:
    meta = _object(data.get("meta"), "meta")
    full = any(key in data for key in ("fixed", "ai_output", "from_path"))
    edited = data.get(EDIT_MARKER) == 1 or (full and isinstance(data.get("tags"), list))
    if edited:
        ai = _object(data.get("ai_output"), "ai_output")
        return StructuredCaption(
            tags=_tokens(data.get("tags"), "tags"),
            nl=_text(data.get("nl", ai.get("nl")), "nl"),
        )

    trigger = _text(meta.get("trigger"), "meta.trigger")
    if isinstance(data.get("tags"), dict):
        fields = data["tags"]
    elif full:
        fixed = _object(data.get("fixed"), "fixed")
        ai = _object(data.get("ai_output"), "ai_output")
        origin = _object(data.get("from_path"), "from_path")
        fields = {
            **{key: fixed.get(key) for key in ("quality", "series", "artist")},
            "count": ai.get("count"),
            "character": data.get("character"),
            "environment": ai.get("environment"),
            "nl": ai.get("nl"),
        }
        for key in ("appearance", "tags"):
            fields[key] = list(
                unique(
                    (
                        *_tokens(ai.get(key), f"ai_output.{key}"),
                        *_tokens(origin.get(key), f"from_path.{key}"),
                        *_tokens(origin.get(f"extra_{key}"), f"from_path.extra_{key}"),
                    )
                )
            )
    else:
        fields = data
    fixed_tokens = []
    for key in _FIXED:
        fixed_tokens.extend(
            _character(fields.get(key)) if key == "character" else _tokens(fields.get(key), key)
        )
    groups = {key: _tokens(fields.get(key), key) for key in _VARIABLE}
    return StructuredCaption(
        fixed=unique(fixed_tokens),
        **groups,
        nl=_text(fields.get("nl"), "nl"),
        trigger=trigger,
    )


def parse_caption(data: Any, *, filename: str = "caption.json") -> StructuredCaption:
    try:
        if not isinstance(data, dict):
            raise ValueError("caption root must be an object")
        return _parse(data)
    except ValueError as error:
        raise ValueError(f"Invalid JSON caption {filename}: {error}") from error


def load_caption(path: str | Path) -> tuple[dict, StructuredCaption]:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid JSON caption {path.name}: {error}") from error
    return data, parse_caption(data, filename=path.name)


def _reject_constant(value: str):
    raise ValueError(f"nonfinite JSON value {value} is not allowed")


def edited_content(path: Path, text: str) -> str:
    """Replace editable tags while preserving source metadata and natural-language prose."""
    data, parsed = load_caption(path) if path.exists() else ({}, StructuredCaption())
    # The flat viewer includes prose. An unchanged prose suffix is not a new tag.
    text = text.strip()
    if parsed.nl:
        if text == parsed.nl:
            text = ""
        elif text.endswith(". " + parsed.nl):
            text = text[: -len(parsed.nl) - 2]
    if isinstance(data.get("tags"), dict) and ORIGINAL_TAGS not in data:
        data[ORIGINAL_TAGS] = data["tags"]
    data["tags"] = list(unique(text.split(",")))
    data[EDIT_MARKER] = 1
    if parsed.nl:
        data["nl"] = parsed.nl
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
