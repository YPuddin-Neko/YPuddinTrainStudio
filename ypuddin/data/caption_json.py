"""Structured caption data and parsing, implemented from the documented JSON formats.

Classification is kept until the training transform; editor overrides are authoritative.
Unknown metadata is retained by the writer, but never interpreted as training text.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EDIT_MARKER = "_ypuddin_caption_edit"
ORIGINAL_TAGS = "_ypuddin_original_tags"
_FIXED = ("quality", "count", "character", "series", "artist")
_VARIABLE = ("appearance", "tags", "environment")


class JSONCaptionError(ValueError):
    def __init__(self, message: str, *, code: str = "caption_json_invalid"):
        super().__init__(message)
        self.code = code


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


def _character_full(value: dict) -> tuple[str, ...]:
    return _tokens(value.get("full"), "character.full")


def _character(value: Any) -> tuple[str, ...]:
    if isinstance(value, dict):
        full = _character_full(value)
        if full:
            return full
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
        raise JSONCaptionError(f"Invalid JSON caption {filename}: {error}") from error


def load_caption(path: str | Path, *, require_known_format: bool = False) -> tuple[dict, StructuredCaption]:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise JSONCaptionError(f"Invalid JSON caption {path.name}: {error}") from error
    parsed = parse_caption(data, filename=path.name)
    if require_known_format and _format(data) == "unknown":
        raise JSONCaptionError(
            f"Invalid JSON caption {path.name}: unrecognized caption format; "
            "use a supported tags/nl structure or select a TXT caption explicitly",
            code="caption_format_unsupported",
        )
    return data, parsed


def _reject_constant(value: str):
    raise ValueError(f"nonfinite JSON value {value} is not allowed")


class CaptionConflictError(ValueError):
    """The caption changed after the editor loaded it."""


def _format(data: dict) -> str:
    full = any(key in data for key in ("fixed", "ai_output", "from_path"))
    if data.get(EDIT_MARKER) == 1 or (full and isinstance(data.get("tags"), list)):
        return "legacy_override"
    if isinstance(data.get("tags"), dict):
        return "nested"
    if full:
        return "full"
    if any(key in data for key in (*_FIXED, "appearance", "environment")):
        return "simple"
    if "tags" in data or "nl" in data:
        return "flat"
    return "unknown"


def _lookup(data: dict, path: tuple[str, ...]) -> tuple[bool, Any]:
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False, None
        current = current[key]
    return True, current


def _assign(data: dict, path: tuple[str, ...], value: str | list[str]) -> None:
    current = data
    for key in path[:-1]:
        if current.get(key) is None:
            current[key] = {}
        if not isinstance(current[key], dict):
            raise ValueError(f"Caption field {'/'.join(path)} has an incompatible parent")
        current = current[key]
    current[path[-1]] = value


def _fields(data: dict, fmt: str) -> list[dict]:
    fields = []

    def add(path, role, default=""):
        path = tuple(path)
        present, value = _lookup(data, path)
        fields.append({
            "path": list(path), "role": role,
            "value": copy.deepcopy(value if value is not None else default), "present": present,
        })

    def character(prefix, default=""):
        _, value = _lookup(data, (*prefix, "character"))
        if isinstance(value, dict):
            # A populated full name is authoritative in the training parser. Do not
            # offer ignored name/variant fields as if edits changed training text.
            if _character_full(value):
                add((*prefix, "character", "full"), "character_full")
            else:
                add((*prefix, "character", "name"), "character_name")
                add((*prefix, "character", "variant"), "character_variant")
        elif fmt == "full" and value is None:
            add(("character", "name"), "character_name")
            add(("character", "variant"), "character_variant")
        else:
            add((*prefix, "character"), "character", default)

    if fmt == "unknown":
        return fields
    if fmt == "legacy_override":
        add(("tags",), "tags", [])
        add(("nl",), "nl", parse_caption(data).nl)
        return fields
    if fmt == "full":
        add(("fixed", "quality"), "quality")
        add(("ai_output", "count"), "count")
        character(())
        for role in ("series", "artist"):
            add(("fixed", role), role)
        for role in ("appearance", "tags"):
            add(("ai_output", role), role, [])
            for key in (role, f"extra_{role}"):
                if key == "appearance" or _lookup(data, ("from_path", key))[0]:
                    add(("from_path", key), role, [])
        add(("ai_output", "environment"), "environment", [])
        add(("ai_output", "nl"), "nl")
    elif fmt in {"nested", "simple"}:
        prefix = ("tags",) if fmt == "nested" else ()
        default = [] if fmt == "nested" else ""
        for role in ("quality", "count"):
            add((*prefix, role), role, default)
        character(prefix, default)
        for role in ("series", "artist"):
            add((*prefix, role), role, default)
        for role in _VARIABLE:
            add((*prefix, role), role, [])
        add((*prefix, "nl"), "nl")
    else:
        add(("tags",), "tags", [])
        add(("nl",), "nl")
    if _lookup(data, ("meta", "trigger"))[0]:
        add(("meta", "trigger"), "trigger")
    return fields


def _read_document(path: Path) -> tuple[dict, str]:
    payload = path.read_bytes()
    try:
        data = json.loads(payload.decode("utf-8-sig"), parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid JSON caption {path.name}: {error}") from error
    parse_caption(data, filename=path.name)
    return data, hashlib.sha256(payload).hexdigest()


def load_caption_structure(path: str | Path | None) -> dict | None:
    if path is None or Path(path).suffix.lower() != ".json" or not Path(path).exists():
        return None
    data, revision = _read_document(Path(path))
    fmt = _format(data)
    return {
        "format": fmt, "document": data, "fields": _fields(data, fmt), "revision": revision,
        "editable": fmt != "unknown", "legacy_override": fmt == "legacy_override",
        "reason": "Unrecognized JSON caption schema; source document is read only"
        if fmt == "unknown" else None,
    }


def _field_tokens(field: dict) -> tuple[str, ...]:
    if field["role"] == "trigger":
        return unique((field["value"],))
    return _tokens(field["value"], "/".join(field["path"]))


def _replace_flat_tags(data: dict, fields: list[dict], fmt: str, text: str) -> None:
    parsed = parse_caption(data)
    text = text.strip()
    if parsed.nl:
        if text == parsed.nl:
            text = ""
        elif text.endswith(". " + parsed.nl):
            text = text[: -len(parsed.nl) - 2]
    current_text = ", ".join(unique((parsed.trigger, *parsed.fixed, *parsed.appearance,
                                     *parsed.tags, *parsed.environment)))
    if text == current_text:
        return
    if text and any(
        "," in token for field in fields if field["role"] != "nl" for token in _field_tokens(field)
    ):
        raise ValueError("Flat editing cannot preserve comma-containing JSON tags; edit their fields instead")
    desired = unique(text.split(","))
    if fmt == "legacy_override":
        # Keep only the existing override's live values. Its old classifications
        # remain history; never make them active again by removing the marker.
        original = fields[0]["value"]
        _assign(data, ("tags",), list(desired) if isinstance(original, list) else ", ".join(desired))
        return
    wanted = {token.casefold(): token for token in desired}
    existing = set()
    for field in fields:
        if field["role"] == "nl":
            continue
        before = _field_tokens(field)
        existing.update(token.casefold() for token in before)
        retained = tuple(wanted[token.casefold()] for token in before if token.casefold() in wanted)
        if retained != before:
            _assign(data, tuple(field["path"]), list(retained)
                    if isinstance(field["value"], list) else ", ".join(retained))
    added = [token for token in desired if token.casefold() not in existing]
    if added:
        target = next(field for field in fields if field["role"] == "tags")
        target_path = tuple(target["path"])
        _, current = _lookup(data, target_path)
        tokens = list(_tokens(current, "/".join(target_path))) + added
        _assign(data, target_path, tokens if isinstance(target["value"], list) else ", ".join(tokens))


def edited_content(
    path: Path, text: str | None = None, *, description: str | None = None,
    fields: list[dict] | None = None, revision: str | None = None,
) -> str:
    """Edit real schema fields, retaining ownership, types and unrelated metadata."""
    data, current_revision = _read_document(path) if path.exists() else ({"tags": []}, "")
    if revision is not None and revision != current_revision:
        raise CaptionConflictError("Caption changed since it was loaded; reload before saving")
    fmt = _format(data)
    if fmt == "unknown":
        raise ValueError("Unrecognized JSON caption schema; source document is read only")
    allowed = _fields(data, fmt)
    if fields is not None:
        if text is not None or description is not None:
            raise ValueError("Choose structured fields or flat caption/description, not both")
        if revision is None:
            raise ValueError("Structured caption edits require caption_revision")
        by_path = {tuple(field["path"]): field for field in allowed}
        seen = set()
        for patch in fields:
            key = tuple(patch["path"])
            if key not in by_path or key in seen:
                raise ValueError(f"Caption field {'/'.join(key)} is not editable or was repeated")
            seen.add(key)
            value, original = patch["value"], by_path[key]["value"]
            if not isinstance(value, type(original)) or isinstance(value, list) and not all(
                isinstance(item, str) for item in value
            ):
                raise ValueError(f"Caption field {'/'.join(key)} must retain its string/array type")
            _assign(data, key, copy.deepcopy(value))
    else:
        if text is not None:
            _replace_flat_tags(data, allowed, fmt, text)
        if description is not None:
            target = next(field for field in allowed if field["role"] == "nl")
            _assign(data, tuple(target["path"]), description.strip())
    parse_caption(data, filename=path.name)
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def category_tokens(path: str | Path) -> dict[str, Any]:
    """Each schema field's tags, the prose and the file's trigger, read the way training reads them."""
    path = Path(path)
    data = _read_document(path)[0]
    return document_categories(data, filename=path.name)


def document_categories(data: dict, *, filename: str = "caption.json") -> dict[str, Any]:
    """Read category values from a caption document without a filesystem round trip."""
    fmt = _format(data)
    parsed = parse_caption(data, filename=filename)
    result: dict[str, Any] = {key: [] for key in (*_FIXED, *_VARIABLE)}
    if fmt == "legacy_override":
        result["tags"] = list(parsed.tags)
    else:
        if fmt == "nested":
            fields = data["tags"]
        elif fmt == "full":
            fixed = _object(data.get("fixed"), "fixed")
            ai = _object(data.get("ai_output"), "ai_output")
            origin = _object(data.get("from_path"), "from_path")
            fields = {
                **{key: fixed.get(key) for key in ("quality", "series", "artist")},
                "count": ai.get("count"),
                "character": data.get("character"),
                "environment": ai.get("environment"),
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
        for key in (*_FIXED, *_VARIABLE):
            result[key] = list(
                _character(fields.get(key)) if key == "character" else _tokens(fields.get(key), key)
            )
    result["nl"] = parsed.nl
    result["trigger"] = parsed.trigger
    return result


def _set_trigger(data: dict, trigger: str) -> None:
    meta = data.get("meta")
    if meta is None:
        data["meta"] = meta = {}
    if not isinstance(meta, dict):
        raise ValueError("meta must be an object to hold the trigger")
    meta["trigger"] = trigger


def with_trigger(content: str, trigger: str, *, filename: str = "caption.json") -> str:
    """A JSON caption document with ``meta.trigger`` set; everything else unchanged."""
    data = json.loads(content)
    _set_trigger(data, trigger.strip())
    parse_caption(data, filename=filename)
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def categorized_content(
    path: str | Path,
    groups: dict[str, list[str]],
    *,
    nl: str | None = None,
    trigger: str = "",
    character: list[str] | None = None,
) -> str:
    """Write count / appearance / tags / environment groups and prose into the file's own fields.

    Quality, series, artist and unknown metadata stay as they are; ``character`` fills the character
    field only when the file has none. A new file uses the flat top-level layout.
    """
    path = Path(path)
    data = _read_document(path)[0] if path.exists() else {}
    fmt = _format(data) if data else "simple"
    if fmt == "unknown":
        raise ValueError("Unrecognized JSON caption schema; source document is read only")
    ordered = ("count", "appearance", "tags", "environment")
    if fmt == "legacy_override":
        # The top-level list overrides every classified field; keep one list, in group order.
        data["tags"] = list(unique(tag for key in ordered for tag in groups.get(key, ())))
        if nl is not None:
            data["nl"] = nl.strip()
    else:
        prefix = ("tags",) if fmt == "nested" else ("ai_output",) if fmt == "full" else ()
        for key in ordered:
            if key not in groups:
                continue
            values = list(unique(groups[key]))
            present, current = _lookup(data, (*prefix, key))
            as_text = isinstance(current, str) or (key == "count" and (not present or current is None))
            _assign(data, (*prefix, key), ", ".join(values) if as_text else values)
        if nl is not None:
            _assign(data, (*prefix, "nl"), nl.strip())
        if character:
            owner = () if fmt == "full" else prefix
            _, current = _lookup(data, (*owner, "character"))
            if not (_character(current) if current is not None else ()):
                _assign(data, (*owner, "character"), ", ".join(unique(character)))
    if trigger.strip():
        _set_trigger(data, trigger.strip())
    parse_caption(data, filename=path.name)
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
