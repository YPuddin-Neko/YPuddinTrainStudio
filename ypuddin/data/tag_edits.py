"""Whole-tag edits that retain structured caption fields and unrelated metadata."""

from __future__ import annotations

import json
from pathlib import Path

from .caption_json import _assign, _fields, _format, _lookup, _reject_constant, parse_caption
from .captions import _split_weighted_tags


def _tokens(value, *, trigger=False):
    values = value if isinstance(value, list) else [value] if trigger else _split_weighted_tags(value or "", ",")
    return [token.strip() for token in values if token.strip()]


def _document(path: Path, original: bytes | None, fallback: str):
    if path.suffix.lower() != ".json":
        text = original.decode("utf-8-sig") if original is not None else fallback
        return None, [{"path": (), "role": "tags", "value": text, "tokens": _tokens(text)}]
    data = json.loads(original.decode("utf-8-sig"), parse_constant=_reject_constant) if original is not None else {"tags": _tokens(fallback)}
    parse_caption(data, filename=path.name)
    fmt = _format(data)
    if fmt == "unknown":
        raise ValueError(f"Unrecognized JSON caption schema in {path.name}; source document is read only")
    fields = [field for field in _fields(data, fmt) if field["role"] != "nl"]
    # Training reads the trigger before every category, even though it lives in meta.
    fields.sort(key=lambda field: field["role"] != "trigger")
    for field in fields:
        field["tokens"] = _tokens(field["value"], trigger=field["role"] == "trigger")
    return data, fields


def caption_tags(path: str | Path | None) -> tuple[str, ...]:
    """Editor/statistics tokens, with weighted groups and JSON array entries intact."""
    if path is None:
        return ()
    path = Path(path)
    _, fields = _document(path, path.read_bytes() if path.exists() else None, "")
    seen, result = set(), []
    for field in fields:
        for token in field["tokens"]:
            if token.casefold() not in seen:
                seen.add(token.casefold())
                result.append(token)
    return tuple(result)


def edit_caption_tags(
    path: Path, original: bytes | None, *, add: list[str], remove: list[str],
    replacements: dict[str, str], position: str, deduplicate: bool, fallback: str = "",
) -> bytes | None:
    """Return replacement bytes only when an operation changes stored tag values."""
    data, fields = _document(path, original, fallback)
    if data is not None and _format(data) in {"gbc", "moss_layout"}:
        raise ValueError("GBC 与 MOSS Layout 标签为只读，不能进行标签编辑。")
    unwanted = {tag.casefold() for tag in remove}
    for field in fields:
        field["updated"] = [replacements.get(tag.casefold(), tag) for tag in field["tokens"]
                            if tag.casefold() not in unwanted]
    existing = {tag.casefold() for field in fields for tag in field["updated"]}
    fresh = []
    for tag in add:
        if tag.casefold() not in existing:
            fresh.append(tag)
            existing.add(tag.casefold())
    target = next(field for field in fields if field["role"] == "tags")
    target["updated"] = fresh + target["updated"] if position == "start" else target["updated"] + fresh
    seen = set()
    if deduplicate:
        for field in fields:
            retained = []
            for tag in field["updated"]:
                if tag.casefold() not in seen:
                    retained.append(tag)
                    seen.add(tag.casefold())
            field["updated"] = retained
    changed = [field for field in fields if field["tokens"] != field["updated"]]
    if not changed:
        return None
    if data is None:
        if any(len(_tokens(token)) != 1 for token in target["updated"]):
            raise ValueError(f"A comma-containing tag needs a JSON array in {path.name}")
        return (", ".join(target["updated"]) + "\n").encode("utf-8")
    for field in changed:
        tokens = field["updated"]
        if not isinstance(field["value"], list) and any(
            len(_tokens(token, trigger=field["role"] == "trigger")) != 1 for token in tokens
        ):
            raise ValueError(f"A comma-containing tag needs an array field in {path.name}")
        _assign(data, tuple(field["path"]), tokens if isinstance(field["value"], list) else ", ".join(tokens))
        if field["role"] == "character_full" and not tokens:
            # Empty full falls back to name/variant in the reader. Clear these alternate
            # name representations so deleting the active character cannot revive them.
            parent = tuple(field["path"][:-1])
            for name in ("name", "variant"):
                present, value = _lookup(data, (*parent, name))
                if present:
                    _assign(data, (*parent, name), [] if isinstance(value, list) else "")
    parse_caption(data, filename=path.name)
    return (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
