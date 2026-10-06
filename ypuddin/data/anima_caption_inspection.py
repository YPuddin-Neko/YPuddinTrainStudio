"""Read-only, model-specific caption advice; never part of training transforms.

Rules are recommendations from https://huggingface.co/circlestone-labs/Anima#prompting.
Only JSON fields with known tag ownership get proposed values. Prose, triggers,
unknown metadata, score_* tokens and legacy flattened documents stay untouched.
"""

from __future__ import annotations

import re
from pathlib import Path

from .caption_json import load_caption_structure

_TAG_ROLES = frozenset({
    "quality", "count", "character", "character_name", "character_variant", "character_full",
    "series", "artist", "appearance", "tags", "environment",
})
_SCORE = re.compile(r"\bscore_[a-z0-9_]+\b", re.IGNORECASE)
_EMOTICON = re.compile(r"[0-9oOxXuUtT=^<>@|+.;()\\_\-]+")
_MESSAGES = {
    "anima_artist_prefix": "Anima recommends @ before an artist name; review the proposed value",
    "anima_tag_spacing": "Anima recommends spaces in tags, except score_* tokens",
    "anima_tag_case": "Anima recommends lowercase tags; prose and triggers are unchanged",
    "anima_text_shuffle": (
        "This TXT may mix prose and tags. Shuffling or dropping separated fragments may break "
        "sentences; review the text or use a separate JSON natural-language field"
    ),
}


def _propose_token(token: str, role: str, triggers: set[str], codes: set[str]) -> str:
    content = token.strip()
    # Dynamic prompts and protected trigger tokens need the user's own semantics.
    if not content or content in triggers or _SCORE.fullmatch(content) or any(char in content for char in "{}<>"):
        return token
    if "_" in content and _EMOTICON.fullmatch(content):
        return token

    def normalize(part):
        if "_" in part:
            codes.add("anima_tag_spacing")
        if part.lower() != part:
            codes.add("anima_tag_case")
        return part.replace("_", " ").lower()

    # Leave score tokens byte-for-byte intact, including their underscore.
    pieces, offset = [], 0
    for match in _SCORE.finditer(content):
        pieces.extend((normalize(content[offset:match.start()]), match.group()))
        offset = match.end()
    pieces.append(normalize(content[offset:]))
    proposed = "".join(pieces)
    if role == "artist" and not proposed.startswith("@"):
        proposed = "@" + proposed
        codes.add("anima_artist_prefix")
    if proposed == content:
        return token
    return token[:len(token) - len(token.lstrip())] + proposed + token[len(token.rstrip()):]


def inspect_anima_caption(path: Path, text: str, caption_config: dict) -> dict:
    """Return warnings and exact field-value previews without writing any file.

    TXT prose detection is deliberately conservative and explicitly a possible
    risk, not a language classifier or a reason to reject valid tag shuffling.
    """
    result = {"profile": "anima", "suggestions": [], "issues": []}
    codes: set[str] = set()
    if path.suffix.lower() == ".json":
        structure = load_caption_structure(path)
        if not structure or not structure["editable"] or structure["legacy_override"]:
            return result
        triggers = {caption_config.get("trigger_word") or ""}
        triggers.update(
            field["value"] for field in structure["fields"]
            if field["role"] == "trigger" and isinstance(field["value"], str)
        )
        for field in structure["fields"]:
            if field["role"] not in _TAG_ROLES or not field["present"]:
                continue
            before = field["value"]
            if isinstance(before, str):
                after = ",".join(
                    _propose_token(token, field["role"], triggers, codes) for token in before.split(",")
                )
            elif isinstance(before, list) and all(isinstance(token, str) for token in before):
                # An array entry is one tag; do not split comma-containing values.
                after = [_propose_token(token, field["role"], triggers, codes) for token in before]
            else:
                continue
            if after != before:
                result["suggestions"].append({
                    "path": field["path"], "role": field["role"], "before": before, "after": after,
                })
    elif caption_config.get("shuffle") or caption_config.get("tag_dropout", 0) > 0:
        separator = caption_config.get("separator", ",")
        fragments = [part.strip() for part in text.split(separator)] if separator else []
        # Ignore fixed fragments and an exact configured trigger, as training does.
        trigger = caption_config.get("trigger_word")
        if trigger:
            fragments = [part for part in fragments if part != trigger]
        fragments = [part for part in fragments if part][caption_config.get("keep_tokens", 0):]
        # Sentence punctuation plus several words is evidence worth reviewing;
        # an ordinary list of tags, parentheses and emoticons is not an error.
        if len(fragments) > 1 and any(
            re.search(r"[.!?。！？](?:\s|$)", part) and len(part.split()) >= 5 for part in fragments
        ):
            codes.add("anima_text_shuffle")
    result["issues"] = [
        {"severity": "warning", "code": code, "message": _MESSAGES[code]} for code in sorted(codes)
    ]
    return result
