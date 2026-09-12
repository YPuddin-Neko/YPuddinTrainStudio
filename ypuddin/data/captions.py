"""Caption loading and train-time transforms (shuffle, dropout, trigger word, wildcards...)."""

from __future__ import annotations

import os
import random
import re
import tempfile
from pathlib import Path

from ypuddin.config import CaptionConfig

from .caption_json import StructuredCaption, edited_content, load_caption, render, unique

_WILDCARD = re.compile(r"\{([^{}]*)\}")


def read_training_caption(path: str | Path | None, fallback: str | None = None) -> str | StructuredCaption:
    if path and Path(path).exists():
        if Path(path).suffix.lower() == ".json":
            return load_caption(path)[1]
        text = Path(path).read_text(encoding="utf-8", errors="replace").strip()
        if text:
            return text
    return fallback or ""


def read_caption(path: str | Path | None, fallback: str | None = None) -> str:
    raw = read_training_caption(path, fallback)
    return raw.text() if isinstance(raw, StructuredCaption) else raw


def read_editable_caption(path: str | Path | None, fallback: str | None = None) -> str:
    """Tag-editor text; structured natural language remains separate and untouched."""
    raw = read_training_caption(path, fallback)
    if isinstance(raw, StructuredCaption):
        return (", ").join(unique((raw.trigger, *raw.fixed, *raw.appearance, *raw.tags, *raw.environment)))
    return raw


def caption_description(path: str | Path | None) -> str:
    """Read-only structured prose for the editor; ordinary TXT has no separate description."""
    raw = read_training_caption(path)
    return raw.nl if isinstance(raw, StructuredCaption) else ""


def caption_content(path: str | Path, text: str) -> str:
    path = Path(path)
    return edited_content(path, text) if path.suffix.lower() == ".json" else text.strip() + "\n"


def write_caption(path: str | Path, text: str) -> None:
    """Atomically replace one caption. Each concurrent writer gets its own temporary file."""
    path = Path(path)
    content = caption_content(path, text)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _expand_wildcards(text: str, rng: random.Random) -> str:
    while (m := _WILDCARD.search(text)) is not None:
        options = m.group(1).split("|")
        text = text[: m.start()] + rng.choice(options).strip() + text[m.end() :]
    return text


def transform_caption(raw: str | StructuredCaption, cfg: CaptionConfig, rng: random.Random) -> str | None:
    """Returns the caption to encode, or ``None`` when this sample is dropped to unconditional."""
    if cfg.caption_dropout > 0 and rng.random() < cfg.caption_dropout:
        return None
    if isinstance(raw, StructuredCaption):
        return _transform_structured(raw, cfg, rng)
    text = raw
    if cfg.wildcard:
        text = _expand_wildcards(text, rng)
    sep = cfg.separator
    tags = [t.strip() for t in text.split(sep)] if text else []
    tags = [t for t in tags if t]
    if cfg.trigger_word:
        tags = [t for t in tags if t != cfg.trigger_word]
    keep = tags[: cfg.keep_tokens]
    rest = tags[cfg.keep_tokens :]
    if cfg.tag_dropout > 0:
        rest = [t for t in rest if rng.random() >= cfg.tag_dropout]
    if cfg.shuffle:
        rng.shuffle(rest)
    tags = ([cfg.trigger_word] if cfg.trigger_word else []) + keep + rest
    joined = (sep + " ").join(tags)
    parts = [p for p in (cfg.prefix.strip(), joined, cfg.suffix.strip()) if p]
    return (sep + " ").join(parts) if cfg.prefix or cfg.suffix else joined


def _transform_structured(raw: StructuredCaption, cfg: CaptionConfig, rng: random.Random) -> str:
    def expand(token):
        return _expand_wildcards(token, rng) if cfg.wildcard else token

    tokens = list(unique(expand(t) for t in (cfg.trigger_word or "", raw.trigger, *raw.fixed)))
    for group in (raw.appearance, raw.tags, raw.environment):
        rest = [expand(token) for token in group]
        if cfg.tag_dropout > 0:
            rest = [token for token in rest if rng.random() >= cfg.tag_dropout]
        if cfg.shuffle:
            rng.shuffle(rest)
        tokens.extend(rest)
    joined = render(unique(tokens), expand(raw.nl), cfg.separator)
    return (cfg.separator + " ").join(
        part for part in (cfg.prefix.strip(), joined, cfg.suffix.strip()) if part
    )


def is_stochastic(cfg: CaptionConfig) -> bool:
    """True when the same raw caption can transform into more than one text (ignoring caption dropout)."""
    return cfg.shuffle or cfg.tag_dropout > 0 or cfg.wildcard


def deterministic_config(cfg: CaptionConfig) -> CaptionConfig:
    """The same transform with every random element disabled (trigger word / prefix / keep_tokens still apply)."""
    return cfg.model_copy(update={"shuffle": False, "tag_dropout": 0.0, "caption_dropout": 0.0})


def transform_caption_deterministic(raw: str | StructuredCaption, cfg: CaptionConfig) -> str:
    """Validation / cache-friendly variant: no shuffle, no dropout; wildcards resolve to a fixed choice."""
    return transform_caption(raw, deterministic_config(cfg), random.Random(0)) or ""


def caption_variants_for_cache(
    raw: str | StructuredCaption, cfg: CaptionConfig, n: int, seed: int
) -> list[str]:
    """Deterministic, bounded set of transformed captions for a sample whose text encodings are pre-cached.

    Caption dropout is excluded on purpose: the dataset applies it at sampling time (the empty caption
    is always cached separately), so the configured dropout probability is honoured exactly instead
    of depending on how many of the ``n`` draws happened to be dropped.
    """
    if not is_stochastic(cfg):
        return [transform_caption_deterministic(raw, cfg)]
    rng = random.Random(seed)
    cfg_nd = cfg.model_copy(update={"caption_dropout": 0.0})
    out = {transform_caption(raw, cfg_nd, rng) or "" for _ in range(n)}
    return sorted(out)
