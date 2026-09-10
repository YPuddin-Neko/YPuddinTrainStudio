"""Caption loading and train-time transforms (shuffle, dropout, trigger word, wildcards...)."""

from __future__ import annotations

import random
import re
from pathlib import Path

from ypuddin.config import CaptionConfig

_WILDCARD = re.compile(r"\{([^{}]*)\}")


def read_caption(path: str | None, fallback: str | None) -> str:
    if path and Path(path).exists():
        text = Path(path).read_text(encoding="utf-8", errors="replace").strip()
        if text:
            return text
    return fallback or ""


def _expand_wildcards(text: str, rng: random.Random) -> str:
    while (m := _WILDCARD.search(text)) is not None:
        options = m.group(1).split("|")
        text = text[: m.start()] + rng.choice(options).strip() + text[m.end() :]
    return text


def transform_caption(raw: str, cfg: CaptionConfig, rng: random.Random) -> str | None:
    """Returns the caption to encode, or ``None`` when this sample is dropped to unconditional."""
    if cfg.caption_dropout > 0 and rng.random() < cfg.caption_dropout:
        return None
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


def caption_variants_for_cache(raw: str, cfg: CaptionConfig, n: int, seed: int) -> list[str]:
    """Deterministic set of transformed captions used when text encodings must be pre-cached."""
    rng = random.Random(seed)
    out: list[str] = []
    for _ in range(n):
        c = transform_caption(raw, cfg, rng)
        out.append("" if c is None else c)
    return sorted(set(out))
