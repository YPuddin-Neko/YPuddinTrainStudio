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


def is_stochastic(cfg: CaptionConfig) -> bool:
    """True when the same raw caption can transform into more than one text (ignoring caption dropout)."""
    return cfg.shuffle or cfg.tag_dropout > 0 or cfg.wildcard


def deterministic_config(cfg: CaptionConfig) -> CaptionConfig:
    """The same transform with every random element disabled (trigger word / prefix / keep_tokens still apply)."""
    return cfg.model_copy(update={"shuffle": False, "tag_dropout": 0.0, "caption_dropout": 0.0})


def transform_caption_deterministic(raw: str, cfg: CaptionConfig) -> str:
    """Validation / cache-friendly variant: no shuffle, no dropout; wildcards resolve to a fixed choice."""
    return transform_caption(raw, deterministic_config(cfg), random.Random(0)) or ""


def caption_variants_for_cache(raw: str, cfg: CaptionConfig, n: int, seed: int) -> list[str]:
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
