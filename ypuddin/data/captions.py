"""Caption loading and train-time transforms (shuffle, dropout, trigger word, wildcards...)."""

from __future__ import annotations

import json
import os
import random
import re
import tempfile
from pathlib import Path

from ypuddin.config import CaptionConfig

from .caption_json import StructuredCaption, edited_content, load_caption, render, unique
from .spatial_captions import SpatialCaption, render_spatial_caption

_WILDCARD = re.compile(r"\{([^{}]*)\}")


def escape_literal_caption_tag(text: str) -> str:
    """Quote literal tag characters before entering a weighted caption."""
    return re.sub(r"(?<!\\)([()\[\]])", lambda match: "\\" + match.group(0), text)


def weighted_caption_issues(text: str) -> list[dict[str, str]]:
    plain = re.sub(r"\\.", "", text)
    stack = []
    balanced = True
    for char in plain:
        if char in "([":
            stack.append(char)
        elif char in ")]":
            if not stack or stack.pop() != {")": "(", "]": "["}[char]:
                # Round and square weight syntax may nest, but each needs its own closer.
                balanced = False
                break
    balanced = balanced and not stack
    issues = []
    if not balanced:
        issues.append({
            "severity": "warning", "code": "weighted_caption_unbalanced",
            "message": "标签中有未配对的权重括号；若括号属于表情或标签本身，请在括号前加反斜杠转义。",
        })
    if re.search(r"\w\s+\([^():]+\)", plain):
        issues.append({
            "severity": "warning", "code": "weighted_caption_literal_parentheses",
            "message": "标签中有“名称 (限定词)”形式；括号会被解释为权重，若需保留字面括号，请在括号前加反斜杠转义。",
        })
    return issues


def read_training_caption(
    path: str | Path | None, fallback: str | None = None, *, require_known_format: bool = False
) -> str | StructuredCaption | SpatialCaption:
    """Read the selected caption; training callers require a recognized JSON format.

    Inspection also uses this reader and keeps unknown documents available for the
    read-only structured editor. Only training may interpret the result as conditioning.
    """
    if path and Path(path).exists():
        if Path(path).suffix.lower() == ".json":
            return load_caption(path, require_known_format=require_known_format)[1]
        text = Path(path).read_text(encoding="utf-8", errors="replace").strip()
        if text:
            return text
    return fallback or ""


def read_caption(path: str | Path | None, fallback: str | None = None) -> str:
    raw = read_training_caption(path, fallback)
    return raw.text() if isinstance(raw, (StructuredCaption, SpatialCaption)) else raw


def read_editable_caption(path: str | Path | None, fallback: str | None = None) -> str:
    """Tag-editor text; structured natural language remains separate and untouched."""
    raw = read_training_caption(path, fallback)
    if isinstance(raw, SpatialCaption):
        return ""
    if isinstance(raw, StructuredCaption):
        return (", ").join(unique((raw.trigger, *raw.fixed, *raw.appearance, *raw.tags, *raw.environment)))
    return raw


def caption_description(path: str | Path | None) -> str:
    """Read-only structured prose for the editor; ordinary TXT has no separate description."""
    raw = read_training_caption(path)
    if isinstance(raw, SpatialCaption):
        return raw.text()
    return raw.nl if isinstance(raw, StructuredCaption) else ""


def caption_content(
    path: str | Path, text: str | None = None, *, description: str | None = None,
    fields: list[dict] | None = None, revision: str | None = None,
) -> str:
    path = Path(path)
    if description is not None and path.suffix.lower() != ".json":
        raise ValueError("separate descriptions require a JSON caption")
    if fields is not None and path.suffix.lower() != ".json":
        raise ValueError("structured fields require a JSON caption")
    if path.suffix.lower() != ".json" and text is None:
        raise ValueError("TXT captions require caption text")
    return (
        edited_content(path, text, description=description, fields=fields, revision=revision)
        if path.suffix.lower() == ".json"
        else text.strip() + "\n"
    )


def write_caption(
    path: str | Path, text: str | None = None, *, description: str | None = None,
    fields: list[dict] | None = None, revision: str | None = None,
) -> None:
    """Atomically replace one caption. Each concurrent writer gets its own temporary file."""
    path = Path(path)
    content = caption_content(path, text, description=description, fields=fields, revision=revision)
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


def _split_weighted_tags(text: str, separator: str, *, boundaries: set[int] | None = None) -> list[str]:
    if not separator:
        raise ValueError("empty separator")
    parts, start, index = [], 0, 0
    round_depth = square_depth = 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if (
            not round_depth and not square_depth and text.startswith(separator, index)
            and (boundaries is None or index in boundaries)
        ):
            parts.append(text[start:index])
            index += len(separator)
            start = index
            continue
        if char == "(":
            round_depth += 1
        elif char == ")":
            round_depth = max(0, round_depth - 1)
        elif char == "[":
            square_depth += 1
        elif char == "]":
            square_depth = max(0, square_depth - 1)
        index += 1
    parts.append(text[start:])
    return parts


def _weighted_groups(tokens: list[str]) -> list[str]:
    # JSON string fields may already have been split on commas. Rejoin only
    # bracketed fragments; complete array entries remain indivisible tags.
    positions, offset = set(), 0
    for token in tokens[:-1]:
        offset += len(token)
        positions.add(offset)
        offset += 2
    return _split_weighted_tags(
        ", ".join(tokens), ", ", boundaries=positions
    ) if tokens else []


def _caption_config_error(cfg: CaptionConfig, family: str | None) -> tuple[str, str] | None:
    if cfg.structured_mode == "natural_language":
        return "structured_mode", "旧自然语言结构标签用途已停用，请重新选择原生格式或 GBC 转 MOSS Layout。"
    if cfg.structured_mode == "gbc_to_moss":
        if family is not None and family != "moss_image":
            return "structured_mode", "GBC 转 MOSS Layout 仅用于 MOSS-Image。"
        if cfg.content_type not in {"auto", "moss_layout", "gbc"}:
            return "structured_mode", "GBC 转 MOSS Layout 需选择 MOSS JSON 或自动识别；其他标签请选择原生格式。"
    if cfg.content_type == "gbc" and cfg.structured_mode == "native":
        return "structured_mode", "当前模型没有原生 GBC 图入口；MOSS-Image 可选择 GBC 转 MOSS Layout。"
    if family is not None and cfg.content_type != "auto":
        from ypuddin.models import get_family

        spec = get_family(family).spec
        supported = spec.caption_content_types if "json" in spec.caption_formats else ("text",)
        content_type = "moss_layout" if cfg.content_type == "gbc" else cfg.content_type
        if content_type not in supported:
            return "content_type", f"{spec.label or family} 不支持 {cfg.content_type} 标签内容。"
    return None


def caption_execution_issues(cfg) -> list[dict[str, str]]:
    """Validate selected caption uses at execution, keeping legacy recipes editable."""
    configs = [("dataset.caption", cfg.dataset.caption)]
    for prefix, sources in (
        ("dataset.sources", cfg.dataset.sources),
        ("validation.sources", cfg.validation.sources if cfg.validation.enabled else []),
    ):
        configs.extend((f"{prefix}.{index}.caption", source.caption)
                       for index, source in enumerate(sources) if source.caption is not None)
    issues = []
    for prefix, caption_cfg in configs:
        error = _caption_config_error(caption_cfg, cfg.model.family)
        if error is not None:
            field, message = error
            issues.append({"loc": f"{prefix}.{field}", "msg": message, "code": "caption_mode"})
    return issues


def validate_caption_content(raw, cfg: CaptionConfig, family: str | None = None) -> None:
    error = _caption_config_error(cfg, family)
    if error is not None:
        raise ValueError(error[1])
    kind = raw.format if isinstance(raw, SpatialCaption) else "anima_json" if isinstance(raw, StructuredCaption) else "text"
    expected = "gbc" if cfg.content_type == "moss_layout" and cfg.structured_mode == "gbc_to_moss" else cfg.content_type
    matches = expected in ("auto", kind) or expected == "text" and kind == "anima_json"
    if not matches:
        if cfg.content_type == "moss_layout" and cfg.structured_mode == "gbc_to_moss":
            raise ValueError("GBC 转 MOSS Layout 需要 GBC 图标注；已有 MOSS Layout 请选原生格式。")
        raise ValueError(f"标签内容为 {kind}，与选择的 {cfg.content_type} 不符。")
    if not isinstance(raw, SpatialCaption):
        return
    if cfg.wildcard or cfg.weighted:
        raise ValueError("布局标签需关闭通配符与提示词权重。")
    if raw.format == "gbc":
        if cfg.structured_mode != "gbc_to_moss":
            raise ValueError("当前模型没有原生 GBC 图入口；MOSS-Image 可选择 GBC 转 MOSS Layout。")
    elif cfg.structured_mode != "native":
        raise ValueError("MOSS Layout 标签请选择原生格式；转换用途仅接收 GBC 图。")
    if family is not None:
        from ypuddin.models import get_family

        if "moss_layout" not in get_family(family).spec.native_caption_types:
            raise ValueError("MOSS Layout 标签仅用于 MOSS-Image。")


def transform_spatial_caption(raw: SpatialCaption, cfg: CaptionConfig, rng: random.Random, *,
                              geometry: dict | None = None, image_fit: str = "crop",
                              flip: bool = False, family: str | None = None) -> str | None:
    validate_caption_content(raw, cfg, family)
    text = render_spatial_caption(raw, mode=cfg.structured_mode, geometry=geometry,
                                  image_fit=image_fit, flip=flip)
    if cfg.caption_dropout > 0 and rng.random() < cfg.caption_dropout:
        return None
    data = json.loads(text)
    # LoRA trigger and surrounding prose belong to the global caption, outside box syntax.
    data["summary"]["caption"] = _transform_structured(
        StructuredCaption(nl=data["summary"]["caption"]), cfg, rng,
    )
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def transform_caption(raw: str | StructuredCaption | SpatialCaption, cfg: CaptionConfig, rng: random.Random) -> str | None:
    """Returns the caption to encode, or ``None`` when this sample is dropped to unconditional."""
    validate_caption_content(raw, cfg)
    if isinstance(raw, SpatialCaption):
        return transform_spatial_caption(raw, cfg, rng)
    if cfg.caption_dropout > 0 and rng.random() < cfg.caption_dropout:
        return None
    if isinstance(raw, StructuredCaption):
        return _transform_structured(raw, cfg, rng)
    text = raw
    if cfg.wildcard:
        text = _expand_wildcards(text, rng)
    sep = cfg.separator
    parts = (_split_weighted_tags(text, sep) if cfg.weighted else text.split(sep)) if text else []
    tags = [t.strip() for t in parts] if text else []
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
        if cfg.weighted:
            rest = _weighted_groups(rest)
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
