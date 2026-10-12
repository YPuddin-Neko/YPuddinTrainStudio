"""Caption batches and approximate text lengths for memory planning."""

from __future__ import annotations

import json
import re
from ctypes import c_float
from pathlib import Path
from typing import Any

from ypuddin.config import SamplePrompt, TrainConfig
from ypuddin.data.captions import is_stochastic
from ypuddin.data.dataset import DataLayout, TrainDataset
from ypuddin.models.base import LatentSpec

_TOKEN_LIMIT = 1024
_PAD_FLOOR = 512
_TEXT_PARTS = re.compile(r"[A-Za-z0-9_]+|[^\s]")


class _TextShapeUnavailable(ValueError):
    def __init__(self, loc: str, error: Exception):
        super().__init__(str(error))
        self.loc = loc


def _clip_config_paths(cfg: TrainConfig) -> tuple[Path, Path]:
    from ypuddin.models.sdxl.loading import component_path, config_asset

    return tuple(config_asset(component_path(cfg.model.dit_path or ".", name, override), name)
                 for name, override in (("text_encoder", cfg.model.text_encoder_path),
                                        ("text_encoder_2", cfg.model.text_encoder_2_path)))


def _pairs(cfg: TrainConfig, spec: LatentSpec, items: list, *, validation: bool) -> tuple:
    dataset = TrainDataset(
        items, latent_spec=spec, latent_cache=None, flip=False, masked_loss=False,
        seed=cfg.validation.seed if validation else cfg.loop.seed, deterministic_captions=validation,
    )
    pairs = dataset.use_cached_captions()
    active = {""}
    stochastic = False
    for item, variants in zip(items, dataset.caption_variants, strict=True):
        if validation or item.caption_cfg.caption_dropout < 1:
            active.update(variants)
            stochastic |= not validation and is_stochastic(item.caption_cfg)
    return pairs, active, stochastic


def _cold_batches(pairs: list[tuple[str, str]]):
    pending: dict[str, None] = {}
    for _, caption in pairs:
        pending[caption] = None
        if len(pending) == 16:
            yield tuple(pending)
            pending.clear()
    if pending:
        yield tuple(pending)


def _approximate_tokens(text: str, *, byte_fallback: bool = False) -> int:
    # ASCII words average several characters per token; count other symbols
    # separately so long CJK captions do not collapse to a few words.
    tokens = 0
    for part in _TEXT_PARTS.findall(text):
        if part.isascii() and (part.isalnum() or "_" in part):
            tokens += (len(part) + 3) // 4
        else:
            size = len(part.encode("utf-8"))
            tokens += size if byte_fallback else (size + 2) // 3
    return tokens


def _length_summary(captions: set[str], lengths: dict[str, int], *, known: bool) -> dict:
    if not known:
        tokens = _TOKEN_LIMIT
        basis = "token_limit_upper_bound"
    else:
        tokens = max((lengths[text] for text in captions), default=1)
        basis = "character_approximation"
    return {"qwen_tokens": tokens, "t5_tokens": tokens,
            "qwen_padded_tokens": max(_PAD_FLOOR, tokens), "t5_padded_tokens": max(_PAD_FLOOR, tokens),
            "basis": basis}


def _sdxl_prompt_shape(prompt: str, mode: str, length: int) -> dict:
    from ypuddin.models.sdxl.caption_weights import parse_caption_weights
    from ypuddin.models.sdxl.prompt_weights import comfy_prompt_spans

    spans = comfy_prompt_spans(prompt) if mode == "comfyui" else parse_caption_weights(prompt, prompt=True)
    capacity = length - 2
    chunks = used = 0
    weighted = False
    for span, factor in spans:
        if mode == "a1111" and span == "BREAK" and factor == -1:
            chunks += 1
            used = 0
            continue
        # CLIP's byte vocabulary can split each CJK character into several tokens.
        tokens = _approximate_tokens(span, byte_fallback=True)
        if not tokens:
            continue
        weighted |= c_float(factor).value != 1
        if mode == "comfyui" and tokens < 8 and used + tokens > capacity:
            chunks += 1
            used = 0
        full, used = divmod(used + tokens, capacity)
        if full and not used:
            full -= 1
            used = capacity
        chunks += full
    chunks += int(bool(used) or not chunks)
    return {"chunks": [chunks, chunks], "weighted": [weighted, weighted], "mode": mode,
            "context_length": length, "basis": "character_approximation"}


def _sdxl_prompts(cfg: TrainConfig, prompts: set[str]) -> dict[str, dict]:
    lengths = []
    for path, loc in zip(_clip_config_paths(cfg), ("model.text_encoder_path", "model.text_encoder_2_path"), strict=True):
        try:
            lengths.append(int(json.loads(path.read_text(encoding="utf-8"))["max_position_embeddings"]))
        except (OSError, ValueError, KeyError) as error:
            raise _TextShapeUnavailable(loc, error) from error
    length = lengths[0]
    if length != lengths[1] or length < 3:
        raise _TextShapeUnavailable("model.text_encoder_2_path", ValueError("SDXL CLIP 编码器的上下文长度不一致"))
    return {prompt: _sdxl_prompt_shape(prompt, cfg.sampling.noise, length) for prompt in prompts}


def text_plan_inputs(cfg: TrainConfig, latent_spec: LatentSpec, layout: DataLayout | None) -> dict[str, Any]:
    """Read captions and prompt geometry without tokenizers, images or weights.

    Lengths are approximate; cache batches still follow actual caption variants
    and deduplication. Unknown online variants retain the truncation upper bound.
    """
    errors = []
    unknowns = []
    train: list[tuple[str, str]] = []
    validation: list[tuple[str, str]] = []
    train_text, val_text = {""}, {""}
    online_stochastic = False
    known = {"train": layout is not None, "validation": layout is not None, "prompt": True}
    if layout is None:
        unknowns.append({"loc": "dataset.sources", "reason": "data_layout_unavailable"})
    if layout is not None:
        for label, items in (("train", layout.items), ("validation", layout.validation_items)):
            try:
                values, active, stochastic = _pairs(cfg, latent_spec, items, validation=label == "validation")
                if label == "train":
                    train = values
                    train_text = active
                    online_stochastic = stochastic
                else:
                    validation = values
                    val_text = active
            except (OSError, ValueError) as error:
                known[label] = False
                errors.append({"loc": "dataset.sources" if label == "train" else "validation.sources", "msg": str(error)})
    prompts = set()
    values = []
    if cfg.sampling is not None and cfg.sampling.enabled:
        try:
            values = [prompt if isinstance(prompt, SamplePrompt) else SamplePrompt.model_validate(prompt)
                      for prompt in cfg.sampling.prompts]
        except (ValueError, TypeError) as error:
            known["prompt"] = False
            errors.append({"loc": "sampling.prompts", "msg": str(error)})
        if cfg.sampling.prompts_file:
            try:
                from .trainer import _load_prompts_file

                values.extend(_load_prompts_file(cfg.sampling.prompts_file))
            except (OSError, ValueError, TypeError) as error:
                known["prompt"] = False
                errors.append({"loc": "sampling.prompts_file", "msg": str(error)})
        prompts = {text for prompt in values for text in (prompt.prompt, prompt.negative)}
    separate = cfg.model.family == "sdxl"
    captions = set(train) | set(validation) | {("_prompts", "")}
    if not separate:
        captions.update(("_prompts", text) for text in prompts)
    ordered = sorted(captions)
    cache_known = known["train"] and known["validation"] and (separate or known["prompt"])
    batch_size = min(16, len({text for _, text in ordered})) if cache_known else 16
    if online_stochastic:
        unknowns.append({"loc": "dataset.caption", "reason": "online_caption_variants"})
    result = {
        "cache_batch_size": batch_size,
        "prompt_cache_batch_size": int(separate and (bool(prompts) or not known["prompt"])),
        "preview_items": [{"width": (prompt.width or cfg.sampling.width) // latent_spec.align * latent_spec.align,
                           "height": (prompt.height or cfg.sampling.height) // latent_spec.align * latent_spec.align,
                           "prompt_steps": prompt.steps, "prompt_cfg": prompt.cfg,
                           "positive": {}, "negative": {}} for prompt in values] if known["prompt"] else None,
        "errors": errors, "unknowns": unknowns,
    }
    if cfg.model.family == "sdxl":
        result["sdxl_prompt_batches"] = []
        if prompts or not known["prompt"]:
            try:
                if not known["prompt"]:
                    raise _TextShapeUnavailable("sampling.prompts_file", ValueError("无法读取完整预览提示词"))
                shapes = _sdxl_prompts(cfg, prompts)
                unique = {(*shape["chunks"], *shape["weighted"]): shape for shape in shapes.values()}
                result["sdxl_prompt_batches"] = [unique[key] for key in sorted(unique)]
                for item, prompt in zip(result["preview_items"], values, strict=True):
                    item["positive"] = {"prompt_spec": shapes[prompt.prompt]}
                    item["negative"] = {"prompt_spec": shapes[prompt.negative]}
            except Exception as error:
                result["sdxl_prompt_batches"] = None
                result["preview_items"] = None
                unknowns.append({"loc": getattr(error, "loc", "sampling.prompts"),
                                 "reason": "sdxl_preview_length", "detail": str(error)})
        return result
    if cfg.model.family != "anima":
        return result
    all_text = {text for _, text in ordered} | prompts
    lengths = {text: min(_TOKEN_LIMIT, _approximate_tokens(text) + 1) for text in all_text}
    result["anima_lengths"] = {
        "cache": _length_summary({text for _, text in ordered}, lengths, known=cache_known),
        "cached_training": _length_summary(train_text, lengths, known=known["train"]),
        "online": _length_summary(train_text, lengths, known=known["train"] and not online_stochastic),
        "validation": _length_summary(val_text, lengths, known=known["validation"]),
        "prompt": _length_summary(prompts, lengths, known=known["prompt"]),
    }
    if result["preview_items"] is not None:
        for item, prompt in zip(result["preview_items"], values, strict=True):
            item["positive"] = _length_summary({prompt.prompt}, lengths, known=True)
            item["negative"] = _length_summary({prompt.negative}, lengths, known=True)
    if cache_known:
        batches = {(len(batch), max(lengths[text] for text in batch))
                   for batch in _cold_batches(ordered)}
        result["anima_cache_batches"] = [{"batch_size": b, "qwen_tokens": tokens, "t5_tokens": tokens}
                                        for b, tokens in sorted(batches)]
    else:
        result["anima_cache_batches"] = [{"batch_size": batch_size, "qwen_tokens": 1024, "t5_tokens": 1024}]
    return result
