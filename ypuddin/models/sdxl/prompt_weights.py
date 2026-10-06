"""Inference prompt weights, separate from sd-scripts training caption weights."""

from __future__ import annotations

import math
import re

from .caption_weights import parse_caption_weights


def comfy_prompt_spans(prompt: str, weight: float = 1.0) -> list[tuple[str, float]]:
    escaped = prompt.replace(r"\(", "\x00L").replace(r"\)", "\x00R")

    def split(text, scale):
        segments, depth, start = [], 0, 0
        for position, char in enumerate(text):
            if char == "(":
                if depth == 0 and position > start:
                    segments.append(text[start:position])
                    start = position
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    segments.append(text[start:position + 1])
                    start = position + 1
        if start < len(text):
            segments.append(text[start:])
        result = []
        for section in segments:
            if len(section) >= 2 and section.startswith("(") and section.endswith(")"):
                body = section[1:-1]
                amount = scale * 1.1
                head, colon, tail = body.rpartition(":")
                if colon and head:
                    try:
                        amount = float(tail)
                        body = head
                    except ValueError:
                        pass
                if not math.isfinite(amount):
                    raise ValueError("提示词权重必须为有限数值")
                result.extend(split(body, amount))
            else:
                result.append((section, scale))
        return result

    return [(text.replace("\x00L", "(").replace("\x00R", ")"), scale) for text, scale in split(escaped, weight)]


def prompt_chunks(tokenizer, prompt: str, mode: str, length: int = 77) -> list[tuple[list[int], list[float]]]:
    """Each chunk retains its BOS/EOS/padding; long prompts are not training captions."""
    if mode not in {"comfyui", "a1111"}:
        raise ValueError(f"未知采样兼容模式：{mode}")
    spans = comfy_prompt_spans(prompt) if mode == "comfyui" else parse_caption_weights(prompt, prompt=True)
    if mode == "comfyui":
        # ComfyUI separates embedding markers even without a textual-inversion directory.
        spans = [(part, factor) for span, factor in spans for part in re.split(r"(?<=\s)(?=embedding:)", span)]
    capacity = length - 2
    chunks, ids, weights = [], [], []
    comma = tokenizer.get_vocab().get(",</w>")
    last_comma = -1

    def finish():
        nonlocal ids, weights, last_comma
        chunks.append((
            [tokenizer.bos_token_id, *ids, tokenizer.eos_token_id]
            + [tokenizer.pad_token_id] * (capacity - len(ids)),
            [1.0, *weights] + [1.0] * (capacity + 1 - len(weights)),
        ))
        ids, weights, last_comma = [], [], -1

    for span, factor in spans:
        if mode == "a1111" and span == "BREAK" and factor == -1:
            finish()
            continue
        tokens = tokenizer(span, add_special_tokens=False, truncation=False, verbose=False).input_ids
        if mode == "comfyui":
            # ComfyUI keeps short weighted segments together at a chunk boundary.
            if len(tokens) < 8 and len(ids) + len(tokens) > capacity:
                finish()
            for token in tokens:
                if len(ids) == capacity:
                    finish()
                ids.append(token)
                weights.append(factor)
        else:
            for token in tokens:
                if token == comma:
                    last_comma = len(ids)
                elif len(ids) == capacity and last_comma >= 0 and len(ids) - last_comma <= 20:
                    moved_ids, moved_weights = ids[last_comma + 1:], weights[last_comma + 1:]
                    ids, weights = ids[:last_comma + 1], weights[:last_comma + 1]
                    finish()
                    ids, weights = moved_ids, moved_weights
                if len(ids) == capacity:
                    finish()
                ids.append(token)
                weights.append(factor)
    if ids or not chunks:
        finish()
    return chunks
