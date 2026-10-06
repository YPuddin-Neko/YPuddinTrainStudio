"""Parse bracket emphasis in SDXL captions into weighted text spans."""

from __future__ import annotations

import math
import re

_EXPLICIT_WEIGHT = re.compile(r":([+-]?(?:\d+(?:\.\d*)?|\.\d+))\)")
_PROMPT_WEIGHT = re.compile(r":\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)")
_PROMPT_BREAK = re.compile(r"\s*\bBREAK\b\s*")


def parse_caption_weights(caption: str, *, prompt: bool = False) -> list[tuple[str, float]]:
    """Parentheses multiply by 1.1, square brackets by its inverse; backslashes escape syntax."""
    characters: list[str] = []
    weights: list[float] = []
    openings: dict[str, list[int]] = {"(": [], "[": []}

    def close(kind: str, factor: float) -> None:
        start = openings[kind].pop()
        for index in range(start, len(weights)):
            weights[index] *= factor

    position = 0
    while position < len(caption):
        if prompt and (match := _PROMPT_BREAK.match(caption, position)):
            characters.append("BREAK")
            weights.append(-1.0)
            position = match.end()
            continue
        char = caption[position]
        if char == "\\":
            position += 1
            if position == len(caption):
                break
            if prompt and caption[position] not in "()[]\\":
                continue
            char = caption[position]
        elif char in openings:
            openings[char].append(len(characters))
            position += 1
            continue
        elif char == ":" and openings["("] and (match := (_PROMPT_WEIGHT if prompt else _EXPLICIT_WEIGHT).match(caption, position)):
            close("(", float(match[1]))
            position = match.end()
            continue
        elif char in ")]":
            kind = "(" if char == ")" else "["
            if openings[kind]:
                close(kind, 1.1 if kind == "(" else 1 / 1.1)
                position += 1
                continue
        characters.append(char)
        weights.append(1.0)
        position += 1

    for kind, factor in (("(", 1.1), ("[", 1 / 1.1)):
        while openings[kind]:
            close(kind, factor)
    if any(not math.isfinite(weight) or abs(weight) > 3.4028234663852886e38 for weight in weights):
        raise ValueError("SDXL caption weights must be finite")
    if not characters:
        return [("", 1.0)]

    spans = []
    start = 0
    for end in range(1, len(characters) + 1):
        if end == len(characters) or weights[end] != weights[start] or (
            prompt and (characters[start] == "BREAK" or characters[end] == "BREAK")
        ):
            spans.append(("".join(characters[start:end]), weights[start]))
            start = end
    return spans
