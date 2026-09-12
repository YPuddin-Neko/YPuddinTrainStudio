"""Sample loss metadata shared by REST and SSE; no nearest-step approximation."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def finite_loss(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def samples_with_loss(events: Iterable[dict[str, Any]]) -> Iterator[tuple[dict[str, Any], float | None]]:
    """Resolve historical samples within their own training/resume event segment.

    New samples carry their own loss (including explicit null). Older samples can
    use only a preceding training step with the exact same positive step number.
    A resume clears the lookup: the same step in an earlier attempt is not proof
    of the resumed optimizer's loss. Events from later in the file never backfill
    an earlier sample.
    """
    losses: dict[int, float | None] = {}
    for event in events:
        kind = event.get("type")
        if kind in {"run.started", "run.resumed"}:
            losses.clear()
        step = event.get("step")
        valid_step = isinstance(step, int) and not isinstance(step, bool) and step > 0
        if kind == "step" and valid_step:
            losses[step] = finite_loss(event.get("loss"))
        elif kind == "sample.saved":
            loss = None
            if valid_step:
                loss = finite_loss(event["loss"]) if "loss" in event else losses.get(step)
            yield event, loss


def sample_event_loss(event: dict[str, Any], path: Path | None = None) -> float | None:
    """Normalize one live/replayed event using the same rules as the REST listing."""
    if "loss" in event:
        return next(samples_with_loss([event]))[1]
    if path is not None:
        for candidate, loss in samples_with_loss(read_events(path)):
            if candidate == event:
                return loss
    return None
