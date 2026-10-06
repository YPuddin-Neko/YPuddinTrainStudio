"""Periodic event settings and their historical nullable representation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import TypeAdapter, ValidationError

# A disabled interval keeps this editable value without scheduling an event.
INTERVAL_DEFAULTS: dict[str, dict[str, tuple[int, bool]]] = {
    "checkpoint": {
        "save_every_steps": (100, False),
        "save_every_epochs": (1, True),
        "save_state_every_steps": (100, True),
        "save_state_every_epochs": (1, False),
    },
    "sampling": {"every_steps": (100, False), "every_epochs": (1, True)},
    "validation": {"every_steps": (100, False), "every_epochs": (1, True)},
}
_BOOL = TypeAdapter(bool)
_INT = TypeAdapter(int)


def interval_disabled(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    try:
        return _INT.validate_python(value) == 0
    except ValidationError:
        return False


def normalize_interval_section(section: str, values: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate only supplied intervals; omitted fields still inherit from the base."""
    result = dict(values)
    for field, (fallback, _) in INTERVAL_DEFAULTS[section].items():
        enabled = f"{field}_enabled"
        if field not in result:
            continue
        disabled = interval_disabled(result[field])
        if enabled not in result:
            result[enabled] = not disabled
        if disabled:
            # A cleared interval wins over an enabled flag retained by an old client.
            result[enabled] = False
            result[field] = fallback
    return result


def normalize_legacy_intervals(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize each config layer before merging it with inherited settings."""
    result = dict(config)
    for section in INTERVAL_DEFAULTS:
        if isinstance(result.get(section), Mapping):
            result[section] = normalize_interval_section(section, result[section])
    return result


def project_legacy_interval_section(section: str, values: Mapping[str, Any]) -> dict[str, Any]:
    """Keep checkpoint identity based on the original nullable event schedule."""
    result = dict(values)
    for field, (fallback, _) in INTERVAL_DEFAULTS[section].items():
        enabled = f"{field}_enabled"
        if enabled in result:
            active = _BOOL.validate_python(result.pop(enabled))
            result[field] = result.get(field, fallback) if active else None
    return result


def project_legacy_intervals(config: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(config)
    for section in INTERVAL_DEFAULTS:
        if isinstance(result.get(section), Mapping):
            result[section] = project_legacy_interval_section(section, result[section])
    return result
