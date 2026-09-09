"""UI hints attached to config fields and exported into JSON Schema as ``x-ui``.

The frontend renders the training form from the JSON Schema alone, so every
presentation decision (grouping, ordering, advanced-only, control type,
conditional visibility) lives here next to the field definition.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

GROUPS: tuple[str, ...] = (
    "model",
    "dataset",
    "caption",
    "adapter",
    "objective",
    "optimizer",
    "scheduler",
    "memory",
    "loop",
    "checkpoint",
    "sampling",
    "validation",
    "logging",
)

CONTROLS = {"select", "number", "slider", "switch", "text", "path", "textarea", "tags", "rules", "prompts"}


def ui(
    group: str | None = None,
    *,
    order: int = 0,
    advanced: bool = False,
    control: str | None = None,
    unit: str | None = None,
    show_when: str | None = None,
    step: float | None = None,
) -> dict[str, Any]:
    """Build the ``json_schema_extra`` payload for a field."""
    if group is not None and group not in GROUPS:
        raise ValueError(f"unknown ui group {group!r}")
    if control is not None and control not in CONTROLS:
        raise ValueError(f"unknown ui control {control!r}")
    payload: dict[str, Any] = {"order": order}
    if group is not None:
        payload["group"] = group
    if advanced:
        payload["advanced"] = True
    if control is not None:
        payload["control"] = control
    if unit is not None:
        payload["unit"] = unit
    if show_when is not None:
        payload["show_when"] = show_when
    if step is not None:
        payload["step"] = step
    return {"x-ui": payload}


def F(default: Any = ..., *, help: str | None = None, ui_: dict[str, Any] | None = None, **kwargs: Any) -> Any:  # noqa: N802
    """``pydantic.Field`` shorthand: ``help`` becomes the description, ``ui_`` the x-ui extra."""
    if ui_ is not None:
        kwargs["json_schema_extra"] = ui_
    if help is not None:
        kwargs["description"] = help
    return Field(default, **kwargs)
