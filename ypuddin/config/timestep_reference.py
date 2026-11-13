"""The reference token count consumed by resolution-dependent timestep sampling."""

from __future__ import annotations

from typing import Any


def supports_resolution_reference(family: str) -> bool:
    from ypuddin.models import get_family

    spec = get_family(family).spec
    return not spec.retired_reason and "resolution_sqrt" in spec.objective_timestep_sampling


def effective_reference_tokens(objective: Any) -> int:
    if (
        objective.timestep_sampling == "resolution_sqrt"
        and getattr(objective, "resolution_reference_mode", "manual") == "auto"
    ):
        value = getattr(objective, "resolution_reference_tokens_resolved", None)
        if type(value) is not int or value <= 0:
            raise ValueError("自动参考 token 尚未确定，请先完成训练计划。")
        return value
    return objective.resolution_reference_tokens
