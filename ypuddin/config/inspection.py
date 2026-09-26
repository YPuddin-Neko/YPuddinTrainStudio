"""Inspect saved fields without loading models or changing the input configuration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from .io import deep_merge
from .optimizer_rules import optimizer_key
from .schema import TrainConfig
from .show_when import ShowWhenError, evaluate

# These sections gate their runtime consumers with the same configuration choices.
# Model paths, resume settings and arbitrary plugin arguments are never pruned by visibility.
_CONDITIONAL_SECTIONS = {"objective", "scheduler", "optimizer", "sampling", "validation"}


def inspect_config(raw: dict[str, Any]) -> dict[str, Any]:
    schema = TrainConfig.json_schema()
    effective = deep_merge(TrainConfig().to_dict(), raw)
    if isinstance(effective.get("optimizer"), dict):
        effective["optimizer"]["type"] = optimizer_key(str(effective["optimizer"].get("type", "adamw")))
    known_optimizer = isinstance(effective.get("optimizer"), dict) and effective["optimizer"].get(
        "type"
    ) in schema.get("x-optimizer-capabilities", {})
    cleaned = deepcopy(raw)
    fields: list[dict[str, Any]] = []

    def resolve(node: dict[str, Any], value: Any) -> dict[str, Any]:
        if "$ref" in node:
            target = schema
            for key in node["$ref"].removeprefix("#/").split("/"):
                target = target[key]
            return resolve({**target, **{k: v for k, v in node.items() if k != "$ref"}}, value)
        if "anyOf" in node:
            for variant in node["anyOf"]:
                variant = resolve(variant, value)
                if (isinstance(value, dict) and variant.get("type") == "object") or (
                    isinstance(value, list) and variant.get("type") == "array"
                ):
                    return {**node, **variant}
        return node

    def add(path: list[str | int], value: Any, kind: str, condition: str | None = None) -> None:
        fields.append(
            {
                "loc": ".".join(map(str, path)),
                "path": path,
                "value": value,
                "kind": kind,
                "condition": condition,
            }
        )

    def walk(value: Any, node: dict[str, Any], path: list[str | int], clean: Any) -> None:
        node = resolve(node, value)
        if isinstance(value, dict) and node.get("type") == "object":
            properties = node.get("properties", {})
            extra = node.get("additionalProperties", True)
            for key, item in value.items():
                child_path = [*path, key]
                if key not in properties:
                    if extra is False:
                        add(child_path, item, "unknown")
                        clean.pop(key, None)
                    elif isinstance(extra, dict):
                        walk(item, extra, child_path, clean[key])
                    continue
                child = resolve(properties[key], item)
                condition = child.get("x-ui", {}).get("show_when")
                if (
                    path
                    and path[0] in _CONDITIONAL_SECTIONS
                    and (path[0] != "optimizer" or known_optimizer)
                    and condition
                    and "default" in child
                    and item != child["default"]
                ):
                    try:
                        if not evaluate(condition, effective):
                            add(child_path, item, "inactive", condition)
                    except (ShowWhenError, TypeError, ValueError):
                        pass
                walk(item, child, child_path, clean[key])
        elif isinstance(value, list) and node.get("type") == "array" and isinstance(node.get("items"), dict):
            for index, item in enumerate(value):
                walk(item, node["items"], [*path, index], clean[index])

    walk(raw, schema, [], cleaned)
    errors, advice = [], []
    try:
        validated = TrainConfig.model_validate(cleaned)
    except ValidationError as exc:
        errors = [{"loc": ".".join(map(str, item["loc"])), "msg": item["msg"]} for item in exc.errors()]
    else:
        from ypuddin.models import get_family
        from ypuddin.train.advice import value_advice

        try:
            advice = value_advice(validated, get_family(validated.model.family).spec.latent.align)
        except KeyError:
            advice = []  # Unknown families are reported by the plan.
    # Invalid values need correction, not deletion disguised as cleanup.
    fields = [
        field
        for field in fields
        if field["kind"] == "unknown"
        or not any(
            not error["loc"] or field["loc"] == error["loc"] or field["loc"].startswith(error["loc"] + ".")
            for error in errors
        )
    ]
    return {"fields": fields, "errors": errors, "advice": advice}
