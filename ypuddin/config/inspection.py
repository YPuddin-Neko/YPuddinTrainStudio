"""Inspect saved fields without loading models or changing the input configuration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from .export import project_config_for_storage
from .issues import validation_issues
from .schema import TrainConfig


def inspect_config(raw: dict[str, Any]) -> dict[str, Any]:
    schema = TrainConfig.json_schema()
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
                walk(item, child, child_path, clean[key])
        elif isinstance(value, list) and node.get("type") == "array" and isinstance(node.get("items"), dict):
            for index, item in enumerate(value):
                walk(item, node["items"], [*path, index], clean[index])

    walk(raw, schema, [], cleaned)
    errors, advice = [], []
    try:
        validated = TrainConfig.model_validate(project_config_for_storage(cleaned))
    except ValidationError as exc:
        errors = validation_issues(exc)
    else:
        from ypuddin.models import get_family
        from ypuddin.train.advice import value_advice

        try:
            advice = value_advice(validated, get_family(validated.model.family).spec.latent.align)
        except KeyError:
            advice = []  # Unknown families are reported by the plan.
    return {"fields": fields, "errors": errors, "advice": advice}
