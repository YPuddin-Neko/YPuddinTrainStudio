"""Project configuration downloads without changing saved training recipes."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .schema import TrainConfig

NATIVE_INACTIVE_BUCKET_FIELDS = frozenset({
    "resolutions", "aspect_ratio_limit", "area_tolerance", "bucket_step", "bucket_no_upscale",
})


def public_config_dict(config: TrainConfig) -> dict[str, Any]:
    """Keep an omitted checkpoint strategy omitted across public config round trips."""
    data = config.to_dict()
    if (
        config.dataset.resolution_mode == "native" and config.checkpoint.resume
        and "native_batching_version" not in config.dataset.model_fields_set
    ):
        data["dataset"].pop("native_batching_version", None)
    return data


def project_config_for_export(config: TrainConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Omit bucket-only geometry from native exports, preserving partial recipes."""
    data = deepcopy(public_config_dict(config) if isinstance(config, TrainConfig) else dict(config))
    dataset = data.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("resolution_mode") != "native":
        return data
    for field in NATIVE_INACTIVE_BUCKET_FIELDS:
        dataset.pop(field, None)
    for section in ("dataset", "validation"):
        values = data.get(section)
        if isinstance(values, dict):
            for source in values.get("sources", []):
                if isinstance(source, dict):
                    source.pop("resolutions", None)
    return data
