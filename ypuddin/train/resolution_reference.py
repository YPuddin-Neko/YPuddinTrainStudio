"""Derive and freeze reference tokens from each family's final training canvases."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ypuddin.config import TrainConfig
from ypuddin.config.timestep_reference import supports_resolution_reference
from ypuddin.data.dataset import DataConfigError
from ypuddin.models import get_family


def automatic_reference(cfg: TrainConfig) -> bool:
    return (
        cfg.objective.timestep_sampling == "resolution_sqrt"
        and cfg.objective.resolution_reference_mode == "auto"
        and supports_resolution_reference(cfg.model.family)
    )


def clear_resolution_reference(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    objective = config.get("objective")
    if isinstance(objective, dict) and "resolution_reference_tokens_resolved" in objective:
        result["objective"] = {
            key: value for key, value in objective.items() if key != "resolution_reference_tokens_resolved"
        }
    return result


def reference_summary(cfg: TrainConfig, counts: Mapping[tuple[int, int], int]) -> dict[str, Any] | None:
    if cfg.objective.timestep_sampling != "resolution_sqrt" or not supports_resolution_reference(cfg.model.family):
        return None
    align = get_family(cfg.model.family).spec.latent.align
    tokens: Counter[int] = Counter()
    for (width, height), count in counts.items():
        if count > 0:
            value = (width // align) * (height // align)
            if value <= 0:
                raise DataConfigError("dataset", "训练画布不能生成有效的图像 token。")
            tokens[value] += count
    total = sum(tokens.values())
    if not total:
        raise DataConfigError("dataset", "没有可用于计算参考 token 的训练样本。")
    cumulative = 0
    # Use the lower median when the two middle samples have different token counts.
    for token_count, count in sorted(tokens.items()):
        cumulative += count
        if cumulative * 2 >= total:
            median = token_count
            break
    objective = cfg.objective
    reference = (
        objective.resolution_reference_tokens_resolved or median
        if automatic_reference(cfg) else objective.resolution_reference_tokens
    )
    low, high = min(tokens), max(tokens)
    return {
        "mode": objective.resolution_reference_mode,
        "training_items": total,
        "token_min": low,
        "token_max": high,
        "weighted_median_tokens": median,
        "reference_tokens": reference,
        "effective_shift_min": objective.shift * math.sqrt(low / reference),
        "effective_shift_max": objective.shift * math.sqrt(high / reference),
    }


def _resume_reference(cfg: TrainConfig) -> int:
    try:
        metadata = json.loads((Path(cfg.checkpoint.resume).expanduser() / "state.json").read_text(encoding="utf-8"))
        baseline = metadata["progress"]["extra"]["adaptive_timestep"]["recipe"]["baseline"]
        value = baseline["resolution_reference_tokens"]
        if baseline.get("resolution_reference_mode") != "auto" or type(value) is not int or value <= 0:
            raise ValueError("恢复点未保存自动参考 token")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise DataConfigError("checkpoint.resume", f"无法读取原训练的自动参考 token：{error}") from error
    supplied = cfg.objective.resolution_reference_tokens_resolved
    if supplied is not None and supplied != value:
        raise DataConfigError("checkpoint.resume", "参考 token 与原训练不同，请沿用原训练配置恢复。")
    return value


def resolve_resolution_reference_config(
    cfg: TrainConfig,
    *,
    plan_result: dict[str, Any] | None = None,
    counts: Mapping[tuple[int, int], int] | None = None,
) -> TrainConfig:
    if not automatic_reference(cfg):
        return cfg
    if cfg.checkpoint.resume:
        value = _resume_reference(cfg)
    elif cfg.objective.resolution_reference_tokens_resolved is not None:
        return cfg
    else:
        summary = None
        if plan_result is not None:
            if not plan_result.get("ok"):
                issue = next(iter(plan_result.get("errors") or []), {})
                raise DataConfigError(issue.get("loc", "objective.resolution_reference_mode"),
                                      issue.get("msg", "训练计划无效，无法确定自动参考 token。"))
            summary = plan_result.get("resolution_reference")
        elif counts is not None:
            summary = reference_summary(cfg, counts)
        value = summary.get("weighted_median_tokens") if isinstance(summary, dict) else None
        if type(value) is not int or value <= 0:
            raise DataConfigError("objective.resolution_reference_mode", "自动参考 token 尚未确定，请先完成训练计划。")
    result = cfg.model_copy(deep=True)
    result.objective.resolution_reference_tokens_resolved = value
    return result
