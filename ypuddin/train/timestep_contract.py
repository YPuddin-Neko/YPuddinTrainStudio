"""Strict resume identity for optional adaptive Anima timestep sampling."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ypuddin.config import TrainConfig

STATE_KEY = "adaptive_timestep"
_INFONOISE_FIELDS = (
    "bins", "warmup_steps", "refresh_every", "ema_decay", "exploration",
    "adaptive_mix", "min_samples", "min_coverage", "gate_sigma",
)


def recipe(cfg: TrainConfig) -> dict[str, Any] | None:
    objective = cfg.objective
    mode = objective.timestep_sampling
    if mode != "resolution_sqrt" and not objective.infonoise_enabled:
        return None
    baseline = {
        "timestep_sampling": mode,
        "t_min": objective.t_min,
        "t_max": objective.t_max,
        "stratified": objective.stratified,
    }
    if mode in {"logit_normal", "shift", "resolution_sqrt"}:
        baseline.update(logit_mean=objective.logit_mean, logit_std=objective.logit_std)
    if mode in {"shift", "resolution_sqrt"}:
        baseline["shift"] = objective.shift
    if mode == "resolution_sqrt":
        baseline["resolution_reference_tokens"] = objective.resolution_reference_tokens
    infonoise = {"enabled": objective.infonoise_enabled}
    if objective.infonoise_enabled:
        infonoise.update({key: getattr(objective, "infonoise_" + key) for key in _INFONOISE_FIELDS})
        infonoise["fixed_gate"] = "sigma-n3"
    return {
        "algorithm": "anima-adaptive-timestep-v1",
        "family": cfg.model.family,
        "baseline": baseline,
        "infonoise": infonoise,
    }


def _same_recipe(saved: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(saved, dict)
            and saved.keys() == expected.keys()
            and all(_same_recipe(saved[key], value) for key, value in expected.items())
        )
    if isinstance(expected, float):
        return type(saved) in (int, float) and saved == expected
    return type(saved) is type(expected) and saved == expected


def validate_resume(cfg: TrainConfig, extra: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate metadata before the caller loads weights, optimizer or sampler arrays."""
    expected = recipe(cfg)
    if not isinstance(extra, Mapping):
        raise ValueError("恢复点的训练附加状态无效，不能精确恢复")
    if STATE_KEY not in extra:
        if expected is None:
            return None
        raise ValueError("恢复点缺少自适应时间步状态，请使用原设置恢复，或从导出权重开始新任务")
    if expected is None:
        raise ValueError("恢复点启用了自适应时间步，请恢复原设置，或从导出权重开始新任务")
    saved = extra[STATE_KEY]
    if (
        not isinstance(saved, dict)
        or set(saved) != {"version", "recipe", "sampler"}
        or type(saved["version"]) is not int
        or saved["version"] != 1
    ):
        raise ValueError("恢复点的自适应时间步状态格式或版本无效，不能精确恢复")
    if not _same_recipe(saved["recipe"], expected):
        raise ValueError("时间步抽样设置与恢复点不同，请恢复原设置，或从导出权重开始新任务")
    if cfg.objective.infonoise_enabled:
        if not isinstance(saved["sampler"], dict):
            raise ValueError("恢复点缺少有效的 InfoNoise 统计，不能精确恢复")
    elif saved["sampler"] is not None:
        raise ValueError("恢复点包含与当前设置不一致的 InfoNoise 统计，不能精确恢复")
    return saved
