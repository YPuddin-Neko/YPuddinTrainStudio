"""Resume identity for optional training features added after legacy checkpoints."""

from __future__ import annotations

from typing import Any

from ypuddin.config import TrainConfig

STATE_KEY = "training_feature_contract"
_DEFAULTS = {
    "training.backbone_lr": None,
    "training.text_encoder_lr": None,
    "training.text_encoder_2_lr": None,
    "training.llm_adapter_lr": None,
    "training.self_attn_lr": None,
    "training.cross_attn_lr": None,
    "training.mlp_lr": None,
    "training.modulation_lr": None,
    "optimizer.cpu_offload": False,
    "optimizer.exclude_bias_norm_from_weight_decay": False,
    "dataset.caption.weighted": False,
    "objective.noise_offset": 0.0,
    "objective.multires_noise_iterations": 0,
    "objective.multires_noise_discount": 0.3,
}


def training_feature_contract(cfg: TrainConfig) -> dict[str, Any]:
    values = {}
    for path in _DEFAULTS:
        value = cfg
        for key in path.split("."):
            value = getattr(value, key)
        values[path] = value
    return {"version": 1, "values": values}


def validate_training_feature_contract(expected: dict[str, Any], saved: Any) -> None:
    if saved is None:
        previous = _DEFAULTS
    elif (
        not isinstance(saved, dict)
        or type(saved.get("version")) is not int
        or saved["version"] != 1
        or set(saved) != {"version", "values"}
        or not isinstance(saved["values"], dict)
        or saved["values"].keys() != _DEFAULTS.keys()
    ):
        raise ValueError("训练状态的组件学习率、优化器卸载或标签与噪声设置记录无效，不能精确恢复")
    else:
        previous = saved["values"]
    changed = [path for path, value in expected["values"].items() if value != previous[path]]
    if changed:
        raise ValueError(
            "以下训练设置与恢复点不同，不能精确恢复：" + "、".join(changed)
            + "。请恢复原设置，或从导出权重开始新任务。"
        )


def validate_training_feature_resume(cfg: TrainConfig, saved: Any) -> None:
    validate_training_feature_contract(training_feature_contract(cfg), saved)
