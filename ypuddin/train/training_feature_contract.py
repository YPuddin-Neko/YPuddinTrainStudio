"""Resume identity for optional training features added after legacy checkpoints."""

from __future__ import annotations

from typing import Any

from ypuddin.config import TrainConfig

STATE_KEY = "training_feature_contract"
_TILED_FAMILIES = {"flux3", "moss_image"}
_LATENT_REVISION = "spatial-v1"
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
    result = {"version": 1, "values": values}
    if cfg.model.family in _TILED_FAMILIES and (cfg.memory.vae_tiling or cfg.memory.cache_encode_tiled):
        result.update(version=2, latent_encoding={
            "family": cfg.model.family,
            "vae_tiling": cfg.memory.vae_tiling,
            "cache_encode_tiled": cfg.memory.cache_encode_tiled,
            "revision": _LATENT_REVISION,
        })
    if cfg.training.task != "standard":
        result["task_recipe"] = {
            "task": cfg.training.task,
            "revision": 1,
            **({"slider": cfg.slider.model_dump(mode="json")} if cfg.training.task == "slider" else {"reference_preprocessing": "single-fit-edge-pad-target-v1"}),
        }
        result["version"] = 3
    return result


def validate_training_feature_contract(expected: dict[str, Any], saved: Any) -> None:
    expected_task = expected.get("task_recipe")
    saved_task = saved.get("task_recipe") if isinstance(saved, dict) else None
    if expected_task != saved_task:
        raise ValueError("训练任务类型或滑块配方与恢复点不同，请恢复原设置或从导出权重开始新任务。")
    if isinstance(saved, dict) and saved.get("version") == 3:
        if not isinstance(saved_task, dict) or saved_task.get("revision") != 1:
            raise ValueError("恢复点的训练任务配方无效")
        saved = {key: value for key, value in saved.items() if key != "task_recipe"}
        saved["version"] = 2 if "latent_encoding" in saved else 1
    previous_latent = None
    if saved is None:
        previous = _DEFAULTS
    elif (
        not isinstance(saved, dict)
        or type(saved.get("version")) is not int
        or saved["version"] not in {1, 2}
        or not isinstance(saved.get("values"), dict)
        or saved["values"].keys() != _DEFAULTS.keys()
    ):
        raise ValueError("训练状态的组件学习率、优化器卸载或标签与噪声设置记录无效，不能精确恢复")
    else:
        if saved["version"] == 1:
            if set(saved) != {"version", "values"}:
                raise ValueError("训练状态的组件学习率、优化器卸载或标签与噪声设置记录无效，不能精确恢复")
        else:
            recipe = saved.get("latent_encoding")
            if (
                set(saved) != {"version", "values", "latent_encoding"}
                or not isinstance(recipe, dict)
                or set(recipe) != {"family", "vae_tiling", "cache_encode_tiled", "revision"}
                or not isinstance(recipe["family"], str)
                or recipe["family"] not in _TILED_FAMILIES
                or type(recipe["vae_tiling"]) is not bool
                or type(recipe["cache_encode_tiled"]) is not bool
                or not (recipe["vae_tiling"] or recipe["cache_encode_tiled"])
                or recipe["revision"] != _LATENT_REVISION
            ):
                raise ValueError("训练状态的 VAE 分块编码记录无效，不能精确恢复。请从导出权重开始新任务。")
            previous_latent = recipe
        previous = saved["values"]
    if expected.get("latent_encoding") != previous_latent:
        if previous_latent is None:
            raise ValueError(
                "恢复点未记录当前 VAE 分块编码方式，不能在精确恢复时启用分块。请从导出权重开始新任务。"
            )
        raise ValueError(
            "VAE 分块编码设置与恢复点不同，不能精确恢复。请恢复原设置，或从导出权重开始新任务。"
        )
    changed = [path for path, value in expected["values"].items() if value != previous[path]]
    if changed:
        raise ValueError(
            "以下训练设置与恢复点不同，不能精确恢复：" + "、".join(changed)
            + "。请恢复原设置，或从导出权重开始新任务。"
        )


def validate_training_feature_resume(cfg: TrainConfig, saved: Any) -> None:
    validate_training_feature_contract(training_feature_contract(cfg), saved)
