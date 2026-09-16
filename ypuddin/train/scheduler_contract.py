"""Authenticate LR scheduler closures before restoring mutable training state.

LambdaLR.state_dict() does not save the Python function or its captured recipe.
Keep config/total_steps semantics aligned with sharded_state, without importing
distributed code into the serialized owner preparation path.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ypuddin.config import TrainConfig, config_hash, load_config
from ypuddin.config.optimizer_rules import optimizer_key
from ypuddin.optim import manages_learning_rate

_SCHEDULER_CLASS = "torch.optim.lr_scheduler.LambdaLR"


def scheduler_recipe(cfg: TrainConfig, total_steps: int) -> dict[str, Any]:
    managed = manages_learning_rate(cfg.optimizer)
    return {
        "version": 1,
        "config": cfg.scheduler.model_dump(mode="json"),
        "total_steps": total_steps,
        "scheduler_class": None if managed else _SCHEDULER_CLASS,
        "managed_by_optimizer": optimizer_key(cfg.optimizer.type) if managed else None,
        "training_kind": "full-model" if cfg.training.mode == "full" else "adapter",
    }


def validate_scheduler_instance(contract: dict[str, Any], scheduler: Any) -> None:
    actual = None if scheduler is None else f"{type(scheduler).__module__}.{type(scheduler).__qualname__}"
    if actual != contract["scheduler_class"]:
        raise ValueError("实际学习率调度器与训练配置不一致，不能保存或恢复训练状态")


def validate_scheduler_recipe(
    contract: dict[str, Any], cfg: TrainConfig, total_steps: int | None = None
) -> None:
    expected = scheduler_recipe(cfg, contract.get("total_steps") if total_steps is None else total_steps)
    if contract.get("config") != expected["config"]:
        raise ValueError("学习率调度配置与原训练不同，不能精确恢复；请恢复原设置或开始新任务")
    if total_steps is not None and contract.get("total_steps") != total_steps:
        raise ValueError("总训练步数与原训练不同，不能精确恢复学习率调度；请恢复原设置或开始新任务")
    if contract.get("training_kind") != expected["training_kind"]:
        raise ValueError(
            "checkpoint training mode differs: full-model weights and adapters are not interchangeable"
        )
    for key in ("version", "scheduler_class", "managed_by_optimizer"):
        if contract.get(key) != expected[key]:
            raise ValueError("训练模式或学习率管理方式与原训练不同，不能精确恢复")


@dataclass(frozen=True)
class ResumeSchedulerContract:
    checkpoint: Path
    metadata_sha256: str
    contract: dict[str, Any]


def read_resume_scheduler_contract(
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    captured: ResumeSchedulerContract | None = None,
) -> ResumeSchedulerContract:
    """Capture authenticated legacy config before same-directory preparation.

    Reuse only for the same checkpoint and unchanged metadata. Never authenticate a
    replacement config against the current job's hash instead of the saved hash.
    """
    path = Path(path).resolve()
    if metadata is None:
        metadata = json.loads((path / "state.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    if captured is not None:
        if captured.checkpoint != path or captured.metadata_sha256 != digest:
            raise ValueError("预检后训练状态已改变，请重新检查学习率调度配置")
        return captured
    progress = metadata.get("progress", {})
    extra = progress.get("extra", {})
    if "scheduler_contract" in extra:
        contract = copy.deepcopy(extra["scheduler_contract"])
    else:
        try:
            original = load_config(path.parent / "config.toml")
            if not metadata.get("config_hash") or config_hash(original) != metadata["config_hash"]:
                raise ValueError("原训练配置已修改或无法确认来源")
            contract = scheduler_recipe(original, progress["total_steps"])
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(
                "旧训练状态缺少学习率调度配置，无法验证精确恢复；"
                "请保留原训练目录中未修改的 config.toml，或从导出权重开始新任务。"
            ) from exc
    if (
        not isinstance(contract, dict)
        or contract.get("version") != 1
        or not isinstance(contract.get("config"), dict)
        or type(contract.get("total_steps")) is not int
        or contract["total_steps"] < 0
        or contract["total_steps"] != progress.get("total_steps")
        or contract.get("training_kind") != metadata.get("training_kind", "adapter")
        or "scheduler_class" not in contract
        or "managed_by_optimizer" not in contract
    ):
        raise ValueError("训练状态的学习率调度合同无效，不能精确恢复")
    return ResumeSchedulerContract(path, digest, contract)
