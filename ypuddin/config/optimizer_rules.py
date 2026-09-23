"""Optimizer controls shared by config validation and the schema-driven editor.

Studio enforces the learning-rate locks through config validation, independently
of the upstream optimizer constructors. D Coef remains adjustable.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

PRODIGY_FIELDS = (
    "d_coef",
    "d0",
    "beta3",
    "use_bias_correction",
    "safeguard_warmup",
    "growth_rate",
    "slice_p",
    "decouple",
)
PPSF_FIELDS = (
    "d_coef",
    "d0",
    "beta3",
    "use_bias_correction",
    "d_limiter",
    "prodigy_steps",
    "schedulefree_c",
    "split_groups",
    "split_groups_mean",
    "factored",
    "factored_fp32",
    "use_stableadamw",
    "stochastic_rounding",
    "weight_decay_by_lr",
    "use_schedulefree",
    "use_speed",
    "use_cautious",
    "use_grams",
    "use_adopt",
    "use_orthograd",
    "use_focus",
)
AUTOMAGIC_FIELDS = ("beta2", "min_lr", "max_lr", "lr_bump", "clip_threshold")


def optimizer_key(name: str) -> str:
    aliases = {
        "prodigyopt.prodigy": "prodigy",
        "prodigyopt.prodigy.prodigy": "prodigy",
        "prodigyplus.prodigyplusschedulefree": "prodigy_plus_sf",
        "prodigyplus.prodigy_plus_schedulefree.prodigyplusschedulefree": "prodigy_plus_sf",
        "ypuddin.optim.automagic.automagic": "automagic",
        "schedulefree.adamwschedulefree": "adamw_sf",
        "schedulefree.adamw_schedulefree.adamwschedulefree": "adamw_sf",
    }
    return aliases.get(name.lower(), name.lower())


def optimizer_specific_fields(name: str) -> tuple[str, ...]:
    return {
        "prodigy": PRODIGY_FIELDS,
        "prodigy_plus_sf": PPSF_FIELDS,
        "automagic": AUTOMAGIC_FIELDS,
    }.get(optimizer_key(name), ())


def _reason(zh: str, en: str) -> dict[str, str]:
    return {"zh": zh, "en": en}


_ADAPTIVE_REASON = _reason(
    "由优化器估计有效步长，基础倍率固定为 1；用 D Coef 调整训练强度。",
    "The optimizer estimates the effective step size. Studio fixes the base multiplier at 1; adjust D Coef to change its scale.",
)
_GROUP_REASON = _reason(
    "当前优化器统一管理步长，不叠加手动分组学习率或倍率。",
    "This optimizer manages step sizes without manual group learning rates or multipliers.",
)
_SF_REASON = _reason(
    "已启用免调度权重平均，外部学习率保持恒定且不预热。",
    "Schedule-Free averaging manages training internally; external learning-rate scheduling and warmup are disabled.",
)
_KAHAN_REASON = _reason(
    "当前优化器已有自己的状态与精度处理，不支持叠加此补偿。",
    "This optimizer manages its own states and precision; the additional Kahan wrapper is unsupported.",
)


def _adaptive(lr: float = 1.0) -> dict[str, Any]:
    return {
        "fixed": {"optimizer.lr": lr, "optimizer.group_lr": {}, "adapter.lr_scale": {}},
        "reason": {
            "optimizer.lr": _ADAPTIVE_REASON,
            "optimizer.group_lr": _GROUP_REASON,
            "adapter.lr_scale": _GROUP_REASON,
            "adapter.rules[].lr": _GROUP_REASON,
        },
        "defaults": {},
        "conditional": [],
    }


def _schedule_free() -> dict[str, Any]:
    fixed: dict[str, Any] = {
        "scheduler.type": "constant",
        "scheduler.warmup_steps": 0,
        "scheduler.min_lr_ratio": 0,
        "optimizer.kahan": False,
    }
    return {
        "fixed": fixed,
        "reason": {k: _KAHAN_REASON if k == "optimizer.kahan" else _SF_REASON for k in fixed},
    }


def optimizer_capabilities() -> dict[str, Any]:
    prodigy = _adaptive()
    prodigy["aliases"] = ["prodigyopt.Prodigy", "prodigyopt.prodigy.Prodigy"]
    prodigy["typed_fields"] = list(PRODIGY_FIELDS)
    prodigy["defaults"] = {"optimizer.betas": [0.9, 0.999], "optimizer.eps": 1e-8}
    ppsf = _adaptive()
    ppsf["aliases"] = [
        "prodigyplus.ProdigyPlusScheduleFree",
        "prodigyplus.prodigy_plus_schedulefree.ProdigyPlusScheduleFree",
    ]
    ppsf["typed_fields"] = list(PPSF_FIELDS)
    ppsf["defaults"] = {"optimizer.betas": [0.9, 0.99], "optimizer.eps": 1e-8, "optimizer.grad_clip_norm": 0}
    ppsf["conditional"] = [{"when": "optimizer.use_schedulefree == true", **_schedule_free()}]
    # Kahan is also unsupported for PPSF's non-SF mode: its state/rounding logic
    # still owns the low-precision master parameters.
    ppsf["fixed"]["optimizer.kahan"] = False
    ppsf["reason"]["optimizer.kahan"] = _KAHAN_REASON
    auto = _adaptive(1e-6)
    auto["aliases"] = ["ypuddin.optim.automagic.Automagic"]
    auto["typed_fields"] = list(AUTOMAGIC_FIELDS)
    auto["fixed"].update(_schedule_free()["fixed"])
    auto_reason = _reason(
        "Automagic 自行调整每个参数的学习率，从 0.000001 开始；可调整下方步长范围与增量。",
        "Automagic adapts per-parameter learning rates from 0.000001. Adjust its bounds and increment below.",
    )
    auto["reason"]["optimizer.lr"] = auto_reason
    auto["reason"].update(
        {k: _KAHAN_REASON if k == "optimizer.kahan" else auto_reason for k in _schedule_free()["fixed"]}
    )
    auto["defaults"] = {"optimizer.eps": 1e-30, "optimizer.grad_clip_norm": 0}
    return {
        "prodigy": prodigy,
        "prodigy_plus_sf": ppsf,
        "automagic": auto,
        "adamw_sf": {
            **_schedule_free(),
            "defaults": {},
            "conditional": [],
            "aliases": [
                "schedulefree.AdamWScheduleFree",
                "schedulefree.adamw_schedulefree.AdamWScheduleFree",
            ],
        },
    }


def optimizer_policy(name: str, *, use_schedulefree: bool = True) -> dict[str, Any]:
    policy = deepcopy(optimizer_capabilities().get(optimizer_key(name), {}))
    if optimizer_key(name) == "prodigy_plus_sf" and use_schedulefree:
        conditional = policy["conditional"][0]
        policy["fixed"].update(conditional["fixed"])
        policy["reason"].update(conditional["reason"])
    return policy


def canonical_optimizer_fragment(raw: dict[str, Any], validated: dict[str, Any]) -> dict[str, Any]:
    """Persist effective optimizer controls while preserving a partial preset.

    Reading an old preset does not rewrite it. An explicit save canonicalizes
    only provided optimizer controls, migrated args and managed schedule values.
    """
    if not isinstance(raw.get("optimizer"), dict):
        return raw
    result = deepcopy(raw)
    original = raw["optimizer"]
    effective = validated["optimizer"]
    optimizer = result["optimizer"]
    for name in original:
        optimizer[name] = deepcopy(effective[name])
    for name in original.get("args", {}):
        if name not in effective["args"] and name in effective:
            optimizer[name] = deepcopy(effective[name])
    policy = optimizer_policy(effective["type"], use_schedulefree=effective["use_schedulefree"])
    for path, value in policy.get("fixed", {}).items():
        group, field = path.split(".")
        if group in {"optimizer", "scheduler"}:
            result.setdefault(group, {})[field] = deepcopy(value)
    return result
