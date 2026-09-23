"""Versioned Apple attention identity without changing existing SDPA checkpoints."""

from __future__ import annotations

from typing import Any


def resolve_metal_attention_runtime(attention: str, device_type: str) -> dict[str, str] | None:
    if attention != "metal_flash":
        return None
    if device_type != "mps":
        raise ValueError("Metal FlashAttention 仅适用于 Apple MPS；其他设备请选择 SDPA 或对应平台后端")
    from ypuddin.models.metal_attention import require_metal_flash

    return require_metal_flash(device_type)


def validate_metal_attention_resume(current: dict[str, str] | None, saved: Any) -> None:
    if current != saved:
        raise ValueError(
            "Metal FlashAttention 与检查点的注意力实现或运行环境不同，不能严格续训；"
            "请保持原后端及版本，或从导出的适配器开始新训练"
        )
