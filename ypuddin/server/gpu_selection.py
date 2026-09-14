"""Per-job accelerator requests, separate from immutable training configuration."""

from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator

DeviceId = Annotated[str, Field(pattern=r"^(cuda:[0-9]+|mps)$")]


class GpuSelection(BaseModel):
    gpu_devices: list[DeviceId] = Field(default_factory=list, max_length=64)

    @field_validator("gpu_devices")
    @classmethod
    def distinct_devices(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("同一张显卡不能重复选择")
        return value


def selection_error(devices: list[str], count: int, inventory: list[dict[str, Any]]) -> str | None:
    if not devices:
        return None
    if len(devices) != count:
        return f"此任务需要 {count} 张显卡，请选择相同数量的显卡，或使用自动分配。"
    available = {gpu["device"] for gpu in inventory}
    if missing := set(devices) - available:
        return f"所选显卡不可用：{', '.join(sorted(missing))}。请重新选择显卡。"
    return None
