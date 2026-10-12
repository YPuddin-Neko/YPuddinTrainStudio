"""Allocator defaults for fresh, single-GPU image training workers."""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

from ypuddin.runtime_profiles import current_profile

if TYPE_CHECKING:
    from ypuddin.config import TrainConfig

log = logging.getLogger(__name__)

_ALLOCATOR_ENV = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF")


def configure_native_allocator(cfg: TrainConfig, *, device: str | None = None) -> bool:
    """Configure variable-sized native allocations before the worker loads its trainer."""
    if (
        sys.platform != "linux"
        or cfg.dataset.resolution_mode != "native"
        or cfg.loop.gpu_count != 1
        or os.environ.get("WORLD_SIZE", "1") != "1"
        # Even an empty value is an explicit allocator choice in PyTorch.
        or any(name in os.environ for name in _ALLOCATOR_ENV)
    ):
        return False
    if device is None:
        if current_profile().endswith("-cpu"):
            return False
    elif device.split(":", 1)[0] != "cuda":
        return False

    import torch

    if (
        not torch.version.cuda
        or torch.version.hip
        or torch.cuda.is_initialized()
        or not torch.cuda.is_available()
        or torch.cuda.memory.get_allocator_backend() != "native"
    ):
        return False
    # Torch can read allocator environment settings during import. Its runtime
    # setter updates the allocator directly, before any model or CUDA tensor loads.
    setter = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
    if setter is None:
        setter = getattr(torch.cuda.memory, "_set_allocator_settings", None)
    if setter is None:
        return False
    try:
        setter("expandable_segments:True")
    except RuntimeError as exc:
        log.warning("CUDA allocator could not enable expandable segments: %s", exc)
        return False
    log.debug("CUDA allocator configured with expandable_segments=True for native-resolution training")
    return True
