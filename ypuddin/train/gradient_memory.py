"""Temporary tensors in the trainer's gradient norm and clipping path."""

from __future__ import annotations

import torch


def gradient_workspace_bytes(param_groups, *, clip_norm: float) -> int:
    parameters = [p for group in param_groups for p in group["params"] if p.requires_grad]
    if not parameters:
        return 0
    if clip_norm > 0:
        # Per-parameter norms, their stack and the clipping coefficient; gradients update in place.
        return 2 * sum(max(4, p.element_size()) for p in parameters) + 32
    # Norm logging visits one gradient at a time: optional FP32 cast plus its square.
    return max(p.numel() * (4 if p.dtype == torch.float32 else 8) + 4 for p in parameters)
