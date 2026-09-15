"""FP32 Conv2d computation with the original forward output dtype boundary."""

from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as F
from torch import nn

from ypuddin.config.compute_policy import FP32_CONV_IMPLEMENTATION_ID

_MARKER = "_ypuddin_fp32_conv_forward"


def _forward(module, input):
    device_type = input.device.type
    output_dtype = (
        torch.get_autocast_dtype(device_type) if torch.is_autocast_enabled(device_type) else input.dtype
    )
    with torch.autocast(device_type, enabled=False):
        x = input.float()
        if module.padding_mode != "zeros":
            x = F.pad(x, module._reversed_padding_repeated_twice, mode=module.padding_mode)
            padding = (0, 0)
        else:
            padding = module.padding
        result = F.conv2d(
            x,
            module.weight.float(),
            module.bias.float() if module.bias is not None else None,
            module.stride,
            padding,
            module.dilation,
            module.groups,
        )
    return result.to(output_dtype)


def install_conv_fp32_forward(backbone):
    """Install the verified ordinary Conv2d strategy without replacing parameters."""
    modules = [module for module in backbone.modules() if isinstance(module, nn.Conv2d)]
    if not modules or any(type(module) is not nn.Conv2d for module in modules):
        raise ValueError("FP32 convolution policy requires ordinary nn.Conv2d modules")
    if any(_MARKER in module.__dict__ for module in modules):
        raise ValueError("FP32 convolution policy is already installed")
    records = []
    for module in modules:
        existed = "forward" in module.__dict__
        previous = module.__dict__.get("forward")
        replacement = MethodType(_forward, module)
        module.forward = replacement
        module.__dict__[_MARKER] = (FP32_CONV_IMPLEMENTATION_ID, replacement)
        records.append((module, replacement, existed, previous))

    def restore():
        for module, replacement, existed, previous in records:
            if module.__dict__.get("forward") is replacement:
                if existed:
                    module.forward = previous
                else:
                    module.__dict__.pop("forward", None)
            marker = module.__dict__.get(_MARKER)
            if marker is not None and marker[1] is replacement:
                module.__dict__.pop(_MARKER)

    return restore, {"conv2d": len(modules)}


def validate_conv_forward_installation(backbone, expected_counts):
    count = 0
    for module in backbone.modules():
        if not isinstance(module, nn.Conv2d):
            continue
        marker = module.__dict__.get(_MARKER)
        ordinary_conv = type(module) is nn.Conv2d
        if not ordinary_conv:
            from torch.distributed.fsdp import FSDPModule

            # FSDP2 dynamically wraps the original class after installation.
            # Accept only that wrapper around an ordinary Conv2d, not an
            # arbitrary Conv2d subclass with different forward semantics.
            ordinary_conv = type(module).__bases__ == (FSDPModule, nn.Conv2d)
        if (
            not ordinary_conv
            or marker is None
            or marker[0] != FP32_CONV_IMPLEMENTATION_ID
            or module.__dict__.get("forward") is not marker[1]
        ):
            raise ValueError("FP32 convolution policy is missing or its forward was replaced")
        count += 1
    if not count or expected_counts != {"conv2d": count}:
        raise ValueError("FP32 convolution installation does not match the selected backbone")
    return expected_counts
