"""Versioned native-dtype boundaries for DTK text LoRA training.

Cover the frozen backbone's VJP, all online text encoders, and LoRA's direct
contractions. FP32 inputs outside BF16 autocast and no-grad LoRA/Linear evaluation
retain their original implementation. The BF16 route retains each original
parameter cast and intermediate rounding; it is not whole-model FP32 training.
"""

from __future__ import annotations

import types

import torch

from ypuddin.adapters.frozen import FrozenLinear
from ypuddin.adapters.linear import AdaptedLinear
from ypuddin.adapters.lora import LoRA
from ypuddin.config.compute_policy import (
    BF16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID,
    BF16_LORA_FP32_IMPLEMENTATION_ID,
)

from .conv_forward import install_conv_fp32_forward, validate_conv_forward_installation
from .linear_backward import (
    _BF16OperandsFP32Compute,
    _uses_bf16,
    install_linear_bf16_forward_fp32_backward,
    install_linear_bf16_operands_fp32_compute,
    validate_linear_backward_installation,
)

_MARKER = "_ypuddin_text_lora_compute"


def _lora_layers(module):
    layers = []
    for child in module.modules():
        if isinstance(child, FrozenLinear) and child.is_fp8:
            raise ValueError("文本 LoRA 可复现策略不支持 FP8 冻结权重")
        if not isinstance(child, AdaptedLinear):
            continue
        if (
            type(child.adapter) is not LoRA
            or child.mode != "bypass"
            or child.dora is not None
            or any(p.dtype != torch.float32 for p in child.adapter.parameters())
        ):
            raise ValueError("文本可复现策略需要实际 LoRA bypass、FP32 适配器参数且不使用 DoRA")
        layers.append(child.adapter)
    return layers


def _install_lora_contractions(module):
    layers = _lora_layers(module)
    if any(_MARKER in layer.__dict__ for layer in layers):
        raise ValueError("文本 LoRA 算子策略不能重复安装")
    records = []

    def bind(original):
        def delta_apply(adapter, x):
            if not torch.is_grad_enabled():
                return original(x)
            # LoRA.delta_apply explicitly casts both parameters to x.dtype,
            # even outside autocast. Retain those CastBackward boundaries.
            dt = x.dtype
            down = adapter.down.to(dt)
            if not _uses_bf16(x, down):
                return original(x)
            h = _BF16OperandsFP32Compute.apply(x, down, None, True)
            mask = adapter._rank_mask(adapter.rank, x.device, dt)
            if mask is not None:
                h = h * mask
            y = _BF16OperandsFP32Compute.apply(h, adapter.up.to(dt), None, True)
            gain = adapter._gain()
            if isinstance(gain, torch.Tensor):
                gain = gain.to(dt)
            return adapter._output_dropout(y * gain)

        return delta_apply

    for layer in layers:
        existed = "delta_apply" in layer.__dict__
        previous = layer.__dict__.get("delta_apply")
        replacement = types.MethodType(bind(layer.delta_apply), layer)
        layer.delta_apply = replacement
        layer.__dict__[_MARKER] = (BF16_LORA_FP32_IMPLEMENTATION_ID, replacement)
        records.append((layer, replacement, existed, previous))

    def restore():
        for layer, replacement, existed, previous in reversed(records):
            if layer.__dict__.get("delta_apply") is replacement:
                if existed:
                    layer.delta_apply = previous
                else:
                    layer.__dict__.pop("delta_apply", None)
            if layer.__dict__.get(_MARKER) == (BF16_LORA_FP32_IMPLEMENTATION_ID, replacement):
                layer.__dict__.pop(_MARKER)

    return restore, len(layers)


def install_text_adapter_compute(modules, policy):
    if sorted(modules) != policy["operator_components"]:
        raise ValueError("文本计算策略的实际模型组件不一致")
    # Validate all real adapter types before modifying any instance method.
    for name, module in modules.items():
        layers = _lora_layers(module)
        if bool(layers) != (name in policy["trainable_components"]):
            raise ValueError("文本计算策略的实际可训练组件不一致")
    restores, counts = [], {}
    try:
        for name, module in modules.items():
            implementation = policy[
                "linear_backward_implementation"
                if name == "backbone"
                else "text_linear_backward_implementation"
            ]
            install = (
                install_linear_bf16_operands_fp32_compute
                if implementation == BF16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID
                else install_linear_bf16_forward_fp32_backward
            )
            undo_linear, linear = install(module)
            restores.append(undo_linear)
            undo_lora, lora = _install_lora_contractions(module)
            restores.append(undo_lora)
            counts[name] = {"linear": linear, "lora": lora}
            if name == "backbone" and "conv_implementation" in policy:
                undo_conv, conv = install_conv_fp32_forward(module)
                restores.append(undo_conv)
                counts[name]["conv"] = conv
        validate_text_adapter_compute(modules, counts, policy)
    except BaseException:
        for restore in reversed(restores):
            restore()
        raise

    def restore():
        for undo in reversed(restores):
            undo()

    return restore, counts


def validate_text_adapter_compute(modules, expected_counts, policy):
    if sorted(modules) != policy["operator_components"] or set(expected_counts) != set(modules):
        raise ValueError("文本计算策略的组件或覆盖记录不完整")
    for name, module in modules.items():
        counts = expected_counts[name]
        implementation = policy[
            "linear_backward_implementation" if name == "backbone" else "text_linear_backward_implementation"
        ]
        validate_linear_backward_installation(
            module, counts["linear"], expected_implementation=implementation
        )
        layers = _lora_layers(module)
        if len(layers) != counts["lora"] or bool(layers) != (name in policy["trainable_components"]):
            raise ValueError("文本 LoRA 策略未完整覆盖实际适配器")
        for layer in layers:
            marker = layer.__dict__.get(_MARKER)
            if (
                marker is None
                or marker[0] != policy["adapter_implementation"]
                or layer.__dict__.get("delta_apply") is not marker[1]
            ):
                raise ValueError("文本 LoRA 收缩策略缺失或被替换")
        if name == "backbone" and "conv_implementation" in policy:
            validate_conv_forward_installation(module, counts.get("conv"))
    return expected_counts
