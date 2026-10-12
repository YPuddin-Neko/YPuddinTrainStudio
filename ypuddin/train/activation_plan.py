"""Constant-cost activation estimates from geometry and offline block measurements."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ypuddin.runtime_profiles import current_profile


@dataclass(frozen=True)
class BackboneMemoryLayout:
    family: str
    channels: int
    stride: int
    patch: int
    hidden: int
    blocks: int
    units: float
    checkpoint_units: float
    whole_units: tuple[tuple[str, float], ...]
    modes: tuple[str, ...]
    activation_element: int
    latent_element: int
    condition_element: int
    condition_width: int
    condition_layers: int
    math_attention: bool
    attention_groups: tuple[tuple[int, int, int], ...]
    auxiliary_bytes_at_512: int
    bridge_width: int
    bridge_active_blocks: int
    bridge_heads: int
    bridge_has_self_attention: bool
    cache_latents: bool
    device_type: str
    mask_inputs: bool
    anima_autocast: bool = False
    block_output_element: int = 0
    method: str = "family_geometry_offline_activation_units"


def _element(dtype):
    return {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2, torch.float64: 8}[dtype]


def _sdxl_attention_groups(backbone):
    groups = []

    def add(container, divisor):
        for attention in getattr(container, "attentions", ()):
            blocks = getattr(attention, "transformer_blocks", ())
            if blocks:
                groups.append((divisor, int(blocks[0].attn1.heads), len(blocks)))

    down = tuple(getattr(backbone, "down_blocks", ()))
    for index, block in enumerate(down):
        add(block, 2**index)
    add(getattr(backbone, "mid_block", None), 2**max(0, len(down) - 1))
    for index, block in enumerate(getattr(backbone, "up_blocks", ())):
        add(block, 2**max(0, len(down) - 1 - index))
    return tuple(groups)


def backbone_memory_layout(backbone, family, cfg, dtype, compute_dtype, device_type, layout=None):
    """Read existing metadata once, without copying or executing the backbone.

    dtype is the loaded/latent dtype; compute_dtype is the activation dtype.
    The result owns no modules, tensors, traces or autograd objects.
    """
    spec = family.spec
    if spec.retired_reason or spec.name not in {"anima", "krea2", "flux2", "sdxl", "toy"}:
        raise ValueError("unsupported memory-estimate family")
    config = getattr(backbone, "config", None)
    hidden = (getattr(backbone, "dim", None) or getattr(backbone, "model_channels", None)
              or getattr(backbone, "inner_dim", None) or getattr(config, "features", 2048))
    if layout is None and hasattr(family, "memory_layout_meta"):
        layout = family.memory_layout_meta(backbone)
    blocks = tuple(layout.blocks) if layout is not None else tuple(getattr(backbone, "blocks", ()))
    count = max(1, len(blocks))
    heads = (getattr(backbone, "num_heads", None) or getattr(config, "heads", None)
             or getattr(config, "num_attention_heads", None) or max(1, hidden // 128))
    units = spec.activation_units
    if spec.name == "flux2":
        double, single = len(backbone.transformer_blocks), len(backbone.single_transformer_blocks)
        count = max(1, double + single)
        # These are the two block measurements behind the existing 26.5 average.
        units = (30.8 * double + 25.1 * single) / count
    attention = getattr(cfg.model, "attention", "auto")
    device = device_type or "cuda"
    math_attention = (device == "cpu" or (device == "mps" and attention != "metal_flash")
                      or (current_profile() == "linux-dtk" and cfg.loop.deterministic)
                      or (device == "cuda" and compute_dtype == torch.float32 and attention in {"auto", "sdpa"}))
    groups = _sdxl_attention_groups(backbone) if spec.name == "sdxl" else ((1, int(heads), count),)
    condition_width = {"anima": 1024, "sdxl": 2048, "toy": 64}.get(spec.name, hidden)
    condition_layers, condition_dtype = 1, compute_dtype
    if spec.name == "anima":
        condition_dtype = torch.float32
    elif spec.name == "krea2":
        condition_width, condition_layers = config.txtdim, config.txtlayers
        condition_dtype = torch.bfloat16
    elif spec.name == "flux2":
        condition_width = config.joint_attention_dim
    bridge_blocks = tuple(getattr(getattr(backbone, "llm_adapter", None), "blocks", ()))
    bridge_width = bridge_heads = bridge_active = 0
    bridge_self = False
    active = bool(cfg.training.train_text_encoder)
    for block in bridge_blocks:
        active = active or any(parameter.requires_grad for parameter in block.parameters())
        bridge_active += active
    if bridge_blocks:
        cross = bridge_blocks[0].cross_attn
        bridge_width, bridge_heads = cross.query_dim, cross.n_heads
        condition_width = cross.context_dim
        bridge_self = bridge_blocks[0].has_self_attn
    auxiliary = family.auxiliary_activation_bytes_for_plan(backbone, 1, compute_dtype)
    return BackboneMemoryLayout(
        spec.name, spec.latent.channels, spec.latent.stride, spec.latent.patch, int(hidden), count,
        float(units), 22.0 if spec.name == "anima" else float(units),
        tuple(spec.backbone_activation_units), tuple(spec.checkpointing_modes),
        _element(compute_dtype), _element(dtype), _element(condition_dtype), int(condition_width),
        int(condition_layers), math_attention, groups, int(auxiliary), int(bridge_width),
        int(bridge_active), int(bridge_heads), bridge_self, cfg.dataset.cache_latents, device,
        cfg.dataset.masked_loss or cfg.dataset.image_fit == "pad",
        anima_autocast=(spec.name == "anima" and device == "cuda"
                        and getattr(cfg.loop, "mixed_precision", "no") != "no"),
        block_output_element=(4 if spec.name == "anima" and compute_dtype == torch.float16
                              else _element(compute_dtype)),
    )


def estimate_backbone_memory(layout, *, width, height, batch_size, checkpointing,
                             text_tokens=512, training=True):
    """Return training or inference workspace, plus separate input byte estimates.

    Activation bytes include checkpoint recomputation and auxiliary saved tensors.
    They exclude parameters, gradients and merged adapter-weight storage. Math
    attention adds score tensors separately from the linear fused coefficients.
    forward_workspace_bytes is the inference alternative, not added to training
    activation_bytes. Inputs include a small objective/mask tensor allowance.
    """
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (width, height, batch_size, text_tokens)):
        raise ValueError("memory geometry must contain positive integers")
    align = layout.stride * layout.patch
    if width % align or height % align:
        raise ValueError("memory geometry must match the latent alignment")
    if checkpointing not in layout.modes:
        raise ValueError("unsupported checkpointing mode")
    b, element = batch_size, layout.activation_element
    if layout.family == "anima":
        text_tokens = max(512, text_tokens)
    latent_pixels = (width // layout.stride) * (height // layout.stride)
    image_tokens = latent_pixels // (layout.patch**2)
    tokens = image_tokens
    if layout.family == "krea2":
        tokens = ((image_tokens + text_tokens + 255) // 256) * 256
    elif layout.family == "flux2":
        tokens += text_tokens
    whole = dict(layout.whole_units)
    unit = b * tokens * element * (1 if whole else layout.hidden)
    factors = whole or {
        "none": layout.units * layout.blocks,
        "block": min(layout.units * layout.blocks, layout.blocks + layout.checkpoint_units + 2),
        "unsloth": min(layout.units * layout.blocks, layout.checkpoint_units + 3),
    }
    activation = unit * factors[checkpointing]
    forward = unit * (whole.get("block", layout.units) if whole else layout.units)
    auxiliary = layout.auxiliary_bytes_at_512 * b * text_tokens / 512
    bridge = b * text_tokens * layout.bridge_width * element * layout.units
    auxiliary += bridge * layout.bridge_active_blocks
    activation += auxiliary
    # A frozen bridge executes before the main blocks; its transient peak is a
    # competing stage, not an extra full saved bridge graph on every main block.
    workspace = bridge_workspace = block_scores = all_scores = 0
    if layout.math_attention:
        largest_scores = 0
        for divisor, heads, count in layout.attention_groups:
            query = ((width // layout.stride // divisor) * (height // layout.stride // divisor)
                     if whole else tokens)
            self_score = b * heads * query**2 * 4
            cross_score = b * heads * query * text_tokens * 4 if layout.family in {"anima", "sdxl"} else 0
            all_scores += (self_score + cross_score) * count
            block_scores = max(block_scores, self_score + cross_score)
            largest_scores = max(largest_scores, self_score, cross_score)
        bridge_score = b * layout.bridge_heads * text_tokens**2 * 4
        activation += all_scores if checkpointing == "none" else 0
        bridge_saved = bridge_score * layout.bridge_active_blocks * (1 + layout.bridge_has_self_attention)
        activation += bridge_saved
        auxiliary += bridge_saved
        workspace, bridge_workspace = 2 * largest_scores, 2 * bridge_score
    if layout.family == "anima":
        image_hidden, text_hidden = b * image_tokens * layout.hidden, b * text_tokens * layout.hidden
        promoted = max(0, 4 - element)
        # Q/K RMSNorm retains FP32 inputs: self Q/K and cross Q/K.
        precision = (3 * image_hidden + text_hidden) * promoted
        if promoted and (layout.anima_autocast or layout.block_output_element > element):
            # BF16 AdaLN retains three FP32 norm inputs beside the residuals
            # and a promoted normalized_x. FP16 uses the same local allowance
            # for its FP32 residual/gate path; inter-block storage is separate.
            precision += image_hidden * (3 * 4 + promoted)
            forward += image_hidden * promoted
        residual_element = layout.block_output_element or element
        forward += image_hidden * max(0, residual_element - element)
        saved_inputs = image_hidden * (element + max(0, layout.blocks - 1) * residual_element)
        # Math SDPA also retains FP32 Q/K/V; fused attention keeps low precision.
        math_casts = (4 * image_hidden + 2 * text_hidden) * promoted if layout.math_attention else 0
        none = unit * layout.units * layout.blocks + (precision + math_casts) * layout.blocks
        if checkpointing == "none":
            activation = none + all_scores + workspace + auxiliary
        else:
            shared = (saved_inputs if checkpointing == "block" else image_hidden * residual_element) + 2 * unit
            # Attention's transient scores and MLP backward peak at different
            # times. The latter still holds one score per attention operation.
            attention_peak = min(none, shared + unit * layout.checkpoint_units + precision) + workspace
            mlp_peak = min(none, shared + unit * layout.units + precision + math_casts) + block_scores
            activation = max(attention_peak, mlp_peak) + auxiliary
        workspace_for_training = 0
    else:
        workspace_for_training = workspace
    latent = b * layout.channels * latent_pixels * layout.latent_element
    condition = b * text_tokens * layout.condition_width * layout.condition_layers * layout.condition_element
    geometry = b * 62  # collated sizes/crop, weights and flags
    # FP32 x0/noise/noised latent/target/loss intermediates; model input casts
    # are small beside the backbone and kept separate from its block units.
    inputs = b * layout.channels * latent_pixels * 4 * 5 + condition + geometry
    if layout.latent_element != 4:
        inputs += latent
    mask = b * width * height * 4 if layout.mask_inputs else 0
    if mask:
        inputs += mask + b * latent_pixels * 4 * (1 + layout.channels)
    host = 0
    if layout.device_type != "cpu":
        host = geometry + mask + (latent if layout.cache_latents else b * 3 * width * height * 4)
    elif not layout.cache_latents:
        inputs += b * 3 * width * height * 4
    if training and checkpointing == "unsloth":
        host += (saved_inputs if layout.family == "anima"
                 else b * tokens * layout.hidden * element * layout.blocks)
    return {
        "activation_bytes": math.ceil(max(activation + workspace_for_training, bridge + bridge_workspace)) if training else 0,
        "forward_workspace_bytes": math.ceil(max(forward + workspace, bridge + bridge_workspace)),
        "auxiliary_activation_bytes": math.ceil(auxiliary) if training else 0,
        "input_bytes": inputs, "host_input_bytes": host, "method": layout.method,
        "attention_basis": "math_score_tensors" if layout.math_attention else "fused_linear_storage",
        "includes_parameter_gradients": False, "includes_adapter_weight_fusion": False,
    }
