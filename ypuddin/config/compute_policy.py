"""Resolve versioned training compute settings without importing a tensor runtime."""

from __future__ import annotations

from typing import Literal, TypedDict

from .schema import TrainConfig

DTK_FULL_FP32_MATH_POLICY_ID = "dtk-full-fp32-math-v1"
DTK_ANIMA_BF16_LINEAR_COMPUTE_POLICY_ID = "dtk-anima-bf16-linear-fp32-compute-v1"
DTK_ANIMA_DDP_BF16_LINEAR_COMPUTE_POLICY_ID = "dtk-anima-ddp-bf16-linear-fp32-compute-v1"
DTK_ANIMA_FSDP_BF16_LINEAR_COMPUTE_POLICY_ID = "dtk-anima-fsdp-bf16-linear-fp32-compute-v1"
BF16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID = "linear-bf16-operands-fp32-compute-v1"
DTK_KREA2_FSDP_BF16_LINEAR_POLICY_ID = "dtk-krea2-fsdp-bf16-linear-fp32-backward-v1"
DTK_SDXL_BF16_CONV_LINEAR_POLICY_ID = "dtk-sdxl-bf16-conv-fp32-linear-backward-v1"
DTK_SDXL_FSDP_BF16_CONV_LINEAR_POLICY_ID = "dtk-sdxl-fsdp-bf16-conv-fp32-linear-backward-v1"
DTK_SDXL_LONG_TEXT_POLICY_ID = "dtk-sdxl-long-text-bf16-conv-fp32-linear-compute-v1"
BF16_LINEAR_BACKWARD_IMPLEMENTATION_ID = "linear-bf16-forward-fp32-backward-v1"
FP32_CONV_IMPLEMENTATION_ID = "conv2d-fp32-output-bf16-v1"
KLEIN9B_TEXT_COMPUTE_ID = "qwen3-bf16-linear-fp32-v1"
BF16_LORA_FP32_IMPLEMENTATION_ID = "lora-bf16-operands-fp32-contractions-v1"
DTK_TEXT_LORA_POLICY_IDS = {
    family: f"dtk-{family}-text-lora-bf16-fp32-contractions-v1" for family in ("anima", "sdxl", "krea2")
}
DTK_SDXL_TEXT_LORA_PREVIEW_POLICY_IDS = {
    strategy: f"dtk-sdxl-text-lora-{strategy}-bf16-compute-preview-v2" for strategy in ("single", "ddp")
}
DTK_TEXT_LORA_ALL_POLICY_IDS = frozenset(DTK_TEXT_LORA_POLICY_IDS.values()) | frozenset(
    DTK_SDXL_TEXT_LORA_PREVIEW_POLICY_IDS.values()
)
DTK_BACKBONE_ADAPTER_POLICY_IDS = {
    (family, algo, strategy): f"dtk-{family}-backbone-{algo}-{strategy}-bf16-compute-v1"
    for family in ("anima", "sdxl")
    for algo, strategies in (("lora", ("single", "ddp", "fsdp")), ("lokr", ("fsdp",)))
    for strategy in strategies
}
BF16_LINEAR_PREVIEW_IMPLEMENTATION_ID = "linear-native-dispatch-bf16-operands-fp32-preview-v1"
FP16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID = "linear-lora-fp16-operands-fp32-compute-v1"
DTK_ANIMA_LORA_SINGLE_FP16_POLICY_ID = "dtk-anima-backbone-lora-single-fp16-compute-v1"
FP16_LINEAR_PREVIEW_IMPLEMENTATION_ID = "linear-native-dispatch-fp16-operands-fp32-preview-v1"
DTK_SDXL_LOKR_SINGLE_FP16_PREVIEW_POLICY_ID = "dtk-sdxl-backbone-lokr-single-fp16-preview-v1"
DTK_SDXL_LONG_TEXT_PREVIEW_POLICY_IDS = {
    (algo, strategy): f"dtk-sdxl-backbone-{algo}-{strategy}-bf16-compute-preview-v2"
    for algo in ("lora", "lokr")
    for strategy in ("single", "ddp", "fsdp")
}
DTK_ANIMA_LORA_FSDP_PREVIEW_POLICY_ID = "dtk-anima-backbone-lora-fsdp-bf16-compute-preview-v2"
DTK_ANIMA_LOKR_FSDP_PREVIEW_POLICY_ID = "dtk-anima-backbone-lokr-fsdp-bf16-compute-preview-v2"
DTK_ANIMA_FSDP_PREVIEW_POLICY_IDS = {
    "lora": DTK_ANIMA_LORA_FSDP_PREVIEW_POLICY_ID,
    "lokr": DTK_ANIMA_LOKR_FSDP_PREVIEW_POLICY_ID,
}
DTK_KLEIN4B_ADAPTER_POLICY_IDS = {
    (
        algo,
        strategy,
    ): f"dtk-klein4b-backbone-{algo}-{strategy}-bf16-compute-preview-v{2 if algo == 'lokr' and strategy == 'fsdp' else 1}"
    for algo in ("lora", "lokr")
    for strategy in ("ddp", "fsdp")
}
DTK_KLEIN9B_ADAPTER_POLICY_IDS = {
    (algo, strategy): (f"dtk-klein9b-backbone-{algo}-{strategy}-bf16-compute-preview-v4")
    for algo in ("lora", "lokr")
    for strategy in ("ddp", "fsdp")
}
DTK_BACKBONE_ADAPTER_ALL_POLICY_IDS = (
    frozenset(DTK_BACKBONE_ADAPTER_POLICY_IDS.values())
    | frozenset(DTK_SDXL_LONG_TEXT_PREVIEW_POLICY_IDS.values())
    | frozenset(DTK_ANIMA_FSDP_PREVIEW_POLICY_IDS.values())
    | frozenset(DTK_KLEIN9B_ADAPTER_POLICY_IDS.values())
    | frozenset(
        value
        for (algo, _strategy), value in DTK_KLEIN4B_ADAPTER_POLICY_IDS.items()
        if algo == "lora" or _strategy == "fsdp"
    )
)


class _RequiredTrainingComputePolicy(TypedDict):
    id: str
    mixed_precision: Literal["no", "bf16", "fp16"]
    allow_tf32: Literal[False]
    attention: Literal["sdpa"]
    sdpa_backend: Literal["math"]


class TrainingComputePolicy(_RequiredTrainingComputePolicy, total=False):
    linear_forward: Literal[
        "native-bf16",
        "bf16-rounded-operands-fp32-contraction-bf16-output",
        "fp16-rounded-operands-fp32-contraction-fp16-output",
    ]
    linear_backward: Literal["fp32-contractions-grad-original-dtype"]
    linear_backward_implementation: str
    fsdp_param_dtype: Literal["bfloat16"]
    fsdp_reduce_dtype: Literal["float32"]
    conv_forward: Literal["fp32-output-bf16"]
    conv_implementation: str
    text_linear_forward: Literal["bf16-rounded-operands-fp32-contraction-bf16-output"]
    text_linear_backward_implementation: str
    adapter_forward: Literal[
        "bf16-rounded-operands-fp32-contractions-bf16-intermediates",
        "fp16-rounded-operands-fp32-contractions-fp16-intermediates",
    ]
    adapter_backward: Literal["fp32-contractions-grad-original-dtype"]
    adapter_implementation: str
    trainable_components: list[str]
    operator_components: list[str]
    distributed_strategy: Literal["single", "ddp", "fsdp"]
    adapter_algorithm: Literal["lora", "lokr"]
    sdxl_max_token_length: Literal[75, 150, 225]
    preview_operator_components: list[str]
    preview_linear_forward: Literal[
        "bf16-rounded-operands-fp32-contraction-bf16-output",
        "fp16-rounded-operands-fp32-contraction-fp16-output",
    ]
    preview_linear_implementation: str


def _text_lora_policy(cfg, device_type, profile):
    if not (
        profile == "linux-dtk"
        and device_type == "cuda"
        and cfg.model.family in DTK_TEXT_LORA_POLICY_IDS
        and cfg.loop.deterministic
        and cfg.loop.mixed_precision == "bf16"
        and cfg.loop.distributed_strategy == "ddp"
        and cfg.training.mode == "adapter"
        and cfg.training.train_text_encoder
        and cfg.adapter.algo == "lora"
        and cfg.adapter.mode in {"auto", "bypass"}
        and cfg.adapter.param_dtype == "fp32"
        and not cfg.adapter.dora
        and all(rule.algo in {None, "lora", "none"} for rule in cfg.adapter.rules)
        and not cfg.memory.base_precision.startswith("fp8")
        and cfg.memory.activation_checkpointing in {"none", "block"}
        and not cfg.memory.compile
        and cfg.memory.blocks_to_swap == 0
        and cfg.dataset.text_encoding in {"auto", "online"}
        and not cfg.memory.offload_text_encoder
    ):
        return None
    text = ["text_encoder"] + (["text_encoder_2"] if cfg.model.family == "sdxl" else [])
    policy: TrainingComputePolicy = {
        "id": DTK_TEXT_LORA_POLICY_IDS[cfg.model.family],
        "mixed_precision": "bf16",
        "allow_tf32": False,
        "attention": "sdpa",
        "sdpa_backend": "math",
        "linear_forward": "bf16-rounded-operands-fp32-contraction-bf16-output",
        "linear_backward": "fp32-contractions-grad-original-dtype",
        "linear_backward_implementation": BF16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID,
        "text_linear_forward": "bf16-rounded-operands-fp32-contraction-bf16-output",
        "text_linear_backward_implementation": BF16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID,
        "adapter_forward": "bf16-rounded-operands-fp32-contractions-bf16-intermediates",
        "adapter_backward": "fp32-contractions-grad-original-dtype",
        "adapter_implementation": BF16_LORA_FP32_IMPLEMENTATION_ID,
        "trainable_components": sorted(text + (["backbone"] if cfg.training.train_backbone else [])),
        "operator_components": sorted(["backbone", *text]),
        "distributed_strategy": "ddp" if cfg.loop.gpu_count > 1 else "single",
    }
    if cfg.model.family == "sdxl":
        policy.update(
            conv_forward="fp32-output-bf16",
            conv_implementation=FP32_CONV_IMPLEMENTATION_ID,
            sdxl_max_token_length=cfg.model.sdxl_max_token_length,
        )
    if (
        cfg.model.family == "sdxl"
        and cfg.model.sdxl_max_token_length == 150
        and cfg.training.train_backbone
        and cfg.loop.gpu_count in (1, 2)
    ):
        policy.update(
            id=DTK_SDXL_TEXT_LORA_PREVIEW_POLICY_IDS[policy["distributed_strategy"]],
            preview_operator_components=["backbone"],
            preview_linear_forward="bf16-rounded-operands-fp32-contraction-bf16-output",
            preview_linear_implementation=BF16_LINEAR_PREVIEW_IMPLEMENTATION_ID,
        )
    return policy


def _backbone_adapter_policy(cfg, device_type, profile):
    strategy = cfg.loop.distributed_strategy if cfg.loop.gpu_count > 1 else "single"
    key = (cfg.model.family, cfg.adapter.algo, strategy)
    if not (
        profile == "linux-dtk"
        and device_type == "cuda"
        and key in DTK_BACKBONE_ADAPTER_POLICY_IDS
        and cfg.loop.deterministic
        and cfg.loop.mixed_precision == "bf16"
        and cfg.training.mode == "adapter"
        and cfg.training.train_backbone
        and not cfg.training.train_text_encoder
        and cfg.adapter.mode in {"auto", "bypass"}
        and cfg.adapter.param_dtype == "fp32"
        and not cfg.adapter.dora
        and all(rule.algo in {None, cfg.adapter.algo, "none"} for rule in cfg.adapter.rules)
        and not cfg.memory.base_precision.startswith("fp8")
        and cfg.memory.activation_checkpointing in {"none", "block"}
        and not cfg.memory.compile
        and cfg.memory.blocks_to_swap == 0
    ):
        return None
    # LoKr keeps its native contractions. Its FSDP backbone uses the same
    # operators as the existing family recipe, with a separate resume identity.
    fp32_forward = (
        cfg.adapter.algo == "lora"
        or cfg.model.family == "anima"
        or (cfg.model.family == "sdxl" and cfg.model.sdxl_max_token_length > 75)
    )
    policy: TrainingComputePolicy = {
        "id": DTK_BACKBONE_ADAPTER_POLICY_IDS[key],
        "mixed_precision": "bf16",
        "allow_tf32": False,
        "attention": "sdpa",
        "sdpa_backend": "math",
        "linear_forward": "bf16-rounded-operands-fp32-contraction-bf16-output"
        if fp32_forward
        else "native-bf16",
        "linear_backward": "fp32-contractions-grad-original-dtype",
        "linear_backward_implementation": BF16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID
        if fp32_forward
        else BF16_LINEAR_BACKWARD_IMPLEMENTATION_ID,
        "adapter_algorithm": cfg.adapter.algo,
        "trainable_components": ["backbone"],
        "operator_components": ["backbone"],
        "distributed_strategy": strategy,
    }
    if cfg.adapter.algo == "lora":
        policy.update(
            adapter_forward="bf16-rounded-operands-fp32-contractions-bf16-intermediates",
            adapter_backward="fp32-contractions-grad-original-dtype",
            adapter_implementation=BF16_LORA_FP32_IMPLEMENTATION_ID,
        )
    if strategy == "fsdp":
        policy.update(fsdp_param_dtype="bfloat16", fsdp_reduce_dtype="float32")
    if cfg.model.family == "sdxl":
        policy.update(
            conv_forward="fp32-output-bf16",
            conv_implementation=FP32_CONV_IMPLEMENTATION_ID,
            sdxl_max_token_length=cfg.model.sdxl_max_token_length,
        )
    return policy


def _klein_adapter_policy(cfg, device_type, profile):
    key = (cfg.adapter.algo, cfg.loop.distributed_strategy)
    policy_ids = {
        "klein-base-4b": DTK_KLEIN4B_ADAPTER_POLICY_IDS,
        "klein-base-9b": DTK_KLEIN9B_ADAPTER_POLICY_IDS,
    }.get(cfg.model.flux2_variant, {})
    if not (
        profile == "linux-dtk"
        and device_type == "cuda"
        and cfg.model.family == "flux2"
        and cfg.model.dtype == "bf16"
        and cfg.memory.base_precision in {"auto", "bf16"}
        and cfg.loop.gpu_count == 2
        and key in policy_ids
        and cfg.loop.deterministic
        and cfg.loop.mixed_precision == "bf16"
        and cfg.training.mode == "adapter"
        and cfg.training.train_backbone
        and not cfg.training.train_text_encoder
        and cfg.adapter.mode in {"auto", "bypass"}
        and cfg.adapter.param_dtype == "fp32"
        and not cfg.adapter.dora
        and all(rule.algo in {None, cfg.adapter.algo, "none"} for rule in cfg.adapter.rules)
        and cfg.memory.activation_checkpointing in {"none", "block"}
        and not cfg.memory.compile
        and not cfg.memory.blocks_to_swap
    ):
        return None
    if cfg.model.flux2_variant == "klein-base-4b" and key == ("lokr", "ddp"):
        # Only 4B LoKr/DDP uses native training with a preview-only identity.
        # 9B also needs stable training backbone contractions under DDP.
        return {
            "id": policy_ids[key],
            "mixed_precision": "bf16",
            "allow_tf32": False,
            "attention": "sdpa",
            "sdpa_backend": "math",
            "adapter_algorithm": "lokr",
            "distributed_strategy": cfg.loop.distributed_strategy,
            "preview_operator_components": ["backbone"],
            "preview_linear_forward": "bf16-rounded-operands-fp32-contraction-bf16-output",
            "preview_linear_implementation": BF16_LINEAR_PREVIEW_IMPLEMENTATION_ID,
        }
    policy: TrainingComputePolicy = {
        "id": policy_ids[key],
        "mixed_precision": "bf16",
        "allow_tf32": False,
        "attention": "sdpa",
        "sdpa_backend": "math",
        "linear_forward": "bf16-rounded-operands-fp32-contraction-bf16-output",
        "linear_backward": "fp32-contractions-grad-original-dtype",
        "linear_backward_implementation": BF16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID,
        "adapter_algorithm": cfg.adapter.algo,
        "trainable_components": ["backbone"],
        "operator_components": ["backbone"],
        "distributed_strategy": cfg.loop.distributed_strategy,
        "preview_operator_components": ["backbone"],
        "preview_linear_forward": "bf16-rounded-operands-fp32-contraction-bf16-output",
        "preview_linear_implementation": BF16_LINEAR_PREVIEW_IMPLEMENTATION_ID,
    }
    if cfg.adapter.algo == "lora":
        policy.update(
            adapter_forward="bf16-rounded-operands-fp32-contractions-bf16-intermediates",
            adapter_backward="fp32-contractions-grad-original-dtype",
            adapter_implementation=BF16_LORA_FP32_IMPLEMENTATION_ID,
        )
    if cfg.model.flux2_variant == "klein-base-9b":
        policy["frozen_text_implementation"] = KLEIN9B_TEXT_COMPUTE_ID
    if cfg.loop.distributed_strategy == "fsdp":
        policy.update(fsdp_param_dtype="bfloat16", fsdp_reduce_dtype="float32")
    return policy


def _with_sdxl_long_text_preview(cfg, policy):
    """Version the measured frozen-text long-caption preview recipe.

    Training contractions are unchanged. The separate preview fields prevent
    old checkpoints from silently claiming the new inference numeric behavior.
    """
    strategy = cfg.loop.distributed_strategy if cfg.loop.gpu_count > 1 else "single"
    key = (cfg.adapter.algo, strategy)
    if not (
        cfg.model.family == "sdxl"
        and cfg.model.sdxl_max_token_length > 75
        and cfg.training.mode == "adapter"
        and not cfg.training.train_text_encoder
        and key in DTK_SDXL_LONG_TEXT_PREVIEW_POLICY_IDS
    ):
        return policy
    return policy | {
        "id": DTK_SDXL_LONG_TEXT_PREVIEW_POLICY_IDS[key],
        "adapter_algorithm": cfg.adapter.algo,
        "trainable_components": ["backbone"],
        "operator_components": ["backbone"],
        "distributed_strategy": strategy,
        "preview_operator_components": ["backbone"],
        "preview_linear_forward": "bf16-rounded-operands-fp32-contraction-bf16-output",
        "preview_linear_implementation": BF16_LINEAR_PREVIEW_IMPLEMENTATION_ID,
    }


def _with_anima_fsdp_preview(cfg, policy):
    """Version frozen-text Anima adapter FSDP previews without changing training."""
    if not (
        cfg.model.family == "anima"
        and cfg.adapter.algo in DTK_ANIMA_FSDP_PREVIEW_POLICY_IDS
        and cfg.loop.distributed_strategy == "fsdp"
        and cfg.loop.gpu_count >= 2
        and policy["id"] == DTK_BACKBONE_ADAPTER_POLICY_IDS[("anima", cfg.adapter.algo, "fsdp")]
    ):
        return policy
    return policy | {
        "id": DTK_ANIMA_FSDP_PREVIEW_POLICY_IDS[cfg.adapter.algo],
        "preview_operator_components": ["backbone"],
        "preview_linear_forward": "bf16-rounded-operands-fp32-contraction-bf16-output",
        "preview_linear_implementation": BF16_LINEAR_PREVIEW_IMPLEMENTATION_ID,
    }


def resolve_training_compute_config(
    cfg: TrainConfig, device_type: str | None, profile: str
) -> tuple[TrainConfig, TrainingComputePolicy | None]:
    """Return an independent effective configuration and its resume identity.

    The runtime must still apply deterministic algorithms and the math SDPA
    backend. This function only resolves the shared planning/training recipe;
    it neither initializes devices nor changes process-wide backend settings.
    """
    effective = cfg.model_copy(deep=True)
    if (
        profile == "linux-dtk"
        and device_type == "cuda"
        and cfg.model.family == "anima"
        and cfg.model.dtype == "bf16"
        and cfg.memory.base_precision in {"auto", "bf16"}
        and cfg.loop.gpu_count == 1
        and cfg.loop.distributed_strategy == "ddp"
        and cfg.loop.deterministic
        and cfg.loop.mixed_precision == "fp16"
        and cfg.training.mode == "adapter"
        and cfg.training.train_backbone
        and not cfg.training.train_text_encoder
        and cfg.adapter.algo == "lora"
        and cfg.adapter.mode in {"auto", "bypass"}
        and cfg.adapter.param_dtype == "fp32"
        and not cfg.adapter.dora
        and all(rule.algo in {None, "lora", "none"} for rule in cfg.adapter.rules)
        and cfg.memory.activation_checkpointing in {"none", "block"}
        and not cfg.memory.compile
        and cfg.memory.blocks_to_swap == 0
    ):
        effective.model.attention = "sdpa"
        effective.memory.allow_tf32 = False
        return effective, {
            "id": DTK_ANIMA_LORA_SINGLE_FP16_POLICY_ID,
            "mixed_precision": "fp16",
            "allow_tf32": False,
            "attention": "sdpa",
            "sdpa_backend": "math",
            "linear_forward": "fp16-rounded-operands-fp32-contraction-fp16-output",
            "linear_backward": "fp32-contractions-grad-original-dtype",
            "linear_backward_implementation": FP16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID,
            "adapter_forward": "fp16-rounded-operands-fp32-contractions-fp16-intermediates",
            "adapter_backward": "fp32-contractions-grad-original-dtype",
            "adapter_implementation": FP16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID,
            "adapter_algorithm": "lora",
            "operator_components": ["backbone"],
            "trainable_components": ["backbone"],
            "distributed_strategy": "single",
        }
    if (
        profile == "linux-dtk"
        and device_type == "cuda"
        and cfg.model.family == "sdxl"
        and cfg.model.dtype == "bf16"
        and cfg.memory.base_precision == "bf16"
        and cfg.model.sdxl_max_token_length == 150
        and cfg.loop.gpu_count == 1
        and cfg.loop.distributed_strategy == "ddp"
        and cfg.loop.deterministic
        and cfg.loop.mixed_precision == "fp16"
        and cfg.training.mode == "adapter"
        and cfg.training.train_backbone
        and not cfg.training.train_text_encoder
        and cfg.adapter.algo == "lokr"
        and cfg.adapter.mode in {"auto", "bypass"}
        and cfg.adapter.param_dtype == "fp32"
        and not cfg.adapter.dora
        and all(rule.algo in {None, "lokr", "none"} for rule in cfg.adapter.rules)
        and cfg.memory.activation_checkpointing in {"none", "block"}
        and not cfg.memory.compile
        and not cfg.memory.blocks_to_swap
    ):
        # Native FP16 training and its GradScaler stay unchanged. Only the
        # measured backbone preview contractions receive a new resume identity.
        effective.memory.allow_tf32 = False
        effective.model.attention = "sdpa"
        return effective, {
            "id": DTK_SDXL_LOKR_SINGLE_FP16_PREVIEW_POLICY_ID,
            "mixed_precision": "fp16",
            "allow_tf32": False,
            "attention": "sdpa",
            "sdpa_backend": "math",
            "preview_operator_components": ["backbone"],
            "preview_linear_forward": "fp16-rounded-operands-fp32-contraction-fp16-output",
            "preview_linear_implementation": FP16_LINEAR_PREVIEW_IMPLEMENTATION_ID,
        }
    klein_policy = _klein_adapter_policy(cfg, device_type, profile)
    if klein_policy is not None:
        effective.memory.allow_tf32 = False
        effective.model.attention = "sdpa"
        return effective, klein_policy
    adapter_policy = _backbone_adapter_policy(cfg, device_type, profile)
    if adapter_policy is not None:
        effective.memory.allow_tf32 = False
        effective.model.attention = "sdpa"
        adapter_policy = _with_sdxl_long_text_preview(cfg, adapter_policy)
        return effective, _with_anima_fsdp_preview(cfg, adapter_policy)
    text_policy = _text_lora_policy(cfg, device_type, profile)
    if text_policy is not None:
        effective.memory.allow_tf32 = False
        effective.model.attention = "sdpa"
        return effective, text_policy
    if not (
        profile == "linux-dtk"
        and device_type == "cuda"
        and cfg.training.train_backbone
        and cfg.loop.deterministic
        and cfg.model.family in {"anima", "sdxl", "krea2"}
    ):
        return effective, None

    sdxl_lokr = (
        cfg.model.family == "sdxl"
        and cfg.training.mode == "adapter"
        and not cfg.training.train_text_encoder
        and (cfg.loop.gpu_count == 1 or cfg.loop.distributed_strategy == "ddp")
        and cfg.loop.mixed_precision == "bf16"
        and cfg.adapter.algo == "lokr"
        and cfg.adapter.mode in {"auto", "bypass"}
        and not cfg.adapter.dora
        and cfg.adapter.param_dtype == "fp32"
        and all(rule.algo in {None, "lokr", "none"} for rule in cfg.adapter.rules)
        and not cfg.memory.base_precision.startswith("fp8")
    )
    sdxl_sharded_full = (
        cfg.model.family == "sdxl"
        and cfg.training.mode == "full"
        and cfg.loop.gpu_count >= 2
        and cfg.loop.distributed_strategy == "fsdp"
    )
    anima_lokr = (
        cfg.model.family == "anima"
        and cfg.training.mode == "adapter"
        and cfg.adapter.algo == "lokr"
        and cfg.adapter.mode in {"auto", "bypass"}
        and not cfg.adapter.dora
        and cfg.adapter.param_dtype == "fp32"
        and all(rule.algo in {None, "lokr", "none"} for rule in cfg.adapter.rules)
        and not cfg.memory.base_precision.startswith("fp8")
    )
    anima_sharded_full = (
        cfg.model.family == "anima"
        and cfg.training.mode == "full"
        and cfg.loop.gpu_count >= 2
        and cfg.loop.distributed_strategy == "fsdp"
    )
    anima_ddp_lokr = anima_lokr and cfg.loop.gpu_count >= 2 and cfg.loop.distributed_strategy == "ddp"
    anima_linear = (
        cfg.model.family == "anima"
        and (cfg.training.mode == "full" or anima_lokr)
        and not cfg.training.train_text_encoder
        and (cfg.loop.gpu_count == 1 or anima_ddp_lokr or anima_sharded_full)
        and cfg.loop.mixed_precision == "bf16"
    )
    if cfg.training.mode != "full" and not sdxl_lokr and not anima_linear:
        return effective, None

    if sdxl_lokr and cfg.model.sdxl_max_token_length > 75:
        if (
            cfg.memory.activation_checkpointing not in {"none", "block"}
            or cfg.memory.compile
            or cfg.memory.blocks_to_swap
        ):
            raise ValueError("SDXL 长文本 BF16 可复现训练需要关闭编译/换块，并使用关闭或逐块重算")
        effective.memory.allow_tf32 = False
        effective.model.attention = "sdpa"
        return effective, _with_sdxl_long_text_preview(
            cfg,
            {
                "id": DTK_SDXL_LONG_TEXT_POLICY_ID,
                "mixed_precision": "bf16",
                "allow_tf32": False,
                "attention": "sdpa",
                "sdpa_backend": "math",
                "linear_forward": "bf16-rounded-operands-fp32-contraction-bf16-output",
                "linear_backward": "fp32-contractions-grad-original-dtype",
                "linear_backward_implementation": BF16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID,
                "conv_forward": "fp32-output-bf16",
                "conv_implementation": FP32_CONV_IMPLEMENTATION_ID,
                "sdxl_max_token_length": cfg.model.sdxl_max_token_length,
            },
        )

    effective.memory.allow_tf32 = False
    effective.model.attention = "sdpa"
    if anima_linear:
        if cfg.memory.activation_checkpointing == "unsloth":
            # Unsloth runs the first forward under no_grad and recomputes with
            # gradients. This policy deliberately preserves native no_grad
            # sampling, so those two forward computations would differ.
            raise ValueError(
                "Anima BF16 可复现训练暂不支持“重算并卸载中间输入”（Unsloth）。"
                "请将“重算中间结果（梯度检查点）”改为“关闭”或“逐块重算”。"
            )
        anima_policy: TrainingComputePolicy = {
            "id": DTK_ANIMA_FSDP_BF16_LINEAR_COMPUTE_POLICY_ID
            if anima_sharded_full
            else DTK_ANIMA_DDP_BF16_LINEAR_COMPUTE_POLICY_ID
            if anima_ddp_lokr
            else DTK_ANIMA_BF16_LINEAR_COMPUTE_POLICY_ID,
            "mixed_precision": "bf16",
            "allow_tf32": False,
            "attention": "sdpa",
            "sdpa_backend": "math",
            "linear_forward": "bf16-rounded-operands-fp32-contraction-bf16-output",
            "linear_backward": "fp32-contractions-grad-original-dtype",
            "linear_backward_implementation": BF16_LINEAR_FP32_COMPUTE_IMPLEMENTATION_ID,
        }
        if anima_sharded_full:
            # Gathered weights and their VJP round to BF16 before FP32
            # reduction; keep this boundary distinct from unsharded masters.
            anima_policy.update(fsdp_param_dtype="bfloat16", fsdp_reduce_dtype="float32")
        return effective, anima_policy
    if (
        cfg.model.family == "krea2"
        and not cfg.training.train_text_encoder
        and cfg.loop.distributed_strategy == "fsdp"
        and cfg.loop.gpu_count >= 2
        and cfg.loop.mixed_precision == "bf16"
    ):
        return effective, {
            "id": DTK_KREA2_FSDP_BF16_LINEAR_POLICY_ID,
            "mixed_precision": "bf16",
            "allow_tf32": False,
            "attention": "sdpa",
            "sdpa_backend": "math",
            "linear_forward": "native-bf16",
            "linear_backward": "fp32-contractions-grad-original-dtype",
            "linear_backward_implementation": BF16_LINEAR_BACKWARD_IMPLEMENTATION_ID,
            "fsdp_param_dtype": "bfloat16",
            "fsdp_reduce_dtype": "float32",
        }
    if (
        cfg.model.family == "sdxl"
        and not cfg.training.train_text_encoder
        and (cfg.loop.gpu_count == 1 or sdxl_lokr or sdxl_sharded_full)
        and cfg.loop.mixed_precision == "bf16"
    ):
        sdxl_policy: TrainingComputePolicy = {
            "id": DTK_SDXL_FSDP_BF16_CONV_LINEAR_POLICY_ID
            if sdxl_sharded_full
            else DTK_SDXL_BF16_CONV_LINEAR_POLICY_ID,
            "mixed_precision": "bf16",
            "allow_tf32": False,
            "attention": "sdpa",
            "sdpa_backend": "math",
            "linear_forward": "native-bf16",
            "linear_backward": "fp32-contractions-grad-original-dtype",
            "linear_backward_implementation": BF16_LINEAR_BACKWARD_IMPLEMENTATION_ID,
            "conv_forward": "fp32-output-bf16",
            "conv_implementation": FP32_CONV_IMPLEMENTATION_ID,
        }
        if sdxl_sharded_full:
            # FSDP gathers BF16 parameters before the FP32 convolution. This
            # rounding boundary differs from the unsharded FP32 master weights.
            sdxl_policy.update(fsdp_param_dtype="bfloat16", fsdp_reduce_dtype="float32")
        return effective, sdxl_policy
    effective.loop.mixed_precision = "no"
    policy: TrainingComputePolicy = {
        "id": DTK_FULL_FP32_MATH_POLICY_ID,
        "mixed_precision": "no",
        "allow_tf32": False,
        "attention": "sdpa",
        "sdpa_backend": "math",
    }
    return effective, policy


def validate_resume_compute_policy(expected: dict | None, saved: dict | None) -> None:
    """Reject a changed compute recipe instead of migrating optimizer state."""
    if expected == saved:
        return
    if expected is not None and saved is None:
        raise ValueError(
            "此训练状态未记录当前计算配方，不能按新的可复现计算策略恢复。"
            "请使用保存该状态的原版本继续，或从导出权重新建训练；不会自动迁移优化器状态。"
        )
    raise ValueError(
        "训练状态的计算配方与当前设置不同，不能进行严格续训。"
        "请恢复原计算策略并使用对应版本继续，或从导出权重新建训练。"
    )
