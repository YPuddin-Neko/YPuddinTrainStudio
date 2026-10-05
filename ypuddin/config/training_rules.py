"""Explicit component-training capabilities and incompatible memory paths."""

from .optimizer_rules import optimizer_key, optimizer_policy

FULL_FAMILIES = frozenset({"anima", "krea2", "sdxl", "flux2", "toy"})
TEXT_ADAPTER_FAMILIES = frozenset({"anima", "krea2", "sdxl", "flux2"})
_ANIMA_LR_FIELDS = frozenset({"llm_adapter_lr", "self_attn_lr", "cross_attn_lr", "mlp_lr", "modulation_lr"})
_FULL_LR_FIELDS = ("backbone_lr", "text_encoder_lr", "text_encoder_2_lr", *sorted(_ANIMA_LR_FIELDS))


def training_capabilities(family: str) -> dict:
    supported = family in FULL_FAMILIES
    return {
        "modes": ["adapter", "full"] if supported else ["adapter"],
        "full_backbone": supported,
        "full_text_encoder": supported,
        "adapter_text_encoder": family in TEXT_ADAPTER_FAMILIES,
        "text_encoder_components": ["CLIP-L", "CLIP-G"] if family == "sdxl" else ["文本编码器"],
        "full_block_swap": False,
        "full_fp8": False,
    }


def trains_conv_adapters(cfg) -> bool:
    """Whether an adapter run adds adapters to convolution layers of the main model."""
    if cfg.training.mode != "adapter" or not cfg.training.train_backbone:
        return False
    if cfg.adapter.layer_types != "linear_conv":
        return False
    try:
        from ypuddin.adapters.rules import trains_convolutions
        from ypuddin.models import get_family

        preset = get_family(cfg.model.family).presets().get(cfg.adapter.preset)
    except Exception:  # noqa: BLE001 - an unknown family or preset fails elsewhere; assume convolutions
        return True
    return preset is None or trains_convolutions(cfg.adapter, preset)


def _caption_noise_errors(cfg) -> list[dict[str, str]]:
    checks = [
        (
            cfg.dataset.caption.weighted and cfg.model.family != "sdxl",
            "dataset.caption.weighted", "标签权重语法仅适用于 SDXL",
        ),
        (
            cfg.objective.noise_offset > 0 and cfg.model.family != "sdxl",
            "objective.noise_offset", "噪声偏移仅适用于 SDXL",
        ),
        (
            cfg.objective.multires_noise_iterations > 0 and cfg.model.family != "sdxl",
            "objective.multires_noise_iterations", "多分辨率噪声仅适用于 SDXL",
        ),
        (
            cfg.objective.noise_offset > 0 and cfg.objective.multires_noise_iterations > 0,
            "objective.multires_noise_iterations", "噪声偏移与多分辨率噪声不能同时启用，请将其中一项设为 0",
        ),
    ]
    for section in ("dataset", "validation"):
        for index, source in enumerate(getattr(cfg, section).sources):
            caption = source.caption
            checks.append((
                caption is not None and "weighted" in caption.model_fields_set
                and caption.weighted != cfg.dataset.caption.weighted,
                f"{section}.sources.{index}.caption.weighted",
                "标签权重语法对整个训练任务生效，请与数据集的标签权重设置保持一致",
            ))
    return [{"loc": loc, "msg": message} for failed, loc, message in checks if failed]


def _full_optimizer_errors(cfg) -> list[dict[str, str]]:
    checks = []
    policy = optimizer_policy(cfg.optimizer.type, use_schedulefree=cfg.optimizer.use_schedulefree)
    managed_groups = "optimizer.group_lr" in policy.get("fixed", {})
    for field in _FULL_LR_FIELDS:
        value = getattr(cfg.training, field)
        if value is None:
            continue
        loc = f"training.{field}"
        checks.extend([
            (cfg.training.mode != "full", loc, "组件和模块学习率仅用于全量微调，适配器训练请清空此设置"),
            (field in _ANIMA_LR_FIELDS and cfg.model.family != "anima", loc, "该模块学习率仅适用于 Anima"),
            (field == "text_encoder_2_lr" and cfg.model.family != "sdxl", loc, "第二文本编码器学习率仅适用于 SDXL"),
            (managed_groups and value > 0, loc, "当前优化器自动管理学习率，请留空；设为 0 仍可冻结该组参数"),
        ])
    checks.append((
        cfg.optimizer.exclude_bias_norm_from_weight_decay and cfg.training.mode != "full",
        "optimizer.exclude_bias_norm_from_weight_decay", "偏置和归一化参数的权重衰减排除仅用于全量微调",
    ))
    if cfg.optimizer.cpu_offload:
        checks.extend([
            (cfg.training.mode != "full", "optimizer.cpu_offload", "优化器 CPU 卸载仅用于全量微调"),
            (optimizer_key(cfg.optimizer.type) != "adamw", "optimizer.cpu_offload", "优化器 CPU 卸载需要选择 AdamW"),
            (cfg.optimizer.kahan, "optimizer.kahan", "优化器 CPU 卸载不支持 Kahan 补偿，请关闭补偿"),
            (cfg.loop.distributed_strategy == "fsdp", "optimizer.cpu_offload", "优化器 CPU 卸载不支持显存分片，请使用单卡或 DDP"),
            (cfg.optimizer.fused_backward, "optimizer.fused_backward", "优化器 CPU 卸载不支持反向即时更新"),
        ])
        for field in ("capturable", "differentiable", "fused", "fused_back_pass"):
            checks.append((
                bool(cfg.optimizer.args.get(field, False)), f"optimizer.args.{field}",
                f"优化器 CPU 卸载不支持 {field}=true，请移除此参数或设为 false",
            ))
    return [{"loc": loc, "msg": message} for failed, loc, message in checks if failed]


def training_errors(cfg) -> list[dict[str, str]]:
    selection = cfg.training
    errors = _caption_noise_errors(cfg) + _full_optimizer_errors(cfg)

    def reject(loc, msg):
        errors.append({"loc": loc, "msg": msg})

    if not selection.train_backbone and not selection.train_text_encoder:
        reject("training", "至少选择主模型或文本编码器参与训练")
    if selection.mode == "adapter":
        if selection.train_text_encoder and cfg.model.family not in TEXT_ADAPTER_FAMILIES:
            reject("training.train_text_encoder", "当前模型族尚未实现文本编码器适配器训练")
        if selection.resume_weights:
            reject("training.resume_weights", "全量模型权重不能作为适配器权重加载")
        adapter = cfg.adapter
        # OrthoLoRA and orthogonal T-LoRA start from their own orthonormal factors.
        orthogonal = adapter.algo == "ortho" or adapter.algo == "tlora" and adapter.tlora_ortho
        if adapter.algo == "tlora":
            if adapter.dora:
                reject("adapter.dora", "T-LoRA 按每张图的噪声强度调整秩，需要分开计算，不能与 DoRA 同时使用")
            if adapter.mode == "merged":
                reject(
                    "adapter.mode",
                    "T-LoRA 按每张图的噪声强度调整秩，不能合并权重后计算，请选择自动或分开计算",
                )
            if (
                isinstance(adapter.rank, int)
                and adapter.tlora_min_rank
                and adapter.tlora_min_rank > adapter.rank
            ):
                reject("adapter.tlora_min_rank", "最小秩不能大于 Rank")
        if adapter.algo == "full" and adapter.dora:
            reject("adapter.dora", "LyCORIS Full 直接训练完整权重，不能与 DoRA 同时使用")
        if orthogonal and adapter.init == "scalar":
            reject("adapter.init", "正交参数从自己的正交方向开始，不能使用「随机权重，零值缩放」")
        if orthogonal and adapter.resume_weights:
            reject("adapter.resume_weights", "导出文件是普通 LoRA，无法还原正交参数；请从完整恢复点继续训练")
    else:
        if cfg.model.family not in FULL_FAMILIES:
            reject("training.mode", "当前模型族尚未实现全量微调")
        if cfg.memory.base_precision not in {"auto", "fp32"}:
            reject("memory.base_precision", "全量微调的主参数使用 FP32，请选择保留精度；混合精度控制前向计算")
        if cfg.memory.blocks_to_swap:
            reject("memory.blocks_to_swap", "现有块换出只管理冻结权重，不能用于全量微调")
        if cfg.adapter.resume_weights:
            reject("adapter.resume_weights", "全量微调请使用 training.resume_weights 加载模型组件")
    if selection.train_text_encoder:
        if cfg.dataset.text_encoding == "cached":
            reject("dataset.text_encoding", "文本编码器训练需要每步重新编码，不能使用预编码缓存")
        if cfg.memory.offload_text_encoder:
            reject("memory.offload_text_encoder", "训练中的文本编码器需保留权重及反向图，不能在编码后卸载")
        if selection.mode == "adapter" and cfg.memory.blocks_to_swap:
            reject("memory.blocks_to_swap", "文本编码器适配器训练暂不支持主模型块换出")
        if selection.mode == "adapter" and cfg.memory.activation_checkpointing not in {"none", "block"}:
            reject("memory.activation_checkpointing", "文本编码器适配器训练的梯度检查点请选择“关闭”或“开启”")
    if cfg.loop.gpu_count > 1 or cfg.loop.distributed_strategy == "fsdp":
        errors.extend(distributed_training_errors(cfg))
    return errors


def distributed_training_errors(cfg) -> list[dict[str, str]]:
    """Shared admission checks for UI plans and externally launched torchrun jobs."""
    sharded = cfg.loop.distributed_strategy == "fsdp"
    checks = [
        (bool(cfg.memory.blocks_to_swap), "memory.blocks_to_swap", "多卡训练暂不支持块换出，请设置为 0"),
        (cfg.memory.compile, "memory.compile", "多卡训练暂不支持编译，请关闭编译"),
        (
            cfg.memory.activation_checkpointing not in {"none", "block"},
            "memory.activation_checkpointing",
            "多卡训练的梯度检查点请选择“关闭”或“开启”",
        ),
    ]
    if sharded:
        checks.extend(
            [
                (cfg.loop.gpu_count < 2, "loop.gpu_count", "显存分片至少需要两张显卡"),
                (
                    not cfg.training.train_backbone,
                    "training.mode",
                    "显存分片需要启用主模型训练",
                ),
                (
                    cfg.training.train_text_encoder,
                    "training.train_text_encoder",
                    "显存分片暂不支持同时训练文本编码器",
                ),
                (
                    cfg.training.mode == "adapter" and cfg.adapter.algo not in {"lora", "lokr"},
                    "adapter.algo",
                    "适配器显存分片请选择 LoRA 或 LoKr",
                ),
                (
                    cfg.training.mode == "adapter" and cfg.memory.base_precision.startswith("fp8"),
                    "memory.base_precision",
                    "适配器显存分片暂不支持 FP8 底模",
                ),
                (
                    cfg.training.mode == "adapter" and cfg.adapter.module_dropout > 0,
                    "adapter.module_dropout",
                    "显存分片需要各卡执行相同的适配层，请将整层丢弃率设为 0；仍可使用普通 Dropout 和 Rank Dropout",
                ),
                (
                    cfg.training.mode == "adapter" and cfg.adapter.param_dtype != "fp32",
                    "adapter.param_dtype",
                    "显存分片的适配器训练参数请选择 FP32；底模仍可使用 BF16 或 FP16",
                ),
                (cfg.loop.ema, "loop.ema", "显存分片暂不支持 EMA，请关闭 EMA"),
                (
                    optimizer_key(cfg.optimizer.type) not in {"adamw", "adafactor", "sgd"},
                    "optimizer.type",
                    "显存分片请选择 AdamW、Adafactor 或 SGD",
                ),
                (cfg.optimizer.kahan, "optimizer.kahan", "显存分片的主参数使用 FP32，请关闭低精度更新补偿"),
            ]
        )
    return [{"loc": loc, "msg": message} for failed, loc, message in checks if failed]
