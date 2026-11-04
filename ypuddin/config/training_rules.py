"""Explicit component-training capabilities and incompatible memory paths."""

from .optimizer_rules import optimizer_key, optimizer_policy

FULL_FAMILIES = frozenset({"anima", "krea2", "sdxl", "flux2", "moss_image", "flux3", "toy"})
FULL_TEXT_FAMILIES = frozenset({"anima", "krea2", "sdxl", "flux2", "toy"})
TEXT_ADAPTER_FAMILIES = frozenset({"anima", "krea2", "sdxl", "flux2"})
NOISE_SHAPE_FAMILIES = frozenset({"anima", "krea2", "sdxl", "flux2", "flux3", "moss_image"})
_ANIMA_LR_FIELDS = frozenset({"llm_adapter_lr", "self_attn_lr", "cross_attn_lr", "mlp_lr", "modulation_lr"})
_FULL_LR_FIELDS = ("backbone_lr", "text_encoder_lr", "text_encoder_2_lr", *sorted(_ANIMA_LR_FIELDS))


def training_capabilities(family: str) -> dict:
    supported = family in FULL_FAMILIES
    result = {
        "tasks": ["standard", "slider", "edit"] if family in {"flux2", "flux3"} else ["standard", "slider"] if supported else ["standard"],
        "slider_modes": ["text", "image"] if supported else [],
        "max_reference_images": 1 if family in {"flux2", "flux3"} else 0,
        "modes": ["adapter", "full"] if supported else ["adapter"],
        "full_backbone": supported,
        "full_text_encoder": family in FULL_TEXT_FAMILIES,
        "adapter_text_encoder": family in TEXT_ADAPTER_FAMILIES,
        "text_encoder_components": ["CLIP-L", "CLIP-G"] if family == "sdxl" else ["文本编码器"],
        "full_block_swap": False,
        "full_fp8": False,
    }
    if family in {"moss_image", "flux3"}:
        result["full_save_dtypes"] = ["fp32", "bf16"]
    if family == "flux3":
        result.update(
            adapter_algorithms=["lora", "lokr", "loha", "full", "ortho"],
            model_dtypes=["auto", "bf16", "fp32"],
            base_precisions=["auto", "bf16", "fp32", "fp8_e4m3", "fp8_e5m2"],
            mixed_precisions=["no", "bf16"],
            distributed_strategies=["ddp"],
            gpu_counts=[1],
        )
    elif family == "moss_image":
        result.update(
            adapter_algorithms=["lora", "lokr", "loha", "full", "ortho", "tlora"],
            model_dtypes=["auto", "bf16", "fp32"],
            base_precisions=["auto", "bf16", "fp32", "fp8_e4m3", "fp8_e5m2"],
            mixed_precisions=["no", "bf16"],
            distributed_strategies=["ddp"],
            gpu_counts=[1],
        )
    return result


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
            cfg.objective.noise_offset > 0 and cfg.model.family not in NOISE_SHAPE_FAMILIES,
            "objective.noise_offset", "当前模型不支持噪声偏移",
        ),
        (
            cfg.objective.multires_noise_iterations > 0 and cfg.model.family not in NOISE_SHAPE_FAMILIES,
            "objective.multires_noise_iterations", "当前模型不支持多分辨率噪声",
        ),
        (
            cfg.objective.noise_offset > 0 and cfg.objective.multires_noise_iterations > 0,
            "objective.multires_noise_iterations", "噪声偏移与多分辨率噪声不能同时启用，请将其中一项设为 0",
        ),
    ]
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


def training_errors(cfg, *, preset: bool = False) -> list[dict[str, str]]:
    selection = cfg.training
    errors = _caption_noise_errors(cfg) + _full_optimizer_errors(cfg)

    def reject(loc, msg):
        errors.append({"loc": loc, "msg": msg})

    task = selection.task
    if task != "standard":
        if selection.mode != "adapter":
            reject("training.mode", "滑块和编辑训练请选择适配器训练")
        if selection.train_text_encoder:
            reject("training.train_text_encoder", "滑块和编辑训练需要冻结文本编码器")
        if cfg.loop.gpu_count != 1 or cfg.loop.distributed_strategy == "fsdp":
            reject("loop.gpu_count", "滑块和编辑训练目前支持单卡")
        if task == "edit" and cfg.model.family not in {"flux2", "flux3"}:
            reject("training.task", "当前模型不支持原生参考图编辑训练")
        if task == "edit" and cfg.model.family == "flux2" and cfg.model.flux2_variant == "dev":
            reject("model.flux2_variant", "编辑训练请选择 FLUX.2 Klein Base 4B 或 9B")
        if task == "slider":
            if cfg.model.family not in FULL_FAMILIES:
                reject("training.task", "当前模型不支持滑块训练")
            if cfg.adapter.algo != "lora":
                reject("adapter.algo", "滑块训练需要普通 LoRA")
            if cfg.adapter.dora:
                reject("adapter.dora", "滑块训练需要关闭 DoRA")
            for index, rule in enumerate(cfg.adapter.rules):
                if rule.algo not in {None, "none", "lora"}:
                    reject(f"adapter.rules.{index}.algo", "滑块训练的目标层只能使用 LoRA 或排除")
            if cfg.objective.infonoise_enabled:
                reject("objective.infonoise_enabled", "滑块训练不支持 InfoNoise")
            if cfg.slider.mode == "text":
                for field in ("neutral_prompt", "positive_prompt", "negative_prompt"):
                    if not getattr(cfg.slider, field).strip():
                        reject(f"slider.{field}", "请填写滑块提示词")
                if cfg.validation.enabled:
                    reject("validation.enabled", "文字滑块不使用图片验证集，请关闭图片验证")
                if cfg.dataset.resolution_mode != "bucket":
                    reject("dataset.resolution_mode", "文字滑块使用滑块分辨率，请选择分桶模式")
                if cfg.memory.blocks_to_swap:
                    reject("memory.blocks_to_swap", "文字滑块的基础样本生成暂不支持块换出")
        field = "reference_dir" if task == "edit" else "pair_path" if cfg.slider.mode == "image" else None
        if field:
            for index, source in enumerate(cfg.dataset.sources):
                if not getattr(source, field):
                    reject(f"dataset.sources.{index}.{field}", "请设置一一配对的图片目录")
    for index, prompt in enumerate(cfg.sampling.prompts):
        if prompt.reference_images and cfg.model.family not in {"flux2", "flux3"}:
            reject(f"sampling.prompts.{index}.reference_images", "当前模型不支持参考图预览")
        if not preset and task == "edit" and cfg.sampling.enabled and not prompt.reference_images:
            reject(f"sampling.prompts.{index}.reference_images", "编辑训练的预览需要一张参考图")

    if not selection.train_backbone and not selection.train_text_encoder:
        reject("training", "至少选择主模型或文本编码器参与训练")
    if selection.mode == "adapter":
        if selection.train_text_encoder and cfg.model.family not in TEXT_ADAPTER_FAMILIES:
            reject("training.train_text_encoder", "当前模型族尚未实现文本编码器适配器训练")
        if selection.resume_weights:
            reject("training.resume_weights", "全量模型权重不能作为适配器权重加载")
        adapter = cfg.adapter
        if adapter.dora and adapter.dora_compute_mode == "standard_fused":
            if cfg.memory.compile:
                reject("memory.compile", "DoRA 标准融合模式暂不支持编译，请关闭编译")
            if cfg.loop.gpu_count > 1 or cfg.loop.distributed_strategy == "fsdp":
                reject("adapter.dora_compute_mode", "DoRA 标准融合模式暂仅支持单卡训练")
        # OrthoLoRA and orthogonal T-LoRA start from their own orthonormal factors.
        algorithms = {adapter.algo, *(rule.algo or adapter.algo for rule in adapter.rules)}
        orthogonal = "ortho" in algorithms or "tlora" in algorithms and adapter.tlora_ortho
        if "tlora" in algorithms and adapter.mode == "merged":
            reject(
                "adapter.mode",
                "T-LoRA 按每张图的噪声强度调整秩，不能合并权重后计算，请选择自动或分开计算",
            )
        if adapter.algo == "tlora":
            if adapter.dora:
                reject("adapter.dora", "T-LoRA 按每张图的噪声强度调整秩，需要分开计算，不能与 DoRA 同时使用")
            if (
                isinstance(adapter.rank, int)
                and adapter.tlora_min_rank
                and adapter.tlora_min_rank > adapter.rank
            ):
                reject("adapter.tlora_min_rank", "最小秩不能大于 Rank")
        for index, rule in enumerate(adapter.rules):
            if (rule.algo or adapter.algo) != "tlora":
                continue
            ranks = [adapter.rank if rule.rank is None else rule.rank]
            if rule.rank is None and adapter.conv_rank is not None and trains_conv_adapters(cfg):
                ranks.append(adapter.conv_rank)
            # A rule can match linear or convolution layers; resolution selects the inherited rank.
            numeric_ranks = [rank for rank in ranks if isinstance(rank, int)]
            if not numeric_ranks:
                reject(f"adapter.rules.{index}.rank", "T-LoRA 的 Rank 请选择正整数")
            elif adapter.tlora_min_rank and adapter.tlora_min_rank > max(numeric_ranks):
                reject(f"adapter.rules.{index}.rank", "该规则的 Rank 不能小于 T-LoRA 最小秩")
        if adapter.algo == "full" and adapter.dora:
            reject("adapter.dora", "LyCORIS Full 直接训练完整权重，不能与 DoRA 同时使用")
        if orthogonal and adapter.init == "scalar":
            reject("adapter.init", "正交参数从自己的正交方向开始，不能使用「随机权重，零值缩放」")
        if orthogonal and adapter.resume_weights:
            reject("adapter.resume_weights", "导出文件是普通 LoRA，无法还原正交参数；请从完整恢复点继续训练")
    else:
        if cfg.model.family not in FULL_FAMILIES:
            reject("training.mode", "当前模型族尚未实现全量微调")
        if selection.train_text_encoder and cfg.model.family not in FULL_TEXT_FAMILIES:
            reject("training.train_text_encoder", "当前模型族的全量微调仅支持主模型，文本编码器保持冻结")
        if cfg.model.family in {"moss_image", "flux3"} and cfg.checkpoint.save_dtype == "fp16":
            reject("checkpoint.save_dtype", "该模型的完整权重保存精度请选择 FP32 或 BF16")
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
