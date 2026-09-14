"""Explicit component-training capabilities and incompatible memory paths."""

FULL_FAMILIES = frozenset({"anima", "krea2", "sdxl", "flux2", "toy"})


def training_capabilities(family: str) -> dict:
    supported = family in FULL_FAMILIES
    return {
        "modes": ["adapter", "full"] if supported else ["adapter"],
        "full_backbone": supported,
        "full_text_encoder": supported,
        "adapter_text_encoder": False,
        "text_encoder_components": ["CLIP-L", "CLIP-G"] if family == "sdxl" else ["文本编码器"],
        "full_block_swap": False,
        "full_fp8": False,
    }


def training_errors(cfg) -> list[dict[str, str]]:
    selection = cfg.training
    errors = []

    def reject(loc, msg):
        errors.append({"loc": loc, "msg": msg})

    if not selection.train_backbone and not selection.train_text_encoder:
        reject("training", "至少选择主模型或文本编码器参与训练")
    if selection.mode == "adapter":
        if selection.train_text_encoder:
            reject("training.train_text_encoder", "当前适配器模式只训练主模型；微调文本编码器请选择全量微调")
        if selection.resume_weights:
            reject("training.resume_weights", "全量模型权重不能作为适配器权重加载")
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
                reject(
                    "memory.offload_text_encoder", "训练中的文本编码器需保留权重及反向图，不能在编码后卸载"
                )
    return errors
