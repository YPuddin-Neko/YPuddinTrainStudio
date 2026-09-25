"""Public training recipes for safetensors metadata."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import PureWindowsPath
from typing import TYPE_CHECKING

from ypuddin.config.optimizer_rules import optimizer_specific_fields
from ypuddin.config.schema import CaptionConfig
from ypuddin.optim.factory import manages_learning_rate

if TYPE_CHECKING:
    from ypuddin.config import TrainConfig
    from ypuddin.data.dataset import DataBundle
    from ypuddin.train.state import Progress

# Keep paths, prompts, caption text and future configuration fields out of shared files.
_FIELDS = {
    "model": "family dtype attention prediction_type zero_terminal_snr sdxl_max_token_length flux2_variant krea2_variant",
    "training": "mode train_backbone train_text_encoder",
    "dataset": "resolutions resolution_mode image_fit native_max_pixels native_max_side native_overflow aspect_ratio_limit area_tolerance bucket_step bucket_no_upscale batch_size flip masked_loss cache_latents text_encoding",
    "objective": "timestep_sampling logit_mean logit_std res_shift_tokens res_shift_mu shift mode_scale stratified t_min t_max loss huber_c weighting snr_gamma ip_noise_gamma scale_v_pred_loss_like_noise_pred v_pred_like_loss debiased_estimation_loss",
    "scheduler": "type warmup_steps min_lr_ratio num_cycles power decay_steps",
    "memory": "base_precision blocks_to_swap activation_checkpointing offload_text_encoder compile allow_tf32",
    "loop": "gpu_count distributed_strategy max_steps epochs grad_accum mixed_precision seed deterministic ema ema_decay nan_skip_limit",
    "checkpoint": "save_dtype",
}
_CAPTION_FIELDS = "keep_tokens shuffle tag_dropout caption_dropout wildcard cache_variants"
_EXTRA_OPTIMIZER_ARGS = set(
    "amsgrad maximize foreach capturable differentiable fused momentum dampening nesterov "
    "scale_parameter relative_step warmup_init clip_threshold decay_rate beta1 eps betas "
    "min_8bit_size percentile_clipping block_wise paged is_paged warmup_steps r weight_lr_power".split()
)


def _select(value, names: str) -> dict:
    data = value.model_dump(mode="json")
    return {key: data[key] for key in names.split() if key in data}


def _numeric(value) -> bool:
    return (
        value is None
        or isinstance(value, (bool, int, float))
        or (isinstance(value, (list, tuple)) and all(_numeric(item) for item in value))
    )


def training_recipe_metadata(
    cfg: TrainConfig, bundle: DataBundle, progress: Progress, *, world_size: int = 1
) -> dict[str, str]:
    recipe = {section: _select(getattr(cfg, section), names) for section, names in _FIELDS.items()}
    recipe["adapter"] = cfg.adapter.model_dump(mode="json", exclude={"resume_weights"})
    recipe["dataset"]["caption"] = _select(cfg.dataset.caption, _CAPTION_FIELDS)
    counts = Counter(record.source_index for record in bundle.records)
    sources = []
    for index, source in enumerate(cfg.dataset.sources):
        public = _select(source, "repeats is_reg prior_weight resolutions")
        public["image_count"] = counts[index]
        public["caption"] = _select(
            source.caption
            if source.caption is not None
            else CaptionConfig()
            if source.is_reg
            else cfg.dataset.caption,
            _CAPTION_FIELDS,
        )
        sources.append(public)
    recipe["dataset"]["sources"] = sources
    recipe["loop"]["gpu_count"] = world_size
    optimizer_fields = "type lr weight_decay betas eps grad_clip_norm kahan group_lr " + " ".join(
        optimizer_specific_fields(cfg.optimizer.type)
    )
    recipe["optimizer"] = _select(cfg.optimizer, optimizer_fields)
    recipe["optimizer"]["args"] = {
        key: value
        for key, value in cfg.optimizer.args.items()
        if key in _EXTRA_OPTIMIZER_ARGS and _numeric(value)
    }
    recipe["optimizer"]["type"] = cfg.optimizer.type.rsplit(".", 1)[-1]
    managed = manages_learning_rate(cfg.optimizer)
    recipe["scheduler"]["active"] = not managed
    recipe["progress"] = {
        "step": progress.step,
        "epoch": progress.epoch,
        "total_steps": progress.total_steps,
        "steps_per_epoch": progress.steps_per_epoch,
    }
    opt = recipe["optimizer"]
    opt_args = {key: value for key, value in opt.items() if key != "type"}
    metadata = {
        "ypuddin.training_recipe_version": "1",
        "ypuddin.training_config": json.dumps(recipe, ensure_ascii=False, allow_nan=False),
        "ss_output_name": cfg.checkpoint.name,
        "ss_sd_model_name": PureWindowsPath(cfg.model.dit_path).name
        if cfg.model.dit_path
        else cfg.model.family,
        "ss_learning_rate": str(cfg.optimizer.lr),
        "ss_network_dropout": str(cfg.adapter.dropout),
        "ss_optimizer": f"{opt['type']}({json.dumps(opt_args, ensure_ascii=False, allow_nan=False)})",
        "ss_lr_scheduler": "optimizer-managed" if managed else cfg.scheduler.type,
        "ss_gradient_accumulation_steps": str(cfg.loop.grad_accum),
        "ss_batch_size_per_device": str(cfg.dataset.batch_size),
        "ss_total_batch_size": str(cfg.dataset.batch_size * cfg.loop.grad_accum * world_size),
        "ss_max_train_steps": str(progress.total_steps),
        "ss_num_train_images": str(bundle.plan.images),
        "ss_num_epochs": str(
            (progress.total_steps + max(progress.steps_per_epoch, 1) - 1) // max(progress.steps_per_epoch, 1)
        ),
        "ss_seed": str(cfg.loop.seed),
        "ss_mixed_precision": cfg.loop.mixed_precision,
        "ss_gradient_checkpointing": str(cfg.memory.activation_checkpointing != "none"),
        "ss_shuffle_caption": str(cfg.dataset.caption.shuffle),
        "ss_keep_tokens": str(cfg.dataset.caption.keep_tokens),
        "ss_caption_dropout_rate": str(cfg.dataset.caption.caption_dropout),
        "ss_caption_tag_dropout_rate": str(cfg.dataset.caption.tag_dropout),
        "ss_flip_aug": str(cfg.dataset.flip),
        "ss_dataset_dirs": json.dumps(
            {
                f"dataset_{index + 1}": {"img_count": source["image_count"], "n_repeats": source["repeats"]}
                for index, source in enumerate(sources)
            }
        ),
    }
    if not managed:
        warmup = cfg.scheduler.warmup_steps
        metadata["ss_lr_warmup_steps"] = str(
            int(round(warmup * progress.total_steps)) if warmup < 1 else int(warmup)
        )
    if cfg.model.family == "sdxl":
        metadata["modelspec.prediction_type"] = cfg.model.prediction_type
        metadata["ss_ip_noise_gamma"] = str(cfg.objective.ip_noise_gamma)
    if cfg.objective.weighting == "min_snr":
        metadata["ss_min_snr_gamma"] = str(cfg.objective.snr_gamma)
    if cfg.dataset.resolution_mode == "native":
        metadata["ss_resolution"] = "native"
    return metadata
