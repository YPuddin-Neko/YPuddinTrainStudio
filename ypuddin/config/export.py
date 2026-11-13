"""Public training recipes projected from runtime applicability, without editing drafts."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .optimizer_rules import (
    AUTOMAGIC_FIELDS,
    MUON_FIELDS,
    PPSF_FIELDS,
    PRODIGY_FIELDS,
    optimizer_key,
    optimizer_specific_fields,
)
from .schema import TrainConfig

NATIVE_INACTIVE_BUCKET_FIELDS = frozenset({
    "resolutions", "aspect_ratio_limit", "area_tolerance", "bucket_step", "bucket_no_upscale",
})
_OPTIMIZER_FIELDS = set(PRODIGY_FIELDS + PPSF_FIELDS + AUTOMAGIC_FIELDS + MUON_FIELDS)
_BANK_SAMPLING_FIELDS = {
    "enabled", "sampler", "scheduler", "noise", "cfg", "guidance", "shift",
    "er_sde_order", "er_sde_s_noise",
}


def public_config_dict(config: TrainConfig) -> dict[str, Any]:
    """Keep an omitted checkpoint strategy omitted across public config round trips."""
    data = config.to_dict()
    if config.dataset.resolution_mode == "native" and config.checkpoint.resume:
        for field in ("native_batching_version", "native_batch_mode"):
            if field not in config.dataset.model_fields_set:
                data["dataset"].pop(field, None)
    return data


def _section(data: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def _drop(data: dict[str, Any], fields) -> None:
    for field in fields:
        data.pop(field, None)


def _only_for(data: dict[str, Any], selector: Mapping[str, Any], key: str, values, fields) -> None:
    # A preset may supply only dependent values for a later merge. Missing selectors
    # must not be interpreted as the defaults of a new complete training recipe.
    if key in selector and selector[key] not in values:
        _drop(data, fields)


def _native_geometry(data: dict[str, Any]) -> None:
    dataset = _section(data, "dataset")
    if dataset.get("resolution_mode") != "native":
        return
    _drop(dataset, NATIVE_INACTIVE_BUCKET_FIELDS)
    for section in ("dataset", "validation"):
        for source in _section(data, section).get("sources", []):
            if isinstance(source, dict):
                source.pop("resolutions", None)


def _project_model(data: dict[str, Any], config: TrainConfig | None) -> None:
    model, training = _section(data, "model"), _section(data, "training")
    adapter, memory = _section(data, "adapter"), _section(data, "memory")
    for fields, families in (
        (("prediction_type", "zero_terminal_snr", "sdxl_max_token_length"), {"sdxl"}),
        (("text_encoder_2_path",), {"sdxl", "flux"}),
        (("training_guidance",), {"flux"}), (("flux2_variant",), {"flux2"}),
        (("krea2_variant",), {"krea2"}),
    ):
        _only_for(model, model, "family", families, fields)
    _only_for(memory, model, "family", {"anima", "krea2"}, ("vae_attention_chunking", "no_half_vae"))
    _only_for(memory, model, "family", {"anima", "krea2", "flux3", "moss_image"}, ("vae_tiling", "cache_encode_tiled"))
    if training.get("mode") == "full":
        data.pop("adapter", None)
        memory.pop("blocks_to_swap", None)
    else:
        # Per-layer algorithm overrides inherit these global parameters.
        if "algo" in adapter and ("rules" in adapter or config is not None):
            algorithms = {adapter["algo"]}
            algorithms.update(rule.get("algo") for rule in adapter.get("rules", []) if isinstance(rule, dict))
            if "lokr" not in algorithms:
                _drop(adapter, ("factor", "decompose_both"))
            if "tlora" not in algorithms:
                _drop(adapter, ("tlora_min_rank", "tlora_power", "tlora_ortho"))
        if adapter.get("dora") is False:
            _drop(adapter, ("dora_axis", "dora_compute_mode", "dora_merge_dtype"))
        _only_for(adapter, adapter, "dora_compute_mode", {"comfyui"}, ("dora_merge_dtype",))
        if training.get("train_backbone") is False or adapter.get("layer_types") == "linear":
            _drop(adapter, ("conv_rank", "conv_alpha"))
        elif config is not None:
            from .training_rules import trains_conv_adapters

            if not trains_conv_adapters(config):
                _drop(adapter, ("conv_rank", "conv_alpha"))
    lr_fields = ("backbone_lr", "text_encoder_lr", "text_encoder_2_lr", "llm_adapter_lr", "self_attn_lr", "cross_attn_lr", "mlp_lr", "modulation_lr")
    _only_for(training, training, "mode", {"full"}, (*lr_fields, "resume_weights"))
    _only_for(training, training, "train_backbone", {True}, ("backbone_lr", *lr_fields[3:]))
    _only_for(training, training, "train_text_encoder", {True}, ("text_encoder_lr", "text_encoder_2_lr"))
    _only_for(training, model, "family", {"sdxl"}, ("text_encoder_2_lr",))
    _only_for(training, model, "family", {"anima"}, lr_fields[3:])


def _project_optimizer(data: dict[str, Any]) -> None:
    optimizer, training = _section(data, "optimizer"), _section(data, "training")
    if isinstance(optimizer.get("type"), str):
        key = optimizer_key(optimizer["type"])
        _drop(optimizer, _OPTIMIZER_FIELDS - set(optimizer_specific_fields(key)))
        if key not in {"adamw", "adam", "adamw8bit", "lion", "lion8bit", "prodigy", "prodigy_plus_sf", "adamw_sf", "came", "muon"}:
            optimizer.pop("betas", None)
        if key not in {"adamw", "adam", "adamw8bit", "adamw_sf", "prodigy", "prodigy_plus_sf", "automagic", "muon"}:
            optimizer.pop("eps", None)
        if key != "adamw":
            optimizer.pop("cpu_offload", None)
        # Partial recipes keep legacy arguments until schema migration resolves precedence.
        if key in {"adamw_sf", "automagic"} or (
            key == "prodigy_plus_sf" and optimizer.get("use_schedulefree") is True
            and "use_schedulefree" not in _section(optimizer, "args")
        ):
            data.pop("scheduler", None)
    # args are passed verbatim to third-party constructors, regardless of typed names.
    _only_for(optimizer, training, "mode", {"full"}, ("cpu_offload", "exclude_bias_norm_from_weight_decay"))
    scheduler = _section(data, "scheduler")
    _only_for(scheduler, scheduler, "type", {"cosine_restarts"}, ("num_cycles",))
    _only_for(scheduler, scheduler, "type", {"polynomial"}, ("power",))
    _only_for(scheduler, scheduler, "type", {"warmup_stable_decay"}, ("decay_steps",))
    if scheduler.get("type") == "constant":
        scheduler.pop("min_lr_ratio", None)


def _project_objective(data: dict[str, Any]) -> None:
    objective, model = _section(data, "objective"), _section(data, "model")
    for fields, samplers in (
        (("logit_mean", "logit_std"), {"logit_normal", "shift", "resolution_shift", "resolution_sqrt"}),
        (("shift",), {"shift", "resolution_sqrt"}),
        (("res_shift_tokens", "res_shift_mu"), {"resolution_shift"}),
        (("resolution_reference_tokens",), {"resolution_sqrt"}),
        (("mode_scale",), {"mode"}),
    ):
        _only_for(objective, objective, "timestep_sampling", samplers, fields)
    _only_for(objective, objective, "weighting", {"snr_like", "min_snr"}, ("snr_gamma",))
    _only_for(objective, objective, "loss", {"huber", "pseudo_huber"}, ("huber_c",))
    if objective.get("multires_noise_iterations") == 0:
        objective.pop("multires_noise_discount", None)
    if objective.get("infonoise_enabled") is False:
        _drop(objective, [key for key in objective if key.startswith("infonoise_") and key != "infonoise_enabled"])
    _only_for(objective, model, "family", {"sdxl"}, ("scale_v_pred_loss_like_noise_pred", "v_pred_like_loss", "debiased_estimation_loss"))
    _only_for(objective, model, "prediction_type", {"v_prediction"}, ("scale_v_pred_loss_like_noise_pred",))
    _only_for(objective, model, "prediction_type", {"epsilon"}, ("v_pred_like_loss",))


def _project_caption(caption: dict[str, Any], dataset: dict[str, Any], model: dict[str, Any]) -> None:
    _only_for(caption, model, "family", {"sdxl"}, ("weighted",))
    if caption.get("shuffle") is False and caption.get("tag_dropout") == 0:
        caption.pop("keep_tokens", None)
        if caption.get("wildcard") is False:
            caption.pop("cache_variants", None)
    if dataset.get("text_encoding") == "online":
        caption.pop("cache_variants", None)
    if caption.get("content_type") in {"anima_json", "gbc", "moss_layout"}:
        caption.pop("keep_tokens", None)


def _project_dataset(data: dict[str, Any]) -> None:
    dataset, training, model = _section(data, "dataset"), _section(data, "training"), _section(data, "model")
    if dataset.get("resolution_mode") == "bucket":
        _drop(dataset, [key for key in dataset if key.startswith("native_")])
    if dataset.get("resolution_mode") == "native":
        _only_for(dataset, dataset, "native_max_pixels_mode", {"custom"}, ("native_max_pixels",))
        _only_for(dataset, dataset, "native_max_pixels_mode", {"auto_vram"}, ("native_max_pixels_resolved",))
    if dataset.get("image_fit") == "pad":
        dataset.pop("crop_anchor", None)
    _project_caption(_section(dataset, "caption"), dataset, model)
    for section in ("dataset", "validation"):
        for source in _section(data, section).get("sources", []):
            if not isinstance(source, dict):
                continue
            if source.get("is_reg") is False:
                source.pop("prior_weight", None)
            _only_for(source, training, "task", {"edit"}, ("reference_dir",))
            _only_for(source, training, "task", {"slider"}, ("pair_path",))
            if _section(data, "slider").get("mode") == "text":
                source.pop("pair_path", None)
            # A source caption is its own override; global shuffle does not control it.
            _project_caption(_section(source, "caption"), dataset, model)


def _project_optional(data: dict[str, Any]) -> None:
    training, slider, sampling = _section(data, "training"), _section(data, "slider"), _section(data, "sampling")
    if "task" in training and training["task"] != "slider":
        data.pop("slider", None)
    elif slider.get("mode") == "image":
        _drop(slider, set(slider) - {"mode"})
    if sampling.get("enabled") is False:
        if "task" in training and training["task"] != "slider" or slider.get("mode") == "image":
            _drop(sampling, set(sampling) - {"enabled"})
        elif training.get("task") == "slider" and slider.get("mode") == "text":
            _drop(sampling, set(sampling) - _BANK_SAMPLING_FIELDS)
    _only_for(sampling, sampling, "sampler", {"er_sde"}, ("er_sde_order", "er_sde_s_noise"))
    _only_for(sampling, sampling, "noise", {"comfyui"}, ("adapter_merge_dtype",))
    _only_for(sampling, training, "mode", {"adapter"}, ("adapter_merge_dtype",))
    validation = _section(data, "validation")
    if validation.get("enabled") is False:
        _drop(validation, set(validation) - {"enabled"})
    loop, checkpoint = _section(data, "loop"), _section(data, "checkpoint")
    if loop.get("ema") is False:
        loop.pop("ema_decay", None)
    if checkpoint.get("save_training_metadata") is False:
        _drop(checkpoint, [key for key in checkpoint if key.startswith("metadata_")])
    for section in (checkpoint, sampling, validation):
        for key, value in list(section.items()):
            if key.endswith("_enabled") and value is False:
                section.pop(key.removesuffix("_enabled"), None)


def _restore_nondefaults(original: Any, projected: Any, defaults: Any) -> None:
    """Retain inactive nondefaults that still participate in old resume identities."""
    if isinstance(original, dict) and isinstance(projected, dict):
        defaults = defaults if isinstance(defaults, dict) else {}
        for key, value in original.items():
            if key in projected:
                _restore_nondefaults(value, projected[key], defaults.get(key))
            elif isinstance(value, dict) and isinstance(defaults.get(key), dict):
                retained = {}
                _restore_nondefaults(value, retained, defaults[key])
                if retained:
                    projected[key] = retained
            elif key not in defaults or value != defaults[key]:
                projected[key] = deepcopy(value)
    elif isinstance(original, list) and isinstance(projected, list):
        for index, (before, after) in enumerate(zip(original, projected, strict=True)):
            default = defaults[index] if isinstance(defaults, list) and index < len(defaults) else None
            _restore_nondefaults(before, after, default)



def _omit_default_null_fields(data: Any, path: list[str | int] | None = None) -> None:
    from .io import _schema_default_none

    path = [] if path is None else path
    if isinstance(data, dict):
        for key, value in list(data.items()):
            field_path = [*path, key]
            if value is None and _schema_default_none(TrainConfig, field_path):
                del data[key]
            else:
                _omit_default_null_fields(value, field_path)
    elif isinstance(data, list):
        for index, value in enumerate(data):
            _omit_default_null_fields(value, [*path, index])


def project_config_for_export(
    config: TrainConfig | Mapping[str, Any], *, preserve_resume_identity: bool = True,
    omit_default_nulls: bool | None = None,
) -> dict[str, Any]:
    """Omit inactive settings; retain unknown preset context and old recovery identity.

    The effective job view opts out of identity preservation and never becomes a
    replacement for its raw execution snapshot.
    """
    full_config = config if isinstance(config, TrainConfig) else None
    data = deepcopy(public_config_dict(config) if full_config is not None else dict(config))
    _native_geometry(data)
    if omit_default_nulls is None:
        omit_default_nulls = full_config is not None
    resume = preserve_resume_identity and bool(_section(data, "checkpoint").get("resume"))
    if resume and full_config is None:
        if omit_default_nulls:
            _omit_default_null_fields(data)
        return data
    original = deepcopy(data) if resume else None
    _project_dataset(data)
    _project_objective(data)
    _project_optional(data)
    _project_optimizer(data)
    _project_model(data, full_config)
    if original is not None:
        # Resolve defaults in the selected branch: managed optimizers, for example,
        # restore a constant scheduler rather than the generic scheduler default.
        restored = TrainConfig.model_validate(data).to_dict()
        _restore_nondefaults(original, data, restored)
    if omit_default_nulls:
        _omit_default_null_fields(data)
    return data
