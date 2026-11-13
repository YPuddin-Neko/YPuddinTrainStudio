"""Public training recipes projected from runtime applicability, without editing drafts."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, get_args

from pydantic import ValidationError

from .optimizer_rules import (
    AUTOMAGIC_FIELDS,
    EMOSENS_FIELDS,
    MUON_FIELDS,
    PPSF_FIELDS,
    PRODIGY_FIELDS,
    optimizer_key,
    optimizer_policy,
    optimizer_specific_fields,
)
from .schema import CaptionConfig, DatasetSourceConfig, ModelConfig, TrainConfig

NATIVE_INACTIVE_BUCKET_FIELDS = frozenset({
    "resolutions", "aspect_ratio_limit", "area_tolerance", "bucket_step", "bucket_no_upscale",
})
_OPTIMIZER_FIELDS = set(PRODIGY_FIELDS + PPSF_FIELDS + AUTOMAGIC_FIELDS + MUON_FIELDS + EMOSENS_FIELDS)
_BANK_SAMPLING_FIELDS = {
    "enabled", "sampler", "scheduler", "noise", "cfg", "guidance", "shift",
    "er_sde_order", "er_sde_s_noise",
}
_MODEL_FAMILIES = frozenset(get_args(ModelConfig.model_fields["family"].annotation))
_TILED_FAMILIES = frozenset({"anima", "krea2", "flux3", "moss_image"})
EXECUTION_CONFIG_VERSION = 1


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
    if (model.get("family") == "sdxl" and model.get("prediction_type") == "epsilon"
            and model.get("zero_terminal_snr") is False):
        model.pop("zero_terminal_snr")
    _only_for(memory, model, "family", {"anima", "krea2"}, ("vae_attention_chunking", "no_half_vae"))
    _only_for(memory, model, "family", _TILED_FAMILIES, ("vae_tiling", "cache_encode_tiled"))
    dataset = _section(data, "dataset")
    if dataset.get("cache_latents") is False:
        memory.pop("cache_encode_tiled", None)
    if training.get("train_text_encoder") is False and (
        dataset.get("text_encoding") == "cached" or (
            dataset.get("text_encoding") == "auto" and model.get("family") in _MODEL_FAMILIES
            and not _online_text(dataset, training, model)
        )
    ):
        memory.pop("offload_text_encoder", None)
    if training.get("mode") == "full":
        data.pop("adapter", None)
        memory.pop("blocks_to_swap", None)
    else:
        no_conv = training.get("train_backbone") is False or adapter.get("layer_types") == "linear"
        conv_context = config
        if (conv_context is None and training.get("mode") == "adapter"
                and training.get("train_backbone") is True and adapter.get("layer_types") == "linear_conv"
                and isinstance(model.get("family"), str) and isinstance(adapter.get("preset"), str)):
            conv_context = SimpleNamespace(
                model=SimpleNamespace(family=model["family"]),
                training=SimpleNamespace(mode="adapter", train_backbone=True),
                adapter=SimpleNamespace(layer_types="linear_conv", preset=adapter["preset"]),
            )
        if not no_conv and conv_context is not None:
            from .training_rules import trains_conv_adapters

            no_conv = not trains_conv_adapters(conv_context)
        _project_adapter_rules(data, inherits_full_rank=(
            adapter.get("rank") == "full" and (
                no_conv or (adapter.get("layer_types") == "linear_conv"
                            and "conv_rank" in adapter and adapter["conv_rank"] is None)
            )
        ))
        # Per-layer algorithm overrides inherit these global parameters.
        if "algo" in adapter and ("rules" in adapter or config is not None):
            algorithms = {adapter["algo"]}
            algorithms.update(rule.get("algo") for rule in adapter.get("rules", []) if isinstance(rule, dict))
            if "lokr" not in algorithms:
                _drop(adapter, ("factor", "decompose_both"))
            if "tlora" not in algorithms:
                _drop(adapter, ("tlora_min_rank", "tlora_power", "tlora_ortho"))
        if (
            training.get("mode") == "adapter"
            and adapter.get("algo") == "lokr" and adapter.get("rank") == "full"
            and isinstance(adapter.get("rules"), list)
            and all(isinstance(rule, dict) and (
                rule.get("algo") == "none" or (
                    (rule.get("algo") or "lokr") == "lokr" and rule.get("rank") in (None, "full")
                )
            ) for rule in adapter["rules"])
            and (no_conv or (adapter.get("layer_types") == "linear_conv"
                             and "conv_rank" in adapter and adapter["conv_rank"] is None))
        ):
            # Both backbone and text adapters inherit full factors here.
            # Their exported tensor alpha is 1; numeric-rank overrides need these values.
            _drop(adapter, ("decompose_both", "rs_lora", "alpha", "rank_dropout", "conv_alpha"))
        if adapter.get("dora") is False:
            _drop(adapter, ("dora_axis", "dora_compute_mode", "dora_merge_dtype"))
        _only_for(adapter, adapter, "dora_compute_mode", {"comfyui"}, ("dora_merge_dtype",))
        if no_conv:
            _drop(adapter, ("conv_rank", "conv_alpha"))
        if (
            training.get("mode") == "adapter" and adapter.get("algo") == "full"
            and isinstance(adapter.get("rules"), list)
            and all(isinstance(rule, dict) and rule.get("algo") in (None, "none", "full")
                    for rule in adapter["rules"])
        ):
            _drop(adapter, ("alpha", "factor", "decompose_both", "rs_lora", "init", "dropout", "rank_dropout",
                            "tlora_min_rank", "tlora_power", "tlora_ortho", "conv_rank", "conv_alpha"))
            if adapter.get("rank") != "full":
                adapter.pop("rank", None)
        if (training.get("mode") == "adapter" and training.get("train_backbone") is False
                and training.get("train_text_encoder") is True):
            _drop(adapter, ("preset", "layer_types"))
    lr_fields = ("backbone_lr", "text_encoder_lr", "text_encoder_2_lr", "llm_adapter_lr", "self_attn_lr", "cross_attn_lr", "mlp_lr", "modulation_lr")
    _only_for(training, training, "mode", {"full"}, (*lr_fields, "resume_weights"))
    _only_for(training, training, "train_backbone", {True}, ("backbone_lr", *lr_fields[3:]))
    _only_for(training, training, "train_text_encoder", {True}, ("text_encoder_lr", "text_encoder_2_lr"))
    _only_for(training, model, "family", {"sdxl"}, ("text_encoder_2_lr",))
    _only_for(training, model, "family", {"anima"}, lr_fields[3:])


def _project_adapter_rules(data: dict[str, Any], *, inherits_full_rank: bool) -> None:
    adapter = _section(data, "adapter")
    for rule in adapter.get("rules", []):
        if not isinstance(rule, dict):
            continue
        algorithm = rule.get("algo") or adapter.get("algo")
        if algorithm == "none":
            _drop(rule, ("rank", "alpha", "factor", "lr", "dropout", "rank_dropout"))
        elif algorithm == "full":
            _drop(rule, ("alpha", "factor", "dropout", "rank_dropout"))
            # The adapter builder rejects the full-rank sentinel before dispatch.
            if rule.get("rank") != "full":
                rule.pop("rank", None)
        elif algorithm == "lokr" and (
            rule.get("rank") == "full" or rule.get("rank") is None and inherits_full_rank
        ):
            _drop(rule, ("alpha", "rank_dropout"))
        elif algorithm in {"lora", "loha", "ortho", "tlora"}:
            rule.pop("factor", None)


def _project_optimizer(data: dict[str, Any]) -> None:
    optimizer, training = _section(data, "optimizer"), _section(data, "training")
    if optimizer.get("weight_decay") == 0:
        optimizer.pop("exclude_bias_norm_from_weight_decay", None)
    if isinstance(optimizer.get("type"), str):
        key = optimizer_key(optimizer["type"])
        _drop(optimizer, _OPTIMIZER_FIELDS - set(optimizer_specific_fields(key)))
        if key not in {"adamw", "adam", "adamw8bit", "lion", "lion8bit", "prodigy", "prodigy_plus_sf", "adamw_sf", "came", "muon", "emosens"}:
            optimizer.pop("betas", None)
        if key not in {"adamw", "adam", "adamw8bit", "adamw_sf", "prodigy", "prodigy_plus_sf", "automagic", "muon", "emosens"}:
            optimizer.pop("eps", None)
        if key != "adamw":
            optimizer.pop("cpu_offload", None)
        if key == "prodigy_plus_sf":
            args = _section(optimizer, "args")
            for selector, inactive, field in (
                ("use_schedulefree", False, "schedulefree_c"),
                ("split_groups", False, "split_groups_mean"),
                ("factored", False, "factored_fp32"),
                ("weight_decay", 0, "weight_decay_by_lr"),
                ("use_speed", True, "d_limiter"),
            ):
                if selector in optimizer and optimizer[selector] == inactive and not ({selector, field} & args.keys()):
                    optimizer.pop(field, None)
        if key == "prodigy" and optimizer.get("weight_decay") == 0:
            args = _section(optimizer, "args")
            if "decouple" not in args and args.get("weight_decay", 0) == 0:
                optimizer.pop("decouple", None)
        # Partial recipes keep legacy arguments until schema migration resolves precedence.
        if key in {"adamw_sf", "automagic", "emosens"} or (
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


def _project_resolution_reference(objective: dict[str, Any]) -> None:
    if "timestep_sampling" not in objective:
        return
    if objective["timestep_sampling"] != "resolution_sqrt":
        _drop(objective, ("resolution_reference_mode", "resolution_reference_tokens_resolved"))
        objective.pop("resolution_reference_tokens", None)
    elif objective.get("resolution_reference_mode") == "auto":
        objective.pop("resolution_reference_tokens", None)
    elif objective.get("resolution_reference_mode") == "manual":
        objective.pop("resolution_reference_tokens_resolved", None)


def _project_objective(data: dict[str, Any]) -> None:
    objective, model = _section(data, "objective"), _section(data, "model")
    for fields, samplers in (
        (("logit_mean", "logit_std"), {"logit_normal", "shift", "resolution_shift", "resolution_sqrt"}),
        (("shift",), {"shift", "resolution_sqrt"}),
        (("res_shift_tokens", "res_shift_mu"), {"resolution_shift"}),
        (("mode_scale",), {"mode"}),
    ):
        _only_for(objective, objective, "timestep_sampling", samplers, fields)
    _project_resolution_reference(objective)
    _only_for(objective, objective, "weighting", {"snr_like", "min_snr"}, ("snr_gamma",))
    _only_for(objective, objective, "loss", {"huber", "pseudo_huber"}, ("huber_c",))
    if objective.get("multires_noise_iterations") == 0:
        objective.pop("multires_noise_discount", None)
    if model.get("family") in _MODEL_FAMILIES - {"anima"}:
        _drop(objective, [key for key in objective if key.startswith("infonoise_")])
    elif objective.get("infonoise_enabled") is False:
        _drop(objective, [key for key in objective if key.startswith("infonoise_") and key != "infonoise_enabled"])
    _only_for(objective, model, "family", {"sdxl"}, ("scale_v_pred_loss_like_noise_pred", "v_pred_like_loss", "debiased_estimation_loss"))
    _only_for(objective, model, "prediction_type", {"v_prediction"}, ("scale_v_pred_loss_like_noise_pred",))
    _only_for(objective, model, "prediction_type", {"epsilon"}, ("v_pred_like_loss",))


def _online_text(dataset: dict[str, Any], training: dict[str, Any], model: dict[str, Any]) -> bool:
    if training.get("train_text_encoder") is True or dataset.get("text_encoding") == "online":
        return True
    if dataset.get("text_encoding") != "auto" or not isinstance(model.get("family"), str):
        return False
    from ypuddin.models import get_family

    try:
        family = get_family(model["family"])
    except (KeyError, ImportError):
        return False
    return "online_text" in family.spec.capabilities


def _project_caption(caption: dict[str, Any], model: dict[str, Any], *, online_text: bool) -> None:
    _only_for(caption, model, "family", {"sdxl"}, ("weighted",))
    if caption.get("shuffle") is False and caption.get("tag_dropout") == 0:
        caption.pop("keep_tokens", None)
        if caption.get("wildcard") is False:
            caption.pop("cache_variants", None)
    if online_text:
        caption.pop("cache_variants", None)
    if caption.get("content_type") in {"anima_json", "gbc", "moss_layout"}:
        caption.pop("keep_tokens", None)


def _project_dataset(data: dict[str, Any]) -> None:
    dataset, training, model = _section(data, "dataset"), _section(data, "training"), _section(data, "model")
    online_text = _online_text(dataset, training, model)
    if dataset.get("resolution_mode") == "bucket":
        _drop(dataset, [key for key in dataset if key.startswith("native_")])
    if dataset.get("resolution_mode") == "native":
        _only_for(dataset, dataset, "native_max_pixels_mode", {"custom"}, ("native_max_pixels",))
        _only_for(dataset, dataset, "native_max_pixels_mode", {"auto_vram"}, ("native_max_pixels_resolved",))
    if dataset.get("image_fit") == "pad":
        dataset.pop("crop_anchor", None)
    _project_caption(_section(dataset, "caption"), model, online_text=online_text)
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
            _project_caption(_section(source, "caption"), model, online_text=online_text)


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
    if _section(data, "model").get("family") in _MODEL_FAMILIES - {"flux"}:
        sampling.pop("guidance", None)
    if _section(data, "model").get("family") == "sdxl" and sampling.get("shift") == 1:
        sampling.pop("shift")
    validation = _section(data, "validation")
    if validation.get("enabled") is False:
        _drop(validation, set(validation) - {"enabled"})
    loop, checkpoint = _section(data, "loop"), _section(data, "checkpoint")
    if loop.get("gpu_count") == 1 and loop.get("distributed_strategy") == "ddp":
        loop.pop("distributed_strategy")
    _section(data, "logging").pop("level", None)
    if loop.get("ema") is False:
        loop.pop("ema_decay", None)
    if checkpoint.get("save_training_metadata") is False:
        _drop(checkpoint, [key for key in checkpoint if key.startswith("metadata_")])
    for section in (checkpoint, sampling, validation):
        for key, value in list(section.items()):
            if key.endswith("_enabled") and value is False:
                section.pop(key.removesuffix("_enabled"), None)


def _project_fixed_controls(data: dict[str, Any]) -> None:
    if (_section(data, "training").get("task") == "slider"
            and _section(data, "slider").get("mode") == "text"):
        dataset = _section(data, "dataset")
        _drop(dataset, ("sources", "resolutions", "aspect_ratio_limit", "area_tolerance", "bucket_step",
                        "bucket_no_upscale", "image_fit", "crop_anchor", "flip", "masked_loss"))
        if dataset.get("resolution_mode") == "bucket":
            dataset.pop("resolution_mode")
        if _section(data, "sampling").get("enabled") is False:
            dataset.pop("cache_latents", None)
        _section(data, "memory").pop("cache_encode_tiled", None)
    optimizer = _section(data, "optimizer")
    if optimizer.get("fused_backward") is False:
        optimizer.pop("fused_backward")
    if isinstance(optimizer.get("type"), str):
        key = optimizer_key(optimizer["type"])
        fixed = optimizer_policy(key, use_schedulefree=optimizer.get("use_schedulefree") is True).get("fixed", {})
        if fixed.get("optimizer.kahan") is False and optimizer.get("kahan") is False:
            optimizer.pop("kahan")
        if fixed.get("optimizer.group_lr") == {} and optimizer.get("group_lr") == {}:
            optimizer.pop("group_lr")
        adapter = _section(data, "adapter")
        if (
            _section(data, "training").get("mode") == "adapter"
            and fixed.get("adapter.lr_scale") == {} and adapter.get("lr_scale") == {}
        ):
            adapter.pop("lr_scale")
    family = _section(data, "model").get("family")
    if family in _MODEL_FAMILIES:
        from .training_rules import training_capabilities

        if not training_capabilities(family)["max_reference_images"]:
            for prompt in _section(data, "sampling").get("prompts", []):
                if isinstance(prompt, dict) and prompt.get("reference_images") == []:
                    prompt.pop("reference_images")
        captions = [_section(_section(data, "dataset"), "caption")]
        for section in ("dataset", "validation"):
            captions.extend(_section(source, "caption") for source in _section(data, section).get("sources", [])
                            if isinstance(source, dict))
        for caption in captions:
            content_type = caption.get("content_type", "auto")
            if caption.get("structured_mode") == "native" and (
                content_type in {"text", "anima_json"}
                or family != "moss_image" and content_type == "auto"
            ):
                caption.pop("structured_mode")


def _restore_nondefaults(original: Any, projected: Any, defaults: Any) -> None:
    """Retain selected contract values that schema defaults would otherwise change."""
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


def _restore_resume_contract(original: dict[str, Any], data: dict[str, Any]) -> None:
    try:
        defaults = TrainConfig.model_validate(data).to_dict()
    except ValidationError:
        # A partial preset may only become valid after merging into its destination.
        defaults = TrainConfig().to_dict()
    for section in ("dataset", "validation"):
        before = _section(original, section)
        if not before:
            continue
        after = data.setdefault(section, {})
        selected = ({"caption": before["caption"]} if "caption" in before else {}) if section == "dataset" else {
            key: value for key, value in before.items() if key != "sources"
        }
        _restore_nondefaults(selected, after, _section(defaults, section))
        if section == "validation":
            for key, value in before.items():
                if key.endswith("_enabled") and key.removesuffix("_enabled") in after:
                    after[key] = deepcopy(value)
        if "sources" not in after:
            continue
        default_sources = _section(defaults, section).get("sources", [])
        for index, (old_source, new_source) in enumerate(zip(before.get("sources", []), after["sources"], strict=True)):
            if not isinstance(old_source, dict) or not isinstance(new_source, dict):
                continue
            selected = {key: old_source[key] for key in ("caption", "prior_weight") if key in old_source}
            source_defaults = default_sources[index] if index < len(default_sources) else {}
            _restore_nondefaults(selected, new_source, source_defaults)
    contracts = {}
    adapter = _section(original, "adapter")
    if "rules" in adapter and _section(original, "training").get("mode") != "full":
        contracts["adapter"] = {"rules": adapter["rules"]}
    if "scheduler" in original:
        contracts["scheduler"] = original["scheduler"]
    optimizer = _section(original, "optimizer")
    keys = {"betas", "eps", "weight_decay", "args", "exclude_bias_norm_from_weight_decay"}
    if isinstance(optimizer.get("type"), str):
        keys.update(optimizer_specific_fields(optimizer["type"]))
    if optimizer:
        contracts["optimizer"] = {key: optimizer[key] for key in keys if key in optimizer}
    memory = _section(original, "memory")
    if _section(original, "model").get("family") in _TILED_FAMILIES and "cache_encode_tiled" in memory:
        contracts["memory"] = {"cache_encode_tiled": memory["cache_encode_tiled"]}
    if (_section(original, "training").get("task") == "slider"
            and _section(original, "slider").get("mode") == "text"):
        sampling = _section(original, "sampling")
        if "guidance" in sampling:
            contracts["sampling"] = {"guidance": sampling["guidance"]}
    _restore_nondefaults(contracts, data, defaults)



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
    """Omit inactive settings; retain unknown preset context and resume contracts.

    The effective job view opts out of identity preservation and never becomes a
    replacement for its raw execution snapshot.
    """
    full_config = config if isinstance(config, TrainConfig) else None
    data = deepcopy(public_config_dict(config) if full_config is not None else dict(config))
    _native_geometry(data)
    if omit_default_nulls is None:
        omit_default_nulls = full_config is not None
    resume = preserve_resume_identity and bool(_section(data, "checkpoint").get("resume"))
    original = deepcopy(data) if resume else None
    _project_dataset(data)
    _project_objective(data)
    _project_optional(data)
    _project_optimizer(data)
    _project_model(data, full_config)
    if full_config is not None or omit_default_nulls is True:
        _project_fixed_controls(data)
    if original is not None:
        _restore_resume_contract(original, data)
    if omit_default_nulls:
        _omit_default_null_fields(data)
    for key in TrainConfig.model_fields:
        if data.get(key) == {}:
            data.pop(key)
    return data


def project_config_for_execution(config: TrainConfig) -> dict[str, Any]:
    """Write an applicable recipe that reloads to the frozen execution values."""
    original = config.to_dict()
    data = project_config_for_export(config, preserve_resume_identity=False)
    # Seeds, data fingerprints, scheduler state and sharded adapter contracts
    # consume values beyond the visible controls. Keep their nondefault values;
    # defaults omitted by the applicability projection reconstruct on load.
    defaults = TrainConfig.model_validate(data).to_dict()
    _restore_nondefaults(original, data, defaults)
    return data


def project_config_for_storage(config: Mapping[str, Any], *, partial: bool = False) -> dict[str, Any]:
    """Save applicable supplied values without expanding defaults into a draft.

    Partial presets inherit missing selectors from their destination. A complete
    recipe uses schema defaults for those selectors. Exact recovery keeps its
    original recipe because inactive values can still participate in its identity.
    """
    from .intervals import normalize_legacy_intervals
    from .io import deep_merge

    raw = deepcopy(dict(config))
    if _section(raw, "checkpoint").get("resume"):
        return raw
    raw = normalize_legacy_intervals(raw)
    try:
        if partial:
            return project_config_for_export(raw, omit_default_nulls=False)

        context = deep_merge(TrainConfig().to_dict(), deepcopy(raw))
        for section in ("dataset", "validation"):
            for source in _section(context, section).get("sources", []):
                if isinstance(source, dict):
                    defaults = DatasetSourceConfig.model_construct().model_dump(mode="json")
                    source.update(deep_merge(defaults, source))
                    if isinstance(source.get("caption"), dict):
                        source["caption"] = deep_merge(CaptionConfig().model_dump(mode="json"), source["caption"])
        projected = project_config_for_export(context, preserve_resume_identity=False, omit_default_nulls=True)
    except (TypeError, ValueError):
        # Keep malformed drafts available for the normal field-error response.
        try:
            TrainConfig.model_validate(raw)
        except ValidationError:
            return raw
        raise

    def supplied(value: Any, active: Any) -> Any:
        if isinstance(value, dict) and isinstance(active, dict):
            return {key: supplied(item, active[key]) for key, item in value.items() if key in active}
        if isinstance(value, list) and isinstance(active, list):
            return [supplied(item, active[index]) for index, item in enumerate(value)]
        return deepcopy(value)

    data = supplied(raw, projected)
    for key in TrainConfig.model_fields:
        if data.get(key) == {}:
            data.pop(key)
    return data
