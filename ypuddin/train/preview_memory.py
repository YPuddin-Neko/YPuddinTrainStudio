"""Condition and image tensors retained between consecutive training previews."""

from __future__ import annotations

import math
from collections.abc import Callable

from ypuddin.models.base import LatentSpec


def text_capacity_complete(profile: dict | None, *, unified_memory: bool,
                           cached_text: bool = False) -> bool:
    if profile is None or not unified_memory:
        return True
    prefix = "cached_" if cached_text else ""
    return all(profile.get(prefix + key) is not None
               for key in ("host_staging_mb", "host_staging_device_mb"))


def text_phase_capacity(profile: dict, *, unified_memory: bool) -> float:
    """Known capacity contributions; check completeness before admitting a job."""
    peak = profile["peak_mb_estimate"]
    host, target = profile.get("host_staging_mb"), profile.get("host_staging_device_mb")
    if unified_memory and host is not None and target is not None:
        peak = max(peak, profile["weights_mb"] + host + target)
    return peak


def cached_text_phase_capacity(profile: dict, *, unified_memory: bool) -> float:
    peak = profile["cached_condition_workspace_mb_estimate"]
    if unified_memory and text_capacity_complete(profile, unified_memory=True, cached_text=True):
        peak = max(peak, profile["cached_host_staging_mb"] + profile["cached_host_staging_device_mb"])
    return peak


def preview_residency(items: list[dict], profile: Callable[[dict], dict | None],
                      latent: LatentSpec, *, cached_text: bool, unified_memory: bool = False) -> list[dict]:
    """Follow the preview loop's condition, latent and decoded-image references.

    The previous prediction closures survive both new text encodes. They are
    replaced before sampling; only the old latent and pixels remain then. The
    sampled latent replaces its predecessor before decoding, while the old
    pixels survive until the next decode returns. Model weights and each
    sampler/decoder's own workspace are supplied by the caller.
    """
    stages = []
    previous_conditions = previous_latent = previous_pixels = 0.0
    for item in items:
        positive, negative = (profile(item[key]) for key in ("positive", "negative"))
        output_key = "cached_condition_outputs_mb" if cached_text else "condition_outputs_mb"
        positive_output = positive[output_key] if positive is not None else 0.0
        negative_output = negative[output_key] if negative is not None else 0.0
        peak_key = "cached_condition_workspace_mb_estimate" if cached_text else "peak_mb_estimate"
        positive_peak = positive[peak_key] if positive is not None else 0.0
        negative_peak = negative[peak_key] if negative is not None else 0.0
        capacity = cached_text_phase_capacity if cached_text else text_phase_capacity
        positive_capacity, negative_capacity = (
            capacity(current, unified_memory=unified_memory) if current is not None else 0.0
            for current in (positive, negative)
        )
        width, height = item["width"], item["height"]
        latent_mb = latent.channels * (width // latent.stride) * (height // latent.stride) * 4 / 2**20
        pixels_mb = 3 * width * height * 4 / 2**20
        stages.append({
            "width": width, "height": height,
            "text_peak_mb": previous_conditions + previous_latent + previous_pixels
            + max(positive_peak, positive_output + negative_peak),
            "text_capacity_mb": previous_conditions + previous_latent + previous_pixels
            + max(positive_capacity, positive_output + negative_capacity),
            "text_capacity_complete": all(text_capacity_complete(
                current, unified_memory=unified_memory, cached_text=cached_text,
            ) for current in (positive, negative)),
            "text_host_peak_mb": max(
                (current.get(("cached_" if cached_text else "") + "host_staging_mb") or 0.0)
                for current in (positive, negative) if current is not None
            ) if positive is not None or negative is not None else 0.0,
            "text_host_complete": all(
                current is not None and current.get(("cached_" if cached_text else "") + "host_staging_mb") is not None
                for current in (positive, negative)
            ),
            "previous_latent_mb": previous_latent,
            "conditions_mb": positive_output + negative_output,
            "sampling_retained_mb": positive_output + negative_output + previous_latent + previous_pixels,
            "decoding_retained_mb": positive_output + negative_output + previous_pixels,
            "previous_image": bool(stages),
        })
        previous_conditions = positive_output + negative_output
        previous_latent, previous_pixels = latent_mb, pixels_mb
    return stages


def preview_working_memory(
    cfg, backbone, training_state, items: list[dict], stages: list[dict], *,
    dtype, device_type: str, backbone_workspace: Callable[[int, int, int, int], int],
    export_layout: tuple | None = None,
) -> dict:
    """Approximate preview workspace from layouts and scalar prompt dimensions.

    Excludes resident model/optimizer/evaluation weights and text/VAE workspace,
    which the caller combines separately. Saved factors are counted once across
    prompts; device caching assumes all factors fit. Actual caching is bounded
    by half the free memory at first use. No model or sampler is executed here.
    """
    import torch

    from ypuddin.adapters.components import ComponentAdapterSet
    from ypuddin.adapters.dora import auto_merge_dtype
    from ypuddin.adapters.inject import AdapterSet
    from ypuddin.adapters.io import SAVE_DTYPES
    from ypuddin.models import get_family
    from ypuddin.sampling.adapter_preview import exported_adapter_bindings

    from .evaluation_memory import _export_layout

    if len(items) != len(stages):
        raise ValueError("preview items and residency stages must match")
    device_type = torch.device(device_type).type
    if device_type not in {"cpu", "cuda", "mps"}:
        raise ValueError("preview workspace requires cpu, cuda or mps placement")
    mb = 2**20
    result = {
        "method": "preview_scalar_workspace_approximation", "rows": [],
        "allocated_additional_mb": 0.0, "host_additional_mb": 0.0, "shared_additional_mb": 0.0,
        "snapshot_host_mb": 0.0, "factor_cache_device_mb": 0.0, "fusion_workspace_mb": 0.0,
        "factor_cache_assumption": "all_exported_factors_fit" if device_type != "cpu" else "cpu_snapshot_reused",
        "scope": "tensor_layout_approximation_excluding_kernel_workspace_and_allocator_reserve",
    }
    if not items:
        return result
    family = get_family(cfg.model.family)
    latent = family.spec.latent
    compute_dtype = torch.float32 if device_type == "cpu" else dtype
    exported_preview = cfg.sampling.noise == "comfyui" and isinstance(training_state, (AdapterSet, ComponentAdapterSet))
    snapshot = cache = fusion = export_device = export_host = 0
    layout_unknown = []
    bindings = exported_adapter_bindings(training_state) if isinstance(training_state, (AdapterSet, ComponentAdapterSet)) else {}
    grouped = {}
    if exported_preview:
        records, _, peak, unknown = (
            export_layout if export_layout is not None else _export_layout(training_state, device_type)
        )
        saved_dtype = SAVE_DTYPES[cfg.checkpoint.save_dtype]
        for record in records:
            elements = math.prod(record["shape"])
            size = elements * (4 if record["key"].endswith(".alpha") else saved_dtype.itemsize)
            snapshot += size
            grouped.setdefault(record["key"].split(".", 1)[0], []).append((record, elements))
        cache = snapshot if device_type != "cpu" else 0
        export_device, export_host = peak["device"], peak["host"] + snapshot
        layout_unknown = list(dict.fromkeys(unknown["device"] + unknown["host"]))
    for prefix, layer in bindings.items():
        if not exported_preview and layer.mode != "merged":
            continue
        selected = cfg.sampling.adapter_merge_dtype
        if exported_preview:
            merge = (SAVE_DTYPES[selected] if selected != "auto" else
                     layer.dora.merge_dtype if layer.dora is not None and layer.dora.compute_mode == "comfyui" else
                     auto_merge_dtype(layer.base, compute_dtype))
        else:
            merge = layer.dora.merge_dtype if layer.dora is not None and layer.dora.compute_mode == "comfyui" else torch.float32
        n = layer.weight.numel()
        # Delta, merged weight and scaled delta can coexist. DoRA also keeps
        # its normalized result; non-unit strength adds an interpolation result.
        dense = 3 + int(layer.dora is not None) + int(layer.dora is not None and layer.multiplier != 1.0)
        workspace = dense * n * merge.itemsize
        workspace += n * merge.itemsize if layer.weight.dtype != merge else 0
        workspace += n * compute_dtype.itemsize if merge != compute_dtype else 0
        if exported_preview and saved_dtype != merge:
            workspace += sum(elements * merge.itemsize for record, elements in grouped.get(prefix, ())
                             if not record["key"].endswith(".alpha"))
        if layer.bias is not None:
            workspace += 3 * layer.bias.numel() * max(merge.itemsize, compute_dtype.itemsize)
        if layer.dora is not None:
            workspace += 3 * layer.dora.dora_scale.numel() * merge.itemsize
        fusion = max(fusion, workspace)

    def placement(device, host):
        return (device + host, 0) if device_type == "cpu" else (device, host)

    def observe(device, host):
        result["allocated_additional_mb"] = max(result["allocated_additional_mb"], device / mb)
        result["host_additional_mb"] = max(result["host_additional_mb"], host / mb)
        result["shared_additional_mb"] = max(result["shared_additional_mb"], (device + host) / mb)

    observe(*placement(export_device, export_host))
    context_device, context_host = placement(cache, snapshot)
    previous_array = 0
    workspace_cache = {}
    for item, stage in zip(items, stages, strict=True):
        width, height = int(item["width"]), int(item["height"])
        if width <= 0 or height <= 0:
            raise ValueError("preview dimensions must be positive")
        defaults = family.spec.sampling
        cfg_scale = item.get("prompt_cfg")
        cfg_scale = cfg.sampling.cfg if cfg_scale is None else cfg_scale
        cfg_scale = defaults.cfg if cfg_scale is None else cfg_scale
        steps = item.get("prompt_steps") or cfg.sampling.steps or defaults.steps
        conditional = not math.isclose(cfg_scale, 1.0)
        prepared = cfg.model.family == "anima" and cfg.sampling.noise == "comfyui" and conditional
        lengths = []
        for condition in (item["positive"], item["negative"]):
            if cfg.model.family == "anima":
                lengths.append(int(condition.get("t5_padded_tokens", max(512, condition.get("t5_tokens", 1024)))))
            elif cfg.model.family == "sdxl":
                prompt = condition.get("prompt_spec", {})
                lengths.append(int(min(prompt.get("chunks", (1, 1))) * prompt.get("context_length", cfg.model.sdxl_max_token_length + 2)))
            else:
                lengths.append(family.spec.text.max_len)
        batch = 2 if prepared and lengths[0] == lengths[1] else 1
        key = (width, height, batch, max(lengths))
        if key not in workspace_cache:
            workspace_cache[key] = max(0, int(backbone_workspace(*key)))
        backbone_bytes = workspace_cache[key]
        latent_bytes = latent.channels * (width // latent.stride) * (height // latent.stride) * 4
        pixels = 3 * width * height * 4
        array = 3 * width * height
        prepared_bytes = (sum(lengths) * int(getattr(backbone, "model_channels", 2048)) * compute_dtype.itemsize
                          if prepared else 0)
        joined_context = prepared_bytes if batch == 2 else 0
        # Named latent-sized terms: x, prediction, denoised/direction, update
        # temporaries, CFG pair/combination, and the sampler's bounded history.
        latent_buffers = 6 + (3 if conditional else 0) + (2 if batch == 2 else 0)
        sampler = cfg.sampling.sampler
        if sampler == "heun":
            latent_buffers += 3 if steps > 1 else 0
        elif sampler == "euler_ancestral":
            latent_buffers += 2
        elif sampler == "er_sde":
            latent_buffers += min(max(0, steps - 1), cfg.sampling.er_sde_order - 1) + 3
            latent_buffers += 2 if cfg.sampling.er_sde_s_noise > 0 else 0
        elif sampler != "euler":
            raise ValueError(f"unsupported preview sampler: {sampler}")
        sampler_bytes = latent_buffers * latent_bytes + 32 * (steps + 1)
        row_device, row_host = placement(cache, snapshot + previous_array)
        sampling_device = row_device + stage["sampling_retained_mb"] * mb + prepared_bytes + joined_context
        sampling_device += backbone_bytes + sampler_bytes + fusion
        # Noise starts on CPU before transfer; schedules also have CPU copies.
        noise_device, noise_host = placement(0, latent_bytes + 32 * (steps + 1))
        sampling_device += noise_device
        sampling_host = row_host + noise_host
        # FP32 pixels stay live during NumPy conversion. Adjacent array
        # expressions overlap; PIL may hold another uint8 image while saving.
        conversion_device = context_device + stage["conditions_mb"] * mb + latent_bytes + pixels
        conversion_host = context_host + previous_array + (2 if device_type == "cpu" else 3) * pixels + 2 * array
        conversion_device, conversion_host = placement(conversion_device, conversion_host)
        clamp_device = context_device + stage["decoding_retained_mb"] * mb + latent_bytes + 2 * pixels
        clamp_host = context_host + previous_array
        clamp_device, clamp_host = placement(clamp_device, clamp_host)
        row = {"width": width, "height": height, "cfg_batch": batch, "text_tokens": max(lengths),
               "backbone_workspace_mb": backbone_bytes / mb, "sampler_workspace_mb": sampler_bytes / mb,
               "prepared_condition_mb": (prepared_bytes + joined_context) / mb,
               "text_context_device_mb": row_device / mb, "text_context_host_mb": row_host / mb,
               "text_context_shared_mb": (row_device + row_host) / mb,
               "decode_context_device_mb": row_device / mb, "decode_context_host_mb": row_host / mb,
               "decode_context_shared_mb": (row_device + row_host) / mb}
        for name, device, host in (("sampling", sampling_device, sampling_host),
                                   ("image_conversion", conversion_device, conversion_host),
                                   ("pixel_clamp", clamp_device, clamp_host)):
            row[f"{name}_additional_mb"] = device / mb
            row[f"{name}_host_additional_mb"] = host / mb
            row[f"{name}_shared_additional_mb"] = (device + host) / mb
            observe(device, host)
        result["rows"].append(row)
        previous_array = array
    result.update(snapshot_host_mb=snapshot / mb, factor_cache_device_mb=cache / mb,
                  fusion_workspace_mb=fusion / mb, export_layout_unknowns=layout_unknown,
                  workspace_geometry_count=len(workspace_cache))
    return result
