"""Frozen text-encoder residency and the tensors used by its encoding phase.

The decoder dimensions come from the same local configs/headers as the loaders. Cached, online and
preview encoding use the same calculation with their respective batch and token counts; callers add
only the other models/states resident during that phase. Allocator and device-runtime allowances are
separate from these tensor bytes.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ypuddin.config import TrainConfig

_MIB = 2**20


def qwen_parameter_elements(config: dict, *, causal_lm: bool) -> int:
    """Qwen3 and Qwen3-VL text decoder parameters, including tied embeddings only once."""
    hidden, intermediate = int(config["hidden_size"]), int(config["intermediate_size"])
    heads, kv_heads = int(config["num_attention_heads"]), int(config["num_key_value_heads"])
    head = int(config.get("head_dim", hidden // heads))
    q, kv = heads * head, kv_heads * head
    per_layer = hidden * (2 * q + 2 * kv) + 3 * hidden * intermediate + 2 * hidden + 2 * head
    if config.get("attention_bias", False):
        per_layer += q + 2 * kv + hidden
    embedding = int(config["vocab_size"]) * hidden
    return (
        embedding + int(config["num_hidden_layers"]) * per_layer + hidden
        + (embedding if causal_lm and not config.get("tie_word_embeddings", False) else 0)
    )


def clip_parameter_elements(config: dict, *, projection: bool) -> int:
    """CLIP text transformer parameters, including dense biases and both LayerNorm parameters."""
    hidden, intermediate = int(config["hidden_size"]), int(config["intermediate_size"])
    per_layer = 4 * hidden * hidden + 2 * hidden * intermediate + 9 * hidden + intermediate
    return (
        (int(config["vocab_size"]) + int(config["max_position_embeddings"])) * hidden
        + int(config["num_hidden_layers"]) * per_layer + 2 * hidden
        + (hidden * int(config["projection_dim"]) if projection else 0)
    )


def _qwen_geometry(cfg: TrainConfig) -> tuple[dict, str, int]:
    family = cfg.model.family
    if family == "krea2":
        from ypuddin.models.krea2.text import text_config_for

        config = text_config_for(cfg.model.text_encoder_path or "").to_dict()
        selected = min(12, int(config["num_hidden_layers"]))
        if cfg.model.dit_path and Path(cfg.model.dit_path).expanduser().is_file():
            from ypuddin.models.krea2.family import _config_from_header

            selected = _config_from_header(Path(cfg.model.dit_path).expanduser()).txtlayers
        return config, "qwen3_vl_text", selected
    if family == "flux2":
        from ypuddin.models.flux2.loading import component
        from ypuddin.models.flux2.single_text import ASSETS, text_config

        if cfg.model.text_encoder_path or cfg.model.dit_path:
            path = component(cfg.model.dit_path or ".", "text_encoder", cfg.model.text_encoder_path)
            config = text_config(path)
        else:
            variant = "8b" if cfg.model.flux2_variant == "klein-base-9b" else "4b"
            config = json.loads((ASSETS / f"qwen3-{variant}.json").read_text(encoding="utf-8"))
        return config, "qwen3_causal_lm", 3
    from ypuddin.models.anima.text import ASSETS

    path = Path(cfg.model.text_encoder_path or "").expanduser()
    location = (path if path.is_dir() else path.parent) / "config.json"
    if not cfg.model.text_encoder_path or not location.is_file():
        location = ASSETS / "qwen3_06b" / "config.json"
    return json.loads(location.read_text(encoding="utf-8")), "qwen3_decoder", 0


def _attention_workspace(batch: int, tokens: int, heads: int, element: int, backend: str) -> int:
    if backend == "fused":
        return 0
    # Math SDPA materializes scores and FP32 softmax; a half-precision result is also retained.
    scores = batch * heads * tokens * tokens
    return scores * (8 + (element if element < 4 else 0)) + batch * tokens * tokens * 4


def _qwen_tensors(
    config: dict, kind: str, selected: int, batch: int, tokens: int, element: int, backend: str,
    linear_compute_element: int | None = None,
) -> dict:
    hidden, intermediate = int(config["hidden_size"]), int(config["intermediate_size"])
    layers, heads = int(config["num_hidden_layers"]), int(config["num_attention_heads"])
    head = int(config.get("head_dim", hidden // heads))
    q, kv = heads * head, int(config["num_key_value_heads"]) * head
    state = batch * tokens * hidden * element
    all_hidden = kind != "qwen3_decoder"
    hidden_output = state * (layers + 1 if all_hidden else 1)
    logits = batch * tokens * int(config["vocab_size"]) * element if kind == "qwen3_causal_lm" else 0
    # Klein stacks then permutes/reshapes selected states into a second contiguous buffer.
    stack = state * selected * (2 if kind == "qwen3_causal_lm" else 1)
    if kind == "qwen3_decoder":
        # Anima's cache encoder converts the output to FP32 before copying it to CPU.
        stack = batch * tokens * hidden * 4 if element < 4 else 0
    elif kind == "qwen3_vl_text":
        # Krea copies one image's selected valid states at a time, casting to BF16 for its cache.
        stack += tokens * hidden * selected * (element + (2 if element != 2 else 0))
    mlp = batch * tokens * (2 * hidden + 3 * intermediate) * element
    attention = batch * tokens * (3 * hidden + 2 * q + 2 * kv) * element
    attention += _attention_workspace(batch, tokens, heads, element, backend)
    # RMSNorm computes variance in FP32; the residual and original dtype input remain alive.
    norm = batch * tokens * (hidden * (2 * element + 8) + 4)
    block = max(mlp, attention, norm)
    linear_cast_workspace = lm_cast_workspace = 0
    if linear_compute_element is not None and linear_compute_element > element:
        # The DTK Klein recipe casts one linear's input/weight and result, then restores BF16.
        contractions = ((hidden, q), (hidden, kv), (q, hidden), (hidden, intermediate), (intermediate, hidden))
        linear_cast_workspace = max(
            max(
                (batch * tokens * (input_dim + output_dim) + input_dim * output_dim) * linear_compute_element,
                batch * tokens * output_dim * (linear_compute_element + element),
            )
            for input_dim, output_dim in contractions
        )
        block = max(block, linear_cast_workspace + state)
        if kind == "qwen3_causal_lm":
            vocab = int(config["vocab_size"])
            lm_cast_workspace = max(
                (batch * tokens * (hidden + vocab) + hidden * vocab) * linear_compute_element,
                batch * tokens * vocab * (linear_compute_element + element),
            )
    retained_before_last = (layers - 1) * state if all_hidden else 0
    peak = max(retained_before_last + block, hidden_output + logits + stack, hidden_output + lm_cast_workspace)
    return {
        "parameter_elements": qwen_parameter_elements(config, causal_lm=kind == "qwen3_causal_lm"),
        "buffer_bytes": head * 4,
        "hidden_output_bytes": hidden_output, "logits_bytes": logits, "selection_workspace_bytes": stack,
        "block_workspace_bytes_estimate": block, "workspace_bytes_estimate": peak,
        "linear_cast_workspace_bytes_estimate": max(linear_cast_workspace, lm_cast_workspace),
        "tokens": tokens, "batch_size": batch, "hidden_size": hidden, "layers": layers,
        "vocab_size": int(config["vocab_size"]), "kind": kind,
    }


def _sdxl_prompt_tensors(configs: list[dict], spec: dict, batch: int, element: int, backend: str) -> tuple:
    mode = spec["mode"]
    if mode not in {"comfyui", "a1111"} or len(spec["chunks"]) != 2 or len(spec["weighted"]) != 2:
        raise ValueError("SDXL preview token geometry is invalid")
    chunks = [int(value) for value in spec["chunks"]]
    if min(chunks) < 1 or (mode == "a1111" and chunks[0] != chunks[1]):
        raise ValueError("SDXL preview chunks do not match the sampling mode")
    components = []
    previous_outputs = previous_conditions = previous_transforms = 0
    workspace = 0
    for index, config in enumerate(configs):
        hidden, layers = int(config["hidden_size"]), int(config["num_hidden_layers"])
        tokens = int(config["max_position_embeddings"])
        if tokens != int(spec["context_length"]):
            raise ValueError("SDXL preview context does not match the CLIP configuration")
        baseline = int(mode == "comfyui" and spec["weighted"][index])
        effective_batch = batch * chunks[index] + baseline
        state = effective_batch * tokens * hidden * element
        outputs = state * (layers + 1)
        attention = 7 * state + _attention_workspace(
            effective_batch, tokens, int(config["num_attention_heads"]), element, backend,
        )
        mlp = effective_batch * tokens * (2 * hidden + 2 * int(config["intermediate_size"])) * element
        encoder_peak = max((layers - 1) * state + max(attention, mlp), outputs + 2 * state)
        # The previous output tuple is still live while the next encoder's RHS runs.
        workspace = max(workspace, previous_outputs + previous_conditions + previous_transforms + encoder_peak)
        condition = batch * chunks[index] * tokens * hidden * 4
        cast = effective_batch * tokens * hidden * 4 if mode == "comfyui" and element < 4 else 0
        transforms = 2 * condition if mode == "a1111" or baseline else 0
        inputs = effective_batch * tokens * 8 + batch * chunks[index] * tokens * (4 + int(bool(baseline)))
        pooled = effective_batch * int(config.get("projection_dim", hidden)) * element if index == 1 else 0
        workspace = max(workspace, previous_conditions + previous_transforms + outputs + cast + transforms + inputs + pooled)
        # Weighted intermediates remain local variables until the next iteration replaces them.
        previous_transforms = cast + transforms
        previous_conditions += condition
        previous_outputs = outputs
        components.append({"kind": "clip_prompt", "parameter_elements": clip_parameter_elements(config, projection=index == 1),
                           "buffer_bytes": tokens * 8, "batch_size": effective_batch, "tokens": tokens,
                           "hidden_size": hidden, "layers": layers, "chunks": chunks[index],
                           "baseline_batch_size": baseline, "hidden_output_bytes": outputs, "logits_bytes": 0,
                           "workspace_bytes_estimate": encoder_peak})
    combined = batch * min(chunks) * int(spec["context_length"]) * sum(int(c["hidden_size"]) for c in configs) * 4
    pooled = batch * int(configs[1]["projection_dim"]) * 4
    workspace = max(workspace, previous_outputs + previous_conditions + previous_transforms + combined + pooled)
    return components, workspace, combined + pooled


def _cached_condition_phase(condition: int, entries: int, *, device_type: str | None,
                            view_only: bool, basis: str) -> dict:
    assembled = entries if view_only else entries + condition
    return {
        "cached_condition_workspace_mb_estimate": (assembled if device_type == "cpu" else condition) / _MIB,
        "cached_host_staging_mb": 0.0 if device_type == "cpu" else assembled / _MIB,
        "cached_host_staging_device_mb": 0.0 if device_type == "cpu" else condition / _MIB,
        "cached_condition_assembly_basis": basis,
    }


def _flux3_text_phase(cfg, *, batch, device_type, token_length, backend, include_condition):
    from safetensors import SafetensorError

    from ypuddin.models.flux3.loading import component_paths
    from ypuddin.models.flux3.text import Flux3Text

    tokens = Flux3Text.max_len if token_length is None else token_length
    if (isinstance(tokens, bool) or not isinstance(tokens, int) or not 0 < tokens <= Flux3Text.max_len
            or (tokens != Flux3Text.max_len and tokens % Flux3Text.pad_multiple)):
        raise ValueError("FLUX.3 text length must be a padded multiple of 80 or the 4096-token limit")
    try:
        _, _, text, _, processor = component_paths(cfg.model)
        full_config, elements = Flux3Text.validate_paths(text, processor)
        config = full_config["text_config"]
        if any(not isinstance(config[key], int) or config[key] <= 0 for key in (
            "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "vocab_size",
        )):
            return None
        # The complete VL model stays resident, but captions run sequentially with B=1.
        part = _qwen_tensors(config, "qwen3_causal_lm", len(Flux3Text.layers), 1, tokens, 2, backend)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, SafetensorError):
        return None
    part.update(kind="qwen3_vl_causal_lm", parameter_elements=elements)
    entry = tokens * int(config["hidden_size"]) * len(Flux3Text.layers) * 2
    entries = batch * entry
    condition = entries + batch * 8  # Padded BF16 embeddings and int64 per-caption lengths.
    call_workspace = part["workspace_bytes_estimate"]
    previous = (batch - 1) * entry if include_condition or device_type == "cpu" else 0
    workspace = previous + call_workspace
    if include_condition:
        workspace = max(workspace, entries + condition)
    else:
        workspace = max(workspace, entries if device_type == "cpu" else entry)
    # Cache creation copies each completed caption to CPU before encoding the next.
    host = entries if not include_condition and device_type != "cpu" else 0
    host_device = call_workspace if host else 0
    # Cached entries are read on CPU; cond_from_cache builds the destination directly.
    cached_workspace = entries + condition if device_type == "cpu" else condition + entry
    basis = "token_limit_upper_bound" if token_length is None else "batch_maximum_padded_length"
    weights = elements * 2 + part["buffer_bytes"]
    return {
        "weights_mb": weights / _MIB,
        "retained_outputs_mb": (previous + part["hidden_output_bytes"] + part["logits_bytes"]) / _MIB,
        "condition_outputs_mb": condition / _MIB, "cached_condition_outputs_mb": condition / _MIB,
        "condition_outputs_basis": basis,
        "host_staging_mb": host / _MIB, "host_staging_device_mb": host_device / _MIB,
        "cached_condition_workspace_mb_estimate": cached_workspace / _MIB,
        "cached_host_staging_mb": 0.0 if device_type == "cpu" else entries / _MIB,
        "cached_host_staging_device_mb": 0.0 if device_type == "cpu" else cached_workspace / _MIB,
        "cached_condition_assembly_basis": basis,
        "workspace_mb_estimate": workspace / _MIB, "peak_mb_estimate": (weights + workspace) / _MIB,
        "batch_size": batch, "dtype": "bfloat16", "attention_backend": backend,
        "length_basis": "caller_padded_length" if token_length is not None else "runtime_truncation_limit",
        "components": [part],
    }


def frozen_text_phase(
    cfg: TrainConfig,
    dtype: torch.dtype,
    *,
    batch_size: int = 16,
    device_type: str | None = None,
    token_length: int | None = None,
    t5_token_length: int | None = None,
    attention_backend: str | None = None,
    linear_compute_dtype: torch.dtype | None = None,
    prompt_spec: dict | None = None,
    include_condition: bool = True,
    cached_condition_kind: str = "training",
) -> dict | None:
    """Text-only peak in MiB; excludes backbone/VAE, optimizer, allocator and driver residency.

    This models frozen encoding, not text-encoder backward. ``token_length`` is the actual padded
    encoder call length when known; otherwise the runtime's configured upper bound is used.
    SDXL previews pass their independent per-encoder chunks and weighting in ``prompt_spec``.
    ``include_condition=False`` stops at ``encode_for_cache`` without assembling a padded condition.
    Cached previews can use views where training batches stack their entries; their assembly
    fields exclude encoder weights and are selected by ``cached_condition_kind``.
    """
    family = cfg.model.family
    if family not in {"anima", "krea2", "sdxl", "flux2", "flux3"}:
        return None
    batch = max(1, int(batch_size))
    element = torch.empty((), dtype=dtype).element_size()
    backend = attention_backend or ("math" if device_type in {"cpu", "mps"} else "fused")
    if backend not in {"fused", "math"}:
        raise ValueError("text attention backend must be fused or math")
    if cached_condition_kind not in {"training", "preview"}:
        raise ValueError("cached condition kind must be training or preview")
    if family == "flux3":
        return _flux3_text_phase(cfg, batch=batch, device_type=device_type, token_length=token_length,
                                 backend=backend, include_condition=include_condition)
    host_staging = host_staging_device = 0
    if family != "sdxl":
        config, kind, selected = _qwen_geometry(cfg)
        tokens = int(token_length or (546 if family == "krea2" else 1024 if family == "anima" else 512))
        linear_element = torch.empty((), dtype=linear_compute_dtype).element_size() if linear_compute_dtype else None
        components = [_qwen_tensors(config, kind, selected, batch, tokens, element, backend, linear_element)]
        workspace = components[0]["workspace_bytes_estimate"]
        retained = components[0]["hidden_output_bytes"] + components[0]["logits_bytes"]
        hidden = int(config["hidden_size"])
        condition_basis = "tensor_shapes"
        if family == "anima":
            qwen_length = max(512, min(1024, tokens))
            t5_length = max(512, min(1024, t5_token_length or 1024))
            condition = batch * (qwen_length * (hidden * 4 + 1) + t5_length * 9)
            if token_length is None or t5_token_length is None:
                condition_basis = "token_limit_upper_bound"
            entries = batch * (tokens * hidden * 4 + min(1024, t5_token_length or 1024) * 8)
        elif family == "krea2":
            condition = batch * 512 * (selected * hidden * 2 + 1)
            condition_basis = "token_limit_upper_bound"
            entries = batch * 512 * selected * hidden * 2
        else:
            condition = batch * 512 * selected * hidden * element
            entries = condition
        cached_condition = condition
        if include_condition and family in {"anima", "krea2"}:
            workspace = max(workspace, condition + (entries if device_type == "cpu" else 0))
            if device_type != "cpu":
                host_staging = entries + condition
                host_staging_device = condition
        cached_basis = (
            "batch_maximum_token_lengths"
            if family == "anima" and batch > 1 and condition_basis == "tensor_shapes" else condition_basis
        )
    else:
        from ypuddin.models.sdxl.loading import component_config, component_path

        configs = [component_config(component_path(cfg.model.dit_path or ".", name, override), name)
                   for name, override in (("text_encoder", cfg.model.text_encoder_path),
                                          ("text_encoder_2", cfg.model.text_encoder_2_path))]
        if prompt_spec is not None:
            components, workspace, retained = _sdxl_prompt_tensors(configs, prompt_spec, batch, element, backend)
            weights = sum(part["parameter_elements"] * element + part["buffer_bytes"] for part in components)
            chunks = prompt_spec["chunks"]
            projection = int(configs[1]["projection_dim"])
            condition = batch * min(chunks) * int(prompt_spec["context_length"]) * sum(int(c["hidden_size"]) for c in configs) * 4
            pooled_element = 4 if prompt_spec["mode"] == "comfyui" else element
            cached_condition = condition + batch * projection * pooled_element
            if prompt_spec["mode"] == "comfyui" and element != 4:
                condition = cached_condition
            else:
                baseline = int(prompt_spec["mode"] == "comfyui" and prompt_spec["weighted"][1])
                condition += (batch * chunks[1] + baseline) * projection * element
            return {
                "weights_mb": weights / _MIB, "retained_outputs_mb": retained / _MIB,
                "condition_outputs_mb": condition / _MIB, "cached_condition_outputs_mb": cached_condition / _MIB,
                "condition_outputs_basis": "tensor_shapes",
                "host_staging_mb": None if host_staging is None else host_staging / _MIB,
                "host_staging_device_mb": None if host_staging_device is None else host_staging_device / _MIB,
                **_cached_condition_phase(cached_condition, cached_condition, device_type=device_type,
                                          view_only=cached_condition_kind == "preview", basis="tensor_shapes"),
                "workspace_mb_estimate": workspace / _MIB, "peak_mb_estimate": (weights + workspace) / _MIB,
                "batch_size": batch, "dtype": str(dtype).removeprefix("torch."), "attention_backend": backend,
                "length_basis": "local_preview_chunks", "components": components,
            }
        chunks = cfg.model.sdxl_max_token_length // 75
        components = []
        previous_hidden = 0
        workspace = retained = 0
        for index, config in enumerate(configs):
            hidden, layers = int(config["hidden_size"]), int(config["num_hidden_layers"])
            tokens = int(config["max_position_embeddings"])
            effective_batch = batch * chunks
            state = effective_batch * tokens * hidden * element
            outputs = state * (layers + 1)
            # RHS encoding completes before Python releases the previous encoder's output tuple.
            attention = 7 * state + _attention_workspace(
                effective_batch, tokens, int(config["num_attention_heads"]), element, backend
            )
            mlp = effective_batch * tokens * (2 * hidden + 2 * int(config["intermediate_size"])) * element
            encoder_peak = max((layers - 1) * state + max(attention, mlp), outputs + 2 * state)
            workspace = max(workspace, previous_hidden + encoder_peak)
            previous_hidden = outputs
            retained += outputs
            components.append({"kind": "clip", "parameter_elements": clip_parameter_elements(config, projection=index == 1),
                               "buffer_bytes": tokens * 8,
                               "batch_size": effective_batch, "tokens": tokens, "hidden_size": hidden, "layers": layers,
                               "hidden_output_bytes": outputs, "logits_bytes": 0, "workspace_bytes_estimate": encoder_peak})
        combined = batch * (cfg.model.sdxl_max_token_length + 2) * sum(int(c["hidden_size"]) for c in configs)
        # Weighted captions multiply the embeddings by FP32 weights before returning the condition.
        workspace = max(workspace, previous_hidden + combined * (4 + element))
        condition_element = 4 if cfg.dataset.caption.weighted and chunks == 1 else element
        condition = combined * condition_element + batch * chunks * int(configs[1]["projection_dim"]) * element
        cached_condition = combined * condition_element + batch * int(configs[1]["projection_dim"]) * element
        condition_basis = "tensor_shapes"
        entries = cached_condition
        cached_basis = condition_basis
    weights = sum(part["parameter_elements"] * element + part["buffer_bytes"] for part in components)
    return {
        "weights_mb": weights / _MIB, "retained_outputs_mb": retained / _MIB,
        "condition_outputs_mb": condition / _MIB, "cached_condition_outputs_mb": cached_condition / _MIB,
        "condition_outputs_basis": condition_basis,
        "host_staging_mb": None if host_staging is None else host_staging / _MIB,
        "host_staging_device_mb": None if host_staging_device is None else host_staging_device / _MIB,
        **_cached_condition_phase(cached_condition, entries, device_type=device_type,
                                  view_only=family == "sdxl" and cached_condition_kind == "preview", basis=cached_basis),
        "workspace_mb_estimate": workspace / _MIB, "peak_mb_estimate": (weights + workspace) / _MIB,
        "batch_size": batch, "dtype": str(dtype).removeprefix("torch."), "attention_backend": backend,
        "length_basis": "caller_padded_length" if token_length else "runtime_truncation_limit",
        "components": components,
    }
