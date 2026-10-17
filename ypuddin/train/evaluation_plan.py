"""Place evaluation and checkpoint tensors beside the caller's residents."""

from __future__ import annotations

from .evaluation_memory import _export_layout, evaluation_export_memory, optimizer_evaluation_memory
from .training_modes import FullTrainingSet


def evaluation_operations(training_state, cfg, optimizer, *, device_type):
    records = optimizer["state_storage_records"]
    storage = ([] if device_type == "cpu" or cfg.optimizer.cpu_offload else
               None if records is None else [record["bytes"] for record in records
                                            if record["placement"] == "device"])
    operations = ["save_state"]
    checkpoint = cfg.checkpoint
    # A plan draft with an invalid checkpoint section has no export settings to size; it is left unestimated.
    if checkpoint is not None and (checkpoint.save_on_finish or checkpoint.save_every_steps_enabled
                                   or checkpoint.save_every_epochs_enabled):
        operations.append("save_weights")
    if cfg.loop.ema:
        # Existing EMA + one update transfer dominates the per-key initialization
        # clone while its destination dictionary is still being built.
        operations.append("ema_update")
    context_active = (
        not isinstance(training_state, FullTrainingSet) or "save_weights" in operations or cfg.loop.ema
        or (cfg.sampling.enabled and (cfg.sampling.at_start or cfg.sampling.every_steps_enabled
                                      or cfg.sampling.every_epochs_enabled))
        or (cfg.validation.enabled and (cfg.validation.every_steps_enabled or cfg.validation.every_epochs_enabled))
    )
    export_layout = _export_layout(training_state, device_type)
    results = [evaluation_export_memory(
        training_state, cfg, operation=operation, meta_device_type=device_type,
        optimizer_device_storage_bytes=storage,
        optimizer_state_clone_storage_bytes=optimizer["kahan_storage_bytes"],
        export_layout=export_layout,
    ) for operation in operations]
    return {
        "context_active": context_active,
        "export_layout": export_layout,
        "mode": optimizer_evaluation_memory(training_state, cfg.optimizer, meta_device_type=device_type),
        "stages": [dict(stage, operation=result["operation"]) for result in results for stage in result["stages"]],
        "non_tensor_unknowns": sorted({reason for result in results for reason in result["non_tensor_unknowns"]}
                                      | ({"checkpoint_settings"} if checkpoint is None else set())),
    }


def project_evaluation_stages(stages, *, resident_mb, host_resident_mb, device_type, processes=1):
    """Host tensor copies consume CPU allocation or MPS shared capacity, not CUDA allocation."""
    allocated = resident_mb
    capacity = resident_mb + (host_resident_mb if device_type == "mps" else 0.0)
    host = host_resident_mb
    allocated_unknown, capacity_unknown, host_unknown = [], [], []
    for row in stages:
        device_bytes, host_bytes = row["additional_device_bytes"], row["additional_host_bytes"]
        device_extra = row["known_additional_device_bytes"] / 2**20
        # Exports, validation and previews run on rank zero; EMA runs on every rank.
        copies = processes if row["operation"] in {"ema_update", "ema_initialize"} else 1
        host_extra = row["known_additional_host_bytes"] / 2**20 * copies
        live = resident_mb + device_extra + (host_extra if device_type == "cpu" else 0.0)
        allocated = max(allocated, live)
        capacity = max(capacity, live + (host_resident_mb + host_extra if device_type == "mps" else 0.0))
        host = max(host, host_resident_mb + host_extra)
        scope = row["operation"] + "." + row["name"] + ":"
        device_reasons = [scope + reason for reason in row["unknown_device_components"] or ["device_storage"]]
        host_reasons = [scope + reason for reason in row["unknown_host_components"] or ["host_storage"]]
        if device_bytes is None:
            allocated_unknown.extend(device_reasons)
            capacity_unknown.extend(device_reasons)
        if host_bytes is None:
            host_unknown.extend(host_reasons)
            if device_type == "cpu":
                allocated_unknown.extend(host_reasons)
            if device_type in {"cpu", "mps"}:
                capacity_unknown.extend(host_reasons)
    return {
        "known_allocated_mb": allocated, "known_capacity_mb": capacity, "known_host_mb": host,
        "allocated_unknowns": sorted(set(allocated_unknown)),
        "capacity_unknowns": sorted(set(capacity_unknown)), "host_unknowns": sorted(set(host_unknown)),
    }


def evaluation_mode_stages(mode, snapshot):
    """Mode hooks retain the raw snapshot; restore copies are a later stage."""
    rows = []
    for name in ("eval", "train"):
        phase = mode[name]
        row = {"name": name, "operation": "optimizer_mode"}
        for place in ("device", "host"):
            workspace = phase[f"{place}_workspace_bytes"]
            value = snapshot[f"snapshot_{place}_bytes"] + (workspace or 0)
            row[f"known_additional_{place}_bytes"] = value
            row[f"additional_{place}_bytes"] = None if workspace is None else value
            row[f"unknown_{place}_components"] = phase["unknown_components"] if workspace is None else []
        rows.append(row)
    return rows
