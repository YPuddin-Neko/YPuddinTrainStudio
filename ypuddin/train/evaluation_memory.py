"""Storage copied while schedule-free weights are used for evaluation."""

from __future__ import annotations

from importlib import metadata
from typing import Any

import torch

from ypuddin.adapters.components import ComponentAdapterSet
from ypuddin.adapters.inject import AdapterSet
from ypuddin.config import OptimizerConfig, TrainConfig
from ypuddin.config.optimizer_rules import optimizer_key
from ypuddin.optim.factory import is_schedule_free

from .training_modes import FullTrainingSet


def _state_entries(training_state):
    if isinstance(training_state, FullTrainingSet):
        entries = {}
        for component, module in training_state.modules.items():
            for key, value in module.state_dict().items():
                native = f"{component}.{training_state._native_key(key)}"
                if native in entries:
                    raise ValueError(f"duplicate native training state key: {native}")
                entries[native] = value
        return entries, True, False
    if isinstance(training_state, ComponentAdapterSet):
        entries = {}
        for component, item in training_state.components.items():
            values, _, _ = _state_entries(item)
            entries.update({f"{component}.{key}": value for key, value in values.items()})
        return entries, False, True
    if isinstance(training_state, AdapterSet):
        entries = {}
        for name, layer in training_state.layers.items():
            for kind, module in (("adapter", layer.adapter), ("dora", layer.dora)):
                if module is not None:
                    entries.update({f"{name}.{kind}.{key}": value for key, value in module.state_dict().items()})
        return entries, False, False
    raise TypeError("unsupported training-state snapshot layout")


def evaluation_snapshot_memory(
    training_state,
    optimizer_cfg: OptimizerConfig,
    *,
    meta_device_type: str | None = None,
) -> dict[str, Any]:
    """Count snapshot tensors without cloning or moving the supplied state.

    Meta tensors describe only the supplied layout; their eventual device must
    be supplied for adapters. Full-model snapshots always copy to CPU. This
    counts state copies, not optimizer eval/train work or transfer workspaces.
    """
    result = {
        "enabled": is_schedule_free(optimizer_cfg),
        "layout_basis": "disabled",
        "layout_scope": "provided_state_dict",
        "runtime_layout_verified": False,
        "storage_records": [],
        "includes_optimizer_mode_workspace": False,
        "includes_transfer_workspace": False,
    }
    for stage in ("snapshot", "restore_workspace"):
        for placement in ("device", "host"):
            result[f"{stage}_{placement}_bytes"] = 0
            result[f"{stage}_{placement}_storage_bytes"] = []
    if not result["enabled"]:
        return result

    entries, copy_to_host, restore_copy = _state_entries(training_state)
    meta_target = torch.device(meta_device_type).type if meta_device_type is not None else None
    if meta_target == "meta":
        raise ValueError("meta snapshot placement needs a real device type")
    has_meta = False
    for key, value in entries.items():
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise TypeError(f"unsupported snapshot tensor layout: {key}")
        if value.is_meta:
            has_meta = True
            if not copy_to_host and meta_target is None:
                raise ValueError("meta adapter snapshots require meta_device_type")
        destination = "cpu" if copy_to_host else (meta_target if value.is_meta else value.device.type)
        placement = "host" if destination == "cpu" else "device"
        size = value.numel() * value.element_size()
        # training_state_dict copies every key, including tied state and buffers.
        result["storage_records"].append({
            "key": key,
            "bytes": size,
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "source_device_type": value.device.type,
            "snapshot_device_type": destination,
            "placement": placement,
            "restore_copy": restore_copy,
        })
        result[f"snapshot_{placement}_bytes"] += size
        result[f"snapshot_{placement}_storage_bytes"].append(size)
        if restore_copy:
            # ComponentAdapterSet keeps its expected snapshot beside the raw one.
            result[f"restore_workspace_{placement}_bytes"] += size
            result[f"restore_workspace_{placement}_storage_bytes"].append(size)
    result["layout_basis"] = "meta_state_dict" if has_meta else "materialized_state_dict"
    result["runtime_layout_verified"] = not has_meta
    return result


def optimizer_evaluation_memory(
    training_state,
    optimizer_cfg: OptimizerConfig,
    *,
    meta_device_type: str | None = None,
    state_colocated: bool = True,
) -> dict[str, Any]:
    """Count mode-switch tensor workspace, excluding snapshots and optimizer state.

    The supported implementations interpolate in place. Their ``z.to(p.device)``
    aliases z only when optimizer state remains on the parameter's device.
    """
    enabled = is_schedule_free(optimizer_cfg)
    key = optimizer_key(optimizer_cfg.type)
    packages = {"adamw_sf": ("schedulefree", "1.4.1"),
                "prodigy_plus_sf": ("prodigy-plus-schedule-free", "2.0.1")}
    package, version = packages.get(key, (None, None))
    installed = None
    if package is not None:
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    unknown = []
    if enabled and (version is None or installed != version):
        unknown.append("optimizer_mode_implementation")
    if enabled and not state_colocated:
        unknown.append("optimizer_z_placement")
    return {
        "enabled": enabled, "optimizer": key, "implementation_version": installed,
        "state_colocated": state_colocated, "includes_snapshot": False,
        "includes_optimizer_state": False, "scope": "tensor_storage",
        "eval": {"device_workspace_bytes": None if unknown else 0,
                 "host_workspace_bytes": None if unknown else 0, "unknown_components": list(unknown)},
        "train": {"device_workspace_bytes": None if unknown else 0,
                  "host_workspace_bytes": None if unknown else 0, "unknown_components": list(unknown)},
    }


def _placement(value, meta_device_type):
    target = torch.device(meta_device_type).type if meta_device_type is not None else None
    device = target if value.is_meta else value.device.type
    if device in (None, "meta"):
        raise ValueError("meta export layout requires a real meta_device_type")
    return "host" if device == "cpu" else "device"


def _export_layout(training_state, meta_device_type):
    """Describe copied export tensors and the peak while constructing them."""
    from ypuddin.adapters.full import Full
    from ypuddin.adapters.loha import LoHa
    from ypuddin.adapters.lokr import LoKr
    from ypuddin.adapters.lora import LoRA
    from ypuddin.adapters.ortho import OrthoLoRA
    from ypuddin.adapters.tlora import TLoRA

    records = []
    live = {"device": 0, "host": 0}
    peak = dict(live)
    unknown = {"device": [], "host": []}

    def record(key, value=None, *, shape=None, dtype=None, placement=None, component=None):
        shape = tuple(value.shape) if shape is None else tuple(shape)
        dtype = value.dtype if dtype is None else dtype
        placement = _placement(value, meta_device_type) if placement is None else placement
        size = dtype.itemsize
        for dimension in shape:
            size *= dimension
        return {"key": key, "bytes": size, "shape": shape, "dtype": str(dtype),
                "placement": placement, "component": component,
                "contiguous": True if value is None else value.is_contiguous()}

    if isinstance(training_state, FullTrainingSet):
        entries, _, _ = _state_entries(training_state)
        for key, value in entries.items():
            item = record(key, value, placement="host", component=key.split(".", 1)[0])
            # to(copy=True) preserves dense strides, but compacts sliced tensors.
            expected_stride = 1
            dense = True
            dimensions = zip(value.shape, value.stride(), strict=True)
            for stride, dimension in sorted((stride, size) for size, stride in dimensions if size > 1):
                if stride != expected_stride:
                    dense = False
                    break
                expected_stride *= dimension
            item["contiguous"] = value.is_contiguous() or not dense
            records.append(item)
            live["host"] += item["bytes"]
        return records, live, dict(live), unknown

    sets = training_state.components.values() if isinstance(training_state, ComponentAdapterSet) else (training_state,)
    for item in sets:
        if not isinstance(item, AdapterSet):
            raise TypeError("unsupported export training-state layout")
        for name, layer in item.layers.items():
            adapter = layer.adapter
            prefix = item.export_key(name)
            local = []
            extra = {"device": 0, "host": 0}
            tensor = next(adapter.parameters())
            where = _placement(tensor, meta_device_type)
            if any(not p.is_contiguous() for p in adapter.parameters()):
                unknown[where].append("noncontiguous_adapter_export")

            def add(suffix, value=None, *, local=local, prefix=prefix, **kwargs):
                local.append(record(f"{prefix}.{suffix}", value, **kwargs))

            if type(adapter) in (LoRA, LoHa, LoKr):
                if type(adapter) is LoRA:
                    pairs = [("lora_down.weight", adapter.down), ("lora_up.weight", adapter.up)]
                    scaled = adapter.up
                elif type(adapter) is LoHa:
                    pairs = [("hada_" + field, getattr(adapter, field))
                             for field in ("w1_a", "w1_b", "w2_a", "w2_b")]
                    scaled = adapter.w1_a
                else:
                    fields = ["w1_a", "w1_b"] if adapter.w1_lowrank else ["w1"]
                    fields += ["w2_a", "w2_b"] if adapter.w2_lowrank else ["w2"]
                    pairs = [("lokr_" + field, getattr(adapter, field)) for field in fields]
                    scaled = getattr(adapter, fields[0])
                for suffix, value in pairs:
                    shape = None
                    if type(adapter) is LoRA:
                        shape = ((adapter.rank, adapter.in_features, *adapter.kernel) if suffix.startswith("lora_down")
                                 else (adapter.out_features, adapter.rank, *((1,) * len(adapter.kernel))))
                    add(suffix, value, shape=shape)
                # The differentiable dictionary keeps the scaled factor and alpha
                # until the independent clones in export_tensors are complete.
                extra[_placement(scaled, meta_device_type)] += scaled.numel() * scaled.element_size()
                extra["host"] += 4
                add("alpha", shape=(), dtype=torch.float32, placement="host")
            elif type(adapter) is Full:
                add("diff", adapter.weight)
                weight = adapter.weight.numel() * adapter.weight.element_size()
                bias = 0
                if adapter.bias is not None:
                    add("diff_b", adapter.bias)
                    bias = adapter.bias.numel() * adapter.bias.element_size()
                # A difference and its clone coexist; the weight result remains
                # live while the subsequent bias difference is cloned.
                extra[where] = max(2 * weight, weight + 2 * bias) - weight - bias
                for field in ("weight", "bias"):
                    value = getattr(adapter, field)
                    base = getattr(adapter, "base_" + field)
                    if value is not None and (value.dtype != base.dtype or value.device != base.device):
                        unknown[where].append("full_adapter_base_placement_or_dtype")
            elif type(adapter) in (OrthoLoRA, TLoRA):
                rank = adapter.rank * (2 if type(adapter) is TLoRA and adapter.ortho else 1)
                dtype = torch.float32 if type(adapter) is OrthoLoRA or adapter.ortho else adapter.down.dtype
                add("lora_down.weight", shape=(rank, adapter.in_features, *adapter.kernel),
                    dtype=dtype, placement=where)
                add("lora_up.weight", shape=(adapter.out_features, rank, *((1,) * len(adapter.kernel))),
                    dtype=dtype, placement=where)
                add("alpha", shape=(), dtype=torch.float32, placement="host")
                if type(adapter) is TLoRA and not adapter.ortho:
                    up = adapter.up.numel() * adapter.up.element_size()
                    down = adapter.down.numel() * adapter.down.element_size()
                    alpha = 4 if where == "host" else 0
                    extra[where] = max(2 * up, up + down + alpha) - up - down - alpha
                else:
                    unknown[where].append("orthogonal_export_workspace")
            else:
                for place in unknown:
                    unknown[place].append("adapter_export_layout")
                continue
            subtotal = {place: sum(r["bytes"] for r in local if r["placement"] == place) for place in live}
            for place in live:
                peak[place] = max(peak[place], live[place] + subtotal[place] + extra[place])
                live[place] += subtotal[place]
            records.extend(local)
            if layer.dora is not None:
                value = layer.dora.dora_scale
                row = record(f"{prefix}.dora_scale", value)
                records.append(row)
                live[row["placement"]] += row["bytes"]
                peak[row["placement"]] = max(peak[row["placement"]], live[row["placement"]])
    return records, live, peak, unknown


def evaluation_export_memory(
    training_state,
    cfg: TrainConfig,
    *,
    operation: str,
    meta_device_type: str | None = None,
    state_colocated: bool = True,
    optimizer_device_storage_bytes: list[int] | tuple[int, ...] | None = None,
    optimizer_state_clone_storage_bytes: list[int] | tuple[int, ...] | None = None,
    export_layout: tuple | None = None,
) -> dict[str, Any]:
    """Count additional storage at the trainer's export boundaries.

    Stage totals include their raw snapshot and export tensors. Model weights,
    optimizer states, and an existing EMA dictionary are caller-owned residents.
    EMA initialization instead includes the new EMA dictionary in its workspace.
    Host serialization and device-kernel scratch are separate from tensor storage.
    Optimizer state clones are additional to resident state. Device serialization
    sizes include those clones when torch.save must transfer them to the host.
    """
    if operation not in {"save_weights", "save_state", "ema_update", "ema_initialize"}:
        raise ValueError(f"unsupported evaluation operation: {operation}")
    full = isinstance(training_state, FullTrainingSet)
    snapshot = evaluation_snapshot_memory(training_state, cfg.optimizer, meta_device_type=meta_device_type)
    mode = optimizer_evaluation_memory(training_state, cfg.optimizer, meta_device_type=meta_device_type,
                                       state_colocated=state_colocated)
    records, exported, build_peak, export_unknown = (
        export_layout if export_layout is not None else _export_layout(training_state, meta_device_type)
    )
    raw = {p: snapshot[f"snapshot_{p}_bytes"] for p in ("device", "host")}
    restore = {p: snapshot[f"restore_workspace_{p}_bytes"] for p in raw}
    enabled = snapshot["enabled"] and operation != "ema_initialize" and not (full and operation == "save_state")
    if not enabled:
        raw = {p: 0 for p in raw}
        restore = dict(raw)
    stages = []
    non_tensor_unknowns = []
    try:
        serialization_version = metadata.version("safetensors")
    except metadata.PackageNotFoundError:
        serialization_version = None
    serialization_unknown = [] if serialization_version == "0.8.0" else ["safetensors_tensor_serialization"]

    def stage(name, *, snap=False, export=False, workspace=None, unknown=None):
        workspace = workspace or {}
        unknown = unknown or {}
        row = {"name": name, "unknown_components": []}
        for place in raw:
            reasons = list(dict.fromkeys(unknown.get(place, [])))
            if export and "adapter_export_layout" in export_unknown[place]:
                reasons.append("adapter_export_layout")
            values = {"snapshot": raw[place] if snap else 0,
                      "export": exported[place] if export else 0,
                      "workspace": workspace.get(place, 0)}
            for part, size in values.items():
                row[f"{part}_{place}_bytes"] = size
            row[f"known_additional_{place}_bytes"] = sum(values.values())
            row[f"additional_{place}_bytes"] = None if reasons else sum(values.values())
            row[f"unknown_{place}_components"] = reasons
            row["unknown_components"].extend(reasons)
        stages.append(row)

    def mode_stage(which, retained):
        value = mode[which]
        stage("optimizer_" + which, snap=True, export=retained,
              workspace={p: value[f"{p}_workspace_bytes"] or 0 for p in raw},
              unknown={p: value["unknown_components"] for p in raw})

    if enabled:
        mode_stage("eval", False)
    stage("export", snap=enabled, workspace=build_peak, unknown=export_unknown)

    def conversion(rows, *, ema=False, preserve_dtype=False):
        """CPU conversion peak beyond source storage, including the loop local t."""
        accumulated = maximum = previous_local = 0
        previous_component = component_current = 0
        current_component = None
        for row in rows:
            n = row["bytes"] // getattr(torch, row["dtype"].removeprefix("torch.")).itemsize
            source_dtype = torch.float32 if ema else getattr(torch, row["dtype"].removeprefix("torch."))
            source_bytes = n * source_dtype.itemsize
            source_host = ema or row["placement"] == "host"
            target_dtype = source_dtype if preserve_dtype or (full and not source_dtype.is_floating_point) else (
                torch.float32 if row["key"].endswith(".alpha") else
                {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[cfg.checkpoint.save_dtype])
            target_bytes = n * target_dtype.itemsize
            same = target_dtype == source_dtype
            contiguous = row["contiguous"] or ema
            transfer = 0 if source_host and contiguous else source_bytes
            output = 0 if source_host and contiguous and same else target_bytes
            if full:
                # _cpu_tensors performs one combined to(), then contiguous().
                output = 0 if source_host and contiguous and same else target_bytes
                transient = target_bytes if not contiguous and not same else 0
                component = row["component"]
                if component != current_component:
                    previous_component, component_current = component_current, 0
                    current_component = component
                maximum = max(maximum, previous_component + component_current + output + transient)
                component_current += output
            else:
                maximum = max(maximum, accumulated + previous_local + transfer)
                maximum = max(maximum, accumulated + transfer + (target_bytes if not same else 0))
                accumulated += output
                previous_local = transfer if not same else 0
                maximum = max(maximum, accumulated + previous_local)
        return maximum

    if operation == "save_weights":
        converted = conversion(records)
        stage("save_weights_conversion", snap=enabled, export=True, workspace={"host": converted})
        stage("save_weights_serialization", snap=enabled, export=True,
              workspace={"host": converted}, unknown={"host": serialization_unknown})
        non_tensor_unknowns.append("host_file_serialization")
        if cfg.loop.ema:
            # Adapter exports remain in _save_weights while the EMA file is saved.
            converted_ema = conversion(records, ema=True)
            stage("save_ema_conversion", snap=enabled, export=not full,
                  workspace={"host": converted_ema})
            stage("save_ema_serialization", snap=enabled, export=not full,
                  workspace={"host": converted_ema}, unknown={"host": serialization_unknown})

    if enabled:
        retained = operation in {"ema_update", "save_state"}
        mode_stage("train", retained)
        stage("restore", snap=True, export=retained, workspace=restore)

    if operation in {"ema_update", "ema_initialize"}:
        current = 0
        workspace = {p: 0 for p in raw}
        for row in records:
            dtype = getattr(torch, row["dtype"].removeprefix("torch."))
            float_bytes = row["bytes"] // dtype.itemsize * 4
            cast = float_bytes if dtype != torch.float32 else 0
            if operation == "ema_update":
                workspace["device"] = max(workspace["device"], cast if row["placement"] == "device" else 0)
                workspace["host"] = max(workspace["host"], float_bytes if row["placement"] == "device" else cast)
            else:
                workspace["device"] = max(workspace["device"], cast if row["placement"] == "device" else 0)
                host_temporary = float_bytes if row["placement"] == "device" else cast
                workspace["host"] = max(workspace["host"], current + host_temporary + float_bytes)
                current += float_bytes
        stage(operation, export=True, workspace=workspace)

    if operation == "save_state":
        training = {p: 0 for p in raw}
        entries, _, _ = _state_entries(training_state)
        if full:
            stage("checkpoint_conversion", export=True, workspace={"host": conversion(records, preserve_dtype=True)})
        else:
            # Raw training state is collected after _evaluation has released raw.
            for value in entries.values():
                training[_placement(value, meta_device_type)] += value.numel() * value.element_size()
            stage("checkpoint_training_state", export=True, workspace=training)
            transfer = conversion(records, preserve_dtype=True)
            training_transfer = sum(value.numel() * value.element_size() for value in entries.values()
                                    if _placement(value, meta_device_type) == "device" or not value.is_contiguous())
            stage("checkpoint_conversion", export=True,
                  workspace={"device": training["device"],
                             "host": training["host"] + max(transfer, training_transfer)})
        try:
            from torch.utils.serialization import config as serialization_config

            pinned_transfer = getattr(getattr(serialization_config, "save", None), "use_pinned_memory_for_d2h", False)
        except ImportError:
            # Older torch.save transfers device storage with ordinary storage.cpu().
            pinned_transfer = False

        device_parameters = any(_placement(value, meta_device_type) == "device" for value in entries.values())
        transfer_unknown = list(serialization_unknown)
        transfer = 0
        if device_parameters and optimizer_device_storage_bytes is None:
            transfer_unknown.append("checkpoint_optimizer_transfer")
        elif optimizer_device_storage_bytes is not None:
            sizes = tuple(optimizer_device_storage_bytes)
            if any(size < 0 for size in sizes):
                raise ValueError("optimizer storage sizes must be nonnegative")
            if pinned_transfer and sizes:
                # The supplied layout is not torch.save's storage traversal order.
                transfer = max(sizes)
                transfer_unknown.append("checkpoint_pinned_storage_order")
            else:
                transfer = max(sizes, default=0)
        clones = {p: 0 for p in raw}
        clone_unknown = {p: [] for p in raw}
        clone_sizes = None if optimizer_state_clone_storage_bytes is None else tuple(optimizer_state_clone_storage_bytes)
        if clone_sizes is not None and any(size < 0 for size in clone_sizes):
            raise ValueError("optimizer state clone sizes must be nonnegative")
        if cfg.optimizer.kahan:
            placements = {_placement(parameter, meta_device_type) for parameter in training_state.parameters()
                          if parameter.dtype in (torch.float16, torch.bfloat16)}
            if clone_sizes is None and placements:
                for place in placements:
                    clone_unknown[place].append("optimizer_state_clone_layout")
            elif clone_sizes:
                if len(placements) == 1:
                    clones[next(iter(placements))] = sum(clone_sizes)
                else:
                    for place in raw:
                        clone_unknown[place].append("optimizer_state_clone_placement")
        elif clone_sizes:
            raise ValueError("state clone layout is supported only for KahanWrapper")
        stage("checkpoint_serialization", export=True,
              workspace={"device": training["device"] + clones["device"],
                         "host": training["host"] + transfer + clones["host"]},
              unknown={"device": clone_unknown["device"], "host": transfer_unknown + clone_unknown["host"]})
        for place in raw:
            stages[-1][f"optimizer_state_clone_{place}_bytes"] = (
                None if clone_unknown[place] else clones[place]
            )
        non_tensor_unknowns.append("host_checkpoint_serialization")

    unknown = sorted({reason for row in stages for reason in row["unknown_components"]})
    return {"operation": operation, "scope": "additional_tensor_storage", "stages": stages,
            "export_storage_records": records, "export_device_bytes": exported["device"],
            "export_host_bytes": exported["host"], "optimizer_mode": mode,
            "includes_model_optimizer_residents": False,
            "includes_existing_ema": False, "includes_new_ema": operation == "ema_initialize",
            "includes_kernel_workspace": False, "unknown_components": unknown,
            "serialization_version": serialization_version,
            "non_tensor_unknowns": non_tensor_unknowns}
