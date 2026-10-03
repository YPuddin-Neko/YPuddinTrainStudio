"""Preview exported adapter tensors without replacing live training parameters."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import MappingProxyType, MethodType

import torch
from torch import Tensor, nn

from ypuddin.adapters import AdapterSet
from ypuddin.adapters.components import ComponentAdapterSet
from ypuddin.adapters.dora import auto_merge_dtype
from ypuddin.adapters.io import SAVE_DTYPES
from ypuddin.adapters.linear import AdaptedLayer

from .adapter_fusion import ComfyAdapterFusion

_ABSENT = object()


def exported_adapter_bindings(adapters: AdapterSet | ComponentAdapterSet) -> dict[str, AdaptedLayer]:
    """Map complete exported prefixes to their live wrappers, including text encoders."""
    components = adapters.components.values() if isinstance(adapters, ComponentAdapterSet) else (adapters,)
    bindings: dict[str, AdaptedLayer] = {}
    for component in components:
        for name, layer in component.layers.items():
            key = component.export_key(name)
            if key in bindings:
                raise ValueError(f"duplicate exported adapter prefix: {key}")
            bindings[key] = layer
    return bindings


class AdapterSnapshot:
    """An immutable CPU copy of an adapter export, read layer by layer.

    ``save_dtype`` applies the checkpoint writer's casting rules; leave it unset for tensors that came from
    a saved adapter. Build it once and reuse it: a model test keeps one per checkpoint for all its cells.
    """

    def __init__(self, tensors: Mapping[str, Tensor], save_dtype: str | None = None) -> None:
        if save_dtype is not None and save_dtype not in SAVE_DTYPES:
            raise ValueError(f"unsupported adapter save dtype: {save_dtype}")
        layers: dict[str, dict[str, Tensor]] = {}
        for key, tensor in tensors.items():
            if not isinstance(tensor, Tensor):
                raise TypeError(f"adapter export must contain tensors: {key}")
            dtype = tensor.dtype
            if save_dtype is not None:
                dtype = torch.float32 if key.endswith(".alpha") else SAVE_DTYPES[save_dtype]
            # Export prefixes never contain a dot, so the first one ends the layer name.
            prefix, _, suffix = key.partition(".")
            layers.setdefault(prefix, {})[suffix] = tensor.detach().to(device="cpu", dtype=dtype, copy=True).contiguous()
        self._layers = layers
        self._fusions: dict[str, ComfyAdapterFusion] = {}

    def fusion(self, prefix: str) -> ComfyAdapterFusion:
        if prefix not in self._fusions:
            self._fusions[prefix] = ComfyAdapterFusion.from_tensors(MappingProxyType(self._layers.get(prefix, {})))
        return self._fusions[prefix]


def _free_memory(device: torch.device) -> int:
    try:
        if device.type == "cuda":
            return torch.cuda.mem_get_info(device)[0]
        if device.type == "mps":
            return max(0, torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory())
    except (AttributeError, RuntimeError):
        pass
    return 0


class _DeviceCopies:
    """Each layer's exported tensors on the device it runs on, copied there once per preview while they fit
    in half of that device's free memory as the first layer runs; beyond that a forward copies them again."""

    def __init__(self) -> None:
        self._copies: dict[str, tuple[torch.device, ComfyAdapterFusion]] = {}
        self._room: dict[torch.device, int] = {}

    def on(self, key: str, fusion: ComfyAdapterFusion, device: torch.device) -> ComfyAdapterFusion:
        if device.type == "cpu":
            return fusion
        kept = self._copies.get(key)
        if kept is not None and kept[0] == device:
            return kept[1]
        if device not in self._room:
            self._room[device] = _free_memory(device) // 2
        if fusion.nbytes > self._room[device]:
            return fusion
        self._room[device] -= fusion.nbytes
        copy = fusion.to(device)
        self._copies[key] = (device, copy)
        return copy


def _merge_dtype(layer: AdaptedLayer, x: Tensor, selected: str) -> torch.dtype:
    if selected != "auto":
        return SAVE_DTYPES[selected]
    if layer.dora is not None and layer.dora.compute_mode == "comfyui":
        return layer.dora.merge_dtype
    # FP8 storage is dequantized to the forward's floating-point compute dtype.
    compute = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
    dtype = auto_merge_dtype(layer.base, compute)
    if dtype not in SAVE_DTYPES.values():
        raise ValueError("adapter preview requires a floating-point inference dtype")
    return dtype


def _forwards(fusion: ComfyAdapterFusion, key: str, selected: str, used: dict[str, str], copies: _DeviceCopies):
    @torch.no_grad()
    def adapter_forward(_adapter, base: Tensor, dtype: torch.dtype):
        return copies.on(key, fusion, base.device).reconstruct_delta(base.shape, device=base.device, dtype=dtype)

    @torch.no_grad()
    def dora_forward(_dora, base: Tensor, delta: Tensor, alpha: float, dtype: torch.dtype, strength: float):
        return copies.on(key, fusion, base.device).fuse(base, delta, alpha, dtype=dtype, strength=strength)

    @torch.no_grad()
    def layer_forward(layer: AdaptedLayer, x: Tensor) -> Tensor:
        dtype = _merge_dtype(layer, x, selected)
        used[key] = str(dtype).removeprefix("torch.")
        with torch.autocast(device_type=x.device.type, enabled=False):
            base = layer.frozen_weight(dtype)
            # Module calls retain FSDP's unshard and reshard hooks.
            delta, alpha = layer.adapter(base, dtype)
            local = copies.on(key, fusion, base.device)
            if layer.dora is None:
                weight = local.fuse(base, delta, alpha, dtype=dtype, strength=layer.multiplier)
            else:
                weight = layer.dora(base, delta, alpha, dtype, layer.multiplier)
            bias = local.fuse_bias(layer.bias, dtype=dtype, strength=layer.multiplier)
        return layer._layer_op(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))

    return layer_forward, adapter_forward, dora_forward


@contextmanager
def exported_adapter_preview(
    bindings: Mapping[str, AdaptedLayer],
    tensors: Mapping[str, Tensor] | AdapterSnapshot,
    *,
    merge_dtype: str = "auto",
    save_dtype: str | None = None,
) -> Iterator[dict[str, str]]:
    """Use an immutable CPU export snapshot for inference, restoring every forward afterwards.

    ``tensors`` is an export, copied here (``save_dtype`` as for :class:`AdapterSnapshot`), or a snapshot
    made earlier. The yielded mapping records the actual fusion dtype of each layer called during the preview.
    """
    if merge_dtype != "auto" and merge_dtype not in SAVE_DTYPES:
        raise ValueError(f"unsupported adapter merge dtype: {merge_dtype}")
    if isinstance(tensors, AdapterSnapshot):
        if save_dtype is not None:
            raise ValueError("an adapter snapshot already holds its saved precision")
        snapshot = tensors
    else:
        snapshot = AdapterSnapshot(tensors, save_dtype)
    # A live export may hold every adapter tensor on the training device; sampling needs only the copy.
    del tensors
    prepared = []
    modules = set()
    for key, layer in bindings.items():
        if not isinstance(layer, AdaptedLayer):
            raise TypeError(f"adapter preview binding is not an adapted layer: {key}")
        fusion = snapshot.fusion(key)
        if ("dora_scale" in fusion.tensors) != (layer.dora is not None):
            raise ValueError(f"exported DoRA magnitude does not match the bound layer: {key}")
        for module in (layer, layer.adapter, layer.dora):
            if module is None:
                continue
            if id(module) in modules:
                raise ValueError(f"adapter preview bindings share a module: {key}")
            modules.add(id(module))
        prepared.append((key, layer, fusion))

    used: dict[str, str] = {}
    copies = _DeviceCopies()
    previous: list[tuple[nn.Module, object]] = []
    with torch.no_grad():
        try:
            for key, layer, fusion in prepared:
                forwards = _forwards(fusion, key, merge_dtype, used, copies)
                for module, forward in zip((layer, layer.adapter, layer.dora), forwards, strict=True):
                    if module is None:
                        continue
                    previous.append((module, module.__dict__.get("forward", _ABSENT)))
                    module.forward = MethodType(forward, module)
            yield used
        finally:
            for module, forward in reversed(previous):
                if forward is _ABSENT:
                    delattr(module, "forward")
                else:
                    module.forward = forward
