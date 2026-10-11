"""Preview exported adapter tensors without replacing live training parameters."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
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


class _FusionDtypes(Mapping[str, str]):
    """Resolve layer names only when reading the observed inference dtypes."""

    def __init__(self):
        self.slots: dict[str, list[torch.dtype | None]] = {}

    def __getitem__(self, key: str) -> str:
        dtype = self.slots[key][0]
        if dtype is None:
            raise KeyError(key)
        return str(dtype).removeprefix("torch.")

    def __iter__(self):
        return (key for key, slot in self.slots.items() if slot[0] is not None)

    def __len__(self):
        return sum(slot[0] is not None for slot in self.slots.values())


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


class _CopySlot:
    def __init__(self) -> None:
        self.kept: tuple[torch.device, ComfyAdapterFusion] | None = None
        self.denied: set[torch.device] = set()


class _DeviceCopies:
    """Retain factors within half of each device's free memory when first needed.

    Compiled blocks prepare their factors before entering the block; other layers prepare on first use.
    Factors beyond the budget stay on CPU and are read again by each forward.
    """

    def __init__(self) -> None:
        self._room: dict[torch.device, int] = {}

    def on(self, slot: _CopySlot, fusion: ComfyAdapterFusion, device: torch.device) -> ComfyAdapterFusion:
        if device.type == "cpu":
            return fusion
        kept = slot.kept
        if kept is not None and kept[0] == device:
            return kept[1]
        if device in slot.denied:
            return fusion
        return self._fill(slot, fusion, device)

    def _fill(self, slot: _CopySlot, fusion: ComfyAdapterFusion, device: torch.device) -> ComfyAdapterFusion:
        if device not in self._room:
            self._room[device] = _free_memory(device) // 2
        if fusion.nbytes > self._room[device]:
            # The budget only decreases, so subsequent calls use the same uncached CPU tensors.
            slot.denied.add(device)
            return fusion
        self._room[device] -= fusion.nbytes
        copy = fusion.to(device)
        slot.kept = (device, copy)
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


def _forwards(
    fusion: ComfyAdapterFusion, selected: str, used: list[torch.dtype | None], copies: _DeviceCopies, cached: _CopySlot,
):
    @torch.no_grad()
    def adapter_forward(_adapter, base: Tensor, dtype: torch.dtype):
        return copies.on(cached, fusion, base.device).reconstruct_delta(base.shape, device=base.device, dtype=dtype)

    @torch.no_grad()
    def dora_forward(_dora, base: Tensor, delta: Tensor, alpha: float, dtype: torch.dtype, strength: float):
        return copies.on(cached, fusion, base.device).fuse(base, delta, alpha, dtype=dtype, strength=strength)

    @torch.no_grad()
    def layer_forward(layer: AdaptedLayer, x: Tensor) -> Tensor:
        dtype = _merge_dtype(layer, x, selected)
        used[0] = dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            base = layer.frozen_weight(dtype)
            # Module calls retain FSDP's unshard and reshard hooks.
            delta, alpha = layer.adapter(base, dtype)
            local = copies.on(cached, fusion, base.device)
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
    compiled_blocks: Sequence[nn.Module] = (),
) -> Iterator[Mapping[str, str]]:
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

    used = _FusionDtypes()
    copies = _DeviceCopies()
    previous: list[tuple[nn.Module, object]] = []
    hooks = []
    caches = {id(layer): _CopySlot() for _, layer, _ in prepared}

    def prepare_block(layers):
        def pre_forward(_block, _args):
            for layer, fusion in layers:
                copies.on(caches[id(layer)], fusion, layer.base.weight.device)
        return pre_forward
    with torch.no_grad():
        try:
            for key, layer, fusion in prepared:
                slot = used.slots[key] = [None]
                forwards = _forwards(fusion, merge_dtype, slot, copies, caches[id(layer)])
                for module, forward in zip((layer, layer.adapter, layer.dora), forwards, strict=True):
                    if module is None:
                        continue
                    previous.append((module, module.__dict__.get("forward", _ABSENT)))
                    module.forward = MethodType(forward, module)
            for block in compiled_blocks:
                members = {id(module) for module in block.modules()}
                layers = [(layer, fusion) for _, layer, fusion in prepared if id(layer) in members]
                if layers:
                    hooks.append(block.register_forward_pre_hook(prepare_block(layers)))
            yield used
        finally:
            for hook in hooks:
                hook.remove()
            for module, forward in reversed(previous):
                if forward is _ABSENT:
                    delattr(module, "forward")
                else:
                    module.forward = forward
