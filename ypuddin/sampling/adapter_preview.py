"""Preview exported adapter tensors without replacing live training parameters."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import MethodType

import torch
from torch import Tensor, nn

from ypuddin.adapters import AdapterSet
from ypuddin.adapters.components import ComponentAdapterSet
from ypuddin.adapters.io import SAVE_DTYPES
from ypuddin.adapters.linear import AdaptedLayer

from .adapter_fusion import ComfyAdapterFusion, fuse_adapter_delta

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


def _snapshot(tensors: Mapping[str, Tensor], save_dtype: str | None) -> dict[str, Tensor]:
    if save_dtype is not None and save_dtype not in SAVE_DTYPES:
        raise ValueError(f"unsupported adapter save dtype: {save_dtype}")
    snapshot = {}
    for key, tensor in tensors.items():
        if not isinstance(tensor, Tensor):
            raise TypeError(f"adapter export must contain tensors: {key}")
        dtype = tensor.dtype
        if save_dtype is not None:
            dtype = torch.float32 if key.endswith(".alpha") else SAVE_DTYPES[save_dtype]
        snapshot[key] = tensor.detach().to(device="cpu", dtype=dtype, copy=True).contiguous()
    return snapshot


def _merge_dtype(layer: AdaptedLayer, x: Tensor, selected: str) -> torch.dtype:
    if selected != "auto":
        return SAVE_DTYPES[selected]
    if layer.dora is not None and layer.dora.compute_mode == "comfyui":
        return layer.dora.merge_dtype
    if layer.weight.dtype in SAVE_DTYPES.values():
        return layer.weight.dtype
    # FP8 storage is dequantized to the forward's floating-point compute dtype.
    dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
    if dtype not in SAVE_DTYPES.values():
        raise ValueError("adapter preview requires a floating-point inference dtype")
    return dtype


def _forwards(fusion: ComfyAdapterFusion, key: str, selected: str, used: dict[str, str]):
    @torch.no_grad()
    def adapter_forward(_adapter, base: Tensor, dtype: torch.dtype):
        return fusion.reconstruct_delta(base.shape, device=base.device, dtype=dtype)

    @torch.no_grad()
    def dora_forward(_dora, base: Tensor, delta: Tensor, alpha: float, dtype: torch.dtype, strength: float):
        return fuse_adapter_delta(
            base, delta, dtype=dtype, alpha=alpha, magnitude=fusion.tensors["dora_scale"], strength=strength,
        )

    @torch.no_grad()
    def layer_forward(layer: AdaptedLayer, x: Tensor) -> Tensor:
        dtype = _merge_dtype(layer, x, selected)
        used[key] = str(dtype).removeprefix("torch.")
        with torch.autocast(device_type=x.device.type, enabled=False):
            base = layer.frozen_weight(dtype)
            # Module calls retain FSDP's unshard and reshard hooks.
            delta, alpha = layer.adapter(base, dtype)
            if layer.dora is None:
                weight = fuse_adapter_delta(base, delta, dtype=dtype, alpha=alpha, strength=layer.multiplier)
            else:
                weight = layer.dora(base, delta, alpha, dtype, layer.multiplier)
            bias = fusion.fuse_bias(layer.bias, dtype=dtype, strength=layer.multiplier)
        return layer._layer_op(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))

    return layer_forward, adapter_forward, dora_forward


@contextmanager
def exported_adapter_preview(
    bindings: Mapping[str, AdaptedLayer],
    tensors: Mapping[str, Tensor],
    *,
    merge_dtype: str = "auto",
    save_dtype: str | None = None,
) -> Iterator[dict[str, str]]:
    """Use an immutable CPU export snapshot for inference, restoring every forward afterwards.

    ``save_dtype`` applies the checkpoint writer's casting rules. Leave it unset
    when ``tensors`` already came from a saved adapter. The yielded mapping records
    the actual fusion dtype of each layer called during the preview.
    """
    if merge_dtype != "auto" and merge_dtype not in SAVE_DTYPES:
        raise ValueError(f"unsupported adapter merge dtype: {merge_dtype}")
    snapshot = _snapshot(tensors, save_dtype)
    prepared = []
    modules = set()
    for key, layer in bindings.items():
        if not isinstance(layer, AdaptedLayer):
            raise TypeError(f"adapter preview binding is not an adapted layer: {key}")
        fusion = ComfyAdapterFusion.from_tensors(snapshot, prefix=key)
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
    previous: list[tuple[nn.Module, object]] = []
    with torch.no_grad():
        try:
            for key, layer, fusion in prepared:
                forwards = _forwards(fusion, key, merge_dtype, used)
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
