"""ComfyUI names for Klein layers without Diffusers LoRA aliases."""

from torch import Tensor

_LAYERS = {
    "double_stream_modulation_img.linear": "double_stream_modulation_img.lin",
    "double_stream_modulation_txt.linear": "double_stream_modulation_txt.lin",
    "single_stream_modulation.linear": "single_stream_modulation.lin",
    "time_guidance_embed.timestep_embedder.linear_1": "time_in.in_layer",
    "time_guidance_embed.timestep_embedder.linear_2": "time_in.out_layer",
}
_EXPORT = {
    "lora_transformer_" + source.replace(".", "_"): "lora_unet_" + target.replace(".", "_")
    for source, target in _LAYERS.items()
}
_IMPORT = {target: source for source, target in _EXPORT.items()}
# The extraction CLI historically used the generic backbone prefix.
_EXPORT.update({
    "lora_unet_" + source.replace(".", "_"): "lora_unet_" + target.replace(".", "_")
    for source, target in _LAYERS.items()
})


def has_legacy_keys(keys) -> bool:
    return any(key.partition(".")[0] in _EXPORT for key in keys)


def remap_adapter_keys(tensors: dict[str, Tensor], *, to_comfy: bool) -> dict[str, Tensor]:
    """Rename only the five Klein file modules; tensor values and training names stay unchanged."""
    aliases = _EXPORT if to_comfy else _IMPORT
    out = {}
    sources = {}
    for key, tensor in tensors.items():
        module, separator, suffix = key.partition(".")
        target = aliases.get(module, module)
        previous = sources.setdefault(target, module)
        if previous != module:
            raise ValueError(f"Klein 适配器的同一层存在重复名称：{previous}、{module}")
        out[target + separator + suffix] = tensor
    return out
