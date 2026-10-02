# Adapter fusion provenance

`adapter_fusion.py` adapts inference arithmetic from ComfyUI 0.38.1,
commit `20ca544ee0436721d8eb5f544665e490609f72c8`, by the ComfyUI authors and
contributors. The upstream code is licensed under GNU GPL version 3; the
license is included in the project's root `LICENSE`.

Upstream sources:

- [LoRA](https://github.com/Comfy-Org/ComfyUI/blob/20ca544ee0436721d8eb5f544665e490609f72c8/comfy/weight_adapter/lora.py)
- [LoKr](https://github.com/Comfy-Org/ComfyUI/blob/20ca544ee0436721d8eb5f544665e490609f72c8/comfy/weight_adapter/lokr.py)
- [LoHa](https://github.com/Comfy-Org/ComfyUI/blob/20ca544ee0436721d8eb5f544665e490609f72c8/comfy/weight_adapter/loha.py)
- [DoRA decomposition](https://github.com/Comfy-Org/ComfyUI/blob/20ca544ee0436721d8eb5f544665e490609f72c8/comfy/weight_adapter/base.py)
- [Full difference patches](https://github.com/Comfy-Org/ComfyUI/blob/20ca544ee0436721d8eb5f544665e490609f72c8/comfy/lora.py)

The adaptation selects a single exported layer, clones the base before fusion,
disables gradients and outer autocast, and raises errors for unsupported formats.
It preserves the upstream operation order, alpha handling, DoRA axis detection,
epsilon and strength interpolation. It covers this trainer's LoRA, LoKr, LoHa
and Full exports, including scalar and rsLoRA gains already folded into their
factors and alpha values.

The requested fusion dtype applies to both the base and intermediate values,
matching the `LowVramPatch` arithmetic. Static patching with a different
intermediate dtype, stochastic rounding, Tucker factors, shape-changing patches,
model-strength patches, offsets and custom patch functions are not implemented.
DoRA output-axis fusion uses the base-weight norm, as this upstream version does;
input-axis fusion uses the merged-weight norm. These inference rules do not
replace the trainer's DoRA computation or change saved training parameters.
