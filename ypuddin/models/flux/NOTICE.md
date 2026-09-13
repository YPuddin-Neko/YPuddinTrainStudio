FLUX.1 implementation references
==============================

The transformer and AutoencoderKL are provided by Diffusers. The original BFL
checkpoint mapping in loading.py follows Diffusers' Apache-2.0 single-file
converter, generalized to config-derived projection widths and strict rejection
of unused tensors:
https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/single_file_utils.py

Packing, position IDs and text conditioning follow FluxPipeline:
https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py

Default dev/schnell geometry, AE scaling and shift are from BFL's official config:
https://github.com/black-forest-labs/flux/blob/main/src/flux/util.py
T5-v1.1-XXL config:
https://huggingface.co/google/t5-v1_1-xxl/blob/main/config.json

Standalone components reuse the project's packaged OpenAI CLIP-L tokenizer
(sdxl/assets/tokenizer) and standard T5 tokenizer (anima/assets/t5_old).
A local pipeline's tokenizers or an explicit tokenizer_path override these.
No model, tokenizer, config or Python code is downloaded by this family.

Supported task: dev/schnell architecture text-to-image training and sampling.
Fill, Control and explicitly identified Kontext pipelines are rejected. A bare
Kontext transformer has the same tensor geometry as dev, so tensor shapes alone
cannot establish its intended task; image-conditioned editing is not implemented.
Block swapping, compilation, Unsloth and external attention processors are not
advertised until verified against this native network's backward lifecycle.

Adapter export compatibility
----------------------------

The main checkpoint uses the `lora_transformer_` prefix with flattened Diffusers
module names, recognized by ComfyUI's FLUX key map. LoRA and LoKr attn-mlp
mapping and delta weights were verified against the actual local ComfyUI loader
and original BFL network in a reduced CPU oracle (not full-weight GPU inference).
Diffusers 0.40 parses the attn-only LoRA export correctly. Its current mixture
converter does not preserve FLUX MLP submodule paths, so attn-mlp exports are not
claimed to load directly in that Diffusers release. The project can restore its
own exported LoRA and LoKr tensors without conversion.
https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/lora.py
