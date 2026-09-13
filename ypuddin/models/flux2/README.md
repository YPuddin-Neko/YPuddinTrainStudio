# FLUX.2 Klein backend boundary

This backend trains only Klein base 4B and Klein base 9B. It uses the installed Diffusers `Flux2Transformer2DModel` and
`AutoencoderKLFlux2`, and Transformers' Qwen3 implementation. It loads
local safetensors assets only. No weight or tokenizer download is triggered.

| Branch | Text encoder | Selected hidden states | Sampling default |
| --- | --- | --- | --- |
| Klein base 4B | Qwen3-4B | 9, 18, 27 | 50 steps, CFG 4, no guidance embedding |
| Klein base 9B | Qwen3-8B | 9, 18, 27 | 50 steps, CFG 4, no guidance embedding |

The original BFL Klein transformer and shared FLUX.2 VAE single-file formats and local Diffusers
component directories are accepted. A single transformer requires explicit
text-encoder, VAE and tokenizer assets. Text encoders must be local HF
directories; quantized text or transformer checkpoints are rejected. Klein base
and distilled weights have identical geometry: automatic selection requires an HF
`model_index.json` declaring `is_distilled=false`. Otherwise the user must select
`model.flux2_variant`; a declaration of distilled weights is always rejected.
Legacy dev settings and dev transformer checkpoints are explicitly rejected before
any transformer payload or text encoder is loaded. They are never reinterpreted as
Klein. Header-only dev identification remains available for historical inspection.
Klein distilled/KV, reference-image editing, online text training, FP8 conversion,
unsloth checkpointing and compile are outside this implementation.

The autoencoder uses its posterior **mode**, packs 2×2 spatial cells into 128
channels, then normalizes with the checkpoint's frozen batchnorm mean/variance.
Decode exactly reverses that normalization and packing. Its cache identity includes
the real weights and normalization configuration. Conditioners retain all 512
tokens, including padding. Qwen3 uses the upstream user template with thinking disabled.
Klein has no embedded guidance input; sampling guidance is classifier-free guidance (CFG).
The transformer sees unit flow time; Diffusers applies its internal ×1000 scaling.
Preview shift uses BFL's empirical function of both image-token count and step count.

`load()` builds a meta transformer and lazy encoders. `materialize_backbone()`
unloads the text/VAE modules before reading transformer weights, and must run before
adapter injection. This avoids keeping the large transformer and text encoder
weights resident together during cache creation. It does not stream text-encoder
layers: text caching still needs enough device memory for the whole text encoder
and its activations. A 16 GB GPU is not a validated environment for full
Klein 9B weights. Cache batch size 1 and Klein base 4B are the more practical starting
point; actual GPU feasibility remains hardware dependent. The public text-encoder
estimate defaults to approximately 4B parameters. Unconfigured meta planning uses
Klein 4B geometry (5 double / 20 single blocks); an explicit Klein 9B selection
uses 8 / 24 blocks. Loaded metadata also records the real local text-weight element
count from safetensors headers, without materializing the encoder.

Block swap requires block checkpointing. Diffusers' checkpoint path passes tensors
positionally to the actual double/single blocks, allowing the swapper's backward
hooks to observe input gradients. CPU tests compare every adapter gradient and the
latent-input gradient against the same real network without swapping, including
LoKr Full. These tests do not measure CUDA streams, transfer peaks or full-model VRAM.

Default exports use `lora_transformer_` followed by the underscored Diffusers module
name. Current ComfyUI's `Flux2(Flux)` dispatch and `flux_to_diffusers` mapping include
all targets in this backend's two presets, including the fused single-stream QKV/MLP
projection. An independent CPU oracle executed the real ComfyUI mapping and adapter
weight routines for LoRA, factorized LoKr and LoKr Full on reduced Klein geometry: all 30
targets matched, delta-weight error was zero, and the maximum network prediction
error after converting patched BFL weights was 4.77e-7. This is not full-weight
ComfyUI application or GPU validation. Diffusers 0.40's FLUX.2 Kohya parser expects BFL `double_blocks` and
`single_blocks` names, so the default file is not advertised as directly loadable
there. The separate PEFT oracle has also passed with PEFT 0.20.0, Diffusers 0.40.0,
Transformers 5.17.0 and Torch 2.14.0: it explicitly converts LoRA to PEFT, folds
alpha/rank into B, loads it into an independent real Diffusers transformer, and
compares predictions at `rtol=2e-5, atol=2e-6`. This confirms the explicit conversion
path, not direct loading of the default file. Diffusers PEFT does not provide an
equivalent LoKr loader. The test skips when the optional `peft` package is absent;
native training does not require PEFT.

On 2026-09-13, all 38 focused CPU tests passed. The suite covers Klein cache →
training → preview → adapter save/reload,
LoRA / LoKr / LoKr Full checkpoint-and-swap gradients, local shard loading, original
BFL conversion, VAE batchnorm normalization, PEFT conversion, and dev rejection
before weight loading. Expected PyTorch hook warnings exercise the no-input-gradient
case; the swapper defers releasing those blocks until backward completes. These
results use reduced real networks, not full downloaded checkpoints or CUDA kernels.

References (implementation delegates to these dependencies; no GPL trainer source
is copied into this package):

- [BFL model geometry](https://github.com/black-forest-labs/flux2/blob/main/src/flux2/model.py)
- [BFL text conditioning](https://github.com/black-forest-labs/flux2/blob/main/src/flux2/text_encoder.py)
- [BFL VAE normalization](https://github.com/black-forest-labs/flux2/blob/main/src/flux2/autoencoder.py)
- [Diffusers Klein pipeline](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py)
- [ComfyUI adapter key mapping](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/lora.py)
- [ComfyUI transformer key mapping](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/utils.py)
