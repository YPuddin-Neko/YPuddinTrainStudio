# MOSS-Image Base

This family loads the original dense Base DiT, frozen Qwen3.5 last hidden states,
and the released small-decoder Flux2 autoencoder or an explicitly selected full
FLUX.2 VAE. Existing model bundles keep their bundled VAE. Image batches keep a real batch
axis through all projections and attention; masks isolate samples and exclude
padding from text refinement and joint attention. Checkpointing supports none/block.

The official `moss_image` training distribution is the 1000-point stage-three law
in report equation (4), with no training schedule shift. Explicit t_min/t_max condition and renormalize the
PMF on the included 0.001-spaced points; stratification divides uniform quantiles
before inverse-CDF sampling. logit_mean/logit_std/shift do not alter this fixed law.
Rectified flow predicts
noise minus clean latent. Only image tokens enter the loss. Same-canvas batches
have equal token counts; the trainer's image mean equals their token mean. Other
explicit timestep/loss settings retain the trainer's existing definitions.
Noise and VAE calculations are FP32. VAE encoding uses the posterior mode. Spatial
noise modifiers operate before 2x2 latent packing. Text uses the local chat template
with thinking enabled, truncation at 4096 tokens, and unpadded encoding per caption.

Sampling uses the official snapped Euler schedule (50 steps, shift 5, CFG 3 by
default). The negative CFG branch skips main block 22. The text and image reference
branches, additional SLG, model compilation, block swapping, quantized weights,
full model/text encoder training, T-LoRA, and distributed training are not enabled.

Component overrides use local paths. A complete official directory works directly;
`model.vae_path` also accepts a standard full FLUX.2 Diffusers VAE directory or
safetensors file, including the original BFL layout. VAE caches follow the selected
weights and configuration; small and full decoders do not share cached latents.
shared installations may use `.ypuddin-moss-image-components.json` with format
`ypuddin-moss-image-components`, version 1, and absolute `text_encoder`, `vae`,
`tokenizer` paths under `components`. Explicit model fields take precedence.

Adapters preserve native module names, including fused `ffn_up_gate` and text
`qkv` projections. Export uses `lora_unet_` plus underscore-separated native paths,
matching ComfyUI's generic native-key mapping when its MOSS-Image plugin is present.
