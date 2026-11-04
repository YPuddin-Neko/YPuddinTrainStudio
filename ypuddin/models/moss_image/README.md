# MOSS-Image Base

This family loads the original dense Base DiT, frozen Qwen3.5 last hidden states,
and the released small-decoder Flux2 autoencoder or an explicitly selected full
FLUX.2 VAE. Existing model bundles keep their bundled VAE. Image batches keep a real batch
axis through all projections and attention; masks isolate samples and exclude
padding from text refinement and joint attention. Checkpointing supports none/block
and `unsloth`, which checkpoints blocks while saving their inputs on CPU.

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
branches, additional SLG, text encoder training, and distributed training
are not enabled.

Adapter training supports CUDA block swapping through `memory.blocks_to_swap`.
Its memory layout includes text-refinement blocks followed by main DiT blocks;
the valid count follows the loaded architecture. Frozen tensors are staged from
CPU while trainable adapters stay on the device. Preview follows the same layout,
including the negative branch's skipped block. Full-DiT block swapping is disabled.

Adapter training can convert selected frozen linear weights to FP8 E4M3 or E5M2
storage at startup. Forward calculations dequantize to BF16 or FP32; this does
not accept arbitrary quantized source checkpoints or quantize the text encoder
and VAE. Full-DiT trainable parameters remain FP32. CUDA block compilation uses
the shared Inductor path and requires `memory.activation_checkpointing=none`;
it cannot be combined with block swapping. With compilation disabled, both
`block` and CPU-offloaded `unsloth` checkpointing remain available, including
with FP8 frozen storage. Compilation does not change saved configuration values.

T-LoRA uses each image's sampled training timestep to mask its adapter ranks, including
rules that override the global adapter algorithm. It uses bypass computation; orthogonal
initialization and its original factors are retained in complete training checkpoints.
Exports are ordinary LoRA for inference. Noise offset and multiresolution noise use the
existing spatial-noise path before latent packing and cannot be enabled together.

Single-device full training updates the complete DiT, including normalization,
text refinement, timestep and modulation parameters. The external text encoder and
VAE stay frozen. Trainable parameters remain FP32, with optional BF16 mixed precision.
Ordinary and EMA exports use native `diffusion_pytorch_model.safetensors` plus the
architecture `config.json` inside a `.model` artifact. Save precision is BF16 or FP32;
the artifact's training config retains explicit references to the frozen components.

Component overrides use local paths. A complete official directory works directly;
`model.vae_path` also accepts a standard full FLUX.2 Diffusers VAE directory or
safetensors file, including the original BFL layout. VAE caches follow the selected
weights and configuration; small and full decoders do not share cached latents.

Optional VAE tiling works with both small and full decoders, using 512-pixel tiles
and 128-pixel overlap. Cache-only tiling uses 1024/128 above 2048² pixels and
otherwise follows ordinary tiling; it does not change preview decoding. Both
options default off. FP32 mean encoding, patch2 and frozen BN are retained; decoded
tiles are blended before the final clamp. Spatial normalization and attention are
tile-local, so results differ from whole-image computation. Enabled settings use
distinct cache identities.

shared installations may use `.ypuddin-moss-image-components.json` with format
`ypuddin-moss-image-components`, version 1, and absolute `text_encoder`, `vae`,
`tokenizer` paths under `components`. Explicit model fields take precedence.

Adapters preserve native module names, including fused `ffn_up_gate` and text
`qkv` projections. Export uses `lora_unet_` plus underscore-separated native paths,
matching ComfyUI's generic native-key mapping when its MOSS-Image plugin is present.
