# FLUX.3 Image Klein Base backend

This family loads the original undistilled Base DiT, its paired autoencoder and a
complete local Qwen3-VL-4B-Instruct processor/encoder directory. The fixed upstream
source revision and licensing are recorded in `NOTICE.md`.

The `flux3` optional dependency group requires Transformers 5.8.1 or later. Normal
CUDA/CPU/MPS startup and the existing isolated PyTorch environment preparation
include it; native framework constraints remain protected. The Linux DTK launcher
retains its vendor dependency rules and does not select this group. Family preflight
reports an incompatible Transformers version before loading model payloads.

Text-to-image training supports adapters or the full DiT, with a frozen BF16 text
encoder and FP32 VAE. LoRA, LoKr, LoHa, Full-diff and Ortho adapters
use the existing adapter runtime. Ortho export remains plain LoRA. T-LoRA,
text encoder training, FP16/FP8 base conversion, compilation, block swap and
multi-GPU training are rejected by the family configuration checks. Reference-image
editing, bounding boxes, the prompt upsampler and distilled flow-map weights are
outside this family entry.

Full training selects every DiT parameter, including normalization, modulation and
conditioning projections; the image-conditioning branch receives gradients only
when nonempty image conditions are supplied. Trainable parameters stay FP32, with
optional BF16 mixed precision. Ordinary and EMA exports contain native
`backbone/model.safetensors`, the architecture `config.json` and an artifact training
config referencing the frozen components. Full model save precision is BF16 or FP32.

Text conditioning retains the official 80-token padding multiple (4096 maximum),
all eight selected hidden layers, and the full padded states for each caption.
Extra batch padding is removed only by the recorded per-caption lengths. The VAE
uses its mean, a 2x2 patch and frozen batch normalization; no external scale/shift
is applied. The training objective predicts noise minus clean latents. Its initial
recipe is logit-normal time sampling with no loss weighting. The preview defaults
are 50 Euler steps, CFG 3.5 and inference time shift 6.93; inference shift does not
set the training time distribution.

Adapters keep original projection names with the `lora_unet` prefix. The project
family/architecture metadata identifies FLUX.3; no undocumented third-party
`ss_base_model_version` convention is asserted. The same original projection names
are recognized by ComfyUI's FLUX.3 fused-qkv mapping. External runtime image quality
must be checked with actual model weights and adapters.

Memory planning reads the complete encoder's safetensors headers, uses sequential
single-caption encoding and records the batch conditioning separately. The DiT
activation coefficients bound the local shape census in
`Test/scripts/measure_flux3_activations.py`: 26 units per effective block and 32
checkpoint recompute units. The census uses meta tensors and a fused-attention
saved-storage surrogate; it is not an accelerator peak measurement. The effective
path has 5 separate-mode stages and 28 shared stages. VAE storage is counted from
its exact meta architecture; convolution workspace uses the existing 128-channel
FP32 KL VAE envelope. Device, allocator and full-model peaks need separate hardware
measurement.

The directory download entries pin separate Hugging Face and ModelScope revisions
and file manifests. Each source publishes its own complete directory only after
every listed file and component passes validation. Hugging Face public metadata
withholds large-file SHA-256 values: those files use the pinned revision and byte
size, with the downloaded SHA-256 recorded; small files use Git blob hashes.
ModelScope provides upstream SHA-256 values for every listed file, all checked
during transfer. Verification records and partial files are never shared across
sources. A same-source retry retains its original manifest; switching sources
starts a new attempt with the selected source's pinned manifest.
