# Third-party notices for `ypuddin/models/krea2/vendor/`

`krea2_mmdit.py` is vendored from **kohya-ss/musubi-tuner** (<https://github.com/kohya-ss/musubi-tuner>) at commit
`8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6` (2026-09), file `src/musubi_tuner/krea2/krea2_mmdit.py`, which is licensed
under the **Apache License, Version 2.0**. musubi-tuner in turn ported the model definition from Krea's public
`references/Krea2/mmdit.py` (krea-ai/krea-2, Apache-2.0). Vendoring removes `ypuddin`'s runtime dependency on the
musubi-tuner tree.

The full Apache-2.0 text is in the repository root `LICENSE`. Per Section 4 of the license the original
attribution is preserved at the top of the vendored file, together with a first-line marker:

    # Vendored from kohya-ss/musubi-tuner (Apache-2.0) at commit 8934cfb; local modifications: attention import, block swap removed, config inference added.

`ypuddin` itself is Apache-2.0, so no license change occurs for downstream users.

## Vendored source files

| Vendored file | Upstream file (musubi-tuner @ 8934cfb) | Upstream provenance chain | Copyright |
|---|---|---|---|
| `krea2_mmdit.py` | `src/musubi_tuner/krea2/krea2_mmdit.py` | krea-ai/krea-2 `references/Krea2/mmdit.py` (SingleStreamDiT) | Krea AI (Apache-2.0); Kohya S. and musubi-tuner contributors (Apache-2.0) |

No model weights, tokenizers or configuration assets are included. The Qwen3-VL-4B-Instruct text encoder is loaded
from the user's own download with `transformers`; the Qwen3 tokenizer files in `ypuddin/models/anima/assets/qwen3_06b`
(vendored from sd-scripts, see `ypuddin/models/anima/vendor/NOTICE.md`) are reused as a fallback tokenizer.

## Local modifications

Rule of thumb: module and parameter **names are unchanged** in every class (`SingleStreamDiT`, `SingleStreamBlock`,
`TextFusionTransformer`, `Attention`, `SwiGLU`, `RMSNorm`, ...), so `state_dict()` keys stay identical to the
official `krea2_raw_bf16.safetensors` / Comfy-Org `krea2_fp8_scaled` checkpoints and to the ComfyUI / musubi LoRA
key conventions (`lora_unet_blocks_0_attn_wq` ...).

### `krea2_mmdit.py`

1. **Attention import**: `from musubi_tuner.modules.attention import AttentionParams, attention` ->
   `from ypuddin.models.anima.vendor.attention import ...` (the same kohya `AttentionParams` API, already vendored
   from sd-scripts). Because Krea 2 uses grouped-query attention (48 query heads / 12 kv heads) the shared
   `attention.py` gained a GQA expansion (`repeat_interleave` of k/v when head counts differ) for its SDPA and sage
   paths; that change is documented in `ypuddin/models/anima/vendor/NOTICE.md`.
2. **Block swap removed**: `blocks_to_swap`, `offloader`, `enable_block_swap`, `move_to_device_except_swap_blocks`,
   `switch_block_swap_for_inference/training`, `prepare_block_swap_before_forward` and the
   `offloader.wait_for_block / submit_move_blocks` calls in the block loop were removed. `ypuddin` swaps blocks with
   external hooks (`ypuddin.memory.BlockSwapper`).
3. **Gradient checkpointing**: `enable_gradient_checkpointing(cpu_offload=False)` keeps upstream's flag but the
   block loop only checkpoints when `self.training and torch.is_grad_enabled()`, so inference/preview passes never
   pay the recompute (`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` as upstream).
4. **Config helpers added** (not in upstream): `KREA2_CONFIG` (the public `single_mmdit_large_wide` geometry:
   features 6144, 28 layers, 48/12 heads, txtdim 2560, 12 text layers), `KREA2_KEY_PREFIXES`
   (`model.diffusion_model.`, `diffusion_model.`, `net.`) and `infer_config(state_dict, patch=2)` which derives a
   `SingleMMDiTConfig` from tensor shapes (works with `safetensors` slices, so no weights are read for planning).
5. Header comment with provenance + `# ruff: noqa`; `__all__`.

## Not vendored

`krea2_utils.py` (checkpoint loading / fp8 optimisation), `krea2_encoder.py` (Qwen3-VL prompt template and layer
selection), `krea2_sampling.py` (resolution-aware `mu` schedule, Euler sampler), `lora_krea2.py` and the
`krea2_train_network.py` / `krea2_cache_*` scripts were **not** copied. Their behaviour (prompt template, layer
indices, `mu` interpolation endpoints, fp8_scaled handling, LoRA target defaults) was re-implemented natively in
`ypuddin/models/krea2/family.py` and `ypuddin/models/krea2/text.py` and is covered by `Test/tests/unit/test_krea2_family.py`.
