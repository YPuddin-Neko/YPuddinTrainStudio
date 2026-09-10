# Third-party notices for `ypuddin/models/anima/vendor/` and `ypuddin/models/anima/assets/`

Everything in these two directories is vendored from **kohya-ss/sd-scripts**
(<https://github.com/kohya-ss/sd-scripts>, `dev`/`sd3`-lineage), at commit
`4e624302e0088e39933b31cbc71f24212e900f5f` ("Merge pull request #2428 from kohya-ss/dev", 2026-09-09),
which is licensed under the **Apache License, Version 2.0**. sd-scripts in turn vendored the upstream sources
listed below, all Apache-2.0. Vendoring removes `ypuddin`'s runtime dependency on the sd-scripts tree.

The full Apache-2.0 text is in the repository root `LICENSE`. Per Section 4 of the license the original
copyright/attribution headers are preserved at the top of every vendored source file, and every file carries
a first-line marker:

    # Vendored from kohya-ss/sd-scripts (Apache-2.0) at commit 4e62430; local modifications: import paths only + noted changes.

`ypuddin` itself is Apache-2.0, so no license change occurs for downstream users.

## Vendored source files

| Vendored file | Upstream file (sd-scripts @ 4e62430) | Upstream provenance chain | Copyright |
|---|---|---|---|
| `cosmos_dit.py` | `library/anima_models.py` | NVIDIA Cosmos-Predict2 (`MiniTrainDIT`) + Anima `LLMAdapter`; Unsloth-Zoo offloaded checkpointing | NVIDIA CORPORATION & AFFILIATES (Apache-2.0); kohya-ss; Unsloth (Daniel Han-Chen & team); DeepSpeed `detach_variable` re-implementation |
| `attention.py` | `library/attention.py` | kohya-ss | kohya-ss (Apache-2.0) |
| `qwen_image_vae.py` | `library/qwen_image_autoencoder_kl.py` | Musubi-Tuner <- HuggingFace diffusers `AutoencoderKLQwenImage` <- Wan-Video Wan2.1 VAE | The Qwen-Image Team, Wan Team and The HuggingFace Team (Apache-2.0); kohya-ss |
| `qwen_image_vae_2d.py` | `library/qwen_image_autoencoder_kl_2d.py` | kohya-ss (2-D re-implementation of the above) | kohya-ss (Apache-2.0) |

## Vendored assets (byte-identical copies, SHA-256 verified at copy time)

| Vendored path | Upstream path | Origin |
|---|---|---|
| `assets/qwen3_06b/config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` | `configs/qwen3_06b/*` | Qwen/Qwen3-0.6B (Alibaba Cloud, Apache-2.0) tokenizer + model config. No model weights are included. |
| `assets/t5_old/config.json`, `spiece.model`, `tokenizer.json` | `configs/t5_old/*` | google/t5-v1_1-xxl ("old" T5 SentencePiece tokenizer, vocab 32128, Apache-2.0). Used only to produce token ids for the LLM adapter; no T5 weights are included. |

## Local modifications

Rule of thumb: module and parameter **names are unchanged** in every class (`Anima`, `Block`, `Attention`,
`LLMAdapter*`, `AutoencoderKLQwenImage*`, ...), so `state_dict()` keys stay compatible with official checkpoints
and with sd-scripts / ComfyUI / diffusion-pipe LoRA key conventions.

### `cosmos_dit.py` (from `anima_models.py`)

1. **Imports**: `from library import custom_offloading_utils, attention` -> `from . import attention`;
   `from .utils import setup_logging; setup_logging()` -> plain `import logging; logger = logging.getLogger(__name__)`.
2. **Block swap removed**: `blocks_to_swap`, `offloader` (`custom_offloading_utils.ModelOffloader`),
   `enable_block_swap`, `move_to_device_except_swap_blocks`, `switch_block_swap_for_inference`,
   `switch_block_swap_for_training`, `prepare_block_swap_before_forward`, and the
   `offloader.wait_for_block` / `submit_move_blocks` calls in the block loop (now `for block in self.blocks`).
   This project swaps blocks with external hooks (`ypuddin.memory`). Gradient checkpointing is kept unchanged
   (`torch.utils.checkpoint`, `use_reentrant=False`, plus the self-contained CPU-offload and Unsloth variants).
3. **torchvision dependency removed**: `torchvision.transforms.functional.resize(padding_mask, (H, W), NEAREST)`
   in `prepare_embedded_sequence` -> `torch.nn.functional.interpolate(mode="nearest")` (what torchvision calls
   internally for tensors; only executed when the mask size differs from the latent size). Two small
   robustness additions in the same spot: a `None` `padding_mask` becomes an all-zero mask (every sd-scripts
   caller passes zeros), and the mask is cast to the latent's dtype/device before concatenation.
4. **RoPE buffers non-persistent**: `VideoRopePosition3DEmb.seq`, `dim_spatial_range`, `dim_temporal_range` are
   registered with `persistent=False` (upstream: persistent). They are pure functions of the constructor args,
   official checkpoints do not contain them (sd-scripts filters them out of `missing_keys`), and excluding them
   from `state_dict()` lets `load_state_dict` succeed regardless of the `max_img_h/w` the saving side used
   (sd-scripts 512 vs. this copy 1024 would otherwise be a shape mismatch on `pos_embedder.seq`). Same approach
   as diffusion-pipe commit `b0aa4f1`. `llm_adapter.rotary_emb.inv_freq` was already non-persistent upstream.
5. **Config helpers added** (not in upstream, which only had a commented-out `get_dit_config`):
   `ANIMA_2B_CONFIG` (the hard-coded dict from `library/anima_utils.py:load_anima_model`, with
   `max_img_h = max_img_w = 1024` instead of 512, `attn_mode="torch"`, `split_attn=False`, and the constructor
   default `mlp_ratio=4.0` spelled out so `infer_dit_config(official_2B_state_dict) == ANIMA_2B_CONFIG`),
   `ANIMA_NUM_HEADS_BY_WIDTH`, `ANIMA_KEY_PREFIXES`,
   `detect_key_prefix`, `strip_key_prefix`, `infer_dit_config` (+ alias `get_dit_config`).
   *Why 1024 is safe*: `max_img_h/w` feed only `len_h/len_w = max_img // patch_spatial`, which size the
   `seq = arange(...)` position table and the `H <= max_h` assertion in `generate_embeddings`. The rotary
   frequencies come from `head_dim` and the `*_extrapolation_ratio` NTK factors alone, and `seq[:H]` is the
   same `0..H-1` for any table at least that long, so outputs are bit-identical for every input that fits in
   both settings -- the larger value is pure extrapolation head-room for high-resolution training, exactly as in
   AnimaLoraStudio / diffusion-pipe. Verified by `tests/unit/test_anima_vendor.py::test_max_img_size_is_pure_extrapolation`.
6. Module docstring documenting the training / sampling forward-call contract; `__all__`; `# ruff: noqa`.

### `attention.py` (from `attention.py`)

1. Only the PyTorch SDPA backend is kept (`attn_mode` in `{"torch", "sdpa", None}`); the xformers, flash-attn and
   sageattention branches and their optional imports are removed. Selecting a removed mode raises
   `NotImplementedError`. `AttentionParams` keeps the same fields/properties so `Attention.forward` in
   `cosmos_dit.py` is unchanged.
2. Deliberate fix: in `split_attn` mode with no explicit `seqlens`, upstream sliced `k`/`v` to the **query**
   length (`k[i:i+1, :q_len]`). That is a no-op for self-attention but silently truncated the text context in
   cross-attention whenever there were fewer image tokens than text tokens. The vendored copy keeps each
   tensor's own full length in that case (masked/`seqlens` paths are unchanged).

### `qwen_image_vae.py` (from `qwen_image_autoencoder_kl.py`)

1. `from library.safetensors_utils import load_safetensors` -> local `load_safetensors(path, device, disable_mmap,
   dtype)` built on `safetensors.torch.load_file` / `safetensors.torch.load` (same signature subset;
   `disable_numpy_memmap` dropped).
2. `setup_logging` -> plain `logging`.
3. The `if __name__ == "__main__":` debugging harness (depended on `library.device_utils`, PIL, argparse) is dropped.
4. Module docstring; `# ruff: noqa`. All model code is otherwise byte-identical (verified with `diff`).

### `qwen_image_vae_2d.py` (from `qwen_image_autoencoder_kl_2d.py`)

1. `from library.qwen_image_autoencoder_kl import (...)` -> `from .qwen_image_vae import (...)`, which also
   supplies `load_safetensors`; `setup_logging` -> plain `logging`.
2. Module docstring; `# ruff: noqa`. Model code otherwise byte-identical (verified with `diff`).

## Not vendored

`library/anima_utils.py` (checkpoint loading with fp8/LoRA merge, tokenizer/text-encoder loaders that need
`transformers`), `library/anima_train_utils.py`, `library/strategy_anima.py`, `library/custom_offloading_utils.py`
and `library/safetensors_utils.py` were **not** copied; their relevant behaviour is documented in the module
docstrings of the vendored files and re-implemented natively in `ypuddin`.
