I have everything needed. Here is the report.

---

# Architectural Report: kohya-ss/sd-scripts (Anima-focused)

`$ROOT` = `/Volumes/Service/Dev/YPuddinTrainStudio/sd-scripts` (git `4e62430`, dated 2026-09-09, branch `main`). All paths below are `$ROOT/<path>:<line>`.

Important context: this checkout is a **post-refactor** sd-scripts. The historic 6000-line `library/train_util.py` has been split; `library/train_util.py` is now a 218-line re-export shim (`library/train_util.py:1-218`). Canonical modules are `dataset.py`, `subset.py`, `dreambooth_dataset.py`, `finetuning_dataset.py`, `caching.py`, `args.py`, `accelerator_setup.py`, `optimizer.py`, `model_io.py`, `checkpoint_io.py`, `loss.py`, `sampling.py`, `logging_util.py`, `hidden_states.py`. Documentation in `.ai/context/01-overview.md:13-37` mandates importing from these directly.

---

## 1. Core training loop — `train_network.py` (`class NetworkTrainer`)

File: `$ROOT/train_network.py` (2047 lines). `NetworkTrainer` at line 66; `train()` spans lines 898–1863 (~965 lines, single method).

### 1.1 Step-by-step flow of `train(args)`

| Step | Lines | What happens |
|---|---|---|
| Arg verification / seeding | 899–912 | `args_util.verify_training_args`, `accelerator_setup.prepare_dataset_args`, `deepspeed_utils.prepare_deepspeed_args`; random seed if None |
| Strategy singletons | 914–920 | `get_tokenize_strategy()` → `TokenizeStrategy.set_strategy()`; `get_latents_caching_strategy()` → `LatentsCachingStrategy.set_strategy()` (class-level globals; must be set before dataset construction) |
| Dataset construction | 923–967 | `BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True)).generate(user_config, args)` → `config_util.generate_dataset_group_by_blueprint` returns `(train_dataset_group, val_dataset_group)`. Fallbacks: DreamBooth subdir config (`generate_dreambooth_subsets_config_by_subdirs`) or FineTuning `in_json`. Alternative: `--dataset_class` arbitrary dataset (966) |
| Collator | 969–972 | `collator_class(current_epoch, current_step, ds)` with two `multiprocessing.Value`s to propagate epoch/step into DataLoader workers |
| `assert_extra_args` hook | 997 | Subclass validation (Anima sets bucket step 16, disables fp8) |
| Accelerator / dtypes | 1001–1006 | `accelerator_setup.prepare_accelerator(args)`; `weight_dtype, save_dtype = prepare_dtype(args)`; `vae_dtype = fp32 if no_half_vae else weight_dtype` |
| `load_target_model` hook | 1009–1015 | returns `(model_version, text_encoder(s), vae, unet_or_None)` |
| Latent caching | 1018–1030 | `train_dataset_group.new_cache_latents(vae, accelerator)`; VAE moved back to CPU |
| Text-encoding strategy + TE output caching | 1034–1042 | `get_text_encoding_strategy` / `get_text_encoder_outputs_caching_strategy` → `set_strategy`; `cache_text_encoder_outputs_if_needed` hook |
| Lazy UNet load | 1044–1046 | `load_unet_lazily` hook (Anima loads DiT here, after TE caching frees VRAM) |
| Network module import | 1049–1051 | `importlib.import_module(args.network_module)` — duck-typed plugin |
| `--base_weights` merge | 1053–1068 | `create_network_from_weights(..., for_inference=True)` then `module.merge_to(...)` |
| Network creation | 1071–1096 | parses `--network_args k=v` into `net_kwargs`; `create_network(1.0, dim, alpha, vae, text_encoder, unet, neuron_dropout=..., **net_kwargs)` or `create_network_from_weights` when `--dim_from_weights` |
| Optional hooks via `hasattr` | 1097–1111 | `prepare_network`, `apply_max_norm_regularization`, then `post_process_network` |
| `network.apply_to(text_encoder, unet, train_text_encoder, train_unet)` | 1116 | monkeypatches `forward` of target Linear/Conv layers |
| `--network_weights` | 1118–1121 | `network.load_weights()` (FIXME at 1119: alpha assumed unchanged) |
| Gradient checkpointing | 1123–1134 | `unet.enable_gradient_checkpointing(cpu_offload=args.cpu_offload_checkpointing)`; TE `gradient_checkpointing_enable()`; `network.enable_gradient_checkpointing()` (no-op for LoRA) |
| Param groups | 1139–1162 | `network.prepare_optimizer_params_with_multiple_te_lrs(text_encoder_lr, unet_lr, learning_rate)` if present else `prepare_optimizer_params` → `(trainable_params, lr_descriptions)` |
| Optimizer | 1174–1175 | `optimizer_util.get_optimizer(args, trainable_params)`; `get_optimizer_train_eval_fn` (schedule-free train/eval toggles) |
| DataLoaders | 1180–1203 | `DataLoader(train_dataset_group, batch_size=1, shuffle=True, collate_fn=collator, ...)` — **the dataset yields whole batches**; DataLoader batch is always 1 |
| Steps/epochs | 1206–1215 | `max_train_steps = max_train_epochs * ceil(len(dl)/num_processes/grad_accum)` |
| LR scheduler | 1218 | `optimizer_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)` |
| full_fp16 / full_bf16 | 1221–1232 | asserts `mixed_precision` matches; `network.to(weight_dtype)` (network weights themselves in half) |
| fp8_base | 1236–1254 | casts unet to `torch.float8_e4m3fn` (plain cast, no scaling); TE to fp8 unless `--fp8_base_unet`; `prepare_text_encoder_fp8` hook keeps embeddings in half |
| Freeze base / cast | 1256–1268 | `unet.requires_grad_(False)`, `unet.to(dtype=unet_weight_dtype)`; TE frozen + cast |
| `accelerator.prepare` | 1271–1306 | DeepSpeed branch wraps everything into one `ds_model`; else `unet = prepare_unet_with_accelerator(...)` (hook), trained TEs prepared, then `network, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(...)`. `training_model = network` |
| train()/eval() | 1308–1323 | with grad-ckpt, `unet.train()`; `prepare_text_encoder_grad_ckpt_workaround` sets embeddings `requires_grad_(True)` |
| `network.prepare_grad_etc(text_encoder, unet)` | 1325 | network sets `requires_grad_(True)` on itself |
| fp16 patch | 1333–1334 | `patch_accelerator_for_fp16_training` monkeypatches GradScaler `_unscale_grads_` to allow fp16 grads |
| Save/load state hooks | 1337–1379 | `register_save_state_pre_hook`/`register_load_state_pre_hook` strip non-network models; write/read `train_state.json` (`current_epoch`, `current_step`) |
| Resume | 1382 | `args_util.resume_from_local_or_hf_if_specified` → `accelerator.load_state(args.resume)` |
| Metadata | 1409–1425 | `_build_metadata(...)` (lines 499–756) builds `ss_*` dict |
| Initial step / skip logic | 1427–1469, 1512–1516, 1582–1591 | `--initial_step/--initial_epoch/--skip_until_initial_step`; uses `accelerator.skip_first_batches` |
| Noise scheduler | 1473 | `get_noise_scheduler(args, device)` hook |
| Trackers | 1475–1479 | `logging_util.init_trackers`; three `LossRecorder`s |
| `sample_at_first` | 1503–1505 | `optimizer_eval_fn(); sample_images(...); optimizer_train_fn()` |
| Validation setup | 1533–1542 | `NUM_VALIDATION_TIMESTEPS = 4` hardcoded (TODO at 1536); `validation_timesteps = linspace(min,max,6)[1:-1]` |
| **Epoch loop** | 1573–1836 | see below |
| Final save | 1841–1863 | `checkpoint_io.get_last_ckpt_name`, `_save_model(force_sync_upload=True)`; optional `save_state_on_train_end` |

**Inner step (1587–1756):**
```python
for step, batch in enumerate(skipped_dataloader or train_dataloader):
    with accelerator.accumulate(training_model):                    # 1593
        on_step_start_for_network(text_encoder, unet)               # network hook (hasattr)
        self.on_step_start(args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train=True)
        loss = self.process_batch(batch, ..., is_train=True, train_text_encoder=..., train_unet=...)  # 1599
        accelerator.backward(loss)                                  # 1617
        if accelerator.sync_gradients:
            self.all_reduce_network(accelerator, network)           # manual DDP grad mean (225-228)
            if args.max_grad_norm != 0.0:
                accelerator.clip_grad_norm_(network.get_trainable_params(), args.max_grad_norm)
            network.update_grad_norms()/update_norms() if hasattr
        optimizer.step(); lr_scheduler.step(); optimizer.zero_grad(set_to_none=True)   # 1629-1631
    if args.scale_weight_norms: network.apply_max_norm_regularization(...)               # 1633-1636
    if accelerator.sync_gradients:
        global_step += 1; sample_images(...); save every N steps (+ save_state, rotate)   # 1658-1690
    loss_recorder.add(...); step logs                                                      # 1692-1712
    if validate_every_n_steps: _run_validation_loop(mode="step")                          # 1716-1753
```
Epoch tail: epoch validation (1759–1799), epoch logging (1802–1804), `save_every_n_epochs` + rotation + `save_state` (1809–1830), epoch sample images (1832).

`process_batch` (371–488): VAE-encode if no cached latents (optionally chunked by `vae_batch_size`), NaN→0, `shift_scale_latents`; fetch `text_encoder_outputs_list` from batch or encode via `text_encoding_strategy.encode_tokens` (weighted-caption path 433–440); `get_noise_pred_and_target` → `(noise_pred, target, timesteps, weighting)`; `loss_util.conditional_loss(pred.float(), target.float(), loss_type, "none", huber_c)` (476); `* weighting`; masked loss (`apply_masked_loss`) if `--masked_loss` or `alpha_masks`; mean over non-batch dims; `* batch["loss_weights"]` (prior-loss weight per sample); `post_process_loss` hook; `.mean()`.

### 1.2 Hook methods subclasses override (all in `train_network.py`)

| Hook | Line | Default |
|---|---|---|
| `assert_extra_args(args, train_ds, val_ds)` | 162 | bucket steps % 64 |
| `load_target_model(args, weight_dtype, accelerator)` | 172 | SD1/2 via `model_io.load_target_model` |
| `load_unet_lazily(args, weight_dtype, accelerator, text_encoders)` | 182 | `NotImplementedError` |
| `get_tokenize_strategy` / `get_tokenizers` | 185/188 | SD CLIP |
| `get_latents_caching_strategy` | 191 | `SdSdxlLatentsCachingStrategy` |
| `get_text_encoding_strategy` / `get_text_encoder_outputs_caching_strategy` | 197/200 | SD / None |
| `get_models_for_text_encoding` | 203 | pass-through |
| `get_text_encoders_train_flags` / `is_train_text_encoder` | 211/214 | `not network_train_unet_only` |
| `cache_text_encoder_outputs_if_needed` | 217 | move TE to device |
| `call_unet` | 221 | `unet(noisy, t, cond).sample` |
| `all_reduce_network` | 225 | manual mean-reduce of grads |
| `sample_images` | 230 | `sampling.sample_images` (SD pipeline) |
| `post_process_network` | 235 | no-op |
| `get_noise_scheduler` | 238 | `DDPMScheduler(scaled_linear, 1000)` + `zero_terminal_snr` |
| `encode_images_to_latents` / `shift_scale_latents` | 247/250 | `vae.encode().latent_dist.sample()` / `*0.18215` |
| `get_noise_pred_and_target` | 253 | DDPM eps/v-pred + differential output preservation (306–327) |
| `post_process_loss` | 331 | min_snr / v-pred scaling / debiased |
| `get_sai_model_spec` / `update_metadata` | 342/345 | SD spec / no-op |
| `is_text_encoder_not_needed_for_training` | 348 | False |
| `prepare_text_encoder_grad_ckpt_workaround` / `prepare_text_encoder_fp8` | 351/355 | CLIP-specific |
| `prepare_unet_with_accelerator` | 358 | `accelerator.prepare(unet)` |
| `on_step_start` / `on_validation_step_end` | 363/366 | no-op |
| `cast_text_encoder` / `cast_vae` / `cast_unet` | 490–497 | True |

### 1.3 Mechanisms

- **Gradient accumulation:** `accelerator.accumulate(training_model)` (1593); `accelerator.sync_gradients` gates clipping/step counting; `Accelerator(gradient_accumulation_steps=...)` in `library/accelerator_setup.py:143-151`.
- **Mixed precision:** `accelerator.autocast()` wraps TE encode (431) and UNet call (287); `weight_dtype` from `prepare_dtype` (`accelerator_setup.py:156-171`). `full_fp16/full_bf16` put the *network* into half (1226/1232) and patch the GradScaler (`accelerator_setup.py:174-186`).
- **Gradient checkpointing:** delegated to model (`unet.enable_gradient_checkpointing(cpu_offload=...)`, 1123–1127). For Anima: `anima_models.py:1164-1166` → per-block `Block.forward` uses `torch.utils.checkpoint(use_reentrant=False)` (1007), a CPU-offload variant (981–1004), or the Unsloth async offloader (`anima_models.py:75-124`, autograd.Function with `non_blocking=True` CPU copies). Requires inputs with `requires_grad` → `noisy_latents.requires_grad_(True)` at 272–276 / `anima_train_network.py:304-308`.
- **Fused backward pass:** only in *full fine-tune* scripts (`anima_train.py:390-409`): `library/adafactor_fused.py:136 patch_adafactor_fused(optimizer)` then `parameter.register_post_accumulate_grad_hook` calling `optimizer.step_param(tensor, group)` and freeing `.grad`. Restricted to Adafactor with `gradient_accumulation_steps == 1` (`optimizer.py:57-63`). Not available in `train_network.py`. `--fused_optimizer_groups` exists only in `sdxl_train.py:381-...`.
- **Block-wise LR:** for SD/SDXL LoRA via `networks/lora.py:389 parse_block_lr_kwargs` (`down_lr_weight/mid_lr_weight/up_lr_weight`, `block_lr_zero_threshold`) and `get_block_index` (719). For Anima: regex-based `network_reg_lrs` / `network_reg_dims` (`networks/lora_anima.py:298-308`, 637–733); full-FT: `anima_train_utils.get_anima_param_groups` (285–370) gives 6 groups (base/self_attn/cross_attn/mlp/mod/llm_adapter) with `lr==0` → freeze.
- **Multi-GPU:** `accelerate` DDP; `InitProcessGroupKwargs` (nccl/gloo, `--ddp_timeout`), `DistributedDataParallelKwargs(gradient_as_bucket_view, static_graph)` (`accelerator_setup.py:120-140`); DeepSpeed via `deepspeed_utils.prepare_deepspeed_plugin/prepare_deepspeed_model` (`library/deepspeed_utils.py:72,121`) which wraps unet+TE+network into one module. Caching is sharded by `i % num_processes == process_index` (`dataset.py:816, 883`). Sample generation splits prompts with `PartialState.split_between_processes` (`anima_train_utils.py:599-608`). Because LoRA forward runs through monkeypatched base-model forwards, DDP reducer doesn't fire → manual `all_reduce_network` (225).

---

## 2. Anima specifics

Files: `$ROOT/anima_train_network.py` (485), `$ROOT/anima_train.py` (769), `$ROOT/library/anima_models.py` (1671), `$ROOT/library/anima_utils.py` (320), `$ROOT/library/anima_train_utils.py` (768), `$ROOT/library/strategy_anima.py` (304), `$ROOT/anima_minimal_inference.py` (1080), `$ROOT/networks/lora_anima.py` (847), `$ROOT/networks/convert_anima_lora_to_comfy.py` (160), `$ROOT/networks/control_net_lllite_anima.py` (988), `$ROOT/anima_train_control_net_lllite.py`.

### 2.1 DiT architecture (`library/anima_models.py`)

Fixed config hardcoded in `anima_utils.load_anima_model` (`anima_utils.py:67-98`) — **not** auto-detected despite docs (`get_dit_config` is commented out at `anima_models.py:1621-1671`):

```python
dit_config = { "max_img_h": 512, "max_img_w": 512, "max_frames": 128,
  "in_channels": 16, "out_channels": 16, "patch_spatial": 2, "patch_temporal": 1,
  "model_channels": 2048, "concat_padding_mask": True, "crossattn_emb_channels": 1024,
  "pos_emb_cls": "rope3d", "pos_emb_learnable": True, "pos_emb_interpolation": "crop",
  "use_adaln_lora": True, "adaln_lora_dim": 256, "num_blocks": 28, "num_heads": 16,
  "extra_per_block_abs_pos_emb": False,
  "rope_h_extrapolation_ratio": 4.0, "rope_w_extrapolation_ratio": 4.0, "rope_t_extrapolation_ratio": 1.0,
  "rope_enable_fps_modulation": False, "use_llm_adapter": True, ... }
```

- `class Anima(nn.Module)` (1033–1390; Cosmos-Predict2 "MiniTrainDIT"). `LATENT_CHANNELS = 16`. Input is 5-D `(B, C, T=1, H, W)`; trainers `unsqueeze(2)` / `squeeze(2)` around the call (`anima_train_network.py:328,339`).
- **Patch embed** `PatchEmbed` (650–689): einops rearrange `b c (t r) (h m) (w n) -> b t h w (c r m n)` + bias-free Linear(17*2*2 → 2048). `concat_padding_mask=True` appends a 1-channel mask → 17 input channels (`build_patch_embed` 1180–1187). Trainers pass an all-zeros `padding_mask` `(B,1,h,w)` (`anima_train_network.py:325`).
- **Positional embedding:** `VideoRopePosition3DEmb` (394–505): head_dim=128 split `dim_h = 128//6*2 = 42, dim_w = 42, dim_t = 44`; theta = `10000 * ntk_factor`, `h_ntk_factor = 4.0 ** (42/40)`; embeddings concatenated `[t,h,w]*2` and returned as `(L,1,1,D)` (491–501). `pos_emb_learnable=True` is passed but only affects kwargs (RoPE class ignores it; `LearnablePosEmbAxis` only used when `extra_per_block_abs_pos_emb`). RoPE applied to q/k **only in self-attention** (`Attention.compute_qkv` 349–351).
- **Timestep embedding:** `Timesteps` sinusoidal (560–582) → `TimestepEmbedding` (585–620) with `use_adaln_lora=True`: `linear_1(2048→2048, bias=False)`, SiLU, `linear_2(2048→3*2048)`; returns `(emb_B_T_D=raw sinusoid, adaln_lora_B_T_3D)`. `t_embedding_norm = RMSNorm(2048)` applied to emb (1336). Timesteps expected in **[0,1]** (trainers divide by 1000: `anima_train_network.py:301`).
- **Block** (762–1029): pre-LN (`nn.LayerNorm(elementwise_affine=False, eps=1e-6)`) with AdaLN-LoRA modulation: `adaln_modulation_{self_attn,cross_attn,mlp} = Sequential(SiLU, Linear(2048→256,bias=False), Linear(256→3*2048,bias=False))` (801–816), output added to the shared `adaln_lora_B_T_3D` and chunked into shift/scale/gate. Each sublayer: `x = x + gate * sublayer(norm(x)*(1+scale)+shift)` (914–952). Order: self-attn (RoPE) → cross-attn to `crossattn_emb` (1024-d context) → `GPT2FeedForward` (GELU MLP, 2048→8192→2048, bias-free, 242–269).
- **Attention** (273–373): bias-free q/k/v/out projections, `q_norm`/`k_norm = RMSNorm(head_dim, eps=1e-6)` (QK-norm), no v-norm; dispatch through `library/attention.py:89 attention(qkv, attn_params)` supporting `torch` SDPA / `xformers` / `flash` / `sageattn` and `split_attn` (`AttentionParams` at `attention.py:31-86`). fp16 stability: `use_fp32 = x.dtype == torch.float16` (1347) casts residual stream to fp32 and modulations under fp32 autocast.
- **FinalLayer** (693–758): LN + AdaLN(shift,scale) → Linear(2048 → 2*2*1*16). `unpatchify` (1244–1252).
- **LLM Adapter** (1394–1616): `LLMAdapter(source_dim=1024, target_dim=1024, model_dim=1024, num_layers=6, self_attn=True)` (1118–1125). Embeds **T5 token ids** (`nn.Embedding(32128, 1024)`), 6× `LLMAdapterTransformerBlock` (RMSNorm; self-attn with RoPE on T5 positions; cross-attn to Qwen3 hidden states with separate RoPE for context; GELU MLP with bias), then `out_proj` + RMSNorm. Output is the DiT cross-attention context; padding positions zeroed via `context[~target_attention_mask] = 0` (1387). Two invocation paths exist: `forward()`→`_preprocess_text_embeds` (1362–1390, used by `anima_train_network.py`) and inside `forward_mini_train_dit` when `t5_input_ids` kwarg is set (1317–1325, used by `anima_train.py:566-573`).
- **Forward** (`forward_mini_train_dit` 1294–1360): adapter → patch embed + RoPE → t-embed → 28 blocks (block-swap `wait_for_block`/`submit_move_blocks` at 1350–1356) → final layer → unpatchify. Note the DiT cross-attention has **no key mask**; padded context tokens are zero vectors that still receive softmax mass (sequence always 512).

### 2.2 Text encoder handling (Qwen3-0.6B)

- Loader `anima_utils.load_qwen3_text_encoder` (192–268): `transformers.AutoModelForCausalLM.from_pretrained(...).model` (i.e. `Qwen3Model` without LM head) from a directory, or `Qwen3ForCausalLM(Qwen3Config)` from bundled `configs/qwen3_06b/config.json` + single safetensors (strips `model.` prefix, 250–256). `use_cache=False`, frozen. Config: hidden_size 1024, 28 layers, 16 heads / 8 KV heads, vocab 151936 (`configs/qwen3_06b/config.json`). `pad_token = eos_token` if missing (186–187).
- Tokenization `AnimaTokenizeStrategy` (`strategy_anima.py:24-71`): Qwen3 tokenizer **and** T5 tokenizer (`configs/t5_old/spiece.model`, `T5TokenizerFast`), both `padding="max_length", truncation=True, max_length=512` (`--qwen3_max_token_length`, `--t5_max_token_length`, defaults 512 at `anima_train_utils.py:85-95`). Returns `[qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask]`. No chat template, no weighted-caption support (`tokenize_with_weights` → `NotImplementedError` from base).
- Encoding `AnimaTextEncodingStrategy.encode_tokens` (84–109): `outputs.last_hidden_state` (**final layer only**, no intermediate hidden states / no `output_hidden_states`), then `prompt_embeds[~attn_mask] = 0`. Returns `[prompt_embeds(B,512,1024), qwen3_attn_mask, t5_input_ids, t5_attn_mask]`.
- Caption dropout with cached outputs: `drop_cached_text_encoder_outputs` (111–150) zeroes embeds/masks and sets T5 ids to `[</s>=1, 0...]` per sample with prob `caption_dropout_rate` stored in the cache. Called from `AnimaNetworkTrainer.process_batch` (`anima_train_network.py:349-399`), which pops the trailing `caption_dropout_rate` tensor.
- TE LoRA: allowed only when `--network_train_unet_only` absent and no TE caching (`anima_train_network.py:65-67`); targets `Qwen3Attention/Qwen3MLP/...` with prefix `lora_te` (`lora_anima.py:385,388`).

### 2.3 VAE

`--vae` → `anima_train_utils.load_qwen_image_vae` (201–222) → `library/qwen_image_autoencoder_kl.AutoencoderKLQwenImage` (857–) (Wan2.1-architecture causal-3D VAE, weights published as Qwen-Image VAE): `base_dim=96, z_dim=16, dim_mult=[1,2,4,4], temperal_downsample=[F,T,T]` → spatial compression 8. Normalization uses **per-channel `latents_mean`/`latents_std`** (877–912); `encode_pixels_to_latents` (1194–1227) takes `[-1,1]` pixels, uses `posterior.mode()` (deterministic), returns `(latents - mean) / std`, 4-D in/out. `decode_to_pixels` (1178–1192) inverts and clamps. `shift_scale_latents` is identity (`anima_train_network.py:264-266`). Options: `--vae_chunk_size` (spatial chunking), `--vae_disable_cache`, `--qwen_image_vae_2d` (2-D conversion in `qwen_image_autoencoder_kl_2d.py`). Bucket resolution must be multiple of 16 (`anima_train_network.py:94`).

### 2.4 Flow-matching formulation

- Scheduler: `sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)` (`anima_train_network.py:256-258`; class at `sd3_train_utils.py:628-864`): `sigmas = t/1000; sigmas = shift*sigmas/(1+(shift-1)*sigmas); timesteps = sigmas*1000` (654–660).
- Noisy input (`flux_train_utils.get_noisy_model_input_and_timesteps` 473–539, shared with FLUX):
```python
if timestep_sampling in ("uniform","sigmoid"):
    r = randn(bsz) (+ per-sample offset); sigmas = sigmoid(sigmoid_scale * r)   # or rand for uniform
elif "shift":   sigmas = sigmoid(randn*sigmoid_scale); sigmas = shift*s/(1+(shift-1)*s)
elif "flux_shift": mu = lin(256→0.5, 4096→1.15)((h//2)*(w//2)); sigmas = time_shift(mu,1.0,sigmoid(...))
else ("sigma"):  u = compute_density_for_timestep_sampling(weighting_scheme, logit_mean, logit_std, mode_scale)
                 indices=(u*1000).long(); timesteps=scheduler.timesteps[indices]; sigmas=get_sigmas(...)
sigmas = sigmas.view(-1,1,1,1)
noisy = (1-sigmas)*latents + sigmas*(noise [+ ip_noise_gamma*xi])
return noisy.to(dtype), timesteps.to(dtype), sigmas
```
  `--discrete_flow_shift` only affects `sigma` and `shift` modes (`_SHIFT_AWARE_TIMESTEP_SAMPLING` 546); per-subset `custom_attributes.timestep_sampling.offset` applies to `sigmoid/shift/flux_shift` (550; read at `anima_train_network.py:291-296`, docs `docs/timestep_sampling_offset.md`). Defaults for Anima: `timestep_sampling=sigmoid`, `discrete_flow_shift=1.0`, `sigmoid_scale=1.0` (`anima_train_utils.py:96-114`).
- Prediction/target (`anima_train_network.py:341-347`): **velocity** `target = noise - latents`; `model_pred` raw (no `model_prediction_type` option, unlike FLUX). Loss weighting `compute_loss_weighting_for_anima(weighting_scheme, sigmas)` (`anima_train_utils.py:228-242`): `sigma_sqrt → sigmas**-2`, `cosmap → 2/(π(1-2σ+2σ²))`, else ones. Applied in `process_batch` before spatial mean (`train_network.py:477-478`). `post_process_loss` is a no-op for Anima (401–402), so `min_snr_gamma`, `debiased_estimation_loss`, `v_pred_like_loss` are accepted but ignored.
- Timesteps passed to the model are `timesteps/1000` (301). Loss: `conditional_loss` L2 by default (`loss.py:98-130`).

### 2.5 State-dict key naming

- **Base DiT:** module keys as defined in `Anima` (e.g. `blocks.0.self_attn.q_proj.weight`, `blocks.0.adaln_modulation_self_attn.1.weight`, `x_embedder.proj.1.weight`, `t_embedder.1.linear_1.weight`, `final_layer.linear.weight`, `llm_adapter.blocks.0.cross_attn.k_proj.weight`, `llm_adapter.embed.weight`). Loader strips `net.` (anima-base) or `model.diffusion_model.` (anima-aesthetics/ComfyUI) prefixes (`anima_utils.py:107-116`) and is strict except RoPE buffers (140–157). Saving re-adds `net.` (`save_anima_model` 297–320).
- **LoRA (sd-scripts format):** `lora_unet_<module path with . → _>.lora_down.weight / .lora_up.weight / .alpha` (`lora_anima.py:387, 473`), e.g. `lora_unet_blocks_0_self_attn_q_proj.lora_down.weight`; TE: `lora_te_layers_0_self_attn_q_proj...`. Default excludes regex `.*(_modulation|_norm|_embedder|final_layer).*` (254). ComfyUI conversion: `networks/convert_anima_lora_to_comfy.py` → `diffusion_model.<dotted>.lora_A/lora_B` and `text_encoders.qwen3_06b.transformer.model.<...>` (14–15). LoKr/LoHa: `lora_unet_<...>.lokr_w1 / lokr_w2_a / lokr_w2_b / lokr_t2 / alpha` and `hada_w1_a/...` (`networks/lokr.py:139-165`).
- Metadata `ss_*` + `modelspec.*` with `modelspec.architecture = "anima-preview"` and `implementation = "https://huggingface.co/circlestone-labs/Anima"` (`sai_model_spec.py:84,97`).

### 2.6 fp8 and block swap

- Training: `--fp8_base/--fp8_base_unet` are force-disabled with a warning; `args.fp8_scaled=False` (`anima_train_network.py:50-54`). Inference only: `--fp8_scaled` → `load_safetensors_with_lora_and_fp8(..., fp8_optimization=True, target_keys=["blocks",""], exclude_keys=["_embedder","norm","adaln","final_layer",".embed."])` (`anima_utils.py:27-29, 118-132`) → `apply_fp8_monkey_patch` replaces `nn.Linear.forward` with dequantize-on-the-fly (`fp8_optimization_utils.py:355-482`, per-tensor/channel/block `scale_weight` buffers; `_scaled_mm` path untested).
- Block swap: `Anima.enable_block_swap(n, device)` (1254–1262, max `num_blocks-2`) → `custom_offloading_utils.ModelOffloader(self.blocks, n, device)`; `move_to_device_except_swap_blocks` (1264–1273); `prepare_block_swap_before_forward` (1289–1292); trainer prepares model with `accelerator.prepare(model, device_placement=[False])` (`anima_train_network.py:432-438`) and re-arms swap after each validation step (451–454). Incompatible with `cpu_offload_checkpointing` and `unsloth_offload_checkpointing` (69–82). `--compile` per-block `torch.compile` via `compile_utils.compile_transformer` (443–447) with Linear layers excluded under swap (`compile_utils.py:28-40`).

### 2.7 `anima_train.py` (full fine-tune) differences

Standalone 700-line procedural loop (no class): loads Qwen3 + tokenizers first (177–194), caches TE outputs then **deletes the encoder** (228), loads VAE, loads DiT on CPU with `dit_weight_dtype=None` (249–251), param groups via `get_anima_param_groups` (277–285), `full_bf16/full_fp16` else fp32 master weights (340–349), optional `--fused_backward_pass` (390–409), block swap w/o DDP placement (372–375), loss with weighting applied after spatial mean (571–580), saves full DiT with `net.` prefix via `anima_train_utils.save_anima_model_on_*` (632, 672, 709). Passes `noise_scheduler=None` to `get_huber_threshold_if_needed` (569).

### 2.8 `anima_minimal_inference.py`

Loads DiT with optional LoRA/LoHa/LoKr *merged into weights at load* via `load_safetensors_with_lora_and_fp8` (`load_dit_model` 235–299; keeps only `lora_unet_` keys 261), or LyCORIS (`lycoris.kohya.create_network_from_weights`, 30–32). Text encode path (358–469) runs `_preprocess_text_embeds` once and caches. Denoising (518–590): `hunyuan_image_utils.get_timesteps_sigmas(steps, flow_shift)`, `t/1000`, CFG with two passes, `hunyuan_image_utils.step` Euler. Latents `(1,16,1,H/8,W/8)` bf16.

---

## 3. `library/` structure

### 3.1 Module inventory (77 files)

| Module | Purpose |
|---|---|
| `accelerator_setup.py` | `prepare_accelerator`, `prepare_dtype`, `prepare_dataset_args`, fp16 GradScaler patch, mutable `HIGH_VRAM` flag |
| `adafactor_fused.py` | Fused Adafactor `step_param` for per-parameter backward-time optimizer steps |
| `anima_models.py` | Anima/Cosmos DiT + LLM adapter + Unsloth checkpointer |
| `anima_train_utils.py` | Anima args, VAE loader, loss weighting, param groups, save, Euler sampler, sample_images |
| `anima_utils.py` | Anima DiT/Qwen3/T5-tokenizer loading, fp8 key lists, `net.`-prefixed save |
| `args.py` | All `add_*_arguments`, `verify_*`, `read_config_from_file`, resume helper |
| `attention.py` | Unified `attention()` (torch/xformers/flash/sageattn, split_attn, varlen) + `AttentionParams` |
| `attention_processors.py` | Legacy FlashAttention processor for diffusers UNet |
| `caching.py` | Image loading for caching; legacy SD npz latent caching |
| `checkpoint_io.py` | Checkpoint/state filename templates, save & rotate |
| `chroma_models.py` | FLUX Chroma variant model |
| `compile_utils.py` | Per-block `torch.compile`, TF32/cudnn switches |
| `config_util.py` | TOML dataset config schema (voluptuous), blueprints → datasets |
| `controlnet_dataset.py` | `ControlNetDataset` |
| `custom_offloading_utils.py` | Block-swap `Offloader`/`ModelOffloader`, CPU offload wrappers |
| `custom_train_functions.py` | SNR weights, zero-terminal-SNR, noise offset, multires noise, masked loss, weighted captions (SD) |
| `dataset.py` | `ImageInfo`, `BucketManager`, `BaseDataset`, `DatasetGroup`, `MinimalDataset`, collator |
| `deepspeed_utils.py` | DeepSpeed args/plugin/model wrapper |
| `device_utils.py` | `clean_memory_on_device`, `synchronize_device`, `get_preferred_device`, `init_ipex` |
| `dreambooth_dataset.py` | `DreamBoothDataset` |
| `finetuning_dataset.py` | `FineTuningDataset` (JSON/JSONL metadata) |
| `flux_models.py`, `flux_train_utils.py`, `flux_utils.py` | FLUX.1 model / training helpers (timestep sampling shared with Anima) |
| `fp8_optimization_utils.py` | Scaled fp8 quantization + Linear monkey-patch |
| `hidden_states.py` | CLIP hidden-state extraction (SD/SDXL) |
| `huggingface_util.py` | HF Hub upload/list |
| `hunyuan_image_*.py` (5) | HunyuanImage-2.1 model/TE/VAE/utils |
| `hypernetwork.py` | Legacy hypernetwork attention patch |
| `ipex/` | Vendored Intel XPU shims |
| `jpeg_xl_util.py` | JXL header size parser |
| `leco_train_util.py` | LECO (concept erasing) training |
| `logging_util.py` | `init_trackers`, `LossRecorder` |
| `lora_utils.py` | Load safetensors while merging LoRA/LoHa/LoKr + fp8 |
| `loss.py` | DDPM timestep/noise sampling, Huber threshold, `conditional_loss` |
| `lpw_stable_diffusion.py`, `sdxl_lpw_stable_diffusion.py` | Long-prompt-weighting pipelines (sample gen for SD/SDXL) |
| `lumina_*.py` (4) | Lumina Image 2.0 |
| `mask_generator.py` | Random inpainting masks |
| `model_io.py` | SD model loading, hashes, `ss_*` keys, SAI spec wrappers |
| `model_util.py` | SD1/2 ckpt↔diffusers conversion, `make_bucket_resolutions`, VAE load |
| `optimizer.py` | `get_optimizer`, `get_scheduler_fix`, schedule-free helpers |
| `original_unet.py`, `sdxl_original_unet.py`, `sdxl_original_control_net.py` | Reimplemented SD UNets/ControlNet |
| `qwen_image_autoencoder_kl.py`, `qwen_image_autoencoder_kl_2d.py` | Wan/Qwen-Image VAE (3D causal) and 2D port |
| `safetensors_utils.py` | Memory-efficient safetensors save/open, split weights, `WeightTransformHooks` |
| `sai_model_spec.py` | ModelSpec 1.0.1 metadata dataclass/builders, `--metadata_*` args |
| `sampling.py` | Prompt file parsing, SD sample generation |
| `sd3_models.py`, `sd3_train_utils.py`, `sd3_utils.py` | SD3 (hosts `FlowMatchEulerDiscreteScheduler`) |
| `sdxl_model_util.py`, `sdxl_train_util.py` | SDXL loading/args |
| `slicing_vae.py` | Sliced SD VAE |
| `strategy_base.py` | Strategy interfaces |
| `strategy_{sd,sdxl,sd3,flux,lumina,hunyuan_image,anima}.py` | Per-model strategies |
| `subset.py` | `BaseSubset`, `DreamBoothSubset`, `FineTuningSubset`, `ControlNetSubset` |
| `timestep_visualization.py` | ASCII/matplotlib timestep histogram |
| `train_util.py` | Backward-compat re-export shim only |
| `utils.py` | logging setup, `str_to_dtype`, `IMAGE_TRANSFORMS`, image load/resize, `GradualLatent` |

### 3.2 `strategy_base.py` interfaces (`$ROOT/library/strategy_base.py`)

All four are singletons via a class attribute; `set_strategy` raises if already set (43–46, 288–292, 337–341, 390–394).

```python
class TokenizeStrategy:                                  # line 21
    _strategy = None
    @classmethod set_strategy(cls, strategy) / get_strategy(cls)
    def tokenize(self, text: str|List[str]) -> List[torch.Tensor]: raise NotImplementedError          # 71
    def tokenize_with_weights(self, text) -> Tuple[List[Tensor], List[Tensor]]: raise NotImplementedError  # 74
    # helpers: _load_tokenizer (52), _get_weighted_input_ids (80, A1111 attention syntax), _get_input_ids (219, CLIP 75/150/225 chunking)

class TextEncodingStrategy:                              # 285
    def encode_tokens(self, tokenize_strategy, models: List[Any], tokens: List[Tensor]) -> List[Tensor]      # 298
    def encode_tokens_with_weights(self, tokenize_strategy, models, tokens, weights) -> List[Tensor]         # 308

class TextEncoderOutputsCachingStrategy:                 # 320
    def __init__(self, cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial=False, is_weighted=False)
    def get_outputs_npz_path(self, image_abs_path) -> str                # 363
    def load_outputs_npz(self, npz_path) -> List[np.ndarray]             # 366
    def is_disk_cached_outputs_expected(self, npz_path) -> bool          # 369
    def cache_batch_outputs(self, tokenize_strategy, models, text_encoding_strategy, batch: List[ImageInfo])  # 372

class LatentsCachingStrategy:                            # 378
    def __init__(self, cache_to_disk, batch_size, skip_disk_cache_validity_check)
    cache_suffix (property, 409); get_image_size_from_disk_cache_path (412: parses "_WxH" from filename)
    def get_latents_npz_path(self, absolute_path, image_size) -> str     # 416
    def is_disk_cached_latents_expected(self, bucket_reso, npz_path, flip_aug, alpha_mask) -> bool  # 419
    def cache_batch_latents(self, model, batch, flip_aug, alpha_mask, random_crop)                  # 424
    # defaults: _default_is_disk_cached_latents_expected(latents_stride,...,multi_resolution) 427
    #           _default_cache_batch_latents(encode_by_vae, vae_device, vae_dtype, image_infos, ...) 478
    #           _default_load_latents_from_disk(latents_stride, npz_path, bucket_reso) 564
    #           save_latents_to_disk(npz_path, latents, original_size, crop_ltrb, flipped, alpha_mask, key_reso_suffix) 608
```

### 3.3 Datasets (`library/dataset.py`, `dreambooth_dataset.py`, `finetuning_dataset.py`, `subset.py`)

- `ImageInfo` (146–178): per-image record — key, `num_repeats`, caption, `is_reg`, path, `caption_dropout_rate`, sizes, `bucket_reso`, in-memory latents/flipped/alpha, `latents_npz`, `text_encoder_outputs(_npz)`.
- `BucketManager` (181–316): `make_buckets()` → `model_util.make_bucket_resolutions(max_reso, min, max, steps)`; `select_bucket(w,h)` (251–311) picks nearest aspect-ratio predefined bucket (upscale mode) or derives a bucket from image size rounded to `reso_steps` (`bucket_no_upscale`); `sort()`, `shuffle()`.
- `BaseDataset` (362–1268): `process_caption` (501–612) implements prefix/suffix, `caption_dropout_rate`, `caption_dropout_every_n_epochs`, wildcard `{a|b}` + multi-line random choice (`enable_wildcard`), `keep_tokens`/`keep_tokens_separator` (fixed prefix and optional fixed suffix), `token_warmup_min/step`, `caption_tag_dropout_rate`, `shuffle_caption`, `secondary_separator`, TI replacements. `make_buckets` (618–715) assigns buckets, repeats images `num_repeats` times, builds `buckets_indices: List[BucketBatchIndex]` of `(bucket_index, batch_size, batch_index)`; `shuffle_buckets` seeds `random` with `seed + epoch` (717–722). `__getitem__(index)` (989–1242) returns a **whole batch dict**: `latents`/`images`, `captions`, `input_ids_list`, `text_encoder_outputs_list` (padded stack via `none_or_stack_elements` 1162–1194), `loss_weights` (prior_loss_weight for reg), `alpha_masks`, `masks/masked_images` (inpainting), `original_sizes_hw/crop_top_lefts/target_sizes_hw`, `flippeds`, `network_multipliers`, `custom_attributes`.
- Caching: `new_cache_latents` (747–852) sorts by bucket area, groups by `Condition(reso, flip_aug, alpha_mask, random_crop)`, shards by process, loads images in a `ThreadPoolExecutor`, delegates to strategy. `new_cache_text_encoder_outputs` (854–907).
- `DatasetGroup(ConcatDataset)` (1271–1352): fan-out of set_* methods; `is_latent_cacheable` (no color_aug/random_crop), `is_text_encoder_output_cacheable(cache_supports_dropout)` (no shuffle/tag-dropout/token-warmup).
- `collator_class` (1554–1571): sets epoch/step on the worker's dataset copy, returns `examples[0]`.
- `DreamBoothDataset` (`dreambooth_dataset.py:30-346`): per-subset directory glob (`glob_images`), caption files `<stem><caption_extension>` (first line, or all lines when wildcard), `class_tokens` fallback, `metadata_cache.json` (`cache_info`), image sizes from latent-cache filenames (146–179), `skip_image_resolution` filtering, train/val split (`split_train_val`, `dataset.py:107-143`), **regularization balancing**: reg images' `num_repeats` are inflated round-robin until `num_reg >= num_train` (328–344); `loss_weights = prior_loss_weight` for reg samples (`dataset.py:1016`).
- `FineTuningDataset` (`finetuning_dataset.py:27-`): JSON `{path: {"caption","tags","train_resolution",...}}` or JSONL rows with `image_path/caption/image_size`; uses `_{W}x{H}<suffix>` npz.
- Cache file formats: latents `<stem>_<W:04d>x<H:04d>_anima.npz` with keys `latents_<h>x<w>`, `original_size_<h>x<w>`, `crop_ltrb_<h>x<w>`, `latents_flipped_*`, `alpha_mask_*` (float32 via `np.savez`, uncompressed; `strategy_base.py:608-647`). TE cache `<stem>_anima_te.npz` with `prompt_embeds`(float32 512×1024), `attn_mask`, `t5_input_ids`(int32), `t5_attn_mask`, `caption_dropout_rate` (`strategy_anima.py:159, 239-247`).
- Config TOML schema (`config_util.py`): dataclasses `BaseSubsetParams`(55–79), `DreamBoothSubsetParams`(83), `FineTuningSubsetParams`(92), `ControlNetSubsetParams`(98), `BaseDatasetParams`(105), `DreamBoothDatasetParams`(116, has `prior_loss_weight`), etc. `ConfigSanitizer` schemas: `SUBSET_ASCENDABLE_SCHEMA` (186–204: color_aug, face_crop_aug_range, flip_aug, num_repeats, random_crop, shuffle_caption, keep_tokens(_separator), secondary_separator, caption_separator, enable_wildcard, token_warmup_*, caption_prefix/suffix, `custom_attributes: dict`, resize_interpolation), dropout schema (206–210), DB-specific (212–221: caption_extension, class_tokens, cache_info, image_dir, is_reg, alpha_mask), FT (223–227: metadata_file), CN (228–235), `DATASET_ASCENDABLE_SCHEMA` (238–251: batch_size, bucket_*, enable_bucket, validation_seed/split, resolution, network_multiplier, resize_interpolation, skip_image_resolution). Values cascade `subset → dataset → general → argparse` (`generate_params_by_fallbacks` 459–477).

### 3.4 `custom_offloading_utils.py` (block swap)

`swap_weight_devices_cuda` (34–75): pairs modules by name, swaps `.weight.data` between a GPU block and a CPU block on a dedicated CUDA stream with `non_blocking=True`, using the GPU tensor storage as the landing buffer (`cuda_data_view.copy_`). `Offloader` (109–163): single-worker `ThreadPoolExecutor`, `futures` per block index. `ModelOffloader` (170–276): registers `register_full_backward_hook` on blocks (208–234) so that during backward, block `num_blocks - k` moving to CPU triggers block `blocks_to_swap - k` onto GPU; `prepare_block_devices_before_forward` (236–256) keeps first `N - blocks_to_swap` on GPU, last `blocks_to_swap` weights on CPU; `submit_move_blocks` (263–276) in forward swaps block `i` out and `N - blocks_to_swap + i` in; `set_forward_only` for inference. Also generic `to_device/to_cpu/create_cpu_offloading_wrapper` (284–342).

### 3.5 `custom_train_functions.py` (DDPM-era loss shaping)

`prepare_scheduler_for_custom_training` (16, computes `all_snr`), `fix_noise_scheduler_betas_for_zero_terminal_snr` (30), `apply_snr_weight` (68, min-SNR-γ; `/(snr+1)` for v-pred), `scale_v_prediction_loss_like_noise_prediction` (79), `add_v_prediction_like_loss` (94), `apply_debiased_estimation` (101, `1/sqrt(snr)` or `1/(snr+1)`), args (115–165: `--min_snr_gamma --scale_v_pred_loss_like_noise_pred --v_pred_like_loss --debiased_estimation_loss --weighted_captions`), `pyramid_noise_like` (458, multires noise), `apply_noise_offset` (471, + adaptive scale), `apply_masked_loss` (487, uses conditioning image R channel or `alpha_masks`, area-interpolated to loss shape), weighted-embedding helpers for CLIP (167–455). None of the SNR functions apply to flow matching.

### 3.6 Other

- `device_utils.py`: `clean_memory_on_device` (42, gc + empty_cache for cuda/xpu/mps), `synchronize_device` (60), `get_preferred_device` (74), `init_ipex` (90).
- `model_io.py` / `sai_model_spec.py`: see §7.
- `utils.py`: `setup_logging` (41, rich handler), `str_to_dtype` (98, incl. fp8 names), `IMAGE_TRANSFORMS = ToTensor+Normalize(0.5,0.5)` (162), `load_image` (170), `get_crop_ltrb` (186, SDXL-style center crop coords), `trim_and_resize_if_required` (206), `resize_image` (256, cv2/PIL interpolations), `GradualLatent` (383, inference hack).

---

## 4. `networks/`

Inventory (`$ROOT/networks/`): `lora.py` (SD/SDXL), `lora_anima.py`, `lora_flux.py`, `lora_sd3.py`, `lora_lumina.py`, `lora_hunyuan_image.py`, `lora_fa.py` (LoRA-FA, frozen A), `lora_diffusers.py` (PEFT-style for diffusers), `dylora.py`, `oft.py` (SD/SDXL, `oft_unet` prefix), `oft_flux.py`, `oft_v2.py` + `boft.py` + `orthogonal_common.py` (OneTrainer/PEFT-compatible OFTv2/BOFT for SD1/2/SDXL only), `loha.py`, `lokr.py`, `network_base.py` (shared `AdditionalNetwork` + `ArchConfig` for LoHa/LoKr), `control_net_lllite.py`, `control_net_lllite_for_train.py`, `control_net_lllite_anima.py`; tools: `merge_lora.py`, `sdxl_merge_lora.py`, `flux_merge_lora.py`, `svd_merge_lora.py`, `resize_lora.py`, `extract_lora_from_models.py`, `extract_lora_from_dylora.py`, `flux_extract_lora.py`, `convert_flux_lora.py`, `convert_anima_lora_to_comfy.py`, `convert_hunyuan_image_lora_to_comfy.py`, `check_lora_weights.py`, `lora_interrogator.py`.

### 4.1 Plugin contract every network module must satisfy (duck-typed by `train_network.py`)

```python
def create_network(multiplier: float, network_dim: Optional[int], network_alpha: Optional[float],
                   vae, text_encoder(s), unet, neuron_dropout: Optional[float] = None, **kwargs) -> nn.Module
    # kwargs are raw strings from --network_args k=v; module must parse them itself  (lora_anima.py:225-336)
def create_network_from_weights(multiplier, file, vae, text_encoder(s), unet,
                                weights_sd=None, for_inference=False, **kwargs) -> (network, weights_sd)   # lora_anima.py:339-376
# Network instance methods used by train_network.py:
apply_to(text_encoder, unet, apply_text_encoder: bool, apply_unet: bool)          # 1116
prepare_optimizer_params(text_encoder_lr, unet_lr, default_lr) -> params | (params, lr_descriptions)  # 1153
  or prepare_optimizer_params_with_multiple_te_lrs(text_encoder_lr: list, unet_lr, default_lr)          # 1140-1151
prepare_grad_etc(text_encoder, unet); on_epoch_start(text_encoder, unet); get_trainable_params()      # 1325, 1579, 1621
save_weights(file, dtype, metadata); load_weights(file)                                                # 888, 1120
merge_to(text_encoder, unet, weights_sd, dtype, device)   # for --base_weights (1066)
set_multiplier(m)  # for differential output preservation (313)
# optional, probed with hasattr: prepare_network(args), apply_max_norm_regularization(max_norm, device) -> (keys_scaled, mean, max),
#   enable_gradient_checkpointing(), on_step_start(te, unet), update_grad_norms/update_norms/weight_norms/grad_norms/combined_weight_norms
```

### 4.2 `networks/lora.py` (SD/SDXL reference implementation)

- `LoRAModule` (25–121): `lora_down`/`lora_up` Linear or Conv2d (3×3 down + 1×1 up), kaiming-uniform down, zero up, `scale = alpha/dim`, `alpha` registered as buffer; `forward` = `org_forward(x) + lora_up(lora_down(x)) * multiplier * scale` with `module_dropout` (skip whole module), `dropout` (on hidden), `rank_dropout` (mask over rank, rescale `1/(1-p)`) (90–121). `apply_to` swaps `org_module.forward` (85–88).
- `LoRAInfModule` (124–387): merge/get_weight/regional (A1111 regional prompter) support.
- `create_network` (416–508): `conv_dim/conv_alpha` (conv LoRA on ResnetBlock2D/Downsample2D/Upsample2D), `block_dims/block_alphas/conv_block_dims/conv_block_alphas` (25 SD / 23 SDXL blocks; `get_block_dims_and_alphas` 515), `down_lr_weight/mid_lr_weight/up_lr_weight` with presets (`get_block_lr_weight` 589: `sine/cosine/linear/reverse_linear/zeros`), `block_lr_zero_threshold`, `rank_dropout`, `module_dropout`, `loraplus_*_lr_ratio`.
- `LoRANetwork` (861–1410): `UNET_TARGET_REPLACE_MODULE=["Transformer2DModel"]`, `TEXT_ENCODER_TARGET_REPLACE_MODULE=["CLIPAttention","CLIPSdpaAttention","CLIPMLP"]`, prefixes `lora_unet` / `lora_te` (`lora_te1`/`lora_te2` for SDXL); `prepare_optimizer_params` (1148) builds groups per TE and per unet with block LR weights & LoRA+; `apply_max_norm_regularization` (1369) rescales up/down when `||ΔW|| > max_norm`.
- `create_network_from_weights` (805) infers `modules_dim/alpha` from `lora_down.weight` shape and `alpha` keys; converts diffusers (PEFT) key names (`convert_diffusers_to_sai_if_needed` 758).

### 4.3 `networks/lora_anima.py`

Described in §2.5; extras: regex `exclude_patterns`/`include_patterns` (fullmatch on original module name, 245–261, 475–481), `network_reg_dims` / `network_reg_lrs` (277–308, applied at 491–497 / 650–713), `train_llm_adapter` (adds `LLMAdapterTransformerBlock` targets, 540–542), `verbose`, `rank_dropout`, `module_dropout`, LoRA+. Only Linear (and 1×1 conv) get default dims; 3×3 conv skipped unless reg_dims match (500–507). No block-index-based LR (regex replaces it). `save_weights` computes `sshs_model_hash/sshs_legacy_hash` (747–771).

### 4.4 `network_base.py` / LoHa / LoKr (LyCORIS-family reimplementation)

`ArchConfig` (19–27) + `detect_arch_config(unet, text_encoders)` (30–63): SDXL by class, Anima by presence of a `Block` module class → `unet_target_modules=["Block","PatchEmbed","TimestepEmbedding","FinalLayer"]`, `te_target_modules` Qwen3 classes, prefix `lora_unet`/`lora_te`, default excludes, `adapter_target_modules=["LLMAdapterTransformerBlock"]`. `AdditionalNetwork` (86–545) generalizes `lora_anima.LoRANetwork` with `module_class`/`module_kwargs` injection, `conv_lora_dim`, `_is_plus_param` for LoRA+ (`lora_up`, `hada_w2_a`, `lokr_w1`). `lokr.py`: `factorization` (24), `LoKrModule` (79–) with `factor`, `use_tucker`, full-matrix fallback when `dim >= max(out_k,in_n)/2`, `get_diff_weight` (211) via `torch.kron`; `create_network` parses `factor`, `use_tucker`, `conv_dim/conv_alpha`, patterns, reg dims/lrs, `train_llm_adapter`. `loha.py` analogous with custom autograd `HadaWeight`. Docs: `docs/loha_lokr.md`.

### 4.5 Other adapters

- `oft.py` (SD/SDXL only; `oft_unet` prefix, `enable_all_linear`, `enable_conv`), `oft_v2.py`/`boft.py` (PEFT/OneTrainer-compatible; SD1/2/SDXL only, `networks/orthogonal_common.py` shares prefixes/parsers), `oft_flux.py`.
- `dylora.py`: `DyLoRAModule` with `unit` rank granularity; `extract_lora_from_dylora.py`.
- `lora_fa.py`: LoRA-FA (frozen `lora_down`), SD/SDXL only.
- `control_net_lllite*.py`: LLLite (lightweight ControlNet) for SDXL; `control_net_lllite_anima.py` adds `LLLiteModuleDiT`/`ControlNetLLLiteDiT`/`AnimaControlNetLLLiteWrapper` (204–535) trained by `anima_train_control_net_lllite.py`.
- **LyCORIS integration:** no first-class hook; `--network_module lycoris.kohya` works purely because LyCORIS exposes `create_network(...)`/`create_network_from_weights(...)` with the same signature. `train_network.py:1081-1083` injects `dropout` into kwargs as a "workaround for LyCORIS". Inference scripts import `lycoris.kohya.create_network_from_weights` when `--lycoris` (`anima_minimal_inference.py:30-32`). No `lycoris` in `requirements.txt`.
- Tools: `svd_merge_lora.py` (merge N LoRAs by SVD to a new rank with LBW per-block weights; SD/SDXL block indexing only), `resize_lora.py` (SVD rank reduction with `sv_ratio/sv_fro/sv_cumulative` dynamic methods; generic over `lora_down/up` keys so works for Anima keys), `extract_lora_from_models.py` (SD/SDXL diff → LoRA via SVD), `flux_extract_lora.py`, `merge_lora.py`/`sdxl_merge_lora.py`/`flux_merge_lora.py` (model-specific). **No Anima merge/extract tool** — merging Anima LoRA into the DiT is only done at load time in `lora_utils.load_safetensors_with_lora_and_fp8`.

---

## 5. Optimizers & schedulers — `$ROOT/library/optimizer.py`

- `get_optimizer(args, trainable_params) -> (name, args_str, optimizer)` (34–434). `--optimizer_args k=v` parsed with `ast.literal_eval` (67–83). Legacy `--use_8bit_adam`/`--use_lion_optimizer` (38–51). Supported names (case-insensitive): `AdamW` (default), `AdamW8bit`, `SGDNesterov`, `SGDNesterov8bit`, `Lion`, `Lion8bit`, `PagedAdamW`, `PagedAdamW8bit`, `PagedAdamW32bit`, `PagedLion8bit`, `DAdaptation`/`DAdaptAdamPreprint`, `DAdaptAdaGrad`, `DAdaptAdam`, `DAdaptAdan`, `DAdaptAdanIP`, `DAdaptLion`, `DAdaptSGD`, `Prodigy`, `Adafactor` (with `relative_step` handling that mutates `args.learning_rate/unet_lr/text_encoder_lr/lr_scheduler` at 267–290 — noted as inverted dependency), `RAdamScheduleFree`, `AdamWScheduleFree`, `SGDScheduleFree`, and a fallback for **any** `module.Class` path via `importlib` (330–343; e.g. `pytorch_optimizer.CAME`, `prodigyplus.ProdigyPlusScheduleFree`). `AdEMAMix8bit/PagedAdEMAMix8bit` are listed in the docstring but have no branch (they hit the `endswith("8bit")` block and fall through with `optimizer_class=None` → generic path fails since no `.`). `ScheduleFreeWrapper` is commented out (345–424).
- Schedule-free: `is_schedulefree_optimizer` (449, name suffix check), `get_optimizer_train_eval_fn` (437) returns `optimizer.train/eval`, called around sampling/saving/validation; `get_dummy_scheduler` (453).
- `get_scheduler_fix(args, optimizer, num_processes)` (474–608): `num_training_steps = max_train_steps * num_processes`; warmup/decay accept int or float ratio; `--lr_scheduler_type module.Class` generic; `adafactor:<lr>` → `AdafactorSchedule`; `piecewise_constant` (diffusers); transformers `SchedulerType`: `constant`, `constant_with_warmup`, `inverse_sqrt`(timescale), `cosine_with_restarts`(num_cycles), `polynomial`(power), `cosine_with_min_lr`(min_lr_ratio), `linear`, `cosine`, `warmup_stable_decay`(num_decay_steps, min_lr_ratio). Args at `args.py:49-178`.
- Per-parameter-group LR: network returns list of `{"params":..., "lr":...}` groups (TE vs unet vs LoRA+ "plus" vs regex groups) with `lr_descriptions` used for logging (`train_network.py:98-112`). D-Adaptation/Prodigy warn that only the first LR is honored (204–207).

---

## 6. Timestep / noise handling

- **DDPM path** (`library/loss.py`): `get_timesteps(min,max,b,device)` (27, `randint`), `get_noise_noisy_latents_and_timesteps` (36–72): noise offset (`--noise_offset`, `--noise_offset_random_strength`, `--adaptive_noise_scale`), `--multires_noise_iterations/--multires_noise_discount` (pyramid noise), `--min_timestep/--max_timestep`, `--ip_noise_gamma(_random_strength)`; `noise_scheduler.add_noise`. `--perlin_noise` arg exists but implementation is commented out (`custom_train_functions.py:506-561`). `--zero_terminal_snr` in `get_noise_scheduler`.
- **Flow-matching path** (`flux_train_utils.get_noisy_model_input_and_timesteps` 473–539): `--timestep_sampling {sigma,uniform,sigmoid,shift,flux_shift}`, `--sigmoid_scale`, `--discrete_flow_shift`, `--weighting_scheme {sigma_sqrt,logit_normal,mode,cosmap,none,uniform}` (+ `--logit_mean --logit_std --mode_scale`; density sampling only when `timestep_sampling=sigma`), `--ip_noise_gamma`. Only `ip_noise_gamma` from the noise-shaping family is honored; `noise_offset`, `multires_noise`, `min/max_timestep` are silently ignored. Diagnostics: `--show_timesteps console|image`, `--show_timesteps_resolution`, `--show_timesteps_offset` (`args.py:689-719`, `anima_train_utils.show_timesteps` 245–281, `timestep_visualization.py`), startup log `log_timestep_sampling_info` (`flux_train_utils.py:592`).
- **Huber** (`loss.py:75-95`): `--loss_type {l1,l2,huber,smooth_l1}`, `--huber_schedule {constant,exponential,snr}` default **`snr`** (`args.py:526-532`), `--huber_c`, `--huber_scale`. Pseudo-Huber `2c(sqrt(d²+c²)-c)`.

---

## 7. Saving / metadata / resume

- LoRA file: `network.save_weights(ckpt_file, save_dtype, metadata)` (`train_network.py:888`) → `safetensors.save_file(state_dict, file, metadata)` with `sshs_model_hash`/`sshs_legacy_hash` (`lora_anima.py:759-769`, hashes via `model_io.precalculate_safetensors_hashes` 82–96). Names: `<output_name>.safetensors`, `<output_name>-<epoch:06d>.safetensors`, `<output_name>-step<step:08d>.safetensors` (`checkpoint_io.py:43-72`). `--save_model_as {ckpt,pt,safetensors}`; `--save_precision {float,fp16,bf16}`.
- `ss_*` metadata built in `_build_metadata` (`train_network.py:525-587`): session id, timestamps, LRs, counts, `ss_gradient_*`, `ss_network_module/dim/alpha/dropout/args`, `ss_mixed_precision`, `ss_base_model_version`, `ss_seed`, noise/loss options, `ss_sd_scripts_commit_hash`, `ss_optimizer`, `ss_datasets` (JSON of dataset/subset config incl. `tag_frequency`, `bucket_info`), `ss_sd_model_hash/ss_new_sd_model_hash`, `ss_vae_*`, validation fields; Anima adds `ss_weighting_scheme/logit_mean/logit_std/mode_scale/timestep_sampling/sigmoid_scale/discrete_flow_shift` (`anima_train_network.py:407-414`); `_save_model` adds `ss_training_finished_at/ss_steps/ss_epoch` (880–882). `--no_metadata` keeps only `SS_METADATA_MINIMUM_KEYS` (`model_io.py:172-179`). All values stringified (747).
- SAI ModelSpec: `sai_model_spec.ModelSpecMetadata` dataclass (103–190) → `modelspec.sai_model_spec="1.0.1"`, `architecture` (`anima-preview` + `/lora` adapter suffix via `determine_architecture` 193), `implementation`, `title`, `resolution`, `date`, `hash_sha256`, optional `--metadata_{title,author,description,license,tags,usage_hint,thumbnail,merged_from,trigger_phrase,preprocessor,is_negative_embedding}` (586). Anima uses `get_sai_model_spec_dataclass(..., anima="preview")` (`model_io.py:283-348`).
- Save cadence: `--save_every_n_epochs`, `--save_every_n_steps`, `--save_n_epoch_ratio`, rotation `--save_last_n_epochs/--save_last_n_steps` (`checkpoint_io.get_remove_epoch_no/get_remove_step_no` 75–95), separate `--save_last_n_epochs_state/--save_last_n_steps_state`.
- State: `--save_state` (`accelerator.save_state(dir)` → `<name>-<epoch>-state/` or `<name>-step<N>-state/`, `checkpoint_io.py:227-273`), `--save_state_on_train_end` (`<name>-state`), HF upload options. Pre-hooks strip base models so only network + optimizer + scheduler + RNG are saved, plus `train_state.json` (`train_network.py:1337-1379`). `--resume <dir>` → `accelerator.load_state` (`args.py:1200-1207`; HF variant 1209–1250); resumed step comes from `train_state.json` unless `--initial_step/--initial_epoch` given; `--skip_until_initial_step` replays the dataloader via `accelerator.skip_first_batches` (1455–1469, 1583–1585).
- Full DiT save (`anima_train.py`): `anima_utils.save_anima_model` with `net.` prefix and `format="pt"` (297–320); `checkpoint_io.save_sd_model_on_*_common` for naming/rotation.

---

## 8. Sample generation during training (Anima)

`AnimaNetworkTrainer.sample_images` (`anima_train_network.py:236-254`) → `anima_train_utils.sample_images` (508–616): gating by `--sample_at_first/--sample_every_n_steps/--sample_every_n_epochs`; unwraps DiT, `switch_block_swap_for_inference`, saves/restores RNG, distributes prompts across processes, calls `_sample_image_inference` (619–768): per-prompt keys `prompt`, `negative_prompt`, `sample_steps` (default 30), `width/height` (512, rounded down to /16), `scale` (CFG, default 7.5), `seed`, `flow_shift` (default 3.0); text encoded via strategies or from `sample_prompts_te_outputs` cache (built during TE caching, `anima_train_network.py:201-219`); LLM adapter applied manually (689–699); `do_sample` (428–505): Euler on `sigmas = linspace(1,0,steps+1)` with shift warp `σ*s/(1+(s-1)σ)`, two-pass CFG, `x += pred*dt`; VAE `decode_to_pixels`; PNG saved as `<output_name>_<e{epoch:06d}|{step:06d}>_<i:02d>_<ts>_<seed>.png` in `output_dir/sample`; W&B image log.

Prompt file (`library/sampling.load_prompts` 218–244, `line_to_prompt_dict` 128–215): `.txt` one prompt per line (`#` comments) with inline flags `--w --h --d(seed) --s(steps) --l(scale) --g --n(negative) --ss --cn --mk --i --ctr --rcfg --fs(flow_shift) --am`; or `.toml` (`[prompt]` defaults + `[[prompt.subset]]`) / `.json` list of dicts. `--sample_sampler` is ignored by Anima (Euler only).

---

## 9. `.ai/` directory and docs

`.ai/claude.prompt.md`, `.ai/Codex.prompt.md`, `.ai/gemini.prompt.md` are 3-line agent stubs that `@include` `.ai/context/01-overview.md` (129 lines): module ownership table post-refactor, import conventions (`_util` aliases; never import via `train_util`), model-family file layout, strategy-pattern checklist for new models, testing approach, block-swap notes. No coding standards beyond that.

Anima-relevant docs (`$ROOT/docs/`): `anima_train_network.md` (LoRA guide: args, target modules, regex dims/LRs, LLM adapter LoRA, TE LoRA, VRAM options, timestep/weighting, caption dropout w/ cache, ComfyUI conversion, metadata), `anima_torch_compile.md`, `anima_train_control_net_lllite.md`, `timestep_sampling_offset.md`, `loha_lokr.md` (LoHa/LoKr for SDXL & Anima), `train_network.md` / `train_network_advanced.md` (shared LoRA options), `config_README-en.md` (dataset TOML), `dataset_metadata.md`, `validation.md`, `masked_loss_README.md`, `flux_train_network.md` / `sd3_train_network.md` (referenced for timestep options). `README.md:54-125` release notes list Anima features (timestep offset PR #2401, torch.compile PR #2379, 2D VAE, LLLite PR #2317).

---

## 10. Weaknesses / tech debt

**Structural**
1. `NetworkTrainer.train()` is a ~965-line method (`train_network.py:898-1863`) mixing dataset setup, model loading, accelerator wiring, resume logic, the loop, validation, saving and logging; `_build_metadata` is 250 lines. `anima_train.py` duplicates ~500 lines of it procedurally with subtly different semantics (weighting after vs before spatial mean; different LLM-adapter call path; `noise_scheduler=None`).
2. Global singletons: strategy classes hold `_strategy` on the class and refuse re-set (`strategy_base.py:43-46` etc.), `accelerator_setup.HIGH_VRAM` module flag, `multiprocessing.Value` epoch/step propagation. One training run per process; not importable as a library.
3. `argparse.Namespace` is the universal config object, threaded into datasets, strategies, networks, loss and mutated at runtime (`args.min_timestep = args.max_timestep = timestep  # dirty hack`, 826; `optimizer.py:271-288` rewriting `args.learning_rate/lr_scheduler`; `assert_extra_args` flipping fp8 flags). ~250 flags total: 155 in `args.py`, 28 in `train_network.py`, 26 Anima, 12 metadata, 9 DeepSpeed, 5 loss, 5 config, 3 logging. Many are model-specific yet globally registered (`--v2`, `--clip_skip`, `--noise_offset`, `--min_snr_gamma` all accepted for Anima and silently ignored).
4. Duck-typed network protocol via `hasattr` (`train_network.py:1097-1109, 1140, 1486, 1624-1627, 1641`); the "contract" exists only implicitly (TODO at 1099). `create_network` receives raw strings and every module re-implements `kwargs.get(...)` parsing (`lora_anima.py:241-308`, `lokr.py`, `oft_v2.py:574-585`).
5. Dataset does its own batching (`DataLoader(batch_size=1)`, `collator` returns `examples[0]`); bucket → batch mapping is rebuilt each `make_buckets`; `__getitem__` (`dataset.py:989-1242`) is a 250-line branchy function handling latents/images/npz/inpainting/alpha/TE caches simultaneously. `network_multipliers` is emitted per batch (1238) but never consumed by the trainer (only `set_multiplier(1.0)` comment at 326).
6. Cache formats: uncompressed `np.savez` float32 (bf16 latents up-cast, doubling size; TODO `strategy_base.py:639`), one npz per image per resolution; TE cache stores full 512×1024 fp32 embeddings (2 MB/caption) even for short prompts; T5 ids always padded to 512; cache validity check is key-presence only, no hash of caption/settings (docs say "delete cache if you change caption_dropout_rate").
7. Bilingual JP/EN strings in every assert/log inflate files and hinder reuse; comments partly Japanese.

**Anima-specific correctness / gaps**
8. Validation timestep sweep does nothing for Anima: `_run_validation_loop` sets `args.min_timestep/max_timestep` (826) but the flow path never reads them → the 4 "fixed" validation timesteps are actually 4 random draws.
9. Huber loss is effectively unusable for Anima: default `--huber_schedule snr` needs `alphas_cumprod` → `NotImplementedError` (`loss.py:84-85`); `exponential` receives timesteps already divided by 1000 (`anima_train_network.py:301` then `train_network.py:475`) so `exp(-alpha*t)` is ~constant; `anima_train.py:569` passes `noise_scheduler=None` (crashes for `exponential`). Only `constant` works.
10. `--llm_adapter_path` is registered (`anima_train_utils.py:42-47`) but never read anywhere; docs claim it loads a separate adapter.
11. Docs claim DiT config is "auto-detected from the state dict"; code hardcodes a single 28-block/2048 config (`anima_utils.py:67-98`); other Anima sizes would fail strict loading.
12. `pos_emb_learnable=True` is passed but unused by `VideoRopePosition3DEmb`; dead kwargs.
13. Cross-attention context is always 512 tokens with no key mask (zeros attend as sinks); Qwen3 encoding uses only `last_hidden_state`; no option for intermediate layers, no chat template.
14. Two LLM-adapter invocation paths (`forward` vs `forward_mini_train_dit` kwargs) — easy to double-apply or skip.
15. `fp8` training unsupported for Anima; `fp8_scaled` inference-only; `torch._scaled_mm` path marked "not tested".
16. Block swap incompatible with CPU-offload checkpointing, DDP device placement handled ad hoc (`device_placement=[False]`); `custom_offloading_utils.swap_weight_devices_no_cuda` "not tested".
17. No Anima merge/extract/resize tooling beyond `resize_lora.py` (generic) and load-time merge; `svd_merge_lora.py` block indexing is SD/SDXL-only.
18. `weighted_captions` unsupported (`tokenize_with_weights` not implemented) yet flag accepted.
19. `train_network.py` LoRA + DDP relies on manual `all_reduce_network` because forward hooks bypass DDP reducer; `find_unused_parameters` semantics untested with regex-selected modules.
20. TODO/FIXME density: 101 occurrences across core/library/networks (e.g. `train_network.py:71,430,524,1099,1119,1391,1492,1536`; `strategy_base.py:224,379,477,639`).

**Tests**
21. `Test/tests/` (~3.6k lines) covers offloading utils, optimizer factory, orthogonal networks, mask generator, Lumina strategy/models/utils, FLUX train utils, SAI spec, 2D VAE, inpainting. **Zero tests for Anima** (`grep -rl anima Test/tests/` empty), none for `train_network.py` loop, dataset/bucketing, caching formats, `lora_anima`, LoKr on Anima. CI (`.github/workflows/tests.yml`) is CPU-only pytest on torch 2.4/2.6, python 3.10.

---

## 11. `requirements.txt` (`$ROOT/requirements.txt`)

Pinned: `accelerate==1.6.0`, `transformers==4.54.1` (needed for `Qwen3Config/Qwen3ForCausalLM`), `diffusers[torch]==0.32.1` (schedulers, `SchedulerMixin`, SD pipelines), `ftfy==6.3.1`, `opencv-python==4.10.0.84`, `einops==0.7.0`, `bitsandbytes` (unpinned; 8-bit/paged optimizers), `lion-pytorch==0.2.3`, `schedulefree==1.4`, `pytorch-optimizer==3.10.0`, `prodigy-plus-schedule-free==1.9.2`, `prodigyopt==1.1.2`, `tensorboard`, `safetensors==0.4.5`, `toml==0.10.2`, `voluptuous==0.15.2` (dataset config schema), `huggingface-hub==0.34.3`, `imagesize==1.4.1`, `numpy` (unpinned), `rich==14.1.0`, `sentencepiece==0.2.1` (T5 tokenizer), `-e .` (installs `library` package via `setup.py`). Torch itself is not pinned (CI installs 2.4.0/2.6.0; docs say PyTorch ≥2.5 to avoid NaN in Anima). Optional/undeclared: `dadaptation` (CI installs 3.2), `xformers`, `flash_attn`, `sageattention`, `lycoris`, `wandb`, `matplotlib`, `deepspeed`, `pillow_avif`/`pillow_jxl`, `triton` (for `--compile`).
