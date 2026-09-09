I now have everything I need. Here is the full architectural report.

---

# diffusion-pipe architectural report (focus: Anima / LoRA / LoKr)

Repo root: `/Volumes/Service/Dev/YPuddinTrainStudio/diffusion-pipe` (HEAD `8f83dbf`, submodule dirs are present but **empty / not checked out**).

**Headline finding:** there is no `models/anima.py`. `type = 'anima'` is routed in `train.py:342-344` to `models/cosmos_predict2.py::CosmosPredict2Pipeline`, which flips into "Anima mode" when the config has `llm_path` instead of `t5_path` (`cosmos_predict2.py:248-268`, sets `self.name = 'anima'` at line 268). Anima = Cosmos-Predict2 DiT (vendored in `models/cosmos_predict2_modeling.py`, NVIDIA Apache-2.0) + a 6-layer `LLMAdapter` (`models/llm_adapter.py`) that maps Qwen3-0.6B hidden states onto old-T5 token positions + a Wan2.1-architecture VAE (`models/wan/vae2_1.py::WanVAE_`).

---

## 1. `train.py` — full flow (975 lines, single script)

### 1.1 CLI (`train.py:41-58`)
`--config`, `--local_rank`, `--resume_from_checkpoint [name]`, `--reset_dataloader`, `--reset_optimizer`, `--reset_optimizer_params`, `--regenerate_cache`, `--cache_only`, `--trust_cache`, `--i_know_what_i_am_doing`, `--master_port`, `--dump_dataset DIR` (Flux only), `--test_sample` (debug: one 512x512 image then quit), plus `deepspeed.add_config_arguments`.

### 1.2 Config loading and TOML schema (`train.py:282-290`, defaults in `set_config_defaults` `train.py:93-143`)
Config is `json.loads(json.dumps(toml.load(f)))` (inline tables aren't picklable for the multiprocess dataset code). No schema validation; every key is read via `config.get(...)` scattered over `train.py`, model files and `utils/dataset.py`.

Top-level keys (all that the code reads):

| key | meaning / where read |
|---|---|
| `output_dir` | run dirs `YYYYMMDD_HH-MM-SS` created here (`train.py:543-547`) |
| `dataset` | path to dataset TOML (`train.py:393`) |
| `eval_datasets` | list of `{name, config}` (or bare str → `eval{i}`) (`train.py:437-447`) |
| `epochs` | required; loop ends when `train_dataloader.epoch > epochs` (`saver.py:140`) |
| `max_steps` | optional hard stop (`train.py:961`) |
| `micro_batch_size_per_gpu` | int or `[[res, bs], ...]` per-resolution (`train.py:396-400`) |
| `image_micro_batch_size_per_gpu`, `eval_micro_batch_size_per_gpu`, `eval_image_micro_batch_size_per_gpu` | same forms (`train.py:402-418`) |
| `pipeline_stages` | default 1 (`train.py:97`, used 605) |
| `gradient_accumulation_steps` | → DeepSpeed `gradient_accumulation_steps` = number of micro-batches per step (`train.py:425`) |
| `gradient_clipping` | default 1.0; forced 0 with gradient release (`train.py:427`) |
| `steps_per_print` | DeepSpeed console log freq (`train.py:428`) |
| `warmup_steps` | default 0; LinearLR warmup via `SequentialLR` (`train.py:858-861`) |
| `lr_scheduler` | `constant` (default) / `linear` (→0 over `epochs*steps_per_epoch`) / `cosine` (eta_min 1e-6) (`train.py:849-857`) |
| `force_constant_lr` | overrides LR + scheduler after resume (`train.py:893-896`) |
| `blocks_to_swap` | block swapping count; asserts `pipeline_stages==1` and `adapter` present (`train.py:576-583`) |
| `disable_block_swap_for_eval` | (`train.py:907`) |
| `activation_checkpointing` | `false` / `true` / `'unsloth'` (`train.py:98-101, 587-603`) |
| `reentrant_activation_checkpointing` | default False; forced True for `'unsloth'` (`train.py:99-101`) |
| `partition_method` | default `'parameters'`; or `'manual'` (`train.py:606`) |
| `partition_split` | list of length `pipeline_stages-1` (`train.py:607`, `utils/pipeline.py:16-51`) |
| `save_every_n_epochs` / `save_every_n_steps` / `save_every_n_examples` | one is required (`train.py:95`); examples→steps conversion at `train.py:643-645` |
| `checkpoint_every_n_epochs` / `checkpoint_every_n_minutes` | DeepSpeed state checkpoint (`utils/saver.py:20-44`) |
| `eval_every_n_epochs` / `eval_every_n_steps` / `eval_every_n_examples` | (`train.py:138-140, 646-648, 945`) |
| `eval_before_first_step` | default True (`train.py:141, 908`) |
| `eval_gradient_accumulation_steps` | default 1 (`train.py:137`) |
| `save_dtype` | dtype for saved weights (`train.py:103-104`, `saver.py:75-76`) |
| `caching_batch_size` | VAE/TE batch size during caching (`train.py:430`) |
| `map_num_proc` | HF `map` workers, default `min(8, cpu_count)` (`train.py:289-290`, `dataset.py:34`) |
| `compile` | `pipeline_model.compile(dynamic=True)` (`train.py:620-621`) |
| `video_clip_mode` | `single_beginning` / `single_middle` (`models/base.py:59, 113-131`) |
| `x_axis_examples` | TB/WandB x axis = examples instead of steps (`train.py:143, 926`) |
| `logging_steps` | default 1 (`train.py:135`) |
| `uncond_fraction` | probability of swapping caption for cached empty-caption embedding (`train.py:288`, `dataset.py:315`) |
| `huber_delta` / `smooth_l1_beta` | alternative losses (`models/base.py:424-427`). NOTE `examples/main_example.toml:44` documents `pseudo_huber_c`, which is **not read anywhere** |
| `resume_from_checkpoint`, `regenerate_cache` | config-level fallbacks for the CLI flags (`train.py:301-308`) |
| `[model]` | `type`, `dtype` (mapped via `DTYPE_MAP`), `transformer_dtype`, `diffusion_model_dtype`, `guidance` (default 1.0), model-specific keys (`train.py:106-113`) |
| `[adapter]` | `type` (`lora`/`lokr`), `rank`, `dtype` (defaults to model dtype), `dropout` (lora), `decompose_factor` (lokr, default -1), `rank_dropout` (lokr), `init_from_existing`, `exclude_modules`. **`alpha` is forbidden** — forced `alpha = rank` (`train.py:118-122`) |
| `[optimizer]` | `type` + kwargs passed straight to the optimizer class; special: `gradient_release`, `beta2_half_life` (`train.py:658-663`) |
| `[monitoring]` | `enable_wandb`, `wandb_api_key`, `wandb_tracker_name`, `wandb_run_name` (`train.py:561-573`) |

Stale key: `train.py:843` reads `config['lora']['dtype']` — the table is `[adapter]`, so this branch never fires.

### 1.3 Startup sequence (`train.py:276-627`)
1. `torch.multiprocessing.set_sharing_strategy('file_system')`, `apply_patches()` (`utils/patches.py:408-439`, monkeypatches PEFT, DeepSpeed schedule/engine, HunyuanVideo, Comfy Flux blocks).
2. `deepspeed.init_distributed()` + `torch.cuda.set_device(local_rank)` (`train.py:296-299`).
3. Model pipeline object constructed from `model.type` (`train.py:310-382`) — the constructor loads VAE + text encoder(s) **but not the DiT**.
4. `DatasetManager` + `Dataset` objects (`train.py:430-447`); `dataset_manager.cache()` runs the full caching flow (`train.py:516`); `--cache_only` quits here.
5. `model.free_vae_and_te()` (no-op for BasePipeline models like Anima; real for ComfyPipeline) then `model.load_diffusion_model()` (`train.py:522-529`).
6. Adapter: `model.configure_adapter(adapter_config)` then optional `load_adapter_weights(init_from_existing)` (`train.py:531-537`). **LoRA vs full fine-tune is selected purely by presence of `[adapter]`**; without it every DiT parameter is trainable (`is_adapter=False`).
7. `run_dir` decided on rank 0 and broadcast; config + dataset TOMLs copied there (`train.py:539-558`). Bug: `shutil.copy(eval_dataset['config'])` at line 557 assumes dict form.
8. WandB init on rank 0 (`train.py:560-573`).
9. Block swap: monkeypatch `deepspeed.pipe.PipelineModule.to` to no-op, then `model.enable_block_swap(n)` (`train.py:575-583`).
10. `layers = model.to_layers()`; activation checkpointing kwargs (`train.py:585-603`):
    - `true` → `functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant=reentrant_flag)`;
    - `'unsloth'` → `utils/unsloth_utils.py::unsloth_checkpoint`;
    - passed as `activation_checkpoint_interval=1, checkpointable_layers=model.checkpointable_layers, activation_checkpoint_func=...`.
11. `ManualPipelineModule(layers, num_stages, partition_method, manual_partition_split, loss_fn=model.get_loss_fn(), dynamic_shape=True)` (`train.py:608-616`); `parameters_to_train = [p for p in pipeline_model.parameters() if p.requires_grad]`.
12. `deepspeed.initialize(args, model=pipeline_model, config=ds_config)` (`train.py:623-627`); `model_engine._support_torch_style_backward = True` hack (631); `global_batch_size = micro_bs * GAS * dp_world_size` (632).
13. Optimizer built via `model_engine._configure_optimizer(get_optimizer, parameters_to_train)` (`train.py:817`) — see §5 for `get_optimizer`. No DeepSpeed fp16/bf16 mode is enabled, so DeepSpeed takes its "fp32" path with bf16 parameters and **no master weights**; correctness relies on Kahan-summation optimizers.
14. `train_data.post_init(dp_rank, dp_world, micro_bs_dict, GAS, image_micro_bs_dict)` (`train.py:825-831`), `PipelineDataLoader` (846), `steps_per_epoch = len(train_dataloader) // GAS` (847), LR scheduler (849-862).

### 1.4 Resume (`train.py:868-891`)
`model_engine.load_checkpoint(run_dir, load_module_strict=False, load_lr_scheduler_states=..., load_optimizer_states=not reset_optimizer)`; `client_state['custom_loader']` restores `PipelineDataLoader` (`dataset.py:1425-1435`: recreates the DataLoader with `SkipFirstNSampler(num_batches_pulled-1)`, sets `recreate_dataloader=True` so the skip happens only once); `step = client_state['step']+1`, `examples` restored. `--reset_dataloader` only restores epoch. `epoch_loss`/`num_steps` are not checkpointed (TODO at `train.py:911`).

### 1.5 Training loop (`train.py:915-965`)
```python
while True:
    model_engine.reset_activation_shape()                 # bucket shapes vary every step
    iterator = get_data_iterator_for_step(train_dataloader, model_engine)  # preload GAS micro-batches (only first/last stage)
    loss = model_engine.train_batch(iterator).item()      # DeepSpeed pipeline schedule: fwd/bwd per micro-batch, then optimizer step
    train_dataloader.sync_epoch()                         # all_gather max epoch across ranks
    new_epoch, checkpointed, saved = saver.process_epoch(epoch, step, examples)
    ... tensorboard/wandb logging every logging_steps ...
    if eval due: evaluate(...)
    if finished_epoch: log train/epoch_loss; break if new_epoch is None
    checkpointed, saved = saver.process_step(step, examples)   # save_every_n_steps, checkpoint_every_n_minutes, signal files
    if max_steps reached: break
    step += 1; examples += global_batch_size
```
After the loop: `saver.save_checkpoint` / `saver.save_model(final_model_name)` unless just done (`train.py:967-971`).

Logged metrics (`train.py:928-943`): `train/loss`, `train/grad_norm` (only if `optimizer._grad_norm` exists — only `GenericOptim` sets it), `train/prodigy_d`, `train/automagic_lrs` (histogram) + `train/automagic_avg_lr`, `train/epoch_loss`; all mirrored to WandB when enabled.

### 1.6 Eval (`train.py:39, 176-242`)
`TIMESTEP_QUANTILES_FOR_EVAL = [0.1 .. 0.9]`. `evaluate()` calls `model.prepare_block_swap_inference()`, wraps in `torch.no_grad()` + `isolate_rng()` seeded with `rank` (deterministic noise across evals), then for each eval dataset and each quantile runs `evaluate_single`: `eval_dataloader.set_eval_quantile(q)` → every batch's `prepare_inputs(batch, timestep_quantile=q)` uses `dist.icdf(q)` so **t is fixed per pass**; loops `model_engine.eval_batch` until the eval dataloader wraps (epoch==2), averages. Metrics: `{name}/loss_quantile_0.10 ... 0.90`, `{name}/loss` (mean of 9), `eval/eval_time_sec`. Cost = 9 full passes over the eval set.

### 1.7 Save/checkpoint (`utils/saver.py`)
- `save_model` → `save_adapter` (`saver.py:58-85`) or `save_full_model` (87-108): each pipeline stage (dp rank 0) dumps `p.requires_grad` params keyed by `p.original_name` with `.default`/`.modules_to_save` stripped into `tmp/state_dict_{stage}.bin`; stage 0 merges and calls `model.save_adapter(save_dir, sd)` / `model.save_model(save_dir, sd)`, copies the TOML. Output dirs `epochN/` or `stepN/`.
- `save_checkpoint` → `model_engine.save_checkpoint(run_dir, client_state={step, examples, custom_loader}, save_latest=True, exclude_frozen_parameters=True)` → `global_stepNNNN/` (`saver.py:118-128`).
- Manual control: touch `<run_dir>/save` or `<run_dir>/save_quit` (`saver.py:151-163`).

---

## 2. `models/base.py` — pipeline abstraction (833 lines)

### 2.1 Class hierarchy
- `CommonPipeline` (`base.py:216-345`): class attrs `framerate=None, pixels_round_to_multiple=16, keep_in_high_precision=[], spatial_compression=8, channels=16, is_video_vae=False`; sampling scaffolding (`prepare_sample_test` 234-253, `sample` 313-342 — only for `--test_sample`, needs `get_conds`/`vae_decode`), `get_call_vae_fn` (255-260), **`get_target_modules` (262-270)** and **`configure_adapter` (272-311)**.
- `BasePipeline(CommonPipeline)` (`base.py:348-445`): for models with their own loaders (Flux, Wan, Cosmos-Predict2/Anima, ...). Uses `self.transformer`.
- `ComfyPipeline(CommonPipeline)` (`base.py:541-833`): for ComfyUI-native models (Z-Image, Flux2, Krea2, MiniMax, ...). Uses `self.diffusion_model`, loads VAE via `comfy.sd.VAE`, TEs via `comfy.sd.load_clip`, DiT via `comfy.sd.load_diffusion_model` with `merge_adapters`, `dequantize` (593-618) and `patch_quantized_modules` (620-635) for training on quantized Comfy weights.

### 2.2 Methods a model must provide (BasePipeline contract, `base.py:352-445`)
```python
class BasePipeline(CommonPipeline):
    name: str; framerate: int|None; checkpointable_layers: list[str]; adapter_target_modules: list[str]
    def __init__(self, config)                                  # load VAE + text encoders (NOT the DiT)
    def load_diffusion_model(self)                              # load DiT into self.transformer; set p.original_name
    def get_vae(self)                                           # nn.Module (or Comfy wrapper) used by DatasetManager
    def get_text_encoders(self) -> list                         # [] if embeddings are computed inside the model
    def get_preprocess_media_file_fn(self) -> PreprocessMediaFile
    def get_call_vae_fn(self, vae) -> fn(pixels[, control][, audio=]) -> dict   # returns e.g. {'latents': ...}
    def get_call_text_encoder_fn(self, te) -> fn(captions: list[str], is_video: list[bool][, control_file]) -> dict
    def prepare_inputs(self, inputs: dict, timestep_quantile=None) -> (features_tuple, (target, mask))
    def to_layers(self) -> list[nn.Module]                      # pipeline decomposition
    def get_loss_fn(self) -> fn(output, label)                  # default MSE / huber / smooth_l1 with mask (418-436)
    def get_param_groups(self, parameters) -> list[dict]        # default [{'params': parameters}] (414-415)
    def configure_adapter(self, adapter_config)                 # default: CommonPipeline.configure_adapter(self.transformer, ...)
    def save_adapter(self, save_dir, peft_state_dict)           # abstract
    def save_model(self, save_dir, state_dict)                  # abstract
    def load_adapter_weights(self, adapter_path)                # generic impl 367-386 (strips transformer./diffusion_model. prefix, adds .default)
    def load_and_fuse_adapter(self, path)                       # 388-392
    def model_specific_dataset_config_validation(self, dataset_config)
    def enable_block_swap(self, n) / prepare_block_swap_training() / prepare_block_swap_inference(disable_block_swap=False)
    def free_vae_and_te(self)
```

### 2.3 Pipeline-parallel layer decomposition pattern
`to_layers()` returns a flat list of `nn.Module`s each with `forward(inputs: tuple[Tensor]) -> tuple[Tensor]` (DeepSpeed requires tensor-only tuples; `None` becomes `torch.tensor([])` in `dataset.py:1278-1279`). Canonical structure (Anima instance at `cosmos_predict2.py:547-645`):
- **`InitialLayer`**: holds embedders (`x_embedder`, `pos_embedder`, `t_embedder`, `t_embedding_norm`) and a *non-registered* back-reference `self.model = [model]` (a list so the whole DiT isn't re-registered on this stage). Computes patch/time/rope embeddings and calls `tensor.requires_grad_(True)` on every float output (`cosmos_predict2.py:586-588`) so DeepSpeed's pipeline backward works when the stage has no trainable params (LoRA-only).
- **`TransformerLayer(block, idx, offloader)`**: `offloader.wait_for_block(idx)` → `block(...)` → `offloader.submit_move_blocks_forward(idx)`; passes the whole tuple through unchanged except the hidden state. These are what `checkpointable_layers = ['TransformerLayer']` targets.
- **`FinalLayer`**: `__getattr__` delegates to the model (for `unpatchify`), returns the prediction tensor.
- Every forward is decorated `@torch.autocast('cuda', dtype=AUTOCAST_DTYPE)` (`common.AUTOCAST_DTYPE = model.dtype`, `train.py:287`), and outputs go through `make_contiguous` (`base.py:37-38`).
- Loss: `loss_fn(output, (target, mask))`; targets are broadcast first-stage → last-stage via `dist.send/recv` (`dataset.py:1387-1405`) because they depend on noise sampled on the first stage; `patches.py:113-160` reorders DeepSpeed's `TrainSchedule` so `LoadMicroBatch` happens before send/recv to avoid the deadlock.

### 2.4 PEFT integration (`base.py:262-311`)
```python
def get_target_modules(self, target_model):
    target_modules = set()
    for name, module in target_model.named_modules():
        if module.__class__.__name__ not in self.adapter_target_modules: continue   # by class NAME
        for full_submodule_name, submodule in module.named_modules(prefix=name):
            if isinstance(submodule, nn.Linear): target_modules.add(full_submodule_name)
    return list(target_modules)

def configure_adapter(self, target_model, adapter_config):
    target_modules = self.get_target_modules(target_model)
    if adapter_type == 'lora':
        peft_config = peft.LoraConfig(r=rank, lora_alpha=alpha(=rank), lora_dropout=dropout, bias='none',
                                      target_modules=target_modules, exclude_modules=adapter_config.get('exclude_modules'))
    elif adapter_type == 'lokr':
        peft_config = peft.LoKrConfig(r=rank, decompose_factor=..., alpha=alpha, rank_dropout=..., target_modules=..., exclude_modules=...)
    self.lora_model = peft.get_peft_model(target_model, peft_config)
    for name, p in target_model.named_parameters():
        p.original_name = name                     # e.g. blocks.0.self_attn.q_proj.lora_A.default.weight
        if p.requires_grad: p.data = p.data.to(adapter_config['dtype'])
```
- Every `nn.Linear` inside any module whose class name is in `adapter_target_modules` gets LoRA — explicit full-path list, not regex.
- `utils/patches.py:34-69, 411` patches PEFT so LoRA weights are not downcast to fp8 when the base layer is fp8.
- Saving: `saver.py:74` builds `{original_name minus '.default'/'.modules_to_save': tensor}`, then the model's `save_adapter` writes `peft_config.save_pretrained(save_dir)` (→ `adapter_config.json`) + `adapter_model.safetensors`. ComfyUI format = prefix `diffusion_model.` (`base.py:703-707`, `cosmos_predict2.py:331-335`); Diffusers-format models prefix `transformer.` instead. Since alpha==rank, no `alpha` tensors are written (ComfyUI default scale 1.0).
- Loading (`base.py:367-386`): regex-strips `^(transformer|diffusion_model)\.`, converts `.weight` → `.default.weight`, `load_state_dict(strict=False)`.

---

## 3. Anima model — `models/cosmos_predict2.py` (+ `cosmos_predict2_modeling.py`, `llm_adapter.py`)

### 3.1 Architecture source
- DiT: `models/cosmos_predict2_modeling.py::MiniTrainDIT` (1145-1476), vendored from NVIDIA Cosmos-Predict2 (`# SPDX Copyright NVIDIA 2025`), attention backend forced to `'torch'` (`:1215`, F.scaled_dot_product_attention via `torch_attention_op` 275-306). **No attention mask in self- or cross-attention.** Block structure (`Block` 937-1143): pre-LN (no affine) + AdaLN-LoRA modulation (`adaln_modulation_{self_attn,cross_attn,mlp}` = SiLU → Linear(D,256) → Linear(256,3D), 991-1005) + `self_attn`/`cross_attn` (`Attention` 309-460: `q_proj, k_proj, v_proj, output_proj`, all bias-free, `q_norm`/`k_norm` RMSNorm per head, 3D RoPE on self-attn only) + `mlp` (`GPT2FeedForward`: `layer1`, GELU, `layer2`, 245-272). Timestep path: `Timesteps` sinusoidal on **raw t** (674-694) → `TimestepEmbedding` returns `(emb_B_T_D=sinusoid, adaln_lora_B_T_3D=linear_2(silu(linear_1(sin))))` (697-731) → `t_embedding_norm` RMSNorm. `x_embedder = PatchEmbed` (Rearrange + Linear, patch 2x2x1, 789-856) with an extra all-zero padding-mask channel (`concat_padding_mask=True`, 1391-1397). `final_layer` = LN + AdaLN(shift,scale) + Linear → unpatchify (859-934, 1411-1419).
- Geometry inferred from the checkpoint in `get_dit_config` (`cosmos_predict2.py:122-168`): `model_channels` = `x_embedder.proj.1.weight.shape[0]`; `in_channels = shape[1]//4 - 1` (=16); `num_blocks = count_blocks(...)` (39-50, commit `632e540` "Support Anima models with different number of layers"); heads by width (2048→16, 5120→40, 1280→20); `crossattn_emb_channels=1024`; `rope3d`, learnable pos emb, `max_img_h/w=1024`, `max_frames=128`, RoPE h/w extrapolation ratio 4.0 for in_channels 16; `use_adaln_lora=True, adaln_lora_dim=256`.
- State-dict prefixes stripped: `^net\.` (Cosmos original) and `^model\.diffusion_model\.` (native ComfyUI) (`cosmos_predict2.py:279-285`, commit `5e64d52`). Weights are materialized with `accelerate.init_empty_weights` + `set_module_tensor_to_device` (303-309); saved full models re-add `net.` (`save_model` 337-339).
- `use_llm_adapter` is enabled if `llm_adapter_path` is given or `llm_adapter.out_proj.weight` is in the DiT state dict (289-301) — Anima checkpoints embed it.

### 3.2 LLM adapter (`models/llm_adapter.py`)
`LLMAdapter(source_dim=1024, target_dim=1024, model_dim=1024, num_layers=6, self_attn=True)` (`cosmos_predict2_modeling.py:1256-1263`): `embed = nn.Embedding(32128, 1024)` over **old-T5 token ids**, `in_proj = Identity`, RoPE (theta 10000), 6× `TransformerBlock` (`llm_adapter.py:117-161`: RMSNorm → self-attn over T5-token sequence (masked by `t5_attn_mask`) → RMSNorm → cross-attn to Qwen3 hidden states (masked by Qwen `attn_mask`) → RMSNorm → MLP GELU; attention class `Attention` with `q_proj/k_proj/v_proj/o_proj`), `out_proj`, final RMSNorm. Output: `(B, 512, 1024)` pseudo-T5 embeddings; padded positions zeroed (`cosmos_predict2.py:608`). This is layer index 1 of the pipeline (`LLMAdapterLayer` 592-610).

### 3.3 Text encoder (Qwen3-0.6B) (`cosmos_predict2.py:248-272, 171-188, 356-363`)
- `llm_path` a directory → generic `AutoTokenizer` + `AutoModelForCausalLM`; a single `.safetensors` → tokenizer/config from `configs/qwen3_06b/` (hidden 1024, 28 layers, 16 heads / 8 KV heads, vocab 151936, `tie_word_embeddings`) and `Qwen3ForCausalLM` built on meta device then filled via `iterate_safetensors`. `self.text_encoder = text_encoder.model` (decoder stack, no LM head), `use_cache=False`, `pad_token = eos_token` if missing, frozen (`requires_grad_(False)`).
- Tokenization: `padding='max_length', max_length=512, truncation=True` for **both** Qwen3 and the old-T5 tokenizer (`configs/t5_old/`); cached per caption: `prompt_embeds (512,1024)`, `attn_mask (512)`, `t5_input_ids (512)`, `t5_attn_mask (512)`.
- Embedding = `outputs.last_hidden_state` (final layer, post final RMSNorm) with padded positions set to 0 (`_compute_text_embeddings` 180-188). No hidden-layer selection, no chat template, no special prefix.
- `cache_text_embeddings = false` keeps Qwen3 inside `InitialLayer` and computes under `torch.no_grad()` (`cosmos_predict2.py:567-569`) — enables on-the-fly captions but never TE gradients.
- `text_encoder_fp8` / `text_encoder_nf4` options only apply to the T5 (Cosmos-Predict2) path, not Qwen3.

### 3.4 VAE (`cosmos_predict2.py:63-119, 209-217, 322-323, 348-354`)
`WanVAE` wrapper around `models/wan/vae2_1.py::WanVAE_` (dim 96, z_dim 16, `temperal_downsample=[F,T,T]`, 8x spatial) loaded from `vae_path` (docs point at `qwen_image_vae.safetensors`, architecturally the Wan2.1 VAE). Hard-coded Wan2.1 per-channel `mean`/`std` (98-105); `scale = [mean, 1/std]` passed to `model.encode(tensor, scale)`. Pixels `[0,1] → *2-1`. Media are processed as 1-frame videos (`support_video=True, framerate=16`, 341-346) so latents are 5D `(16, 1, H/8, W/8)`. `get_vae()` returns the raw `nn.Module`.

### 3.5 Flow-matching inputs, timestep sampling, loss (`cosmos_predict2.py:372-422, 506-544`)
```python
def prepare_inputs(self, inputs, timestep_quantile=None):
    latents = inputs['latents'].float(); mask = inputs['mask']
    ...  # mask -> (bs,1,1,h,w) nearest-exact resize to latent size
    timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')
    dist = Normal(0,1) if logit_normal else Uniform(0,1)
    t = dist.icdf(full((bs,), timestep_quantile)) if timestep_quantile is not None else dist.sample((bs,))
    if logit_normal: t = torch.sigmoid(t * self.model_config.get('sigmoid_scale', 1.0))
    if shift := self.model_config.get('shift'):      t = (t*shift) / (1 + (shift-1)*t)
    elif self.model_config.get('flux_shift'):        t = time_shift(get_lin_function(y1=0.5,y2=1.15)((h//2)*(w//2)), 1.0, t)
    noise = torch.randn_like(latents); t_expanded = t.view(-1,1,1,1,1)
    noisy_latents = (1 - t_expanded)*latents + t_expanded*noise
    target = noise - latents
    return (noisy_latents, t.view(-1,1), *prompt_embeds_or_batch_encoding), (target, mask)
```
- t ∈ (0,1) with t=1 pure noise; the DiT receives **raw t** (not ×1000, unlike Wan `wan.py:370`). Author's note (365-371): NVIDIA's formulation is equivalent to rectified flow with an implicit `t² + (1-t)²` loss weight, which was **deliberately dropped**; the raw `final_layer` output is trained directly as velocity `noise - x`. No `min_t`/`max_t` here (only Wan has it).
- Loss (`get_loss_fn` 506-544): fp32 MSE (or huber/smooth-L1) × mask, mean; optional `multiscale_loss_weight`: for side length ≥ 0.9·1024 px adds avg-pooled 2x MSE terms (weights normalised).
- Per-group LRs (`get_param_groups` 463-504): `self_attn_lr`, `cross_attn_lr`, `mlp_lr`, `mod_lr` (adaln_modulation), `llm_adapter_lr`, base `optimizer.lr`; `lr = 0` sets `requires_grad_(False)` (and the saver therefore skips those params).

### 3.6 LoRA targets and saved key format
- `adapter_target_modules = ['Block', 'TransformerBlock']` (`cosmos_predict2.py:195-198`) → every `nn.Linear` inside DiT `Block`s: `blocks.N.{self_attn,cross_attn}.{q_proj,k_proj,v_proj,output_proj}`, `blocks.N.mlp.{layer1,layer2}`, `blocks.N.adaln_modulation_{self_attn,cross_attn,mlp}.{1,2}`; plus inside `llm_adapter.blocks.N` (`TransformerBlock`): `{self_attn,cross_attn}.{q_proj,k_proj,v_proj,o_proj}`, `mlp.{0,2}`. Not targeted: `x_embedder`, `t_embedder`, `final_layer`, `llm_adapter.{embed,in_proj,out_proj}`. Unlike MiniMax (`docs/minimax_h3_notes.md`), **AdaLN linears are LoRA-targeted** for Anima. Use `exclude_modules` to trim.
- Saved keys: `diffusion_model.blocks.N.self_attn.q_proj.lora_A.weight` / `...lora_B.weight` (PEFT names with `.default` removed, `diffusion_model.` prefixed, `cosmos_predict2.py:331-335`) + `adapter_config.json`. LoKr: `diffusion_model.<module>.lokr_w1`, `lokr_w2` etc. (generic `.default` strip in `saver.py:74`, flagged TODO).
- `init_from_existing` works through `BasePipeline.load_adapter_weights` (transformer./diffusion_model. prefixes accepted).

### 3.7 fp8 / quantization
- Supported: `transformer_dtype = 'float8_e5m2'|'float8'|...` casts all ≥2-D DiT weights except `KEEP_IN_HIGH_PRECISION = ['x_embedder','t_embedder','t_embedding_norm','final_layer']` and 1-D params (`cosmos_predict2.py:33, 308, 314`); autocast upcasts on the fly. Docs (`supported_models.md:272-274`) warn e4m3fn works badly for Cosmos-Predict2, use e5m2.
- **Why Anima is excluded from "train on quantized ComfyUI weights"**: that feature lives in `ComfyPipeline` (`comfy.sd.load_diffusion_model` → `QuantizedTensor` params → `patch_quantized_modules`/`dequantize`, `base.py:593-674`). Anima is a `BasePipeline` with its own loader (`utils/common.py::load_state_dict`), which **raises** on any `scale_input`/`scale_weight` key (`common.py:83-84, 101-102`), so fp8_scaled/int8 ComfyUI checkpoints are rejected outright.

### 3.8 Other Anima quirks
- Class attr `name = 'cosmos_predict2'` but instance sets `'anima'` → cache dir `<dir>/cache/anima/` (268).
- Padding tokens are zero vectors, not masked (cross-attention has no mask), so cache/inference must reproduce the zeroing exactly.
- RoPE buffers were made `persistent=False` and `max_img_h/w` raised to 1024 (commit `b0aa4f1`) so high-res training doesn't fail on checkpoint key mismatch.
- Relative paths `configs/t5_old/...`, `configs/qwen3_06b` — must run from the repo root (220-223, 256-257).
- `sample()`/`--test_sample` unusable: `CosmosPredict2Pipeline` defines no `get_conds`/`vae_decode`.

---

## 4. Dataset pipeline — `utils/dataset.py` (1480 lines), `utils/cache.py`, `models/base.py::PreprocessMediaFile`

### 4.1 Dataset TOML schema (`examples/dataset.toml`, read in `DirectoryDataset.__init__` `dataset.py:449-522`, `_set_defaults` 731-736)
Top-level (overridable per `[[directory]]`): `resolutions` (list of side lengths or `[w,h]` pairs → areas; >3 aborts without `--i_know_what_i_am_doing`, 524-531), `enable_ar_bucket`, `min_ar`/`max_ar`/`num_ar_buckets` (geomspace, 502-505) or explicit `ar_buckets`, `frame_buckets` (1 always added, 508-513), `size_buckets` (`[w,h,frames]` explicit, disables AR logic, 460-466), `cache_shuffle_num` + `cache_shuffle_delimiter` (pre-cache tag shuffling into N caption variants, 48-57, 761-763), `shuffle_tags` (legacy → `cache_shuffle_num=1`), `caption_prefix`, `num_repeats`, `shuffle_metadata` (default True), `skip_empty_caption` (default True), `online_captions` (read `captions.json` at train time, 515-522), `subsample_ratio` (985-987). Per directory: `path`, `mask_path`, `default_mask_file`, `control_path` (edit datasets), and any override above. Caption sources: `<stem>.txt` or `captions.json` mapping filename → list of captions (multi-caption). `.tar` archives are enumerated as image sources (636-640). The cache root is **inside the dataset dir**: `<path>/cache/<model.name>/` (482).

### 4.2 Object hierarchy
`Dataset` (927-1048) → `DirectoryDataset` per `[[directory]]` → `ARBucketDataset` per (AR, frames) (399-445) → `SizeBucketDataset` per (AR, w, h, frames) (207-337) — or directly `SizeBucketDataset` when `size_buckets` is used. `Dataset.post_init` (953-987) regroups all `SizeBucketDataset`s across directories by identical size bucket into `ConcatenatedBatchedDataset`s (342-396).

### 4.3 Metadata + fingerprinted caching
- `_get_ungrouped_metadata` (621-729): enumerate files → HF `datasets.Dataset.from_dict({image_spec, caption_file, mask_file[, control_file]})` → optional `captions.json` map → deterministic shuffle seeded by `md5(path)` → `save_to_disk(metadata/metadata_intermediate)` → `map(_metadata_map_fn, batched=True, batch_size=1, num_proc=NUM_PROC, cache_file_name=metadata/metadata.arrow)` which opens each file (PIL / Comfy `VideoFromFile`), computes `log_ar`, frames (`int(source_frames * model.framerate / source_fps)`), assigns AR or size bucket (838-871; videos never map to frame bucket 1).
- `_group_metadata_and_save_to_disk` (589-619): groups by bucket key → `metadata/grouped_metadata_<key>/` + `metadata/grouping_keys.json`. `--trust_cache` skips re-enumeration (`load_from_cache_file` gates).
- `ARBucketDataset.cache_latents` (416-439): for each resolution computes `w = sqrt(area*ar)`, `h = area/w`, rounded to `model.pixels_round_to_multiple` (16), writes `ar_frames_<ar>_<f>/metadata/metadata_<ar>x<w>x<h>x<f>.arrow`.
- `_map_and_cache` (85-161): fingerprint = `Hasher.hash([extra_args..., dataset._fingerprint])` → `utils/cache.py::Cache(dir, fingerprint, shard_size_gb=10)`: SQLite `metadata.db` (tables `fingerprint`, `items(shard, shard_index)`, `shard_N(offset,size)`) + `shard_N.bin` files, each item `torch.save`d as bytes (`cache.py:109-133`). Fingerprint mismatch → `clear()`. **Resumable**: already-cached prefix is skipped via `dataset.select(range(cache_size, dataset_size))`. Mapping runs through a `multiprocess.Pool(NUM_PROC)` with per-worker rank from a `Manager().Queue()`.
- Latents per size bucket: `cache_<ar>x<w>x<h>x<f>/latents/` (`SizeBucketDataset.cache_latents` 234-299) + `iteration_order/` HF dataset of `(image_spec, latents_idx, caption, caption_number)` (built so that equal-caption-count datasets read shards mostly sequentially, 264-287).
- Text embeddings **per AR bucket** (shared across resolutions): `ar_frames_<ar>_<f>/text_embeddings_<i>/` via `_cache_text_embeddings` (178-202; captions flattened so each caption is one row, `TextEmbeddingDataset` maps `(image_spec, caption_number)` → row). Uncond (empty caption) embedding: `<cache_root>/uncond_text_embeddings_<i>/` (905-921), also exposed as `model.uncond_dict` (1048) for CFG-augmented training.

### 4.4 Caching process architecture (`_cache_fn` 1051-1136, `DatasetManager` 1141-1270)
Rank 0 spawns a `multiprocess.Process(_cache_fn)` which runs the HF `map`s; its pool workers decode/resize media (`PreprocessMediaFile.__call__` `base.py:133-212`: RGBA→white bg, `ImageOps.fit` crop to bucket, rounding to 16 px / 4 frames, mask from R channel, framerate resample, clip extraction) and push `(task_id, tensor, control_tensor, audio_list, pipe)` on a broadcast `Manager().Queue()`. **Every GPU rank** loops `queue.get()` → `_handle_task` (1229-1270): moves only the needed submodel (`[vae] + text_encoders`, id 0 = VAE, i≥1 = TE) to CUDA, calls `call_vae_fn` / `call_text_encoder_fns[i-1]`, sends CPU results back through the worker's `mp.Pipe`. `torch.set_num_threads(1)` workaround for HF map hangs (1056-1059); tensors over pipes use `utils/reduction.py` (torch multiprocessing reductions ported to `multiprocess`). After caching: submodels moved to `meta` (freed) except SDXL VAE; `mm.unload_all_models()`. Then every rank reloads datasets from cache (`ds.cache_latents(None, trust_cache=True)`).

### 4.5 Batch formation
- `ConcatenatedBatchedDataset.post_init` (347-380): pick `global_batch_size` (per-resolution dict matched by nearest `sqrt(w*h)`; image vs video dict by `frames==1`), truncate to a multiple of it (warns/drops tiny buckets, 392-396), `batch_size = global/dp_world`. `__getitem__(idx)` returns this DP rank's slice.
- `Dataset.post_init` shuffles bucket order deterministically (seed 0) → `iteration_order = [(bucket_i, j)]`; `__getitem__` collates (`_collate` 1005-1034: stack equal-shape tensors, masks → ones for missing). One DataLoader item = **GAS × micro_bs examples for this rank**.
- `SizeBucketDataset.__getitem__` (309-334): latents row + text-embedding dict (or uncond with prob `UNCOND_FRACTION`) + `caption` string; `__len__ = len(iteration_order) * num_repeats`.
- `PipelineDataLoader` (1302-1435): infinite iterator; `_pull_batches_from_dataloader` calls `model.prepare_inputs(batch, timestep_quantile)`, broadcasts targets to the last stage, `split_batch` into GAS micro-batches (1273-1281); tracks `epoch`, `num_batches_pulled`; `sync_epoch` all-gathers; `state_dict`/`load_state_dict` for resume; `num_dataloader_workers=1` (train), 0 (eval).

### 4.6 Video / audio / edit hooks
`PreprocessMediaFile(config, support_video, support_audio, framerate, audio_sample_rate, round_height, round_width, round_frames)`; `DatasetManager` detects `audio` kwarg on the VAE fn and 3-arg TE fns (`control_file`) by signature (1148-1153); `is_video` list passed to TE fns; frame buckets + `video_clip_mode`; `control_path` for Kontext/Qwen-Edit/Flux2 edit; `mask_path` for loss masks.

### 4.7 Eval datasets
Each `eval_datasets[i].config` is a full dataset TOML with the same machinery; its own `eval_micro_batch_size_per_gpu` and `eval_gradient_accumulation_steps` define bucket rounding (docs warn small eval sets lose images). Eval loader created with `num_dataloader_workers=0` (`train.py:898-901`).

---

## 5. `optimizers/`

Dispatch is in `train.py:650-815` (`get_optimizer`): `type` lower-cased → `adamw` (torch.optim.AdamW), `adamw8bit` (bitsandbytes), `adamw_optimi` / `stableadamw` (torch-optimi, Kahan built-in), `sgd`, `adamw8bitkahan`, `offload` (torchao `CPUOffloadOptimizer(AdamW, fused=True)`), `automagic`, `genericoptim`, else `getattr(pytorch_optimizer, type)` (Prodigy, etc.). All other `[optimizer]` keys become kwargs. Param groups: `model.get_param_groups(params)` then split into weight-decay / no-weight-decay (1-D params and `llm_adapter.embed*`, `train.py:789-813`). `beta2_half_life` → `beta2 = 0.5 ** (global_batch_size / half_life)` (658-663). `DummyOptimizer` when a stage has no trainable params (`train.py:61-76`).

| optimizer | file | notes |
|---|---|---|
| `AdamW8bitKahan` | `optimizers/adamw_8bit.py:6-124` | subclass of `bitsandbytes.optim.AdamW8bit`; adds `stabilize` (StableAdamW RMS-scaled LR, 43-47) and Kahan compensation |
| `Automagic` | `optimizers/automagic.py:34-467` | AI-Toolkit port: Adafactor-style factored 2nd moment, per-element LR mask (int8 `Auto8bitTensor`) bumped by sign agreement, Kahan for bf16 (308-318), custom state_dict for lr_mask |
| `GenericOptim` | `optimizers/generic_optim.py:250-657` | sn-sm port + Muon/AdaMuon/NorMuon (`zeropower_via_newtonschulz5` 152-179, Polar Express 191-233), subset-norm / subspace-momentum projectors, `cpu_offload`, `kahan_buffer_offload`, `automagic` mode (568-597), `skip_invalid_grads`, computes `self._grad_norm` (390-391, 508-511) — the only optimizer that logs grad norm; `train.py:760-785` splits 2-D vs other params and injects `rank/proj_type/update_proj_gap`, `mpu` |
| `GradientReleaseOptimizerWrapper` | `optimizers/gradient_release.py:5-27` | holds per-parameter optimizers; `step`/`zero_grad` are no-ops (hooks do the work) |
| projectors | `optimizers/projectors/{svd_projector,approx_svd,topk_norm_projector,uniform_projector}.py` | GaLore-style low-rank gradient projections for GenericOptim |
| utils | `optimizers/optimizer_utils.py` | `copy_stochastic` (stochastic rounding), `Auto8bitTensor`, quanto `QBytesTensor` handling |

**Kahan summation mechanism** (`adamw_8bit.py:12-14, 122-124`): a bf16 `shift` buffer per parameter (same dtype as `p`) accumulates the rounding error lost when adding a tiny update to a bf16 weight:
```python
def init_state(self, group, p, gindex, pindex):
    super().init_state(group, p, gindex, pindex)
    self.state[p]['shift'] = self.get_state_buffer(p, dtype=p.dtype)   # compensation buffer
...
# bitsandbytes kernel writes the update INTO `shift` (passed where `p` normally goes), then:
buffer = p.clone()
p.add_(shift)                 # apply accumulated update (rounded to bf16)
shift.add_(buffer.sub_(p))    # shift = shift + (old_p - new_p) = the part of the update that was lost to rounding
```
Same pattern in `automagic.py:308-318` and `generic_optim.py:486-497` (there `p.grad` is reused as the temp buffer and the shift can live on CPU). This substitutes for fp32 master weights.

**Gradient release** (`train.py:702-759`): one optimizer per parameter, stepped in `register_post_accumulate_grad_hook`; asserts DP world size 1, patches `PipelineEngine._INSTRUCTION_MAP[ReduceGrads]` to no-op, monkeypatches `p.add_` to write `p.data` (author: "unbelievably hacky and not mathematically sound"), scales betas/momentum by `**(1/GAS)`, disables grad clipping.

---

## 6. `utils/` modules

| module | purpose |
|---|---|
| `utils/common.py` | `DTYPE_MAP` (14-21, incl. `float8`, `float8_e4m3fn`, `float8_e5m2`), `VIDEO_EXTENSIONS` from imageio, `AUTOCAST_DTYPE` global, rank helpers, `zero_first`/`one_at_a_time` barriers, `load_state_dict` (76-85, rejects fp8_scaled), `iterate_safetensors` (88-103), rounding helpers, `time_shift`/`get_lin_function` (Flux shift), `get_t_distribution` / `slice_t_distribution` / `sample_t` (124-160: 10k-bucket discretised icdf table used by Wan for `min_t`/`max_t`) |
| `utils/dataset.py` | everything in §4 |
| `utils/cache.py` | SQLite-indexed binary shard cache (§4.3) |
| `utils/offloading.py` | Musubi-tuner block swap adapted for PEFT: `swap_weight_devices_cuda` (43-99) swaps `.weight.data` (or `_qdata`) between a CPU-resident and GPU-resident block on a side stream, **skipping any module with `'lora'` in its name** so trainable params stay on GPU; `weights_to_device` (127-135); `ModelOffloader` (201-318): `prepare_block_devices_before_forward` puts first `num_blocks-blocks_to_swap` blocks on GPU and the rest on CPU, `wait_for_block`/`submit_move_blocks_forward` in the forward path, `register_full_backward_hook` per block for the backward direction, 1-thread `ThreadPoolExecutor`; honours reentrant checkpointing (skips swaps during recompute) and `forward_only` for eval |
| `utils/saver.py` | §1.7 |
| `utils/pipeline.py` | `ManualPipelineModule(PipelineModule)` with `partition_method='manual'` + `manual_partition_split` (11-53) |
| `utils/patches.py` | `apply_patches()` (408-439): PEFT fp8 downcast fix (34-69); HunyuanVideo TE loader; DeepSpeed `TrainSchedule.steps` reordering (113-160); `DeepSpeedEngine._broadcast_model` that skips frozen params and handles CPU-resident (block-swapped) params (163-172); `clip_fp32_gradients` → custom `clip_grad_norm_` tolerant of stages with no grads (175-246); `reduction.init_reductions()`; torch inductor `copy_args_to_cpu_if_needed` patch (249-294); Comfy Flux `DoubleStreamBlock`/`SingleStreamBlock` forwards rewritten without in-place ops (297-405) |
| `utils/unsloth_utils.py` | `Unsloth_Offloaded_Gradient_Checkpointer` (24-79): autograd Function that saves inputs ≥ 5M elements to CPU (non-blocking), recomputes forward in backward, skips tensors flagged `no_backward` (used by MiniMax CFG branch) |
| `utils/isolate_rng.py` | Lightning's `isolate_rng` context manager (torch/cuda/numpy/python RNG save+restore) used by eval |
| `utils/reduction.py` | copy of `torch.multiprocessing.reductions` re-targeted at the `multiprocess` library so tensors cross Queues/Pipes by shared memory |

Model save formats: adapters → `adapter_model.safetensors` (+ `adapter_config.json`, config TOML copy); full models → `model.safetensors` with model-specific prefixes (`net.` for Cosmos-P2/Anima `cosmos_predict2.py:337-339`; raw keys for ComfyPipeline `base.py:733-734`; SDXL original checkpoint layout).

---

## 7. Loss and timestep sampling (cross-model)

- Distribution: `timestep_sample_method` = `logit_normal` (default; `sigmoid(N(0,1) * sigmoid_scale)`) or `uniform`; then optional `shift` (`t*s / (1+(s-1)t)`) **or** `flux_shift` (resolution-dependent `mu` via `get_lin_function(y1=0.5,y2=1.15)` on token count, `time_shift(mu,1.0,t)`). Implemented per model (Anima: `cosmos_predict2.py:391-414`; Krea2 `krea2.py:92-115`; MiniMax `minimax_h3.py:368-394` adds `image_shift` for single-frame batches). Wan is the only model using the shared discretised `get_t_distribution` with `min_t`/`max_t` (`wan.py:78, 352-361`; docs `supported_models.md:345-355`).
- Quantile eval: `timestep_quantile` → `dist.icdf(q)` before sigmoid/shift, so the 9 eval quantiles track the *training* distribution.
- Loss weighting options: none generic. Model-specific: Anima/Cosmos-P2 `multiscale_loss_weight`; SDXL `min_snr_gamma`, `debiased_estimation_loss`, `v_pred` (`sdxl.py:335-383`); `huber_delta` / `smooth_l1_beta` in the default loss. NVIDIA's `t²+(1-t)²` weighting intentionally omitted for Anima.
- Masked loss: mask (R channel, fp16) resized nearest-exact to latent grid, multiplied elementwise, then plain `.mean()` (note: not renormalised by mask sum).
- Conditioning dropout: `uncond_fraction` swaps in the cached empty-caption embedding (`dataset.py:315-317, 330-332`).
- **CFG-augmented training (MiniMax H3 only)** (`minimax_h3.py:409-420, 655-673, 689-702, 730-742`; `docs/minimax_h3_notes.md`): with `[model].cfg = N > 1`, `prepare_inputs` appends the batch-repeated **uncond** text embeds (from `model.uncond_dict`) to the features; every pipeline layer runs a second `torch.no_grad()` branch on the uncond stream (tensors flagged `no_backward` so the unsloth checkpointer doesn't save them); `FinalLayer` recombines `out = (cond + (cfg-1)*uncond) / cfg` before the standard flow-matching MSE, so the guidance-distilled model's raw velocity is what gets fitted and distillation is preserved. Not wired for Anima.

---

## 8. Multi-GPU design and single-GPU experience

- Hybrid PP × DP via DeepSpeed's `PipelineModule` topology: `pipeline_stages` = PP degree, DP degree = `num_gpus / pipeline_stages` (`model_engine.grid`). `partition_method` `'parameters'` is monkeypatched to count **all** params, not just trainable, so LoRA runs partition evenly (`train.py:79-90`); `'manual'` + `partition_split` for uneven GPUs (`utils/pipeline.py`). `dynamic_shape=True` + `reset_activation_shape()` every step because bucket shapes vary. First/last-stage process group for target broadcast (`train.py:821-823`, `dataset.py:1387-1405`). Only first/last stages read the dataloader (`get_data_iterator_for_step` `train.py:167-173`). `communication_data_type` set to the model dtype (843-844).
- Gradient release constrains DP world size to 1 (`train.py:713-716`).
- **Single-GPU user still pays for all of it**: must launch with `deepspeed --num_gpus=1 train.py --deepspeed --config ...` (README), NCCL init (`NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` recommended for RTX 40xx), `deepspeed.init_distributed`, a 1-stage `PipelineEngine` interpreting `LoadMicroBatch/ForwardPass/BackwardPass/ReduceGrads/OptimizerStep` instructions, `broadcast_object_list`/`barrier` calls, DeepSpeed checkpoint format (`global_stepN/`), and ~8 monkeypatches of DeepSpeed internals pinned to `deepspeed==0.18.4` (`_count_layer_params`, `TrainSchedule.steps`, `_broadcast_model`, `clip_fp32_gradients`, `_support_torch_style_backward`, `PipelineModule.to`, `_report_progress`, `_INSTRUCTION_MAP[ReduceGrads]`). No native Windows (README "Windows support").

---

## 9. Weaknesses / limitations (verified)

1. **Hard DeepSpeed dependency** (23 import sites across 13 files; `requirements.txt` pins `deepspeed==0.18.4`); README states native Windows is "difficult or impossible", WSL2 only.
2. **Hard ComfyUI + HunyuanVideo submodule dependency even for Anima**: `models/base.py:8,24-29` imports `comfy.*` at module import; `utils/dataset.py:12,25,29` imports `comfy_api` / `comfy.model_management`; `utils/patches.py:4-5,27-31` imports `hyvideo` and `comfy.ldm.flux`. Plus `comfy-kitchen`, `comfy-aimdo`, `bitsandbytes`, `optimum-quanto`, `torchaudio` (imported in `base.py:12`, **not in requirements.txt**).
3. **No text-encoder training**: embeddings are pre-cached (README "training LoRAs for text encoders is not currently supported"); Anima's Qwen3 is `requires_grad_(False)` (`cosmos_predict2.py:272`) and the non-cached path is under `torch.no_grad()` (567). SDXL is the sole exception.
4. **No sample image generation during training** — confirmed. Only `--test_sample` (one image then `quit()`, `train.py:526-527, 635-641`) which requires `get_conds`/`vae_decode` that Anima's pipeline lacks. No validation images, no EMA, no FID/CLIP metrics; eval = loss only.
5. **No web API / callbacks / progress reporting** (no flask/fastapi/gradio anywhere); control only via CLI flags and `save`/`save_quit` signal files.
6. **Monolithic `train.py`** (975 lines; ~700 in the `__main__` block); config is an untyped dict with defaults spread over 4+ files; stale/mismatched keys (`config['lora']` 843, `pseudo_huber_c` documented but unread, `eval_datasets` str-form breaks line 557).
7. **bf16 params without master weights**: choosing `adamw`/`adamw8bit`/`sgd` with bf16 LoRA silently loses small updates; correctness is delegated to Kahan optimizers. Gradient release is self-described as unsound (`train.py:720`).
8. **Adapter rigidity**: `alpha` forbidden (=rank); targets are whole-class `nn.Linear` sweeps (Anima LoRA hits AdaLN modulation linears too); no per-module rank, no DoRA/LoHa, no LoRA on embedders/final layer; block swap only with LoRA and `pipeline_stages=1`; LoKr saving relies on a generic `.default` strip (TODO `saver.py:73`).
9. **Caching design**: all latents/embeddings must be pre-cached (no runtime augmentation, random crop, caption dropout other than `uncond_fraction`; tag shuffle only via N pre-cached variants); HF Datasets + forked `multiprocess` + `torch.set_num_threads(1)` hang workaround (`dataset.py:1054-1059`); TE RAM not fully freed after caching (docs recommend two-phase `--cache_only` run; TODO `dataset.py:1213`); text embeddings are fixed 512-token padded tensors (1 MB/caption for Anima); cache lives inside the dataset directory.
10. **Anima-specific**: fp8 only via naive dtype cast (no scaling), fp8_scaled rejected, excluded from quantized-weight training; relative `configs/` paths; the deliberately dropped NVIDIA loss weighting is a divergence from the reference objective; raw `t` fed to the DiT while other models use `t*1000` (convention inconsistency to verify against ComfyUI's `comfy/ldm/cosmos/predict2.py`, which is not checked out here); cross-attention cannot mask padding.
11. **Eval cost**: 9 quantiles × full eval set per eval; small eval sets lose examples to batch-size rounding.
12. Hard-coded oddities: `mp.current_process().authkey = b'afsaskgfdjh4'` (`train.py:35`), `model_management.EXTRA_RESERVED_VRAM = 2000 MiB` (`base.py:34`), `TIMESTEP_QUANTILES_FOR_EVAL`, dataloader workers fixed at 1/0.
13. TODO/FIXME inventory (non-vendored): `train.py:590` (block swap vs DeepSpeed non-reentrant checkpoint), `:669` (fused Adam build failure), `:720` (gradient release hack), `:911` (epoch_loss not checkpointed); `utils/dataset.py:169, 910, 933, 1054, 1204, 1213`; `utils/saver.py:73`; `utils/patches.py:410` (PEFT fp8 bug); `optimizers/generic_optim.py:496, 648`; `models/base.py:108, 306` (LoKr init); `models/minimax_h3.py:449, 651, 652, 666, 691`; `models/cosmos.py:118, 147`; `models/chroma.py:51`; `models/qwen_image.py:216`; `models/ltx2.py:177, 202, 495, 619`; `models/sdxl.py` (7 TODOs); `models/hunyuan_image.py:240`.

---

## 10. `docs/supported_models.md` extracts

**Anima** (`supported_models.md:542-562`), verbatim:
```
## Anima
[model]
type = 'anima'
transformer_path = '/data2/imagegen_models/comfyui-models/anima-preview.safetensors'
vae_path = '/data2/imagegen_models/comfyui-models/qwen_image_vae.safetensors'
llm_path = '/data2/imagegen_models/comfyui-models/qwen_3_06b_base.safetensors'
dtype = 'bfloat16'
# Comment out to train the llm_adapter, or adjust the learning rate to be >0.
llm_adapter_lr = 0

Use the official ComfyUI format model files (https://huggingface.co/circlestone-labs/Anima).

Notes:
- Might need to use lower learning rate than other models.
- You can control the llm_adapter learning rate separately. This is an adapter that processes the Qwen3 embeddings before feeding into the diffusion model.
  - Setting `llm_adapter_lr=0` disables training it entirely. This probably makes training more stable for small datasets.
  - If you have a larger dataset or a lot of brand-new concepts, you can try training the llm_adapter and see if it helps.

Anima LoRAs are saved in ComfyUI format.
```
Summary table row (line 26): `Anima | LoRA ✅ | Full Fine Tune ✅ | fp8/quantization ✅`.

**Krea 2** (`supported_models.md:620-633`), verbatim:
```
## Krea 2
[model]
type = 'krea2'
diffusion_model = '/data2/imagegen_models/comfyui-models/krea2_raw.safetensors'
vae = '/data2/imagegen_models/comfyui-models/qwen_image_vae.safetensors'
text_encoders = [
    {path = '/data2/imagegen_models/comfyui-models/qwen3vl_4b_bf16.safetensors', type = 'krea2'}
]
dtype = 'bfloat16'
diffusion_model_dtype = 'float8'
timestep_sample_method = 'logit_normal'

This configuration can train a rank 32 LoRA at 512 resolution with 24GB VRAM.
```
(Implementation `models/krea2.py`: `ComfyPipeline`, `adapter_target_modules = ['SingleStreamBlock','TextFusionTransformer']` plus `txtmlp` override, Wan21 latent format, same InitialLayer/TransformerLayer/FinalLayer pattern.)

**LoRA output-format notes across the doc**: SDXL → Kohya sd-scripts format (line 54); Flux, SD3, AuraFlow, Flux Kontext → Diffusers format (`transformer.` prefix; works in ComfyUI) (74, 255, 309, 442); HunyuanVideo → Diffusers-style with `transformer.` prefix (112); LTX-Video, Cosmos, Lumina 2, Wan2.1/2.2, Chroma, HiDream, Cosmos-Predict2, OmniGen2, Qwen-Image(-Edit), HunyuanImage-2.1 (key names differ from the original model), Z-Image ("different from Diffusers format"), HunyuanVideo-1.5, Flux 2, Anima, Ernie-Image, LTX 2.3, Ideogram4, Krea 2, MiniMax H3 → ComfyUI format (`diffusion_model.` prefix). README "Output files": each saved dir has safetensors weights, PEFT `adapter_config.json`, and the diffusion-pipe config copy. Related Cosmos-Predict2 caveat (272-274): fp8 e4m3fn degrades badly, use e5m2; LoRAs applied on e5m2 weights in ComfyUI produce noisy images (GGUF fine).

---

## 11. `requirements.txt` and `.gitmodules`

`requirements.txt` (verbatim): `deepspeed==0.18.4, toml, transformers, diffusers>=0.35.1, datasets, pillow, sentencepiece, protobuf, peft, torch-optimi, tensorboard, tqdm, safetensors, bitsandbytes, imageio[ffmpeg], av, einops, accelerate, loguru, omegaconf, iopath, termcolor, hydra-core, easydict, ftfy, pytorch-optimizer, wandb, optimum-quanto, wheel, ninja, comfy-kitchen, comfy-aimdo, scipy, av>=16.0.0`. Not listed but required: `torch`/`torchvision` (installed manually per README), `torchaudio` (imported by `models/base.py`), `multiprocess` (transitive via `datasets`), optional `flash-attn`, `transformer_engine` (original Cosmos only), `torchao` (`offload` optimizer), `fast_hadamard_transform` (approx SVD).

`.gitmodules` (9 submodules, all empty in this checkout): `submodules/HunyuanVideo` (Tencent/HunyuanVideo), `submodules/Cosmos` (NVIDIA/Cosmos), `submodules/Lumina_2` (Alpha-VLLM/Lumina-Image-2.0), `submodules/flow` (lodestone-rock/flow, Chroma), `submodules/HiDream` (HiDream-ai/HiDream-I1), `submodules/LTX_Video` (Lightricks/LTX-Video), `submodules/OmniGen2` (VectorSpaceLab/OmniGen2), `submodules/HunyuanImage-2.1` (Tencent-Hunyuan/HunyuanImage-2.1), `submodules/ComfyUI` (comfyanonymous/ComfyUI). Anima itself needs none of them for its model code (DiT and adapter are vendored), but `base.py`/`dataset.py`/`patches.py` import ComfyUI and HunyuanVideo unconditionally.

Local bundled configs relevant to Anima: `configs/qwen3_06b/{config.json, tokenizer.json, tokenizer_config.json, vocab.json, merges.txt}` and `configs/t5_old/` (old-T5 tokenizer used for adapter target ids).
