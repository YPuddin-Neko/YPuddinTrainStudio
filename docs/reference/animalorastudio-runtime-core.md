# AnimaLoraStudio — Runtime Training Core Report (Section 1) + Requirements (Section 5)

Repo: `/Volumes/Service/Dev/YPuddinTrainStudio/AnimaLoraStudio`, v0.26.2 (2026-09-05). All paths below are relative to the repo root unless absolute. Line numbers are from the current tree.

---

## 1. `runtime/` training core

### 1.1 Directory layout and module responsibilities

```
runtime/
  anima_train.py        141   entry point; sys.path injection; re-exports; main() = 8 phases + loop
  anima_daemon.py      1926   JSON-over-stdio inference daemon (imports `anima_train as _T`, reuses families)
  anima_generate.py     527   CLI generation (legacy; studio no longer spawns it)
  anima_reg_ai.py       699   prior-preservation reg-set generator (base model, no LoRA)
  train_monitor.py      214   in-memory monitor state -> monitor_state.json writer (polled by studio)
  training/
    bootstrap.py        171   ensure_dependencies / load_yaml_config / apply_yaml_config / init_progress
    cli.py              169   parse_args (argparse auto-generated from pydantic TrainingConfig) + interactive prompts
    context.py          232   TrainingContext dataclass (all mutable state) + emit() + handle_interrupt()
    loop.py             991   THE training loop: run(ctx)
    state.py            247   save_training_state / load_training_state (.pt)
    snapshot.py         137   pause/auto-epoch paths, config snapshot JSON, emit_event(__EVENT__: protocol)
    sample_runner.py    180   run_sample(): periodic preview generation wrapper
    dataset.py         1743   BucketManager / ImageDataset / MergedDataset / BucketBatchSampler / CachedLatentDataset / NaViT packer
    text_cache.py       278   TextCacheStore: sidecar + prompt-bundle safetensors cache protocol
    block_swap.py       613   PinnedPacker / PinnedBlockSwap (CPU-pinned layer offload with hooks)
    sysmem.py           307   RAM/VRAM budget guards, pinned budget, trim_working_set, NVML helpers
    vae.py              384   VAEWrapper (auto/on/off tiling, OOM fallback) + load_vae (Wan2.1)
    model_loading.py    288   safetensors loading, prefix remap, path resolution, enable_xformers
    noise.py            100   make_noise (noise_offset / pyramid), noise_params_from_args
    timestep_sampling.py 127  sample_t (6 modes), apply_resolution_shift, latent_token_counts
    loss_weighting.py    64   compute_loss_weight (min_snr / detail_inv_t / cosmap)
    observability.py    404   WandBMonitor + ASCII loss curve
    phases/             bootstrap, models, dataset, text_cache, optimizer, resume, finalize  (run(ctx) each)
    families/           spec.py, protocol.py, latent_spaces.py, anima/*, krea2/*
    adapters/           __init__ (registry), protocol, lycoris, ortho, tlora
    optimizers/         __init__ (registry), adamw, automagic, came, lion, prodigy, prodigy_plus_schedulefree, soap, soap_sf
    schedulers/         __init__ (registry), cosine, cosine_with_restart, cosine_with_warmup
    losses/             __init__ (registry), protocol, mse, huber
    timestep_samplers/  __init__ (registry), protocol, baseline, infonoise, krea2_shift
    inference_samplers/ __init__ (registry), er_sde, dpmpp_3m_sde
modeling/
  anima/anima_modeling.py         257  Anima(MiniTrainDIT) + LLMAdapter (from ComfyUI comfy/ldm/anima)
  anima/cosmos_predict2_modeling.py 2072 NVIDIA Cosmos-Predict2 MiniTrainDIT (Apache-2.0) + attention backends
  krea2/krea2_modeling.py         536  Krea2 SingleStreamDiT (GPL-3.0 from ComfyUI + musubi-tuner layout)
  wan/vae2_1.py                   661  Wan2.1 VAE (Alibaba)
utils/
  lycoris_adapter.py   676  LycorisAdapter: wrapper around pip `lycoris` LycorisNetwork (+ T-LoRA mask)
  ortho_adapter.py     399  OrthoLoRA (PSOFT-style) adapter, MIT-derived from sorryhyun/anima_lora
  optimizer_utils.py  1878  Automagic/Lion/CAME impls + factory + schedule-free helpers + monitor metrics
  soap_optimizer.py    708  SOAP / SOAPScheduleFree (MIT, Nikhil Vyas)
  caption_utils.py     314  JSON caption -> string builder (trigger word, categories, shuffle, dropout)
  lycoris_patch.py     114  monkeypatch for lycoris 3.4.0 LokrModule.get_weight device bug
```

Important coupling: the runtime **imports the studio package** (`studio.infrastructure.logging`, `studio.schema.TrainingConfig`, `studio.domain.common` capability tables, `studio.infrastructure.log_messages.msg()` i18n keys). `runtime/anima_train.py:33-50` injects repo root + `runtime/` into `sys.path`; `training.*` is imported as a top-level package (not `runtime.training`). The runtime cannot run without `studio/`.

`main()` (`runtime/anima_train.py:117-137`):
```python
args = parse_args()
ctx = TrainingContext(args=args)
phases.bootstrap.run(ctx)  # yaml+CLI -> TrainingConfig, family, seed, dtype, dirs, wandb, loss_fn
phases.models.run(ctx)  # paths, fp8 check, load VAE+TE (+DiT unless deferred), inject adapter, block swap
phases.dataset.run(ctx)  # datasets, latent cache, samplers, dataloader, VAE roundtrip self-test
phases.text_cache.run(ctx)  # Krea2: pre-encode captions to sidecars, release TE
phases.models.finish(ctx)  # Krea2: load DiT now (TE gone) + inject adapter
phases.optimizer.run(ctx)  # param groups, optimizer, total_steps, scheduler, timestep sampler
phases.resume.run(ctx)  # progress UI, --resume-state, signal handlers, step-0 baseline samples
loop.run(ctx)
phases.finalize.run(ctx)  # final save, eval event, block swap release, wandb finish
```

### 1.2 Plugin registry system

Seven registries, all the same shape: a module per variant exposing `build(...)`, a `BUILDERS` dict in `__init__.py`, and a `validate_schema_consistency()` that asserts `set(TrainingConfig.model_fields[<field>].annotation.__args__) == set(BUILDERS)` (run at startup: `runtime/training/phases/bootstrap.py:99-106`). Interfaces are `typing.Protocol` + `@runtime_checkable` (duck typing), not ABCs. Adding a variant = new module + one dict line + one `Literal` value in `studio/domain/training.py`.

| Registry | File | Dispatch key | Interface |
|---|---|---|---|
| model families | `runtime/training/families/__init__.py:27-70` | `args.model_family` | `ModelFamily` Protocol (`families/protocol.py:19-55`) + frozen `ModelSpec` |
| adapters | `runtime/training/adapters/__init__.py:24-40` | `args.lora_type` | `AdapterProtocol` (`adapters/protocol.py:36-88`) |
| optimizers | `runtime/training/optimizers/__init__.py:31-63` | `args.optimizer_type` | `build(args, params, lr, wd)` + optional `validate(args)` |
| lr schedulers | `runtime/training/schedulers/__init__.py:22-45` | `args.lr_scheduler` (`"none"` -> `None`) | `build(args, optimizer, total_steps)` |
| losses | `runtime/training/losses/__init__.py:26-41` | `args.loss_type` | `LossProtocol.compute(pred, target, t)` |
| timestep samplers | `runtime/training/timestep_samplers/__init__.py:25-48` | `krea2_shift` mode / `infonoise_enabled` / baseline | `TimestepSamplerProtocol` |
| inference samplers | `runtime/training/inference_samplers/__init__.py:22-31` | `sample_sampler_name` | `sample(denoise_fn, x, sigmas, **kw)` |

Family registry is only half-pluggable: `get_family()` is an if/elif on `"anima"`/`"krea2"` (`families/__init__.py:44-60`); `SPECS` are validated at import by `validate_spec` (`families/spec.py:121-160`, rejects `temporal=True`, unknown capabilities, cached_varlen+caption_tag_ops).

`ModelSpec` (`runtime/training/families/spec.py:97-110`, with sub-specs at L21-95):
```python
@dataclass(frozen=True)
class ModelSpec:
    family_id: str
    display_name: str
    objective: Literal["rectified_flow"]  # only legal value in v1
    latent: LatentSpec  # fingerprint, channels, spatial_stride, patch_spatial/temporal, temporal, rgb_factors/bias
    text: TextSpec  # strategy: "online" | "cached_varlen"; max_seq_len; fingerprint
    sampling: SamplingDefaults  # samplers, schedulers, default_sampler/scheduler/steps/cfg, shift_policy
    capabilities: frozenset[
        str
    ]  # subset of KNOWN_CAPABILITIES (L114-118): navit sra leap compile_blocks caption_tag_ops online_text text_cache masked_loss block_swap
    lora: LoraOutputSpec  # prefix="lora_unet", preset_name
    config_defaults: Mapping[str, Any]
```

`ModelFamily` Protocol (`runtime/training/families/protocol.py:19-55`):
```python
class ModelFamily(Protocol):
    spec: ModelSpec
    def load_dit(self, path, device, dtype, *, attention_backend="flash_attn", repo_root=None, purpose="train") -> Any
    def load_vae(self, path, device, dtype, *, tiling="auto") -> Any
    def load_text(self, text_encoder_path, device, dtype, *, t5_tokenizer_path="", comfy_qwen=False, t5_fast=False, purpose="train", cache_enabled=True) -> Any
    def prepare_text_cache(self, captions, extra_prompts, *, cache_entries=(), cache_root=None, text=None, device=None, dtype=None) -> None
    def encode_text_for_batch(self, text, dit, captions, device, dtype, *, comfy_encoding=True, kv_trim=True) -> Any   # opaque cond
    def forward_train(self, dit, noisy, t, cond, *, use_checkpoint=False)   # returns v_pred
    def sample_image(self, *args, **kwargs)
    def lora_preset(self) -> dict; def lora_metadata(self) -> dict[str, str]; def convert_lora_state_dict(self, sd) -> dict
```
(Both concrete families also add `swapped_param_ratio(blocks_to_swap, *, checkpoint_path)` used by VRAM budgeting.)

`AdapterProtocol` + `StepContext` (`runtime/training/adapters/protocol.py:22-88`):
```python
@dataclass(frozen=True)
class StepContext: global_step: int; total_steps: Optional[int]; epoch: int; sigma_t: Tensor; args: object

class AdapterProtocol(Protocol):
    def inject(self, model: nn.Module) -> None
    def get_param_groups(self, weight_decay: float) -> list[dict]
    def save(self, path: Path) -> None; def load(self, path: Path) -> None
    # optional hooks (default no-op):
    def on_step_begin(self, ctx: StepContext) -> None          # T-LoRA rank mask
    def regularization_loss(self, ctx: StepContext) -> Optional[Tensor]
    def excludes_weight_decay(self, param_name: str) -> bool
```

`TimestepSamplerProtocol` (`runtime/training/timestep_samplers/protocol.py:19-75`): `sample(bs, device, *, token_counts=None)`, optional `record(t, raw_mse)`, `maybe_refresh(global_step)`, `status()`, `state_dict()`, `load_state_dict()`; a sampler declares `requires_token_counts = True` to receive per-sample token counts (`loop.py:190-198`) and `applies_resolution_shift = True` to be mutually exclusive with `timestep_shift_resolution_aware` (`loop.py:281-284`).

`LossProtocol` (`runtime/training/losses/protocol.py:19-46`): `compute(pred, target, t) -> per-element tensor` (no reduction; reduction + weighting happen in the loop).

### 1.3 Main training loop — `runtime/training/loop.py::run(ctx)` (L256-991)

Per micro-batch (inside `for epoch in range(ctx.start_epoch, args.epochs)` L314 / `for batch_idx, batch in enumerate(ctx.dataloader)` L333):

1. **Data** (L338-356): `captions = batch["captions"]`; latents from (a) NaViT list `batch["navit_latents"]`, (b) cached `batch["latents"]` (npz), or (c) live `ctx.vae.model.encode(pixels.unsqueeze(2), ctx.vae.scale)` under `no_grad`. Latents are always 5-D `[B,C,1,H,W]`. On ARB bucket shape change, `torch.cuda.empty_cache()` (`_BucketSwitchCacheRelease`, L212-253; Windows WDDM fix).
2. **Text cond** (L364-381): `ctx.family.encode_text_for_batch(ctx.text_stack, ctx.model, captions, device, dtype, comfy_encoding=..., kv_trim=...)`; Anima encodes online every step; Krea2 reads sidecar caches. `cond` is opaque to the loop.
3. **Timestep** (L385-402): `t = ctx.timestep_sampler.sample(bs, device[, token_counts])`, then optional SD3 resolution shift `apply_resolution_shift(t, token_counts, base_tokens)`.
4. **Adapter hook** (L406-414): `ctx.injector.on_step_begin(StepContext(global_step, total_steps, epoch, sigma_t=t, args))`.
5. **Noise** (L417-429): `make_noise(latents, noise_offset, pyramid_iters, pyramid_discount)`; leap dice roll with Python `random` (L431-441).
6. **Forward + loss** under `torch.autocast("cuda", dtype=ctx.dtype)` (L452), three exclusive paths:
   - NaViT packed (L453-488, Anima only): `navit_packed_forward_and_loss(...)`.
   - Leap/FlowBP self-distillation (L489-548, Anima only): 4 variants from `families/anima/leap.py`.
   - **Standard rectified flow** (L549-603):
```python
noisy  = (1 - t_exp) * latents + t_exp * noise           # t=1 is pure noise
target = noise - latents                                  # velocity
pred   = ctx.family.forward_train(ctx.model, noisy, t, cross, use_checkpoint=args.grad_checkpoint)
loss_per_sample = ctx.loss_fn.compute(pred.float(), target.float(), t)   # per-element, fp32
with torch.no_grad():                                     # raw MSE for InfoNoise (main-set samples only)
    _raw_mse = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dims...)
ctx.timestep_sampler.record(t.detach()[~is_reg], _raw_mse[~is_reg])
if "loss_weight" in batch: loss_per_sample *= w                            # reg_weight per sample
lw = compute_loss_weight(t, scheme=args.loss_weighting, ...)              # min_snr / detail_inv_t / cosmap
loss = _masked_mean(loss_per_sample, spatial_mask) if spatial_mask is not None else loss_per_sample.mean()
```
   - Then SRA alignment loss (L606-619) and `injector.regularization_loss()` (L623-625).
7. **Backward / accumulation** (L629-651): `_accumulation_step(batch_idx, dl_len, grad_accum)` (L141-160) returns `(group_size, is_group_end)`; the epoch's tail group steps even if smaller than `grad_accum` and is normalized by the *actual* group size. Non-finite loss skips only this micro-batch's backward (L634-645). `loss = loss / group_size; scaler.scale(loss).backward()` or `loss.backward()`.
8. **Optimizer step** at group end (L653-699): skip if no grads; `scaler.unscale_`; any non-finite grad -> skip step + `zero_grad` (L664-681); `clip_grad_norm_(trainable_params, grad_clip_max_norm)`; `scaler.step/update` or `optimizer.step()`; `scheduler.step()` unless `prodigy_plus_schedulefree` (L690-691); `zero_grad()`; `global_step += 1`; `timestep_sampler.maybe_refresh(global_step)`. 50 consecutive skipped steps -> ERROR log (L299-312), no auto-stop.
9. **Logging** (L703-815): `loss_val = loss.item()*group_size`; `get_optimizer_monitor_metrics()` (`utils/optimizer_utils.py:1811-1878`, Prodigy `d*lr`); `train_monitor.update_monitor(...)` -> JSON file; speed EMA; `wandb_monitor.log({...})`; progress line every `log_every` steps to logger `training.progress` (pipe mode), or Rich / `\r` in a TTY.
10. **Periodic IO** (L818-872): `sample_steps` -> `run_sample`; `save_every_steps` -> `injector.save(output_dir/{name}_step{N}.safetensors)` inside `optimizer_eval_mode` (schedule-free averaged weights); `save_state_every_steps` -> `save_training_state(state_dir/training_state_step{N}.pt)`; `max_steps` break.
11. **Epoch end** (L874-991): NaN summary; `train/loss_epoch` to wandb; `save_every_epochs` LoRA; `sample_every`; `save_state_every_epochs`; **forced** `auto_epoch_state.pt` + `auto_epoch_state.config.json` (L954-987) followed by `emit_event("auto_epoch_backup_written", ...)`.

Mixed precision (`phases/bootstrap.py:175-184`): `bf16` -> autocast bf16, no scaler; `fp16` -> `torch.cuda.amp.GradScaler()` + VAE forced fp32 (`ctx.vae_dtype`); `no` -> fp32. Base weights are cast to `ctx.dtype` at load. **LoRA parameters take the model's first non-fp8 float dtype** (`utils/lycoris_adapter.py:236-244`), i.e. bf16 trainable params + bf16 optimizer state when training in bf16 (fp32 only for all-fp8 base). fp8 base: see 1.5.

Gradient checkpointing: Anima = manually unrolled per-block `torch.utils.checkpoint(..., use_reentrant=False)` (`families/anima/forward.py:11-37`: `prepare_embedded_sequence` -> `t_embedder` -> blocks -> `final_layer` -> `unpatchify`); Krea2 = inside `SingleStreamDiT.forward` (`modeling/krea2/krea2_modeling.py:508-522`), per block, only when `self.training and torch.is_grad_enabled()`.

EMA: **none** for weights (only `speed_ema` and InfoNoise's MSE EMA). Multi-GPU: none (single `cuda`; GPU selection via `CUDA_VISIBLE_DEVICES` in `studio/services/runtime/gpu_select.py`).

### 1.4 Model family abstraction: Anima vs Krea 2

| | Anima (`families/anima/`) | Krea 2 (`families/krea2/`) |
|---|---|---|
| DiT | `modeling/anima/anima_modeling.py:216` `Anima(MiniTrainDIT)` = Cosmos-Predict2 DiT + `LLMAdapter` (6 layers, dim 1024, L159-215). Config inferred from checkpoint (`loader.py:163-193`): `x_embedder.proj.1.weight` -> `model_channels` 2048 -> 28 blocks/16 heads (2B) or 5120 -> 36 blocks/40 heads (14B); `max_img_h/w=1024`, `patch_spatial=2`, `rope3d` learnable, `adaln_lora_dim=256`, `crossattn_emb_channels=1024`, `concat_padding_mask=True` | `modeling/krea2/krea2_modeling.py:328` `SingleStreamDiT`, `Krea2Config` (L59-94): features 6144, 28 layers, 48 heads / 12 kv heads (GQA), text dim 2560 x 12 stacked layers, `TextFusionTransformer` (12 layers), patch 2, 16 ch, RoPE theta 1e3; single stream = concat(text, image) tokens (L487), text RoPE pos zeros (L489), output slice (L525). ~12.9B params |
| Text encoder | Qwen3-0.6B via `AutoModelForCausalLM` last hidden state (`text_encoding.py:22-74`) or standalone Comfy port `comfy_qwen.py`; plus T5-XXL **tokenizer only** (ids fed to LLMAdapter; `loader.py:247-265`, downloads `google/t5-v1_1-xxl` if missing). Encoded **online every step** (`family.py:84-142`): `dit.preprocess_text_embeds(qwen_emb, t5_ids, t5xxl_weights)` -> pad to floor 512 -> optional `kv_trim` to 64/128/256/512 | Qwen3-VL-4B-Instruct (`Qwen3VLForConditionalGeneration`) with chat template (`text_encoding.py:41-49`, 34 prefix / 5 suffix tokens), hidden layers `(2,5,8,...,35)` stacked -> `(seq, 12, 2560)` (L38, L563-567), valid tokens gathered incl. suffix after interior padding (L61-78), caption padded to floor 512 (L472-494). Strategy `cached_varlen`: pre-encoded to sidecars, TE released before DiT load (`phases/models.py:236-240`, `:322-338`); HF sharded dir or Comfy single-file fp8_scaled TE (L180-274) |
| VAE / latent | shared: Wan2.1 (`modeling/wan/vae2_1.py`), `load_vae` (`training/vae.py:347-384`, dim 96, z 16, hard-coded mean/std L365-372), `WAN21_F8C16` (`families/latent_spaces.py:43-52`, fingerprint `wan21-f8c16`, stride 8, patch 2 -> align 16 px) | same instance -> latent npz cache is shared across families |
| Objective | rectified flow, `ConstantShift(3.0)` (`anima/__init__.py:48`); training `timestep_sampling=logit_normal`, `timestep_shift=3.0` | rectified flow, `ConstantShift(1.15)` but **meaning `mu` (pre-exp)**, not a factor (`krea2/__init__.py:127-132`); training default `timestep_sampling=krea2_shift` (per-image dynamic shift) |
| `forward_train` | zero `pad_mask [B,1,H,W]`, `t.view(-1,1)` (`family.py:145-159`) | `dit(noisy, t, cond.context, attention_mask=cond.attention_mask, use_checkpoint=...)` (`krea2/__init__.py:216-223`) |
| Sampling | Comfy KSampler parity: samplers `er_sde`, `dpmpp_3m_sde`; schedulers `simple`, `sgm_uniform` (`anima/sampling.py:35-36`); 25 steps / CFG 4.0 | FlowMatchEuler only: `euler` + `simple`, fixed mu 1.15, Raw 28 steps / g=4.5, Turbo 8 / 0.0, guidance `cond + g*(cond-uncond)` (`krea2/sampling.py:32-47`) |
| LoRA targets | `*q_proj,*k_proj,*v_proj,*output_proj,*mlp.layer1,*mlp.layer2`, exclude `llm_adapter*` (`anima/preset.py:18-30`) | `["*"]` = all 264 `nn.Linear` incl. attention gates and text-fusion stack (`krea2/preset.py:20-29`) |
| Attention | flash_attn / xformers / SDPA global switch (`cosmos_predict2_modeling.py:96-125`) | SDPA only (`krea2/__init__.py:148-149`) |
| Capabilities (`studio/domain/common.py:21-27`) | navit, sra, leap, compile_blocks, caption_tag_ops, online_text, masked_loss, block_swap | masked_loss, text_cache, block_swap (no caption shuffle/dropout because cache key = caption hash) |

Family config defaults overlay (`studio/domain/common.py:32-47`): krea2 forces `shuffle_caption=False`, `tag_dropout=0`, `attention_backend="none"`, `timestep_sampling="krea2_shift"`, `sample_sampler_name="euler"`, `sample_infer_steps=28`, `sample_cfg_scale=4.5`.

### 1.5 fp8 base weights (Krea2 only)

- Detection: `checkpoint_contains_fp8()` reads safetensors header (`families/krea2/loader.py:252-267`); `_inspect()` (L122-229) validates key set / shapes against a meta-device `SingleStreamDiT` and collects `{layer}.weight_scale` F32 scalars (fp8_scaled format, Comfy `__metadata__._quantization_metadata`, parsed in `quant_fp8.py:43-72`) or accepts pure-cast fp8 without scales.
- Loading (`load_krea2_model`, L422-525): fp8 tensors are moved to the device **unchanged** (L488-498); everything else is cast to the compute dtype; `load_state_dict(strict=True, assign=True)`; `requires_grad_(False)`.
- Compute: `patch_fp8_linears()` (`quant_fp8.py:91-131`) monkeypatches each fp8 `nn.Linear.forward` with
```python
def _fp8_linear_forward(self, input):
    weight = self.weight.to(input.dtype)  # dequant on the fly, per forward
    if (scale := getattr(self, "weight_scale", None)) is not None:
        weight = weight * scale.to(input.dtype)  # per-tensor scale, cast first (ComfyUI parity)
    return F.linear(input, weight, bias)
```
  i.e. **no fp8 matmul (`_scaled_mm`), no fused dequant kernels** — a transient bf16 copy of each weight per forward (and again in checkpoint recompute). Hence `_validate_fp8_base` (`phases/models.py:243-275`) requires `grad_checkpoint=True` and rejects `lora_dora` (DoRA reads base norms).
- Adapter interaction: LoKr on fp8 base forced into `bypass_mode` (`utils/lycoris_adapter.py:192-198`); adapter dtype = first non-fp8 dtype else fp32 (L236-244).
- Block swap keeps fp8 dtype in pinned memory; scales are pinned to the compute device (`loader.py:521-524`).
- Inference-only extra: LoRA merge back into fp8 with requantization + stochastic rounding, ComfyUI bit-parity (`families/krea2/lora_fp8_merge.py`).
- Anima: no fp8 path at all (weights cast through `_load_weights_best_effort`).

### 1.6 Block swap / layer offloading — `runtime/training/block_swap.py`

- Mechanism: the **last N** blocks of `model.blocks` keep frozen params as CPU-pinned master copies; two GPU "slots" (double buffer) sized for one block (L242-253, L311-315). `_fetch()` copies on a dedicated CUDA stream with `ready`/`done` events (L337-354) and **rebinds `param.data`** to the slot buffer (L356-369) — module identity never changes, so LyCORIS wrappers (which hold `org_module`) keep working and fp8 `weight_scale` buffers stay paired. Trainable (LoRA) params are never swapped (L287-288).
- Wiring: `attach()` registers 4 hooks per swapped block (L444-495): forward pre (fetch + prefetch i+1), forward post (release slot), **full-backward pre (re-fetch + prefetch i-1)**, full-backward post. The backward hooks are mandatory (checkpoint recompute does not fire forward hooks; missing them silently corrupted grads ~300x noise floor per the comments; test `Test/tests/test_block_swap_grad_fidelity.py`).
- `PinnedPacker` (L84-184): pre-allocates pinned memory as power-of-two chunks (64 MB granularity, 256 B alignment) so the host caching allocator doesn't round each tensor up to 2^n (1.47x overhead observed).
- Loaders place swapped blocks directly on CPU (never touching VRAM): Anima `place_model_for_block_swap` (`families/anima/loader.py:58-102`), Krea2 via `PinnedPacker` during load (`krea2/loader.py:464-505`). Budget guards: `check_pinned_budget` (80% of available RAM minus ~4 GB base, `sysmem.py:176-207`); VRAM guard discount by parameter ratio (`phases/models.py:64-83`).
- Setup after adapter injection (`phases/models.py:158-181`); teardown `close()` + `release_pinned_host_cache()` (`torch._C._host_emptyCache`) in finalize (`phases/finalize.py:20-53`). `restore_masters()` (L406-426) and `move_module_excluding()` (L569-597) serve the inference daemon. Config field `blocks_to_swap` (`studio/domain/training.py:168`), CUDA only, trailing blocks only, cost ~8 ms/block per docs.

### 1.7 Text-encoder cache and latent cache

**Text cache** (`runtime/training/text_cache.py`): per-image sidecar `<image filename>.text.safetensors` (L28, L72-76) with metadata `{cache_kind, format_version=1, text_fingerprint, caption_sha256, cache_key}` where `cache_key = sha256("text-cache-v1\0<TE fingerprint>\0<caption>")` (L58-69); any mismatch -> miss (L172-190). Sample/negative prompts go to a bundle `<task root>/.text-cache/prompts.<fp12>.safetensors` (L79-91). Atomic `tmp + os.replace` writes (L110-125). Tensors are variable-length (not padded). Krea2 fingerprint `qwen3-vl-4b-instruct-krea2-12x2560-v1` (+`-tefp8` for the fp8 TE, `text_encoding.py:34, 358-359`). Phase (`phases/text_cache.py`): dedupes by image (raises if one image resolves to two captions, L38-58), adds sample prompts + negative (L61-77), calls `family.prepare_text_cache`; Krea2 encodes misses in batches of `cache_batch_size=1` then `release_model()` (`text_encoding.py:679-708`). Misses during training are repaired by **reloading the 4B TE and releasing it again** (L732-792). Anima: strategy `online`, no cache.

**Latent cache** (`runtime/training/dataset.py::CachedLatentDataset` L966-1436): npz beside each image, `img.npz` or `img.r{reso}.npz` when the same image fans out to multiple resolutions (L1064-1073). Keys: `latent` (fp32 `[C,1,h,w]`), `latent_flipped` (if `flip_augment`), `mask`/`mask_flipped`, `bucket_w/h`, `latent_fingerprint`, `layout_version` (L1304-1311). Invalidation `_is_cache_valid` (L1075-1155): missing file, `npz mtime < image mtime`, missing `latent`, fingerprint/layout mismatch (legacy files without fingerprint grandfathered as `wan21-f8c16`, L43), missing `latent_flipped` when flip is on, bucket size mismatch, mask presence/mtime mismatch. **No content hash** (mtime only). Encoding batched by bucket size (`_encode_and_save`, L1218-1378), tiled encode for images > 4 MP when `cache_encode_tiled`. `__getitem__` re-opens the npz per sample (no RAM cache), picks flipped 50%, and computes the caption **at load time** (shuffle/dropout re-rolled every epoch, L1412-1414).

### 1.8 Dataset pipeline

- `BucketManager` (`dataset.py:163-245`): buckets = all `(w,h)` in `[min,max]` step 64 with area within ±10% of `base²` and AR ≤ `aspect_ratio_limit` (2.0); bounds derived from `base/√R .. base·√R`; 37 buckets at 1024. `get_bucket` picks min `(|AR diff|, |area diff|)`. Must stay in sync with TS port `studio/web/src/lib/trainBuckets.ts`.
- `ImageDataset` (L248-809): extensions L258; folder naming `[Npx_][R_]label` (L415-443: per-folder resolution override snapped to /64 in [256,4096], Kohya repeats); `_scan` (L482-547) root images + recursive subfolders, each unique image fanned out to `repeat × len(resolutions)` samples carrying `target_reso`; captions: JSON preferred (`utils/caption_utils.build_caption_from_json` L55-156: `meta.trigger` always first and protected, fixed quality/count/character/series/artist, shuffled+dropout appearance/tags/environment with ≥1 kept, dedupe, optional NL suffix) else `.txt`/`.caption` (`_process_caption_txt` L595-616: comma split, `keep_tokens` prefix protected, shuffle, independent per-tag dropout); `caption_override` for reg sets; JSON preflight fail-fast (L549-593); `get_with_flip` (L750-809): resize-cover LANCZOS + center crop to bucket, optional H-flip, mask sidecar `{stem}.mask` NEAREST-resized then BOX-downsampled /8; pixels in [-1,1]. NaViT native sizes via `plan_native_fit_image` (L67-160).
- Regularization: `MergedDataset` (L825-876) concatenates main + reg; reg items get `loss_weight=reg_weight`, `is_reg=True`; built with `tag_dropout=0` and optional `reg_caption` override (`phases/dataset.py:146-188`); latent-cached separately (L206-215).
- Batching: `BucketBatchSampler` (L879-956) groups indices by `bucket_for_index`, seeded shuffle `seed+epoch`, `drop_last=False` (short tail batches), `__len__` = Σ per-bucket ceil; `NavitPackBatchSampler` (L1628-1718) packs by token budget (`next_fit` or windowed FFD, L1488-1555). Collates L1452-1481 / L1721-1743 emit `latents|pixel_values|navit_latents`, `captions`, `masks`, `loss_weight`, `is_reg`. Windows forces `num_workers=0` (`phases/dataset.py:222-227`). VAE encode->decode self-test image (L279-297).
- Not present: caption dropout to empty/unconditional (no `caption_dropout_rate`), no dataset config file (TOML/JSON subsets), no per-image resolution lists beyond folder prefix, no random crop.

### 1.9 Adapter implementations

- Registry (`adapters/__init__.py:24-30`): `lora|lokr|loha -> lycoris.build`, `ortho -> ortho.build`, `tlora -> tlora.build` (Ortho backend when `tlora_use_ortho=True` [default], else LyCORIS locon + rank mask; `adapters/tlora.py`).
- **LyCORIS**: depends on the pip package `lycoris-lora>=3.4.0,<4.0` (not vendored). `utils/lycoris_adapter.py::LycorisAdapter` (L94-549): `LycorisNetwork.apply_preset(preset)` (L160), algo map `lora/tlora -> "locon"` (L163-165), extras `factor` (LoKr), `weight_decompose` (DoRA), `rs_lora`, `bypass_mode=True` for plain LoRA (2x faster than ΔW rebuild, L182-183) and for LoKr on fp8 (L196-198); constructs `LycorisNetwork(model, multiplier=1.0, lora_dim=rank, alpha, dropout, rank_dropout, module_dropout, network_module=..., **extra).apply_to()` (L201-212) while filtering lycoris' stdout spam (L49-91); `lora_reg_dims` per-layer rank override by regex with re-init (L578-672); T-LoRA installs a mask by monkeypatching each module's `make_weight` (L284-332) with `r = (1-t̄)^α·(rank-min_rank)+min_rank` from the **batch-mean** t (L334-354); syncs device/dtype (L236-244); hijacks `model.train()` so the external `LycorisNetwork` follows eval/train (L252-262); `detach()` for daemon hot-swap (L361-414). Param groups: LoKr `lokr_w1*` excluded from weight decay (L423-448). Save (L466-499): `network.state_dict()` -> keys `lora_unet_<module.path.with_underscores>.{lora_down.weight,lora_up.weight,alpha | lokr_w1,lokr_w2_a,...}` (ComfyUI-compatible), per-layer `.alpha` rewritten to `scale*lora_dim` (L552-575), metadata `ss_network_dim`, `ss_network_alpha`, `ss_network_module="lycoris.kohya"`, `ss_network_args` JSON `{algo, factor, dropout, rank_dropout, module_dropout, weight_decompose, rs_lora, lora_reg_dims?, model_family, preset}`. Load `strict=False` (L501-526). `utils/lycoris_patch.py` patches the 3.4.0 `LokrModule.get_weight` CPU-mask bug.
- **OrthoLoRA** (`utils/ortho_adapter.py`, MIT-derived from sorryhyun/anima_lora): `OrthoLoRALinear` (L64-208) freezes SVD bases `P,Q` (`svd_lowrank`, fallback full SVD), trains skew matrices `S_p,S_q` (Cayley rotation via `linalg.solve`) and diagonal `lambda`; forward L152-178; saves a **distilled plain LoRA** (`lora_down/lora_up/alpha`, L180-196) so inference is standard; `OrthoLoRAAdapter` (L210-399) replaces `nn.Linear` matched by the family preset via fnmatch (L241-267), metadata `algo="lora", source_algo="ortho"|"tlora_ortho"` (L374-383); resume from a distilled file is only approximate (L198-207, L357-364).
- Rank/alpha: `lora_rank` (schema L279), `lora_alpha` default 32.0 (L284), `lokr_factor` 8, dropout trio, DoRA/rs-LoRA flags. `injector.metadata_extra = family.lora_metadata()` (`phases/models.py:194`); `resume_lora` rejected across families (L197-205, `_read_lora_family` L341-353). `regularization_loss` is `None` for every shipped adapter.

### 1.10 Optimizers, schedulers, losses, timestep samplers

**Optimizers** (`runtime/training/optimizers/__init__.py:31-46`; implementations in `utils/optimizer_utils.py` unless noted):
`adamw` (torch `AdamW`, `create_standard_adamw` L360-406) · `automagic` v1/v2 (`Automagic` L500 / `Automagic2` L788, Ostris-derived; v2 = fused backward, validator forbids `grad_accum>1`, fp16, warns grad_clip; requires `lr_scheduler=none`) · `came` (L1137, Luo 2023) · `lion` (L1047) · `prodigy` (pip `prodigyopt`, L1363-1434) · `prodigy_plus_schedulefree` (pip `prodigy-plus-schedule-free>=2.0`, L1435-1545; validator forces `lr_scheduler=none`) · `soap` (`utils/soap_optimizer.py:47`) · `soap_sf` (`SOAPScheduleFree` L441; `lr_scheduler=none`). `adamw8bit` (bitsandbytes) exists in `_dispatch_optimizer` (L191) but is not registered/schema-exposed (dead). Schedule-free support: `optimizer_eval_mode()` (L1667-1694) around every save/sample; `optimizer.train()` at start (`phases/resume.py:110-111`) and after state load (`state.py:232-239`).

**LR schedulers** (`schedulers/__init__.py:22-27`): `none`, `cosine` (`CosineAnnealingLR`, T_max=total_steps, `lr_scheduler_eta_min`), `cosine_with_restart` (`T_0=lr_scheduler_t0`=500, `T_mult`), `cosine_with_warmup` (`LambdaLR`, linear warmup `lr_scheduler_warmup_steps`=100). No linear/constant/polynomial/REX.

**Losses** (`losses/__init__.py:26-29`): `mse`, `huber` (constant `huber_c`=0.15). Weighting (`loss_weighting.py:19-64`): `none`, `min_snr` (SNR=((1-t)/t)², w=min(γ/SNR,1), γ=5), `detail_inv_t` (1/t clamped [1,5]), `cosmap`; `weight_cap_ratio`. Extras: masked loss (`loop.py:120-138`), reg `loss_weight`, SRA v2 alignment (Anima, `families/anima/sra_align.py`, block 4 hidden -> 5-layer MLP -> clean latent, weight 0.2 with none/linear/cosine/jump decay `loop.py:64-99`), Leap/FlowBP self-distillation (`families/anima/leap.py`, `leap_ratio` 0.6 per micro-batch). Noise: `noise_offset`, pyramid noise (`noise.py:42-100`, gated by `noise_enhancement_type`). Absent: debiased estimation, timestep-dependent Huber, LPIPS, eps/x0 prediction variants, weight EMA.

**Timestep samplers** (`timestep_samplers/__init__.py:25-48`): `baseline` wraps `sample_t` (`timestep_sampling.py:19-73`) modes `logit_normal` (sigmoid(randn) then Möbius shift `s=timestep_shift`=3.0), `uniform`, `logit_normal_low` (1/s), `mode` (SD3), `mixed_uniform_low`, `mixed_uniform_logit` (`timestep_mix_low_prob`), plus post-hoc `timestep_schedule_shift` Möbius (L76-86); `infonoise` (`infonoise.py:46+`, arXiv 2602.18647: K=64 log-σ bins, FIFO B=256, EMA β=0.9, warmup 20% of total steps, refresh every M=100 steps, gate pivot c=0.15, records main-set raw MSE only, serializable for resume); `krea2_shift` (`krea2_shift.py:63-105`: logit-normal then per-image `mu = lerp(0.5..1.15 over seq_len 256..6400)`, `t' = t·e^mu/(1+(e^mu-1)t)`, `sigmoid_scale` hard-coded 1.0). Optional SD3 resolution shift `apply_resolution_shift` (`timestep_sampling.py:115-127`, `s=sqrt(n_i/n_base)`) via `timestep_shift_resolution_aware`, exclusive with `krea2_shift`.

### 1.11 Checkpoint save / resume / pause-at-epoch-end

- `save_training_state` (`runtime/training/state.py:31-114`) -> `torch.save` (tmp + `os.replace`) of `{lora_state_dict, optimizer_state_dict, epoch, global_step, loss_history, rng_state{torch,cuda,random}, monitor_state, model_family, scheduler_state_dict?, sra_aligner_state?, timestep_sampler_state?, scaler_state?}`. `load_training_state` (L117-247): cross-family fail-fast, LoRA `strict=False`, SRA, optimizer, scheduler, GradScaler, RNG, sampler state, then `optimizer.train()` for schedule-free.
- Locations: user periodic saves -> `<output_dir>/state/task_<LORA_TASK_ID|unknown>/training_state_{step|epoch}{N}.pt` (`context.py:122-133`); LoRA files -> `<output_dir>/<output_name>_step{N}|_epoch{N}|.safetensors`; system auto-backup -> `studio_data/tasks/<id>/state/auto_epoch_state.pt` + `.config.json` (`context.py:135-149`, `snapshot.py:50-62`), written **unconditionally every epoch end** (`loop.py:954-987`). `write_config_snapshot` freezes `vars(args)` + `sample_prompts` (`snapshot.py:88-126`).
- Pause: supervisor sends SIGINT (POSIX) / CTRL_BREAK -> SIGBREAK (Windows); handler `TrainingContext.handle_interrupt` (`context.py:180-232`) finishes wandb, prints `__EVENT__:pause_state:{"state_path","config_path","step"}` and `sys.exit(0)`. **No mid-epoch save**: resume uses the last epoch backup; a pause before the first epoch ends is treated as cancel (design: ADR 0006 Addendum 1).
- Resume: `--resume-state <pt>`; if a sibling `.config.json` exists, `_maybe_apply_pause_snapshot` (`phases/bootstrap.py:26-67`) overrides all args except `resume_state`/`config`; `phases/resume.py:59-93` loads state, restores monitor history, emits `resume_state_loaded`. `for epoch in range(start_epoch, epochs)` — **dataloader position is not saved**, so step-level saves replay the partial epoch (acknowledged in `context.py:195-200`). `resume_lora` = warm-start weights only.
- Event protocol to the supervisor: stdout lines `__EVENT__:<type>:<json>` (`snapshot.py:24, 129-137`): `train_loop_started`, `auto_epoch_backup_written`, `pause_state`, `resume_state_loaded`, `eval_training_finished`.

### 1.12 Sample generation during training

`runtime/training/sample_runner.py::run_sample` (L27-153): resolves size (aligned to 16 px), steps/cfg/sampler/scheduler from args or `family.spec.sampling` defaults; clears the T-LoRA mask; `optimizer_eval_mode` (schedule-free averaged weights) + `model.eval()`; seeds `sample_seed+offset` (`sample_seed=0` is replaced by a random seed once at start, `phases/bootstrap.py:70-84`); `ctx.family.sample_image(model, vae, text_stack, prompt, ...)`; PNG + wandb + monitor; `empty_cache` before/after with VRAM watermark logging. Baseline samples at step 0 for up to 3 prompts (`phases/resume.py:120-135`); then every `sample_steps` steps / `sample_every` epochs, rotating `sample_prompts`.

Anima `sample_image` (`families/anima/sampling.py:263-537`) is a ComfyUI KSampler re-implementation: raw prompt -> Qwen, SDTokenizer-style weighted T5 ids, CFG-batched forward (L400-438), sigma schedules `simple`/`sgm_uniform` with `time_snr_shift(3.0)` (L135-183), CPU-seeded initial noise (L186-213), `x0 = x - σ·v` (L455), sampler via `inference_samplers` registry (`er_sde.py` ER-SDE-Solver-3; `dpmpp_3m_sde.py` with torchsde Brownian tree), optional DiT/Qwen offload for fp32 VAE decode (L500-534), xformers-NaN retry. Krea2 `sample_image` (`families/krea2/sampling.py`) is FlowMatchEuler with fixed mu; `Krea2Family.sample_image` adds TE/DiT VRAM orchestration (`krea2/__init__.py:225-305`). **Shared with the studio daemon**: `runtime/anima_daemon.py:46` does `import anima_train as _T` and calls the same `get_family()/load_dit/load_text/sample_image`, `PinnedBlockSwap`, `quant_fp8`, plus `studio/services/inference/core.apply_loras`; training previews use HF Qwen (Comfy-*style*), the daemon can use `comfy_qwen` + xformers for bit parity (`sampling.py:276-281`).

### 1.13 CLI entry point and config schema

- Invocation (built by `studio/supervisor/cmd_builder.py:44-73`): `python runtime/anima_train.py --config <yaml> [--monitor-state-file studio_data/tasks/<id>/monitor/state.json] [--resume-state <pt>]`; CLI-only flags `--interactive --auto-install --no-live-curve` (`cli.py:36-50`). Env contract: `LORA_TASK_ID`, `LORA_RAM_GUARD`, `WANDB_*` (`observability.py:319+`), `ANIMA_LOG_LEVEL`, process/trace ids, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (Linux only, `anima_train.py:27-28`).
- Schema = pydantic v2 `TrainingConfig` in `studio/domain/training.py:34-1319` (re-exported by `studio/schema.py`), **164 fields**, `extra="ignore"`, groups: model(5) dataset(9) caption(6) system(~20) lora(13) training(~48) noise_augmentation(4) timestep_sampling(~13) loss(~22) output(9) sample(11) eval_validation(4) monitor(3). Each field carries UI metadata via `_meta(group, control, show_when=..., disable_when=..., advanced=..., cli_alias=...)` (`studio/domain/common.py:11-13`) — the training schema doubles as the web form definition. Argparse is generated from the model (`studio/infrastructure/argparse_bridge.py:199-219`) with suppressed defaults; `namespace_from_config` (L227-260) merges YAML < explicit CLI, constructs `TrainingConfig(**merged)` (all migrations/validators fire), and dumps back into an `argparse.Namespace` consumed everywhere as `args`.
- Key validators: `resolution` -> `list[int]` snapped to /64 in [256,4096] (L957-978); family capability violations (L980-990); family defaults overlay (L992-1002); legacy key migrations (L1004-1012); sampler/scheduler coercion per family (L1014-1053); declarative pin/disable rules from field metadata (`config_rules.py`, L1055-1098); range checks (L1100-1144).
- Representative fields:
```python
model_family: Literal["anima", "krea2"] = "anima"
lora_type: Literal["lora", "lokr", "loha", "ortho", "tlora"] = "lora"
lora_rank: int
lora_alpha: float = 32.0
optimizer_type: Literal[
    "adamw", "automagic", "came", "lion", "prodigy", "prodigy_plus_schedulefree", "soap", "soap_sf"
] = "adamw"
lr_scheduler: Literal["none", "cosine", "cosine_with_restart", "cosine_with_warmup"]
mixed_precision: Literal["bf16", "fp16", "no"]
grad_checkpoint: bool
grad_accum: int = 4
batch_size: int = 1
timestep_sampling: Literal[
    "logit_normal",
    "uniform",
    "logit_normal_low",
    "mode",
    "mixed_uniform_low",
    "mixed_uniform_logit",
    "krea2_shift",
]
loss_type: Literal["mse", "huber"]
loss_weighting: Literal["none", "min_snr", "detail_inv_t", "cosmap"]
cache_latents: bool = True
text_encoder_cache: bool = True
blocks_to_swap: int = 0
navit_packing: bool
sample_sampler_name: Literal["er_sde", "dpmpp_3m_sde", "euler"]
sample_scheduler: Literal["simple", "sgm_uniform"]
```

### 1.14 Runtime-specific weaknesses observed while reading (beyond the ADR/CHANGELOG survey)

1. `runtime` -> `studio` import dependency (logging, schema, i18n `msg()` keys, capability tables) — the trainer is not standalone; `training.*` is a top-level import name that only works after `sys.path` surgery, and `dataset.py:318-329` loads `caption_utils.py` via `importlib.util.spec_from_file_location`.
2. `loop.py::run` is ~740 lines of nested control flow with Anima-only branches (leap, NaViT, SRA, `pad_mask`) inside the "family-agnostic" loop; `_ANIMA_SPEC` is used as the default latent spec in `dataset.py:38`, `timestep_sampling.py:17`, `phases/models.py:229`.
3. Precision: LoRA params/optimizer state inherit bf16 from the base model (no fp32 master weights); `torch.cuda.amp.GradScaler` deprecated API; loss upcast is the only fp32 stage.
4. Per-step CPU syncs: `torch.isfinite(p.grad).all()` per parameter every optimizer step (`loop.py:664-667`), `loss.item()`; InfoNoise samples on CPU/numpy.
5. Resume granularity: only epoch boundaries are truly consistent; step-level `.pt` replays the partial epoch; pause loses everything since the last epoch end.
6. Cache invalidation is mtime-based (no content hash); npz reopened per sample; Krea2 text miss repair reloads the 4B TE mid-training.
7. fp8 = dequant-per-forward monkeypatch (memory/time cost, needs checkpointing, no DoRA); Anima has no fp8/quantized path; no int8/NF4 anywhere.
8. Hard-coded constants: `max_img_h/w=1024` and RoPE extrapolation ratios in `anima/loader.py:179-193`; Anima block/heads table (2048->28/16, 5120->36/40); pad floor 512; `krea2_shift` `sigmoid_scale=1.0`; VAE mean/std in `vae.py:365-372`; single VAE/latent space (`temporal=False` enforced, no video).
9. Missing features: weight EMA, multi-GPU/DDP, full fine-tune or TE training, unconditional caption dropout, LoRA+ (separate A/B lr), DoRA on fp8, more schedulers (linear/constant/REX), only 2 samplers per family, `adamw8bit` dead path, `regularization_loss` hook unused by any adapter, T-LoRA mask uses batch-mean t.
10. Process-global mutable state: attention backend flags in `cosmos_predict2_modeling.py:96-125`, `model.train` monkeypatch, `sys.stdout` filter during LyCORIS injection, `train_monitor.MONITOR_STATE` global.
11. `lycoris-lora` pinned `<4.0` because the adapter relies on lycoris private internals (`make_weight` monkeypatch, `lokr_w2_a/b` attribute names, `restore` API probing at `lycoris_adapter.py:374-397`).

---

## 5. `requirements.txt` — key dependencies and pins

| Package | Pin | Role |
|---|---|---|
| `torch` / `torchvision` | `>=2.0.0` / `>=0.15.0` (no upper bound; torch index chosen by `tools/select_torch_index.py`, studio installer) | core |
| `transformers` | `>=4.57.0` (needs `Qwen3VLForConditionalGeneration` for Krea2 TE) | Qwen3-0.6B / Qwen3-VL-4B, T5 tokenizer |
| `diffusers` | `>=0.32.0` | referenced for parity checks; not used in the training loop |
| `accelerate` | `>=0.24.0` | `init_empty_weights` for single-file Qwen3-VL load |
| `peft` | `>=0.8.0` | listed; LoRA is done via LyCORIS/Ortho, not PEFT |
| `lycoris-lora` | `>=3.4.0,<4.0` | LoRA/LoKr/LoHa backend (v4 breaks private helpers; `torch.compile` issue) |
| `prodigyopt` | `>=1.0` | Prodigy |
| `prodigy-plus-schedule-free` | `>=2.0.0` | PPSF (state_dict incompatible with 1.9.x) |
| `bitsandbytes` | commented out (optional; Windows) | `adamw8bit` (unreachable from schema) |
| `optimum-quanto` | commented out (optional) | Automagic quantized path (`QBytesTensor`) |
| `safetensors` | `>=0.4.0` | weights, LoRA output, text cache |
| `torchsde` | `>=0.2.6` | Brownian tree noise for `dpmpp_3m_sde` |
| `xformers` / `flash-attn` | commented (optional; e.g. `xformers==0.0.30` for torch 2.7/cu128) | attention backends (Anima) |
| `sentencepiece`, `protobuf` | `>=0.1.99`, `>=3.20.0` | T5 tokenizer |
| `einops` | `>=0.7.0` | models |
| `wandb` | `>=0.16.0` | observability |
| `Pillow` `>=10`, `pillow-jxl-plugin` `>=1.3.0`, `numpy` `>=1.24`, `ImageHash` `>=4.3` | image IO / dedup |
| `lpips` `>=0.1.4`, `pytorch-fid` `>=0.3.0`, `pytorch-msssim` `>=0.2.1`, `scipy`, `scikit-image`, `scikit-learn` `>=1.3`, `pandas`, `matplotlib` | studio eval/preprocess side |
| `spandrel` `>=0.4.0` | upscaler loader (preprocess) |
| `fastapi` `>=0.104`, `uvicorn` `>=0.24`, `pydantic` `>=2.5.0`, `python-multipart`, `pyyaml` `>=6.0.1`, `psutil` `>=5.9`, `rich` `>=13.7`, `nvidia-ml-py` `>=12.0`, `requests[socks]` `>=2.31`, `httpx[socks]` `>=0.27`, `huggingface_hub` `>=0.20`, `modelscope` `>=1.20`, `concurrent-log-handler` `>=0.9.25`, `omegaconf` `>=2.3`, `tqdm` | studio backend / downloads / logging |
| `onnxruntime[-gpu]` | not listed (installed from Settings) | WD14/CLTagger |
| Python | 3.10–3.13 (`studio.sh:137-146`) | |

Notable: no lock file, almost all floors with no ceilings except `lycoris-lora<4.0`; `pydantic>=2.5` is load-bearing for the config pipeline; `transformers>=4.57` is the Krea2 gate.
