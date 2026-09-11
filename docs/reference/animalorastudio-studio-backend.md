I now have everything needed. Here is the report.

---

# AnimaLoraStudio — `studio/` Backend Architecture Report

Version under review: `studio/__init__.py:41` → `__version__ = "0.26.2"`. All line numbers refer to files under `/Volumes/Service/Dev/YPuddinTrainStudio/AnimaLoraStudio/`.

## A. Framework & layout

**Framework**: FastAPI (`fastapi>=0.104`, `uvicorn>=0.24`, `pydantic>=2.5`, SQLite via stdlib `sqlite3`, PyYAML). ~53k lines of Python across `studio/`, 208 route decorators in 27 routers.

**Layering** (documented in `studio/__init__.py:5-22`, ADR-0008): four strict top-down layers plus two cross-cutting packages:
```
api/             FastAPI app + 27 routers + pydantic request schemas + deps
  ↓
services/        11 sub-packages: tagging / booru / reg / inference / models /
                 preprocess / projects / dataset / presets / runtime / data_io
  ↓
domain/          pure pydantic models (TrainingConfig, GenerateConfig, RegAiConfig, LoraEntry, XY)
  ↓
infrastructure/  paths / db / event_bus / secrets / logging / argparse_bridge / migrations
supervisor/  (scheduler daemon thread)   workers/  (5 subprocess entry points)  — cross-layer
```
`studio/services/__init__.py:3-4` states the contract: services are "pure functions + dataclasses", *do not depend on db; db operations happen at the call site*. In practice this is violated in several places (e.g. `services/projects/jobs.py`, `services/eval_session.py`, `services/version_config.py` all import `db`).

**App construction**:
- `studio/api/app.py:57-121` — `app = FastAPI(title="AnimaStudio", version=__version__, lifespan=lifespan)`; middleware: `_SelectiveGZipMiddleware(minimum_size=1000)` (skips `/api/events` SSE and `/samples/*`, `studio/api/middleware.py:17-26`) and a pure-ASGI `TraceIdMiddleware` (`studio/api/trace_middleware.py:29-77`: reads/mints `X-Trace-Id`, binds a ContextVar, writes it back on the response, stashes it in `scope["state"]` for the 500 fallback). Router registration order matters in one spot: `queue/io` before `queue/lifecycle` so `/api/queue/{task_id}/snapshot/config` isn't shadowed (`app.py:101-106`).
- `studio/api/lifespan.py:101-260` — startup: `setup_logging("webui")` → Windows Proactor `ConnectionResetError` filter → **single-instance lock** on `studio_data/.server.lock` (fcntl/msvcrt, `infrastructure/single_instance.py`) → `ensure_dirs()` + `db.init_db()` → `cleanup_stale_generate_tempdirs()` → `disk_cache.init(...)` (encrypted session cache) → 3 background threads (TAEFlux download, tag-dictionary download, legacy eval cleanup) → `bus.attach_loop(loop)` → `Supervisor(on_event=bus.publish).start()` stored on `app.state.supervisor` → `SystemStatsSampler` (2.5 s CPU/RAM/GPU via psutil/pynvml → SSE `system_stats_updated`). Shutdown reverses it and `generate_cache.clear_all()`.
- `studio/api/deps.py:18-32` — `_supervisor()` pulls the Supervisor from `app.state` (late import to avoid an import cycle); raises `DomainError(code="system.starting", 503)` before startup completes.
- `studio/api/exception_handlers.py:150-160` — 4 handlers. Error envelope (`_error_envelope`, lines 58-71):
  ```python
  {"error": {"code": "task.not_found", "message": "...", "trace_id": "...", "details": {...}?}}
  ```
  `DomainError` (`studio/domain/errors.py:45`, class attrs `http_status=400`, `default_code="domain.error"`, subclasses `NotFoundError/ConflictError/ValidationError/ForbiddenError/PresetNotFoundError`) → envelope; bare `HTTPException` → backstop envelope with `code="http.<status>"`; `RequestValidationError` keeps starlette's `{"detail": [...]}` (the one exception); uncaught `Exception` → sanitized 500 (traceback only to `studio.log`).
- `studio/api/responses.py` — `EMPTY_STATE` (line 15) for `/api/state`; `packaged_zip_response` (explicitly `Accept-Ranges: none` because zips are regenerated per request, line 28-72); `_thumb_response` (weak ETag + no-cache).
- `studio/api/main.py:11-49` — `main()` applies `gpu_select.apply_gpu_selection_env()` before importing uvicorn, then `uvicorn.run("studio.server:app", host=127.0.0.1, port=8765, timeout_graceful_shutdown=3)`.
- `studio/server.py` — a 55-line shim: re-exports `app`, `main`, `HTTPException`, six path constants (for old test fixtures), and mounts the SPA `SPAStaticFiles(WEB_DIST)` at `/` only if `dist/` exists (line 46-51).
- **Shims**: `studio/db.py`, `studio/paths.py`, `studio/secrets.py` are `sys.modules[__name__] = _real` aliases to `infrastructure/*`; `studio/schema.py` re-exports `domain/*`; `services/model_downloader.py` and `services/updater.py` likewise.

**CLI** (`studio/cli.py`, 886 lines): `python -m studio [run|dev|build|test]` (default `run`, `main():864-882`).
- `run` (`cmd_run:665-750`): loop → `_ensure_python_deps()` (pip install if `fastapi` missing, Tencent mirror fallback) → `_apply_update_pending()` (git pull from `.update_pending`) → rebuild frontend if `dist/` missing or stale (`_web_dist_is_stale():529-574`: git-HEAD marker `dist/.built-from` OR src mtime) → `_apply_pending_install()` (deferred pip installs, e.g. torch) → `_apply_gpu_selection()` → `_check_torch_cuda()` → `_try_enable_flash_attn()` → `_check_onnxruntime()` → `subprocess.call([python, "-m", "studio.server", ...])`. After exit: if `tmp/restart` exists, re-loop; if `cli.py`/`studio.sh`/`studio.bat` sha256 changed, exit code **42** so the shell wrapper re-execs itself.
- `dev`: `ProcGroup` spawns `npm run dev --port 5173` + `uvicorn --reload` (`cmd_dev:752-798`). `build`: npm install/build. `test`: pytest then vitest.
- `--torch cu128|cu126|cu124|cu118|cpu` forces a torch reinstall (`_maybe_force_torch:635-662`).

**Launchers**: `studio.sh` / `studio.bat` — pure-ASCII, set `PYTHONUTF8=1`; parse `--mirror`, `--reinstall` (rm venv with confirmation), `--torch=<tag>`; find Python ≥3.10 (`python3.13…3.10`, or `py -3` on Windows); create `venv/`; **install CUDA torch first** via `tools/select_torch_index.py` so `requirements.txt`'s bare `torch>=2.0.0` doesn't pull the CPU wheel; `pip install -r requirements.txt` (Tencent PyPI mirror fallback); stale-dep sync via `tools/check_requirements_changed.py` + marker `venv/.studio-requirements.sha256`; then the restart loop (`while true; python -m studio; [ -f tmp/restart ] || break; exit 42 → exec "$0"`).

## B. Job queue / supervisor

### Data model (SQLite, `studio_data/studio.db`, WAL, `PRAGMA user_version` = 21)

Base schema `studio/infrastructure/db.py:16-34`; migrations v2–v20 in `studio/infrastructure/migrations/` (`__init__.py:39-60` ordered list). **`tasks` is the unified ledger for everything** (R-2/R-3 "台账合并"): columns after all migrations:

```
tasks: id, name, config_name, status, priority, created_at, started_at, finished_at,
       pid, exit_code, output_dir, error_msg,
       project_id, version_id (v2), monitor_state_path (v3), config_path (v4),
       task_type DEFAULT 'train' (v5),
       paused_state_path, paused_config_path, paused_step, paused_at (v6),
       request_trace_id (v10), last_state_path, last_config_path,
       last_state_epoch, last_state_step (v13), generate_params, generate_cover (v14),
       scheduled_at (v15), params (v17, JSON for data-job kinds), generate_images (v20)
projects: id, slug UNIQUE, title, active_version_id, created_at, updated_at, note, archived_at (v12)
versions: id, project_id FK CASCADE, label, config_name, created_at, output_lora_path, note,
          trigger_word (v7), status, phase, last_failure_reason (v8)   [stage dropped in v9]
project_jobs: frozen legacy table (v18 marks residual pending/running → canceled)
queue_settings: key/value  (only key: "queue.held")
eval_sessions / eval_candidates / eval_metric_results (v19)
```
`db.py:36-66` defines the vocabularies:
```python
VALID_STATUSES = {"pending", "running", "done", "failed", "canceled", "paused", "scheduled"}
TERMINAL_STATUSES = {"done", "failed", "canceled"}
LIVE_STATUSES = ("running", "paused", "pending", "scheduled")
GPU_TASK_TYPES = ("train", "reg_ai", "generate")
JOB_TASK_TYPES = (
    "download",
    "preprocess",
    "tag",
    "reg_build",
    "eval_session",
    "eval_samples",
    "eval_clip",
    "eval_dino",
    "eval_tag",
    "eval_ccip",
)  # last 5 legacy
```
DAO is plain functions (`create_task:119-160`, `update_task:315-323` builds `SET k=?` from kwargs, `list_tasks_page`, `promote_due_scheduled:282-304`, `reorder:332-342`, `get/set_queue_held`). Every call opens a **fresh connection** (`connection_for`, `db.py:80-86`); there is no pool. `services/projects/jobs.py` is a "job" façade over the same table (`create_job` → `db.create_task(task_type=kind, params=...)`, `as_job()` injects `kind` + `log_path`).

### States & transitions

```
scheduled --(tick: scheduled_at<=now, promote_due_scheduled)--> pending
pending   --(dispatch, _spawn_task/_spawn_job/_submit_to_daemon)--> running
running   --rc==0--> done | --rc!=0--> failed | --cancel_pending--> canceled
running   --pause_pending && __EVENT__:pause_state received--> paused
running   --pause_pending but no pause_state (first epoch / killed)--> canceled   (Addendum 1 "方案 Δ")
paused    --POST /resume--> pending (cmd gets --resume-state) | --POST /cancel--> canceled
failed|canceled --POST /resume (if last_state_path exists)--> pending (last_* copied into paused_*)
done|failed|canceled --POST /retry--> new task (copies config_path/project_id/version_id/task_type/params)
running (orphan at startup) --> failed "supervisor restart while task was running"
```
The three-way terminal decision is `_finish_slot` (`supervisor/core.py:1576-1583`). Version status is pushed in lock-step via `finalizer._maybe_finalize_version` (`supervisor/finalizer.py:16-76`: done→completed + `output_lora_path` = `<output_dir>/<output_name>_final.safetensors`; failed→failed + `last_failure_reason`; canceled→canceled; paused leaves version at `training`).

### Scheduling policy (`supervisor/core.py`, `resources.py`, `slot.py`)

- Single daemon thread `studio-supervisor`, `POLL_INTERVAL=1.0s`, `TERMINATE_GRACE=30s` (`core.py:192-193`, `_loop:411-425`). `_tick` (464-487): promote scheduled → reap busy slots (`proc.poll()`) → dispatch to idle slots.
- **Two slots** (`slot.py:15-16`): `SLOT_TRAIN` runs `tasks` of kind `task`; `SLOT_DATA` runs `job`s. `_Slot` dataclass (`slot.py:21-58`) carries `proc, kind, id, job_kind, log_fp, tailer, state_poller, cancel_pending, pause_pending, pause_state_path/config_path/step, train_loop_started, last_auto_epoch_state_path, eval_training_finished_payload`.
- **Resource classes** (`resources.py:20-53`) are static per kind — there is no real VRAM/CPU measurement:
  ```python
  TASK_TYPE_RESOURCE_CLASS = {"train": EXCLUSIVE, "reg_ai": EXCLUSIVE, "generate": EXCLUSIVE}
  JOB_KIND_RESOURCE_CLASS  = {"download": IO, "preprocess": LIGHT, "tag": LIGHT, "reg_build": LIGHT,
                              "eval_session": EXCLUSIVE, ...legacy eval_* }
  # unknown kind → EXCLUSIVE (conservative)
  ```
  Rule: at most **one exclusive** item system-wide (`_exclusive_busy:533-544` = TRAIN slot busy OR daemon has active generate OR DATA slot running an exclusive job). `light` runs alongside exclusive iff `secrets.queue.light_tasks_during_train` (default True); `io` always runs. Global `queue.held` (DB kv) stops all new dispatch.
- `_dispatch_exclusive_tasks` (546-595): single FIFO over `train/reg_ai/generate/eval_session` ordered `priority DESC, created_at ASC` (no cross-type priority, "D-R3 平级"). Routing: `generate` → `_submit_to_daemon` (skips if `config_path` still NULL — enqueue race window); `eval_session` → DATA slot subprocess; `train/reg_ai` → TRAIN slot. Before spawning any exclusive work, `_maybe_yield_daemon` (608-633) asks the resident inference daemon to unload its model ("exclusive lease revocation") and waits a tick.
- `_dispatch_data` (635-667): iterates `project_jobs.list_pending_fifo` and spawns the first `light/io` job that passes admission.

### Launching a training job

`_spawn_task` (`core.py:683-768`): resolves `task.config_path` (falls back to `studio_data/presets/{config_name}.yaml`), freezes a copy to `studio_data/tasks/<id>/snapshot/config.yaml` (`services/task_snapshot.py`), sets `monitor_state_path = tasks/<id>/monitor/state.json`, opens `tasks/<id>/run.log` (`wb`), optionally moves a validation split out of `train/` (`eval_validation.split_for_task`), then `_popen`. Command (`supervisor/cmd_builder.py:39-73`):
```python
cmd = [sys.executable, "runtime/anima_train.py", "--config", <config.yaml>,
       "--monitor-state-file", <tasks/<id>/monitor/state.json>]
if task["paused_state_path"]: cmd += ["--resume-state", <path>]
# reg_ai → runtime/anima_reg_ai.py; generate never goes through here (daemon)
```
Env (`_popen:1462-1548`): `stdout=log_fp, stderr=STDOUT, cwd=REPO_ROOT`, `CREATE_NEW_PROCESS_GROUP` on Windows; injects `PYTHONIOENCODING/PYTHONUTF8/PYTHONUNBUFFERED`, `HF_HUB_DISABLE_PROGRESS_BARS`, `TRANSFORMERS/DIFFUSERS_VERBOSITY=error`, `ANIMA_LOG_LEVEL=DEBUG`, `ANIMA_UI_LANG`, `LORA_RAM_GUARD=0` (if disabled), `WANDB_*` (from `secrets.wandb.active` preset), `LORA_TASK_ID`, `ANIMA_TRACE_ID`, `ANIMA_PROCESS_NAME=worker:train/<id>`. Data jobs: `python -m studio.workers.<kind>_worker --job-id N` (`cmd_builder.py:88-97`), log opened `ab`.

### Pause / resume

Pause is a **signal**: `_send_pause_signal` (`core.py:1733-1753`) sends `SIGINT` (POSIX) or `CTRL_BREAK_EVENT` (Windows, mapped to SIGBREAK in the child); the trainer's `handle_interrupt` saves state and exits. Cancel uses `SIGTERM` on POSIX but **`taskkill /T /F` immediately on Windows** (because CTRL_BREAK is taken by pause, `1710-1730`). `is_task_pausable` (343-363) requires `train_loop_started` and `last_auto_epoch_state_path` (first epoch's auto backup written) — the UI hides the button otherwise. No pause timeout: the UI modal decides after 30 s whether to cancel (`_signal_pause_async:1755-1766`). Resume (`api/routers/queue/lifecycle.py:345-421`): validates `state_path` & `config_path` exist → `status=pending` (+ for failed/canceled copies `last_*` into `paused_*`) → `cmd_builder` adds `--resume-state`; child emits `resume_state_loaded` → `_clear_pause_fields`.

### Logs / progress / metrics flow

Worker → supervisor communication is **stdout text with an inline event protocol**, not JSON-lines or sockets:
- Child stdout+stderr → `tasks/<id>/run.log`. `LogTailer` (`infrastructure/log_tail.py:41-114`, 0.3 s poll, byte-offset cursor) feeds `_make_task_log_callback` (`core.py:848-916`). Lines starting with `__EVENT__:<type>:<json>` (`cmd_builder.py:_EVENT_MARKER`) are parsed (`_parse_event_marker:122-129`) and mirrored to slot state — `pause_state`, `train_loop_started`, `auto_epoch_backup_written` (also persisted to `last_state_*` columns, `_persist_last_state:1768-1790`), `resume_state_loaded`, `eval_training_finished` — then re-published as typed SSE events. Everything else → SSE `task_log_appended {task_id, text, seq, end_offset}`; malformed markers → `event_malformed`.
- Metrics: the trainer writes `monitor/state.json` (`losses[], lr_history[], optimizer_metrics_history[], samples[], step, epoch, speed, config`). `MonitorStatePoller` (`log_tail.py:117-259`, 0.5 s mtime poll, ≥1 s publish throttle) computes a **delta** (`appended_losses/appended_lr/appended_samples`, config only when changed) → SSE `monitor_progress`. Cold start is `GET /api/state?task_id=` (`api/routers/health.py:49`, full file) then deltas.
- Transport: one SSE endpoint `GET /api/events` (`api/routers/events_sse.py:22-41`, 15 s keepalive) over `infrastructure/event_bus.py` — a thread-safe fan-out (`publish` → `loop.call_soon_threadsafe(_safe_put, q, event)`, per-subscriber `asyncio.Queue(maxsize=512)`, slow consumers drop with throttled warnings). Bus also has first/last-subscriber callbacks used to flush the generate image cache 30 s after the last client disconnects (`lifespan.py:205-232`). No WebSocket. Historical log paging: `GET /api/logs/{task_id}?tail|before|after` (`api/routers/logs.py`) strips `__EVENT__` lines and shares the `end_offset` coordinate system with SSE.
- `infrastructure/task_log.py` — `TaskLogLike` protocol / `TaskLog(logger)` adapter so services' progress callbacks carry log levels instead of `[error]`-style prefixes.
- `infrastructure/_messages_{train,worker,server}.py` + `log_messages.py` — an **i18n catalog for user-visible INFO log lines** (`msg("cli.torch_gpu", version=..)`, zh/en chosen via `ANIMA_UI_LANG`), ~320 keys in the server file alone; duplicates across the three dicts raise at import.

### Training config schema & pipeline

`studio/domain/training.py:35` `class TrainingConfig(BaseModel)` (`extra="ignore"`), ~170 fields, each with `json_schema_extra=_meta(group, control, show_when=..., disable_when=..., disable_value=..., advanced=...)` consumed by the frontend SchemaForm and `GET /api/schema`. Groups (`domain/common.py:GROUP_ORDER`, line 176): model → dataset → caption → lora → training → noise_augmentation → timestep_sampling → loss → system → output → sample → eval_validation → monitor. Field families: model paths (`model_family: Literal["anima","krea2"]`, `transformer_path`, `vae_path`, `text_encoder_path`, `t5_tokenizer_path`, `text_encoder_cache`); dataset (`data_dir`, `resolution: list[int]` snapped to /64 & deduped, `aspect_ratio_limit`, `reg_data_dir/reg_caption/reg_weight`); caption (`shuffle_caption`, `keep_tokens`, `flip_augment`, `tag_dropout`, `prefer_json`, `caption_comfy_encoding`); system (`cache_latents`, `blocks_to_swap`, 9 `navit_*`, 4 `cache_encode_*`, `mixed_precision`, `attention_backend`, `num_workers`, `vae_tiling`, `kv_trim`); LoRA (`lora_type: lora|lokr|loha|ortho|tlora`, `lora_rank/alpha`, `lokr_factor`, `tlora_*`, `lora_dora/rs/dropout/rank_dropout/module_dropout`, `lora_reg_dims`); training (`epochs`, `max_steps`, `batch_size`, `grad_checkpoint/accum`, `learning_rate`, `lr_scheduler*`, `optimizer_type: adamw|automagic|came|lion|prodigy|prodigy_plus_schedulefree|soap|soap_sf` + ~35 optimizer-specific fields, `weight_decay`, `grad_clip_max_norm`); noise/timestep (`noise_enhancement_type`, `noise_offset`, `pyramid_noise_*`, `timestep_sampling`, `timestep_shift*`, `infonoise_*`); loss (`loss_type mse|huber`, `loss_weighting none|min_snr|detail_inv_t|cosmap`, `masked_loss`, `leap_*`, `sra_*`); output (`output_dir`, `output_name`, `save_every_*`, `save_state_every_*`, `seed`, `resume_lora/state`); sampling (`sample_every/steps/infer_steps/cfg_scale/sampler_name/scheduler/width/height/seed/prompt(s)`); eval (`eval_validation_enabled/split_ratio/split_seed`, `eval_checkpoint_skip_count`); monitor (`loss_curve_steps`, `no_progress`, `log_every`).

Validators (`training.py:957-1144`): `_normalize_resolution`; before-validators `_apply_family_config_defaults` (overlay `FAMILY_CONFIG_DEFAULTS[family]`, e.g. krea2 → `shuffle_caption=False`, `sample_sampler_name="euler"`), `_migrate_save_keys`, `_migrate_noise_enhancement`, `_coerce_sample_sampler_scheduler` (grandfather anima's legacy values, fail-fast for krea2), `_pin_setdefaults`; after-validators `_validate_family_capabilities` (`FIELD_CAPABILITY_REQUIREMENTS` vs `FAMILY_CAPABILITIES`, `common.py:22-31`: anima = navit/sra/leap/caption_tag_ops/…, krea2 = masked_loss/text_cache/block_swap), `_enforce_disable_rules`, plus range checks. `domain/config_rules.py` is the declarative rule engine: `disable_when/disable_value` = "pin" rules, `option_disable_when` = "forbid" rules, `eval_show_when` mirrors the frontend's `evalShowWhen` string semantics (`_js_str`). `domain/config_prune.py:prune_inactive_fields` drops fields whose `show_when` is false (or `hidden` + default) before YAML is written.

Config persistence: **YAML files**, not DB. Global presets in `studio_data/presets/<name>.yaml` (`services/presets/io.py`, name regex `[A-Za-z0-9_-]+`); per-version private config at `projects/<id>-<slug>/versions/<label>/config.yaml` (`services/version_config.py:87-91`). `write_version_config` → `_tolerant_validate` (`presets/io.py:152-209`: migrate `flash_attn/xformers` → `attention_backend`, drop unknown keys, apply disable-rule fixes, then per-field fallback to defaults, reporting `dropped/defaulted`) → `prune_inactive_fields` → `render_config_yaml`. `PROJECT_SPECIFIC_FIELDS` (`data_dir, reg_data_dir, output_dir, output_name, resume_lora, resume_state`) are written once at creation and never back-filled. Enqueue (`api/routers/projects/training.py:831-899`) inserts a `tasks` row with `config_path=<version config.yaml>` and does **not** rewrite the YAML — the trainer subprocess reads the file directly. `infrastructure/argparse_bridge.py` reverse-compiles `TrainingConfig` into the trainer's argparse so CLI/YAML/Web share one schema. `services/data_io/train_io.py` exports/imports `train.zip` (schema 1) and `bundle.zip` (schema 2: `manifest.json`, `train/`, `reg/`, `presets/*.yaml`).

## C. Project / Version / Preset model

DB tables above. Domain rules: `projects.slug` is immutable (path anchor, `services/projects/projects.py:45-60`), `versions.label` matches `^(?!\.+$)[A-Za-z0-9_.-]+$` (`versions.py:160`, blocks `..`). Version lifecycle (`versions.py:27-62`):
```python
class VersionStatus:  PREPARING, TRAINING, COMPLETED, FAILED, CANCELED
class VersionPhase:   ORDER = (CURATING, PREPROCESSING, TAGGING, EDITING, REGULARIZING, READY)
                      SKIPPABLE = {PREPROCESSING, REGULARIZING}
```
`derive_status_from_tasks` (`versions.py:87-121`) derives status from GPU-type tasks only; `reconcile_version_status` self-heals mismatches on read. `services/projects/phase.py` implements the forward-only cursor: `check_curating` (≥1 train image), `check_tagging/editing` (100 % caption coverage), `check_preprocessing/regularizing` (no active job), `check_ready` (config exists & validates); `advance_phase`/`skip_phase`.

**On-disk layout** (`infrastructure/paths.py`, `projects.py:68-79`, `versions.py:184,379-385`, `version_config.py:53-79`):
```
<repo>/studio_data_location.json          optional pointer → custom STUDIO_DATA (needs restart)
studio_data/
  studio.db, secrets.json, .server.lock, .update_pending|.update_cache|.last_version|.update_log
  presets/<name>.yaml                     global presets
  logs/<id>.log                           legacy task logs (read-only)
  thumb_cache/, tag_dictionary/{active.json,source.sqlite}
  .cache/generate/session-<uuid>/*.bin    encrypted test-image cache (per process)
  test/<date>/{single,xy/<folder>}/*.png  persisted test generations
  tasks/<id>/{snapshot/config.yaml, monitor/state.json, samples/*.png, run.log, state/, eval/}
  eval/sessions/<id>/{plan.json, reference_manifest.json, samples/<run_id>/, report.json}
  projects/<id>-<slug>/
    project.json, download/, preprocess/
    versions/<label>/
      version.json, config.yaml, .unlocked.json
      train/<N>_data/{*.png,*.txt|*.json,*.mask,*.npz}, train/manifest.json
      validation/<N>_data/…   (held-out split)
      reg/{meta.json,<folder>/…}
      output/{<slug>_<label>_final|_epochN|_stepN}.safetensors, output/state/task_<id>/
<repo>/models/ (or secrets.models.root)/{diffusion_models,vae,text_encoders,t5_tokenizer,wd14,eval,taeflux,loras,upscale_models}
<repo>/tmp/restart, <repo>/data_exports/
```
Presets flow (`services/presets/__init__.py`): `fork_preset_for_version` copies a global preset into the version config, applies project fields and (if `secrets.models.auto_sync_paths`) the 4 global model paths; `save_version_config_as_preset` reverses, clearing project fields to defaults. Routers: `api/routers/projects/{crud,ingestion,curation,training,exports,eval_samples,eval_metrics}.py`, `api/routers/presets.py` (+`GET /api/schema`, `POST /api/schema/preview-yaml`).

## D. Inference daemon

`services/inference/daemon.py:77` `class InferenceDaemon` — server-side proxy for a **long-lived subprocess** `runtime/anima_daemon.py` (`_DAEMON_SCRIPT`, line 51), process-wide singleton via `get_daemon()` (line ~874). Lazy spawn on first `generate` task (`core.py:1141-1151`).

- **IPC**: line-delimited **JSON over stdin/stdout**; stderr is a log channel. `Popen(stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True, bufsize=1)` (`daemon.py:356-367`). Requests (`submit_task:485-491`): `{"id": "task-<tid>-<seq>", "action": "generate", "task_id", "config", "output_dir"}`; also `{"action":"cancel","target_id"}`, `{"id":"_unload","action":"unload"}`; `ping`. Stop = close stdin (EOF) then kill after 10 s (`stop:405-437`). Responses/events (`_handle_event`): `ready`, `loaded`, `unloaded` (global); `started`, `phase`, `image_started`, `preview_step` (TAEFlux JPEG b64), `image_done` (base64 PNG — decoded server-side into the encrypted cache and stripped before forwarding), `done`, `canceled`, `error`.
- **State machine**: `stopped → starting → idle ⇄ busy`, `unloading` (`daemon.py:43-47`); `is_model_loaded` tracked separately (model stays resident across tasks). Idle-timeout timer auto-unloads (`secrets.generate.idle_timeout_minutes`), per-image task-timeout timer hard-kills a hung daemon (`_on_task_timeout:260-289`, generation-counter guarded). Supervisor revokes the "exclusive lease" via `request_unload()` before training.
- **Config**: `POST /api/generate` (`api/routers/generate.py:128-319`) builds a `GenerateConfig` (`domain/generate.py:41`: prompts, w/h, steps, cfg, sampler/scheduler, `lora_configs: list[LoraEntry{path,scale,project_id,version_id}]`, `xy_matrix`, `vae_precision`, `lora_merge_precision`, `attention_backend`, `vram_policy`, `ram_guard`, `blocks_to_swap`, `distilled`) → `force_comfy_parity_runtime_config` → writes `anima_gen_<tid>/config.json` and sets `tasks.config_path`; supervisor `_submit_to_daemon` injects `__monitor_state_file` and passes the dict. Server-only keys (`_anima_params_snapshot_`, `save_test_images_at_dispatch`) are stripped/consumed in `submit_task`.
- **LoRA / fp8**: `services/inference/core.py` (`LoRAMeta:143`, `read_lora_meta`, `apply_loras:319`) reads rank/alpha/algo/DoRA/rs-LoRA/`lora_reg_dims`/`model_family` from safetensors metadata and injects one adapter per LoRA; the fp8 handling itself (fp8-scaled Krea 2 weights, `lora_merge_precision`) lives in the runtime; studio only carries the settings. Krea 2 `text_encoder="fp8"` selects the `Comfy-Org/Krea-2` fp8 TE dir (`generate.py:138-149`).
- **Images**: `services/inference/disk_cache.py` — per-process session dir `studio_data/.cache/generate/session-<uuid>/`, files are `[16B nonce][SHAKE-128 keystream XOR payload]` (threat model: passive disk scanners on cloud hosts), 200 files / 500 MB cap, wiped at shutdown and on next startup. `services/generate_storage.py` optionally persists PNGs to `studio_data/test/<date>/...` and is the single writer of `tasks.generate_images`. `upscaler.py` is a spandrel-based tiled upscaler used by preprocess.

## E. Evaluation pipeline

One evaluation = one `eval_session` task row + one worker process (`workers/eval_session_worker.py`, issue #465 refactor). Model (`services/eval_session.py:35-70`, `_v19` tables): `eval_sessions{task_id, parent_task_id, project_id, version_id, trigger manual|after_training, status pending|running|done|partial|failed|canceled, stage, plan_json}`, `eval_candidates{role checkpoint|baseline, checkpoint_path (relative to version dir), checkpoint_digest (size+mtime hash), epoch, step, status, samples_done/total, run_id}`, `eval_metric_results{candidate_id, metric_key, status, value, model_ref, sample_count, reason, details_json}` (UNIQUE on candidate+metric). Stages run sequentially inside the worker: `generate` → `metric:<runner>` per enabled runner → `aggregate` (`eval_session_worker.py:46-62`); DB is the source of truth, so a killed session resumes by skipping `done` candidates/metrics.

Metrics registry (`services/eval_registry.py:22-51`): `clip_t`, `clip_i` (runner `clip`, `openai/clip-vit-base-patch32`), `dino_i` (`facebook/dinov2-small`), `ccip_i` (`deepghs/ccip_onnx` variant `ccip-caformer-24-randaug-pruned`, ONNX, anime character identity), `tag_recall` (WD14 re-tagging of generated images vs prompt tags). Reference/prompt set = the version's held-out `validation/` folder (`services/eval_validation.py`, split by moving images out of `train/` once before training). Generation reuses the inference daemon in-process (`services/eval_generation.py:DaemonSampleGenerator`, `cache_images=False`) so the base model loads once per session with LoRA hot-swap. `services/eval_model_pool.py:ModelPool` holds each metric model for the duration of its stage and releases VRAM between stages. Triggers: `eval_auto.queue_training_finished_eval` (called from `_finish_slot` when the trainer emitted `eval_training_finished`) and the manual route in `api/routers/projects/eval_samples.py`; `eval_cleanup.py` is a one-shot purge of the pre-#465 fan-out rows.

## F. Model download center

`services/models/{catalog,paths,sources,downloader}.py` + `families/{anima,krea2}.py`. Root = `secrets.models.root` or `<repo>/models` (`paths.py:models_root`).

Catalog entries (asset id → source → local target):
- `anima_main` — HF `circlestone-labs/Anima`, variants `1.0` (`anima-base-v1.0.safetensors`, latest), `preview3-base`, `preview2`, `preview` → `diffusion_models/` (`families/anima.py:14-24`); `anima_vae` — same repo `split_files/vae/qwen_image_vae.safetensors` → `vae/` (shared with Krea 2); `qwen3` — `Qwen/Qwen3-0.6B-Base` (6 files) → `text_encoders/`; `t5_tokenizer` — `google/t5-v1_1-xxl` (3 tokenizer files only) → `t5_tokenizer/`.
- `krea2_main` — `raw` (`krea/Krea-2-Raw`, 26.3 GB), `raw_fp8` (`Comfy-Org/Krea-2/diffusion_models/krea2_raw_fp8_scaled.safetensors`, 13.1 GB), `turbo` (`krea/Krea-2-Turbo`, purpose=inference → distilled sampling defaults), `turbo_fp8` (`families/krea2.py:16-56`); `krea2_text_encoder` — `Qwen/Qwen3-VL-4B-Instruct` (12 files) → `text_encoders/Qwen_Qwen3-VL-4B-Instruct/`; `krea2_text_encoder_fp8` — `Comfy-Org/Krea-2/text_encoders/qwen3vl_4b_fp8_scaled.safetensors` + small config files.
- Tooling: `wd14` — `SmilingWolf/*` (`model.onnx`, `selected_tags.csv`); `cltagger` — `cella110n/cl_tagger` (`cl_tagger_1_02/model.onnx`) and gated `cella110n/cl_tagger_v2` (`v2_01a/model.onnx` + `.data` + `model_vocabulary.json`); `eval_clip/eval_dino/eval_ccip`; `upscaler` — `4x-AnimeSharp` (HF `Kim2091/AnimeSharp`), `R-ESRGAN_4x+Anime6B`, `4x_foolhardy_Remacri`, `ESRGAN_4x` (ModelScope `libfishopen/upscaler`); `taeflux` — `madebyollin/taef1` (preview decoder). Plus `*_custom` user-registered candidates stored in `secrets.model_sources`.
- Sources (`sources.py`): HuggingFace (per-call `endpoint=` from `secrets.huggingface.endpoint` or `HF_ENDPOINT`; token from env or secrets) and ModelScope (per-type `secrets.download_sources[type]` or `MODELSCOPE_SOURCE` env; mirror maps `SmilingWolf/*→fireicewolf/*`, `openai/clip-*→AI-ModelScope/*`, Anima TE packed as a single safetensors). Downloads run in daemon threads (`downloader.start_download_async:685-742`), status kept in a module dict with a 30-line log ring and broadcast as SSE `model_download_changed`; `trigger(model_id, variant)` (line 853) is the routing switch; `delete_asset` resolves paths server-side only.

## G. Other services (brief)

- **Tagging** (`services/tagging/`): `Tagger` protocol + factory (`base.py`); `wd14.py`/`cltagger.py` via `onnx_base.py` (onnxruntime, CPU or GPU EP); `llm.py` OpenAI-compatible vision endpoint (chat/responses, JSON output, presets in `secrets.llm_tagger`, JoyCaption is a preset shim); `caption_format.py`/`caption_snapshot.py`. Worker: `workers/tag_worker.py`.
- **Preprocess** (`services/preprocess/`): `core.py` (upscale/crop/inpaint pipeline, ADR 0004/0010), `manifest.py` (per-version `train/manifest.json` origin/state), `duplicates.py` (1479 lines, duplicate/same-scene review groups), `masks.py` (`.mask` grayscale sidecars for masked loss). Worker: `preprocess_worker.py`.
- **Reg builder** (`services/reg/builder.py`, 940 lines): greedy tag-descending booru search ported from a personal script with hard-coded heuristics (tag sequence 10→5→3→2→1, `max_rounds=50`); `analysis.py` scoring, `postprocess.py` aspect-ratio KMeans crop clustering, `dedup.py`. Worker: `reg_build_worker.py`. `reg_ai` (prior generation with base model) runs as a TRAIN-slot task via `runtime/anima_reg_ai.py` (`domain/reg.py:RegAiConfig`).
- **Booru** (`services/booru/`): `api.py` (Gelbooru/Danbooru HTTP), `pool.py` (Session keepalive, dual token bucket API 2 rps / CDN 5 rps, 429 backoff), `downloader.py`, `gallery.py`. Worker: `download_worker.py`.
- **Secrets/settings** (`infrastructure/secrets.py`, 1419 lines): single pydantic `Secrets` model persisted to `studio_data/secrets.json`; sub-configs `gelbooru, danbooru, huggingface, wandb(presets), modelscope, eval_metrics, download, reg, llm_tagger, wd14, cltagger, models(root/selected/selected_te/custom/auto_sync_paths), queue, training(ram_guard), generate, system(ui_language, gpu_index…), proxy, tag_dictionary, model_sources`; masked GET (`***`) with deep-merge PUT; ~200 lines of legacy schema migration.
- **Updater / runtime installs** (`services/runtime/`): `updater.py` git-based self-update via flag files (`tmp/restart`, `.update_pending`, `.update_force`, `.last_version`, `.update_log`), applied by `cli.py` between server runs; `pending_install.py` defers pip installs that would hit Windows `.pyd` locks; `torch.py` (detect CUDA build, recommend `cuXXX` index, reinstall), `onnxruntime.py` (CPU vs GPU package switch), `flash_attention.py`, `xformers.py`, `gpu_select.py` (`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES`). Routers: `api/routers/{system,installs}.py`.
- **System stats / diagnostics**: `services/system_stats.py` (psutil + lazy pynvml), `services/diagnostics.py` (zip with `env.json`, task row, `run.log`, snapshot, `state.json`, time-windowed `studio.log`, redacted). Others: `lora_catalog.py` (TTL-cached LoRA scan), `studio_data.py`/`models_storage.py` (copy-migrate storage roots), `announcements.py` (markdown posts in `docs/announcements/`), `proxy_manager.py`, `infrastructure/tag_dictionary.py` (~31k-entry Danbooru zh/en dictionary), `infrastructure/logging.py` (JSON-lines `studio.log`, human console formatter, trace/job ContextVars).

## H. Frontend (stack only)

`studio/web/package.json`: React 18.3, TypeScript 5.6, Vite 5, Tailwind 3.4, react-router-dom 6, i18next 26 / react-i18next 17, react-virtuoso, @dnd-kit, react-markdown + remark-gfm; tests with Vitest 2 + Testing Library; ESLint 9. `src/` top-level: `api/` (typed fetch client with `X-Trace-Id` handling), `components/` (93 files, shared UI), `context/` (`ProjectContext`), `i18n/` (locales), `lib/` (44 files: schema evaluation mirror `schema.ts`, settings drawer, crop clustering, helpers), `pages/` (92 files: Projects, Queue, QueueDetail, `project/*` phase pages, `queue/*`), `styles/` (design tokens, responsive CSS), `tagDict/` (tag autocomplete store), `test/` (setup). Built `dist/` is served by FastAPI at `/`.

## I. Weaknesses / tech debt

1. **Monolith supervisor**: `supervisor/core.py` is 1814 lines / one class with 37 methods sharing mutable `self` state; the module docstring (`core.py:15-20`) explicitly decides *not* to split it. Late imports inside methods (`core.py:689, 983, 1003, 1114, 1410`) paper over layering cycles (`supervisor → services → infrastructure → services`).
2. **Layer contract violations**: `services/__init__.py:4` says services never touch `db`, yet `services/projects/jobs.py`, `eval_session.py`, `eval_auto.py`, `version_config.py`, `eval_validation.py` all import and use `db`. Migrations `_v11`/`_v13` lazily import `services.projects` (infrastructure → services).
3. **Polling everywhere**: 1 s supervisor tick opens 2–3 fresh SQLite connections per tick (`_queue_held`, `_promote_due_scheduled`, `_next_pending_task_in` loads *all* pending rows), `LogTailer` 0.3 s and `MonitorStatePoller` 0.5 s per running task, daemon `start()` busy-waits at 50 ms. `secrets.load()` (`secrets.py:944-954`) re-reads and re-validates `secrets.json` on **every call** with no cache; `_popen` calls it four times per spawn and the tick calls it whenever data jobs are pending.
4. **Text-protocol IPC**: worker→supervisor events ride on the same stdout stream as arbitrary library logs (`__EVENT__:` prefix); a library that prints a line starting that way, or a partially-flushed line, produces `event_malformed`. Trainer and daemon speak different protocols (marker lines vs JSON-lines).
5. **Scheduling is nominal, not measured**: resource classes are static labels (`resources.py`), no VRAM/CPU probing; a "light" tagger on a 8 GB card can still OOM alongside training. Unknown kinds default to exclusive.
6. **Documented race windows**: `generate` enqueue creates the task before `config_path` is set — the dispatcher skips it but also **blocks the FIFO head** until the row is fixed (`core.py:568-574`); pause vs. process exit degrades to `canceled` if `pause_state` isn't emitted in time (`core.py:1570-1579`); `is_pausable` depends on in-memory slot state that resets on restart; the 30 s cancel grace timer polls `proc.poll()` from a second thread alongside the supervisor loop (`core.py:1690-1702`).
7. **Single-process / single-instance by design**: file lock on `studio_data/.server.lock` (`single_instance.py`), one supervisor thread, one inference daemon, one TRAIN slot — no multi-GPU or multi-node path; `gpu_select` only pins a single card via env.
8. **Windows-specific code** in 7 files (19 branches): `CREATE_NEW_PROCESS_GROUP`, `CTRL_BREAK_EVENT` for pause, `taskkill /T /F` for cancel (so cancel is a hard kill on Windows with no graceful shutdown), `msvcrt.locking`, Proactor loop filters, `npm.cmd/.ps1` discovery, `.pyd` lock workaround (`pending_install`), cp936/cp932 encoding fixes.
9. **Config is files, index is DB**: YAML presets/version configs and `state.json` are authoritative on disk while `tasks` references them by absolute path (`config_path`, `monitor_state_path`, `paused_state_path`); moving `studio_data` requires the pointer file + restart, and legacy path fallbacks are scattered (`logs/<id>.log`, `versions/<label>/monitor/task_<id>/`, `output/state/task_<id>/`).
10. **Compat shims & legacy surface**: `studio/{db,paths,secrets,schema}.py`, `services/{model_downloader,updater}.py`, `services/tagging/joycaption.py`, frozen `project_jobs` table, five legacy `eval_*` task types kept in `VALID_TASK_TYPES`, `custom_anima_paths`/`selected_anima` computed fields, ~240 "legacy/兼容/shim/退役" markers. Zero `TODO|FIXME|HACK|XXX` in the tree — debt is tracked in docstrings/`docs/todo/` rather than inline markers.
11. **Security/robustness nits**: `db.update_task` interpolates column names from `**fields` (internal callers only, but no allow-list); `/api/system/{restart,update,rollback,init_git}` and `open-folder` are unauthenticated (loopback-only check exists just for `open-folder`, `queue/outputs.py:34`); the encrypted image cache is XOR with a SHAKE keystream and no integrity (explicitly scoped to "passive scanner" threat model, `disk_cache.py:13-17`).
12. **Hard-coded domain assumptions**: reg builder thresholds copied verbatim from a personal script (`reg/builder.py:5-16`, incl. a Windows desktop path in the docstring); `_final.safetensors` naming baked into `finalizer.py:65`; error-message extraction depends on the `LOG_LINE_RE` line contract and a 256 KB tail (`core.py:132-188`); ModelScope mirror maps and size estimates are literal tables in `sources.py`/`catalog.py`.
13. **Comments and user-facing strings are Chinese-first** (validators raise Chinese messages, e.g. `training.py:986, 1097`), while the error envelope expects i18n by `code` — mixed-language contract for API consumers.
