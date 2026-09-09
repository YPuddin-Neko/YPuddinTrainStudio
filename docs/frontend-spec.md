# 前端需求文档（v1）— YPuddin Train Studio Web UI

面向：负责 `frontend/` 的前端会话（Kimi前端）。后端与训练核心由另一会话（Claude，"训练器构建"）负责。
本文是**双方的接口契约**：接口路径、数据形状、事件类型一旦在此定义，后端实现必须与之一致；如需变更，先改此文档再改代码。

> 整体架构见 [`design/00-architecture.md`](design/00-architecture.md)；可以参考 `../AnimaLoraStudio/studio/web/`（React + Vite + Tailwind）的交互思路，但**不要复制其代码**（GPL-3.0）。

---

## 0. 协作方式

- 你只在 `frontend/` 目录内工作（以及 `.handoff/frontend-status.md`）。不要修改 `ypuddin/`、`tests/`、`docs/design/`。
- 每完成一个里程碑（见 §8），更新 `.handoff/frontend-status.md`，格式：

  ```markdown
  # Frontend status
  updated: 2026-09-10T12:00
  milestone: FE-M1
  status: done | in_progress | blocked
  ## Done
  - ...
  ## How to run
  - cd frontend && npm i && npm run dev
  ## Tests
  - npm run lint: pass ; npm run test: 12 passed ; npm run build: pass
  ## Questions / blockers for backend
  - ...
  ```

- 后端就绪前，用 **MSW（Mock Service Worker）** 按本文契约造 mock（放在 `frontend/src/mocks/`），使所有页面在无后端时可运行。真实后端上线后，`docs/api/openapi.json` 会出现，请用 `openapi-typescript` 生成类型并替换手写类型。
- 开发服务器把 `/api` 代理到 `http://127.0.0.1:8765`。生产构建产物 `frontend/dist/` 由后端静态托管在 `/`。

## 1. 技术栈（要求）

- React 18 + TypeScript（strict）+ Vite 5；Tailwind CSS；React Router 6。
- 数据层：TanStack Query（REST）+ 一个共享的 `useEventStream()` hook（单条 `EventSource` 连接 `/api/events`，多组件订阅，断线自动重连并带 `Last-Event-ID`）。
- 图表：需支持 **10 万点级别实时曲线**（loss/lr 每步一个点）流畅渲染 —— 推荐 uPlot 或 ECharts（canvas）；不要用 SVG 图表库画训练曲线。
- 表单：**由 JSON Schema 驱动**（见 §4），不要手写训练参数表单。
- 国际化：i18next，中文为默认、英文可切换；所有文案进 `locales/zh-CN.json`、`locales/en.json`。
- 主题：明/暗两套，跟随系统 + 手动切换。
- 质量：ESLint + Prettier；Vitest + Testing Library；`npm run build` 必须零警告通过。
- 桌面宽屏优先（≥1280px），窄屏只需不坍塌。

## 2. 目录约定

```
frontend/
  package.json  vite.config.ts  tsconfig.json  tailwind.config.ts  index.html
  src/
    main.tsx  App.tsx  router.tsx
    api/         client.ts（fetch 封装：错误信封、X-Trace-Id）、types.ts（先手写，后由 openapi 生成）、hooks/*
    events/      useEventStream.ts、eventTypes.ts
    schema/      SchemaForm/*（JSON Schema → 表单）、showWhen.ts（表达式解释器）
    pages/       Dashboard/ Projects/ ProjectDetail/ Dataset/ TrainConfig/ Queue/ JobDetail/ Artifacts/ Models/ Settings/
    components/  通用组件（表格、抽屉、图表、图片网格、日志视图、标签编辑器…）
    mocks/       MSW handlers + 假数据生成器
    i18n/        locales/zh-CN.json  locales/en.json
    styles/
  tests/         组件与 hooks 测试
```

## 3. 页面与功能

### 3.1 Dashboard `/`
- 顶部系统状态条：GPU（名称、显存已用/总量、利用率、温度）、CPU、RAM、磁盘（数据根目录）。来源 `GET /api/system/stats` 初始化 + SSE `system.stats` 更新（≈2 s 一次）。
- 正在运行的任务卡：名称、阶段、进度条（step/total、ETA）、实时 loss 迷你图、最近一张 sample 缩略图；点击进入任务详情。
- 队列摘要（等待/已排期数量）、最近完成任务、最近产物（LoRA 文件）。

### 3.2 项目 `/projects`、`/projects/:id`
- 项目列表：名称、备注、创建时间、数据集数、任务数、最近产物；新建/重命名/归档/删除（删除需二次确认，提示会删除磁盘上的项目目录）。
- 项目详情 tabs：**数据集**（该项目注册的数据源）、**训练配置**（该项目的配置草稿与已保存版本）、**任务**（本项目的任务历史）、**产物**（LoRA 文件）。

### 3.3 数据集 `/datasets/:id`
- 注册数据源：填写目录路径（服务器本机路径）+ 选项（repeats、caption 扩展名、是否正则集、prior 权重、class prompt）。后端会扫描并建立索引（异步任务，进度走 SSE `job.cache_progress`）。
- 概览：图片数、caption 覆盖率、分辩率/长宽比分布直方图、**分桶预览**（`POST /api/plan` 返回的 buckets：每个 (w,h) 的图片数）、缓存状态（latents / 文本缓存命中率）。
- 图片网格（虚拟滚动，缩略图 `GET /api/datasets/{id}/images/{hash}/thumb?size=256`），点选查看大图 + caption 编辑器：
  - caption 以逗号分隔 tag 显示为 chips，支持增删改、拖动排序、批量给选中图片加/删 tag、搜索过滤含某 tag 的图片。
  - 保存走 `PUT /api/datasets/{id}/images/{hash}/caption`（后端直接改 `.txt`）。
- 顶部动作：重新扫描、预缓存（发起 `cache` 类型任务）、删除数据源（仅移除注册，不删文件）。

### 3.4 训练配置 `/projects/:id/train`（核心页面）
- 左侧：**Schema 驱动表单**（§4）。分组显示为可折叠区块或 tabs（顺序由 `x-ui.group` 与 `x-ui.order` 决定）；"高级"开关隐藏 `x-ui.advanced=true` 字段；`x-ui.show_when` 不满足的字段隐藏。
- 顶部：预设选择器（`GET /api/presets`）、"从预设加载"、"另存为预设"、"重置为默认"、导入/导出 TOML。
- 右侧 **Plan 面板**：表单每次改动（防抖 500 ms）调用 `POST /api/plan`，展示：每 epoch 步数、总步数、分桶列表、可训练参数量、优化器状态大小、**显存估算（按分辩率的峰值）与建议**、以及 `warnings[]`/`errors[]`（校验错误定位到字段并高亮）。
- 底部：**入队**（`POST /api/jobs` type=train）、可选"排期在 …"（`scheduled_at`）与优先级。
- 配置 JSON 与后端 `TrainConfig` 一一对应；前端不做业务校验，只呈现后端返回的错误。

### 3.5 队列 `/queue`
- 表格：ID、名称、类型（train/cache/sample/convert）、项目、状态（`queued/scheduled/running/paused/pausing/cancelling/completed/failed/cancelled`）、进度、优先级、创建/开始/结束时间、GPU。
- 动作：暂停 / 恢复 / 取消 / 立即保存检查点 / 删除记录 / 重试（复制配置新建任务）/ 拖动调整优先级（`PATCH /api/jobs/{id}` priority）/ 全局"暂停调度"开关。
- 状态实时来自 SSE `job.state`。

### 3.6 任务详情 `/jobs/:id`（实时监控）
- 头部：状态、阶段时间线（`preparing → caching → training → finalizing`）、step/total、epoch、ETA、速度（it/s、img/s）、显存峰值。
- 图表（可切换对数轴、EMA 平滑系数、按 step/epoch/时间横轴）：loss（原始 + 平滑）、每参数组 lr、grad norm、验证损失（每个固定时间步一条线 + 均值）、吞吐与显存。
  - 初始化：`GET /api/jobs/{id}/metrics`（全量历史，列式数组）；增量：SSE `job.step` / `job.validation`。
- Samples 画廊：按 step 分组的预览图（`GET /api/jobs/{id}/samples`，SSE `job.sample` 追加），支持同一 prompt 跨 step 对比（横向滑动）。
- 检查点列表：step、文件、大小、下载、转换（ComfyUI/PEFT 格式）、"用它继续训练"。
- 日志：虚拟滚动、跟随底部、级别过滤、搜索；初始 `GET /api/jobs/{id}/log?offset=`，增量 SSE `job.log`（后端节流 ≤10 Hz，合并多行）。
- 配置快照查看（只读 + 与所用预设的 diff）。

### 3.7 产物 `/artifacts`
- 列出所有 LoRA/LoKr 文件：名称、项目、任务、算法、rank/alpha/factor、大小、时间；查看元数据（`ss_*`、`modelspec.*`、`ypuddin.*`）与每模块参数量表；转换导出；删除。

### 3.8 模型权重 `/models`
- 已注册权重：模型族、类型（`dit / text_encoder / vae / tokenizer`）、路径、大小、精度（bf16/fp8）、状态（存在/缺失）；添加本地路径、扫描目录、设为默认。下载中心留位（v2）。

### 3.9 设置 `/settings`
- 路径：数据根目录、缓存目录、模型目录、输出目录；服务：端口、是否允许非本机访问；界面：语言、主题、图表默认平滑。`GET/PUT /api/settings`。

## 4. Schema 驱动表单规范

`GET /api/schema/train` 返回 JSON Schema（draft 2020-12，pydantic v2 生成），对象为 `TrainConfig`，嵌套子对象 `model / dataset / adapter / objective / optimizer / scheduler / memory / loop / checkpoint / sampling / validation / logging`。每个字段可能带：

```json
"x-ui": {
  "group": "adapter",          // 分组键，标题从 i18n 取 groups.adapter
  "order": 20,                 // 组内排序
  "advanced": false,
  "control": "select | number | slider | switch | text | path | textarea | tags | rules",
  "unit": "steps",
  "help": "…",                 // 帮助文本（zh），可能同时有 help_en
  "show_when": "adapter.algo == 'lokr'",
  "min": 1, "max": 1024, "step": 1
}
```

`show_when` 表达式语法（前端需实现解释器，后端有同构实现）：

- 操作数：字段路径 `a.b.c`、字符串 `'x'`、数字、`true/false/null`
- 运算：`== != < <= > >= in && || !`，括号；`in` 右侧为字符串/数字数组 `['lora','lokr']`
- 示例：`"adapter.algo in ['lokr'] && adapter.rank != 'full'"`

渲染规则：`enum` → select；`boolean` → switch；`integer/number` → number（有 min/max 且 `control=slider` 时用滑块）；`string` + `control=path` → 路径输入（带"浏览服务器目录"弹窗，`GET /api/fs/list?path=`）；`array of string` + `control=tags` → chips；`adapter.rules` 用专用组件（可增删排序的规则行：match / algo / rank / alpha / factor / lr）。`anyOf` 含 `null` 的可选字段提供"未设置"状态。未知字段按类型兜底渲染，不能崩。

## 5. REST 契约（v1）

基址 `/api`，JSON；时间为 ISO-8601 UTC；ID 为字符串；分页 `?page=1&page_size=50` → `{items, total, page, page_size}`。
错误信封：`{"error": {"code": "job.not_found", "message": "…", "trace_id": "…", "details": {}}}`；请求头可带 `X-Trace-Id`，响应总是回传。

| 方法 & 路径 | 说明 / 响应 |
|---|---|
| `GET /health` | `{version, api_version: 1, torch, cuda, gpus: [{index,name,total_mb}]}` |
| `GET /system/stats` | `{cpu_pct, ram: {used_mb,total_mb}, disks: [{path,used_gb,total_gb}], gpus: [{index,util_pct,mem_used_mb,mem_total_mb,temp_c}]}` |
| `GET /system/info` | 环境详情（python/torch/依赖版本、平台） |
| `GET /settings` · `PUT /settings` | `{paths: {data_root,cache_dir,models_dir,output_dir}, server: {host,port}, ui: {language,theme}}` |
| `GET /fs/list?path=` | `{path, parent, entries: [{name,is_dir,size,mtime}]}`（限制在允许的根目录内） |
| `GET /schema/train` | JSON Schema（§4） |
| `POST /config/validate` | body: config → `{ok, errors: [{loc: "adapter.rank", msg}], warnings: [...]}` |
| `POST /plan` | body: `{config, dataset_ids?}` → `Plan`（§6.1） |
| `GET /presets` · `POST /presets` · `GET/PUT/DELETE /presets/{name}` | `Preset {name, description, config(partial), builtin, updated_at}` |
| `GET /projects` · `POST /projects` · `GET/PATCH/DELETE /projects/{id}` | `Project {id, name, note, created_at, updated_at, archived, dataset_ids, stats: {jobs, artifacts}}` |
| `GET /projects/{id}/config` · `PUT /projects/{id}/config` | 项目当前训练配置草稿（完整 config） |
| `GET /projects/{id}/datasets` · `POST /projects/{id}/datasets` | 注册数据源 `DatasetSource {id, project_id, path, repeats, caption_ext, is_reg, prior_weight, class_prompt, created_at}` |
| `GET /datasets/{id}` | `DatasetInfo {source, stats: {images, captioned, avg_tags, resolutions: [{w,h,count}], ar_hist: [...]}, cache: {latents: {cached,total}, text: {cached,total}}, index_status: "ready|indexing|stale"}` |
| `POST /datasets/{id}/rescan` · `DELETE /datasets/{id}` | |
| `GET /datasets/{id}/images?page=&q=` | `{items: [{hash, rel_path, width, height, caption, has_mask}], total,…}` |
| `GET /datasets/{id}/images/{hash}/thumb?size=256` · `GET .../file` | 图片 |
| `GET/PUT /datasets/{id}/images/{hash}/caption` | `{caption}` |
| `POST /datasets/{id}/tags/batch` | `{hashes: [...], add: [...], remove: [...]}` |
| `GET /jobs?status=&project_id=&page=` · `POST /jobs` | `Job`（§6.2）；创建 body `{type: "train|cache|sample|convert", project_id?, name, config?, priority?, scheduled_at?}` |
| `GET /jobs/{id}` · `PATCH /jobs/{id}` (priority, name) · `DELETE /jobs/{id}` | |
| `POST /jobs/{id}/pause|resume|cancel|save|retry` | 返回更新后的 `Job` |
| `GET /jobs/{id}/metrics?since_step=` | `{steps: [], loss: [], loss_ema: [], lr: {group: []}, grad_norm: [], vram_mb: [], it_s: [], validation: [{step, per_t: {t: loss}, mean}]}` |
| `GET /jobs/{id}/samples` | `[{step, prompt_index, prompt, seed, url, width, height, created_at}]` |
| `GET /jobs/{id}/checkpoints` | `[{step, kind: "weights|full", path, size, created_at, artifact_id?}]` |
| `GET /jobs/{id}/log?offset=&limit=` | `{lines: [{ts, level, msg}], next_offset}` |
| `GET /jobs/{id}/config` | 任务配置快照 |
| `GET /queue/settings` · `PUT /queue/settings` | `{held: bool, max_concurrent: 1}` |
| `GET /artifacts?project_id=` · `GET /artifacts/{id}` · `DELETE` | `Artifact {id, project_id, job_id, name, path, size, algo, rank, alpha, factor, family, created_at, metadata}` |
| `POST /artifacts/{id}/convert` | `{format: "comfyui|peft|kohya"}` → 新 `Artifact` |
| `GET /artifacts/{id}/download` | 文件 |
| `GET /models` · `POST /models` · `DELETE /models/{id}` · `POST /models/scan` | `ModelAsset {id, family, kind, path, size, dtype, exists, is_default}` |
| `GET /events` | SSE（§7） |

## 6. 关键数据形状

### 6.1 Plan
```json
{
  "ok": true,
  "errors": [], "warnings": [{"code": "vram.tight", "msg": "…"}],
  "steps_per_epoch": 120, "total_steps": 2400, "epochs": 20,
  "buckets": [{"w": 1024, "h": 1024, "images": 80, "batches": 40}],
  "params": {"trainable": 12345678, "base": 2000000000},
  "memory": {"weights_mb": 4200, "adapter_mb": 48, "optimizer_mb": 96,
             "activations_mb_by_bucket": [{"w":1024,"h":1024,"mb":6100}],
             "peak_mb_estimate": 11800, "gpu_total_mb": 24576,
             "suggestions": ["enable memory.block_swap=8", "…"]},
  "text_encoding": "online|cached",
  "eta_estimate_s": null
}
```

### 6.2 Job
```json
{
  "id": "j_01H…", "type": "train", "name": "chara-v1", "project_id": "p_…",
  "status": "running", "priority": 0, "scheduled_at": null,
  "created_at": "…", "started_at": "…", "finished_at": null,
  "progress": {"phase": "training", "step": 310, "total_steps": 2400, "epoch": 2, "eta_s": 5400,
               "it_s": 1.7, "vram_peak_mb": 11200},
  "latest": {"loss": 0.123, "loss_ema": 0.131, "lr": {"default": 1e-4}},
  "error": null,
  "resume_from": null,
  "artifact_ids": []
}
```

## 7. SSE 事件（`GET /api/events`）

`event: <type>`，`data: <json>`，`id: <递增序号>`。所有任务事件都带 `job_id`。

| type | data |
|---|---|
| `system.stats` | 同 `GET /system/stats` |
| `job.state` | `{job_id, status, progress?, error?}` |
| `job.phase` | `{job_id, phase, message?}` |
| `job.cache_progress` | `{job_id, kind: "latents|text|index", done, total}` |
| `job.step` | `{job_id, step, epoch, loss, loss_ema, lr: {group: v}, grad_norm, it_s, vram_mb, eta_s}` |
| `job.validation` | `{job_id, step, per_t: {"0.1": 0.2, …}, mean}` |
| `job.sample` | `{job_id, step, prompt_index, prompt, seed, url, width, height}` |
| `job.checkpoint` | `{job_id, step, kind, path, artifact_id?}` |
| `job.warning` | `{job_id, code, msg}` |
| `job.log` | `{job_id, lines: [{ts, level, msg}], next_offset}` |
| `queue.changed` | `{}`（提示刷新队列列表） |
| `dataset.changed` | `{dataset_id}` |

## 8. 里程碑与验收

| 里程碑 | 内容 | 验收 |
|---|---|---|
| FE-M1 | 脚手架、路由、布局（侧栏 + 顶栏 + 系统状态条）、i18n、主题、MSW mock、API 客户端与错误信封处理、`useEventStream` | `npm run lint/test/build` 通过；所有路由可访问并显示 mock 数据 |
| FE-M2 | Schema 表单引擎（含 `show_when` 解释器 + 单测）、预设、Plan 面板、入队 | 用 `docs/api/train-schema.example.json` 渲染无崩溃；表达式单测 ≥20 条 |
| FE-M3 | 队列页、任务详情（图表、samples、日志、检查点）、Dashboard 实时卡 | mock 事件流下 10 万点曲线 60 fps |
| FE-M4 | 项目/数据集页（图片网格、caption 编辑、批量 tag、分桶预览）、产物、模型、设置 | 端到端 mock 流程可走通 |
| FE-M5 | 对接真实后端（openapi 生成类型、去 mock 开关）、修 bug | 真实训练一次全流程可用 |

## 9. 交互与视觉基调

- 训练监控以**信息密度**优先：数字用等宽字体，曲线可全屏；颜色区分参数组时用固定调色板并在图例里可点击隐藏。
- 危险操作（删除项目、取消任务）二次确认；长任务的按钮显示乐观状态（"pausing…"）直到 SSE 确认。
- 所有列表虚拟化；缩略图懒加载；日志窗口不超过 5 万行内存缓存（旧行丢弃，需要时按 offset 回拉）。
