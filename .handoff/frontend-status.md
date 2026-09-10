# Frontend status
updated: 2026-09-10T09:12
milestone: FE-M5（真实后端对接，toy 全流程走通）
status: done

## Done
- ✅ **openapi-typescript 类型生成**：`src/api/generated.ts` 已由 `docs/api/openapi.json` 生成（41 路径）。由于后端多数响应为 `additionalProperties: true` 的宽松 object，自动生成的类型不含字段细节，前端保留手写 `src/api/types.ts` 并按真实响应逐一对齐修正（见下方契约差异记录）。
- ✅ **MSW 开关**：`VITE_USE_MOCK=false npm run dev` 即关闭 MSW 直连真实后端；vite 代理 `/api → 127.0.0.1:8765`（修正了早期错误的 rewrite 规则，此前会把 `/api` 前缀吃掉导致全 404）。
- ✅ **真实 E2E 全流程走通**（脚本 `frontend/test-real-flow.js`，puppeteer-core 驱动本机 Chrome；截图在 `frontend/screenshots/01~11`）：
  1. Dashboard：真实 CPU/RAM/Disk 数据渲染正常（`01-dashboard.png`）。
  2. UI 新建项目 `toy-e2e-*` 成功（`02`），项目页支持新建/重命名/归档/删除（此前 New Project 按钮无 handler，已补全）。
  3. 注册数据集：`POST /api/projects/{pid}/datasets`（path=/tmp/ypuddin_demo，UI 属 FE-M4 范围，经 API 完成）。
  4. 训练配置页：选择 `toy-smoke` 预设（深合并覆盖）、SourcesEditor 添加 dataset.sources 指向数据集、sampling.enabled 开启、添加 prompt（64×64、steps 2），Plan 面板返回真实数据（total_steps=7、trainable=3072、buckets 3 个）（`03`）。
  5. 入队：`POST /api/jobs`（携带 project_id，成功后自动跳转任务详情）。
  6. 队列页：真实任务列表、状态徽标、进度条、时间戳正常（`05`）。
  7. 任务详情：job 训练完成（completed，7/7 步），loss/EMA/grad_norm 曲线、samples 画廊（真实生成的 PNG 经 `/api/jobs/{id}/files?kind=sample` 加载）、checkpoints（lora-epoch0001 + lora-final）、日志、配置快照全部正常（`06~09`）。
  8. 产物页：真实 LoKr 文件列表（algo/kind/rank/alpha/factor/大小/时间），下载、元数据查看、Convert 下拉（实测 UI 转 kohya 成功）、删除（`10~11`）。
- ✅ **对接中修复的前端 bug**：
  - `POST /api/jobs` 未携带 `project_id`（任务归属丢失）→ 已修（从路由参数注入）。
  - 初始 config 中 `optimizer.type: adamw8bit` 在无 bitsandbytes 环境下训练必失败 → 初始值改为后端默认 `adamw`。
  - SSE `job.sample` 未按 job_id 过滤且未去重（混入其他任务的 sample）→ 已修；`job.step` 增加 step 单调去重（重连重放不再产生重复 x 轴）。
  - `Job.progress`/`Job.latest` 排队态为 `{}`、`created_at` 为 Unix 秒浮点（原代码按 ISO/ms 处理渲染 1970 年）→ 类型与渲染已兼容。
  - SamplePrompt 字段名 `negative`（前端原写 `negative_prompt`，extra=forbid 会被拒）→ 已修。
  - 数字格式化：it/s 保留两位小数、trainable 参数量 K/M 自适应、VRAM null 显示 `--`。
- ✅ `npm run lint` / `npm run test`（23 通过）/ `npm run build` 全部通过；路由 React.lazy 拆分与 echarts 独立 chunk 保持有效。

## In progress / Not done
- 数据集 UI（FE-M4：图片网格、caption 编辑、批量 tag、分桶预览）、模型权重页 UI、设置页完整表单仍是骨架/未做。
- 项目详情页 tabs 内容未实现（FE-M4 范围）。

## How to run
- 后端：`cd xiangmuyuanma && .venv/bin/ypuddin serve --port 8765 --data-root /tmp/ypuddin_data`
- 前端（真实后端）：`cd frontend && VITE_USE_MOCK=false npm run dev` → http://localhost:3000
- 前端（MSW mock）：`npm run dev`（默认开启）
- E2E 复现：`node test-real-flow.js`（需本机装有 Chrome）

## Tests
- `npm run lint`: pass (0 warnings)
- `npm run test`: 23 passed (5 files)
- `npm run build`: pass
- 真实后端 E2E：新建项目 → 数据集 → 预设/配置 → Plan → 入队 → 完成 → 产物转换，全链通过（截图 11 张）

## Questions / blockers for backend
1. **Plan 响应与 frontend-spec §6.1 不一致**：buckets 元素用 `items`（spec 用 `images`）、响应多出顶层 `images`/`captioned` 字段、缺少 `eta_estimate_s`、`gpu_total_mb` 为 null。前端已按真实形状兼容，请确认以哪份契约为准并同步文档。
2. **时间字段类型**：`created_at` 等为 Unix 秒浮点（spec 说 ISO-8601 UTC）。前端已兼容 number/string 两种，请确认 openapi 中的正式类型（影响后续生成类型的可用性）。
3. **progress.phase 取值**：实际观测到 `injecting`（spec 只列 preparing/caching/training/finalizing）。前端时间线对未知 phase 做了兜底，请提供完整 phase 列表以便渲染。
4. **建议**：`/config/validate` 对采样/训练分辨率加上限保护——E2E 脚本曾误填 102464×102464，进程在采样阶段长时间无反馈地占用 CPU（现象像"卡死"），容易被用户误判为挂起。
5. 数据集注册后 `stats: {}`、`index_status: indexing`，FE-M4 需要知道索引完成的事件/轮询约定（`dataset.changed` 事件的触发时机）。
