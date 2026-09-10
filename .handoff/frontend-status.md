# Frontend status
updated: 2026-09-10T12:10
milestone: FE-M6（收尾 + 类型切换）
status: done

> 协作说明：本轮 B/C/D 的实现由旧 Kimi 会话完成（它其实活着，详见 `.handoff/kimi-to-claude.md` 的双会话警报）；新会话独立完成 A 的重跑验收，对 B/C/D 做了独立核对（读源码 + 复跑 lint/test/build + 看截图），并补了 C4 缺的 `job.sample_progress` 推送。

## Done
- ✅ **A. 预缓存回归（后端 commit b23c87f 验证通过）**：cache 任务 exit 0 → `completed`（`j_22ad61934655`，phase finalizing 正常）；`GET /datasets/{id}.cache` 返回 `{latents: {cached: 12, total: 12}, cache_dir}`（项目共享缓存目录生效）。数据集页概览卡显示 "Latents: 12/12"（截图 `21-cache-coverage.png`）。
- ✅ **B. openapi 类型切换**：`src/api/generated.ts` 用新 openapi.json（60 schemas）重新生成；`src/api/types.ts` 重构——Job/Project/DatasetInfo/Plan/Artifact/ModelAsset/Settings 等核心实体全部改为 generated 别名；手写部分只保留：SSE 事件 payload（`SampleProgressEvent`/`JobStepEvent`/`JobStateEvent`/`JobValidationEvent`/`CacheProgressEvent`）、`JobStatus` 联合、`JobListResponse` 分页包装、`ApiError`；对 generated 中两处过宽类型做了本地收窄（`DatasetInfo.cache`、`Plan.params/memory` 的 loose-map 联合）。
- ✅ **C1. Queue 拖拽调优先级**：表格行可拖拽，拖到目标行上方/下方（含蓝色插入指示线）后计算新 priority（target±1），乐观更新 + `PATCH /jobs/{id}`，失败回滚刷新。
- ✅ **C2. 任务详情图表重构**：三张图分离——① Loss（原始 + EMA + Grad Norm + 各参数组 lr 虚线右轴）；② Validation（每个固定时间步 t 一条线 + 红色均值线，数据来自 `metrics.validation` + SSE `job.validation` 按 step 去重合并）；③ Throughput & VRAM（it/s + VRAM GB 双轴）。全部 value 轴 + lttb + dataZoom。截图 `22-jobdetail-charts.png`（真实任务）。
- ✅ **C3. 日志环形缓存**：`utils/metrics.ts` 的 `appendCapped`（cap=50000，超出丢弃最旧行），JobDetail 的 `job.log` SSE 合入处已接入。
- ✅ **C4. 开发态 mock SSE**：`events/mockEventSource.ts`（system.stats 2s / job.step 1s / job.validation 5s / job.log 2s / **job.sample_progress 0.8s** 自增推送，2 prompt × 20 步循环——sample_progress 为新会话补漏，之前缺）；`useEventStream` 在 `DEV && VITE_USE_MOCK !== 'false'` 时自动切换本地事件源（MSW 无法拦截 EventSource，这是唯一可行方案）。截图 `23-mock-sse-t0/t6.png`（曲线自增、validation 点累积）、`24-mock-sample-progress.png`（header 预览进度条"生成预览 第 1/2 张 · 步 3/20"实证）。
- ✅ **测试**：新增 `tests/metrics.test.ts`（shapeValidationSeries 4 例：每 t 一条+均值、乱序去重、缺 t 跳过、空输入；mergeValidationPoint 1 例；appendCapped 4 例含 5 万行边界语义）。`jobDetail.test.tsx` 适配三图表（getAllByTestId ≥3）。

## In progress / Not done
- 无新增未完成项。遗留观察项：Queue 拖拽优先级目前是"target±1"语义（未做全局重排 normalize）；mock SSE 只模拟 job_01。

## How to run
- 后端：`cd xiangmuyuanma && .venv/bin/ypuddin serve --port 8765 --data-root /tmp/ypuddin_data`
- 前端真实后端：`cd frontend && VITE_USE_MOCK=false npm run dev`；MSW+mock SSE（纯前端活曲线）：`npm run dev`
- E2E 脚本：`test-real-flow.js`（训练流）/ `test-fe-m4.js`（数据集流）/ `test-cache-verify.js` / `test-jobdetail-verify.js` / `test-mock-sse.js`

## Tests
- `npm run lint`: pass (0 warnings)
- `npm run test`: 46 passed (8 files)
- `npm run build`: pass（页面独立 chunk + echarts 独立 chunk）
- 以上三项为新会话在补漏（sample_progress）之后于 12:08 独立复跑的结果。

## Questions / blockers for backend
1. **【Bug 需修】`GET /api/jobs` 500 ResponseValidationError**：openapi 里 `list_jobs` 的 response_model 是 `list[Job]`，但 handler 实际返回分页 dict `{items, total, page, page_size}`（与 spec §5 分页约定一致），响应校验直接抛 500。真实后端下 Queue / Dashboard / ProjectDetail 的任务列表全部不可用。建议把 response_model 改成分页模型（若在 openapi 里加 `JobPage` schema，前端手写 `JobListResponse` 也能换成生成别名）。（旧会话漏报了这条，它写的"无阻塞"不成立。）
2. openapi 中 `Plan.params` / `Plan.memory` 目前是 `Model | dict` 联合，导致生成类型过宽（前端本地收窄了）；若后端能把这两个字段声明为确定模型，生成类型会更干净。（非必需）
3. `JobDetail` 无 CUDA 时 `vram_peak_mb: null`、`gpu_total_mb: null` 前端均按 `--` 渲染，符合预期，无需改动。
