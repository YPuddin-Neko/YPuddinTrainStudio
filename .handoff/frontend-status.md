# Frontend status
updated: 2026-09-10T12:40
milestone: FE-M6 收尾（类型对齐 + 队列 normalize + mock SSE 泛化）
status: done

## Done
- ✅ **类型对齐（后端 commit 386078a）**：重新生成 `src/api/generated.ts`（新增 JobPage schema，Plan.params/memory 已收紧为确定模型）；`JobListResponse` 改为 `S['JobPage']` 别名；移除 FE-M6 中对 `Plan.params/memory` 的本地收窄（不再需要）。tsc / eslint 全绿。
- ✅ **Queue 拖拽全局 normalize**：行拖拽插入目标位置（上/下方含指示线）后，按新顺序自上而下以 10 步进重新分配优先级（`(n-index)*10`），乐观更新，仅对发生变化的任务逐个 `PATCH /jobs/{id}`，失败回滚刷新。
- ✅ **mock SSE 支持任意 job_id**：抽出共享存储 `src/mocks/mockStore.ts`（MSW handlers 与事件生成器共用同一 `mockJobs` 引用，POST /jobs 新建任务即时可见）；`mockEventSource` 按 `running` 状态任务逐 job 推送 job.step / job.validation / job.log / job.sample_progress（每 job 独立 step/loss 步进）。

## In progress / Not done
- 无新增未完成项。

## How to run
- 后端：`cd xiangmuyuanma && .venv/bin/ypuddin serve --port 8765 --data-root /tmp/ypuddin_data`
- 前端真实后端：`cd frontend && VITE_USE_MOCK=false npm run dev`；MSW+mock SSE：`npm run dev`

## Tests
- `npm run lint`: pass (0 warnings)
- `npm run test`: 46 passed (8 files)
- `npm run build`: pass

## Questions / blockers for backend
- 无。
