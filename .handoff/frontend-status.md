# Frontend status
updated: 2026-09-10T07:48
milestone: FE-M2 & FE-M3 (Full Spec Implementation)
status: done

## Done
- ✅ **A. SchemaForm 专用控件与动态校验 (FE-M2)**:
  - 完美支持 150 个字段的真实 Schema，深度解析 `$ref` 与嵌套子对象。
  - 实现全套专属控件：
    - `control=rules`：可增删/排序/配置规则的 `AdapterRule` 列表（match, algo, rank, alpha, factor, lr）。
    - `control=prompts`：可增删的多维度 `SamplePrompt` 列表。
    - `dataset.sources / validation.sources`：完整的数据源列表控件（包含 repeats, caption_ext, is_reg, prior_weight 等字段）。
    - `control=path`：路径输入框 + "浏览"弹窗（调用 `GET /api/fs/list?path=`，支持层级穿梭与选择）。
    - `additionalProperties` 对象：通用的键值对编辑器 `KeyValueEditor`（支持自由添加属性与即时修改）。
    - `optimizer.betas`：双值浮点滑块输入。
  - 接入 `POST /config/validate` 校验错误定位与高亮（`errors[{loc, msg}]`），并在 Plan 面板展示 warnings 与显存优化建议。
  - 编写了完整的 `tests/schemaForm.test.tsx` 测试，验证叶子节点渲染、高级选项开关、show_when 联动及专用控件交互。
- ✅ **B. 任务详情 /jobs/:id 实时监控 (FE-M3)**:
  - 头部指标（step/total、ETA、速度、显存峰值）与阶段时间线（`preparing → caching → training → finalizing`）。
  - ECharts 图表（支持 Loss 原始与 EMA 平滑切换、Grad Norm、VRAM、速度等多轴折线，集成 LTTB 与 dataZoom 保证海量点流畅渲染）。
  - 集成 Samples 画廊、检查点（Checkpoints）列表、日志视图（Log 流，支持级别过滤与跟随底部）、配置快照只读视图。
  - 编写了 `tests/jobDetail.test.tsx` 测试验证。
- ✅ **C. 队列 /queue 与 Dashboard (FE-M3)**:
  - 队列表格包含全字段列、动作操作（pause/resume/cancel/save/retry/delete）与乐观状态（`pausing...` 等）。
  - 支持全局暂停调度开关（`GET/PUT /api/queue/settings`）与优先级即时调整（`PATCH /api/jobs/{id}`）。
  - Dashboard 具备系统状态条（GPU/CPU/RAM/Disk）、正在运行任务大卡片、队列摘要与最近产物。
  - 编写了 `tests/queue.test.tsx` 与 `tests/dashboard.test.tsx` 测试。
- ✅ **D. MSW Handlers 完善**:
  - 覆盖 `/api/system/stats`、`/api/projects`、`/api/jobs`、`/api/jobs/:id/metrics`、`/api/jobs/:id/samples`、`/api/jobs/:id/checkpoints`、`/api/jobs/:id/log`、`/api/fs/list`、`/api/queue/settings`、`/api/config/validate`、`/api/plan`、`/api/presets` 等全部接口。

## In progress / Not done
- 暂无未完成项。所有 A1~A3, B1~B4, C1~C3, D 要求已全部实现并由测试证明。

## How to run
- `cd frontend && npm i && npm run dev`
- 浏览器访问 `http://localhost:3000`

## Tests
- `npm run lint`: pass (0 warnings, 0 errors)
- `npm run test`: 30 passed (5 test files)
- `npm run build`: pass (tsc -b + vite build 成功)

## Questions / blockers for backend
- 暂无 blocker。
