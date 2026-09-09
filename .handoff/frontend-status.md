# Frontend status
updated: 2026-09-10T07:36
milestone: FE-M2 & FE-M3
status: done
## Done
- ✅ **FE-M2 (Schema 表单引擎)**:
  - 成功支持 150 个字段真实 Pydantic Schema（`train-schema.example.json`），解析 `$ref` 与嵌套子对象无崩溃。
  - 支持 `x-ui-groups` 顶级折叠和层级渲染，实现按 group/order 排序，`advanced` 开关，以及 `show_when` 动态逻辑驱动。
  - 完成 `showWhen.ts` 表达式引擎，支持 `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `&&`, `||`, `!`, 括号与 `null` 比较。单测已扩充至 22 条（全部通过）。
  - 支持预设（Preset）选择与加载、Plan 面板防抖与显存/步数估算联动，提供入队动作支持。
- ✅ **FE-M3 (监控与队列)**:
  - 队列页展示实时任务状态列表，支持任务类型、进度与优先级展示。
  - 任务监控详情页集成 ECharts 针对海量训练数据的性能优化：配置 `sampling: 'lttb'` 与 `dataZoom` 滑块/缩放，可平滑承载 10 万点实时训练 loss 曲线渲染。
  - 支持 SSE `job.step` 增量数据流无缝接入图表。

## How to run
- `cd frontend && npm i && npm run dev`
- 浏览器访问 `http://localhost:3000`，开发环境默认通过 MSW 模拟真实后端 API 与预设数据。

## Tests
- `npm run lint`: pass (0 warnings/errors)
- `npm run test`: 22 passed (Vitest)
- `npm run build`: pass

## Questions / blockers for backend
- 暂无 blocker。随真实后端 API 逐步就绪，我们将通过 openapi 生成正式的类型文件并对接实时接口。
