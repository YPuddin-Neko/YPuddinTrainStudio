# Frontend status
updated: 2026-09-10T07:30
milestone: FE-M1
status: done
## Done
- ✅ 脚手架、路由、布局（侧栏 + 顶栏）、i18n（中/英文）、主题（明/暗自动与手动切换）已完成。
- ✅ MSW mock 数据已接入核心接口（Projects, Jobs, Stats, Settings 等），保证无后端可运行。
- ✅ API 客户端（错误信封处理）、useEventStream（SSE 断线重连机制）已实现。
- ✅ FE-M2 阶段开启：Schema 表单引擎（`SchemaForm`）与 `show_when` 表达式解释器（`showWhen.ts`）基础版本已开发完成，已集成进 TrainConfig 页面并自带单测。

## How to run
- 开发环境启动：`cd frontend && npm i && npm run dev`
- 代理默认指向 `/api` -> `http://127.0.0.1:8765`（MSW 会在开发环境拦截，可通过环境变量 `VITE_API_BASE_URL` 绕过）

## Tests
- `npm run lint`: pass
- `npm run test`: 5 passed
- `npm run build`: pass

## Questions / blockers for backend
- 1. `docs/api/train-schema.example.json` 请尽快提供，以便我们验证 `SchemaForm` 对完整多层嵌套结构的兼容性。
- 2. SSE `useEventStream` 目前支持断线重连，需确认后端 `/api/events` 接口是否完全按照 `Last-Event-ID` header 或 query param 做状态续传（当前前端使用 query param `last_event_id` 传递）。
