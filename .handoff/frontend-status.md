# Frontend status
updated: 2026-09-10T07:53
milestone: FE-M2 & FE-M3 (Parser Refactor & Code-Splitting)
status: done

## Done
- ✅ **1. 重构 showWhen 解释器（1:1 同构后端实现，去除任何动态执行风险）**:
  - 彻底移除了 `new Function` / `eval`，通过词法分析（`tokenize`）+ 递归下降解析（`Parser`）构建 AST，严格按照语法规则（`or -> and -> not -> cmp -> atom`）求值。
  - 严格支持单双引号字符串、数字、`true`/`false`/`null`、数组 `[...]`、括号表达式 `(...)`、`in` 包含运算符、以及 `==`, `!=`, `<`, `<=`, `>`, `>=`。
  - 对齐语义：缺失路径与非存在属性读作 `null`；严格相等判断；不等式比较若任一侧为 `null` 均返回 `false`。
  - 解析错误抛出 `ShowWhenError`，由 `SchemaForm` 组件捕获并降级输出错误日志，避免吞掉语法异常。
  - 完整迁移了 `ypuddin/tests/unit/test_config.py` 中 `test_show_when` 的 11 个核心测试用例，并补齐了非法语法抛错测试。
- ✅ **2. 路由懒加载与代码拆分 (Code-Splitting)**:
  - 使用 `React.lazy` 对所有页面（`Dashboard`, `Projects`, `ProjectDetail`, `TrainConfig`, `Queue`, `JobDetail`, `Artifacts`, `Models`, `Settings`）进行按需加载拆分。
  - 在 `vite.config.ts` 中将 `echarts` 单独打包（`echarts-DQf7QbRo.js`），各页面 chunk 维持在极小体积（大多数 < 15 kB）。
- ✅ **3. 专用控件与全量测试体系**:
  - `SchemaForm`（150 字段、`rules`, `prompts`, `sources`, `key-value`, `betas`, `path` 浏览与高亮定位）及全部专用控件测试通过。
  - `JobDetail`（头部、时间线、ECharts 曲线、画廊、检查点、日志、配置快照）测试通过。
  - `Queue`（全字段列表、动作与乐观状态、全局调度开关、优先级调整）与 `Dashboard` 测试通过。

## In progress / Not done
- 待接收后端真实的 `docs/api/openapi.json`，以便运行 `openapi-typescript` 自动化生成严格的后端类型定义并去除临时手写类型。

## How to run
- `cd frontend && npm i && npm run dev`
- 访问 `http://localhost:3000`（MSW 开发 Mock 环境全量就绪）

## Tests
- `npm run lint`: pass (0 warnings, 0 errors)
- `npm run test`: 23 passed (5 test files)
- `npm run build`: pass (路由代码拆分生效，echarts 独立 chunk)

## Questions / blockers for backend
- 暂无 blocker，等待后端 OpenAPI 导出文件。
