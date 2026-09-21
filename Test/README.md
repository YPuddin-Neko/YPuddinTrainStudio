# 测试工作区

项目的研发测试集中在本目录。用户使用文档保留在 `docs/`，产品代码位于 `ypuddin/` 和 `frontend/src/`。

| 目录 | 内容 | 提交 Git |
| --- | --- | --- |
| `tests/` | Python 单元、集成、端到端测试及多进程 worker | 是 |
| `frontend/tests/` | 前端组件和交互回归测试 | 是 |
| `frontend/mocks/`、`frontend/public/` | 开发模式的模拟数据、事件流和 MSW worker | 是 |
| `frontend/browser/` | 浏览器流程与截图脚本 | 是 |
| `scripts/` | GPU 严格续训、分桶对照等验收工具 | 是 |
| `reports/`、`validation/` | 本地测试报告和机器可读验收记录 | 否 |
| `screenshots/`、`artifacts/` | 截图、测试缓存和临时产物 | 否 |
| `remote-testing/` | 历史远程验收工作目录、原始证据和本地连接材料 | 否 |

## Python

在仓库根目录运行：

```bash
python -m pytest
python -m pytest Test/tests/unit/test_repository_content.py
```

使用项目虚拟环境时，将 `python` 换成相应环境的解释器。`pyproject.toml` 已将默认测试目录设为 `Test/tests`，pytest 缓存写入 `Test/artifacts/pytest-cache`。GPU 与多进程用例仍需要对应硬件和运行权限。

严格续训验收入口：

```bash
python Test/scripts/verify_frozen_adapter_resume.py --help
```

## 前端

先在 `frontend/` 安装依赖，然后从仓库根目录运行：

```bash
npm --prefix frontend test
npm --prefix frontend run test:watch
npm --prefix frontend run build
npm --prefix frontend run lint
```

Vitest 配置位于 `Test/frontend/vitest.config.ts`，复用 `frontend/node_modules`，不需要额外安装一套依赖。浏览器脚本也复用该环境；脚本中的浏览器路径、服务地址和测试数据需按实际机器配置。截图输出统一写入 `Test/screenshots/frontend/`。

开发模拟模式仍通过 `VITE_USE_MOCK=true` 启用，MSW worker 只由 Vite 开发服务提供；生产构建不复制该 worker。

## 本地记录

报告和截图已从旧位置迁入本目录，本地内容保留，Git 不再跟踪。旧报告与 `remote-testing/` 下的历史脚本可能记录迁移前的绝对路径；它们作为历史证据保留原文，不保证直接重放。新的验收应使用本目录维护的测试入口。

软件自带的 `doctor`／`smoke` 诊断命令和“模型测试”页面属于产品功能，仍保留在产品源码中。
