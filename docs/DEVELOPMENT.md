# 开发指南

## 开发环境

先使用当前平台的启动脚本完成依赖安装。开发服务可使用独立数据目录：

```bash
./studio-cpu.sh dev --data-root /path/to/development-data --no-browser
```

后端默认端口为 8123，Vite 默认端口为 3000。前端代理连接实际后端；`--fe-port` 可更改前端端口。平台环境的激活命令由启动入口的 `shell` 子命令输出。

前端命令在 `frontend/` 中执行：

```bash
npm ci
npm run dev
npm run lint
npm run build
```

启动脚本构建前端，以及运行 `dev`、`test` 前，先检查 Node.js 版本和前端依赖：依赖清单、Node/npm 环境未变且安装完整时复用已安装依赖，否则重新安装。源码或构建产物变化时重新构建；构建失败时保留已安装依赖。

后端检查在已激活的项目环境中执行：

```bash
python -m pip install -e '.[dev]'
python -m ruff check ypuddin scripts
```

启动入口的 `test` 子命令安装开发依赖后运行本机 `Test/` 中的后端（pytest）与前端（Vitest）测试，该目录不随仓库发布。

## 命令行

| 命令 | 功能 |
| --- | --- |
| `ypuddin validate config.toml` | 校验配置 |
| `ypuddin plan config.toml` | 检查数据、尺寸、步数与资源估算 |
| `ypuddin train config.toml` | 执行训练 |
| `ypuddin cache config.toml` | 预编码图像和文本缓存 |
| `ypuddin smoke config.toml` | 用真实权重训练几步、生成一次预览并保存后重新加载权重，报告写入 `outputs/smoke/smoke-report.json` |
| `ypuddin schema` | 输出配置 JSON Schema |
| `ypuddin inspect weights.safetensors` | 查看权重元数据、模块数与参数量；`ypuddin -v inspect …` 另列出各模块 |
| `ypuddin serve --port 8123 --data-root /data/studio` | 启动 Web 服务 |

配置支持 TOML 和 JSON，`--preset` 叠加预设文件，`--set a.b=c` 覆盖单个字段。训练、缓存、计划和自检命令支持 `--device`。`smoke` 未设数据源时使用合成图片，`--out` 更改报告目录；启动入口的 `smoke` 子命令转发到该命令。

权重转换、合并和提取示例：

```bash
ypuddin convert input.safetensors --to comfyui --family anima -o output.safetensors
ypuddin merge --base base.safetensors --adapter lora.safetensors --family anima -o merged.safetensors
ypuddin extract --base base.safetensors --tuned tuned.safetensors --algo lokr --family anima -o extracted.safetensors
```

## 接口与配置定义

配置模型位于 [`ypuddin/config/schema.py`](../ypuddin/config/schema.py)。后端根据字段约束、模型能力和运行平台检查请求。参数页运行时读取 `/api/schema/train`；字段名称与选项在 `frontend/src/utils/configPresentation.ts`，说明在 `fieldCopy.ts`，随模型与机器变化的内容在 `fieldContext.ts`，分组在 `frontend/src/schema/SchemaForm/ParameterFields.tsx`。

配置校验失败的每项错误包含 `loc`（字段路径）、`msg`（pydantic 原文）、`type`（错误类型，如 `less_than`）和 `ctx`（界限等简单取值，如 `{"lt": 1}`）；训练计划自身的检查只给出 `loc` 与 `msg`。请求体校验失败返回 422，错误列表位于 `error.details.errors`。前端在 `frontend/src/utils/validationMessages.ts` 中按 `type` 与 `ctx` 以字段单位生成提示。

更新公开定义时，在已激活的项目环境中从源码根目录执行：

```bash
python scripts/export_api.py
./frontend/node_modules/.bin/openapi-typescript docs/api/openapi.json -o frontend/src/api/generated.ts
```

第一条命令输出 `docs/api/openapi.json`、`docs/api/train-schema.example.json`，并同步只供本地测试读取的 `frontend/src/schema/train-schema.json`；第二条命令据此重新生成前端 API 类型 `frontend/src/api/generated.ts`。更新接口后执行前端构建。

## 源码打包

先构建前端，再从源码根目录执行：

```bash
python scripts/package_source.py --check-git
python scripts/package_source.py /path/to/YPuddinTrainStudio-source.zip
```

打包脚本校验 Git 索引及前端构建指纹，包含程序、公开文档、许可证和编译后的前端；排除 Python 环境、模型、运行数据、凭据与本地产物。归档的 `SOURCE_MANIFEST.json` 包含文件大小、SHA-256 清单，以及版本、提交、分支、打包时间和源码修改状态，用于压缩包安装的版本识别。已存在的目标文件不会被覆盖。

## 第三方代码

第三方源码、资源和依赖来源记录在所在目录的说明及许可证文件中：模型后端在 `ypuddin/models/` 各系列目录及其 `vendor/`、`assets/` 的 `NOTICE.md`，Automagic 优化器在 `ypuddin/optim/NOTICE.md`，ER-SDE 采样器在 `ypuddin/sampling/ER_SDE_PROVENANCE.md` 与 `ER_SDE_LICENSE.txt`。修改这些模块时保留来源与许可声明，避免将运行权重纳入源码包。
