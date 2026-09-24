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

后端检查在已激活的项目环境中执行：

```bash
python -m pip install -e '.[dev]'
python -m ruff check ypuddin scripts
```

## 命令行

| 命令 | 功能 |
| --- | --- |
| `ypuddin validate config.toml` | 校验配置 |
| `ypuddin plan config.toml` | 检查数据、尺寸、步数与资源估算 |
| `ypuddin train config.toml` | 执行训练 |
| `ypuddin cache config.toml` | 预编码图像和文本缓存 |
| `ypuddin schema` | 输出配置 JSON Schema |
| `ypuddin inspect weights.safetensors` | 查看权重元数据与模块 |
| `ypuddin serve --port 8123 --data-root /data/studio` | 启动 Web 服务 |

配置支持 TOML 和 JSON，`--preset` 叠加预设文件，`--set a.b=c` 覆盖单个字段。训练、缓存和计划命令支持 `--device`。

权重转换、合并和提取示例：

```bash
ypuddin convert input.safetensors --to comfyui --family anima -o output.safetensors
ypuddin merge --base base.safetensors --adapter lora.safetensors --family anima -o merged.safetensors
ypuddin extract --base base.safetensors --tuned tuned.safetensors --algo lokr --family anima -o extracted.safetensors
```

## 接口与配置定义

配置模型位于 [`ypuddin/config/schema.py`](../ypuddin/config/schema.py)。后端根据字段约束、模型能力和运行平台检查请求，前端读取生成的 Schema 渲染参数表单。

更新公开定义：

```bash
python scripts/export_api.py
```

输出 `docs/api/openapi.json`、`docs/api/train-schema.example.json`，并同步 `frontend/src/schema/train-schema.json`。前端 API 类型位于 `frontend/src/api/generated.ts`；更新接口后需同步类型并执行前端构建。

## 源码打包

先构建前端，再从源码根目录执行：

```bash
python scripts/package_source.py --check-git
python scripts/package_source.py /path/to/YPuddinTrainStudio-source.zip
```

打包脚本校验 Git 索引及前端构建指纹，包含程序、公开文档、许可证和编译后的前端；排除 Python 环境、模型、运行数据、凭据与本地产物。归档附带文件大小及 SHA-256 清单，已存在的目标文件不会被覆盖。

## 第三方代码

模型后端的第三方源码、资源和依赖来源记录在各模块的 `NOTICE.md` 及许可证文件中。修改这些模块时保留来源与许可声明，避免将运行权重纳入源码包。
