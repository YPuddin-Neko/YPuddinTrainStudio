# 开发说明

后端位于 `ypuddin/`，前端位于 `frontend/`。先按 [安装与启动](guide/install.md) 准备当前平台的环境。

## 本地开发

在仓库根目录运行对应启动入口的 `dev` 子命令，例如 CPU 环境：

```bash
./studio-cpu.sh dev --no-browser
```

后端默认监听 `127.0.0.1:8123`，Vite 默认监听 `127.0.0.1:3000`。前端通过代理连接真实后端。用 `--data-root <独立目录>` 隔离开发数据，避免操作日常项目。

激活环境的方法可以通过启动入口的 `shell` 子命令查看。前端检查命令在 `frontend/` 中执行：

```bash
npm ci
npm run lint
npm run build
```

Python 代码检查需要在当前虚拟环境安装开发依赖后执行：

```bash
python -m pip install -e '.[dev]'
python -m ruff check ypuddin scripts
```

## 本地测试

测试源码、测试数据和测试记录保存在开发机的 `Test/`，不随 Git 仓库或源码包分发。产品构建不依赖该目录。

已有本地测试目录时，可运行对应启动入口的 `test` 子命令，或在仓库根目录运行：

```bash
python -m pytest
```

前端测试在 `frontend/` 中运行：

```bash
npx --no-install vitest run --config ../Test/frontend/vitest.config.ts
```

## 源码打包

在已激活的环境中，先构建前端，再从仓库根目录检查 Git 索引并生成包：

```bash
python scripts/package_source.py --check-git
python scripts/package_source.py /path/to/YPuddinTrainStudio-source.zip
```

打包包含源码、使用文档、许可证和当前前端构建。脚本会检查构建指纹，排除本地测试、环境、模型权重、运行数据和凭据，并生成文件哈希清单；输出文件已存在时拒绝覆盖。

`--check-git` 同时检查已跟踪文件。仅添加忽略规则不能移除已进入 Git 索引的数据；需要保留本地文件时使用 `git rm --cached <路径>`，不要删除实际数据。
