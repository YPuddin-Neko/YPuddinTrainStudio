# 目录结构

训练器有两棵独立的目录树：**源码目录**跟着 Git 更新；**数据目录**保存项目、模型和训练结果，更新代码时不会被迁移或覆盖。两边各有一个 `environment` 目录，用途不同，见 [两个 environment](#两个-environment)。

## 源码目录

```text
YPuddinTrainStudio/
├── studio-windows-cuda.bat      Windows + NVIDIA CUDA，x86_64
├── studio-cpu.bat               Windows 仅 CPU，x86_64 / arm64
├── studio-linux-cuda.sh         Linux + NVIDIA CUDA，x86_64
├── studio-linux-dtk.sh          Linux + 海光 DTK，x86_64
├── studio-macos.command         macOS Apple 芯片（MPS）
├── studio-cpu.sh                Linux / macOS 仅 CPU
├── scripts/                     启动器共用逻辑（找 Python、建环境、装依赖、构建前端、启动服务）和打包脚本
├── ypuddin/                     Python 后端：训练核心、命令行、Web 服务
├── frontend/
│   ├── src/                     前端源码
│   └── dist/                    前端构建结果，不进 Git
├── environment/<平台>/venv/     各平台的 Python 环境
├── docs/                        文档
└── studio_data/                 默认数据目录，可用 --data-root 换到别处
```

`frontend/dist/` 不在 Git 里。`git pull` 之后重新运行启动脚本，启动器会比对前端源码的指纹，过期时自动执行 `npm ci` 和 `npm run build`。构建需要 Node.js 20.19+ 或 22.12+；没有 Node 且页面与源码不一致时会直接报错，不会继续使用旧页面。

## 数据目录

默认 `studio_data/`，用 `--data-root` 或 **设置 → 存储路径** 更换。

```text
studio_data/
├── studio.db                    SQLite 主库：项目、版本、数据集、任务、产物、模型登记
├── settings.json                设置中保存的内容
├── secrets.json                 模型站点的访问密钥（明文，见下文）
├── project/<项目 ID>/
│   └── v<N>/                    版本目录：v1、v2、v3…
│       ├── config.json          本版本的训练配置
│       ├── traindata/           训练图片、标签和遮罩（本版本的独立副本）
│       ├── reg/                 正则图片，按批次分目录
│       ├── cache/               本版本任务共享的编码缓存
│       ├── samples/<任务 ID>/   每次训练的预览图
│       └── output/<任务 ID>/    每次训练的配置快照、事件、日志、权重和完整断点
├── projects/<项目 ID>/          旧版本创建的项目（旧布局，继续可用）
├── cache/                       图片索引、无项目任务的共享缓存、模型测试指纹缓存
├── models/                      默认模型目录
├── runs/                        不属于任何项目的任务
├── presets/                     保存的参数预设
├── datasets/                    数据集索引
├── thumbs/                      缩略图缓存
├── network/                     网络代理的凭据
└── environment/                 界面准备的其他 PyTorch 版本和服务控制文件
```

- 上传和导入的图片保存为当前版本的独立副本，编辑标签和遮罩只改副本。通过数据来源或 TOML 直接引用的外部目录使用原文件。
- 新版本可以复制图片、标签、遮罩和验证数据，也可以只继承参数或使用空白配置；任务、预览和训练结果不会复制。
- 任务保存的是配置快照，不是图片快照。要修改旧版本的数据，建议先复制出一个新版本。
- 模型下载过程中会出现 `models/.downloads/<任务 ID>/` 暂存目录，完成或取消后清理。
- `secrets.json` 没有加密。备份时一起保存，分享源码或打包时排除，不要放进项目 TOML、日志、截图或公开的问题报告里。

## 两个 environment

| 路径 | 由谁创建 | 内容 |
| --- | --- | --- |
| `<源码目录>/environment/` | 启动脚本 | 各平台的基础 Python 环境，跟着源码目录 |
| `<数据目录>/environment/` | 运行中的服务 | 界面里准备的其他 PyTorch 版本、重启请求等服务控制文件 |

数据目录里的结构：

```text
<数据目录>/environment/<平台>/
├── runtimes/<操作 ID>/     界面准备的其他 PyTorch 版本（完整的虚拟环境）
├── service/
│   ├── selected.json       下次启动使用哪个环境
│   └── restart-*.json      重启请求
├── cache/                  安装器下载缓存
└── installer/              安装器工作文件
```

旧版本装在 `environment/profiles/<平台>/` 下。启动器发现这个旧目录时会打印路径，并照常安装到新位置，不会自动删除；确认新环境可用后可以自行删除。

## 路径设置何时生效

- **缓存目录**（`paths.cache_dir`）和 **自定义输出目录**（`paths.output_mode = custom` 时的 `paths.output_dir`）只影响新建的任务，已有任务继续使用原来的缓存、权重和断点位置。自定义缓存目录还存放扫描索引、缩略图和按平台分开的软件包缓存；自定义输出按 `<项目 ID>/vN/<任务 ID>` 分目录。
- **模型目录**（`paths.models_dir`）是默认的模型扫描和下载位置。
- **基础环境目录**（`paths.bootstrap_env_dir`）在下次从启动脚本启动时生效，见 [运行环境](environment.md)。
- **数据目录** 修改后在服务重启时切换，已有数据不会自动迁移，需要先自己复制。
- **监听地址和端口**（`server.host`／`server.port`）保存后点击“重启服务”生效；下次通过启动脚本启动时，显式传入的 `--host`／`--port` 优先。

数据和模型路径建议填写绝对路径，相对路径按服务的工作目录解析。

## 更新与备份

- **跟着 Git 更新**：源码目录中除 `environment/`、`venv/`、`frontend/dist/`、`studio_data/` 以外的内容。
- **不随更新迁移**：数据目录的全部内容，以及已经建好的环境。
- **需要备份**：`studio.db`、`settings.json`、`secrets.json`、`project/`（旧布局为 `projects/`）、`presets/`、自定义输出目录，以及外部原始数据集。模型权重另行保存或记录下载来源。
- **可以删除、会自动重建**：`frontend/dist/`、`cache/`、`thumbs/`。删除 `environment/` 会导致下次启动重新安装环境。

完整断点不能用推理权重代替；只备份导出的权重无法精确续训。
