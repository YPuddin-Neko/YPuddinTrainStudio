# YPuddin Train Studio

一个面向扩散模型的模块化适配器训练器，包含 LoRA / LoKr / LoHa / DoRA、数据缓存、任务队列、服务 API 与 Web 界面。已接入 **Anima、Krea 2、SDXL、FLUX.2 Klein base 4B/9B**。训练循环统一，模型的条件编码、目标函数、采样和内存能力分别实现。

源码版本 **v0.5.9**，2026-09-13 新增模型接入说明见 [SDXL / FLUX 支持范围与验收](docs/MODEL_FAMILIES_2026-09-13.md)。SDXL 默认下载为光辉 Illustrious-XL v0.1，也可手选其他本地 SDXL base 模型，并明确选择 ε / v 预测。FLUX 仅保留 Klein 基础版 4B/9B；旧 FLUX.1 / dev 配置和数据仍保留，但不再提供新建或训练入口。完整目录与独立组件均按真实结构检查；Klein 蒸馏版和图像编辑模型不在本次接入范围。

Anima、SDXL 光辉 v0.1、Klein base 4B、Krea2 Raw 已完成所列配置的 Windows 正式权重短训、保存、采样与复载；详情和失败边界见 [2026-09-14 Windows 验收](docs/WINDOWS_ACCEPTANCE_2026-09-14.md)。这些结果不代表长训收敛、同条件性能排名、完整 ComfyUI 画面或所有参数组合通过。Klein 9B 尚无对应正式验收，不能用 4B 单卡结果代替。

海光 Linux 已提供独立 DTK 启动入口。DTK 26.04 / 厂商 Torch 2.7.1 环境完成了 Krea2 Raw BF16、512 分辨率 LoKr 的单卡与双卡 DDP 训练验证，以及四个模型族的 20 项微型权重单卡测试；正式训练矩阵、精确恢复、安装流程和待完成项目见 [2026-09-14 DTK 验收](docs/DTK_ACCEPTANCE_2026-09-14.md)。微型权重测试不代表四个模型族的正式大模型均已验收，海光 DDP 结果也不代表 Windows DDP 通过。

界面提供项目概览、独立版本、数据导入、标签与遮罩编辑、训练配置、任务指标和产物管理。最新 UI 流程见 [2026-09-13 工作区验收](docs/UI_WORKFLOWS_2026-09-13.md)；[训练流程、分桶与三个训练器的对比](docs/TRAINING_PARAMETERS.md)、[JSON 标签](docs/JSON_CAPTIONS.md)、[整图保留与原生分辨率](docs/native-resolution.md)有对应说明。Krea2 的加载、缓存和卸载流程见 [显存说明](docs/KREA2_MEMORY.md)。历史报告中的测试数量、服务路径和“未接入模型”仅代表当时状态。

项目按 `project/<id>/vN/` 保存训练图、正则图、缓存、采样与产物，支持自定义输出根；已有数据不随代码更新自动迁移。正则图支持本地底模生成、网站收集与已有图片导入；标签页编辑已有 caption，不提供 WD14 自动打标。参考代码分析位于 `docs/reference/`，不能据此宣称性能优于参考实现。

## 仓库结构

```
studio.sh/.bat    一键安装 + 启动
ypuddin/          Python 后端包（训练核心 + 服务 API），CLI 入口 `ypuddin`
frontend/         Web 前端（动态配置表单、数据集、任务监控与产物管理）
docs/deploy.md    部署与运行指南
docs/design/      架构设计文档（ADR 风格）
docs/reference/   对四个参考项目的深度分析报告
tests/            CPU 可跑的单元 / 集成测试（用玩具模型族端到端验证训练循环）
scripts/          bootstrap.py（studio.sh/.bat 的实现）与开发辅助脚本
HANDOVER.md       当前交接说明与验证边界
```

## 快速开始

```bash
./studio.sh        # Linux / macOS
studio.bat         # Windows
```

首次部署按平台创建独立的 `environment/profiles/<平台>/venv`、选择 PyTorch 安装来源、构建前端并启动服务。已有根目录 `venv` 的旧部署继续使用原环境，不搬移或重建。明确选择平台可用 `studio-windows-cuda.bat`、`studio-linux-cuda.sh`、`studio-linux-dtk.sh`、`studio-macos.command` 或 `studio-cpu.bat/.sh`；这些入口不共用依赖。海光需要匹配的 DTK 用户态运行库与厂商 wheel，准备步骤见 [DTK 独立环境](docs/RUNTIME_DTK.md)；启动器不会安装系统驱动或用 CUDA 包代替。默认地址为 `http://127.0.0.1:8765/`。前端构建需要 Node 20.19+ 或 22.12+；MPS 当前按 FP32 执行，内存预算按统一内存估算。目录、升级与切换边界见 [环境说明](docs/ENVIRONMENT_LIFECYCLE_2026-09-14.md)。
`./studio.sh doctor` 查看本机环境；`./studio.sh smoke --set model.dit_path=… --set model.text_encoder_path=… --set model.vae_path=…`
用真实权重自检整条训练链路。完整说明（参数、手动安装、目录结构、常驻服务、远程访问、排障）见 [`docs/deploy.md`](docs/deploy.md)。

升级前保留数据和断点。停止旧服务、更新源码后重新运行启动脚本，才会检查依赖与前端指纹；没有联网程序自更新或生产热更新。v0.4 将旧项目关联到兼容 v1，不移动旧数据、任务目录或断点。完整训练状态仍使用 v2 格式，旧指纹断点不能直接当作新版的精确续训状态使用。缓存依据实际权重、分词器和数据内容识别，修改 caption 或 mask 会影响数据指纹；详细兼容与迁移方式见部署指南。

## 开发环境

```bash
uv venv --python 3.12 venv
uv pip install --python venv/bin/python -e ".[dev,models,server,optim,logging]"
venv/bin/pytest                      # 全部 CPU 测试
./studio.sh dev                       # 后端 + Vite 热更新前端
```

CUDA 机器上额外安装 `.[cuda,optim]`。

前端开发默认连接真实服务；仅在开发模式显式设置 `VITE_USE_MOCK=true` 时启用演示数据和模拟 SSE。版本数据仍可编辑，高级 TOML 的外部路径也不会自动复制；需要保留实验数据时先创建完整数据副本的新版本。

## 许可证

本仓库代码以 **Apache-2.0** 发布。参考实现中 sd-scripts、LyCORIS、musubi-tuner 为 Apache-2.0，
AnimaLoraStudio 与 diffusion-pipe 为 GPL-3.0（含 ComfyUI 派生代码）——本项目**不复制** GPL 代码，
只在阅读理解后独立实现，以保持宽松许可。模型权重（Anima / Qwen3 / VAE）各有其自身条款。
