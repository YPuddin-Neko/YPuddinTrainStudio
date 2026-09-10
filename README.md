# YPuddin Train Studio

一个面向扩散模型（首发支持 **Anima**：Cosmos-Predict2 风格 DiT + Qwen3-0.6B + Qwen-Image VAE）的
**模块化 LoRA / LoKr 训练器**，附带供 Web 前端调用的服务 API。

它不是现有训练脚本的 GUI 套壳：训练核心、适配器（LoRA / LoKr / LoHa / DoRA）、数据流水线、
显存编排与任务服务全部自研，设计目标是同时超越
[sd-scripts](../sd-scripts)、[diffusion-pipe](../diffusion-pipe) 与 [AnimaLoraStudio](../AnimaLoraStudio)
各自的长处（详见 [`docs/design/00-architecture.md`](docs/design/00-architecture.md)）。

## 仓库结构

```
studio.sh/.bat    一键安装 + 启动
ypuddin/          Python 后端包（训练核心 + 服务 API），CLI 入口 `ypuddin`
frontend/         Web 前端（由独立会话负责，见 docs/frontend-spec.md）
docs/deploy.md    部署与运行指南
docs/design/      架构设计文档（ADR 风格）
docs/reference/   对四个参考项目的深度分析报告
tests/            CPU 可跑的单元 / 集成测试（用玩具模型族端到端验证训练循环）
scripts/          bootstrap.py（studio.sh/.bat 的实现）与开发辅助脚本
.handoff/         与前端会话交接用的状态文件
```

## 快速开始

```bash
./studio.sh        # Linux / macOS
studio.bat         # Windows
```

第一次运行自动创建 `venv`、按显卡驱动安装对应 CUDA 版 PyTorch、构建前端并启动服务，浏览器打开 `http://127.0.0.1:8765/`。
`./studio.sh doctor` 查看本机环境；`./studio.sh smoke --set model.dit_path=… --set model.text_encoder_path=… --set model.vae_path=…`
用真实权重自检整条训练链路。完整说明（参数、手动安装、目录结构、常驻服务、远程访问、排障）见 [`docs/deploy.md`](docs/deploy.md)。

## 开发环境

```bash
uv venv --python 3.12 venv
uv pip install --python venv/bin/python -e ".[dev,models,server]"
venv/bin/pytest                      # 全部 CPU 测试
./studio.sh dev                       # 后端 + Vite 热更新前端
```

CUDA 机器上额外安装 `.[cuda,optim]`。

## 许可证

本仓库代码以 **Apache-2.0** 发布。参考实现中 sd-scripts、LyCORIS、musubi-tuner 为 Apache-2.0，
AnimaLoraStudio 与 diffusion-pipe 为 GPL-3.0（含 ComfyUI 派生代码）——本项目**不复制** GPL 代码，
只在阅读理解后独立实现，以保持宽松许可。模型权重（Anima / Qwen3 / VAE）各有其自身条款。
