# YPuddin Train Studio

一个面向扩散模型的**模块化适配器训练器**，包含 LoRA / LoKr / LoHa / DoRA、数据缓存、任务队列、服务 API 与 Web 界面。已接入 **Anima**（Cosmos-Predict2 风格 DiT + Qwen3-0.6B + Qwen-Image VAE）和 **Krea 2**（单流 MMDiT + Qwen3-VL-4B + Qwen-Image VAE）。训练核心与两族模型通过统一接口连接；模型组件包含按 Apache-2.0 引入的上游实现，来源见各 `vendor/NOTICE.md`。

当前处于集成验证阶段：已有 CPU 玩具模型与缩小版真实组件的训练、暂停恢复、采样和服务回归测试；**官方完整权重在 NVIDIA GPU 上的训练、实际 ComfyUI 加载与质量验收、速度和显存基准仍待完成**。本轮最终验收记录见[v0.4 项目版本报告](docs/UI_VERSIONS_2026-09-11.md)，功能状态见 [`docs/design/03-status.md`](docs/design/03-status.md)。参考项目分析保存在 `docs/reference/`，不作为性能优于参考实现的结论。

当前版本 **v0.4.0**：项目内管理独立实验版本，支持复制图片、标签、Mask 和验证集，或只继承参数、创建空白版本。数据、模型、训练参数、任务结果共用四步工作区；采样和权重按任务/版本归属查看。设置改为四类宽抽屉，训练产物回到项目结果；顶栏按 CPU、内存、GPU、硬盘显示真实读数。操作、迁移和验收边界见[项目版本说明](docs/UI_VERSIONS_2026-09-11.md)。

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

第一次运行自动创建 `venv`、选择 PyTorch 安装来源、有可用 Node 时构建前端并启动服务，默认浏览器地址为 `http://127.0.0.1:8765/`。构建前端需要 Node 20.19+ 或 22.12+；macOS 使用支持 MPS 的 PyTorch，当前 MPS 训练按 FP32 执行，内存预算按统一内存估算。
`./studio.sh doctor` 查看本机环境；`./studio.sh smoke --set model.dit_path=… --set model.text_encoder_path=… --set model.vae_path=…`
用真实权重自检整条训练链路。完整说明（参数、手动安装、目录结构、常驻服务、远程访问、排障）见 [`docs/deploy.md`](docs/deploy.md)。

升级前保留数据和断点。停止旧服务、更新源码后重新运行启动脚本，才会检查依赖与前端指纹；没有联网程序自更新或生产热更新。v0.4 将旧项目关联到兼容 v1，不移动旧数据、任务目录或断点。完整训练状态仍使用 v2 格式，旧指纹断点不能直接当作新版的精确续训状态使用。缓存依据实际权重、分词器和数据内容识别，修改 caption 或 mask 会影响数据指纹；详细兼容与迁移方式见部署指南。

## 开发环境

```bash
uv venv --python 3.12 venv
uv pip install --python venv/bin/python -e ".[dev,models,server]"
venv/bin/pytest                      # 全部 CPU 测试
./studio.sh dev                       # 后端 + Vite 热更新前端
```

CUDA 机器上额外安装 `.[cuda,optim]`。

前端开发默认连接真实服务；仅在开发模式显式设置 `VITE_USE_MOCK=true` 时启用演示数据和模拟 SSE。版本数据仍可编辑，高级 TOML 的外部路径也不会自动复制；需要保留实验数据时先创建完整数据副本的新版本。

## 许可证

本仓库代码以 **Apache-2.0** 发布。参考实现中 sd-scripts、LyCORIS、musubi-tuner 为 Apache-2.0，
AnimaLoraStudio 与 diffusion-pipe 为 GPL-3.0（含 ComfyUI 派生代码）——本项目**不复制** GPL 代码，
只在阅读理解后独立实现，以保持宽松许可。模型权重（Anima / Qwen3 / VAE）各有其自身条款。
