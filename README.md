# YPuddin Train Studio

本地运行的扩散模型训练工具。通过浏览器管理模型、图片、标签、训练参数和任务，支持 **LoRA／LoKr 训练、全量微调、断点续训和 XYZ 出图对比**。

[安装与启动](docs/guide/install.md) · [使用文档](docs/README.md) · [训练说明](docs/guide/training.md) · [更新记录](CHANGELOG.md)

## 能做什么

- **管理训练方案**：按项目和版本保存配置、数据与产物；保存参数预设，导入或导出 TOML 配置。
- **整理训练数据**：导入图片与 TXT／JSON 标签，编辑标签和遮罩，检查图片，修改目录名和重复次数。图片可以暂时移出训练，重新加入时保留原图、标签和遮罩。
- **配置训练**：选择模型和适配器，设置优化器、精度、缓存、分桶或原生尺寸；查看实际训练尺寸与步数估算。
- **运行和恢复任务**：按显卡安排任务，查看损失曲线、实时日志和采样图，暂停并保存完整训练状态，之后继续训练。
- **比较训练结果**：用 XYZ 网格对比不同权重、适配器强度、采样步数和随机种子。
- **管理环境与下载**：登记本地模型或下载推荐权重，切换下载源、配置代理，管理 PyTorch 与当前平台适用的注意力扩展。

## 支持的模型

| 模型 | 准备哪些权重 | 注意事项 |
| --- | --- | --- |
| **Anima** | Anima 主模型、Qwen3-0.6B、Qwen-Image VAE | 主模型使用 BF16／FP16／FP32 原始权重 |
| **SDXL** | 完整 SDXL checkpoint，或分别提供 UNet、CLIP-L、CLIP-G、VAE | 预测方式需与底模一致；冻结文本编码器时使用文字缓存 |
| **Krea 2** | Raw 主模型、Qwen3-VL-4B、Qwen-Image VAE | Raw 用于训练；Turbo 用于出图，支持搭配 Raw 训练的适配器 |
| **FLUX.2 Klein base 4B／9B** | Base 主模型、对应 Qwen3-4B／8B、FLUX.2 VAE | 只训练 base 版本；9B 官方权重需先取得访问权限 |

可用参数随模型、设备和训练方式变化。各模型的文件格式、加载方式与限制见 [模型与权重](docs/guide/models.md)。

## 安装与启动

准备 **Python 3.10–3.12**（推荐 3.12）。从源码首次构建界面还需要 **Node.js 20.19+（20.x）或 22.12+**；GPU 环境需先安装匹配的驱动。

```bash
git clone https://github.com/YPuddin-Neko/YPuddinTrainStudio.git
cd YPuddinTrainStudio
```

运行与你的机器对应的入口：

| 环境 | 启动方式 |
| --- | --- |
| Windows + NVIDIA CUDA（x86_64） | 双击 `studio-windows-cuda.bat` |
| Linux + NVIDIA CUDA（x86_64） | `./studio-linux-cuda.sh` |
| Linux + 海光 DTK（x86_64） | `./studio-linux-dtk.sh`，先按 [DTK 指南](docs/guide/dtk.md) 准备运行库和厂商包 |
| macOS Apple 芯片 | `./studio-macos.command` |
| Windows 仅 CPU | 双击 `studio-cpu.bat` |
| Linux／macOS 仅 CPU | `./studio-cpu.sh` |

首次启动会创建独立环境、安装依赖并构建界面，然后打开 **<http://127.0.0.1:8123/>**。以后继续使用同一个入口即可；设置中可以关闭“启动时打开浏览器”。CPU 入口适合管理数据和配置，大模型训练需要相应的计算资源。

- **NVIDIA**：使用 CUDA，注意力扩展在运行环境页按需安装。
- **Apple**：使用 MPS 和统一内存，当前训练路径为 FP32；可选 [Metal FlashAttention](docs/guide/attention.md#apple-metal-flashattention)。
- **海光**：使用匹配的 DTK／HIP 厂商环境，不能安装 NVIDIA 的 CUDA 包。
- **多卡**：Linux CUDA／DTK 支持 DDP 和 FSDP；原生 Windows CUDA 支持 DDP。配置限制见 [多卡训练](docs/guide/multi-gpu.md)。

更换端口、指定数据目录、离线准备、远程部署和故障排查见 [安装与启动](docs/guide/install.md)。服务没有登录认证；远程访问请使用 SSH 隧道或带认证的反向代理，不要直接暴露到公网。

## 开始一次训练

1. **准备模型**：在模型权重页下载所需组件，或登记已有文件。下载源与访问密钥在设置中管理。
2. **创建项目与版本**：选择模型系列，导入图片和同名标签；在数据页设置重复次数，按需编辑标签、遮罩和训练参与状态。
3. **设置参数**：选择模型文件和训练方式，填写分辨率、学习率、训练时长及保存频率。参数下方是简短说明，旁边的 `?` 提供详细用法。
4. **检查并启动**：查看数据检查、尺寸分布和训练计划，处理错误后选择运行显卡，加入队列。缺少的有效缓存会在训练前自动准备。
5. **查看和比较结果**：在任务页查看日志、损失与预览，在模型测试页用相同提示词和种子比较保存的权重。

**继续原来的训练进度，要恢复完整训练状态；导出的适配器权重用于出图，或作为新训练的起点。** 修改图片、标签、遮罩或关键训练设置后，原状态可能无法精确恢复。

## 数据保存与更新

默认数据目录是 `studio_data/`，保存项目、版本、配置、模型登记和训练产物。模型、缓存和输出目录可在设置中单独配置；修改路径不会自动搬迁已有文件，详见 [目录结构与备份](docs/guide/folders.md)。

更新前先保存训练状态、停止服务，并备份数据目录及外部数据和输出目录，然后执行：

```bash
git pull
```

再运行原来的启动脚本。启动器会检查依赖和前端构建。更新代码不会删除训练数据；请保留完整断点，以便恢复训练。

## 文档导航

| 需要了解 | 文档 |
| --- | --- |
| 安装、启动参数、远程访问与排错 | [安装与启动](docs/guide/install.md) |
| 下载模型、组件格式与存放位置 | [模型与权重](docs/guide/models.md) |
| 训练方式、分桶、缓存、显存与 LoKr Full | [训练说明](docs/guide/training.md) |
| 原图如何缩放、补边，两个尺寸上限怎么填 | [原生尺寸训练](docs/guide/native-resolution.md) |
| JSON 标签读取与编辑 | [JSON 标签](docs/guide/json-captions.md) |
| PyTorch 切换、下载源、代理与重启 | [运行环境](docs/guide/environment.md) |
| SDPA、FlashAttention 与 Apple Metal | [注意力加速](docs/guide/attention.md) |
| 多卡训练和不同显卡并行任务 | [多卡训练](docs/guide/multi-gpu.md) · [任务调度](docs/guide/gpu-scheduling.md) |
| 权重和采样参数对比 | [模型测试](docs/guide/xyz-sampling.md) |
| 目录、备份与开发入口 | [目录结构](docs/guide/folders.md) · [开发说明](docs/development.md) |

## 许可证

项目采用 [Apache-2.0](LICENSE)。第三方组件遵循各自目录中的许可证和声明；模型权重的使用与分发遵循发布方的许可条款。
