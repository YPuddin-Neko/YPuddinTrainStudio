<div align="center">

<img src="frontend/public/icon-192.png" width="160" alt="YPuddin Train Studio">

# YPuddin Train Studio

一个本地运行的训练工作台，支持 Anima、SDXL、Krea 2、FLUX.2 Klein 图像训练，以及 VoxCPM 1.5、GPT-SoVITS v5 语音微调

[![License](https://img.shields.io/badge/License-GPLv3-blue)](LICENSE) [![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)](#支持的模型与平台) [![Version](https://img.shields.io/badge/Version-v0.6.0-orange)](CHANGELOG.md) [![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB)](#快速开始) [![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C)](docs/RUNTIME.md) [![React](https://img.shields.io/badge/React-18-61DAFB)](frontend/package.json)

[快速开始](#快速开始) · [功能介绍](#功能介绍) · [文档](#文档)

</div>

## 功能介绍

- **整理训练数据**：导入与筛选图片，使用 Tagger 或视觉大模型打标，编辑 TXT / JSON 标签，制作遮罩和准备正则图片。
- **配置训练**：支持 LoRA、LoKr、LoHa、OrthoLoRA、T-LoRA、LyCORIS Full 与全量微调；可选择分桶或原生分辨率，查看图片尺寸变化和显存估算。
- **安排显卡与任务**：不同显卡可同时运行不同任务；图像训练可通过 DDP / FSDP 共同训练一个模型，支持暂停、保存恢复点和继续训练。
- **查看训练过程**：实时日志、损失与硬件曲线、定期预览，以及按任务管理的权重和恢复点。
- **测试训练结果**：选择保存的权重，对比模型版本、强度和采样参数；保留生成记录，放大或下载图片。OrthoLoRA 和 T-LoRA 导出为通用 LoRA，适配器兼容范围见[内置适配器](docs/ADAPTERS.md)。
- **训练语音**：创建语音项目，选择 VoxCPM 1.5 LoRA 或 GPT-SoVITS v5dev / v5turbo，按版本配置独立环境、录音转写和训练参数，排队训练并生成试听音频。配置方式见 [语音训练](docs/TTS.md)。

## 支持的模型与平台

| 模型 | 训练权重 | 文本编码器 |
| --- | --- | --- |
| Anima | Anima 原始 BF16 / FP16 / FP32 权重 | Qwen3-0.6B |
| SDXL | 完整 checkpoint 或独立组件 | CLIP-L、CLIP-G |
| Krea 2 | Raw；支持 BF16 和受支持的 FP8 scaled 格式 | Qwen3-VL-4B |
| FLUX.2 Klein | Base 4B / 9B | 对应 Qwen3-4B / Qwen3-8B |

Krea 2 Turbo 用于采样。Klein 蒸馏版、KV 版及 FLUX.2 dev 不用于训练。各模型所需的 VAE、文件格式和加载限制见 [模型配置](docs/MODELS.md)。

| 平台 | 计算后端 | 单任务多卡 |
| --- | --- | --- |
| Windows + NVIDIA，x86_64 | CUDA | DDP |
| Linux + NVIDIA，x86_64 | CUDA | DDP、FSDP |
| Linux + 海光，x86_64 | DTK / HIP | DDP、FSDP |
| macOS Apple Silicon | MPS | 不支持 |
| Windows / Linux / macOS，无 GPU | CPU | 不支持 |

多卡的精度、优化器和训练对象限制见 [多卡训练](docs/MULTI_GPU.md)。

以上模型与平台表适用于图像训练。语音训练使用所选引擎的独立环境和完整预训练模型目录，支持单卡 CUDA 训练及排队试听。VoxCPM 1.5 使用 LoRA；GPT-SoVITS v5 支持 GPT 全参微调与 SoVITS LoRA，可分别或依次执行。任务支持取消和重新训练，不支持暂停恢复或多卡训练。音频格式、环境安装和源码版本要求见 [语音训练](docs/TTS.md)。

## 快速开始

环境要求：

- Python 3.10–3.12。
- Node.js 20.19+（20.x）、22.12+（22.x）或更新主版本，用于构建前端。
- GPU 训练需安装对应驱动；海光还需匹配的 DTK 运行库，海光版 PyTorch 的准备方式见 [DTK 部署](docs/RUNTIME_DTK.md)。

```bash
git clone https://github.com/YPuddin-Neko/YPuddinTrainStudio.git
cd YPuddinTrainStudio
```

按平台运行启动脚本：

| 平台 | 启动入口 |
| --- | --- |
| Windows CUDA | `studio-windows-cuda.bat` |
| Linux CUDA | `./studio-linux-cuda.sh` |
| Linux DTK | `./studio-linux-dtk.sh` |
| macOS Apple Silicon | `./studio-macos.command` |
| Windows CPU | `studio-cpu.bat` |
| Linux / macOS CPU | `./studio-cpu.sh` |

启动脚本会创建独立 Python 环境、安装依赖并构建前端。默认访问地址为 <http://127.0.0.1:8123/>，浏览器自动打开可在设置中关闭。

启动参数、常驻服务和远程访问配置见 [安装与部署](docs/INSTALLATION.md)。

## 使用

1. 在“设置 → 模型权重”中下载或登记主模型、文本编码器和 VAE。
2. 创建项目与版本，导入图片及同名标签文件。
3. 在“数据集检查”中查看问题，在“数据集筛选”中确定参与训练的图片；打开数据集可设置每轮重复次数。
4. 在“图片打标”中生成标签，在“标签编辑”中逐图修改。
5. 配置模型文件、训练算法、分辨率、优化器和保存频率，检查训练计划后启动任务。
6. 在任务页查看进度、日志与预览；在模型测试页比较保存的训练权重。

训练过程自动生成缺少的缓存。暂停和恢复使用完整训练状态；导出的权重可用于推理，或作为新任务的初始权重。

## 更新与备份

默认数据目录为 `studio_data/`。更新前保存训练状态、停止服务，并备份数据目录及自定义输出目录，然后执行 `git pull`，重新运行原启动脚本。

数据目录、模型目录、缓存与输出位置的用途及迁移要求见 [存储与备份](docs/STORAGE.md)。

## 文档

- [安装与部署](docs/INSTALLATION.md) · [运行环境](docs/RUNTIME.md) · [海光 DTK](docs/RUNTIME_DTK.md)
- [模型配置](docs/MODELS.md) · [数据集管理](docs/DATASETS.md) · [JSON 标签](docs/JSON_CAPTIONS.md)
- [训练配置](docs/TRAINING.md) · [原生分辨率](docs/NATIVE_RESOLUTION.md) · [多卡训练](docs/MULTI_GPU.md)
- [语音训练](docs/TTS.md)
- [注意力后端](docs/ATTENTION.md) · [模型测试](docs/MODEL_TESTING.md) · [存储与备份](docs/STORAGE.md)
- [开发指南](docs/DEVELOPMENT.md) · [架构](docs/ARCHITECTURE.md) · [更新记录](CHANGELOG.md)

## 许可证

本项目采用 [GNU General Public License v3.0](LICENSE)。第三方组件的许可证和来源声明位于对应模块目录；模型权重遵循各发布方的许可条款。
