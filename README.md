<div align="center">

<img src="frontend/public/icon-192.png" width="160" alt="YPuddin Train Studio">

# YPuddin Train Studio

一个本地运行的 LoRA / LoKr 训练器，支持 Anima、SDXL、Krea 2 与 FLUX.2 Klein

[![License](https://img.shields.io/badge/License-GPLv3-blue)](LICENSE) [![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)](#支持的模型与平台) [![Version](https://img.shields.io/badge/Version-v0.5.9-orange)](CHANGELOG.md) [![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB)](#快速开始) [![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C)](docs/RUNTIME.md) [![React](https://img.shields.io/badge/React-18-61DAFB)](frontend/package.json)

[快速开始](#快速开始) · [功能介绍](#功能介绍) · [文档](#文档)

</div>

## 功能介绍

- **训练算法**：LoRA、LoKr、LoHa、OrthoLoRA、T-LoRA 和 LyCORIS Full，可选 DoRA、rsLoRA，SDXL 可同时训练卷积层；也支持主模型或文本编码器的全量微调。
- **数据集**：图片导入、数据集检查与筛选，WD / PixAI / CL Tagger 等打标模型、视觉大模型和辅助打标，TXT / JSON 标签编辑，自动头部遮罩，图像涂抹与遮罩编辑，从图站下载训练图和正则图，用底模生成正则图。
- **训练**：分桶与原生分辨率、图像和文本缓存、梯度累积与梯度检查点、显存估算，DDP / FSDP 多卡训练。
- **任务**：任务队列与显卡分配、强制开始、实时日志和指标曲线、训练预览、按完整状态暂停与恢复、任务归档。
- **结果**：XYZ 模型对比，连续测试保留已加载的底模，可手动“释放显存”；训练权重导出，导出的 LoRA 可直接在 ComfyUI 中加载，OrthoLoRA 和 T-LoRA 也导出为通用的 LoRA 文件。
- **环境**：模型下载、访问密钥、代理与下载源，PyTorch 版本切换，注意力扩展、8-bit 优化器和打标与遮罩所需 ONNX Runtime 的安装，视觉大模型服务设置。

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
- [注意力后端](docs/ATTENTION.md) · [模型测试](docs/MODEL_TESTING.md) · [存储与备份](docs/STORAGE.md)
- [开发指南](docs/DEVELOPMENT.md) · [架构](docs/ARCHITECTURE.md) · [更新记录](CHANGELOG.md)

## 许可证

本项目采用 [GNU General Public License v3.0](LICENSE)。第三方组件的许可证和来源声明位于对应模块目录；模型权重遵循各发布方的许可条款。
