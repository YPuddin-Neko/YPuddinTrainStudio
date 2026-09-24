# YPuddin Train Studio

用于 Anima、SDXL、Krea 2 和 FLUX.2 Klein 的本地训练工具，提供 Web 界面、LoRA / LoKr 训练及全量微调。

## 功能

- 项目与版本管理、参数预设、TOML 配置导入导出。
- 图片导入与筛选、TXT / JSON 标签编辑、图像裁剪与遮罩编辑。
- 分桶与原生分辨率训练、图像和文本缓存、梯度累积、梯度检查点。
- 任务队列、显卡分配、实时日志、损失曲线、训练预览和断点恢复。
- XYZ 图像对比、训练权重导出与格式转换。
- 模型下载、访问密钥、代理设置、PyTorch 环境及注意力扩展管理。

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

## 安装

环境要求：

- Python 3.10–3.12。
- Node.js 20.19+（20.x）、22.12+（22.x）或更新主版本，用于构建前端。
- GPU 训练需安装对应驱动；海光还需准备匹配的 DTK 运行库与厂商 Python 包。

```bash
git clone https://github.com/YPuddin-Neko/YPuddinTrainStudio.git
cd YPuddinTrainStudio
```

按平台运行启动脚本：

| 平台 | 启动入口 |
| --- | --- |
| Windows CUDA | `studio-windows-cuda.bat` |
| Linux CUDA | `./studio-linux-cuda.sh` |
| Linux DTK | `./studio-linux-dtk.sh`，安装步骤见 [DTK 部署](docs/RUNTIME_DTK.md) |
| macOS Apple Silicon | `./studio-macos.command` |
| Windows CPU | `studio-cpu.bat` |
| Linux / macOS CPU | `./studio-cpu.sh` |

启动脚本会创建独立 Python 环境、安装依赖并构建前端。默认访问地址为 <http://127.0.0.1:8123/>。浏览器自动启动可在设置中关闭。

启动参数、常驻服务和远程访问配置见 [安装与部署](docs/INSTALLATION.md)。

## 使用

1. 在模型管理中下载或登记主模型、文本编码器和 VAE。
2. 创建项目与版本，导入图片及同名标签文件。
3. 在“训练集筛选”中确定参与训练的图片，在目录管理中设置重复次数。
4. 配置模型文件、训练方式、分辨率、优化器和保存频率，检查训练计划后启动任务。
5. 在任务页查看进度、日志与预览；在模型测试页比较保存的训练权重。

训练过程自动生成缺少的缓存。暂停和恢复使用完整训练状态；导出的 LoRA / LoKr 文件用于推理或新任务的权重初始化。

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

本项目采用 [Apache-2.0](LICENSE)。第三方组件的许可证和来源声明位于对应模块目录；模型权重遵循各发布方的许可条款。
