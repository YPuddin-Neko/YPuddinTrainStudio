# YPuddin Train Studio

面向扩散模型的本地训练工具，提供 Web 界面、数据集管理、LoRA／LoKr 训练、全量微调和模型测试。

支持 Anima、SDXL、Krea2，以及 FLUX.2 Klein base 4B／9B。不同模型与平台可用的训练选项有所区别，界面会根据当前配置显示。

## 主要功能

- **项目与版本**：为不同训练方案分别管理配置、数据和产物。
- **数据集管理**：导入图片，编辑标签与遮罩，配置重复次数、分桶或原生分辨率训练。
- **训练配置**：选择模型、适配器、优化器、缓存、精度和运行显卡。
- **任务队列**：查看训练进度、损失、日志与采样图，保存检查点并恢复训练。
- **模型测试**：独立生成预览，比较模型和参数组合。
- **环境管理**：管理 PyTorch、注意力加速扩展、模型下载、下载源与代理。

## 快速开始

克隆仓库并进入目录：

```bash
git clone https://github.com/YPuddin-Neko/YPuddinTrainStudio.git
cd YPuddinTrainStudio
```

选择与你的机器对应的启动脚本：

| 环境 | 启动入口 |
| --- | --- |
| Windows + NVIDIA CUDA（x86_64） | 双击 `studio-windows-cuda.bat` |
| Linux + NVIDIA CUDA（x86_64） | `./studio-linux-cuda.sh` |
| Linux + 海光 DTK（x86_64） | `./studio-linux-dtk.sh` |
| macOS Apple 芯片 | `./studio-macos.command` |
| Windows 仅 CPU | 双击 `studio-cpu.bat` |
| Linux / macOS 仅 CPU | `./studio-cpu.sh` |

首次启动会准备独立 Python 环境、安装依赖并构建前端。建议使用 Python 3.12；构建前端需要 Node.js 20.19+（20.x）或 22.12+。海光环境需先准备匹配的 DTK 运行库与厂商 PyTorch，见 [DTK 安装说明](docs/RUNTIME_DTK.md)。

启动后访问 **http://127.0.0.1:8765/**。

1. 在设置中检查运行环境，下载或选择本地模型。
2. 创建项目与版本，导入训练图片和标签。
3. 配置训练参数，选择显卡并开始训练。
4. 在任务页面查看进度，使用模型测试检查训练产物。

依赖下载默认使用中科大源，可在设置中切换。完整安装、远程访问和故障排查见 [部署指南](docs/deploy.md)。

## 数据与更新

默认运行数据保存在 `studio_data/`，环境与模型等路径可单独配置。模型、训练图片、检查点、密钥和本地缓存不应提交到 Git。

更新前保存训练状态并停止服务，执行 `git pull` 后重新运行启动脚本。重要数据与检查点请另行备份。

## 使用文档

- [训练参数与分桶](docs/TRAINING_PARAMETERS.md)
- [JSON 标签格式](docs/JSON_CAPTIONS.md)
- [原生分辨率与整图保留](docs/native-resolution.md)
- [文件夹布局](docs/LAYOUT.md)
- [Krea2 显存配置](docs/KREA2_MEMORY.md)
- [支持范围与限制](docs/SUPPORT.md)

## 开发

后端代码位于 `ypuddin/`，前端位于 `frontend/`，测试位于 `Test/tests/`。开发环境配置见 [部署指南](docs/deploy.md)，架构文档位于 [`docs/design/`](docs/design/)。

## 许可证

项目许可证见 [Apache-2.0](LICENSE)。第三方组件的声明与许可证见对应目录中的 `NOTICE`／`LICENSE` 文件；模型权重遵循各自的许可条款。

模型目录分类与共用组件：[模型存放目录](docs/MODEL_STORAGE.md)。
