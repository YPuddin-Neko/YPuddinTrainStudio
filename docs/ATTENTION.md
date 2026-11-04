# 注意力后端

训练参数中的“注意力后端”控制主模型的注意力实现。可选项按平台和模型筛选；扩展安装在“设置 → 运行环境”中完成。

| 后端 | 平台 | 条件 |
| --- | --- | --- |
| PyTorch SDPA | CUDA、DTK、MPS、CPU | 随 PyTorch 提供；DTK 部分路径另有厂商动态库依赖 |
| xFormers | CUDA、DTK | 构建需匹配 PyTorch 与设备；DTK 使用厂商包 |
| FlashAttention 2 | CUDA、DTK | 使用对应平台的扩展包；SDXL 不提供 |
| SageAttention | CUDA | 可在运行环境页安装；参数页不提供，Anima、Krea 2 的配置文件指定 `sage` 时只用于预览、验证等无梯度计算，训练计算使用 SDPA |
| Metal FlashAttention | Apple Silicon | 使用匹配版本的 `mtlattn` 预编译包 |

安装成功不等于设备内核可用。运行环境页显示导入与计算检测结果；失败时按具体原因处理。

## FLUX.3 与 MOSS-Image

两者提供自动、SDPA 和 FlashAttention 2，自动保持 SDPA。加载模型时按实际设备、计算精度和注意力形状检查所选后端，训练检查包含反向计算。准备阶段无法使用所选后端时，会检查 SDPA 并记录实际选择；精确续训须保持恢复点记录的后端，后续执行中的内核错误会终止任务。

- **FLUX.3**：NVIDIA CUDA 的 BF16 计算可使用 FA2 变长序列内核，同一批次的不同文本长度按样本边界分别计算。FP32 使用 SDPA。
- **MOSS-Image**：FA2 用于同批次文本长度相同的 BF16 计算；批量大小为 1 时符合文本长度条件。需要 padding mask 的不同长度批次，以及 FP32 计算，使用 SDPA。

## Windows CUDA

Windows FlashAttention 使用 [mjun0812 的社区预编译包](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases)。页面按当前服务的 Python、PyTorch 和 PyTorch CUDA 构建筛选文件，不使用驱动支持的最高 CUDA 版本代替包版本。

下载后校验文件大小与散列，确认安装计划，再执行安装和设备检测。无匹配包时可查看原因或上传兼容 wheel。安装器不自动执行源码编译，也不替换受保护的基础运行时。

xFormers 内核还需支持当前显卡架构。出现 `No operator found` 等错误时，检查构建和设备检测；可改用当前环境可用的 SDPA。

任务日志中的 `A matching Triton is not available` 只表示 xFormers 的部分可选 Triton 内核不可用，`triton not found; flop counting will not work for triton kernels` 只表示 Triton 内核的 FLOPs 统计不可用；两者都不会中止训练。Triton 的安装见 [运行环境](RUNTIME.md#扩展安装)。

## Apple Metal FlashAttention

当前安装器使用 `mtlattn 0.4.1`，要求 Apple Silicon、macOS 15+、CPython 3.11 / 3.12、PyTorch 2.13.x。该预编译包不适用于 PyTorch 2.14。

1. 使用 `studio-macos.command` 启动。
2. 在运行环境中准备并切换到兼容的 Apple MPS / PyTorch 版本。
3. 安装 Metal FlashAttention。
4. 重启并重新检测，在训练参数中选择该后端。

主模型的 FP32、head dimension 64 / 128、无 mask、无注意力 dropout 的非因果注意力可进入 Metal 内核；其他形状使用 SDPA。文本编码器和 VAE 保持各自实现。内核执行错误会终止任务。

MPS 当前使用 FP32，不支持 `torch.compile`。SDPA 与 Metal 的完整训练状态不能互换，切换后端需从权重开始新任务。

## 海光 DTK

DTK 使用 SourceFind 厂商目录中的 FlashAttention 和 xFormers 构建。基础框架安装、包组合与 SDPA 动态库要求见 [DTK 部署](RUNTIME_DTK.md#注意力扩展)。
