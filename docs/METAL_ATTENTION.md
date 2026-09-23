# Apple Metal FlashAttention

Apple 环境可以选择 PyTorch 内置 SDPA，或安装可选的 **Metal FlashAttention**。后者使用 `mtlattn` 的 Metal 前向与反向内核，不是 NVIDIA 的 `flash-attn` 包。

## 安装与使用

1. 用 `studio-macos.command` 启动 Apple MPS 环境。
2. 在“设置 → 运行环境 → PyTorch 版本”选择兼容的 **2.13.x / Apple MPS**。需要切换时，程序会创建独立环境，原环境保留。
3. 在下方 **Metal FlashAttention** 中检查安装条件、确认安装。程序下载已核对的预编译包，无需本机编译。
4. 重启服务并重新检测；显示“检测通过”后，在训练参数的“注意力后端”选择 **Metal FlashAttention · Apple**。

需要 Apple Silicon、macOS 15 或更新版本、CPython 3.11／3.12、PyTorch 2.13.x 和 `mtlattn 0.4.1`。PyTorch 2.14 环境不能安装这个预编译包，需要先切换到 2.13.x 环境。CPU、CUDA 和海光环境不显示此选项。

## 计算范围

Anima、Krea2、SDXL 和 Klein 的主模型可以使用此选项。当前加速范围是 FP32、head dimension 64／128、无 attention mask、无 attention dropout、非因果的自注意力和交叉注意力。其他形状或语义使用原生 SDPA；Metal 内核报错时任务直接失败，不会自动改用 SDPA。文本编码器和 VAE 继续使用原路径。

MPS 仍按 FP32 训练，不支持 `torch.compile`。带 mask 等超出上述范围的调用不会加速，实际速度与内存收益随模型、尺寸和机器变化。

完整断点记录注意力实现、扩展、PyTorch、Python 和 macOS 身份。SDPA 与 Metal 状态不能混作严格续训；切换后端可以从导出的适配器开始新训练。

上游：[mtlattn](https://github.com/lastowl/mtlattn)、[0.4.1 发布包](https://pypi.org/project/mtlattn/0.4.1/)。安装器会校验 wheel 文件名、大小和 SHA256。
