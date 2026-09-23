# Apple Metal FlashAttention

Apple 环境可以选择 PyTorch 内置 SDPA，或安装可选的 **Metal FlashAttention**。后者使用 `mtlattn` 的 Metal 前向与反向内核，不是 NVIDIA 的 `flash-attn` 包。

## 安装与使用

1. 用 `studio-macos.command` 启动 Apple MPS 环境。
2. 在“设置 → 运行环境 → PyTorch 版本”选择兼容的 **2.13.x / Apple MPS**。需要切换时，程序会创建独立环境，原环境保留。
3. 在下方 **Metal FlashAttention** 中检查安装条件、确认安装。程序下载已核对的预编译包，无需本机编译。
4. 重启服务并重新检测；显示“检测正常”后，在训练参数的“注意力后端”选择 **Metal FlashAttention · Apple**。

当前验证组合为 Apple Silicon、macOS 15 或更新版本、CPython 3.11／3.12、PyTorch 2.13.x 和 `mtlattn 0.4.1`。PyTorch 2.14 环境不能直接混装这个二进制包，也不会被安装器自动降级。CPU、CUDA 和海光环境不显示此包；它们保留各自的后端。

## 计算范围

Anima、Krea2、SDXL 和 Klein 的主模型可以使用此选项。当前加速范围是 FP32、head dimension 64／128、无 attention mask、无 attention dropout、非因果的自注意力和交叉注意力。其他形状或语义在调用内核前使用原生 SDPA；选中的 Metal 内核若报错会明确失败，不以静默重试掩盖故障。文本编码器和 VAE 继续使用原路径。

本版不更改 MPS 的 FP32 训练规则，不支持 `torch.compile`。选择此后端不代表模型内每一个注意力调用都会加速，尤其是带 mask 的调用；实际速度与内存收益需按模型、尺寸和机器测量。

完整断点记录注意力实现、扩展、PyTorch、Python 和 macOS 身份。SDPA 与 Metal 状态不能混作严格续训；切换后端可以从导出的适配器开始新训练。

## 验证记录

在 Apple M4／macOS 15.7.9 上，预编译包的实际输出和 Q／K／V 梯度与 CPU 参考比较通过。四个模型族的缩小真实网络分别验证了 LoRA／LoKr、梯度更新、块重算及新进程恢复，共八组通过。缩小 Klein 还通过完整 CLI 的文本编码、VAE、训练、保存、冷恢复和采样，最终训练张量、优化器／随机状态及预览一致。

以上是执行与恢复正确性检查，使用缩小网络和合成素材，不代表官方完整模型容量或长期训练画质已验收。详细机器记录保留在本地 `Test/reports/`。

上游：[mtlattn](https://github.com/lastowl/mtlattn)、[0.4.1 发布包](https://pypi.org/project/mtlattn/0.4.1/)。安装器固定验证 wheel 文件名、大小及 SHA256；没有复制或全局替换第三方注意力实现。
