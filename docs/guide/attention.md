# 注意力加速

训练参数的“注意力后端”决定主模型使用哪种注意力实现。SDPA 随 PyTorch 提供；其他后端需要先在 **设置 → 运行环境** 安装并通过本机检测，界面按当前平台筛选选项。

| 后端 | 平台 | 说明 |
| --- | --- | --- |
| SDPA | 全部 | PyTorch 内置；DTK 的部分路径还依赖厂商 FlashAttention 动态库 |
| xFormers | Windows／Linux CUDA、海光 DTK | 需匹配 PyTorch 和设备；DTK 厂商包还依赖 FlashAttention |
| FlashAttention 2 | Windows／Linux CUDA、海光 DTK | Windows 使用社区预编译包，海光使用厂商包 |
| SageAttention | CUDA | 只用于不需要梯度的采样，训练反向仍使用 SDPA |
| Metal FlashAttention | Apple Silicon | 需要 PyTorch 2.13.x，见下文 |

“版本匹配”只说明安装包和当前环境的版本对得上，能否使用以安装后的本机前向、反向检测为准。检测失败的扩展不会显示为可用。

## Windows

Windows 页面提供 **mjun0812 社区预编译的 FlashAttention 2**，不是 Dao-AILab 官方发布的 Windows 包。构建列表从 [维护者发布页](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases) 查询，查询失败时使用缓存或内置的发布信息。

- 按服务实际使用的 Python、PyTorch 和 PyTorch 的 CUDA 版本筛选，不按 NVIDIA 驱动显示的最高 CUDA 版本选择。
- 流程：选择构建 → 下载并校验 SHA256 和大小 → 检查安装计划 → 确认安装 → 本机内核检测。
- 下载失败不会安装，页面保留手动下载和上传入口；没有匹配的构建时，可以上传自己准备的兼容 wheel。
- 安装计划不能替换 Torch、Triton 等受保护的运行时，Windows 上不会隐式从源码编译。

xFormers 的“已安装”不代表它的 CUDA 内核支持当前显卡。例如 RTX 50 系显卡配旧版 xFormers 时可能出现 `No operator found` 或 `GPU ... too new`。运行环境页会分别显示导入状态和实际检测结果；SDPA 检测通过时可以继续使用 SDPA，不要为此降级 PyTorch。

## Apple Metal FlashAttention

Apple 环境可以安装可选的 **Metal FlashAttention**，它使用 `mtlattn` 的 Metal 前向和反向内核，不是 NVIDIA 的 `flash-attn`。

安装步骤：

1. 用 `studio-macos.command` 启动。
2. 在 **设置 → 运行环境 → PyTorch 版本** 选择 **2.13.x / Apple MPS**；需要切换时会创建新环境，原环境保留。
3. 在 **Metal FlashAttention** 中检查安装条件并确认安装。程序下载已核对的预编译包，不需要本机编译。
4. 重启服务并重新检测。显示“检测通过”后，在训练参数的“注意力后端”选择 **Metal FlashAttention · Apple**。

要求 Apple Silicon、macOS 15 或更新、CPython 3.11／3.12、PyTorch 2.13.x 和 `mtlattn 0.4.1`。PyTorch 2.14 环境不能安装这个预编译包。安装器会校验 wheel 的文件名、大小和 SHA256。

使用范围：

- Anima、Krea 2、SDXL 和 Klein 的主模型可以使用；文本编码器和 VAE 保持原来的实现。
- 加速 FP32、head dimension 64／128、无注意力 mask、无注意力 dropout 的非因果自注意力和交叉注意力；其他形状仍使用 SDPA。
- Metal 内核报错时任务直接失败，不会悄悄改用 SDPA。
- MPS 按 FP32 训练，不支持 `torch.compile`。
- 断点记录注意力实现和环境。SDPA 与 Metal 的训练状态不能互相精确续训；换后端后可以从导出的适配器开始新训练。

上游：[mtlattn](https://github.com/lastowl/mtlattn)、[0.4.1 发布包](https://pypi.org/project/mtlattn/0.4.1/)。

## 海光 DTK

海光环境使用 SourceFind 厂商目录中的 FlashAttention 和 xFormers，见 [海光 DTK](dtk.md#注意力扩展)。
