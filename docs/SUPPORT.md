# 支持范围

- 支持 Anima、SDXL、Krea2、FLUX.2 Klein base 4B／9B；具体训练选项按模型能力显示。
- Windows 原生多卡只支持 DDP；Linux CUDA／DTK 可以使用 FSDP 分配多卡显存。
- FP16 动态梯度缩放目前只开放单卡，不接受 FP16 可训练参数；建议使用 FP32 适配器参数。
- SDXL 冻结文本编码器时使用文本缓存，目前不开放在线编码。
