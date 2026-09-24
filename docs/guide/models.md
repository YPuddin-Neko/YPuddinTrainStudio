# 模型与权重

## 支持的模型

| 模型 | 规模 | 训练目标 | 需要的文件 |
| --- | --- | --- | --- |
| Anima | 2B DiT | Rectified Flow | DiT（含 LLM adapter）、Qwen3-0.6B 文本编码器、Qwen-Image VAE |
| SDXL | 2.6B UNet | DDPM（ε 或 v 预测） | 完整 SDXL checkpoint，或 UNet、CLIP-L、CLIP-G、VAE |
| Krea 2 Raw | 12.9B DiT | Rectified Flow | DiT、Qwen3-VL-4B 文本编码器、Qwen-Image VAE，见 [Krea 2](krea2.md) |
| FLUX.2 Klein base 4B／9B | 4B／9B | Rectified Flow | Klein base DiT、对应的 Qwen3-4B／Qwen3-8B 文本编码器、FLUX.2 VAE |

可选择适配器训练或全量微调主模型、文本编码器；具体算法、训练对象和精度组合由模型与平台共同决定。多卡尤其是 FSDP 有额外限制，见 [多卡训练](multi-gpu.md)。

FLUX.1 和 FLUX.2 dev 已停用：原来的项目、配置、模型文件和训练结果都保留，可以查看，但需要新建受支持模型的配置才能训练。

## Anima

| 字段 | 文件 | 说明 |
| --- | --- | --- |
| `model.dit_path` | `anima-base-*.safetensors`，或带 `model.diffusion_model.` 前缀的 ComfyUI 格式 | DiT 主干 + LLM adapter，BF16 约 4 GB |
| `model.text_encoder_path` | Qwen3-0.6B（HF 目录或单文件 safetensors） | 单文件时使用内置分词器 |
| `model.vae_path` | `qwen_image_vae.safetensors` | 16 通道、8 倍下采样 |

Anima 的 DiT 和 Qwen 编码器不能直接加载现成的 FP8 或带量化缩放的文件，请使用原始 BF16／FP16／FP32 权重。适配器训练需要省显存时，可以把底模存储精度设为 FP8：程序启动时量化适配器覆盖的冻结线性层，原文件不改动。

冻结文本编码器时，Anima 默认每步重新编码标签（Qwen3-0.6B 较小，可以常驻），标签打乱、丢弃不受缓存限制；也可以改用文字缓存。其他模型冻结文本编码器时都使用文字缓存。

## SDXL

- 默认下载光辉 Illustrious-XL v0.1，也可以选择其他 SDXL 底模。
- 按底模说明选择 ε 或 v 预测，并按需开启末端零信噪比；把 ε 模型设成 v 预测不会让它变成 v 模型。
- SDXL 使用自己的 DDPM 噪声日程，不能套用 Flow 模型的时间偏移和加权；预览采样支持 Euler、Heun。
- 冻结文本编码器时使用双 CLIP 文字缓存。
- 单卡不提供分块换出。

## FLUX.2 Klein

- 只训练 **base** 版本。蒸馏版、KV 版和图像编辑模型不能训练。
- 蒸馏版与基础版的权重尺寸可能相同，单个文件无法可靠区分，请选择发布说明中标明 base 的文件。目录元数据充足时会自动识别，明确为蒸馏版的权重会被拒绝。
- 推荐下载使用独立文件：Base 4B 或 9B 的 DiT、对应的 Qwen3-4B 或 Qwen3-8B 文本编码器，以及两者共用的 FLUX.2 VAE。标准 BF16 单文件所需的配置和分词器已内置，不需要另下完整 Diffusers 包；已有的完整模型包也能直接使用。
- 9B 官方权重需要先在发布方页面接受访问协议，并在设置里配置访问密钥。FP8／FP4 量化文件不作为训练推荐项。
- 9B 的文本编码器也更大，仅靠降低训练分辨率不能保证 16 GB 显卡可用。
- 默认预览 50 步、CFG 4。

Klein 和 Krea 2 一样先缓存图片和文字、卸载编码器，再读取主模型权重。文本编码阶段仍需要把整个文本编码器放到设备上；训练时可以用分块换出和梯度检查点减少显存，代价是更多电脑内存或训练时间。训练预览会临时加载 VAE，预览时的显存峰值比纯训练阶段高。

## 模型存放目录

默认位置是 `studio_data/models/`，可在 **设置 → 存储路径** 修改。下载器按组件和模型系列存放：

```text
models/
├── diffusion_models/
│   ├── anima/
│   ├── krea2/
│   ├── sdxl/
│   └── flux2/
├── vae/
│   ├── shared/       Anima 与 Krea 2 共用的 Qwen-Image VAE
│   ├── sdxl/
│   └── flux2/
└── text_encoders/
    ├── anima/
    ├── sdxl/
    ├── flux2/
    └── shared/       多个模型共用的文本编码器
```

- 系列目录下按资源标识再分子目录，不同来源的同名文件不会互相覆盖。
- 完整 Diffusers 模型包放在 `diffusion_models/<系列>/` 下，保持包内结构，不要拆开其中的组件。
- 扫描会递归读取模型目录，按配置和权重结构识别组件，不根据文件夹名称判断。
- 训练参数中的模型路径未填写时，“浏览”从对应组件目录打开。
- 已登记的旧路径不会被下载器移动或覆盖。

## 下载与登记

**模型权重 → 准备模型** 按模型列出主模型、文本编码器和 VAE，可以在该资源提供的 Hugging Face 或魔搭来源中选择。下载源会保存，失败后重试使用当前选择；下载中可以取消。

- 推荐文件下载后核对大小、SHA-256 和组件类型，通过后才登记；重新下载会从头开始。
- 共用的 VAE 和文本编码器可以直接复用。
- 本地文件第一次设为推荐默认时要读取整个文件计算 SHA，大文件需要等待；之后按文件状态复用校验结果。
- 受限仓库需要先在发布平台取得权限，并在 **设置 → 访问密钥** 保存令牌。
- 自定义下载不会标记为推荐文件。

## 训练结果的格式

- 适配器保存为 `.safetensors`，使用 kohya 键名（`lora_unet_*`）。`ypuddin convert --to comfyui` 转为 ComfyUI 键名，LoRA 也可以转为 PEFT 格式。
- 全量微调保存为 `.model` 目录，包含训练组件的原生权重、组件配置和清单；未参与训练的 VAE、分词器等继续引用原模型。结果页可以打包下载 ZIP。
