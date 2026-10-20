# 模型配置

## 组件

| 模型 | 主模型 | 文本编码器 | VAE |
| --- | --- | --- | --- |
| Anima | Anima DiT | Qwen3-0.6B | Qwen-Image VAE |
| SDXL | SDXL UNet | CLIP-L、CLIP-G | SDXL VAE |
| Krea 2 | Krea 2 Raw DiT | Qwen3-VL-4B | Qwen-Image VAE |
| FLUX.2 Klein 4B | Klein Base 4B DiT | Qwen3-4B | FLUX.2 VAE |
| FLUX.2 Klein 9B | Klein Base 9B DiT | Qwen3-8B | FLUX.2 VAE |
| FLUX.3 Klein 5.5B | Image Klein Base DiT | Qwen3-VL-4B-Instruct | FLUX.3 配套 AE（与 FLUX.2 VAE 同权重） |
| MOSS-Image 5B | MOSS-Image Base DiT | 配套 Qwen3.5-9B | MOSS-Image 小解码器 AutoencoderKLFlux2 |

“设置 → 模型权重”支持本地登记和下载。扫描读取权重文件头及模型配置，识别组件、模型系列和存储精度。无法确认的信息需按模型发布说明补充。

## 下载与登记

在“设置 → 模型权重 → 准备模型”选择组件及“下载来源”。推荐资源提供对应的 Hugging Face 或 ModelScope 来源；需要访问授权的仓库，先取得权限，再在“设置 → 访问密钥”保存访问令牌。

推荐文件下载完成后校验文件大小、SHA-256 和组件类型，随后登记；无论从哪个来源下载，本地都使用同一个文件名。下载可以取消；失败后切换来源再重试，使用新的来源地址。所选下载来源保存在当前浏览器。

本地权重可通过扫描登记。登记不移动原文件，训练配置仍记录实际路径。完整 Diffusers 模型目录应保留其内部配置、词表、VAE 和文本编码器结构。

## Anima

主模型和 Qwen3 编码器支持原始 BF16 / FP16 / FP32 权重。主模型接受常见的 `net.` 或 `model.diffusion_model.` 键名前缀；现成 FP8 量化权重不用于此加载路径。

Anima 同时使用 Qwen3 文本特征和旧版 T5 分词编号。程序内置两套分词器资源，默认无需单独下载。`model.tokenizer_path` 可覆盖 T5 分词器目录；该目录只包含分词资源，不需要 T5 文本编码器权重。

冻结文本编码器时，Anima 默认每步编码标签，也可选择文本缓存。LoRA / LoKr 训练可在支持的平台使用运行时 FP8 底模存储，具体限制见 [训练配置](TRAINING.md#精度与显存)。

## SDXL

支持完整 SDXL checkpoint，也可分别配置 UNet、CLIP-L、CLIP-G 和 VAE。完整模型包含相应组件时，可沿用模型内的组件。

预测方式必须与底模一致，支持 ε 和 v 预测。末端零信噪比等设置按底模要求配置。冻结文本编码器时采用双 CLIP 文本缓存；单卡不提供分块换出。

UNet 含卷积层。适配器训练默认只训练线性层，可在“训练层类型”中同时训练 ResNet、上下采样等卷积层，见[训练层类型](TRAINING.md#训练层类型)；全量微调始终包含卷积层。

单文件 checkpoint 带有 `v_pred`、`ztsnr` 标记键时（如 NoobAI-XL V 预测 1.0），训练前检查要求预测方式为 v_prediction，并开启零终点信噪比；模型测试选用这类底模采样时自动按标记设置。没有标记的文件按参数中的选择处理。

推荐下载包含 Illustrious-XL v0.1（默认）、NoobAI-XL V 预测 1.0 和 NoobAI-XL E 预测 1.1。NoobAI-XL 1.1 的 ModelScope 来源是官方仓库；V 预测 1.0 在 ModelScope 只有社区镜像，下载后按相同的 SHA-256 校验。

## Krea 2

训练使用 Raw 权重。Turbo 是采样模型，可以搭配 Raw 训练出的 LoRA / LoKr，在 [模型测试](MODEL_TESTING.md) 中出图。

主模型支持 BF16 和 Comfy-Org 的 FP8 scaled 格式。FP8 scaled 加载保留文件中的权重与逐张量缩放值；不接受任意 MXFP8、NVFP4 或其他量化格式。FP8 文本编码器单文件在加载时应用缩放并转换到计算精度。

Raw 与 Turbo 的权重形状相同。推荐文件经校验后记录类型；未登记的本地文件或自定义下载需要按发布说明确认类型。选择已登记的文件时，页面自动带入类型。

开启图像和文本缓存、并冻结文本编码器时，Krea 2 在编码器卸载后加载主模型。文本编码、主模型加载和采样仍有独立的内存峰值；仅降低训练分辨率不能消除模型加载开销。

## FLUX.2 Klein

训练范围为 Base 4B 和 Base 9B。蒸馏版、KV 版和图像编辑模型不在当前训练范围内。

支持官方格式的 DiT 单文件与本地 Diffusers 目录。独立 DiT 文件需配套文本编码器及 VAE；标准单文件所需的模型配置和分词器资源已内置。主模型及文本编码器的量化文件不用于当前加载路径。

Base 和蒸馏模型可能具有相同的权重形状。完整目录的元数据或经校验的下载记录可用于识别；信息不足时须按发布说明选择类型。Klein 9B 官方权重需要仓库访问授权。

文本编码阶段需要容纳完整的 Qwen 编码器及其激活。缓存完成后卸载编码器再加载主模型。LoRA / LoKr 训练中的梯度检查点、分块换出和精度限制由配置检查处理。

## FLUX.3 Klein 5.5B

训练使用 Image Klein Base 的原始 BF16 / FP32 权重，接受完整本地目录或其中的 `model.safetensors`。蒸馏版及 FP8／NVFP4 推理权重不用于训练。

文本编码器使用未量化的 Qwen3-VL-4B-Instruct，支持完整本地目录，或配有匹配配置及 processor 的 BF16 单文件。完整模型包可提供文本编码器、processor 和 VAE；独立主模型需选择配套组件。经身份校验的相同文本编码器与 FLUX.2 VAE 可复用已有文件，文本特征缓存仍使用 FLUX.3 自身的编码规则。

运行环境需要 Transformers 5.8.1 或更新版本。当前支持单卡适配器训练、SDPA 和普通梯度检查点；不提供全量微调、文本编码器训练、T-LoRA、编译、块换出或多卡训练。

## MOSS-Image 5B

在模型系列中选择 `MOSS-Image 5B`，DiT 条目为 `MOSS-Image Base 5B`。各条目只下载对应组件；顶部“下载缺失组件”补齐 DiT、文本编码器和 VAE。

DiT 推荐下载使用 Comfy-Org 的 `moss_image_base_bf16.safetensors` 单文件，支持 Hugging Face 和 ModelScope 两个来源。该文件无需配套 `config.json`；导入和加载时会校验官方 Base BF16 文件身份，不接受 SFT、INT8、GPTQ 或 FP16 修正版。已有且通过校验的 Base 目录也会显示为可用，无需重新下载。

主模型继续接受官方完整目录、其中的 `transformer/` 目录，或带有匹配 `config.json` 的原始 safetensors 单文件。分片目录必须保留索引和全部分片。独立文本编码器需要配套 `config.json`、完整权重及本地 `tokenizer/`；程序优先从 `text_encoder/` 的同级目录发现分词器，也可明确填写“分词器”路径。独立 VAE 同样需要匹配的配置和权重。DiT 单文件不包含这些组件；已有的文本编码器、分词器和 VAE 可继续选择使用，“下载缺失组件”只补齐尚缺的组件。

使用 MOSS-Image 发布的 Qwen3.5-9B 文本编码器；Krea 2 / FLUX.3 的 Qwen3-VL 不能替代。配套小解码器 VAE 与现有 FLUX 系列 AE 权重不同，需独立准备。文本编码器固定 BF16，VAE 固定 FP32。DiT 支持未量化 BF16 / FP32，训练时文本编码器与 VAE 保持冻结。

已有 Anima 的训练器 JSON 标签和 Danbooru TXT 标签可直接复用；最终文本由 MOSS-Image 的配套编码器处理，不自动转换成官方 JSON Layout，见 [JSON 标签](JSON_CAPTIONS.md#moss-image-标签复用)。

运行环境需要 PyTorch 2.9 或更新版本、Transformers 5.8.1 或更新版本，以及 Diffusers 0.40 或更新版本。当前支持单卡适配器训练、SDPA 或 FlashAttention 2 注意力和普通梯度检查点；不提供全量微调、文本编码器训练、T-LoRA、FP8 底模、编译、块换出或多卡训练。FlashAttention 2 用于同一批次内文本长度相同的 BF16 计算（批量大小为 1 时总是如此），其余批次使用 SDPA。训练时间步与预览设置见 [训练配置](TRAINING.md#moss-image-时间步)。

## 精度

“跟随模型”根据权重精度和运行平台选择加载方式。Apple MPS 和 CPU 路径使用 FP32。底模存储精度、混合计算精度、可训练参数精度及导出精度是独立设置，详见 [训练配置](TRAINING.md#精度与显存)。

## 文件位置

默认模型目录为 `studio_data/models/`，可在“设置 → 存储路径”的“模型目录”修改。下载器按组件和系列组织文件：

```text
models/
├── diffusion_models/<系列>/<SHA-256 前 12 位>/
├── text_encoders/<系列>/<SHA-256 前 12 位>/
├── vae/<系列或 shared>/<SHA-256 前 12 位>/
├── tagger/<系列>/<版本>/
└── mask/<类型>/<版本>/
```

主模型系列目录为 `anima`、`sdxl`、`krea2`、`flux2`、`flux3`、`moss_image`。文本编码器和 VAE 按系列或共享组件保存；共享以实际配置和权重身份为准。单文件下载放在单独的子目录中：推荐文件以文件 SHA-256 的前 12 位命名，自定义下载以来源地址 SHA-256 的前 12 位命名。`tagger/` 和 `mask/` 存放打标模型与遮罩检测模型，见 [数据集管理](DATASETS.md#模型与运行环境)。已登记路径保持不变。

已下载的 MOSS-Image 完整 Base 包仍可识别、登记和使用。准备缺失组件时复用已校验的相同组件，保留下载器建立的目录结构和共享关联。
