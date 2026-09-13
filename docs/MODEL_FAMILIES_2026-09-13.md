# SDXL、FLUX.1、FLUX.2 接入与验收

2026-09-13。新增原生 ModelFamily 后端、缓存/训练/采样接口、版本模型类型、本地权重识别与前端配置。没有调用其他训练器代跑，也没有修改现有项目的图片、标签或模型文件。完整权重 GPU 验收与训练质量仍独立于下面的小模型测试。

## 支持范围

| 模型 | 训练条件与组件 | 本轮实现 | 明确限制 |
| --- | --- | --- | --- |
| SDXL | UNet、CLIP-L + CLIP-G、SDXL VAE；DDPM | ε / v 预测、可选零终端 SNR、LoRA / LoKr、遮罩、分桶、预览与保存恢复 | SDXL base 架构；不包含 refiner；不支持 FP8、block swap、FlashAttention、compile、在线文本编码 |
| FLUX.1 | dev / schnell；CLIP-L + T5-XXL、FLUX AE；flow | 区分 dev/schnell 条件和默认采样、LoRA / LoKr、冻结主干 FP8 存储、block checkpoint、遮罩 | 目前 auto/SDPA；不支持 block swap、compile、Kontext/Fill/Control |
| FLUX.2 | dev、Klein base 4B / 9B；Mistral3 / Qwen3、FLUX.2 VAE；flow | 分变体文本特征、VAE BN 归一化、LoRA / LoKr / LoKr Full、block checkpoint + block swap、遮罩 | 不支持 Klein 蒸馏/KV、编辑条件、FP8/量化文本编码器、compile、在线文本编码 |

模型族支持的注意力、采样器、调度器和目标函数由后端 API 提供，前端只展示对应选项。CLI/API 同样在开始训练前拒绝不支持的组合，不会因为表单隐藏了字段而静默忽略其值。

## 模型选择

新建项目和版本可以选择 SDXL、FLUX.1、FLUX.2。不同模型族使用独立的默认训练配置；同族复制保留参数，跨族复制保留数据准备配置并重建训练参数。已注册且存在的默认组件按模型族自动填入。手动选定的路径优先。

完整 Diffusers 目录以根目录登记；扫描不会将内部权重分片当成几个独立模型。独立 DiT/UNet 可以单独登记，但训练前必须补齐实际所需的文本编码器、VAE 和分词器。检测基于 safetensors 头部及受限的 JSON 配置，不执行 pickle，也不靠文件名猜模型。

SDXL 支持完整 A1111/LDM `.safetensors` checkpoint 和本地 Diffusers 目录，用户可手选第三方 SDXL base 权重。完整模型中的双 CLIP/VAE 默认直接使用，覆盖路径放在高级设置；“SDXL 预测方式”在普通设置直接显示。模型元数据不可靠时，由用户明确选择 ε 或 v；切回 ε 会清除零终端 SNR。

默认下载是 [Illustrious-XL（光辉）v0.1](https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/blob/f08f0826ffe32183ba2d1f4106dd5b32e195a02e/Illustrious-XL-v0.1.safetensors)，使用常规 ε 预测。目录内置的是下载记录，不内置巨型权重：

- HF revision：`f08f0826ffe32183ba2d1f4106dd5b32e195a02e`
- 文件：`Illustrious-XL-v0.1.safetensors`，6,938,040,760 字节。
- SHA-256：`3e15ba00387db678ab4a099f75771c4f5ac67fda9e7100a01d263eaf30145aa9`。
- 只提供已验证的 HF 来源；没有把未验证的魔搭同名模型当成同一权重。

FLUX.1 独立 DiT 需要 CLIP-L、T5-XXL、AE，界面分别显示。完整 pipeline 目录可使用其内含组件。FLUX.2 文本编码器需要完整本地 HF 目录，包括配置、权重及 tokenizer/processor；单文件 TE 下载会在联网前被拒绝。

Klein base 与蒸馏版的张量尺寸相同，不能靠权重尺寸判断。自动识别需要目录元数据明确声明 `is_distilled=false`；单文件缺少依据时必须选择 Klein base 4B 或 9B，已知蒸馏声明会被拒绝。默认自动识别不猜测它是 base。

## 训练与预览语义

SDXL 使用 1000 步 scaled-linear DDPM 噪声日程。ε 目标是加入的噪声；v 目标结合干净 latent 和噪声，采样时做对应转换。零终端 SNR 仅能配合 v 预测。双 CLIP 的倒数第二层、第二编码器 pooled embedding，以及每张图的原始尺寸/裁切坐标/目标尺寸共同传入 UNet；整图填充时裁切偏移为零。

SDXL 当前预览为 Euler / Heun + uniform，训练与采样都使用自己的 DDPM 实现。不能将 Anima 的 flow 时间偏移或 ER-SDE 直接套到 SDXL；本轮没有对这些未接入组合提供入口。

FLUX 的“模型引导值”与 CFG 是不同参数：前者送入带 guidance embedding 的模型；后者混合正、负提示词预测。训练引导值默认 1；预览按加载后的真实变体取默认值，用户设置优先：

| 变体 | 预览步数 | CFG | 模型引导值 |
| --- | ---: | ---: | ---: |
| FLUX.1 dev | 28 | 1 | 3.5 |
| FLUX.1 schnell | 4 | 1 | 无 guidance embedding |
| FLUX.2 dev | 50 | 1 | 4 |
| FLUX.2 Klein base 4B / 9B | 50 | 4 | 无 guidance embedding |

FLUX.1 dev 保留 512 个 T5 token，schnell 保留 256 个。FLUX.2 保留 512 个 token，dev 取 Mistral3 的第 10/20/30 层，Klein 取 Qwen3 的第 9/18/27 层。FLUX.2 预览时间偏移同时依据图片 token 数和实际步数计算。正则图生成复用同一模型采样接口，并记录实际 shift/guidance。

## 内存与卸载

FLUX.1 / FLUX.2 先创建不分配真实权重的 meta 主干。VAE 与文本编码器按缓存阶段加载，用完卸载；缓存完成后才读取主干真实权重并注入适配器，避免准备阶段主干和大文本编码器同时驻留。已有有效缓存直接复用。

训练时只更新适配器；冻结主干仍参与前后向。block checkpoint 用重算减少保存的激活；FLUX.2 block swap 把指定冻结块留在 CPU、按前后向换入，必须配合 block checkpoint。独立正则生成在 VAE 解码前把主干停放 CPU；训练中的预览目前仍会叠加常驻主干、优化器与 VAE，缓存模式在本轮预览结束后卸载 VAE。两条路径不能当作相同的峰值。训练预览进一步分阶段卸载需要单独验证参数/优化器引用、swap 钩子与异常后的恢复。

这不等于逐层卸载文本编码器：建文本缓存时整个 TE 仍需放进所选设备。FLUX.2 dev 的 24B TE、Klein 9B 的 8B TE 均不能据此承诺在 16GB 显卡上可跑。缩小分辨率能降低图像激活开销，不能解决文本编码器权重本身的占用。完整 GPU 试验应优先 Klein base 4B，单张缓存、低分辨率短训练，并单独测准备峰值和训练峰值。

Anima/Krea 的既有路径和新 FLUX 路径并不完全相同，不能用一个模型的测试显存代表其他族。与 AnimaLoraStudio 的源码比较以两边实际加载顺序、块调度和损失语义为依据，没有得出本项目整体更快或更省显存的结论。

## 验证证据与边界

- SDXL：真实缩小 UNet + 双 CLIP + VAE，覆盖 ε/Euler 与 v/零终端 SNR/Heun；缓存→两步训练→预览→保存回读，LoRA/LoKr 梯度、完整单文件转换、缺权重拒绝与目录迁移身份。
- FLUX.1：真实缩小 dev/schnell，双编码器、AE、原始 BFL 权重转换、checkpoint 梯度及完整 Trainer 链路。与独立原始网络对照前向最大误差 `3.58e-7`。
- FLUX.2：真实缩小 dev/Klein；实际 Mistral3/PixtralProcessor、Qwen3 条件、VAE、分片加载、完整 Trainer 链路。checkpoint/swap 与不开启时逐个适配器梯度一致，包含 LoKr Full。
- 独立导出对照调用参考 ComfyUI 实际 LoRA/LoKr 加载计算：FLUX.1 每组 34 个默认目标、FLUX.2 六组各 30 个目标均映射成功，权重增量误差为零。FLUX.2 回到原始命名再严格转换的前向误差不超过 `5.07e-7`。
- 默认 FLUX 导出前缀为 `lora_transformer`，本项目恢复与上述 ComfyUI 数值映射已验。Diffusers 0.40 的 FLUX.1 MLP 转换、FLUX.2 BFL/Kohya 命名转换有不同限制，不能宣称默认文件可直接用其 pipeline 加载。另有显式 LoRA→PEFT 转换对照，PEFT 0.20 的真实独立加载通过；这不代表 PEFT 支持 LoKr。
- 本轮没有下载完整模型或新增 Windows GPU 训练，没有验证完整权重的速度、显存、图像质量，也没有启动 ComfyUI 应用做最终画面验收。

测试入口：`tests/unit/test_sdxl_family.py`、`test_ddpm_objective.py`、`test_ddpm_sampling.py`、`test_flux_family.py`、`test_flux2_family.py`、`test_conditioning_geometry.py`、`test_model_inspection.py`、`test_flux_model_inspection.py`、`test_model_scan.py`、`frontend/tests/sdxlModelIntegration.test.tsx`。最终全量后端 **1473 通过 / 3 CUDA 跳过**；最后热启动修复后相关训练/恢复回归 **45 通过 / 2 CUDA 跳过**（与全量有重叠，不相加）。前端 **76 文件 / 539 通过**，ESLint、TypeScript、生产构建、Ruff 与 Git 内容检查通过。隔离真实浏览器完成项目/版本创建、v/Klein 选择保存与采样选项检查。机器记录见 [验收 JSON](validation/model-families-2026-09-13.json)。

模型实现通过已安装的 Diffusers/Transformers 接口复用 Apache-2.0 实现；SDXL 分词器/config 的固定来源和许可证在 `ypuddin/models/sdxl/assets/`。参考训练器和 ComfyUI 仅用于只读比较/独立验收，没有把 GPL 源码拷入本项目。
