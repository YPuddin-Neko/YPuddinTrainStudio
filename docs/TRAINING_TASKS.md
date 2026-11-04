# 滑块与原生编辑 LoRA

在训练参数和参数预设的“训练方式”下拉中选择滑块或编辑训练。滑块与编辑使用适配器模式、单设备和冻结文本编码器。

配置文件中的 `training.task` 区分 `standard`、`slider` 和 `edit`，默认 `standard` 沿用普通训练；`training.mode` 区分适配器和全量微调。

| 任务 | 模型 | 数据 |
| --- | --- | --- |
| `slider` / `text` | SDXL、Anima、Krea 2 Raw、FLUX.2 Klein Base、FLUX.3 Klein Base、MOSS-Image | 中性、正向和负向提示词 |
| `slider` / `image` | 同上 | 严格配对的正向、负向图片 |
| `edit` | FLUX.2 Klein Base 4B / 9B、FLUX.3 Klein Base | 目标图片、每张目标对应的一张参考图、编辑指令 |

滑块目标层必须使用普通 LoRA，关闭 DoRA；目标层规则同样不能改用其他适配器算法。编辑任务使用底模原生参考图条件。MOSS-Image 的文生图训练不提供原生编辑入口。

## 在页面中配置

1. 打开版本的“训练参数”，在“训练方式”中选择“滑块 LoRA”或“编辑 LoRA”。滑块另选“文字方向”或“图片配对”。参数预设使用同一入口。
2. 文字滑块填写中性、正向和负向提示词，设置每轮样本数、生成分辨率和生成步数，无需先导入图片。
3. 图片滑块或编辑 LoRA 沿用原来的图片导入流程。在“数据与分桶”的训练数据源中选择训练图片目录，再为各来源填写负向配对目录或参考图片目录。正向／目标图使用训练图片目录；负向／参考图由对应字段引用。
4. 开启训练预览时，编辑 LoRA 的每条预览提示词需配置一张参考图。完成参数检查后启动任务，结果仍在原任务页保存与管理。

切换训练方式会保留各方式的参数和数据源。导入页不要求额外上传配对图片；目录由训练参数决定。参数预设不携带数据目录，载入后需在当前版本绑定来源与配对目录。

## 文字滑块

```toml
[training]
mode = "adapter"
task = "slider"

[adapter]
algo = "lora"
dora = false

[slider]
mode = "text"
neutral_prompt = "a portrait"
positive_prompt = "a smiling portrait"
negative_prompt = "a serious portrait"
guidance = 3.0
anchor_prompts = []
anchor_weight = 1.0
samples_per_epoch = 16
resolution = 512
sampling_steps = 20
```

文字模式不要求上传图片。任务开始时使用冻结底模和中性提示词生成概念样本，实际采样结果保存在任务的 `slider-bank` 中。每个样本使用 `loop.seed + 1000 + index` 的独立种子；采样器、调度和引导参数沿用采样配置。生成过程不推进训练随机数状态。`resolution` 向下对齐到模型要求的倍数，作为概念样本的正方形尺寸。

已有图片来源和配对目录会保留，文字模式只使用生成的概念样本。切回图片训练时，按所选训练方式检查对应的数据与配对目录。

训练在同一噪声输入和时间步上取得冻结底模的中性、正向、负向预测。正方向学习 `neutral + guidance * (positive - negative)`，负方向学习相反偏移；两个方向各占一半权重。使用当前模型自身的预测参数化和损失加权。`anchor_prompts` 可加入需保持的底模预测，损失按提示词数取平均后乘以 `anchor_weight`。

每个概念样本计一次训练项，两个方向不会把样本数或优化步数翻倍。无原图时计划显示 `images=0`、`items=samples_per_epoch`。当前文字模式使用分桶批次，尺寸由 `slider.resolution` 决定；不使用图片验证集或块换出。

## 图片滑块

```toml
[training]
task = "slider"

[slider]
mode = "image"

[[dataset.sources]]
path = "/data/smile"
pair_path = "/data/serious"
```

`path` 是正向图片目录，`pair_path` 是负向图片目录。按相对路径去掉图片扩展名后一一配对；例如 `people/001.png` 可以对应 `people/001.jpg`。重复名称、缺少对应项、多余对应项或原始尺寸不同都会报出配置错误。两端使用同步裁剪、翻转和最终画布，标签取自正向图片，宜描述两端共同内容。

每对图片共用时间步和噪声，分别在 LoRA 倍率 `+1`、`-1` 下训练当前模型的原生目标。负向图片有自己的损失遮罩。每对只计一个逻辑样本，保留批量和梯度累积的含义。文字和图片滑块均不支持 InfoNoise。

## 原生编辑

```toml
[training]
task = "edit"

[[dataset.sources]]
path = "/data/edited"
reference_dir = "/data/original"
```

配对规则同上，参考图与目标图可以有不同尺寸。目标图按现有数据设置处理，参考图保持比例缩放后使用边缘填充到目标的最终画布。模型同时收到带噪目标、干净参考图和目标图片标签中的编辑指令；只对目标预测计算损失。

FLUX.2 使用原生参考图位置标识，输出裁回目标 token；FLUX.3 使用独立的 `image_cond` 输入和原生参考图标识。FLUX.3 可选择 `edit-attn-only` 或 `edit-attn-mlp` 目标预设，将参考分支一并纳入适配器。

训练预览的每条 `sampling.prompts` 可填写 `reference_images`，编辑任务必须提供一张。模型测试的请求使用同名字段；当前最多一张。模型测试提交时复制参考图到任务私有目录，历史和重试使用冻结副本，原来源图片的后续变化不会替换已经提交的输入。

## 保存、恢复与使用

恢复点保存任务配方；普通、滑块和编辑任务不能相互精确续训。修改文字端点、概念库、配对内容、对应顺序或参考图处理语义后，需要恢复原设置，或从导出适配器开始新任务。文字滑块的概念库缺失或变化时不会悄悄重新生成；移动恢复点时需保留其概念库。

导出仍为现有 LoRA 格式，并记录 `ypuddin.training_task` 和配方版本。滑块可在模型测试中使用适配器强度轴比较 `-1 / 0 / +1`；编辑 LoRA 需在支持对应底模原生参考图条件的推理流程中使用。

文字双向预测配方参考 [Fizgig 的训练实现](https://github.com/shootthesound/Fizgig/blob/694b7609169200e07159aa05b8819a1c06c758cd/src/fizgig/families/train.py)，配对噪声原则参考 [AI Toolkit 的图片参考滑块训练](https://github.com/ostris/ai-toolkit/blob/e9c72b762748e4d31837fbeb3b87d01a5b143321/extensions_built_in/image_reference_slider_trainer/ImageReferenceSliderTrainerProcess.py)。原生编辑输入遵循 [FLUX.2 Klein](https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora) 和 [FLUX.3 Klein](https://github.com/black-forest-labs/flux-3-image-klein) 的模型条件。
