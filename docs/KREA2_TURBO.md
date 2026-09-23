# Krea 2 Turbo 用于采样

**训练仍使用 Raw；Turbo 用于生成图片和比较 LoRA / LoKr 效果。** Turbo 是蒸馏模型，程序会拒绝用它开始训练。它可以登记到模型库，但不能设为训练默认，也不会被“准备训练模型”自动选中。

在模型管理中下载标有“仅采样”的 Turbo，然后在 XYZ 采样页选择它作为采样底模。文本编码器仍是 Krea2 的 Qwen3-VL-4B，VAE 仍是共享的 Qwen-Image VAE，不需要为 Raw / Turbo 各下载一套编码器。只有同属 Krea2 的适配器可以用于这条路径；实际效果还取决于训练数据和适配器。

## 默认参数

- 8 步，Euler；CFG 为 0 时只运行正向提示词条件，不计算负向提示词分支。
- Turbo 使用固定 `mu=1.15`，对应界面里的 Flow Shift 约 `3.15819`。这两个数是不同的表示，不能直接互换。
- 手动设置的步数、CFG 和 Shift 优先。Turbo 的正 CFG 采用官方 `正向 + CFG × (正向 − 负向)` 约定；Raw 保持本程序已有的 CFG 约定与分辨率偏移，不改变旧配置。

上述 Turbo 默认值依据 [Krea 官方模型说明](https://huggingface.co/krea/Krea-2-Turbo) 和 [官方采样代码](https://github.com/krea-ai/krea-2/blob/main/sampling.py)。

## 下载与模型识别

推荐列表提供以下两个经过固定版本、文件大小和 SHA-256 约束的条目。下载完成并校验后才登记。

| 条目 ID | 文件大小 | 用途 |
|---|---:|---|
| `krea2-turbo-fp8` | 13,141,730,784 字节 | FP8 scaled，仅采样 |
| `krea2-turbo-bf16` | 26,283,332,608 字节 | BF16，仅采样 |

来源为 [Comfy-Org/Krea-2 固定版本目录](https://huggingface.co/Comfy-Org/Krea-2/tree/e5ea8b4dd7f38f348b138eb0fe29f92c0e367e96/diffusion_models)。FP8 减少权重存储，运行时还需为激活、缓存和临时计算结果预留内存或显存。

Raw 和 Turbo 的权重结构相同，文件名也可以随意修改。程序不会靠名称或形状猜版本：推荐文件校验完成后会在旁边写一个小型 `.ypuddin.json` 记录，绑定当前文件的大小、时间和文件身份；修改、重新复制文件后，需要重新校验或明确选择版本。未验证的本地模型或自定义下载需要用户按发布说明选择 Raw / Turbo。旧训练配置继续默认 Raw；自动模式不能确认时会报错，不会静默猜测。
