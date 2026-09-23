# LoRA／LoKr 显存分片

## 行为与配置

冻结文本编码器、只训练主模型的 LoRA／LoKr 可选择 `loop.distributed_strategy = "fsdp"` 和至少两张显卡。与 DDP 每卡一份完整底模不同，FSDP 将冻结底模和适配器参数分片；只有适配器产生梯度和优化器状态。小于卡数的微小参数保留副本并显式同步梯度。

- 底模支持 BF16、FP16、FP32；不为分片而自动把底模升成 FP32。适配器训练参数使用 FP32。
- 适配器与底模分别归组，避免旧版 FSDP2 同组混合存储精度限制。底模仍按原生重复模块分组；适配层独立分片，通信次数可能增加，未进行实机吞吐优化验收。
- 支持 AdamW、Adafactor、SGD；不支持文本编码器训练、FP8 底模、块换出、编译和 EMA。整层丢弃会导致各卡使用不同适配层，因此启动前拒绝；普通 Dropout 和 Rank Dropout 保留。
- Windows 原生 FSDP 仍不支持；Linux CUDA／DTK 的硬件验收与 Windows 单卡 CUDA 验收分别记录。检测到两张卡不等于环境具备 FSDP2 所需通信能力。

示例增量配置：

```json
{
  "training": {"mode": "adapter", "train_backbone": true, "train_text_encoder": false},
  "adapter": {"algo": "lora", "param_dtype": "fp32", "module_dropout": 0},
  "loop": {"gpu_count": 2, "distributed_strategy": "fsdp", "mixed_precision": "bf16"},
  "memory": {"base_precision": "auto", "blocks_to_swap": 0, "activation_checkpointing": "block", "compile": false},
  "optimizer": {"type": "adamw", "kahan": false}
}
```

LoKr 将 `adapter.algo` 改为 `lokr`。模型路径、数据、批量、步数和其他参数使用自己的训练配置。

## 保存和恢复

导出仍是普通 LoRA／LoKr safetensors，不需要完整底模导出。各卡逐个汇集适配器到 CPU，按原算法折叠 scalar／alpha；不在 GPU 上重建整份底模。

完整状态保存适配器原始参数、优化器、调度器、每卡随机数、数据位置及模型身份。训练种类和适配器配置单独校验；不能把全参状态当作适配器状态，或更改 alpha／算法后宣称精确续训。FSDP、DDP、不同卡数之间不支持无损切换完整状态。

## 海光 BF16 可复现训练

冻结文本编码器的 Anima／SDXL LoRA，以及 LoKr FSDP，在开启 `loop.deterministic` 时使用分别记录的计算策略。LoRA 的主模型线性层和适配器矩阵运算采用先舍入 BF16、再以 FP32 运算的方式，中间结果及输出仍按 BF16 舍入；SDXL 卷积使用 FP32 后转回 BF16。LoKr 因子运算保持原生实现，主模型线性层按模型和标签长度选择对应策略。前端会显示服务器确认的实际计算方式。

Anima LoRA／LoKr FSDP 多卡，以及 SDXL 单卡、DDP／FSDP LoRA／LoKr 使用 150 或 225 token 时，训练预览另有版本化的线性层计算策略：仅在主模型预测时使用 BF16 操数、FP32 矩阵运算并返回 BF16，以稳定海光原生 BF16 预览在相同检查点下的计算结果。该预览策略不改变适配器因子、文本编码器、VAE、训练反向或验证损失的计算。Anima 单卡和 DDP 不启用这一预览策略。

这不会降低底模存储精度；FP32 临时运算可能增加显存和耗时。新策略只作用于明确支持的配置，不改变未开启可复现训练的计算方式。新旧策略、算法、DDP／FSDP 和运行环境均会作为续训合同核对；旧版原生 BF16 路径的完整状态不能混入新策略。如需沿用旧状态，应保留原版本及原环境；也可以从普通适配器导出开始新的训练。

Anima 的 LoRA／LoKr 预览分别使用 `dtk-anima-backbone-lora-fsdp-bf16-compute-preview-v2` 和 `dtk-anima-backbone-lokr-fsdp-bf16-compute-preview-v2` 计算合同。原生 BF16 路径曾在相同训练状态下出现 VAE 解码之前的预览张量差异；不能仅凭最终训练权重一致判断预览也可复现。验收因此分别核对训练状态与预览，并在独立空用户内核缓存下比较连续训练、恢复训练和多轮结果。此策略针对已观测的主模型预览差异，不将其归因为某个尚未确认的厂商内核。

## 其他海光可复现配置

以下策略只在海光 DTK、开启可复现训练且配置满足范围时启用；默认训练路径不因此改变。底模仍使用 BF16 存储，适配器使用 FP32 参数。前端展示服务器确认的实际计算方式，关闭可复现训练后恢复原来的配置选项。

| 配置 | 训练计算 | 训练预览 |
| --- | --- | --- |
| Anima 单卡 LoRA，FP16 混合精度，冻结文本编码器 | 主模型线性层与 LoRA 保留 FP16 操数舍入、中间结果和输出，矩阵运算使用 FP32；AMP 和 GradScaler 继续工作 | 保持原生实现 |
| SDXL 单卡 LoKr，FP16 混合精度，150 token，BF16 底模，冻结文本编码器 | 保持原生 FP16 训练和 GradScaler | 仅主模型线性层的矩阵运算使用 FP32，保留 FP16 舍入与输出 |
| SDXL 单卡／双卡 DDP LoRA，BF16，150 token，同时训练主模型与两个文本编码器 | 沿用文本编码器 LoRA 的 BF16 舍入、FP32 运算策略 | 增加主模型线性层的 BF16 舍入、FP32 运算；文本预览保持原生实现 |
| Klein Base 4B 双卡 DDP／FSDP LoRA，BF16，冻结文本编码器 | 主模型线性层与 LoRA 保留 BF16 舍入和输出，矩阵运算使用 FP32 | 主模型线性层的 BF16 舍入、FP32 运算 |
| Klein Base 4B 双卡 DDP LoKr，BF16，冻结文本编码器 | 保持原生 BF16 训练 | 主模型线性层的 BF16 舍入、FP32 运算 |
| Klein Base 4B 双卡 FSDP LoKr，BF16，冻结文本编码器 | 主模型线性层的 BF16 舍入、FP32 运算；LoKr 因子运算保持原生实现 | 主模型线性层的 BF16 舍入、FP32 运算 |
| Klein Base 9B 双卡 DDP／FSDP LoRA，BF16，冻结文本编码器 | 主模型线性层与 LoRA 保留 BF16 舍入和输出，矩阵运算使用 FP32；冻结 Qwen3 线性层采用相同精度边界 | 主模型线性层的 BF16 舍入、FP32 运算 |
| Klein Base 9B 双卡 DDP／FSDP LoKr，BF16，冻结文本编码器 | 主模型及冻结 Qwen3 线性层保留 BF16 舍入和输出，矩阵运算使用 FP32；LoKr 因子运算保持原生实现 | 主模型线性层的 BF16 舍入、FP32 运算 |

Klein 策略要求明确选择“基础版 4B”（`model.flux2_variant = "klein-base-4b"`）或“基础版 9B”（`klein-base-9b`）；自动识别不据此推断同一计算策略。这些配置不涵盖 DoRA、混合适配器算法、编译、层换出或多卡 FP16。策略与运行环境写入完整状态；切换策略后不能将旧状态当成严格续训。普通适配器导出仍可用于新训练或推理。实际显存、速度和正式硬件通过范围以对应验收记录为准，短程状态一致不代表长期学习质量已验证。

9B 的冻结 Qwen3 策略为 `qwen3-bf16-linear-fp32-v1`，训练策略使用独立的 `dtk-klein9b-backbone-{lora|lokr}-{ddp|fsdp}-bf16-compute-preview-v4` 身份。文本缓存键和完整续训模型身份包含实际编码策略，旧原生编码缓存不能混用。FP32 临时矩阵运算可能增加显存和耗时，不代表整个文本编码器改为 FP32 存储，也不启用文本编码器训练。

SDXL 的预测方式必须匹配底模：普通 epsilon 模型使用 epsilon；v-pred 模型使用 v prediction，并按发布方说明设置 Zero SNR。切换选项不会将普通 epsilon 权重变成 v-pred 权重。

显存估算按真实 dtype 计算冻结底模分片，仅给训练参数计入梯度与优化器状态。每卡还需容纳当前汇集的模块、激活、缓存和通信临时空间；总显存相加并不代表任何尺寸都能装下，也不保证两倍速度。

验收脚本 `Test/scripts/verify_frozen_adapter_resume.py` 支持 `--strategy fsdp --device cuda --processes 2`，指定新的输出目录。其通过标准仍是权重、完整状态和预览精确一致。各次硬件结果保存在本地 `Test/reports/` 和 `Test/remote-testing/`；本地 CPU 结果不能替代双 GPU 正式模型结果。
