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

SDXL 双卡 LoRA／LoKr 使用 150 或 225 token 时，训练预览另有版本化的线性层计算策略：仅在主模型预测时使用 BF16 操数、FP32 矩阵运算并返回 BF16，避免海光原生 BF16 预览在相同检查点下产生不同结果。该预览策略不改变适配器因子、文本编码器、VAE、训练反向或验证损失的计算。

这不会降低底模存储精度；FP32 临时运算可能增加显存和耗时。新策略只作用于明确支持的配置，不改变未开启可复现训练的计算方式。新旧策略、算法、DDP／FSDP 和运行环境均会作为续训合同核对；旧版原生 BF16 路径的完整状态不能混入新策略。如需沿用旧状态，应保留原版本及原环境；也可以从普通适配器导出开始新的训练。

海光双卡 Anima LoRA FSDP 仍有一项已知限制：曾观察到连续训练与恢复训练的预览像素不同，而同次适配器权重、优化器、调度器、随机数和数据位置完全一致。后续两轮独立复测均一致，但尚未定位首次差异的原因，因此不能保证这一路径的预览始终逐像素复现。训练状态的精确续训和预览图像的一致性应分别判断。

显存估算按真实 dtype 计算冻结底模分片，仅给训练参数计入梯度与优化器状态。每卡还需容纳当前汇集的模块、激活、缓存和通信临时空间；总显存相加并不代表任何尺寸都能装下，也不保证两倍速度。

验收脚本 `Test/scripts/verify_frozen_adapter_resume.py` 支持 `--strategy fsdp --device cuda --processes 2`，指定新的输出目录。其通过标准仍是权重、完整状态和预览精确一致。各次硬件结果保存在本地 `Test/reports/` 和 `Test/remote-testing/`；本地 CPU 结果不能替代双 GPU 正式模型结果。
