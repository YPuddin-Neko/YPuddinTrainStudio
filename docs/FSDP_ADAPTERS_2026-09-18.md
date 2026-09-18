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

显存估算按真实 dtype 计算冻结底模分片，仅给训练参数计入梯度与优化器状态。每卡还需容纳当前汇集的模块、激活、缓存和通信临时空间；总显存相加并不代表任何尺寸都能装下，也不保证两倍速度。

验收脚本支持 `--strategy fsdp --device cuda --processes 2`，指定新的输出目录。其通过标准仍是权重、完整状态和预览精确一致。当前测试结果见后续本轮报告；本地 CPU 结果不能替代双 GPU 正式模型结果。
