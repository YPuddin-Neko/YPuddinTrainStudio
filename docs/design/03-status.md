# 实现状态（对照 00-architecture.md 里程碑）

更新：2026-09-10

| 里程碑 | 状态 | 证据 |
|---|---|---|
| M0 骨架 / 配置 / 事件 / toy 族 / 最小循环 | ✅ | `tests/unit/test_config.py`（19）、`tests/e2e/test_toy_training.py`：暂停恢复**逐位一致** |
| M1 适配器引擎（LoRA / LoKr / LoHa / Full / DoRA）+ 存取转换 + 工具 | ✅ merge / extract（SVD→LoRA、最近 Kronecker 积→LoKr）/ resize，`tests/unit/test_tools.py` | `tests/unit/test_adapters.py`（62）：factorization 表与 LyCORIS 一致、ΔW≡kron、bypass≡merged、第三方 alpha 约定、fp8 冻结层、注入/还原、存取往返 |
| M2 数据流水线（索引 / 分桶 / 缓存 / caption / 可恢复采样器）+ 验证集 | ✅ | `tests/unit/test_data.py`；e2e 中验证损失确实下降 |
| M3 Anima 族 | 🟡 代码完成并通过 CPU 随机权重测试，**待 GPU 机器上用官方权重验证** | `ypuddin/models/anima/`：vendored Cosmos-Predict2 DiT + Qwen-Image VAE（Apache-2.0，见 `vendor/NOTICE.md`）、Qwen3+T5 文本管线、族封装；`tests/unit/test_anima_vendor.py`（20）、`test_anima_family.py`（7）：两种前缀加载、几何推断、与 sd-scripts 式调用逐位接近、预设命中数、LoKr 前反向、2B 参数量 |
| M4 显存子系统 | 🟡 Block Swap ✅（CPU/MPS 逐位一致）；fp8 冻结底模 ✅（CPU 反量化路径）；激活检查点由族实现；Kahan ✅；8-bit 需 CUDA | `tests/unit/test_block_swap.py`、`test_optim.py` |
| M5 服务 API + 队列 + SSE + 前端对接 | ✅ 后端；🟡 前端对接中 | `tests/e2e/test_service.py`（真实 uvicorn + 子进程训练 + SSE + API 暂停/恢复） |
| M6 文档 / 预设 / 基准对比 | 🟡 | 本目录 + `docs/reference/`；基准待 GPU |

## 与参考项目的差异（已落地的"更好"）

1. **精确暂停 / 恢复**：任意优化器步边界写完整状态（适配器、优化器、调度器、采样器位置、RNG、EMA），恢复后每一步 loss 与不间断训练相同，最终权重 `atol=0` 相等。AnimaLoraStudio 退化为"epoch 末备份"，sd-scripts/diffusion-pipe 依赖 dataloader 重放。
2. **LoKr 自研且修正了 LyCORIS 的 bug**：scale/scalar/multiplier 只在一处应用（LyCORIS 的 `get_diff_weight` 双乘 scale 且丢 scalar）；Linear 默认走 Kronecker 结构化 bypass（不材料化 ΔW）；显式 factorization 写进元数据，加载不靡启发式推断；dropout 语义各路径一致。
3. **Block Swap 前后向双钩子 + 推迟释放**：修掉了"输入无梯度时 full backward hook 提前触发"的隐患（MPS 上会直接报错、CPU 上静默）；开/关 swap 训练结果逐位相同。
4. **确定性验证损失**：固定验证集 × 固定时间步分位数 × 固定噪声种子；成本可配。
5. **结构化事件流**：训练进程写 `events.jsonl`，监督器尾读并推 SSE；stdout 只是日志，永不解析。
6. **内容哈希缓存**：latent 键 = (内容哈希, 桶尺寸, 编码器指纹, 翻转)；改 caption / 重命名文件不失效；文本缓存去 padding 存储。
7. **类型化配置 + Schema 驱动表单**：150 字段的 pydantic 模型导出 JSON Schema（带 `x-ui`），前端零手写表单；`show_when` 前后端同构解析器。
8. **CPU 可测的端到端**：toy 族让 112+ 个测试在几十秒内跑完，包括完整训练与服务流程。

## 待办（按优先级）

1. GPU 机器上用官方 Anima 权重验证：加载、bf16 训练一次、采样出图、LoRA/LoKr 载入 ComfyUI。
2. 激活检查点的 unsloth 式 CPU 卸载；fp8 `_scaled_mm` 路径；bitsandbytes 8-bit 优化器验证。
3. 多 GPU（torchrun DDP）：采样器已按 rank 切分，训练器还需 DDP 包装与 rank 0 保存。
4. 前端：真实后端对接收尾、数据集页图片网格与 caption 编辑体验。
5. 基准：同一数据集/配置下与 sd-scripts、diffusion-pipe 的速度、显存、验证损失对比。

## 运行方式

```bash
cd xiangmuyuanma
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev,models,server]"
.venv/bin/pytest -q                         # 全部 CPU 测试
.venv/bin/ypuddin plan config.toml          # 不加载权重的预检
.venv/bin/ypuddin train config.toml         # 训练（事件写到 <output_dir>/events.jsonl）
.venv/bin/ypuddin serve --port 8765 --data-root ./studio_data   # 服务（前端 dist 存在时同域托管）
```

Anima 配置最小示例：

```toml
[model]
family = "anima"
dit_path = "/models/anima-base-v1.0.safetensors"
text_encoder_path = "/models/Qwen3-0.6B-Base"        # HF 目录或单文件 safetensors
vae_path = "/models/qwen_image_vae.safetensors"
dtype = "bf16"

[dataset]
sources = [{ path = "/data/chara", repeats = 2 }]
resolutions = [1024]
batch_size = 1
caption = { trigger_word = "chara_name", shuffle = true, keep_tokens = 1 }

[adapter]
algo = "lokr"; rank = "full"; alpha = 1.0; factor = 8; preset = "attn-mlp"

[optimizer]
type = "adamw"; lr = 1e-4

[memory]
activation_checkpointing = "block"; blocks_to_swap = 0

[loop]
epochs = 10; grad_accum = 1; mixed_precision = "bf16"

[checkpoint]
output_dir = "outputs/chara-lokr"; name = "chara"; save_every_epochs = 1

[sampling]
enabled = true; every_epochs = 1; prompts = [{ prompt = "chara_name, 1girl, smile" }]

[validation]
enabled = true; split_ratio = 0.1
```
