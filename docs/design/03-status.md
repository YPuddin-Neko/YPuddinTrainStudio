# 实现状态（对照 00-architecture.md 里程碑）

更新：2026-09-10

| 里程碑 | 状态 | 证据 |
|---|---|---|
| M0 骨架 / 配置 / 事件 / toy 族 / 最小循环 | ✅ | `tests/unit/test_config.py`（19）、`tests/e2e/test_toy_training.py`：暂停恢复**逐位一致** |
| M1 适配器引擎（LoRA / LoKr / LoHa / Full / DoRA）+ 存取转换 + 工具 | ✅ merge / extract（SVD→LoRA、最近 Kronecker 积→LoKr）/ resize，`tests/unit/test_tools.py` | `tests/unit/test_adapters.py`（62）：factorization 表与 LyCORIS 一致、ΔW≡kron、bypass≡merged、第三方 alpha 约定、fp8 冻结层、注入/还原、存取往返 |
| M2 数据流水线（索引 / 分桶 / 缓存 / caption / 可恢复采样器）+ 验证集 | ✅ | `tests/unit/test_data.py`；e2e 中验证损失确实下降 |
| M3 Anima 族 | 🟡 代码完成；**整条训练链路已在 CPU 上用缩小版组件跑通**（真实 transformers Qwen3 → LLM adapter → DiT → VAE → 缓存 → LoKr → 验证 → 采样 → 导出 → ComfyUI 键转换），**待 GPU 机器上用官方权重验证**（`ypuddin smoke`） | `ypuddin/models/anima/`：vendored Cosmos-Predict2 DiT + Qwen-Image VAE（Apache-2.0，见 `vendor/NOTICE.md`）、Qwen3+T5 文本管线、族封装；`tests/unit/test_anima_vendor.py`（21）、`test_anima_family.py`（10）、`tests/e2e/test_anima_pipeline.py`（3：online/cached 文本模式全流程 + 转换往返） |
| M3b Krea 2 族 | 🟡 代码完成；**整条训练链路已在 CPU 上用缩小版组件跑通**（真实 transformers Qwen3-VL 解码器 → 12 层隐状态堆叠 → SingleStreamDiT → VAE → 缓存 → LoKr → 验证 → 分辨率自适应 shift 采样 → 导出 → 合并回 fp8_scaled 底模），**待 GPU 机器上用官方权重验证** | `ypuddin/models/krea2/`：vendored musubi-tuner `SingleStreamDiT`（Apache-2.0，见 `vendor/NOTICE.md`；GQA 由共享 attention 处理）、Qwen3-VL 文本管线（官方提示词模板 / 层选择 / 去前缀 / 去 padding 缓存）、族封装（bare / `model.diffusion_model.` / Comfy-Org **fp8_scaled** 三种检查点，fp8 层带文件 scale 直接冻结）、4 个目标预设、planner 几何推断；`tests/unit/test_krea2_family.py`（17）、`tests/e2e/test_krea2_pipeline.py`（3） |
| M4 显存子系统 | 🟡 Block Swap ✅（CPU/MPS 逐位一致）；fp8 冻结底模 ✅（CPU 反量化路径）；激活检查点由族实现；Kahan ✅；8-bit 需 CUDA | `tests/unit/test_block_swap.py`、`test_optim.py` |
| M5 服务 API + 队列 + SSE + 前端对接 | ✅ 后端（全部 JSON 端点带响应模型；`GET /families` 让族/预设/文本模式数据驱动）；✅ 前端 FE-M1~M6 全部验收（真实后端全流程、数据集页、项目/模型/设置页、图表拆分、类型由 openapi 生成；截图 `frontend/screenshots/01~24`；Vitest 46）；🟡 FE-M7（Krea 2 接入前端）已下发 Kimi | `tests/e2e/test_service.py`（真实 uvicorn + 子进程训练 / 预缓存任务 + SSE + API 暂停/恢复 + 端点扫描） |
| M6 文档 / 预设 / 基准对比 | 🟡 内置预设 anima×2 / krea2×2 / toy；部署文档 `docs/deploy.md` | 本目录 + `docs/reference/`；基准待 GPU |

## 与参考项目的差异（已落地的"更好"）

1. **精确暂停 / 恢复**：任意优化器步边界写完整状态（适配器、优化器、调度器、采样器位置、RNG、EMA），恢复后每一步 loss 与不间断训练相同，最终权重 `atol=0` 相等。AnimaLoraStudio 退化为"epoch 末备份"，sd-scripts/diffusion-pipe 依赖 dataloader 重放。
2. **LoKr 自研且修正了 LyCORIS 的 bug**：scale/scalar/multiplier 只在一处应用（LyCORIS 的 `get_diff_weight` 双乘 scale 且丢 scalar）；Linear 默认走 Kronecker 结构化 bypass（不材料化 ΔW）；显式 factorization 写进元数据，加载不靡启发式推断；dropout 语义各路径一致。
3. **Block Swap 前后向双钩子 + 推迟释放**：修掉了"输入无梯度时 full backward hook 提前触发"的隐患（MPS 上会直接报错、CPU 上静默）；开/关 swap 训练结果逐位相同。
4. **确定性验证损失**：固定验证集 × 固定时间步分位数 × 固定噪声种子；成本可配。
5. **结构化事件流**：训练进程写 `events.jsonl`，监督器尾读并推 SSE；stdout 只是日志，永不解析。
6. **内容哈希缓存**：latent 键 = (内容哈希, 桶尺寸, 编码器指纹, 翻转)；改 caption / 重命名文件不失效；文本缓存去 padding 存储。
7. **类型化配置 + Schema 驱动表单**：150 字段的 pydantic 模型导出 JSON Schema（带 `x-ui`），前端零手写表单；`show_when` 前后端同构解析器。
8. **CPU 可测的端到端**：toy 族让 170 个测试在几十秒内跑完，包括完整训练与服务流程。
9. **cached 文本模式支持 caption 增强**：预缓存每张图有界、确定性的 caption 变体（`caption.cache_variants`），shuffle / tag_dropout / wildcard 在卸载文本编码器后仍可用（sd-scripts 在缓存 TE 输出时直接禁止这些选项）；caption_dropout 在采样时按概率精确生效；采样提示词与负面词一并预缓存。
10. **工具链对官方权重文件格式友好**：`extract` / `merge` 直接吃 `net.`（anima-base）或 `model.diffusion_model.`（ComfyUI）前缀的整模型文件，适配器键始终是 kohya 裸模块名；`convert` 转 ComfyUI 时 LoKr/LoHa 模块的 alpha 与权重同留（LoRA 走 PEFT 键）。
11. **一条命令自检**：`ypuddin smoke` 用真实 trainer 跑几步 + 出图 + 保存/回读 + 报告（含峰值显存与 traceback），新机器/新权重排障不用来回猜。
12. **多族同一套代码路径**：Krea 2（12.9B fp8_scaled 底模 + 4B 文本编码器）与 Anima 共用 trainer / 缓存 / 适配器 / 服务，族只描述“加载、前向、预设、显存布局、采样 shift”；ComfyUI 的 fp8_scaled 文件按 fp8 直接冻结（沿用文件 scale，不回 bf16），`merge` 合并回去仍是 ComfyUI 可读的 fp8_scaled；musubi-tuner 对同一模型需要单独的 `krea2_*` 脚本与缓存流程。
13. **服务契约有类型**：全部 JSON 端点带 pydantic 响应模型，OpenAPI 直接生成前端 TS 类型；`ypuddin serve` 托管前端时支持深链接刷新。
14. **激活卸载可选**：`memory.activation_checkpointing = "unsloth"` 走非阻塞 CPU 卸载的检查点（与逐块重算在 CPU 上梯度一致）；`model.attention = "sage"` 可选 SageAttention（仅图像自注意力，掩码交叉注意力回落 SDPA）。

## 待办（按优先级）

1. GPU 机器上用官方 Anima / Krea 2 权重验证：加载、bf16 / fp8 训练一次、采样出图、LoRA/LoKr 载入 ComfyUI。
2. fp8 `_scaled_mm` 路径；bitsandbytes 8-bit 优化器验证；unsloth 卸载 / sage / compile 在 CUDA 上的实测。
3. 多 GPU（torchrun DDP）：采样器已按 rank 切分，训练器还需 DDP 包装与 rank 0 保存。
4. 前端：按 GPU 实测反馈打磨（显存曲线、ETA、错误提示），i18n 文案补全。
5. 基准：同一数据集/配置下与 sd-scripts、diffusion-pipe 的速度、显存、验证损失对比。

## 运行方式

部署与日常使用见 `docs/deploy.md`（`studio.sh` / `studio.bat` 一键安装启动）。以下是开发者视角的命令：

```bash
cd xiangmuyuanma
uv venv --python 3.12 venv && uv pip install --python venv/bin/python -e ".[dev,models,server]"
venv/bin/pytest -q                         # 全部 CPU 测试
venv/bin/ypuddin plan config.toml          # 不加载权重的预检
venv/bin/ypuddin train config.toml         # 训练（事件写到 <output_dir>/events.jsonl）
venv/bin/ypuddin serve --port 8765 --data-root ./studio_data   # 服务（前端 dist 存在时同域托管）
```

**GPU 机器首次验证（一条命令）**：真实跑 3 步 + 出一张 512 预览 + 保存/回读适配器，输出时序 / 峰值显存 / loss / 键格式，报告写到 `outputs/smoke/smoke-report.json`（失败时含完整 traceback，直接贴给我即可）：

```bash
venv/bin/ypuddin smoke \
  --set model.family=anima \
  --set model.dit_path=/models/anima-base-v1.0.safetensors \
  --set model.text_encoder_path=/models/Qwen3-0.6B-Base \
  --set model.vae_path=/models/qwen_image_vae.safetensors \
  --set adapter.algo=lokr --set adapter.rank=full --set adapter.factor=8 \
  --set memory.activation_checkpointing=block
# 显存紧张时追加：--set memory.blocks_to_swap=10 --set dataset.text_encoding=cached --resolution 512
# 也可直接给配置文件：ypuddin smoke my.toml --steps 5
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
