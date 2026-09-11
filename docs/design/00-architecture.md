# YPuddin Trainer 总体架构设计

## v0.4 现行架构补充（2026-09-11）

当前实现与最终验收入口为 [v0.4 项目版本报告](../UI_VERSIONS_2026-09-11.md)。本轮再次实际运行 AnimaLoraStudio 0.27.0，并独立实现项目/版本工作区与设置组织；最终测试和浏览器证据由本轮验收补充，不套用历史数量。

- **版本是后端实体**：SQLite 的 `project_versions` 保存来源、准备状态、归档、进度和维护标识；数据源、任务和产物保存 `version_id`。项目的 `active_version_id` 仅用于默认进入，显式版本路由/API 和旧任务绑定始终优先。
- **真实文件副本**：`server/versions.py` 将训练源、显式验证源、图片、caption 和 Mask 复制到 `projects/<pid>/versions/<vid>/datasets/`，重写来源路径并重新索引，不使用硬链接。也支持保留参数的空数据版本、默认配置空白版本。复制期间锁住源版本的写操作；失败显示原因并保留原数据。
- **兼容迁移**：启动时事务式补表/归属列，把原项目关联到兼容 v1；旧文件、任务运行路径和任务配置快照不移动、不重写。新任务输出与缓存按项目/版本分目录；旧任务恢复和重试保持原归属。
- **工作区和结果**：项目页与训练页共用项目/版本标识、版本栏和“数据 → 模型 → 参数 → 结果”四步导航；版本可比较、归档、查看目录。任务结果按版本查询，采样按任务读取，产物保留任务/版本关系。顶栏按 CPU/内存/GPU/硬盘排列，GPU 单独显示占用/显存/功率/温度。
- **设置与开发默认值**：设置从工作区打开为宽抽屉，固定运行环境、模型权重、存储路径、界面与服务四类，训练产物回到项目结果。环境计划/应用/重启门禁保持；模型下载由后端继续。开发模式默认连接真实服务，只有 `VITE_USE_MOCK=true` 显式启用模拟。
- **保证边界**：任务保存配置快照，版本数据仍可编辑，精确续训依赖状态指纹；高级 TOML 的外部路径不自动变成独立文件副本。正式权重、Windows NVIDIA、真实 CUDA 扩展安装与性能基准仍需单独验证。

## v0.3 历史架构补充（2026-09-11）

本节保留 v0.3 的实现与验收快照；项目目录、缓存归属和设置产物入口已由顶部 v0.4 补充替代。历史记录见 [v0.3 界面改造报告](../UI_REDESIGN_2026-09-11.md)，不能视为当前完整功能清单或正式模型/CUDA 的验收。

- **训练工作台**：项目配置仍以 `TrainConfig` JSON Schema 为单一字段来源。React 工作台将其组织为“训练参数 / 数据与分桶 / 模型与输出 / 采样与高级”四个标签，使用紧凑表单、搜索与条件显示；错误通过字段位置切换标签并聚焦，配置草稿自动保存，页面内导航等待保存。分桶和步数消费后端 Plan，入队再次进行服务端预检。
- **真实数据与遮罩链路**：上传、索引与项目配置同步后进入训练；手工 Mask 编辑提供笔刷、撤销、重做和灰度 PNG 保存。白色参与学习，黑色忽略，训练需启用 `dataset.masked_loss`。遮罩保存独立于 latent 缓存，使用 revision 冲突检查、项目活动任务锁和原子文件替换；编辑上限为 16,777,216 像素、单边 8192，不包含自动头部检测。
- **设置与环境服务**：`/settings/environment` 下的 runtime / models / artifacts 分别承载运行环境、模型权重和训练产物；`/settings/preferences` 承载存储、监听与界面偏好。`routes_environment.py` / `EnvironmentManager` 提供可选依赖状态、wheel 上传、变更计划和显式应用，保护现有 Torch / CUDA / NumPy 等运行时。环境维护状态阻止调度，修改后要求重启；实际 CUDA wheel 安装尚未实测。
- **执行与数据边界**：FastAPI / SQLite 管理项目、任务与产物，监督器启动独立训练或缓存进程，以结构化事件转换为 SSE。单设备任务独占与内存准入继续生效；MPS 按 FP32 和统一内存处理。编码器缓存使用真实资产内容指纹，完整续训状态为 state v2，旧数据指纹状态不兼容。多 GPU DDP、正式权重兼容与显存速度保证不能由 Toy 测试推导。
- **交付更新**：`git pull` 或替换发布源码后，停止并重新运行 `studio.bat` / `studio.sh`，启动器才检查依赖安装签名及前端源码、输出内容指纹。已有预编译前端匹配时无需 Node；需重建时要求 Node.js 20.19+（20.x）或 22.12+。签名匹配跳过安装不是环境健康验证；没有生产热更新或联网程序自更新。

本轮实际隔离运行 AnimaLoraStudio **0.27.0 / `3d9d2e86045b879cd19c01ac4aa2337f45a283ab`**，并在浏览器查看训练、数据预览、Mask、环境与模型下载。参考原目录未改动，本项目独立实现，未复制参考源码。已完成的本轮快照为后端 371 passed / 3 CUDA skipped、前端 115 项；浏览器 Toy MPS 遮罩训练完成 5 步和产物下载。详细证据与未验证边界以上述报告为准。

---

## 历史设计基线（v0.1）

以下保留 2026-09-10 的设计内容及后续局部修订，供理解决策来源；其中目标、接口草图、里程碑及对参考项目的比较不应视为现行功能清单，也不代表对 AnimaLoraStudio 0.27.0 的评价。与本节冲突时，以顶部 v0.4 补充、当前源码和实际验收为准。

历史状态：Accepted（v0.1 设计基线） · 日期：2026-09-10

本文回答三个问题：我们要做什么、为什么现有训练器不够好、我们打算怎么做。
对四个参考项目的逐文件分析见 [`../reference/`](../reference/)。

---

## 1. 目标与非目标

**目标**

1. 一个**可作为库导入**、**类型化配置**、**无全局状态**的扩散模型 LoRA / LoKr 训练核心（`ypuddin`），首发支持 **Anima**（Cosmos-Predict2 风格 MiniTrainDIT + Qwen3-0.6B + 6 层 LLM Adapter + Qwen-Image VAE，Rectified Flow）。
2. **自研适配器引擎**：LoRA / LoKr / LoHa / 全量微调，DoRA 权重分解作为可选开关；LoKr 以 LyCORIS 为数学参考重新实现，并做出 LyCORIS 没有的改进（Kronecker 结构感知的 bypass 前向、fp8 冻结底模、更好的初始化与提取工具）。
3. **显存工程一等公民**：fp8（带缩放）冻结底模、前后向都挂钩的 Block Swap、Unsloth 式激活卸载、Kahan 补偿的 bf16 参数、8-bit 优化器、文本编码器/VAE 编码后卸载。
4. **正确性优先**：任意优化器步边界都可**精确暂停 / 恢复**（完整状态快照，含采样器位置与 RNG）；**确定性验证损失**（固定验证集 × 固定时间步网格 × 固定噪声），让不同 run、不同 checkpoint 可比。
5. **可观测**：结构化事件流（JSON Lines），服务端以 SSE 推给前端；训练进程与服务之间**不用解析 stdout**。
6. **面向前端的服务 API**：项目 / 数据集 / 预设 / 任务队列 / 实时监控 / 产物管理 / 模型权重注册；配置表单由 **JSON Schema + UI 提示**自动渲染。
7. **CPU 可测**：一个玩具模型族（Toy DiT）让整条训练链路在几秒内跑完端到端测试；所有 CUDA 专属能力都是可选依赖。
8. 许可证 **Apache-2.0**。

**非目标（v1）**

- 视频模型、非 Rectified-Flow 目标（设计上留扩展位，v1 明确拒绝）。
- 流水线并行（DeepSpeed）。多 GPU 走 DDP（torchrun），FSDP2 作为后续项。
- Booru 抓图 / 打标 / 正则集生成等数据生产流水线（属于 Studio 上层应用，可后续作为独立服务接入）。
- InfoNoise / LeapAlign / SRA / NaViT 等研究性功能。它们对应的插件位（有状态时间步采样器、损失、数据打包）预留。

---

## 2. 参考项目的核心短板（我们要解决的问题）

| 问题 | sd-scripts | diffusion-pipe | AnimaLoraStudio |
|---|---|---|---|
| 配置 | `argparse.Namespace` 贯穿全局，~250 个 flag，Anima 下大量 flag 被静默忽略 | 无类型 dict，键散落 4+ 文件，存在陈旧键 | pydantic 但 ~170 字段平铺一张表 |
| 可复用性 | `train()` 单方法 965 行；策略类是类级单例，一进程只能跑一次 | 单文件 975 行；硬依赖 DeepSpeed，Windows 不可用；即使训练 Anima 也无条件 import ComfyUI/HunyuanVideo 子模块 | 训练核心可独立 CLI，但监督器 1814 行单类 |
| 适配器 | 网络模块靳 `hasattr` 鸭子类型；LoKr 自实现但无 Anima 合并/提取工具 | 只走 PEFT；`alpha` 强制等于 rank；按类名扫全部 Linear（AdaLN 也被打上 LoRA） | LoKr 依赖 `lycoris-lora` 外部包（自研版曾有 factor 回退成满秩的 bug） |
| Anima fp8 | 明确不支持 | 仅裸 cast（e4m3fn 效果差需 e5m2）；`fp8_scaled` 权重直接拒绝 | 支持（Krea 2 为主） |
| 暂停恢复 | `accelerate` state，靠 `skip_first_batches` 重放 | DeepSpeed checkpoint，dataloader 状态自定义 | 审计出 7 个恢复 bug 后退化为"暂停 = 取消 + epoch 末备份" |
| 验证损失 | Anima 路径下 4 个"固定"验证时间步实际是随机抽样（bug） | 9 分位数固定时间步，确定性 RNG（好设计，但代价 = 9 遍验证集） | 事后 CLIP/DINO/CCIP 指标，无训练期验证损失 |
| 缓存 | 每图每分辩率一个未压缩 fp32 `.npz`；TE 缓存固定 512×1024 fp32（2 MB/caption）；有效性只查键存在 | HF Datasets + 指纹，可续；缓存放在数据集目录内 | 文本缓存 sidecar，可用 |
| 进程通信 | 无（脚本） | 无（信号文件） | stdout 里混 `__EVENT__:` 标记行，与库日志共享通道 |
| Huber 损失（Anima） | 默认 `snr` 调度直接 `NotImplementedError`，`exponential` 因时间步已 /1000 而近似常数 | 有 `huber_delta` | 有 |
| 测试 | Anima 相关 **0** 个测试 | 极少 | 244 个测试文件（优点） |

同时要**继承**的优点：sd-scripts 的数据集配置级联与 caption 处理、`ss_*`/ModelSpec 元数据、逐块 `torch.compile`；diffusion-pipe 的分位数验证损失、Kahan bf16、按模块组分学习率、从 checkpoint 推断模型几何；AnimaLoraStudio 的插件注册表 + `ModelSpec` 能力门控、前后向双钩子 Block Swap 与 pinned 打包、Schema 驱动表单、SSE 事件总线、trace id 与错误信封。

---

## 3. 总体分层

```
┌───────────────────────────────────────────────────────────────────┐
│ frontend/  (React + TS，独立会话负责)                               │
└──────────────▲────────────────────────────▲───────────────────────┘
               │ REST (OpenAPI)              │ SSE /api/events
┌──────────────┴────────────────────────────┴───────────────────────┐
│ ypuddin.server   FastAPI · SQLite · JobSupervisor · EventBus       │
│   projects / datasets / presets / jobs / artifacts / models / sys  │
└──────────────▲────────────────────────────────────────────────────┘
               │ 子进程: `ypuddin train --config … --events <pipe>`   (JSON Lines)
┌──────────────┴────────────────────────────────────────────────────┐
│ ypuddin (核心库)                                                    │
│  config      pydantic v2 模型 · JSON Schema + UI 提示 · 预设 · 规则  │
│  models      ModelFamily 协议 · anima/ · toy/                       │
│  adapters    LoRA / LoKr / LoHa / Full · DoRA · 规则匹配 · 存取 · 转换│
│  data        索引 · 分桶 · 缓存 · caption 变换 · 可恢复采样器         │
│  objectives  Rectified Flow · 时间步采样器 · 损失加权 · 噪声增强      │
│  optim       优化器工厂 · Kahan 包装 · 调度器 · 参数组               │
│  memory      fp8 量化 · Block Swap · 激活检查点 · VRAM 规划          │
│  train       Trainer · 阶段 · 检查点 · EMA · 事件 · 采样出图         │
│  sampling    Euler Flow 采样器（预览）· CFG                           │
│  tools       merge / extract / resize / convert                     │
│  cli         `ypuddin train|cache|plan|convert|serve …`             │
└───────────────────────────────────────────────────────────────────┘
```

依赖方向自上而下：`server → train → {models, adapters, data, objectives, optim, memory}` → `config`。核心库任何模块**不得** import `server`。

---

## 4. 配置系统（`ypuddin.config`）

- **单一真相**：`TrainConfig`（pydantic v2，`extra="forbid"`）。子模型按关注点拆分：`model`、`dataset`、`adapter`、`objective`、`optimizer`、`scheduler`、`memory`、`loop`、`checkpoint`、`sampling`、`validation`、`logging`。前端表单直接消费 `TrainConfig.model_json_schema()`。
- **UI 提示**通过 `Annotated[T, Field(...), UI(group=, order=, advanced=, control=, show_when=, unit=, help=)]` 注入到 JSON Schema 的 `x-ui` 扩展字段。`show_when` 是一个小型表达式（`"adapter.algo == 'lokr'"`），后端与前端各有一个解释器，后端在校验期用同一表达式做"隐藏字段不得非默认"的一致性检查。
- **能力门控**：`ModelFamily.spec.capabilities`（如 `block_swap`、`fp8_base`、`text_encoder_train`、`masked_loss`）决定哪些字段可用；校验器统一在 `config/rules.py`，**禁止**在训练代码里 `if family == "anima"`。
- **预设**：`presets/*.toml` 是部分覆盖（patch），`resolve(preset, overrides) -> TrainConfig`；每个 run 保存**完全解析后的**配置 + 配置哈希。
- **规划（plan）**：`ypuddin plan config.toml` 不加载权重即输出：分桶结果、每 epoch 步数、总步数、参数量、优化器状态大小、按分辩率的激活估算、VRAM 预算与建议（开 swap / 检查点 / 8-bit）。服务端在入队前调用它做预检。

---

## 5. 模型族协议（`ypuddin.models`）

```python
@dataclass(frozen=True)
class ModelSpec:
    name: str  # "anima"
    objective: Literal["rectified_flow"]
    latent: LatentSpec  # channels=16, stride=8, patch=2, temporal=False
    text: TextSpec  # max_len=512, pad_floor=True, encoders=("qwen3",)
    capabilities: frozenset[str]  # {"block_swap","fp8_base","llm_adapter","text_encoder_train",...}
    t_convention: Literal["unit"]  # 模型接收 t∈(0,1)（Anima/Cosmos），而非 t*1000
    sampling_defaults: SamplingDefaults  # steps=25, cfg=4.0, shift=3.0, sampler="euler"


class ModelFamily(Protocol):
    spec: ModelSpec

    def load(self, paths: ModelPaths, *, dtype, device, memory: MemoryPlan) -> LoadedModel: ...
    # 文本：tokenize → encode，返回 opaque 的 TextCond（含 fingerprint 用于缓存键）
    def text_pipeline(self, loaded) -> TextPipeline: ...
    # 图像：pixels[-1,1] → latents（已归一化）；decode 反向；fingerprint 用于缓存键
    def latent_pipeline(self, loaded) -> LatentPipeline: ...
    # 主干前向：x_t (B,C,H,W)、t (B,) ∈ (0,1)、cond → 速度预测
    def forward(self, loaded, x_t, t, cond: TextCond, *, extra) -> Tensor: ...
    # 适配器可打的模块分组（用于规则与预设）："attn_qkv","attn_out","mlp","adaln","llm_adapter","te",...
    def adapter_groups(self, loaded) -> dict[str, list[str]]: ...
    # 我们的规范模块名 ↔ 各生态键名（kohya lora_unet_*, ComfyUI diffusion_model.*, PEFT）
    def key_mapping(self) -> KeyMapping: ...
    # 显存计划：可交换的 block 列表、必须保持高精度的模块名、每 block 参数量
    def memory_layout(self, loaded) -> MemoryLayout: ...

    # 预览采样所需的最小推理：调用 forward + CFG，由 sampling 模块驱动
```

**循环不变量**（借鉴 AnimaLoraStudio 03-interface-evolution 并加严）：latents 为 4D `(B,C,H,W)`（temporal 族以后再扩）；`t ∈ (0,1)` fp32；`x_t = (1-t)·x0 + t·ε`；`target = ε − x0`；autocast 由训练循环持有；模型族不得触碰全局 RNG；文本条件对循环是不透明对象。

**Anima 实现要点**（见 `01-anima-family.md`，摘要）：

- DiT 几何**从 checkpoint 推断**（`x_embedder.proj.1.weight` 给宽度与输入通道、按 `blocks.N` 计数层数、宽度→头数表），兼容 `net.` 与 `model.diffusion_model.` 两种前缀；LLM Adapter 权重内嵌于 DiT checkpoint。
- 文本：Qwen3-0.6B `last_hidden_state`，padding 位置置零；T5（老版 spiece）token ids 送 LLM Adapter；长度以 512 为下限、允许更长；LLM Adapter 属于主干，可选训练组。
- VAE：Qwen-Image（Wan2.1 结构）16 通道、8× 下采样、逐通道 mean/std 归一化、`mode()` 确定性编码；图像训练用等价 2D 卷积版本（约 3× 省显存 2× 快）。
- 默认适配器目标：DiT 各 block 的 `q_proj/k_proj/v_proj/output_proj` + `mlp.layer1/layer2`；AdaLN 调制、`llm_adapter`、Qwen3 为可选组。
- 时间步：模型直接接收 `t`；预览采样 Euler + 常数 shift（默认 3.0）。
- 交叉注意力无 key mask（上游结构如此），因此缓存必须精确复现"padding 置零"。

`toy` 族：2 层 128 宽的迷你 DiT、`nn.Embedding` 充当文本编码器、单层卷积充当 VAE，具备与 Anima 相同的接口与能力标志，用于 CPU 端到端测试。

---

## 6. 适配器引擎（`ypuddin.adapters`）

详细数学与实现见 `02-adapters-lokr.md`。此处只列架构决策：

- **真实子模块，不做 forward 猴子补丁**：目标 `nn.Linear` 被替换为 `AdaptedLinear(base, adapter)`，`base` 冻结（可为 fp8 存储 + 缩放）。这样 DDP 的归约、`torch.compile`、`state_dict` 都按常规工作，也不需要 sd-scripts 那种手工 `all_reduce`。
- **三种执行路径**：`merged`（材料化 ΔW 后一次 matmul，适合 bf16 底模 + LoRA/LoHa）、`bypass`（`y = W x + Δ(x)`，底模为 fp8/量化时必需）、`kron_bypass`（LoKr 专用：用 `(A⊗B) vec(X) = vec(B X Aᵀ)` 避免材料化 ΔW）。运行时按底模精度与形状自动选择，可强制。
- **规则式目标选择**：`rules: [{match: "blocks.*.self_attn.{q,k,v}_proj", algo: lokr, factor: 8, dim: 16, alpha: 16, lr: 1e-4}, {match: "*mlp*", dim: 8}, ...]`，按序首个匹配生效；预设如 `anima/attn-mlp`、`anima/full-linear`、`anima/with-adapter`。
- **存取格式**：规范内部名 → kohya 风格 `lora_unet_<path_with_underscores>.{lora_down,lora_up,lokr_w1,lokr_w2_a,lokr_w2_b,hada_w1_a,…}` + `.alpha`（+ `dora_scale`），附 `ss_*`、`modelspec.*` 与 `ypuddin.*`（规则、指纹、配置哈希）元数据；可导出 ComfyUI/PEFT 键名；可导入 kohya / LyCORIS / PEFT / ComfyUI 任一格式。
- **工具**：merge（合入底模，含 fp8 底模的重量化）、extract（底模差分 → LoRA 用 SVD；→ LoKr 用最近 Kronecker 积分解 + 低秩）、resize（SVD 降秩，含 LoKr）、convert（键格式互转）。

---

## 7. 数据流水线（`ypuddin.data`）

- **来源**：`dataset.sources: [{path, repeats, caption_ext, class_prompt, is_reg, prior_weight, transforms...}]`，级联覆盖（source → dataset 默认）。
- **索引**：扫描 → 每图内容哈希（blake2b 64-bit 于文件字节）→ 尺寸/AR 元数据写入 `index.sqlite`；重命名文件不失效；caption 在**训练时**读取，改 caption 不重建 latent 缓存。
- **分桶**：多分辩率列表（如 `[1024, 768]`）、`aspect_ratio_limit`、步长由族决定（Anima 16）、面积容差 ±10%、`bucket_no_upscale` 选项。`BucketBatchSampler` 是确定性的（seed + epoch），**可 `state_dict`/`load_state_dict`**，`drop_last=False`，绝不双训。
- **缓存**（集中目录 `cache_dir/`，不在数据集内）：latents 键 `(content_hash, w, h, latent_fingerprint, flip)`，bf16 `safetensors`，一图一文件；文本键 `(caption_hash, text_fingerprint, max_len)`，**按真实长度存储**（去 padding），加载时再 pad 到批内最大长度并重建 mask —— Anima 文本 512×1024 fp32 从 2 MB 降到平均几十 KB。缓存任务多进程解码 + GPU 编码，可中断续跑，`ypuddin cache` 单独可用。
- **在线文本编码**：Anima 的 Qwen3-0.6B 仅 1.2 GB（bf16），允许常驻并在线编码，从而 caption 增强（tag 洗牌、tag dropout、通配符）**不受缓存限制**；默认策略由 plan 根据显存决定。
- **caption 变换**：prefix/suffix、trigger word 注入、`keep_tokens`、shuffle、tag dropout、caption dropout（→ 无条件嵌入，用于 CFG）、通配符 `{a|b}`、二级分隔符。
- **正则集与损失权重**：`is_reg` 来源以 `prior_weight` 加权；掩码损失通过同名 `.mask.png` 或 alpha 通道。
- **验证集**：按内容哈希确定性切分 `validation.split_ratio`，与训练集互斥；也可显式指定目录。

---

## 8. 训练目标与调度（`ypuddin.objectives`）

- Rectified Flow：`x_t = (1-t) x0 + t ε`，预测速度 `v = ε − x0`。
- 时间步采样器（有状态者实现 `state_dict`）：`uniform`、`logit_normal(mean, std)`、`shift(s)`（Möbius 变换 `t·s/(1+(s−1)t)`）、`resolution_shift`（Flux 式 μ 随 token 数线性）、`mode(scale)`、`cosmap`、以及 **`stratified`**（每批在 (0,1) 分层抽样后再变换，降低梯度方差）。
- 损失：`mse`、`huber(delta)`、`pseudo_huber(c)`（对 RF 正确实现，`c` 常数或按 `t` 调度），加权：`none`、`sigma_sqrt`、`cosmap`、`snr_like`（`1/(t²+(1−t)²)`）、`cosmos`（NVIDIA 原始 `t²+(1−t)²`，diffusion-pipe 有意去掉、我们作为选项保留）。
- 噪声增强：`ip_noise_gamma`、pyramid noise；uncond 概率（caption dropout）。
- **确定性验证**：`validation.timesteps = [0.1,0.3,0.5,0.7,0.9]`（分位数经采样器 icdf 映射）× 固定噪声种子 × 固定验证集，报告每时间步与总均值；成本可配（子集大小）。

---

## 9. 训练循环与状态（`ypuddin.train`）

```
Trainer(config)
  .prepare()   解析配置 · 构建族 · 索引/缓存数据 · 注入适配器 · 构建优化器/调度器 · 显存计划
  .run()       while step < max: micro-batches × grad_accum → step → hooks(log/val/sample/save)
  .checkpoint(path, kind="full"|"weights")   .resume(path)
```

- 步为一等公民（epoch 是派生量）；梯度累积在微批层面；`sync_gradients` 边界才计步、裁剪、调度。
- 非有限损失：跳过该微批并计数，连续 N 次升级为错误（沿用 AnimaLoraStudio 教训）。
- **完整检查点**（原子写入）：适配器权重、优化器状态、调度器、`BucketBatchSampler` 状态、时间步采样器状态、EMA、RNG（python/numpy/torch/cuda/mps）、step/epoch、配置哈希、数据指纹。恢复时校验指纹并逐项还原 → **任意步边界可暂停恢复**。
- **控制通道**：`control/` 目录下的命令文件（`pause`、`save`、`stop`）+ SIGINT/SIGTERM（Windows 上以文件为主）。暂停 = 在下一个优化器步边界写完整检查点后退出（状态 `paused`）。
- EMA（可选，默认 CPU 上对适配器参数），保存 `-ema` 变体。
- 优化器：AdamW / AdamW8bit / Lion / Prodigy / Prodigy+ScheduleFree / Adafactor / CAME + 通用 `module.Class` 透传；`kahan=true` 在 bf16 参数上补偿（与 Schedule-Free 组合显式拒绝）；`fused_backward=true` 尚未实现，配置校验显式拒绝。
- 调度器：constant / linear / cosine / cosine_restarts / polynomial / warmup_stable_decay / rex；schedule-free 感知（train/eval 切换）。
- 采样出图：Euler + shift，CFG，多提示词，按步/epoch 触发，在训练进程内以 `no_grad` 运行并可临时切换 swap 为推理模式；产物写 `samples/` 并发事件。
- 事件（JSON Lines → `events.jsonl` + 管道）：`run.started`、`phase.changed`、`cache.progress`、`step`（loss/lr/grad_norm/it_s/vram/eta）、`validation`、`sample.saved`、`checkpoint.saved`、`warning`、`run.finished|failed|paused`。TensorBoard / W&B 作为可选 sink。
- 多 GPU 训练是后续目标，当前没有 DDP 包装；服务可在不同 GPU 上各运行一个独立单设备任务。

---

## 10. 显存子系统（`ypuddin.memory`）

- **fp8 冻结底模**：加载时把 `memory_layout.quantizable` 的 2D 权重量化为 `float8_e4m3fn` + 逐张量（或逐 128 行块）fp32 缩放；前向在 bypass 路径反量化到 bf16 做 matmul；兼容读取 ComfyUI `fp8_scaled`（`scale_weight`）文件与裸 fp8 文件；LoRA 参数始终全精度。merge 工具支持"反量化 → 加 ΔW → 重量化（`amax/448`）"。
- **Block Swap**：最后 N 个 block 的权重驻留 **pinned 主机内存**（打包进少量大块，规避 2^n 取整），前向 pre/post hook 与反向 pre/post hook 在专用 CUDA 流上预取/换出，`param.data` 原地交换（不替换模块对象）；LoRA 参数永不交换；推理模式可整体关闭。
- **激活检查点**：`none | block | unsloth`（block 输入异步卸载到 CPU，反向时重算）。
- **VRAM 规划器**：`plan` 命令与服务预检共用且采用相同数据布局。MPS 按 FP32 估算，不将 CPU swap 视为统一内存释放。训练事件 `vram_metric=peak_allocated` 表示 CUDA PyTorch 分配峰值，`current_allocated` 表示 MPS 当前 PyTorch 分配量，不代表整机或驱动的硬件峰值。

---

## 11. 服务与任务（`ypuddin.server`）

- FastAPI；标准库 `sqlite3` + RLock + WAL；表为 `projects`、`datasets`、`jobs`、`artifacts`、`models`、`kv`。配置草稿、预设和事件文件单独落盘。
- **JobSupervisor**：每任务启动 `ypuddin train/cache … --device` 子进程，独立 `events.jsonl` 被监督器尾读进入 EventBus → SSE `/api/events`（支持有限内存环中的 Last-Event-ID 重放）。stdout/stderr 单独落 `run.log`；重连时前端重新拉取快照。
- 队列：优先级 FIFO、定时任务、每加速器独占。CUDA 空闲显存来自 PyTorch，NVML 仅补充可用遥测；MPS 来自系统可用统一内存。空闲容量与 Plan 估算用于可关闭的 `memory_admission` 准入，不支持 DDP。
- 控制：`POST /jobs/{id}/pause|resume|cancel|save` 写命令文件。缓存准备中可暂停并复用缓存，恢复准备中再次暂停保留原恢复目标；训练暂停在优化器步边界保存完整状态。取消超时仅杀对应原子进程，服务停止先请求暂停再限时等待。CLI 给 SSE 连接 5 秒退出宽限，随后进入服务清理。
- 错误信封 `{"error": {"code","message","trace_id","details"}}`，`X-Trace-Id` 贯穿。
- OpenAPI JSON 导出到 `docs/api/openapi.json` 供前端生成类型。训练事件及服务转换以 `train/trainer.py`、`server/supervisor.py` 为准；SSE payload 未统一为完整 Pydantic 模型。
- 前端需求见 `../frontend-spec.md`。

---

## 12. 测试策略

- `tests/unit/`：配置校验与 schema、规则匹配、LoKr/LoRA/LoHa 数学（ΔW 一致、bypass = merged、DoRA 范数、保存/加载往返、键名转换往返、factorization 表）、分桶与采样器（可恢复、无双训、drop_last=False）、缓存键与去 padding、时间步采样器分布、损失加权。
- `tests/e2e/`：toy 族 + 合成数据：cache → train 30 步 → save → 从第 15 步检查点 resume → 与不间断训练**逐位一致**；采样出图产生文件；服务 API 走一遍 create → enqueue → 事件 → 产物。
- CUDA-only 路径（fp8、bitsandbytes、block swap 真实 H2D）以 `pytest.mark.cuda` 标记，在 GPU 机器上跑。

---

## 13. 里程碑

| # | 内容 | 验收 |
|---|---|---|
| M0 | 仓库骨架、配置系统、事件模型、toy 族、Trainer 最小循环 | e2e 跑通、resume 逐位一致 |
| M1 | 适配器引擎（LoRA/LoKr/LoHa/DoRA）、存取与转换、工具 | 数学单测、与 LyCORIS 数值对拍 |
| M2 | 数据流水线（索引/分桶/缓存/caption）、验证集 | 单测 + toy e2e 使用真实数据流 |
| M3 | Anima 族（加载/文本/VAE/前向/采样） | GPU 机器上小数据集出图正常、LoRA 可载入 ComfyUI |
| M4 | 显存子系统（fp8、block swap、检查点、Kahan、8-bit） | GPU 上显存曲线与速度基准 |
| M5 | 服务 API + 任务队列 + SSE；前端对接 | httpx 集成测试；前端可跑通一次训练 |
| M6 | 文档、预设、基准对比（对 sd-scripts / diffusion-pipe 同配置） | 报告 |

---

## 14. 关键决策记录（简）

- **D1 不用 DeepSpeed**：单卡与 Windows 是主要场景；DDP 足够，FSDP2 后续。
- **D2 适配器用真实子模块替换**：换取 DDP/compile/state_dict 的常规行为，代价是需要一次模块树重写（可逆）。
- **D3 kohya 键名为内部规范存储格式**：ComfyUI、A1111 生态直接可用；同时提供 ComfyUI/PEFT 键导出。
- **D4 缓存集中、内容哈希键**：数据集目录保持干净；重命名/移动不失效；多项目共享。
- **D5 事件走独立 JSONL 文件**：杜绝 stdout 解析的脆弱性；日志与事件分离。
- **D6 精确暂停恢复而非 epoch 末备份**：一切有状态组件必须实现 `state_dict`，这是设计约束而非事后补丁。
- **D7 Apache-2.0，不复制 GPL 代码**：Anima 模型结构以 NVIDIA Cosmos-Predict2（Apache-2.0）与 sd-scripts（Apache-2.0）为参考独立实现；ComfyUI 派生的采样器/文本编码细节只做行为对齐，不搬代码。


## 15. 2026-09-11 修复后的状态边界

模型先在 CPU 加载主干并注入适配器，缓存所需 VAE/文本编码器错峰加载；之后按换块布局搬运训练主干。实际 VAE/文本权重、配置和 tokenizer 内容形成缓存指纹，DiT 与两编码器身份写入 format 2 完整状态。缓存键不含 mask，mask 每次读取；数据指纹含 caption/mask/验证源且采样顺序按内容稳定排序。新版恢复校验不能兼容旧文件名排序时明确拒绝。

服务对新任务冻结绝对路径并强制独立事件/输出路径。修改设置中的 cache/output/models 只影响后续任务；已有运行路径保留。host/port 重启生效且显式 CLI 参数优先；data_root 只由启动参数决定。详情见 `../FIX_REPORT_2026-09-11.md`。
