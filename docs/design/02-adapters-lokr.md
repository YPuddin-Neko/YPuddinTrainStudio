# 适配器引擎与 LoKr 实现设计

状态：Accepted · 日期：2026-09-10 · 参考：[`../reference/lycoris-lokr.md`](../reference/lycoris-lokr.md)

LyCORIS 是 LoKr 的数学参考实现；本文定义我们**自研**实现的数学、参数布局、执行路径、磁盘格式与测试，并明确在哪些点上比 LyCORIS 做得更好。

---

## 1. 从 LyCORIS 学到的（要复用的约定）

- LoKr：`ΔW = s · (W1 ⊗ W2)`，`W1 ∈ ℝ^{a×c}`（小，"权重缩放"），`W2 ∈ ℝ^{b×d}`（大），`out = a·b`，`in = c·d`；`W2` 可再低秩为 `W2a (b×r) @ W2b (r×d)`；`decompose_both` 时 `W1` 亦可低秩。
- `factorization(dim, factor)`：`factor=-1` 取最接近 √dim 的整除对；`factor=f>0` 且 `f | dim` 取 `(min(f, dim/f), max(...))`，**小的一侧永远给 W1**；`f ∤ dim` 时退到 ≤ f 的最大因数。参考表（我们的实现必须逐项一致）：

  | dim | −1 | 4 | 8 | 16 |
  |---|---|---|---|---|
  | 1024 | (32,32) | (4,256) | (8,128) | (16,64) |
  | 2048 | (32,64) | (4,512) | (8,256) | (16,128) |
  | 3072 | (48,64) | (4,768) | (8,384) | (16,192) |
  | 8192 | (64,128) | (4,2048) | (8,1024) | (16,512) |

- 缩放：`s = alpha / r`（`rs_lora` 时 `alpha / √r`）；当 W1、W2 都是满矩阵时 `s := 1`、alpha 被忽略。
- **磁盘上的 alpha 约定**（与 ComfyUI / A1111 / kohya 加载器兼容的关键）：加载器以 `rank = lokr_w2_b.shape[0]`（或 `lokr_w1_b`）计算 `scale = alpha_file / rank`；没有低秩因子时视 `scale = 1` 并忽略 alpha。因此 `alpha_file = s · rank`。
- 初始化：`W1` kaiming-uniform，`W2`（或 `W2b`）置零 ⇒ `ΔW(0) = 0`；`use_scalar` 模式全部随机初始化并训练一个初值为 0 的标量，保存时把标量折进 `W1`。
- bypass 恒等式：`(W1 ⊗ W2) vec(X) = vec(W1 · X · W2ᵀ)`，`X = reshape(x, (c, d))`。实现：`x (…, in) → (…, c, d)` → 右乘 `W2ᵀ` 得 `(…, c, b)` → 转置 `(…, b, c)` → 右乘 `W1ᵀ` 得 `(…, b, a)` → 转置 `(…, a, b)` → 展平 `(…, a·b)`。每 token 代价 `b·c·(d + a)`，对比材料化 ΔW 的 `out·in`（3072²、factor 8：1.2M vs 9.4M）。

## 2. LyCORIS 的问题（我们必须避免）

1. `get_diff_weight` 对 `scale` 乘了两次且丢掉 `scalar` ⇒ `alpha ≠ dim` 时 `merge_to`/`onfly_merge`/`tools/merge.py` 结果强度错误（LoKr、LoHa 都有）。**我们：scale / scalar / multiplier 只在一个函数里应用，并用测试断言 `forward_bypass ≡ forward_merged ≡ base + F.linear(x, ΔW)`**，覆盖 `alpha≠dim`、`use_scalar`、`rs_lora`、非方阵。
2. 默认 rebuild 路径每步材料化 ΔW 且再做一次完整 GEMM；bypass 反而是"量化专用"。**我们：Linear 默认 `kron_bypass`。**
3. dropout 语义按模式不同（rebuild 有 rank_dropout 无 dropout，bypass 反之），且 rank_dropout 掩码建在 CPU。**我们：`dropout`（对 Δ(x)）与 `rank_dropout`（对秩轴）在所有路径语义一致，掩码建在参数所在设备。**
4. 加载时靠启发式反推 `factor`（Tucker/unbalanced 会推错）。**我们：在元数据里显式写出每个模块的 `[[a,b],[c,d]]` 与 rank，加载不推断。**
5. 类级全局预设状态（`apply_preset` 改类属性、缺省键不重置）。**我们：所有配置都是实例级数据对象。**
6. `wd` 时 `multiplier` 只插值范数项，`multiplier=0 ≠` 底模。**我们：`multiplier` 缩放 ΔW，再对 `W + m·ΔW` 做 DoRA 重标定；合并工具对外的 strength 语义与 ComfyUI 一致。**
7. `parametrize` 对非方阵把 in/out 写反；functional 与 module 的 `gamma`/`full_matrix` 语义漂移。**我们：不提供两套 API，只有一套模块 + 纯函数被模块调用。**

## 3. 模块设计

```
ypuddin/adapters/
  base.py        AdapterModule(ABC): delta_weight() / delta_apply(x) / state for save / rank-axis dropout
  lora.py        LoRA          (down r×in, up out×r)                   路径: bypass | merged
  lokr.py        LoKr          (W1 a×c, W2 b×d | W2a b×r, W2b r×d)     路径: kron_bypass | merged
  loha.py        LoHa          (w1a,w1b,w2a,w2b Hadamard)               路径: merged
  full.py        Full          (直接训练 weight，保存 diff)               路径: merged
  dora.py        DoRA 权重分解（对 lora/lokr/loha 的可选包装）
  linear.py      AdaptedLinear(base, adapter, mode, multiplier)        真实子模块替换
  frozen.py      FrozenLinear：bf16/fp16/fp32 或 fp8(e4m3fn)+scale 的冻结底层，提供 forward 与 dequant()
  inject.py      inject(model, AdapterConfig, groups) -> AdapterSet ；eject()
  rules.py       目标选择：preset + 有序 rules（glob/regex），首个匹配生效，algo="none" 排除
  factorize.py   factorization(dim, factor) 与校验/日志
  io.py          保存/加载：kohya 键 + alpha + dora_scale + 元数据；导入 kohya/LyCORIS/PEFT/ComfyUI
  convert.py     键格式互转（kohya ⇄ ComfyUI/PEFT）
```

### 3.1 `AdapterModule` 契约

```python
class AdapterModule(nn.Module):
    kind: str                       # "lora" | "lokr" | "loha" | "full"
    out_features: int; in_features: int
    def delta_weight(self) -> Tensor          # 形状 (out, in)，已含 scale 与 scalar；multiplier 不在此处
    def delta_apply(self, x: Tensor) -> Tensor # 等价于 F.linear(x, delta_weight())，允许结构化快速路径
    supports_bypass: bool
    def export_tensors(self) -> dict[str, Tensor]   # kohya 键后缀 -> 张量（scalar 已折入，alpha 已按约定换算）
    @classmethod
    def from_tensors(cls, tensors, meta) -> "AdapterModule"
    def extra_metadata(self) -> dict          # 显式 factorization / rank / init 等
```

`AdaptedLinear.forward`：

```python
def forward(self, x):
    if self.module_dropout and self.training and rand() < p: return self.base(x)
    if self.mode == "bypass":
        return self.base(x) + self.multiplier * self.adapter.delta_apply(x)   # dropout 施加在 delta_apply 内
    W = self.base.dequant() + self.multiplier * self.adapter.delta_weight()
    if self.dora is not None: W = self.dora.rescale(W)                          # m · W / ||W||_row
    return F.linear(x, W.to(x.dtype), self.base.bias)
```

模式选择：`auto` ⇒ `lokr` 且 Linear ⇒ `kron_bypass`；`lora` ⇒ `bypass`；`loha`/`full` ⇒ `merged`；底层为 fp8 且算法支持 bypass ⇒ 强制 bypass；`dora=true` ⇒ `merged`（v1；LoKr 的 O(out·in) 范数快速路径留作后续）。

### 3.2 LoKr 参数与形状

```python
(a, b) = factorization(out, factor); (c, d) = factorization(in, factor)
w1: (a, c)                      # 或 w1_a (a, r), w1_b (r, c)   当 decompose_both 且 r < max(a,c)/2
w2: (b, d)                      # 或 w2_a (b, r), w2_b (r, d)   当 r < max(b,d)/2 且未指定 full_matrix
scale = 1 if (w1 与 w2 均满)  else alpha / (√r if rs_lora else r)
```

- `rank="full"` 等价于 LyCORIS 的"dim=10000"惯用法，不再依赖魔数。
- 初始化：`w1 ~ kaiming_uniform(a=√5)`；`w2`（满）或 `w2_b` 置零；`w2_a ~ kaiming_uniform`。`init="scalar"`：全部随机 + `scalar=0` 可训练。
- `delta_apply` 使用 §1 的 Kronecker 结构化路径；低秩 W2 时对 `X W2ᵀ` 分两步（先 `W2bᵀ` 再 `W2aᵀ`）。
- 保存：满 W2 时把 `scale·scalar` **折入 w1**（这样任何加载器都按 `scale=1` 正确还原）；低秩时 `alpha_file = scale·r`，`scalar` 折入 `w1`。
- 可选每类参数 LR 倍率（`lr_scale = {w1: 1.0, w2: 1.0}`）与 `w1` 免权重衰减（AnimaLoraStudio 经验）。

### 3.3 LoRA / LoHa / Full

- LoRA：`down (r×in)` kaiming-uniform，`up (out×r)` 零；`scale = alpha/r`（`rs_lora` 可选）；`rank_dropout` 在秩轴（`1/(1-p)` 补偿）；LoRA+（`lr_scale.up`）。
- LoHa：`ΔW = s·(w1a@w1b) ⊙ (w2a@w2b)`，初始化 `w1b~N(0,1), w1a~N(0,0.1), w2b~N(0,1), w2a=0`；merged 路径；反向用自定义 autograd 函数避免同时缓存两个乘积（LyCORIS 技巧）。
- Full：`weight` 可训练（fp32 主权重，bf16 计算）；保存 `diff = W − W₀`（LyCORIS `diff` 键）；主要用于 `llm_adapter` 或 `final_layer` 这类小模块的全量训练。

### 3.4 DoRA

`dora_scale (out,1)` 初值 `||W₀||_row`（fp32），`W' = (W₀ + m·ΔW) · dora_scale / ||W₀ + m·ΔW||_row`。存 `dora_scale` 键；`fp8` 底模下允许（在 dequant 后计算，AnimaLoraStudio 禁止是因其 fp8 补丁路径拿不到范数）。

### 3.5 目标选择规则

```toml
[adapter]
algo = "lokr"; rank = "full"; alpha = 1; factor = 8
preset = "anima/attn-mlp"            # 族提供的分组: attn_qkv, attn_out, mlp, adaln, llm_adapter, embedders, final
[[adapter.rules]]
match = "blocks.*.mlp.*"             # glob（默认）或 "re:^blocks\\.(0|1|2)\\..*"
rank = 32; alpha = 32
[[adapter.rules]]
match = "llm_adapter.*"
algo = "none"                        # 排除
```

解析结果是 `list[ResolvedTarget(name, algo, params)]`，写入 run 目录与安全张量元数据（可审计、可复现）。

## 4. 磁盘格式

- 键：`lora_unet_<canonical path, "." → "_">.<suffix>`；文本编码器 `lora_te_<...>`。后缀：`lora_down.weight / lora_up.weight / alpha`；`lokr_w1 | lokr_w1_a lokr_w1_b`、`lokr_w2 | lokr_w2_a lokr_w2_b`、`alpha`；`hada_w1_a hada_w1_b hada_w2_a hada_w2_b alpha`；`diff`；可选 `dora_scale`。
- 元数据：`ss_network_module="ypuddin.adapters"`、`ss_network_dim`、`ss_network_alpha`、`ss_network_args`（JSON）、`ss_base_model_version`、`ss_resolution`…；`modelspec.sai_model_spec="1.0.1"`、`modelspec.architecture="anima/lora"`、`modelspec.implementation`、`modelspec.title/date/hash_sha256`；`ypuddin.version`、`ypuddin.config_hash`、`ypuddin.targets`（每模块 algo/rank/alpha/factorization）、`ypuddin.dataset_fingerprint`。
- 导出：`ypuddin convert --to comfyui`（`diffusion_model.<path>.lora_A/lora_B.weight`，LoKr 保持 kohya 键）；`--to peft`（`adapter_config.json` + `adapter_model.safetensors`）。
- 导入：识别 kohya、LyCORIS（`lycoris_` 前缀）、PEFT、ComfyUI 键；LoKr 缺失显式 factorization 时才回退到形状推断并给出警告。

## 5. 工具

- `merge`：ΔW（含 DoRA）合入底模；fp8 底模：反量化 → 加 → 以 `amax/448` 重量化（可选随机取整）。
- `extract`：`W_ft − W_base` → LoRA（SVD，rank 固定 / 阈值 / 比例）；→ **LoKr**：最近 Kronecker 积（Van Loan–Pitsianis：对重排矩阵 `R(ΔW) ∈ ℝ^{ac×bd}` 取秩 1 SVD 得 `W1, W2`），再对 `W2` 低秩截断；报告相对残差。这是 LyCORIS 没有的能力。
- `resize`：LoRA/LoKr 的 SVD 降秩。
- `inspect`：打印键、形状、元数据、每模块参数量与等效秩。

## 6. 测试矩阵（CPU）

| 测试 | 断言 |
|---|---|
| `test_factorization` | 与 §1 表逐项一致；非整除 factor 的回退；素数维度退化并告警 |
| `test_lokr_delta` | `delta_weight()` ≡ `s·kron(W1, W2)`（满/半低秩/双低秩、非方阵、conv-free） |
| `test_lokr_bypass_equivalence` | `delta_apply(x)` ≡ `F.linear(x, delta_weight())`，任意前导维度，fp32 容差 1e-5 |
| `test_modes_equivalence` | `AdaptedLinear` bypass ≡ merged ≡ `base(x)+F.linear(x, ΔW)`，覆盖 `alpha≠r`、`scalar`、`rs_lora`、`multiplier∈{0,0.5,1}` |
| `test_dora` | 初始 `dora_scale=||W₀||`，`ΔW=0` 时输出 ≡ 底模；范数重标定正确 |
| `test_io_roundtrip` | export → 文件 → import → 输出逐位一致；alpha 约定：用"第三方加载器"公式 `alpha_file/rank` 重建 ΔW 与我们一致 |
| `test_rules` | preset + rules 的匹配顺序与排除；解析结果可序列化 |
| `test_inject_eject` | 替换后 `named_parameters` 只多出适配器参数；`eject()` 后 state_dict 与原模型逐位一致 |
| `test_init_zero_delta` | 所有算法初始输出 ≡ 底模 |
| `test_extract_lokr` | 对随机 `W1⊗W2` 构造的差分能精确恢复（残差 < 1e-6） |
| `test_convert_comfyui` | kohya ⇄ ComfyUI 键往返 |
