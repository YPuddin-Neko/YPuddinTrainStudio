# 内置适配器

训练器的适配器实现在 [`ypuddin/adapters/`](../ypuddin/adapters/)。训练时调用本项目的 PyTorch 模块，不调用 `lycoris-lora` 包。运行环境页的 LyCORIS 版本是导出格式的参考信息；更新或安装 LyCORIS 不会替换这些训练模块。

## 计算与参数

`AdaptedLinear` 包装所选的线性层，底模权重保存在 `FrozenLinear`；`AdaptedConv` 包装所选的卷积层（Conv1d / Conv2d / Conv3d），底模仍是原卷积模块。两者共用 `AdaptedLayer`：分开计算执行 `base(x) + adapter(x)`；合并计算先得到 `W0 + ΔW`（Full 另加 `b0 + Δb`）再做该层自己的线性或卷积运算。DoRA、LoHa、LyCORIS Full 和分组卷积只能合并计算，T-LoRA 只能分开计算；“权重计算方式”手动选择另一种方式会报错。

T-LoRA 和 LyCORIS Full 不使用 DoRA：选择这两种算法时，“启用 DoRA”自动关闭并置灰；“逐层覆盖规则”中使用这两种算法的层不启用 DoRA。海光 DTK 的可复现计算配方不支持 DoRA，开启 DoRA 时“可复现训练”关闭并置灰。

| 算法 | 权重增量 | 实现 |
| --- | --- | --- |
| LoRA | `scale × B @ A` | [lora.py](../ypuddin/adapters/lora.py) |
| LoKr | `scale × (W1 ⊗ W2)`；因子可再次低秩拆分 | [lokr.py](../ypuddin/adapters/lokr.py) |
| LoHa | `scale × (B1 @ A1) ⊙ (B2 @ A2)` | [loha.py](../ypuddin/adapters/loha.py) |
| OrthoLoRA | 在冻结的主要奇异子空间内训练旋转和缩放 | [ortho.py](../ypuddin/adapters/ortho.py) |
| T-LoRA | 每个样本按噪声强度使用不同数量的秩 | [tlora.py](../ypuddin/adapters/tlora.py) |
| LyCORIS Full | 直接训练所选层权重和原有偏置，`ΔW = W − W0`、`Δb = b − b0` | [full.py](../ypuddin/adapters/full.py) |

普通低秩缩放为 `alpha / rank`，rsLoRA 为 `alpha / sqrt(rank)`。LoKr 的两个因子均为完整矩阵时缩放固定为 1，Rank 和 Alpha 不参与计算。LoKr Full 仍是 Kronecker 结构；LyCORIS Full 保存整层差值，两者不同。

LoKr 分开计算用因子乘法避免生成完整 `ΔW`。LoHa 的自定义反向会重新计算两组低秩乘积，以减少保存的中间张量。这些是普通 PyTorch 实现，不使用 LyCORIS 的融合 GPU 内核。

OrthoLoRA 冻结底模权重的主要奇异向量与奇异值，训练 Cayley 旋转及两侧缩放，初始增量为零。T-LoRA 的掩码逐样本广播到序列维度；正交初始化还保存一份冻结起点，并从可训练增量中减去起点。预览与导出使用全部秩。

输出丢弃率、秩丢弃率和模块丢弃率的取值不小于 0 且小于 1：取 1 时全部被丢弃，适配器学不到任何东西。输出丢弃率只在分开计算时生效，DoRA、LoHa 等合并计算不使用；LoKr 的秩丢弃率仅在 W2 低秩拆分时生效；LyCORIS Full 只使用模块丢弃率。

## 卷积层

“训练层类型”选择只训练线性层，或同时训练卷积层。只有 SDXL 的 UNet 含卷积层，其他模型不显示此项。卷积层只在训练层范围给出卷积范围时加入：

| SDXL 训练层范围 | 只训练线性层 | 线性层和卷积层 |
| --- | --- | --- |
| 精简范围（attn-only） | 注意力投影，560 层 | 不含卷积层，固定为只训练线性层 |
| 常规范围（attn-mlp，默认） | 注意力和前馈，700 层 | 另加 ResNet 模块的 `conv1`、`conv2`、`conv_shortcut`、`time_emb_proj` 与上下采样卷积，共 766 层，其中卷积 49 个 |
| 全部层（all-layers） | UNet 全部 743 个线性层 | 另加全部 51 个卷积层，包括输入、输出卷积 |

卷积部分与 kohya LoCon 的范围相同（ResNet 与上下采样模块）；全部层对应 LyCORIS `full` 预设覆盖的线性层和卷积层，另含尺寸嵌入。归一化层不训练。“逐层覆盖规则”只在所选范围含卷积范围时匹配卷积层。

大于 1×1 的卷积使用“卷积层 Rank / Alpha”，与 kohya、LyCORIS 的 `conv_dim` / `conv_alpha` 相同，留空时沿用 Rank / Alpha；1×1 卷积使用线性层的 Rank 和 Alpha。导出文件的 `ss_network_args` 记录实际使用的 `conv_dim` 和 `conv_alpha`。

各算法在卷积层上的结构与导出布局与 LyCORIS 相同，不使用 Tucker 分解：

| 算法 | 卷积层的结构 | 导出张量 |
| --- | --- | --- |
| LoRA、OrthoLoRA、T-LoRA | 把权重视为 `(out, in·k…)` 矩阵；`down` 按该层的步长、填充、膨胀卷积到 `r` 个通道，`up` 以 1×1 卷积混合 | `lora_down.weight (r, in, *k)`、`lora_up.weight (out, r, 1…)` |
| LoKr | 输入、输出通道按 Factor 分解，卷积核留在 W2：`ΔW = W1[…, None…] ⊗ W2`；分开计算把输入分成 `c` 组分别与 W2 卷积，再用 W1 混合各组 | `lokr_w1 (a, c)`；`lokr_w2 (b, d, *k)`，或 `lokr_w2_a (b, r)` + `lokr_w2_b (r, d·k…)` |
| LoHa | 两组低秩矩阵的逐元素积，`w*_b` 展开卷积核 | `hada_w1_b (r, in·k…)` 等 |
| LyCORIS Full | 训练整层卷积核与偏置 | `diff (out, in, *k)`、`diff_b (out)` |

卷积层的 DoRA 幅度覆盖卷积核：按输出通道为 `(out, 1, 1…)`，按输入通道为 `(1, in, 1…)`，与 LyCORIS、ComfyUI 的形状一致。

卷积层的冻结权重保持加载精度，“底模存储精度”（含 FP8）只转换线性层。海光 DTK 的可复现计算配方只覆盖线性适配器；训练卷积层时这些配方不生效，按常规计算执行。

## 导出与恢复

LoRA、LoKr、LoHa 和 Full 使用对应的 Kohya / LyCORIS 键布局，Full 在层有偏置时另存 `diff_b`。OrthoLoRA 导出为同秩 LoRA；正交初始化的 T-LoRA 导出为两倍秩的 LoRA，用两个项精确表示训练后的增量。导出文件只包含推理权重；完整恢复点另存优化器、调度器、随机状态和原始训练参数。

OrthoLoRA 和正交 T-LoRA 的通用 LoRA 文件不能还原原始训练参数，继续原任务应使用完整恢复点。以普通 LoRA 加载这些文件开展新训练是另一种参数化。

Anima、Krea 2、FLUX.2 的文本编码器使用 `lora_te_` 前缀，SDXL 使用 `lora_te1_` / `lora_te2_`。读取端保留旧键名兼容，旧产物的修复入口只改键名。

DoRA 在合并权重上按输入或输出通道进行幅度归一化。方向会改变训练参数与导出形状；推理端支持范围见[训练配置](TRAINING.md)。不能仅改文件形状来切换方向。

## 与 LyCORIS 的边界

本项目训练线性层和卷积层（见上文“卷积层”），不实现 LyCORIS 的 Tucker 分解（`use_tucker`）和 DyLoRA，也不读取带 `lora_mid`、`lokr_t2`、`hada_t1` 的 Tucker 文件。卷积层的计算由本项目实现，LyCORIS 上游对卷积 bypass、融合内核因子分解、kernel dispatch 和 DyLoRA 梯度路由的修复不对应这里相同的执行路径。

卷积层文件可由 LyCORIS 的 kohya 加载器、合并工具和 ComfyUI 直接读取，包括 LoKr 的各种拆分形式、带偏置的 LyCORIS Full 和按输出通道的 DoRA。

Full 保留底层权重，前向通过合并权重计算，不依赖删除原层权重后再调用其前向。缩放由适配器统一计算，导出编码到 alpha 或因子中；本项目合并工具重建后应用一次增量，不再额外乘一次 alpha/rank。它们与上游历史问题的实现路径不同。

选择范围由[规则解析](../ypuddin/adapters/rules.py)决定：第一条匹配的用户规则优先于模型预设，`algo="none"` 排除该层。这里不使用 LyCORIS 的 kohya 预设解析器。

## 与官方配置对照

同名算法不意味着所有参数取值的行为都相同。比较前应对齐训练层、初始化、有效缩放、DoRA 方向、数值精度和随机策略：

- 默认线性 LoRA、LoKr、LoHa 的增量公式相同。若参数与数值运算完全一致，其梯度更新没有质量上的优劣。
- 本项目的 rank dropout 在低秩轴采样并按 `1/(1-p)` 补偿。LyCORIS 的部分合并路径在增量的输出行采样，补偿也由额外选项控制。非零 dropout 不能仅按参数同名作等价配置。
- 本项目 Full 与官方 Full 一样训练所选层的权重和原有偏置，偏置增量导出为 `diff_b`。
- 卷积层的 Rank 规则与 kohya、LyCORIS 相同：大于 1×1 的卷积用 `conv_dim` / `conv_alpha`，1×1 卷积用线性层的 Rank。官方默认 `full` 预设会训练卷积层；本项目默认只训练线性层，需要时在“训练层类型”中选择。训练范围不同时参数量和归纳偏置也不同，不能只按步数对比。
- 本项目不实现 Tucker 分解和 DyLoRA。
- 本项目的 OrthoLoRA、T-LoRA 参数化、掩码与通用 LoRA 导出约定独立实现；不能把切换为官方模块视为仅更换加速后端。

Loss 曲线还受数据顺序、噪声与时间步采样、损失加权、有效批次、优化器及预览设置影响。较低或较平滑的训练 Loss 本身不能证明生成质量更高。

## GPU 内核加速

LyCORIS 4.x 的实验性加速使用 Triton / TileLang 融合内核，并提供 `torch.compile` 和普通 PyTorch 回退。覆盖 LoRA、LoKr、LoHa、DoRA、Full 等算法的部分路径；具体约束随形状、精度和计算模式变化。[官方后端说明](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/kernels/backends.md)

本项目不使用这些内核，`LYCORIS_KERNEL_BACKEND` 不会改变内置适配器的执行方式。

收益应按完整训练步测量，而不是只看某个内核。官方 RTX 4090 / FP16 表中，LoRA 合并路径前向加反向墙钟比值为 1.46，LoKr 分开计算为 1.47，但 LoKr 合并路径为 0.73（小于 1 代表更慢）。这不能直接换算为本训练器的速度。[官方基准](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/kernels/benchmarks.md)

融合会改变浮点运算顺序、临时显存与首次编译时间，即使数学公式相同也不保证逐位一致。训练效果、完整步耗时及不同平台的支持需要分别比较；不能把 NVIDIA 的结果直接套到 Apple MPS 或海光 DTK。
