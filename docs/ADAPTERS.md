# 内置适配器

训练器的适配器实现在 [`ypuddin/adapters/`](../ypuddin/adapters/)。训练时调用本项目的 PyTorch 模块，不调用 `lycoris-lora` 包。运行环境页的 LyCORIS 版本是导出格式的参考信息；更新或安装 LyCORIS 不会替换这些训练模块。

## 计算与参数

`AdaptedLinear` 包装所选的线性层，底模权重保存在 `FrozenLinear`。分开计算模式执行 `base(x) + adapter(x)`；合并模式先得到 `W0 + ΔW` 再做线性运算。DoRA 使用合并模式，T-LoRA 使用分开计算。

| 算法 | 权重增量 | 实现 |
| --- | --- | --- |
| LoRA | `scale × B @ A` | [lora.py](../ypuddin/adapters/lora.py) |
| LoKr | `scale × (W1 ⊗ W2)`；因子可再次低秩拆分 | [lokr.py](../ypuddin/adapters/lokr.py) |
| LoHa | `scale × (B1 @ A1) ⊙ (B2 @ A2)` | [loha.py](../ypuddin/adapters/loha.py) |
| OrthoLoRA | 在冻结的主要奇异子空间内训练旋转和缩放 | [ortho.py](../ypuddin/adapters/ortho.py) |
| T-LoRA | 每个样本按噪声强度使用不同数量的秩 | [tlora.py](../ypuddin/adapters/tlora.py) |
| LyCORIS Full | 直接训练所选层权重，`ΔW = W − W0` | [full.py](../ypuddin/adapters/full.py) |

普通低秩缩放为 `alpha / rank`，rsLoRA 为 `alpha / sqrt(rank)`。LoKr 的两个因子均为完整矩阵时缩放固定为 1，Rank 和 Alpha 不参与计算。LoKr Full 仍是 Kronecker 结构；LyCORIS Full 保存整层差值，两者不同。

LoKr 分开计算用因子乘法避免生成完整 `ΔW`。LoHa 的自定义反向会重新计算两组低秩乘积，以减少保存的中间张量。这些是现有的 PyTorch 实现，不是 LyCORIS 的融合 GPU 内核。

OrthoLoRA 冻结底模权重的主要奇异向量与奇异值，训练 Cayley 旋转及两侧缩放，初始增量为零。T-LoRA 的掩码逐样本广播到序列维度；正交初始化还保存一份冻结起点，并从可训练增量中减去起点。预览与导出使用全部秩。

## 导出与恢复

LoRA、LoKr、LoHa 和 Full 使用对应的 Kohya / LyCORIS 键布局。OrthoLoRA 导出为同秩 LoRA；正交初始化的 T-LoRA 导出为两倍秩的 LoRA，用两个项精确表示训练后的增量。导出文件只包含推理权重；完整恢复点另存优化器、调度器、随机状态和原始训练参数。

OrthoLoRA 和正交 T-LoRA 的通用 LoRA 文件不能还原原始训练参数，继续原任务应使用完整恢复点。以普通 LoRA 加载这些文件开展新训练是另一种参数化。

Anima、Krea 2、FLUX.2 的文本编码器使用 `lora_te_` 前缀，SDXL 使用 `lora_te1_` / `lora_te2_`。读取端保留旧键名兼容，旧产物的修复入口只改键名。

DoRA 在合并权重上按输入或输出通道进行幅度归一化。方向会改变训练参数与导出形状；推理端支持范围见[训练配置](TRAINING.md)。不能仅改文件形状来切换方向。

## 与 LyCORIS 的边界

本项目的训练目标是线性层，没有 LyCORIS 的卷积 LoKr 分支，也没有 DyLoRA。LyCORIS 上游对卷积 bypass、融合内核因子分解、kernel dispatch 和 DyLoRA 梯度路由的修复，不会自动应用到本项目，也不对应这里相同的执行路径。

Full 保留底层权重，前向通过合并权重计算，不依赖删除原层权重后再调用其前向。缩放由适配器统一计算，导出编码到 alpha 或因子中；本项目合并工具重建后应用一次增量，不再额外乘一次 alpha/rank。它们与上游历史问题的实现路径不同，不能据此推断其他未检查组合都没有问题。

选择范围由[规则解析](../ypuddin/adapters/rules.py)决定：第一条匹配的用户规则优先于模型预设，`algo="none"` 排除该层。这里不使用 LyCORIS 的 kohya 预设解析器。

## 与官方配置对照

同名算法不意味着所有参数取值的行为都相同。比较前应对齐训练层、初始化、有效缩放、DoRA 方向、数值精度和随机策略：

- 默认线性 LoRA、LoKr、LoHa 的增量公式相同。若参数与数值运算完全一致，其梯度更新没有质量上的优劣。
- 本项目的 rank dropout 在低秩轴采样并按 `1/(1-p)` 补偿。LyCORIS 的部分合并路径在增量的输出行采样，补偿也由额外选项控制。非零 dropout 不能仅按参数同名作等价配置。
- 本项目 Full 训练选定线性层的权重，保留偏置；官方 Full 在原层有偏置时也训练偏置。
- 本项目不训练卷积适配器或 DyLoRA。官方的额外训练范围与算法增加了可调能力，也改变了参数量和归纳偏置，不能只按步数对比。
- 本项目的 OrthoLoRA、T-LoRA 参数化、掩码与通用 LoRA 导出约定独立实现；不能把切换为官方模块视为仅更换加速后端。

Loss 曲线还受数据顺序、噪声与时间步采样、损失加权、有效批次、优化器及预览设置影响。较低或较平滑的训练 Loss 本身不能证明生成质量更高。

## GPU 内核加速

LyCORIS 4.x 的实验性加速使用 Triton / TileLang 融合内核，并提供 `torch.compile` 和普通 PyTorch 回退。覆盖 LoRA、LoKr、LoHa、DoRA、Full 等算法的部分路径；具体约束随形状、精度和计算模式变化。[官方后端说明](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/kernels/backends.md)

本项目尚未接入这些内核。`LYCORIS_KERNEL_BACKEND` 不会改变内置适配器的执行方式。接入时可以保留现有参数与导出逻辑，只替换匹配的计算函数；需同时处理混合精度、可训练缩放、dropout、DoRA 方向、检查点重算和多卡分片，不满足条件时仍使用原实现。

收益应按完整训练步测量，而不是只看某个内核。官方 RTX 4090 / FP16 表中，LoRA 合并路径前向加反向墙钟比值为 1.46，LoKr 分开计算为 1.47，但 LoKr 合并路径为 0.73（小于 1 代表更慢）。这不能直接换算为本训练器的速度。[官方基准](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/kernels/benchmarks.md)

融合会改变浮点运算顺序、临时显存与首次编译时间，即使数学公式相同也不保证逐位一致。训练效果、完整步耗时及不同平台的支持需要分别比较；不能把 NVIDIA 的结果直接套到 Apple MPS 或海光 DTK。
