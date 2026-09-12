# 训练参数：当前实现与使用含义

本文按本项目当前源码解释参数，不是一套保证效果的训练配方。界面中的已保存值优先于默认值；新模型版本、导入配置或预设也可能改变初值。比较实验时，应固定数据、模型、种子和预览条件，再逐项改变参数。

## 先分清三种“调度”

| 设置 | 实际控制什么 | 当前通用默认 | 不控制什么 |
| --- | --- | --- | --- |
| `objective.timestep_sampling` | 每张训练图抽到的噪声强度 `t` | `shift`：logit-normal 后应用 shift=3 | 不决定预览图的积分步数 |
| `sampling.scheduler` | 生成预览时，每一步经过的噪声时间点 | `uniform`：保留旧版均匀时间网格再应用采样 shift | 不改变优化器学习率 |
| `scheduler.type` | 学习率随优化更新次数变化的倍率 | `cosine`，预热默认 0 | 不生成图像噪声 |

训练采用 rectified flow：`x_t=(1-t)x₀+tε`，模型预测速度 `v=ε-x₀`。`t` 越接近 1，噪声越多；训练会避开精确端点。训练 `shift>1` 把分布推向较高噪声；`t_min/t_max` 是对结果裁剪，不是在区间内重新抽样。

预览的 `sampler` 决定数值求解方式：Euler 为旧默认，Heun 在非末步多做一次预测修正，ER-SDE 使用历史项与随机项。调度器另选 `uniform/simple/sgm_uniform/normal`。ER-SDE 默认最高三阶，从历史足够时才升阶；`er_sde_s_noise=0` 关闭沿途追加噪声，初始噪声仍由种子生成，且仍保留 ER-SDE 的漂移项，不能将其视为换成了 Euler/ODE 求解器。ER-SDE 从有限信噪比的近纯噪声位置启动，其端点处理不等同于 Euler。更多步数、更高阶数或更大 CFG 都不保证效果更好。

未填写预览步数/CFG/shift 时，使用模型族值：Anima 为 25/4/3；Krea 2 为 28/5.5/按图像 token 数计算 shift；Toy 为 8/2/1。单条提示词可覆盖种子、宽高、步数和 CFG。Krea 2 在训练选择 `resolution_shift` 时，其预览默认 shift 还会使用该配置的 token/mu 参考点；显式 `sampling.shift` 优先。

源码：[训练目标](../ypuddin/objectives/flow.py)、[采样分发与网格](../ypuddin/sampling/dispatch.py)、[训练预览调用](../ypuddin/train/trainer.py)、[Krea 2 默认值](../ypuddin/models/krea2/family.py)。

## LoKr 的 full，不等于完整底模训练

| 配置 | 含义 |
| --- | --- |
| `adapter.algo=lokr, rank=full` | 保留完整的 Kronecker 因子 W1、W2，不再把因子拆成低秩矩阵；仍学习 LoKr 增量 |
| `adapter.algo=full` | 训练所选线性层的完整权重，导出与底模的差分 |
| `adapter.preset=full-linear` | 扩大目标线性层范围；不改变 LoKr/LoRA/full 等算法 |

LoKr 的更新为 `ΔW=scale·(W1⊗W2)`。例如 3072×3072 线性层、`factor=-1, rank=full`，因子大小为 48×48 和 64×64，共 6400 个参数；该层完整权重有 9,437,184 个参数。这个例子说明结构差异，不代表所有层的比例相同。

`factor=-1` 选接近平方根的整数因子；正数优先使用指定因子，不能整除时选择可用因子并提示。整数 rank 达到实现的阈值时，对应因子也会自动保留完整矩阵。两个因子都完整时，缩放固定为 1，alpha/rsLoRA 不再改变它；`decompose_both` 在 `rank=full` 时无效。目标预设与按层规则决定实际训练哪些层，选择算法 `full` 不会自动训练整个模型。

源码：[LoKr](../ypuddin/adapters/lokr.py)、[因子分解](../ypuddin/adapters/factorize.py)、[缩放](../ypuddin/adapters/base.py)、[目标规则](../ypuddin/adapters/rules.py)、[完整层差分](../ypuddin/adapters/full.py)；形状与导出回归见 [适配器测试](../tests/unit/test_adapters.py)。

## 分桶、原生尺寸和每轮图片数

分桶的 `resolutions=[1024]` 指目标面积约为 1024²，不是强制裁为 1024×1024。候选尺寸按 `bucket_step` 网格生成，在基准面积容差和最大长宽比内选取；分配时最小化原图与桶的对数长宽比距离，再等比缩放至覆盖桶、中心裁剪。候选列表始终保留一个对齐后的正方形回退；极严格容差不保证这个回退也落在面积带内。

默认步长通常为 64，最大长边/短边比为 2，面积容差为 ±10%。更狭长的图不会因超出桶的长宽比而自动跳过，会进入最近的桶并裁剪。`bucket_no_upscale` 缩小所选桶以容纳小图，并按模型尺寸要求对齐；它不是原生尺寸模式。

原生模式绕过候选桶：预算内不重采样，仅中心裁去尺寸对齐边缘；超出 `native_max_pixels/native_max_side` 时等比缩小，或按 `native_overflow=error` 报错。不拉伸、不补边，不会把小图放大到统一大小；小于模型对齐尺寸的图不能用于这一模式。默认像素预算 1,048,576，单边 4096。像素预算约束输入张量，不是整体显存上限。

原生逻辑批次可包含不同尺寸，实际前向会按相同尺寸与像素预算分组，训练器再按图片数累积梯度；这不是 NaViT 序列打包。分桶批次则仅包含同尺寸训练项。单卡默认保留不足一批的尾部，不复制图片补齐；多卡为保证各卡步数一致会截去不能均分的尾部批次。

`repeats` 只在训练项列表中重复，不复制磁盘文件。分桶模式每张图按“来源分辨率数 × repeats”展开，原生模式只按 repeats 展开。例如 20 张图、两个基准分辨率、repeats=3，分桶每轮为 120 个训练项；这不等于 120 次更新。批次、梯度累积、验证排除和多卡尾部策略还会影响实际步数，应看数据计划。

源码：[候选桶与中心裁剪](../ypuddin/data/buckets.py)、[图像转换](../ypuddin/data/images.py)、[训练项展开](../ypuddin/data/dataset.py)、[原生尺寸与组批](../ypuddin/data/native.py)、[确定性批次与尾部](../ypuddin/data/sampler.py)。

### 与参考项目的设计关系

下表依据仓库内既有参考审读记录，说明思想和实现差异，不表示对所有上游版本的逐像素兼容。本项目的分桶/原生数据核心在 `ypuddin/data` 独立实现，运行不导入父目录参考项目；本文没有复制参考项目的 GPL 源码。

| 项目/范围 | 候选尺寸与组批设计 | 与本项目的关键区别 | 参考记录 |
| --- | --- | --- | --- |
| 本项目 | 面积带内的整数尺寸网格；对数长宽比距离；显式来源配置与可恢复批次位置 | 原生模式按实际尺寸分组前向，不做序列打包；单卡不重复尾部图片补批 | 上述本项目源文件 |
| sd-scripts | `make_bucket_resolutions` 预生成尺寸，按长宽比选择；no-upscale 路径可从原图尺寸推导桶 | 数据集按 blueprint/subset 组织，数据项本身可返回整个批次；参数与缓存契约不同，不能直接套用其配置 | [sd-scripts 数据链](reference/sd-scripts.md)（参见该文 `BucketManager` / `BaseDataset` 段） |
| diffusion-pipe | 长宽比可用几何间隔或显式列表，再按面积计算并对齐宽高；支持显式尺寸/帧桶 | 分层 AR/SizeBucketDataset、DeepSpeed/DP 批次与分片缓存；记录中的全局批次整除策略会截断尾部 | [diffusion-pipe 第 4 节](reference/diffusion-pipe.md) |
| AnimaLoraStudio | 基准面积 ±10%、step=64、最大长宽比；按绝对长宽比差与面积差选桶 | 本项目改用对数长宽比距离，未沿用其前端 TS 桶算法副本；参考还包含 NaViT 打包等本项目未接入的路径 | [运行核心审读](reference/animalorastudio-runtime-core.md)中 `BucketManager` / `ImageDataset` 段 |

这些共通概念不证明算法实现或历史来源完全相同；完整第三方归属仍以项目许可证与 NOTICE 为准。

## 正则图、标签与验证集

`is_reg=true` 的图片仍参与训练，用于类别先验保持。其 `prior_weight` 是单图损失倍率，默认 1；0 表示不贡献该图的损失梯度，但图片仍占批次并参与批次平均，因此不等价于删除这些图片。正则与主体出现的比例取决于各自图片数、repeats 和分辨率展开，没有自动配成一一对应。默认正则标签使用独立空的标签变换配置，不继承主体触发词；显式来源覆盖优先。

自动验证划分依据内容哈希、排除正则来源；显式验证数据中的重复图片也从训练项排除。`validation.timesteps` 实际是训练分布的分位数，不是未经变换的原始 t。修改分布、验证图片或种子后，旧损失与新损失不能直接横向比较。

新来源的标签格式默认 `auto`，同名 JSON 优先于 TXT；显式后缀仅读取该格式，旧 `.txt` 配置不会自动改写。损坏的 JSON 不会回退到 TXT。支持结构、手动编辑时的元数据保留、恢复与缓存边界见 [JSON 标签说明](JSON_CAPTIONS.md)，不能把一个 JSON 对象当作普通文本训练。普通标签查看/编辑与自动打标是不同能力，当前不提供自动打标入口。

源码：[来源与验证划分](../ypuddin/data/dataset.py)、[损失加权与平均](../ypuddin/objectives/flow.py)、[验证调用](../ypuddin/train/trainer.py)。

## 优化器、精度与恢复的边界

- 优化器下拉列出真实注册项；8-bit 选项需要 CUDA/bitsandbytes，其他扩展需要对应依赖。缺少依赖会报错，不会静默换成 AdamW。自定义 `module.Class` 仍可使用，额外参数覆盖通用参数；具体参数须符合对应构造函数。
- `model.dtype`、`loop.mixed_precision`、`memory.base_precision`、`adapter.param_dtype`、`checkpoint.save_dtype` 分别控制模型加载、训练 autocast、冻结权重存储、可训练参数和导出文件，不能互相替代。CPU/MPS 当前按 fp32 加载且训练关闭 autocast；FP8、8-bit 优化器和 CUDA 注意力扩展不能据此推断在 MPS 可用。
- Block swap 用传输换设备驻留量；激活检查点用重算换激活存储；compile 有首次编译成本且不能与 block swap 组合。实际峰值受模型、图像、优化器和平台影响，没有一组通用显存或性能保证。
- `optimizer.fused_backward=true` 尚未实现并明确拒绝。Kahan 不能与 Schedule-Free 同用。原生模式不等于 packed attention；本项目未因新增噪声调度器而增加 SDXL/Flux 等训练模型族。
- 完整 state 保存原始训练参数、优化器、数据位置和 RNG，用于恢复；普通权重导出用于推理或热启动，不能恢复全部训练进度。模型资产或数据身份变化可能拒绝恢复。相同种子不承诺跨设备、依赖版本逐位相同。
- 步骤与轮次触发器各自生效，保存和预览都可能在同一优化步各触发一次。预览独立 RNG，不参与训练梯度；当前不存在中途保存积分器历史后接着生成一张图的功能。

源码：[优化器与学习率实现](../ypuddin/optim/factory.py)、[能力检查](../ypuddin/train/plan.py)、[训练及恢复](../ypuddin/train/trainer.py)、[状态文件](../ypuddin/train/state.py)。
