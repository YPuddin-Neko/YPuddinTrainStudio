# 训练参数：当前实现与使用含义

本文按 v0.5.6 当前源码解释参数，不是一套保证效果的训练配方。界面中的已保存值优先于默认值；新模型版本、导入配置或预设也可能改变初值。比较实验时，应固定数据、模型、种子和预览条件，再逐项改变参数。

## 先分清三种“调度”

| 设置 | 实际控制什么 | 当前通用默认 | 不控制什么 |
| --- | --- | --- | --- |
| `objective.timestep_sampling` | 每张训练图抽到的噪声强度 `t` | `shift`：logit-normal 后应用 shift=3 | 不决定预览图的积分步数 |
| `sampling.scheduler` | 生成预览时，每一步经过的噪声时间点 | `uniform`：保留旧版均匀时间网格再应用采样 shift | 不改变优化器学习率 |
| `scheduler.type` | 学习率随优化更新次数变化的倍率 | `cosine`，预热默认 0 | 不生成图像噪声 |

训练采用 rectified flow：`x_t=(1-t)x₀+tε`，模型预测速度 `v=ε-x₀`。`t` 越接近 1，噪声越多；训练会避开精确端点。训练 `shift>1` 把分布推向较高噪声；`t_min/t_max` 是对结果裁剪，不是在区间内重新抽样。

预览的 `sampler` 决定数值求解方式。界面直接使用 Euler、Heun、ER-SDE 名称，不把“一阶、二阶”当作质量档位：Euler 是旧默认，每个积分步评估一次速度；Heun 先预测再校正，除末步外通常多评估一次速度；ER-SDE 使用历史项与随机项，默认阶数由内部按历史长度逐步处理。CFG 引导可能让每次速度评估分别计算正向和负向条件，因此“一次评估”不总是一次模型前向。调度器另选 `uniform/simple/sgm_uniform/normal`。ER-SDE 默认最高三阶，从历史足够时才升阶；`er_sde_s_noise=0` 关闭沿途追加噪声，初始噪声仍由种子生成，且仍保留 ER-SDE 的漂移项，不能将其视为换成了 Euler/ODE 求解器。ER-SDE 从有限信噪比的近纯噪声位置启动，其端点处理不等同于 Euler。更多步数、更高阶数或更大 CFG 都不保证效果更好。

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

## 分桶、原生尺寸和完整画面

`resolutions=[1024]` 表示候选桶的目标面积约为 1024²，不是把所有图片变成 1024×1024。当前 Anima / Krea 2 新项目仍使用这个默认值。界面通常填写一个基准分辨率；高级多分辨率列表继续兼容。候选宽高按 `bucket_step` 网格生成，限定面积容差与最大长宽比；分配时最小化原图与桶的对数长宽比距离，再以面积差打破平局。正方形回退始终存在，极严格容差不保证它也在面积带内。

图像如何进入目标尺寸由 `dataset.image_fit` 决定，分桶与原生模式均使用这个设置：

| 图像适配 | 像素处理 | 直接损失区域 | 默认与兼容 |
| --- | --- | --- | --- |
| `pad`：保留完整画面 | 原图等比缩放至能完整放入画布，居中；剩余区域复制最邻近的边缘像素 | 图像有效区域；纯补边区域不计直接损失 | Web 新项目、从模型族默认新建的配置使用此项 |
| `crop`：裁切填满尺寸（旧模式） | 分桶先等比缩放至覆盖桶，再中心裁剪；原生按旧对齐裁剪路径处理 | 保留的图像区域，可再叠加用户 Mask | 缺少此字段的旧配置仍是此项；已有显式选择不迁移 |

**保留完整画面不等于没有训练影响。** RGB 补边仍参与 VAE 编码与模型上下文计算，代码没有移除这些 token，也没有为注意力添加隔离遮罩。有效区 Mask 按面积映射到 latent 损失：纯补边单元为零，跨边界单元使用面积权重。它避免把纯补边位置当成直接监督目标，但不能保证与完全没有补边的训练等价，不能承诺效果或显存更好。

补边有效区始终生效，不依赖 `dataset.masked_loss` 开关；该开关控制用户 `.mask.png` / alpha 是否另行加权。启用用户 Mask 后，它与 RGB 使用相同尺寸、偏移和翻转，再与有效区相乘。原生或 `pad` 模式下，用户 Mask 尺寸必须与 EXIF 校正后的原图一致。训练、验证、在线编码和 latent 缓存均走同一几何规则。

默认桶步长通常为 64，最大长边/短边比为 2，面积容差为 ±10%。更狭长的图不会自动跳过：它会进入最近的桶，按所选方式补边或裁切。`bucket_no_upscale` 会缩小小图的目标桶；`pad` 同时限制图像缩放不超过原尺寸。它仍使用桶分配，不等于原生模式；旧 `crop` 对齐规则保留。

原生模式绕过候选桶。`native + pad` 在预算允许时不重采样，只把画布向上对齐到模型尺寸单位；预算包含补边。超出 `native_max_pixels/native_max_side` 后，默认寻找满足对齐画布预算的最大可用等比缩小比例；`native_overflow=error` 则报错，不会静默跳过。它不会放大小图，也能通过补边容纳短边小于对齐单位的图。`native + crop` 保留旧行为：向下对齐并中心裁去边缘，超预算后缩小再裁切，短边小于模型对齐单位时仍报错。默认像素预算为 1,048,576、单边上限为 4096；这是画布/前向像素预算，不是整体显存上限。

原生逻辑批次可包含不同尺寸，实际前向按相同尺寸与像素预算分组，再按图片数累积梯度；**未实现 NaViT 序列打包**。分桶批次仅包含同尺寸训练项。单卡默认保留不足一批的尾部，不复制图片补齐；底层 sampler 的多 rank 切分会截去不能均分的尾部批次，但当前训练器未接入 DDP，不能把该接口当作多卡训练支持。

数据计划会报告所选适配方式、实际缩放后宽高、有效区域、补边/裁切量和画布比例；明细最多列 100 种图像/尺寸组合，聚合统计仍覆盖全部训练项。latent 缓存为 `pad` 加入独立几何标记，避免误用旧裁切缓存；`crop` 保留旧缓存键和缺省数据指纹。主动从 `crop` 改成 `pad` 会改变数据身份，不能把它当作原训练的精确续训；可以另开版本或以权重热启动。

兼容提醒：Pydantic 的 `DatasetConfig.image_fit` 仍以 `crop` 作为缺省，确保旧 JSON/TOML 读取不改语义。直接使用 CLI/手写配置而希望补边时，请明确设置 `dataset.image_fit="pad"`；不要仅依据 Web 新项目的默认值推断所有入口。

`repeats` 只展开训练项，不复制磁盘文件。分桶每张图按“来源分辨率数 × repeats”展开，原生只按 repeats 展开。例如 20 张图、两个基准分辨率、repeats=3，分桶每轮为 120 个训练项，不是 120 次优化更新。批次、梯度累积、验证排除及尾部策略会影响步数，应以数据计划为准。

源码：[候选桶与适配几何](../ypuddin/data/buckets.py)、[RGB 与有效区](../ypuddin/data/images.py)、[缓存身份与训练项](../ypuddin/data/dataset.py)、[原生尺寸与组批](../ypuddin/data/native.py)、[数据计划](../ypuddin/train/plan.py)、[新项目默认值](../ypuddin/server/family_config.py)。[完整画面回归](../tests/unit/test_image_fit.py)覆盖四角保留、补边损失/梯度、用户 Mask、在线/缓存、训练/验证与恢复。

### 与三个参考训练器的实际差异

以下是 2026-09-12 对父目录本地参考副本的源码核对，行号属于该副本，不代表所有上游版本。参考路径以父目录为根；它们不随本项目运行或打包。本项目 `ypuddin/data` 的实现独立，未复制这些参考训练器的 GPL 数据流水线源码；模型 vendor 的既有许可与来源另见 NOTICE。

| 项目/模式 | 尺寸与适配事实 | 是否裁切 | 本地参考证据 |
| --- | --- | --- | --- |
| 本项目 `bucket + pad` | 面积带整数网格、对数长宽比距离；完整等比缩放、边缘复制补边、有效区损失权重 | 不裁掉原图画面；缩小仍会损失细节 | `ypuddin/data/buckets.py:54,79,107`；`data/images.py:60,83` |
| 本项目 `native + pad` | 每图向上对齐，预算计算包含补边，必要时等比缩小；分组前向 | 不裁掉原图画面 | `ypuddin/data/native.py:31,69` |
| AnimaLoraStudio 分桶 | 基准面积 ±10%、step=64；按绝对长宽比差选择；像素阶段 resize-cover 后中心 crop | 是；比例相符时可以不丢画面 | `AnimaLoraStudio/runtime/training/dataset.py:193–245,780–787`；Mask 同步 `794–797` |
| AnimaLoraStudio native | floor-16 对齐；超预算缩小；使用同一 resize-cover/中心 crop 像素路径 | 对齐边缘会裁切；“零 padding”不等于“零裁切” | 同文件 `67–90,395–407,764–787` |
| sd-scripts 分桶 / no-upscale | 预生成候选桶、按绝对长宽比差选桶；no-upscale 推导缩小尺寸后向下对齐 | 中心或随机裁切；no-upscale 也明确用裁切代替补边 | `sd-scripts/library/model_util.py:1388–1416`；`library/dataset.py:251–306`；`library/utils.py:206–226` |
| diffusion-pipe AR / 显式尺寸桶 | AR 可显式或几何间隔，按对数距离选取；面积推导宽高后对齐；RGB 使用 `ImageOps.fit` | 是；未指定尺寸桶时仍先对齐，再走 fit | `diffusion-pipe/utils/dataset.py:419–424,495–507,809–818`；`models/base.py:41–53,166–186` |

比较针对训练图像的实际转换，不把 token 序列补齐、视频帧或其它模型的 padding 混作图像策略。更完整的上下文见 [sd-scripts](reference/sd-scripts.md)、[diffusion-pipe](reference/diffusion-pipe.md)、[Anima 运行核心](reference/animalorastudio-runtime-core.md)。

## 正则图、标签与验证集

`is_reg=true` 的图片仍参与训练，用于类别先验保持。其 `prior_weight` 是单图损失倍率，默认 1；0 表示不贡献该图的损失梯度，但图片仍占批次并参与批次平均，因此不等价于删除这些图片。正则与主体出现的比例取决于各自图片数、repeats 和分辨率展开，没有自动配成一一对应。默认正则标签使用独立空的标签变换配置，不继承主体触发词；显式来源覆盖优先。

自动验证划分依据内容哈希、排除正则来源；显式验证数据中的重复图片也从训练项排除。`validation.timesteps` 实际是训练分布的分位数，不是未经变换的原始 t。修改分布、验证图片或种子后，旧损失与新损失不能直接横向比较。

新来源的标签格式默认 `auto`，同名 JSON 优先于 TXT；显式后缀仅读取该格式，旧 `.txt` 配置不会自动改写。损坏的 JSON 不会回退到 TXT。支持结构、手动编辑时的元数据保留、恢复与缓存边界见 [JSON 标签说明](JSON_CAPTIONS.md)，不能把一个 JSON 对象当作普通文本训练。普通标签查看/编辑与自动打标是不同能力，当前不提供自动打标入口。

源码：[来源与验证划分](../ypuddin/data/dataset.py)、[损失加权与平均](../ypuddin/objectives/flow.py)、[验证调用](../ypuddin/train/trainer.py)。

## 标签处理与高级参数入口

普通视图按数据来源显示标签格式（自动 / TXT / JSON / 自定义）；自动模式仍是同名 JSON 优先 TXT。无需再次填写已写入标签的触发词。`prefix/suffix/trigger_word` 已移入高级视图，已有值继续生效；普通视图会提示旧配置含额外标签改写，避免隐藏值被误以为已清除。`keep_tokens` 在开启打乱标签或 tag dropout 时才显示，默认 0。

`dataset.text_encoding`、训练图像缓存 `cache_latents` 和 `memory.offload_text_encoder` 移入高级视图，默认值和已保存值不因隐藏而修改。`auto` 按模型能力选择：每步处理标签支持随步变化；训练前缓存标签先计算条件、使用有界随机变体，并可卸载编码器。**两种方式都在训练电脑本地处理**，不是在线云服务。图像缓存默认在训练启动前于本机准备；关闭后逐批本机编码。

底模加载精度 `model.dtype`、导出精度 `checkpoint.save_dtype`、完整状态恢复路径和自定义权重名前缀也放入高级视图。项目任务中默认名 `lora`、空名或缺省名会在入队时转换为安全的“项目显示名_v版本号”；非默认自定义名保留，旧任务快照与文件不改。输出目录仍按项目/版本/任务 ID 隔离，不由文件名前缀替代目录隔离。

源码：[字段元数据](../ypuddin/config/schema.py)、[标签格式与高级视图](../frontend/src/schema/SchemaForm/SchemaForm.tsx)、[项目权重命名](../ypuddin/server/output_binding.py)。

## 当前 Loss、平均 Loss 与显示 EMA

任务顶部的 `Loss` 是最近记录的优化步损失。`平均 Loss` 使用训练器实际累计的 `loss_sum / loss_count`：每次成功的优化器更新计数一次，低频日志间隔中的更新也会计入，跳过的非有限损失更新不计入。它是**优化步损失的算术平均**，不是 EMA，也不是跨全数据集重新计算的验证损失；批次尾部大小不同时，也不等价于把整个过程所有图片再按图片数做一次总体平均。

累计状态存于 `progress.extra` 并进入完整 state。正常恢复继续原累计；旧 state 没有累计历史时，从此次恢复后的成功更新开始计数，`loss_mean_scope="since_resume"` 明示范围；新训练为 `run`。step 事件携带 `loss_mean/loss_count/loss_mean_scope`，服务保存到 `job.latest`。界面显示截至最近收到事件的累计值，不是每次渲染猜测尚未上报的更新。

旧任务没有累计值时，界面只计算已记录日志中有效 Loss 的平均值，并在说明中标记“已有日志中训练步”；未知显示未知，真实 0 保留。图表的“显示 EMA”仅平滑曲线，不参与上述平均，也不改变训练参数或权重 EMA。

源码：[成功更新与事件](../ypuddin/train/trainer.py)、[完整状态](../ypuddin/train/state.py)、[SSE 与 latest](../ypuddin/server/supervisor.py)、[监控显示](../frontend/src/pages/JobDetail/JobDetail.tsx)；[稀疏日志/恢复测试](../tests/unit/test_training_loss_mean.py)、[七项指标交互测试](../frontend/tests/jobSummary.test.tsx)。

## 优化器、精度与恢复的边界

- 优化器下拉列出真实注册项；8-bit 选项需要 CUDA/bitsandbytes，其他扩展需要对应依赖。缺少依赖会报错，不会静默换成 AdamW。自定义 `module.Class` 仍可使用，额外参数覆盖通用参数；具体参数须符合对应构造函数。
- `model.dtype`、`loop.mixed_precision`、`memory.base_precision`、`adapter.param_dtype`、`checkpoint.save_dtype` 分别控制模型加载、训练 autocast、冻结权重存储、可训练参数和导出文件，不能互相替代。CPU/MPS 当前按 fp32 加载且训练关闭 autocast；FP8、8-bit 优化器和 CUDA 注意力扩展不能据此推断在 MPS 可用。
- Block swap 用传输换设备驻留量；激活检查点用重算换激活存储；compile 有首次编译成本且不能与 block swap 组合。实际峰值受模型、图像、优化器和平台影响，没有一组通用显存或性能保证。
- `optimizer.fused_backward=true` 尚未实现并明确拒绝。Kahan 不能与 Schedule-Free 同用。原生模式不等于 packed attention；本项目未因新增噪声调度器而增加 SDXL/Flux 等训练模型族。
- 完整 state 保存原始训练参数、优化器、数据位置和 RNG，用于恢复；普通权重导出用于推理或热启动，不能恢复全部训练进度。模型资产或数据身份变化可能拒绝恢复。相同种子不承诺跨设备、依赖版本逐位相同。
- 步骤与轮次触发器各自生效，保存和预览都可能在同一优化步各触发一次。预览独立 RNG，不参与训练梯度；当前不存在中途保存积分器历史后接着生成一张图的功能。

源码：[优化器与学习率实现](../ypuddin/optim/factory.py)、[能力检查](../ypuddin/train/plan.py)、[训练及恢复](../ypuddin/train/trainer.py)、[状态文件](../ypuddin/train/state.py)。
