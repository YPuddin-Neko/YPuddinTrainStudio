# 分桶实现对照：YPuddin、AnimaLoraStudio、sd-scripts / lora-scripts、diffusion-pipe

本报告对照本地可审计源码，说明同一张图会被怎样分配和变换，以及不同实现的边界。**我们的分桶与 ALS 思路相近，但不是完全相同的算法；`no_upscale`、原生尺寸和多卡尾批也不能按名字视为等价。** 这里没有进行训练质量或吞吐基准，因此不判断哪个更快、效果更好。

## 版本与取证范围

检查日期：2026-09-16。只读取源码，静态例子不打开用户图片、不加载模型、不初始化 GPU；没有修改任何分桶算法。

| 项目 | 本地检出提交 | 本次读取的主要实现 |
|---|---|---|
| YPuddin Train Studio | `977d466e21588b8c306bbd8382cd6c367d2bb855` | `ypuddin/data/buckets.py`、`dataset.py`、`sampler.py`、`native.py`、`images.py` |
| AnimaLoraStudio（ALS） | `3d9d2e86045b879cd19c01ac4aa2337f45a283ab` | `runtime/training/dataset.py`、`phases/dataset.py` |
| sd-scripts | `4e624302e0088e39933b31cbc71f24212e900f5f` | `library/dataset.py`、`model_util.py`、`utils.py` |
| lora-scripts | `5a42a3fb1cb6b16195e6964f12a88d58fb64aef2` | 内置 `scripts/dev/library/dataset.py` 与 `scripts/stable/library/train_util.py` |
| diffusion-pipe | `8f83dbf25d03219df705570ec03e62be04bc402f` | `utils/dataset.py`、`models/base.py` |

YPuddin 工作区同时存在其他功能修改；这里的算法分析以读取时文件为准。静态例子记录了实际抽取源码的 SHA。lora-scripts 是训练器入口与界面层，不能当作另一套独立分桶算法：其目录同时包含 stable/dev 两份 sd-scripts 代码；本地 catalog 默认指向 dev（`mikazuki/catalog/inspector.py:168`）。本报告分别核对了两份桶管理器，而不是只看界面选项。

## 1. 桶尺寸怎么生成、图片怎么选桶

| 项目 | 候选尺寸生成 | 分配图片的距离与限制 |
|---|---|---|
| 我们 | 基准面积 `base²`，默认 ±10%，宽高步长通常 64，限制长短边比例默认 2。遍历宽度后，只检查接近 `base² / width` 的三个高度：`h-step`、`h`、`h+step` | 最小化 `abs(log(bucket_ar)-log(image_ar))`；同距离再选最接近基准面积的桶。极端长图仍分到限制内最近桶 |
| ALS | 同样使用 `base²`、±10%、默认 64 步长、默认比例上限 2，但在推导范围内穷举宽高组合 | 最小化绝对比例差 `abs(image_ar-bucket_ar)`；同差距再比较面积 |
| sd-scripts / lora-scripts | 从最小边到最大边枚举宽度，按最大面积向下取整高度并加入转置；桶面积通常不超过最大面积，没有 ALS 式 ±10% 面积带 | 已有完全相同尺寸优先，否则选绝对比例差最小桶；最小/最大边界间接决定可覆盖比例 |
| diffusion-pipe | 先指定比例列表，或在 `min_ar..max_ar` 之间等比生成；再对每个基准面积解出宽高，并按模型粒度分别四舍五入。也支持直接指定 `(宽, 高, 帧数)` 列表 | 在对数比例空间找最近比例；视频另检查可用帧桶。四舍五入后最终比例和面积会稍变 |

源码依据：我们 [buckets.py](../ypuddin/data/buckets.py) 31–88 行；[ALS dataset.py](../../AnimaLoraStudio/runtime/training/dataset.py) 193–245 行；[sd-scripts dataset.py](../../sd-scripts/library/dataset.py) 229–270 行、[model_util.py](../../sd-scripts/library/model_util.py) 1388–1416 行；[diffusion-pipe dataset.py](../../diffusion-pipe/utils/dataset.py) 416–425、495–507、838–871 行。

**具体差异：** 同为比例上限 2、步长 64、面积容差 10%，实际运行两边原始生成方法得到：

| 基准 | 我们的桶数 | ALS 桶数 |
|---:|---:|---:|
| 512 | 9 | 9 |
| 1024 | 33 | 37 |
| 1536 | 51 | 83 |
| 2048 | 69 | 147 |
| 4096 | 137 | 567 |

差异来自候选枚举方式和边界，不是“同算法只是界面不同”。更多桶可能更贴近部分比例，也会让样本分散到更多尺寸；这里只陈述机制，没有实测它对特定数据集的吞吐或质量净影响。

`area_tolerance` 是候选桶的面积容差，**不是允许裁掉多少画面**。另外，我们会强制保留对齐后的方桶作为兜底；因此非常规基准、严格零容差等边界下，不能把面积带当作所有输出绝对不越过的硬限制（`buckets.py:72`）。

## 2. 进入桶后，是裁切还是补边

- **我们支持两种独立选择。** `crop` 使用等比覆盖缩放再中心裁切；`pad` 使用等比完整放入画布，RGB 通过边缘像素延展补齐，并产生有效区域掩码。补边不计入直接训练损失，但仍是模型前向看到的上下文，不能称为“补边完全无影响”。新项目是否选择 pad 由项目初始化决定；缺少该字段的旧配置沿用 schema 的 crop 默认值。
- **ALS 常规桶路径** 使用 Lanczos 覆盖缩放和中心裁切。这里未发现该路径与我们等价的完整画面补边开关。
- **sd-scripts 常规路径** 按分桶结果缩放，然后中心裁切；开启 `random_crop` 可以改为随机裁切。它与我们固定中心裁切不完全相同。
- **diffusion-pipe 通用图片预处理** 使用 `ImageOps.fit`，属于覆盖缩放后裁切；透明图先合成白底。模型私有预处理还可能有额外处理，本报告不把通用路径结论扩大到所有视频、控制图和模型分支。

源码依据：[我们的 images.py](../ypuddin/data/images.py) 35–90 行、[buckets.py](../ypuddin/data/buckets.py) 98–131 行；[ALS dataset.py](../../AnimaLoraStudio/runtime/training/dataset.py) 764–787 行；[sd-scripts utils.py](../../sd-scripts/library/utils.py) 206–234 行；[diffusion-pipe models/base.py](../../diffusion-pipe/models/base.py) 41–53、173–197 行。

## 3. `no_upscale` 不是原生尺寸的统一别名

| 路径 | 小图和长图的实际处理 |
|---|---|
| 我们的 `bucket_no_upscale` | 先选常规桶；若原图任一边小于桶边，按比例缩小该桶，再按模型 `align` 向下对齐。仍保留“先选桶”的比例约束；不是直接保留原图比例 |
| sd-scripts 的 `bucket_no_upscale` | 不再依赖预生成桶的 min/max 范围。超过最大面积才等比缩小；否则保留原尺寸，再按 `reso_steps` 向下取整生成动态桶并裁边 |
| ALS 常规 ARB | 本次桶管理器没有同名 no-upscale 参数；其原生尺寸是另一条显式路径 |
| diffusion-pipe 通用 ARB | 本次通用桶路径按配置面积得到目标尺寸，可以放大小图；没有核实到与上述两种行为相同的通用 no-upscale 开关 |

依据：[我们的 buckets.py:84](../ypuddin/data/buckets.py)、[sd-scripts dataset.py](../../sd-scripts/library/dataset.py) 272–310、660–665 行。我们的 `align` 与 sd-scripts 的 `reso_steps` 含义也不同，不应直接把两个设置值互换。极小于对齐粒度的输入需要额外检查；本报告的小例均大于对齐粒度，不把它们外推成所有病态尺寸保证。

## 4. 五种尺寸的可复核小例

统一展示基准 1024、步长 64。我们/ALS 的比例上限为 2、容差 10%；sd-scripts 的最小边 512、最大边 2048；diffusion-pipe 使用其示例配置的 7 个等比分布比例（0.5–2），本例明确把尺寸取整粒度设为 64。**这是一组可比的显式参数，不代表各模型的默认配置完全相同。** 尺寸顺序均为宽×高。

| 原图 | 我们常规桶 | ALS | sd-scripts / lora stable / lora dev | diffusion-pipe |
|---|---|---|---|---|
| 1024×1024 | 1024×1024 | 1024×1024 | 1024×1024 | 1024×1024 |
| 1200×800 | 1280×832 | 1280×832 | 1216×832 | 1280×832 |
| 800×1200 | 832×1280 | 832×1280 | 832×1216 | 832×1280 |
| 2048×512 | 1408×704 | 1408×704 | 2048×512 | 1472×704 |
| 400×300 | 1216×896 | 1216×896 | 1152×896 | 1152×896 |

这五张恰好在我们与 ALS 中选到相同尺寸，**不代表两套候选集合或距离函数等价**。例如 4:1 原图受我们/ALS 的 2:1 桶限制，进入 1408×704 桶：crop 会从缩放后的 2816×704 中只保留一半横向画面；pad 则把完整图缩至 1408×352，画布一半是补边。这是几何事实，不是质量评分。

启用各自 no-upscale 后，对比更明显：

| 原图 | 我们 no-upscale（align=16） | sd-scripts no-upscale（step=64） |
|---|---|---|
| 1200×800 | 1200×768 | 1152×768 |
| 2048×512 | 1024×512 | 2048×512 |
| 400×300 | 400×288 | 384×256 |

原始数值、源码 SHA、具体参数与脚本保存在本次工作区外证据：[static-examples.json](../../remote-testing/bucketing-comparison-20260916/static-examples.json)、[static_examples.py](../../remote-testing/bucketing-comparison-20260916/static_examples.py)。脚本抽取并执行现有原始类/函数，不重新实现它们的桶选择算法；没有图片解码或模型计算。

## 5. 多分辨率、重复次数与尾批

**多分辨率并不是每次随机选一个尺寸。**

- 我们对每个来源选取 `source.resolutions` 或全局 `dataset.resolutions`，每张图在每个尺寸建立 `repeats` 份训练条目。原生模式只建一档；不能把 `[512,1024]` 误解为总样本数不变。
- ALS 同样会在多档分辨率展开图片，也支持目录的尺寸覆盖；原生模式明确收拢多档，避免重复相同原生样本。
- diffusion-pipe 的每个比例组会遍历全部 `resolutions` 建尺寸数据集，再应用来源重复次数；其示例配置也明确说明每张图会用于所有指定面积。
- sd-scripts 是每个 dataset 自己的 resolution/bucket 配置。可以用多个 dataset 配同一来源实现多档，但这里没有发现与上述全局列表完全相同的自动展开接口；不要只对照参数名称推算 epoch。

依据：[我们的 dataset.py](../ypuddin/data/dataset.py) 171–218 行；[ALS dataset.py](../../AnimaLoraStudio/runtime/training/dataset.py) 275–294、486–520 行；[diffusion-pipe dataset.py](../../diffusion-pipe/utils/dataset.py) 229–232、337、419–435 行。

| 项目 | 单卡桶尾 | 多卡分配要点 |
|---|---|---|
| 我们 | `drop_last=False`，保留每桶短批，不复制图补满 | 先形成并打乱同尺寸批，再把完整批分给 rank；只保留可整除卡数的批次数，最后不足一组的批会丢弃。不同 rank 同一步不一定同一桶；不能宣称“多卡每轮所有图片必用且绝不丢尾” |
| ALS | 虽然采样器构造默认 `drop_last=True`，实际训练缓存/非缓存入口都显式传 False，保留短批 | 本次该采样器与入口没有独立 rank 分片实现；未据此验证 ALS 的多卡尾批行为，不能套用我们的结论 |
| sd-scripts / lora | 每桶按 ceil 生成批索引，最后一批切片可以不足 batch_size | 后续 DataLoader 交给 Accelerate。是否为跨 rank 整齐批次补齐，还取决于安装的 Accelerate 版本/配置；仅看桶管理器不能证明不重复 |
| diffusion-pipe | 每个尺寸数据集按全局 batch 截断；不足一整个全局 batch 的尺寸桶可能被全部丢弃并警告 | 全局 batch 必须整除数据并行卡数；每个 rank 取该全局批次中的连续片段 |

依据：[我们的 sampler.py](../ypuddin/data/sampler.py) 34–57 行、[distributed.py](../ypuddin/train/distributed.py) 158–188 行；[ALS phases/dataset.py](../../AnimaLoraStudio/runtime/training/phases/dataset.py) 248–275 行；[sd-scripts dataset.py](../../sd-scripts/library/dataset.py) 707–715、989–992 行及 [train_network.py](../../sd-scripts/train_network.py) 1187、1303 行；[diffusion-pipe dataset.py](../../diffusion-pipe/utils/dataset.py) 347–396 行。

## 6. 原生尺寸与 NaViT 要分开看

我们已实现原生尺寸，但它是**按形状拆微批、按图片数累积梯度**，不是把不同图片 token 拼成一次块对角注意力。正常预算内，crop 只做向下对齐裁边且避免重新采样；pad 向上对齐并保留完整图。超像素/单边预算可缩小或报错，不悄悄跳过。400×300、align16 的例子分别是 crop 400×288、pad 400×304。

**我们目前明确禁止原生模式用于多卡 DDP/FSDP。** 验证层会提示“多卡训练暂不支持原图异形批次，请使用标准分桶”；不能因为 `NativeBatchSampler` 有 rank 分支就写成产品已支持。依据：[native.py](../ypuddin/data/native.py) 31–101、123–144 行；[training_rules.py](../ypuddin/config/training_rules.py) 54–62 行。

ALS 有两层独立概念：原生尺寸规划绕过 ARB；另有显式 `navit_packing` 路径按 token 预算打包多张异形图，通过 Anima 专用块对角前向处理。它不是只把普通 batch sampler 改名；也不能据此认为我们已经有同等 NaViT 实现。依据：[ALS phases/dataset.py](../../AnimaLoraStudio/runtime/training/phases/dataset.py) 229–247 行、[dataset.py](../../AnimaLoraStudio/runtime/training/dataset.py) 1628–1669 行及 [families/anima/navit.py](../../AnimaLoraStudio/runtime/training/families/anima/navit.py) 52–73 行。

本次没有对 sd-scripts 或 diffusion-pipe 全部模型分支做 NaViT 能力审计；这里仅比较查到的通用分桶和图像路径，不把“本次未核实”写成项目绝对不支持。

## 使用上的实际含义与未完成比较

现在可以根据数据特点明确选择：需要保留完整构图时检查 image_fit；想避免小图放大时区分 no-upscale 与原生尺寸；多分辨率要把训练条目增长计入步数；多卡必须检查每个桶和全局批次是否足够。我们的数据计划、桶分布和实际条目数比只看“1024”这一项更能说明最终输入。

尚未完成的是相同图片、相同模型、相同有效批量下的多实现吞吐/显存/收敛对照，以及 ALS、sd-scripts 实际安装环境的多卡尾批实验。本报告不能替代这些测试，也没有因本次分析新增或修改产品分桶策略。
