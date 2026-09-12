# v0.5.4 工作区、采样指标与项目管理变更记录

> 日期：2026-09-12。本文记录本轮源码、全量回归、浏览器检查及正式 8876 服务升级的数据保留证明。它不替代源码包的独立交付证明，也不将 v0.5.3 的历史结果记作本轮结果。

## 项目导航与页面层级

项目侧栏现在只有三个主阶段：**训练数据 → 训练参数 → 训练结果**。模型族、底模组件路径与训练超参数统一在“训练参数”的“模型”分区编辑，不再维护另一套“模型准备”表单和保存状态。下载与登记权重仍在环境设置的模型页。

旧 `?step=models` 入口转到对应项目、版本的 `/train?tab=model`；数据页的“选择训练模型”和正则图的模型配置入口使用同一路径。训练分区通过 URL 保留，版本切换保留当前参数分区，仍先处理当前版本草稿。任务调度留在全局队列，第三阶段只查看本版本的模型权重、采样图和训练记录。

本轮保留既有数据导航保护：逐图标签保存失败时留在编辑器，打开的遮罩画布不能被浏览器后退或前进静默丢弃；设置抽屉保留项目背景。合并模型入口没有取消这些保护或把不同版本草稿混在一起。

主要实现：[ProjectWorkflow](../frontend/src/components/ProjectWorkflow.tsx)、[ProjectDetail](../frontend/src/pages/ProjectDetail/ProjectDetail.tsx)、[TrainConfig](../frontend/src/pages/TrainConfig/TrainConfig.tsx)、[projectVersions](../frontend/src/utils/projectVersions.ts)。

## 边框、底色与吸顶

项目正文标题使用页面底色，移除重复外框；数据子步骤保留一条底部分隔线。标签搜索图标放入输入框，搜索工具条不再叠加额外底线。训练工具条、参数分区和固定操作区域的底色与所在页面衔接，减少没有外框却出现异色矩形的情况。

项目标题和任务监控栏的吸顶区域补齐滚动容器顶部 padding 的遮盖，避免滚动后旧参数从标题上方露出。模型工具条在设置抽屉中使用对应表面底色。上述改动是针对已观察到的边框、浅色块和透出问题；已执行的页面和视口检查见下文证据矩阵。

相关样式集中在 `frontend/src/styles/project-sidebar.css`、`frontend/src/styles/index.css`、`frontend/src/pages/TrainConfig/training-workspace.css`、`frontend/src/components/datasets/caption-viewer.css`、`frontend/src/components/datasets/dataset-pipeline.css` 和 `frontend/src/pages/JobDetail/job-detail.css`。

## 采样参数与训练图表

采样参数按用途配对排列，开关、数值输入和可选值控件使用一致的行高。`sampling.output_dir` 是服务分配的任务目录，不再作为可编辑训练字段展示。

- 采样步数、CFG、shift 留空时继承模型族默认，界面显示对应提示。将可选数值切换为显式值时，优先使用族默认或合法下限，避免步数等字段被初始化为非法的 0。
- 每条提示词的 seed、宽、高、steps、CFG 留空表示继承全局设置，不写入假的 42/1024 等覆盖值；CFG 与 seed 的 0 是有效输入。提示词可单独删除，删除操作有明确名称。
- 模型路径候选按模型族与组件类型筛选，排除已知失效路径；模型库变更或窗口重新获得焦点时刷新候选，不覆盖用户已经填写的路径。

任务详情默认展示主损失图：原始损失与用于显示的 EMA 平滑损失。平滑系数只重新计算图表，不改变训练参数、优化器或适配器权重的 EMA。学习率、梯度范数、验证和性能诊断分开，放在可展开区域中。`w1` / `w2` 表示 LoKr 的两类矩阵参数组，不表示 Toy 模型的两层。

图例、坐标提示和说明支持中文与英文；提示值限制有效数字，避免长小数遮挡；图例、坐标轴和底部缩放滑条各有空间，滚动图例已注册。缩放使用底部滑块；不以 Ctrl/普通滚轮作为必要操作。MPS 保持“当前训练分配量”语义，CUDA 对应峰值分配量；没有记录的指标不补成 0。

相关实现：[SchemaForm](../frontend/src/schema/SchemaForm/SchemaForm.tsx)、[JobDetail](../frontend/src/pages/JobDetail/JobDetail.tsx)、[metricPresentation](../frontend/src/pages/JobDetail/metricPresentation.ts)、[EChart](../frontend/src/components/EChart.tsx)。

## 采样卡的精确训练损失

训练循环在每个优化器步记录该步真实的 `group_loss`，写入可恢复进度，并在 `sample.saved` 事件中携带 `loss`。因此即使 `log_every` 较稀疏，采样也不需要猜测最近一条曲线值。REST `/jobs/{id}/samples` 与 SSE `job.sample` 使用同一规则，任务详情和版本结果共用 [SampleLoss](../frontend/src/components/SampleLoss.tsx) 展示。

- 正数 step 的有限 loss 显示为约五位有效数字；真实的 0 会显示。
- step 0 标明“初始采样 · 未训练”；没有精确记录时显示“未记录”。
- 新事件显式 `loss: null` 不被其他指标覆盖。旧事件仅可使用同一次运行/恢复段内、在该采样之前出现的同一正数 step 记录；不能借用相邻步、未来事件或恢复前的同号 step。
- 这是对应训练步的损失，不是单张采样图的质量评分，也不是该图片单独计算的损失。

相关后端：[trainer.py](../ypuddin/train/trainer.py)、[sample_events.py](../ypuddin/server/sample_events.py)、[supervisor.py](../ypuddin/server/supervisor.py)。

## 项目卡片、分类与手动封面

项目页改成封面卡片，保留单一进入项目的链接、项目名称/ID、分类、活动版本模型族与数量摘要。更多菜单提供编辑、归档/恢复和删除；支持键盘方向键、Home/End、Escape 及弹窗焦点恢复。分类可以选择常用项或输入自定义值，与模型族分开存储。

列表支持分类、关键词和归档筛选及每页 24 个项目；筛选变化回到首页，记录减少时校正超出范围的页码。当前前端取得包含归档的项目集合再本地筛选分页，避免“显示已归档”勾选后仍无法恢复项目；服务另提供带 `page/page_size` 的分页查询及分类计数接口。

创建和编辑弹窗先保存项目信息，再提交用户选择的封面。封面失败会说明项目信息已经保存，保留所选文件供重试；新建项目在此阶段固定已创建的 ID 和模型族，重试不会再次创建项目。取消尚未提交的封面选择不改磁盘，已经保存的项目也不会因关闭失败提示而被删除。

封面由 [project_covers.py](../ypuddin/server/project_covers.py) 独立管理：仅接受静态 JPEG、PNG、WebP，限制 8 MiB、1600 万像素和单边 8192；重新编码最长边 640 的 WebP，不沿用上传文件名或图片元数据。文件位于所属项目 `.studio/<随机标识>.webp`，不登记训练数据源。路径检查拒绝符号链接重定向；替换失败保留原封面，读取接口只返回已登记封面。

| 接口 | 用途 |
|---|---|
| `GET /api/projects` | 兼容原列表返回；显式分页时返回 items/total/page/page_size，支持分类、关键词和归档条件 |
| `GET /api/project-categories` | 分类名称、计数及未分类数量 |
| `POST /api/projects` / `PATCH /api/projects/{id}` | 创建项目及修改显示名称、备注、分类等元数据；分类显式 null 可清除 |
| `POST /api/projects/{id}/cover` | multipart 的单个 `file` 上传 |
| `GET` / `DELETE /api/projects/{id}/cover` | 读取或移除该项目封面 |

数据库只增加可空的 `category` 与 `cover_key` 字段；升级不移动旧项目目录或修改旧训练配置。新项目继续使用手填 ASCII ID 与独立多语言显示名称。前端入口：[Projects](../frontend/src/pages/Projects/Projects.tsx)、[ProjectEditor](../frontend/src/pages/Projects/ProjectEditor.tsx)、[ProjectCardMenu](../frontend/src/pages/Projects/ProjectCardMenu.tsx)。

## 新项目与版本的模型族兼容

创建项目可指定初始模型族；创建版本可沿用来源族或选择另一已接入的族。当前训练后端支持 **Anima、Krea 2、Toy**，Toy 用于诊断。**Flux、SDXL 尚未接入，界面禁选，后端拒绝创建；不能把候选名称当作已支持。**

[family_config.py](../ypuddin/server/family_config.py) 定义初始与跨族配置规则：

- 同族保留原训练配置，不因模型库默认发生变化而覆盖显式路径；未指定族时保留原版本复制语义。
- 跨族从完整 `TrainConfig` 默认配置重建，只保留 dataset/validation 数据准备设置，并将文本编码模式调整为目标族兼容默认。旧适配器规则、组学习率、恢复权重、换出块数、采样提示词及覆盖等不会带入新族。通用 rank/alpha/factor/scheduler 保持配置模型默认，不额外套用推荐配方。
- 族内目标 preset 和采样值来自模型族契约。Krea 2 使用 cached 文本、resolution_shift 的 `(256, 6400)` 端点和分块重计算；不自动启用 FP8 或指定 CUDA，不下载权重。
- 默认权重只从目标族已登记的 `is_default` 资产中选择，并核对组件角色、文件/目录类型、存在性和允许路径；失效项留空，不能借用别族路径。
- 是否复制图片由原有 copy/empty 选项单独决定；copy 仍独立复制图片、标签、遮罩及验证源，empty 清空来源。版本快照照旧重定位缓存/任务目录并清除 resume，不复制训练记录、采样和权重产物。旧版本配置和已启动任务快照不改写。

版本响应增加 `family` 投影。尚无配置的复制中版本可返回 null；旧 ready 版本缺配置时沿用历史 Anima 默认；坏配置返回 null，避免仅为显示族名称导致项目列表失败。

## 错误恢复与数据索引回退

[RouteError](../frontend/src/components/RouteError.tsx) 接到 Data Router 错误边界，对旧 chunk、动态模块加载失败及页面异常提供中英文反馈、重新载入按钮和可展开详情。它给出恢复入口，不假装失败的页面已经加载成功。

数据来源存在但模型等配置尚未完成、Plan 无法生成实际分桶时，[BucketInspector](../frontend/src/pages/TrainConfig/BucketInspector.tsx) 区分“已配置来源”和“没有图片”。只在匹配当前来源的索引均 ready 时显示已索引图片/标签数量；重复后样本、分桶和训练估算仍显示未知，并引导检查待配置项。索引数量不冒充已通过的训练计划。核心配置失败继续阻止编辑和启动，辅助任务等查询失败保留局部错误与重试。

## Krea 2 文本加载的内存修复

全量测试期间定位到 Krea 2 单文件文本加载在验证几何前先随机初始化完整 4B Qwen3-VL 文本模型的问题。缺少 `config.json` 的小型错误权重也会触发大分配，导致长时间没有测试进度；只读进程采样确认当时在 `torch.normal_`，并非网络重试。

[text.py](../ypuddin/models/krea2/text.py) 现在先在 meta 设备建立模型骨架，按原规则校验和装载 checkpoint；成功后仅在 CPU 重建未保存在权重中的 RoPE 频率缓冲区，检查无 meta 张量残留，再执行原精度/设备迁移。HF 目录加载路径不变。另修正官方布局测试缺少 meta 上下文的问题，避免仅检查布局时意外分配完整 12.9B DiT。

尺寸不兼容测试仍实际构造官方形状并检查错误，在构造前断言 meta，防止重新引入大分配。有效单文件权重在 FP32/BF16 下与原 eager 文本模型的输出逐位一致；原 FP32 HF 对照、FP8 权重和缩小版真实 Krea 2 管线继续通过。两份定向文件合计 **25 passed，8.42 秒**，日志为 `/private/tmp/ypuddin-v054-krea-text-loader.log`。meta 初始化消除无用随机权重副本，不代表正式模型加载不再需要容纳实际权重的内存。

## 本轮测试与构建结果

| 检查 | 本轮结果 | 范围与说明 |
|---|---|---|
| 前端全量 | **53 个文件、309 项通过** | 使用 `maxWorkers=2`；包含导航/草稿/归档、分类封面、采样控件、图表、RouteError 和索引回退。并发负载下的早前 timeout 不作为这次通过结果。 |
| 模型族文案补充回归 | **13 项通过** | 全量之后的针对性检查，单独记录，不与 309 相加冒充另一轮全量。 |
| 后端全量 | **609 收集，606 通过、3 CUDA 跳过，142.10 秒** | `tests/unit tests/e2e`；允许本地 loopback 后重跑，避免沙箱对本地 HTTP 测试的限制。XML 无 failure/error，跳过不算通过。 |
| Krea 2 定向 | **25 项通过，8.42 秒** | 官方几何、文本单文件/HF/FP8 和缩小架构完整管线；包含本节内存修复回归。 |
| 静态与构建检查 | **全仓 Ruff、前端 ESLint、TypeScript 与最终生产构建通过** | 最终构建 exit 0；最后一轮完整前端回归仍为 53 文件、309 项通过。图表缩放与窄屏标题改动已重新构建并用浏览器复验。 |

后端本轮日志与 JUnit 分别为 `/private/tmp/ypuddin-v054-backend-final.log`、`/private/tmp/ypuddin-v054-backend-final.xml`。这些临时路径是本机运行证据；交付时以随包验证记录为准，不要求接手者拥有原机器的 `/private/tmp`。

## 浏览器与数据证据矩阵

项目卡片与版本族场景在隔离服务 **8894** 执行，项目夹具为 `ui_v054_cards`；正式 **8876** 的升级与旧任务采样另有实际检查，二者分开记录。下表中的截图均已存在于本轮目录。

| 场景 | 环境 / 视口 | 实际操作与结果 | 证据 |
|---|---|---|---|
| 项目卡片、分类和取消编辑 | 8894，1004×773 | 三列封面卡，每卡约 248px；用手动文件 `portrait_0001.png` 创建项目；“人物 LoRA”筛选显示 1/3；编辑时移除封面再取消，原封面保留。 | [项目封面与分类](screenshots/v0.5.4/projects-covers.png) |
| 移动项目列表 | 8894，390×844 | 卡片收敛为单列，保留项目入口和操作。 | [移动项目列表](screenshots/v0.5.4/projects-mobile.png) |
| 移动创建弹窗 | 8894，390×844 | 创建弹窗完整显示封面、名称、ID、分类、模型族与提交区域。 | [移动创建弹窗](screenshots/v0.5.4/project-create-mobile.png) |
| 跨模型族空版本 | 8894，`ui_v054_cards` | v1 为 Anima，创建 v2 时选择 Krea 2 和 empty；创建成功，侧栏模型族随版本切换。更新后的截图展示从 Krea 2 来源选择 Anima 的动态说明，仅用于检查文案，不改写已实际执行的 Anima → Krea 2 创建方向。 | [版本模型类型](screenshots/v0.5.4/version-model-family.png) |
| 封面、分类与目录归属 | 隔离后端，只读核对 | 数据库记录 cover/category，v1/v2 配置分别保存于独立版本目录；核对各自模型族，不把侧栏显示作为唯一证据。 | 后端只读检查及 `test_project_presentation.py`、`test_version_family.py` 回归 |
| 标签查看与搜索 | 本轮桌面页面、600 图夹具 | 检查标签筛选、搜索输入和工具区；沿用既有完整标签查看与分页交互。 | [标签查看与搜索](screenshots/v0.5.4/captions-focused.png) |
| 采样 nullable 控件 | 本轮桌面参数页 | 检查字段排列与留空继承、显式数值输入；精确语义另由采样 schema 回归覆盖。 | [采样参数排列](screenshots/v0.5.4/sampling-layout.png) |
| 主损失图 | 本轮任务详情页 | 检查原始/显示 EMA 主图、平滑控件与分开的诊断区域；loss 不作为采样图片质量分数。 | [主损失图](screenshots/v0.5.4/loss-chart.png) |
| 深色模型页 | 本轮模型页面、深色主题 | 检查模型工具栏与设置表面底色的衔接。 | [模型页深色主题](screenshots/v0.5.4/models-dark.png) |
| 深色项目页 | 本轮项目页面、深色主题 | 最终配色下按钮文字与背景对比度正常。 | [项目页深色主题](screenshots/v0.5.4/projects-dark.png) |
| 正式旧任务采样损失 | 8876，`j_ce4c92d82604` | step 10 显示训练损失 1.6005，step 5 显示 1.3799；step 0 标明“初始采样 · 未训练”。 | [正式任务精确损失](screenshots/v0.5.4/samples-exact-loss.png) |

## 正式服务升级与旧数据保留

正式 8876 已按原数据根从 **0.5.3 重启为 0.5.4**。`/private/tmp/ypuddin-v054-main-restart-proof.json` 对升级前后快照的比较为 `preserved: true`、`differences: {}`：projects、project_versions、datasets、jobs、artifacts、models 六张业务表保持原有逻辑内容，**192 个业务文件**字节哈希不变，凭据配置状态不变。新加可空分类/封面字段按 null 归一后比较；运行时缓存与 SQLite WAL 不在业务文件比较范围。没有把新增字段本身误报为旧数据丢失，也没有读取或披露凭据明文。

旧任务采样损失已在正式浏览器核对，具体 step 与显示值见上表。这证明该旧任务的兼容读取与展示，不表示重新执行了完整正式模型训练。

图表最终复验通过：移除 ECharts inside zoom 后，在图内使用普通滚轮，页面滚动位置从 0 变为 153 px；范围缩放保留底部滑块。390 × 844 窄屏下，任务标题随页面滚动，避免遮住诊断图；实际滚动到 1208 px 时标题已经移出视口，页面宽度仍为 390 px。见 [窄屏诊断](screenshots/v0.5.4/diagnostics-mobile.png)。

正式版本结果页及放大预览也核对了第 5 步 loss 1.3799，与任务详情一致，见 [版本采样](screenshots/v0.5.4/version-samples-loss.png)。队列在隔离服务使用 251 个任务夹具、70 个保持暂停的等待任务验证，训练历史第 2 / 10 页共 181 条，见 [队列分页](screenshots/v0.5.4/queue-history.png)。本轮没有触发这些等待任务。

## 验证边界

本轮未做 Windows/NVIDIA 实机训练、全新 Windows 环境安装、完整 Anima/Krea 2 权重生产训练、ComfyUI 实际加载与质量/性能验收。Toy、缩小架构测试和 QA 生成图片/任务/权重夹具用于检验实现与交互，不代表正式训练质量。官方模型族已接入不等于上述生产验证已完成。

没有执行用户取消的 `npm audit`。本报告不宣布全部功能完成，不改变历史版本报告；最终源码包、构建清单、提交和解包验证以本轮发布证据为准。
