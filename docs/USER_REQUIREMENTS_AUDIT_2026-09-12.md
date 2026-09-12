# 用户明确要求逐项核对（2026-09-12）

> 下表保留整改前 v0.5.2 的审计结论；v0.5.3、v0.5.4 历史验收与当前 v0.5.5 补充在文末。“未闭环”一栏中的界面问题应结合文末读取，Windows/NVIDIA 验证限制仍然适用。

本次依据用户要求整理文件 `/tmp/ypuddin-user-requirements.txt` 的 15 条逐项审计。核对基线是本地提交 `003b443585913e74de8f261a847f32e0adb51234`（v0.5.2）和本轮工作区；审计时已有新的数据准备/正则样式修改，项目顶部布局正在重新设计。本文只读核查源码、交接文档及已有验证记录，没有联网、运行 GPU、安装依赖、操作浏览器或执行 `npm audit`。

**结论：主要功能后端已有真实链路，但不能判定 15 条全部完成。** 项目顶部多层导航是用户本轮明确指出的未解决体验问题，相关统一布局和大数据集可达性应按新页面重新验收。Windows NVIDIA、完整 Anima/Krea 权重和新环境安装没有生产级实测。其余功能检查未发现可以确认的新后端缺项；“未发现”不等于对全部输入与硬件作出保证。

表中的“实现证据”说明代码实际做了什么；“验证边界”说明哪些结论有运行证据；“确定遗漏/未闭环”只列明确未完成事项，不把推测性优化当作缺陷。

## 15 条要求核对

| # | 用户明确要求 | 当前实现证据 | 真实验证边界 | 确定遗漏 / 未闭环 |
|---|---|---|---|---|
| 1 | 实际参考 AnimaLoraStudio 0.27.0 的前端与完整训练流程，不能只看片段，也不照搬局限 | 参考 HEAD 为 `3d9d2e86045b879cd19c01ac4aa2337f45a283ab`；已在隔离副本运行真实 8890/8891 服务。既有训练/Mask/环境浏览记录见 [v0.3](UI_REDESIGN_2026-09-11.md)、[v0.4](UI_VERSIONS_2026-09-11.md)；本轮重新恢复项目 API 和已有示例项目，供实际对比。实现仍使用本项目的 Trainer、版本服务及 React 页面。 | 本轮此前实看参考设置/模型准备；当前项目顶部的重新对比由界面验收执行。没有运行参考的完整正式模型训练，不能宣称比较过其生产训练质量/性能。 | 本轮新的项目工作区设计对照尚未以最终页面闭环；不能仅凭旧参考截图称已满足最新投诉。 |
| 2 | 完整可用的项目→数据导入→参数→启动→监控→结果训练器 | [routes_work.py](../ypuddin/server/routes_work.py) 提供项目/数据集/任务/采样/日志/产物；[supervisor.py](../ypuddin/server/supervisor.py) 启动真实训练子进程；[trainer.py](../ypuddin/train/trainer.py) 执行训练。前端 ProjectDetail、TrainConfig、JobDetail、VersionResults 连接这些 API。 | 已有 Toy 和缩小版真实架构回归及历史 Toy/MPS 浏览器训练、下载记录；0.5.2 的大列表验收使用隔离夹具，并非正式模型训练。 | 核心链路未发现新的确定漏接；Windows/NVIDIA 完整模型端到端仍未验收。 |
| 3 | 紧凑一致的下拉、按钮、弹窗、就近错误、尺寸变化与高级参数间距 | [StudioSelect.tsx](../frontend/src/components/StudioSelect.tsx)、[Dialog.tsx](../frontend/src/components/Dialog.tsx) 和独立 dialog.css；SchemaForm 分组布局；AccessKeys 逐来源反馈；配置/辅助请求失败隔离均已实现。 | 0.5.2 有 1004/390/1280 视口、明暗主题的有限页面记录；不能推出所有页面、所有尺寸一致。 | 当前项目顶部仍被用户明确否定，不能把通用组件通过测试等同于全页布局通过。由本轮视觉整改验收。 |
| 4 | 顶栏 CPU→内存→GPU→硬盘，GPU 四读数与图标对齐，正常不显示连接提示 | [SystemTelemetry.tsx](../frontend/src/components/SystemTelemetry.tsx) 按该顺序渲染，GPU 占用/内存/功率/温度独立；[Layout.tsx](../frontend/src/components/Layout.tsx) 仅异常时显示连接反馈；[hardware.py](../ypuddin/server/hardware.py) 区分 MPS 统一内存与 CUDA，并提供 NVML/nvidia-smi 回退。 | Apple 本机显示统一内存，不伪造独立显存或功率。NVIDIA 数据采集有模拟/单元测试，Windows 实机读数尚无证据。 | 数据契约未发现确定遗漏；真实 Windows 功率等读数及最终图标排列仍需对应环境/视图验收。 |
| 5 | 多语言显示名、手填 ASCII ID、`project/<id>/vN/{traindata,reg,samples/<jobid>,output/<jobid>}`、独立任务 ID、自定义输出、旧数据不破坏 | [routes_work.py](../ypuddin/server/routes_work.py) 的 ProjectBody/create_project 分离 name/id，校验 ASCII、Windows 设备名和大小写冲突；[context.py](../ypuddin/server/context.py) 的 project_dir/version_dir/dataset_dir/samples_dir/job_output_dir/runs_dir 构造实际目录。create_job 固定新的 `j_*` 与版本、采样/输出路径。 | `test_project_layout.py` 覆盖默认/自定义根、连续版本、旧路径可读与失败保留。自定义输出根追加 `<pid>/vN/<jid>`；采样仍按版本 `samples/<jid>` 单独保存。旧 layout=1 和旧任务保持其记录路径。 | 未发现确定功能遗漏；不能把路径回归当作已在 Windows 所有文件系统与磁盘场景验收。 |
| 6 | 数据/版本隔离、分桶可视预览、更完善原生分辨率训练 | [versions.py](../ypuddin/server/versions.py) 用 copy2 复制图片、caption、Mask 和显式 validation 来源，不建硬链接；复制/导入持版本锁。[data/native.py](../ypuddin/data/native.py) 保留各图尺寸、对齐裁剪、像素预算和异尺寸逻辑批次；[trainer.py](../ypuddin/train/trainer.py) 的 `_run_native_epoch` 按实际图片数累积/归一化梯度；Plan 与数据布局一致。 | 原生几何、EXIF、Mask 同变换、尾批梯度和恢复有回归；历史 Toy/MPS 原生训练有实测。原生不是 NaViT 序列打包，预算只限制张量几何，不保证不会 OOM。高级 TOML 外部引用不会自动复制；版本数据也不是每任务不可变图片快照。 | 受管理导入/复制与原生训练未发现新的确定缺项。外部 TOML 和可编辑版本是明确限制，不能声称任意路径都自动物理隔离。 |
| 7 | 完整数据处理，不要自动打标，保留已有标签查看与手动标签/遮罩编辑 | [dataset_pipeline.py](../ypuddin/server/dataset_pipeline.py) 提供检查、排除、恢复、裁剪/缩放、caption 操作及备份；公开 action 不再接受 tag。CaptionViewer 读取完整既有 caption；[routes_work.py](../ypuddin/server/routes_work.py) 保留逐图标签写入；[routes_dataset_masks.py](../ypuddin/server/routes_dataset_masks.py) 保留 revision 校验、灰度 PNG 原子保存与训练写锁。 | 处理/撤销、备份损坏拒绝、caption 扩展名、Mask 尺寸与冲突有真实文件回归；历史浏览器画 Mask、保存重开已验证。Mask 编辑限制为 16,777,216 像素/8192 单边，白色参与学习、黑色忽略；修改后需启用 masked_loss。 | 未发现自动打标重新暴露或手动标签/Mask 被删除。当前标签页面滚动设计由本轮界面复核，不能只凭 600 图夹具概括所有数据。 |
| 8 | 自动正则 AI 生成和网络收集，凭据集中保存共用 | [regularization.py](../ypuddin/server/regularization.py) 使用版本模型和用户类别提示词，独立 worker 无训练适配器；站点固定 Danbooru/Gelbooru，保存标签、去重和暂存批次，完成才注册 is_reg。[model_credentials.py](../ypuddin/server/model_credentials.py)、[routes_credentials.py](../ypuddin/server/routes_credentials.py) 保存四来源，正则默认读取同一 store；前端不再每次输入站点密钥。 | 真 Toy 子进程生成/取消和发布到训练源已验证；两站接口、错误、认证头、重定向使用受控响应测试。正则默认不继承主体 trigger、不参与自动 val split；损失按混合样本权重平均，不是两组独立均值相加。 | 功能链路未发现新的确定漏接；正式模型 AI 出图、真实账号权限及网站当下响应未验收。 |
| 9 | 环境仅核心 Py/Torch/CUDA 与 attention，自动适配和补必需依赖，不让用户逐项装 W&B/打标/优化器 | [EnvironmentManagerPanel.tsx](../frontend/src/components/EnvironmentManagerPanel.tsx) 只管理 xformers/flash-attn，显示 Python/Torch/计算后端；旧 Sage 仅标采样。[bootstrap.py](../scripts/bootstrap.py) 安装默认 models/server/optim/logging，NVIDIA 追加监控，并以签名+真实缺包检查补依赖、保护既有 Torch/CUDA/NumPy。W&B/onnxruntime 安装请求已撤下。 | bootstrap 回归覆盖缺包修复、首次 CPU 构建与原生依赖保护；旧环境中的 Torch 构建不会因驱动变化自动换成另一构建，坏/过旧环境需显式重建。已修改扩展需重启，队列有维护门禁。 | 自动补常规依赖已实现；本轮没有 Windows/Linux 全新环境实际安装、CUDA wheel 安装验收，不能宣称已生产适配所有机器。 |
| 10 | HF/魔搭下载、路径、令牌、组件版本选择与完整失败/进度交互 | [model_recommendations.py](../ypuddin/server/model_recommendations.py) 明确 8 个组件候选与双平台仓库/文件/版本、大小/SHA；[model_downloads.py](../ypuddin/server/model_downloads.py) 下载、取消、失败、从头重试、验证后注册；本地候选 `/use` 验 SHA，共享 VAE 可复用；Models 分准备/本地/下载记录并支持自定义源与目录登记。 | 本地 HTTP、小型真实 safetensors 覆盖来源映射、并发、损坏拒绝、保留重试校验和凭据保护。官方元数据已在此前核对；此次审计未联网。任意自定义文件不自动获得官方推荐身份。 | 未发现当前明确要求的下载功能缺漏；**没有断点续传**，UI 已说明重试从头开始；也未实下载/训练完整底模，不能宣称大模型全链路通过。 |
| 11 | 全局队列与项目结果职责明确，采样隔离，前端日志可看 | [routes_work.py](../ypuddin/server/routes_work.py) `list_jobs` 支持 group/type/q/版本和 SQL 分页；日志有 tail/offset/has_more 且限制读取字节；job_samples/job_file 与 artifacts 查询按任务/版本。Queue 是跨项目任务，VersionResults 是指定版本结果，JobDetail 查看真实日志/采样。 | 已有队列第二页筛选→任务日志→返回原 URL、48 样本、1400 UTF-8 日志与正确版本权重夹具记录。96 字节权重是界面夹具，不是训练成果。 | 未发现新的确定 API 漏接；模型下载/正则使用独立任务服务，并非全都混入训练队列。 |
| 12 | 大数据集下一步/分页/启动始终可达，统一考虑全部页面 | ProjectWorkflow、工作区高度测量、训练固定操作区、CaptionViewer 顶部分页和独立滚动、Queue 顶部分页均已有；数据集/产物等页面也有对应布局实现。 | 600 图夹具及长参数滚动只验证过部分路径与尺寸。“始终/全部页面”不能由有限测试证明，尤其当前用户再次指出顶部占高。 | 全页可达性仍需本轮导航整改后统一实看；不能沿用旧截图直接判通过。 |
| 13 | 项目列表紧凑，顶部不应六行导航占约 266px | Projects 已有紧凑样式，ProjectWorkspaceHeader/ProjectWorkflow/数据步骤仍是当前工作区结构；根任务正在针对最新投诉重做。 | 用户最新反馈本身说明既有发布页未满足预期。本次只读审计不操作浏览器，也不把此前“通过”的视觉结论覆盖用户反馈。 | **确定未闭环：顶部导航层数与占高。** 新页面完成后需实际对照参考并验证项目列表/各步骤，不可仅提交 CSS 宣称解决。 |
| 14 | 侧栏不放版本号，品牌/主题/语言合理，训练参数命名明确 | Layout 侧栏不读取服务版本；[ServiceInfo.tsx](../frontend/src/pages/Settings/ServiceInfo.tsx) 在设置“界面与服务”读取真实 `/system/info`。底栏有 system/light/dark 和当前语言；导航已用“训练参数”等用途名称。 | 位置与保存交互已有回归；品牌、命名是否足够清晰属于本轮界面统一验收，不能由代码断言主观满意度。 | 版本号迁移未发现遗漏；其余视觉/命名跟随本轮页面核对。 |
| 15 | 更新本地 Git、交付可用构建、禁止 npm audit，目标 Windows NVIDIA，不冒充生产完成 | 本地提交 `003b443` 与 v0.5.2 ZIP 已有；[验证记录](validation/v0.5.2.json) 记录 562 后端通过/3 CUDA 跳过、263 前端测试/45 文件、lint/类型/生产 build。包旁 verification JSON 记录独立解包导入、SPA/资源、manifest 与运行数据排除。 | 解包使用已有 Python 依赖，未做全新安装；Windows NVIDIA、完整模型/ComfyUI 与性能均未验收。此后工作区 CSS 整改不在该旧 ZIP 中；没有执行 npm audit。 | **确定未闭环：本轮整改的最终构建/交付，以及目标 Windows NVIDIA 生产验收。** 旧提交/旧包可追溯，但不能代替新整改包或生产成功证据。 |

## 当前目录和数据责任复核

新建项目默认路径是实际后端拼接结果，非仅页面上的目录示意：

```text
<data_root>/
  studio.db
  datasets/<dataset_id>.json       # 全局索引记录
  cache/index.sqlite              # 索引内容指纹缓存
  secrets.json                    # 本机凭据，源码/交付包排除
  project/<project_id>/vN/
    config.json
    traindata/<import_batch>/
    reg/<regularization_batch>/
    samples/<job_id>/
    output/<job_id>/
    cache/
```

自定义 output_mode 使用 `<configured_output>/<project_id>/vN/<job_id>`；项目任务采样继续位于版本 samples 目录。自定义缓存按项目 ID 和版本 API ID 分隔，不是按显示名称。旧项目 layout=1、旧运行目录与已保存 samples_dir 均保留；新任务不会因为 active_version_id 后来改变而归到新版本。手动登记高级外部路径仍可能引用用户外部目录，不能把“版本隔离”解释成系统会复制一切外部 TOML 引用。

## HANDOVER 与证据使用

[HANDOVER.md](../HANDOVER.md) 对中央凭据、模型校验、正则、原生梯度、旧数据保护与未测 Windows/NVIDIA 的说明基本与当前源码一致。其 562/3、263/45 数量与 v0.5.2 JSON 相符，但它记录的是已发布基线；“导航/分页固定可达”只能理解为该轮所测页面，不应作为驳回本轮顶部布局投诉的依据。

已存在的 `YPuddinTrainStudio-v0.5.2-2026-09-12.verification.json` 绑定 `003b443`，记录 ZIP 486 项、13,513,276 字节、SHA-256 `0722be04649df12f6b17666ca310867a0bd3f32e6f7f66ec2b78718dc8ebf9ef`。本次只读取该历史交付证明，没有重新打包或复跑其校验；后续整改需生成对应新证据。

本轮没有确认到新的必须先修才能连通的后端功能缺项，因而没有修改实现。明确尚未闭环的要求是 **#13 导航层次、关联的 #3/#12 全页布局验收、#15 新整改交付及 Windows/NVIDIA 正式模型验证**。参考项目完整正式模型训练、真实站点账号和大模型下载等未执行部分也已逐项标明，不能用单测、模拟响应或本机局部成功替代。

## v0.5.3 整改后验收补充

2026-09-12 的最终实现已将项目身份、版本选择与四个主要阶段放入侧栏，正文只保留标题及一行数据子步骤。项目列表实测单行 48px，数据导入内容从 y=163 开始，不再占用原来的六排导航。上传按钮白字消失、嵌套路径窗口 CSS 污染、移动侧栏被固定标题遮挡、高级选项分组间距、队列筛选换行及结果页重复工具栏均已处理。

针对 #3/#12/#13，正式构建在 1004×773 / 390×844、深浅主题下验证了项目列表、数据导入、标签查看、正则表单、路径选择器、高级参数、分桶、设置、队列和结果浏览。600 张图片独立滚动后，下一步仍在 y=114–142；队列第二页返回上下文、48 采样两页和 1400 行中文日志分页通过。此结论覆盖记录中的操作，不宣称所有可能页面状态均已穷尽。

复核另外发现并修复了浏览器后退丢草稿、设置跳转丢背景，以及搜索下拉可以提交隐藏选项的问题。标签文件实际不可写时后退会保留草稿，恢复权限重试后才离开；遮罩清空后后退仍保留画布和撤销。正式 Data Router 的双向历史与设置背景另有回归。

最终 48 个前端测试文件、282 项通过；TypeScript、ESLint、生产构建通过。8876 按原数据根重启为 0.5.3，六张业务表内容及凭据配置状态保持，39 个静态资源 SHA 一致。完整数值、截图和服务证据见 [v0.5.3 验收记录](validation/v0.5.3.json)，布局说明见 [整改报告](UI_WORKSPACE_V053_2026-09-12.md)。新归档的精确 Git 提交、SHA 和独立解包验证以包旁 `YPuddinTrainStudio-v0.5.3-2026-09-12.verification.json` 为准。

#15 中的 Windows/NVIDIA 正式模型训练、新机器安装以及真实账号大模型下载/正则站点部分依旧未做实机验收；未将这次布局修复标记为全部生产验证完成。没有执行 npm audit。

## v0.5.4 历史补充

本节保留 v0.5.4 实施时的逐项记录，不改写上述 v0.5.2 审计和 v0.5.3 历史验收。表内待验收项应结合已补齐的 [v0.5.4 工作区变更报告](UI_WORKSPACE_V054_2026-09-12.md) 阅读；其中浏览器、测试和升级结果属于 v0.5.4，不能作为 v0.5.5 的新结果。

| 对应要求 / 新反馈 | 本轮实现 | 仍需区分的边界 |
|---|---|---|
| #2/#12/#13：项目流程清楚，避免独立模型页与参数重复 | 三个主阶段为训练数据、训练参数、训练结果；旧模型准备入口转到所属版本 `/train?tab=model`，模型组件和训练参数共用草稿保存；全局队列与版本结果职责不变。 | 合并入口后的导航、保存、归档和分区切换已有定向回归；各视口真实操作结果按本轮矩阵补齐。 |
| #3/#12：标题、数据子步骤、标签搜索与参数区重复边框/底色，滚动透出旧字段 | 标题使用页面底色且无重复框；数据导航保留单底线；标签搜索图标内置；吸顶补齐顶部 padding 遮盖，模型抽屉使用对应底色。 | 已保存阶段截图，不以代码或旧版截图替代最终深浅主题、窄屏与滚动验收。 |
| 新反馈：采样字段难用，Loss EMA、学习率 w1/w2 与梯度含义不清 | nullable 采样值保留继承语义，显式值采用族默认/合法下限；每提示词覆盖可编辑；主损失与诊断图分离，显示 EMA 不影响训练；w1/w2 说明为 LoKr 参数组。 | 图表平滑不等于改善训练，未知读数不填 0。 |
| 新反馈：采样图没有 loss | 每步真实训练损失随采样 REST/SSE 展示；step 0 标明初始采样，旧记录缺精确同 step 值时显示未记录，不借最近点或其他恢复段。 | 该值是训练步损失，不是单张图片质量评分；夹具图片不能证明生产质量。 |
| #5/#13 与新增项目管理要求：分类、封面、模型族 | 项目卡片与编辑弹窗支持独立分类及手动封面；限额和重新编码后的封面保存在项目 `.studio`，不进入训练源；封面失败重试复用已创建项目。分类筛选、归档和分页可组合。 | 旧数据库仅加可空元数据，旧目录不迁移；实际浏览器上传/重试/归档流程的最终证据待统一填入。 |
| 新增版本模型类型要求 | 新项目/版本接受 Anima/Krea 2/Toy；同族保留配置，跨族从通用默认配置重建并保留 dataset/validation，按目标族调整文本/采样/目标函数；数据 copy/empty 独立，旧版本和任务不改写。 | Flux/SDXL 未接入并禁选；不自动启用 FP8/CUDA或下载底模，失效默认权重留空。 |
| #2/#3：加载错误与“有数据却显示空数据”的误导 | RouteError 提供资源或页面异常的就近恢复说明；Plan 尚未就绪时使用匹配来源的 ready 索引数量，并明确分桶/估算未知和待配置入口。 | 索引成功不等于训练配置验证成功；核心配置读取失败仍阻止可写表单和启动。 |

#15 的 Windows/NVIDIA 真机、完整模型训练/质量和新环境安装仍未实测；本节不将全部要求标为完成，不复用历史测试总数。未执行 `npm audit`，未把生成的 QA 数据集、采样图或小型权重夹具当成正式模型训练成果。


## v0.5.5 参数、预设与 JSON 标签补充

本轮针对用户新增要求，落实独立参数管理、帮助浮层、结构化标签和实际采样选择。以下为当前源码及本轮证据，不改变旧表的历史审计基线。完整说明见 [v0.5.5 参数变更报告](UI_PARAMETERS_V055_2026-09-12.md)。

| 对应要求 / 新反馈 | 本轮实施 | 验证与限制 |
|---|---|---|
| #3：问号浮层越过侧栏边界，空白处无法关闭 | ConfigHelp 使用 body 门户和边界避让，外部点击、Escape、焦点移出、外层滚动、路由离开关闭，多个提示互斥；说明内部可阅读和滚动 | 交互/模拟几何回归与 1004/390 真实截图均有记录；不以单测代替所有视口验收 |
| 新反馈：预设只能在项目里编写/加载，缺独立管理 | `/presets` 支持按模型分组、搜索、实际参数表单创建/编辑/复制/删除；内置只读，但保留帮助和分组交互 | 后端校验和原子写入，错误保留输入；浏览器完成保存与加载前预览，内置不允许覆盖 |
| #5/#12：参数复用不能覆盖版本数据与文件 | 加载先看差异，再显式应用；跨族禁用。保存/加载均过滤模型文件、训练/验证来源、缓存、采样文件、输出、日志和恢复路径，保持当前模型族 | 同族取消/应用、跨族禁用、路径保留有回归；不将预设加载成功当作训练 Plan 已通过 |
| #7：已有 JSON 标签应能查看、编辑和训练 | 新来源 auto 优先 JSON，其次 TXT；支持明确结构化格式，索引/编辑/缓存/训练/版本复制使用同一解析。逐图编辑保留 `nl`、`meta` 等内容 | 真实 JSON 查看、编辑、保存及文件保留核对；坏 JSON 不回退。旧 TXT 配置不自动改写；逐图保存不自动生成流水线撤销记录 |
| 新反馈：优化器输入不直观、参数缺解释 | 显示真实注册优化器与自定义类入口，说明设备/依赖限制、精度、训练噪声与学习率调度差异 | 8-bit 依赖 CUDA/bitsandbytes；缺失报错，不自动降级；没有声称所有算法已在 NVIDIA 通过 |
| 新反馈：LoKr Full、低秩与 Alpha 含义不清 | 明确 Full/低秩模式，Full 时隐藏不生效的 Alpha 且保留原值；区分 LoKr 因子、完整层训练、目标层预设 | 模式切换有实际控件回归；未把完整因子写成完整底模训练 |
| 新反馈：采样没有 ER-SDE 或独立调度器 | 实际接入 Euler/Heun/ER-SDE 和 uniform/simple/sgm_uniform/normal，ER 最高阶数及噪声强度可配，旧默认保留 | 数值/集成回归包含独立方程对照与 RNG 隔离；没有实现 DPM++ 3M SDE，也不声称官方模型画质或速度优于参考 |
| 数据来源参数容易误用 | 新来源重复次数默认 1，目录前缀仅给可应用建议；正则参数命名“正则损失权重”，与重复次数分开 | auto 格式、正则权重和版本来源请求有回归；权重 0 的图仍占批次，不等同于删除 |

本轮全量为 **前端 57 文件 / 331 项通过；后端 771 通过 / 3 CUDA 跳过（774 项，134.143 秒）**。Ruff、ESLint、TypeScript 与生产构建通过。截图位于 `docs/screenshots/v0.5.5/`，明细见本轮报告；正式 8876 服务已按原数据根升级到 0.5.5，192 个业务文件、六张业务表、settings 和凭据配置状态保持，128 个源码输入/50 个构建输出哈希匹配。服务证明为 `/private/tmp/ypuddin-v055-service-proof.json`。

#15 中的 Windows/NVIDIA 实机、完整权重效果与性能、新机器安装等仍无本轮验收。Flux/SDXL 未接入且禁选；QA 图片和小型权重夹具不是生产训练质量证明，不将本节写成全部用户要求或整个产品已完成。
