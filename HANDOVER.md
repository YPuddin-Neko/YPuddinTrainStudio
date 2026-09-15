# YPuddin Train Studio — 项目交接报告

> 写给接手本项目的模型/工程师。本文自洽：读完这一份 + 点开的几个文件，就能不需要前任任何上下文地继续开发。
> 当前验收更新：2026-09-13 · 仓库：`xiangmuyuanma/` · 源码仍为 0.5.9。最新参数布局、导入进度与独立标签页见 [UI 工作区验收](docs/UI_WORKFLOWS_2026-09-13.md)：前端最终 494 项通过，真实浏览器验证四种窗口尺寸、目录/ZIP 导入、TXT/JSON 编辑，分段 HTTP 验证真实进度。此前后端 1085 通过 / 3 CUDA 跳过、前端 456 通过及 MPS 训练记录见 [实际流程验收报告](docs/FULL_ACCEPTANCE_2026-09-13.md)。两轮均使用隔离 8877 QA 服务；旧 8876 记录的数据根已不存在，不能把历史升级记录当作当前正式服务状态。

## 2026-09-15 当前查证入口

优先阅读 [当前交付报告](docs/DELIVERY_REPORT_2026-09-15.md) 和 [大模型分片训练指南](docs/FSDP_TRAINING_2026-09-15.md)。当前已实现 DDP 数据并行及 FSDP2 主模型全量微调：DDP 在各卡保留完整训练副本，FSDP2 分配参数、梯度和优化器状态；组合限制与实测范围分别以对应指南为准。

Windows 追加预检与后续真机检查入口见 [Windows 验收说明](docs/WINDOWS_READINESS_2026-09-15.md)。它记录本地模拟与真实宿主执行的区别，不改变原生 Windows 单任务多卡的限制。

海光双卡正式 Krea 2 的 r2 快照已完成容量验证、连续 8 步、从第 4 步冷恢复到第 8 步，以及两份完整模型的原生重载；**执行通过，最终权重逐位一致性未通过**。后续源码快照不能直接继承 r2 的实机结论，最新状态以以上报告为准。下面 2026-09-14 及更早章节保留历史验收正文，其中“未实现 DDP”“正式 GPU 尚未验收”等描述不代表当前能力。

## 2026-09-14 当前查证入口

优先阅读 [训练流程、分桶及三方对比](docs/TRAINING_PARAMETERS.md) 和 [Windows 正式权重验收](docs/WINDOWS_ACCEPTANCE_2026-09-14.md)。前者按固定源码快照区分默认/旧配置、Flow/SDXL、缓存后卸载/移 CPU、XY/XYZ 与 LoKr 运算后端；后者保留正式短训范围、数值失败及未测项目。下面“正式 GPU 尚未验收”等说法属于对应历史轮次，不能覆盖 9 月 14 日已有证据；DDP、Klein 9B、长训和全参数组合仍不能写成已通过。

## 2026-09-13 模型后端新增

优先阅读 [SDXL / Klein 接入说明](docs/MODEL_FAMILIES_2026-09-13.md)。当前公开 registry 支持 SDXL 与 FLUX.2 Klein；下面历史“Flux、SDXL 尚未接入”的记录已被本节取代。SDXL 默认光辉 v0.1，保留第三方本地模型与 ε/v 选择；FLUX 仅保留 Klein base 4B/9B；旧 FLUX.1 和 FLUX.2 dev 配置可读，执行明确拒绝，不改写已有数据。

核心新增 `ModelFamily.build_objective/sample_latents/materialize_backbone/sampling_defaults/sampling_shift_for_model`；默认 RF 路径兼容既有模型，SDXL 使用 DDPM，FLUX 主干在编码缓存后物化。FamilyInfo 提供权重角色、可选组件、可下载类型和实际支持的参数选项。配置、版本、预设与正则生成均已接入。新模型小型真实网络测试和外部数值对照不代表完整 GPU 验收；本轮未修改 Windows 服务或下载大模型。

## 0. 先读这三个文件

2026-09-13 最新入口：先读 `docs/UI_WORKFLOWS_2026-09-13.md`。参数页最大宽度及容器响应式、左下角折叠、主题滚动条、底部操作记录已调整；标签工作区与遮罩页面分开，JSON 自然语言与未知字段保留，同哈希图片编辑使用相对路径。导入进度来自独立后端跟踪器；训练缓存页是可选提前生成，正常训练仍自动准备缓存。本轮未追加官方权重 GPU 训练或 Windows 实测。

2026-09-13 当前入口：先看 `docs/FULL_ACCEPTANCE_2026-09-13.md`。本轮修复目录/多概念 ZIP 同步、服务器目录导入、版本复制保目录及嵌套对话框键盘交互，实际完成 Toy/MPS 80 步暂停恢复和浏览器启动的 10 步训练。官方全尺寸权重、CUDA、Windows 和操作系统文件管理器目录拖放尚未实际验证；DDP 尚未实现。下面 2026-09-12 的版本、服务和测试计数保留为历史证据，不覆盖本轮结论。

2026-09-12 仓库内容补修：先看 `docs/REPOSITORY_CONTENT_2026-09-12.md`。Git 忽略补齐本机数据/凭据/数据库/归档及大小写变体；`python scripts/package_source.py --check-git` 检查索引，打包前会自动阻断误跟踪的本机文件。运行程序仍为 v0.5.9；本次内容规则验收单独记录，不覆盖该版本的 UI 和服务验收。

1. 本文。
   本轮补丁先读 `docs/ADAPTER_FORM_LAYOUT_2026-09-12.md` 与 `docs/validation/v0.5.9.json`。`docs/MAC_GPU_SENSORS_2026-09-12.md` 和 `docs/MAC_GPU_TELEMETRY_2026-09-12.md` 保留功率、温度及利用率采集记录，下面的 v0.5.6 报告保留完整架构与工作流说明。
2. `docs/UI_REVIEW_V056_2026-09-12.md` —— 本轮来源用途自动识别、整图保留、图像绘制、输出绑定、模型检测与运行环境说明；`docs/TRAINING_PARAMETERS.md`、`docs/JSON_CAPTIONS.md` 给出实际参数与格式契约。`docs/UI_PARAMETERS_V055_2026-09-12.md` 保留独立预设、加载保护、Help 浮层、JSON 标签、优化器/LoKr 与 ER-SDE 的历史实现和验证。`docs/UI_WORKSPACE_V054_2026-09-12.md` 保留三阶段导航、边框吸顶、采样参数/损失、项目分类封面、版本模型族与错误恢复的历史验收。`docs/UI_WORKSPACE_V053_2026-09-12.md` 保留 v0.5.3 侧栏、导入、结果及移动叠层证据；`docs/USER_REQUIREMENTS_AUDIT_2026-09-12.md` 逐项对应历史要求并追加本轮变化。`docs/UI_DESIGN_REVIEW_2026-09-12.md` 保留 v0.5.2 中央访问密钥、官方模型候选、队列/版本结果、导航标签与请求隔离。`docs/UI_SIMPLIFICATION_2026-09-12.md` 保留 v0.5.1 项目目录、正则图及自动打标移除记录；`docs/native-resolution.md` 解释原生尺寸与梯度规则。UI_PIPELINE、UI_VERSIONS、UI_WORKFLOW、UI_REDESIGN、FIX_REPORT 与 COMPLETION_AUDIT 是历史证据。
3. `docs/design/03-status.md` —— 逐组件状态表与运行方式。

## 1. 项目定位

一个桌面化的 diffusion **LoRA / LoKr 训练器**。训练循环、适配器、数据流水线、服务 API 与 CLI 独立实现，模型组件包含按 Apache-2.0 引入的上游代码（见各 `vendor/NOTICE.md`）。`docs/reference/*.md` 保留早期参考审计，不能据此推断参考项目当前版本的状态或本项目性能更好。

- **模型族**：`anima`（Anima 2B，Cosmos-Predict2 DiT + Qwen3-0.6B + Qwen-Image VAE）、`krea2`（Krea 2 Raw 12.9B 单流 MMDiT + Qwen3-VL-4B + 同款 VAE）、`sdxl` 和 `flux2`（Klein）；`flux` 仅供历史格式识别；另有 `toy` 族供 CPU 测试。新族 = 实现 `ModelFamily` 协议（`ypuddin/models/`）。
- **适配器**：LoKr（自研，与 LyCORIS 文件格式兼容）、LoRA、LoHa、Full、DoRA 包装。目标选择用 preset + 有序 rules（`ypuddin/adapters/rules.py`）。
- **形态**：Python 包 `ypuddin`（CLI + FastAPI 服务）+ `frontend/`（React/Vite 界面，可选）。一键脚本 `studio.sh` / `studio.bat`。
- 许可证 Apache-2.0（参考项目里 diffusion-pipe 与 AnimaLoraStudio 是 GPL——只读不抄；sd-scripts / musubi-tuner 是 Apache-2.0，vendor 的代码见 §7）。

## 2. 2026-09-12 状态记录（v0.5.9 共享适配器表单）

训练参数与预设编辑共用的适配器表单按算法/目标层、LoKr 形式/分解因子、Rank/Alpha 排列，窄容器切为单列，控件统一为 34px。Full 隐藏不适用的 Rank/Alpha 输入，切回低秩或 LoRA/LoHa 恢复有效数字 Rank；模式切换不覆盖其余适配器参数。后端功能与配置契约不变，功率估算、温度均值/最高值/传感器数量及未知值语义保持。

本轮 40 项前端回归、TypeScript、ESLint 和生产构建通过，未重跑后端测试。正式 8876 已升级到 0.5.9：新鲜快照中的 194 个业务文件、六张业务表、设置与凭据状态完整保留，137 个源码输入及 53 个构建输出哈希匹配，实际传感器 API 契约通过。完整交互与浏览器证据见[适配器表单说明](docs/ADAPTER_FORM_LAYOUT_2026-09-12.md)和[验证记录](docs/validation/v0.5.9.json)；本机升级证明为 `/private/tmp/ypuddin-v059-service-proof.json`，包校验以发布包旁 verification JSON 为准。

### 上轮 v0.5.8 Apple GPU 功率与温度

本轮将 `power_w` 接到 IOReport 的 GPU 功率估算，将 `temp_c` 接到 SMC 可用 GPU 区域传感器的均温；最高温度与有效传感器数量供解释读数。功率不是插座电表实测，均温不是单一热点温度，二者也不是训练进程专属指标。采集与界面保留未知值及来源说明，不能由 GPU 占用或统一内存比例推算。最终实现、实机边界与验证见本轮传感器说明和验证记录；发布包校验以相邻 verification JSON 为准。

正式 8876 已使用原数据根升级到 0.5.8。本次新鲜快照的 194 个业务文件、六张业务表、设置与凭据状态完整保留，137 个源码输入及 53 个构建输出哈希匹配。正式 API 验收时读到 IOReport 估算功率约 0.96 W（采样约 0.261 秒），SMC 均温约 80.94°C、最高 83.85°C、8 个有效 GPU 区域传感器，来源与元数据均匹配。读数只表示该次采样，不代表持续负载；本机升级证明为 `/private/tmp/ypuddin-v058-service-proof.json`。

### 上轮 v0.5.7 Apple GPU 占用监控

MPS 设备通过无 sudo 的 `ioreg` 读取唯一 AGX 加速器的驱动利用率，2 秒缓存、2 秒超时；无法读取或无法唯一对应设备时返回 `null`，失败后不沿用上次成功数值。界面明确这是全系统 GPU 利用率，不是训练进程利用率，也不是统一内存比例；驱动统计窗口未知，功率和温度仍分别保留不可用状态。本轮只改遥测采集与说明，不修改训练调度、项目配置或素材。验收进度与限制见[补丁说明](docs/MAC_GPU_TELEMETRY_2026-09-12.md)和[验证记录](docs/validation/v0.5.7.json)。

正式 8876 已使用原数据根升级到 0.5.7。本次新鲜快照中的 194 个业务文件、六张业务表、设置与凭据配置状态完整保留，136 个源码输入及 53 个构建输出哈希匹配；正式 API 实读 GPU 利用率为 54%，来源 `ioreg`，统一内存与同次系统 RAM 统计一致。这个百分比是验收时读数，不代表持续负载。本机升级证明为 `/private/tmp/ypuddin-v057-service-proof.json`。

### 上轮 v0.5.6 数据与训练参数复查

版本内来源按真实 `traindata` / `reg` 目录归属自动确定用途；外部与旧目录保留原元数据，不移动原文件。新建任务的默认权重名使用安全化项目名称与版本号，保存位置绑定到版本的 `output/<job_id>`，旧任务与自定义名称保持。新项目默认整图保留，旧配置仍按既有裁剪语义读取；图像绘制与遮罩有独立保存、恢复和冲突保护。模型检测只读取受限大小的 safetensors 头部与配置，不执行 pickle；无法识别的字段需手动确认。环境显示真实 PyTorch/CUDA/设备与 distributed/NCCL 能力，当前每个训练任务仍只用一张卡，没有 DDP 训练实现。

本轮后端 **858 passed / 3 CUDA skipped（861 收集）**，前端 **65 个文件、388 项通过**，Ruff、ESLint、TypeScript 与生产构建通过。正式 8876 已在原数据根升级到 **0.5.6**：192 个业务文件、六张业务表、设置与凭据配置状态完整保留，136 个源码输入及 53 个构建输出哈希一致。升级未刷新用户页面或修改队列设置。实际浏览器证据及 Windows/NVIDIA、完整大模型训练等未验证边界见[本轮报告](docs/UI_REVIEW_V056_2026-09-12.md)和[验证记录](docs/validation/v0.5.6.json)；本机升级证明为 `/private/tmp/ypuddin-v056-service-proof.json`。提交和发布包校验以包旁验证文件为准。

以下 v0.5.5 及更早段落为历史版本记录，不代表本轮测试计数或当前预设入口：

v0.5.5 新增独立 `/presets` 管理页，使用真实 SchemaForm 创建、编辑、复制和删除参数预设；内置只读仍可展开分组和查看帮助。项目加载先显示描述和参数差异，同族确认后应用，跨族禁用；模型文件、数据来源、缓存、恢复和运行输出路径不会被预设替换。后端校验、内置保护、并发写入与失败保留同时接通。

共用 Help 使用门户浮层与边界避让，支持空白点击、Escape、焦点移出、滚动、路由切换关闭及多提示互斥。JSON 标签从上传/本机导入贯通解析、索引、逐图编辑、缓存、训练和版本复制；新来源 auto 优先 JSON，旧显式 TXT 保留。编辑保留描述和元数据，坏 JSON 明确报错，不以 TXT 或原始 JSON 文本冒充成功。

采样新增 Heun、ER-SDE 和独立噪声调度选择，旧 Euler+uniform 默认保持。ER-SDE 来源为作者 MIT 实现的独立 rectified-flow 适配，许可与数值边界见 [ER_SDE_PROVENANCE.md](ypuddin/sampling/ER_SDE_PROVENANCE.md)。优化器下拉映射真实注册项，缺依赖报错；LoKr Full 与低秩模式明确区分，Full 隐藏不生效的 Alpha。重复次数与正则损失权重按实际训练语义说明。

本轮前端 **57 个文件、331 项通过**，后端 **771 passed / 3 CUDA skipped（774 收集，134.143 秒）**；Ruff、ESLint、TypeScript 和生产构建通过。预设、Help 的 1004/390 界面，以及 JSON 查看/编辑/保存后描述与元数据保留已有真实浏览器和文件核对证据。正式 8876 服务已按原数据根运行 0.5.5；192 个业务文件、六张业务表、settings 与凭据配置状态保持，128 个源码输入和 50 个构建输出哈希匹配。证据、源文件与限制见[本轮报告](docs/UI_PARAMETERS_V055_2026-09-12.md)。

**历史 v0.5.4** 将项目流程合并为“训练数据 → 训练参数 → 训练结果”，模型配置进入训练参数分区；统一标题/工具条边框与吸顶遮盖，重排采样字段。任务图表将主损失与学习率/梯度诊断分开，显示 EMA 只影响图表；采样 REST/SSE 携带精确同训练步 loss，未知与初始采样明确标记。项目增加分类与手动封面，新项目/版本可指定已接入模型族，跨族重建默认训练配置但保留数据准备设置。RouteError 提供资源加载失败恢复入口；Plan 未就绪时只使用明确来源的 ready 索引数量，不伪造分桶或训练估算。详见[v0.5.4 变更报告](docs/UI_WORKSPACE_V054_2026-09-12.md)，其中统计和正式服务升级证明只对应历史 v0.5.4。

Flux/SDXL 当前未接入并禁选；Toy 是测试族。旧项目元数据迁移不移动目录，封面不进入训练数据；同族版本保留配置，跨族只使用目标族有效默认模型路径，不自动启用 FP8/CUDA。Windows/NVIDIA、完整 Anima/Krea 2 模型质量与新机部署仍无本轮实机验收；QA 生成夹具不能作为生产质量证据。

**历史 v0.5.3**：测试、生产构建与实际浏览器检查见 `docs/validation/v0.5.3.json`、`docs/screenshots/v0.5.3/` 和 [工作区复查报告](docs/UI_WORKSPACE_V053_2026-09-12.md)。当时变化是统一侧栏项目/版本/阶段、紧凑列表和表单、结果及队列工具、样式覆盖修正、标签导航保存保护。该版提交与 ZIP 独立解包校验见发布包旁的 `*.verification.json`，不是本轮 v0.5.6 的验收证明。

**历史 v0.5.2**：后端 562 passed / 3 CUDA skipped（565 收集），前端 45 文件 / 263 测试通过，Ruff/ESLint/TypeScript/生产构建通过。当时没有改训练后端；这些历史总数不作为 v0.5.6 的重跑结果。旧 v0.5.1 的 536/3、前端 220 同样仅供追溯。

当前是具备实际训练、数据上传到训练启动的 Web 工作流的集成验证版本；官方 Anima / Krea 2 全尺寸权重和 NVIDIA 路径仍待验收，不能称为所有功能已完成。

- 设置增加独立“访问密钥”，四来源共用本机安全存储：HF/ModelScope 令牌、Danbooru 用户名/API Key、Gelbooru 用户 ID/API Key。GET 仅返回 configured；正则收集默认读取存储，前端不再一次性输入站点密钥。旧接口保留兼容，运行任务保持启动时凭据。
- `model_recommendations.py` 明确官方组件与两来源映射，推荐下载校验大小/SHA-256/组件后才登记；重试保留原校验。跨来源共享 hash 目标目录互斥；本地候选在 `/use` 验 SHA，按文件状态缓存结果，已验证 VAE 可跨模型族复用。官方目录信息核对与本地小文件测试不代表完整权重训练验收。
- 全局队列负责跨项目任务筛选/分页，版本结果按明确版本查询，任务详情保留曲线/有界日志/任务采样与产物。项目和训练导航、标签查看器工具及分页固定可达；不同区域独立滚动。
- 配置与任务/数据集/预设/模型库辅助请求隔离。辅助失败显示局部重试，不伪造空数据、不用默认配置覆盖草稿；配置失败时禁止挂载可写表单，显式版本保存和迟到响应隔离均有回归。
- 训练参数按项目/版本保存 session 草稿，浏览器 Back 后可恢复未保存修改，并只合并用户改过的字段；串行保存避免旧请求覆盖新输入，保存完成仅清理对应草稿。通用 Dialog 自带独立样式，冷路由弹窗不依赖其他页面预先加载 CSS。

以下为本版继承的目录与训练契约；标明历史的数量和已撤下功能仅供追溯。

- v0.5.1 新项目采用用户填写的 ASCII ID、独立多语言显示名称、`project/<id>/vN` 物理目录；traindata/reg 与 samples/output 分离，采样和产物按任务 ID 隔离。新增 output_mode 明确默认项目路径与自定义根；旧项目和旧任务路径不迁移。
- 正则图提供本地无适配器底模生成、限定站点网络收集与手动导入；后台独立任务具备版本锁、GPU/环境互斥、暂存发布、取消和恢复。仅完整批次登记 is_reg 数据源；不继承主体 trigger、不参与自动 val split。当前混合加权均值损失不等同于独立 train/reg 两项平均。
- WD14 自动打标、WD14 专用模型推荐下载和推理 API 已撤下，前端改成图片与完整原标签查看；历史标签、备份和已下载资产保留。W&B 入口与默认依赖撤下，旧配置读取兼容。v0.5.2 的训练底模推荐是独立服务。
- 顶栏移除正常状态占位，侧栏品牌和底部等宽主题/语言控件重做；版本号移到设置服务信息。StudioSelect统一弹出菜单/键盘/主题/视窗定位；设置运行环境只显示主要后端与注意力加速。
- 启动默认 models/server/optim/logging，NVIDIA额外安装监控；增量升级同时检查脚本/依赖指纹和真实缺包，保护已安装Torch/CUDA/Numpy，不在运行环境页逐包管理普通依赖。当前证据见 `docs/validation/v0.5.1.json`；下文旧统计及WD14验收均为历史记录。

- v0.5 数据操作由 `server/dataset_pipeline.py` 管理，检查、排除、裁剪、缩放和标签修改都有操作记录与备份。准备到入队原子移交版本锁；修改数据会使旧准备结果过期。可视裁剪与 Mask 同步，撤销有外部修改冲突保护。
- 原生模式由 `data/native.py` 和 `train/trainer.py::_run_native_epoch` 实现：尺寸对齐裁剪、像素预算、异尺寸逻辑批次、按图片数归一化梯度。未实现 NaViT 序列打包；不额外依赖变长注意力或强制 latent 缓存。不要把分组数误作逻辑批次/优化器步数。
- 预设应用保留当前版本模型族、底模/分词器文件、训练/验证数据源、缓存、输出、日志、采样提示文件和恢复路径；保存预设移除这些项目字段。完整列表见 v0.5.5 报告。修改来源 caption 格式会同步本版本数据索引与逐图编辑器，图片/Mask 后缀不可用作标签后缀。
- **历史 v0.5.0（v0.5.1 已移除此功能）**：本地 WD14 曾在可取消的隔离子进程运行；官方固定版本的 ONNX/标签表通过模型目录下载，ONNX Runtime 可从环境页安装，图片不上传云端。推理结果由数据流水线统一备份、发布和撤销；环境维护与打标互斥。
- HF/ModelScope 凭证与公开设置分离，`secrets.json` 中是本机凭证，不是加密保险箱。API 只回传已配置状态，镜像匿名、跨域跳转移除认证头；不要把令牌写进配置、日志、截图、源码包或公共错误响应。
- 高级参数块按顺序排列、组内分列，概率编辑显示百分比。SSE 有无事件检测和重连，避免服务重启后监控静默保留旧值。本轮后端 **483 通过 / 3 CUDA 跳过**、前端 **176 通过**；类型检查、lint、生产构建、真实 WD14 CPU 推理、两次 6 步 Toy/MPS 原生训练及 16 份采样/权重下载核对通过。浏览器证据见 UI_PIPELINE 与 `docs/validation/v0.5.0.json`，下面的测试数量均为历史记录。

- v0.4.0 实际运行并对照 AnimaLoraStudio 0.27.0，统一项目页/训练页的身份、版本栏和四步工作区。顶部顺序为 CPU/内存/GPU/硬盘，GPU 保留占用/显存/功率/温度四读数。设置改为运行环境/模型权重/存储路径/界面与服务四类抽屉，训练产物回到项目版本的“任务与结果”。开发前端默认连接真实服务，mock 需显式 opt-in。
- `server/versions.py` 提供真实数据副本与操作锁；复制包含训练源、验证源、caption 和 Mask，不复制任务与产物。旧项目迁移为兼容 v1，不移动旧文件；新任务的配置、缓存、运行目录、采样和产物绑定明确版本。高级 TOML 外部引用不自动复制，任务也没有单独的不可变图片快照。详见 UI_VERSIONS 文档。
- **v0.4 后端 417 通过 / 3 CUDA 跳过，前端 149 通过**；lint、TypeScript、生产构建、真实十步 Toy/MPS、三张采样与三个权重下载核对通过。版本复制后的标签/遮罩互不影响，归档/恢复、抽屉上下文和页面尺寸已验收，详见 UI_VERSIONS 与 `docs/validation/v0.4.0.json`。本轮没有执行用户取消的在线 `npm audit`。
- v0.3.0 历史改造包括紧凑参数分区、实时分桶、字段预检定位、固定启动栏、实际遮罩绘制和环境依赖管理；当时模型/产物合并入设置的入口已被 v0.4 替代。环境变更仍串行化，重启前阻止任务使用旧进程依赖。Anima/Krea 共用 xformers/flash-attn 路径，Sage 仅采样。
- v0.3 历史验收为后端 **371 passed / 3 CUDA skipped**，前端 **115 passed（25 个文件）**；Lint、TypeScript 与生产构建通过。当时浏览器完成启用遮罩的 5 步 Toy/MPS 训练，最终产物包含 60 个可读张量。历史机器记录见 `docs/validation/v0.3.0.json`，截图见 `docs/screenshots/v0.3.0/`。
- v0.2.0 新增项目四步工作区、浏览器图片/目录/ZIP 上传自动同步配置、常用参数与首屏启动、模型组件下载和默认路径、环境诊断、NVML/nvidia-smi 功率采集与前端内容指纹。此前后端测试通过不代表前端体验已经完成；最新验收见 UI_WORKFLOW 文档。
- 前一轮修复了设备 RNG、scalar/dropout/Kahan 续训、mask 缓存、实际编码器指纹、验证源与分桶、分阶段模型加载、准备阶段暂停、队列设备分配、保存设置不生效、前端配置及实时数据断链。
- 前一轮训练核心回归 **278 passed / 3 CUDA skipped**；前端 **72 tests**、lint、TypeScript、production build 通过，包含本机实际 MPS 运算。浏览器已完成 TOML 导入、12 步 MPS 训练、初始/周期预览、权重下载、从第 6 步续训至第 12 步；60 个最终权重张量逐位相同。
- 新完整状态为 format 2，保存原始训练参数与模型资产身份；新数据指纹包含 caption / mask / 验证数据。**旧版完整状态可能不兼容**，不得跳过检查强行恢复。已有权重可通过 `adapter.resume_weights` 热启动新训练；不要删除旧状态或数据。
- Schedule-Free 模式、TensorBoard 和初始采样已接通，真实 Schedule-Free/TensorBoard 已测；早期 W&B sink 只做过模拟契约测试，当前不提供入口或默认安装。`optimizer.fused_backward=true` 当前显式拒绝，尚未实现。
- MPS 使用 FP32、不启用 autocast。系统仪表盘显示统一内存；训练指标显示当前 PyTorch 分配量。CUDA 的训练指标是 PyTorch 分配峰值，两者不等同。
- 队列按设备独占运行多个单设备任务，**未实现 DDP**。显存估算用于准入，可关闭 `memory_admission`；估算不是容量保证。

## 3. 仓库布局

```
ypuddin/            后端包本体
  adapters/         LoKr/LoRA/LoHa/Full/DoRA、注入、目标规则、IO/格式转换
  models/           ModelFamily 协议；anima/、krea2/、sdxl/、flux/、flux2/、toy/
  data/             数据集注册、分桶、caption 增强、内容哈希缓存
  train/            训练循环、暂停/恢复、验证、采样预览
  memory/           block swap、fp8 与激活卸载
  server/           FastAPI：项目/版本/任务/数据集/模型权重/SSE；versions.py 管理数据副本
  cli.py            ypuddin 命令入口
frontend/           React 18 + Vite 8 + TanStack Query + Tailwind；schema 驱动表单
scripts/bootstrap.py  一键部署全部逻辑（studio.sh/.bat 只是壳）
docs/design/        设计文档（00 架构、02 适配器、03 状态）——改代码要同步改这里
docs/reference/     对四个参考项目的源码级分析（写新功能前先查）
docs/deploy.md      部署/排障文档（§9 排障表常更新）
docs/api/openapi.json  由后端导出，前端 generated.ts 的类型来源
tests/              unit + e2e（toy 族让完整训练/服务流程在 CPU 几十秒跑完）
```

## 4. 关键设计决策（为什么这么做的，一句话版）

- **进程通信**：训练子进程写独立 JSONL 事件流，监督器读事件流更新 SQLite/SSE；stdout 只是日志。不要退回解析 stdout。
- **暂停/恢复**：任意步边界存全套状态（适配器/优化器/调度器/采样器位置/RNG/EMA），在相同资产、数据与兼容配置/运行环境下，恢复后与不间断训练**逐位一致**（CPU/MPS 回归证明；CUDA 待验收）。数据被修改时不得绕过状态指纹检查。
- **版本与缓存**：latent/文本缓存键 = 内容哈希（图片内容 × 桶 × 编码器指纹 × 翻转），同版本任务共享；项目任务创建时把 `dataset.cache_dir` 绑定到所属版本缓存目录。任务的 `version_id`、配置和运行路径保存后不跟随活动版本切换。文本缓存存的是**增强后**的 caption 变体（有界、确定性），所以卸载文本编码器后 shuffle/tag_dropout 仍可用。
- **LoKr 自研而非依赖 LyCORIS**：LyCORIS 4.0.0 的 `merge_to`/`get_diff_weight` 在 `alpha≠rank` 且 w2 低秩时把 scale 乘两次（实测，合并结果错一半）。我们的实现与之**文件格式兼容已实测**（`tests/unit/test_lycoris_compat.py` 用真 LyCORIS 加载我们的文件，输出逐位一致；需 `pip install lycoris-lora` 才跑，否则跳过）。细节：`docs/design/02-adapters-lokr.md` §7。
- **Block swap**：前后向双钩子 + 推迟释放（修掉了"块输入无梯度时 backward hook 提前触发"）；开/关 swap 结果逐位一致（有测试）。
- **配置**：分组 pydantic 模型 → JSON Schema（带 x-ui 提示）→ 前端零手写表单；新增配置项的完整链路见 00 架构文档。
- **部署**：包源镜像优先（中科大→清华→阿里→官方兜底，逐源回退）；torch CUDA 版本按显卡计算能力+驱动选（RTX 50 系强制 cu128）；uv 缓存与项目跨盘时自动 `UV_LINK_MODE=copy`。`studio.bat` 必须纯 ASCII + CRLF（.gitattributes 强制），不要往里写中文或 chcp。

## 5. 测试与验证体系

- `venv/bin/python -m pytest tests/ -q`（全量）；前端 `cd frontend && npm run lint && npm run test && npm run build`。用户本轮取消在线 `npm audit`，不要把历史审计记录当作当前依赖的审计结果。
- 四条铁律级测试：暂停/恢复逐位一致、swap 开/关逐位一致、`forward_bypass ≡ merged ≡ base+F.linear(x,ΔW)`（含 alpha≠rank）、LyCORIS 交叉加载。
- Anima/Krea2 各有一个 e2e：用**缩小版真实架构**组件在 CPU 跑完整链路（加载→文本→缓存→训练→验证→采样→导出→ComfyUI 键转换→合并回带前缀底模），fp8_scaled 底模也在其列。
- 前端单元测试可用 MSW，真实后端浏览器验收单独记录；v0.4 证据以 UI_VERSIONS 为准，历史截图在 `docs/screenshots/v0.3.0/` 和 `frontend/screenshots/`。

## 6. 接手后的 critical path（按序）

1. **官方完整权重与 NVIDIA 真机验证**。在有 N 卡的机器上：
   ```bash
   ./studio.sh          # 或 studio.bat；一键装环境起服务
   ./studio.sh smoke --set model.dit_path=<官方权重> --set model.text_encoder_path=... --set model.vae_path=...
   ```
   Anima 与 krea2 各跑一遍（Krea 2 的权重清单见 `docs/deploy.md` §5.1）。产出 `outputs/smoke/smoke-report.json`。最可能出问题：官方权重键名、Qwen3 单文件格式、bf16/fp8 显存行为——CPU 测试覆盖不到这些。
2. 真机数字出来后：block swap / unsloth 卸载 / sage / fp8 的收益实测，与 sd-scripts、diffusion-pipe 的基准对比（速度、显存、出图）。
3. 然后才轮到：多卡（采样器已按 rank 切分，训练器未包 DDP）、更多模型族、前端深度交互（见 §10.3）。英文语言包长尾已补齐（§10.1）。

## 7. 已知风险 / 坑

- **官方权重加载是最大未验证点**（兼容 `net.` / `model.diffusion_model.` / 裸键 / ComfyUI fp8_scaled 的代码都有，但只测过构造的键名）。
- LyCORIS 生态兼容只验证到"它的加载器读我们的文件"；ComfyUI 本体还没挂过我们训出的文件（同一套 kohya 键约定，风险低）。
- 文本管线对空 caption 已修（每行至少保留一个有效 token，防融合注意力 NaN）——改动时保持这个不变量。
- 前端 `JobListResponse` = openapi 的 `JobPage` 别名（分页信封），别再当成 `list[Job]`（出过一次 500 回归）。
- 接手时已存在的参考文档和前端修改均已保留。不要用 `git checkout` / reset 清理不属于自己的改动。
- `.handoff/` 曾是本机多 agent 交接目录，已从 git 移除（`.gitignore`）；如果新工作流不需要，直接删除目录即可。

## 8. 第三方代码与许可证

- 本仓库 Apache-2.0。vendor 的代码（均 Apache-2.0，各目录有 `NOTICE.md` 记录来源 commit 与改动）：`ypuddin/models/anima/vendor/`（sd-scripts：Cosmos-Predict2 DiT、Qwen-Image VAE，已去掉 block swap 与 sd-scripts 依赖）、`ypuddin/models/krea2/vendor/`（musubi-tuner 8934cfb：SingleStreamDiT，同样处理）。
- 同级目录的四个参考仓库（AnimaLoraStudio/、diffusion-pipe/、sd-scripts/、LyCORIS/）**不属于本项目**，运行时不依赖；其中 diffusion-pipe 与 AnimaLoraStudio 是 GPL-3.0——可读可参考，不要把代码搬进本仓库。
- 顶栏使用的官方 Lucide GPU 图标按 ISC 许可引入，完整声明在 `frontend/public/licenses/lucide-gpu.txt`，生产构建同时保留 `licenses/lucide-gpu.txt`。

## 9. 协作模式备注（可选继承）

此前前端由独立 agent 会话（Kimi）实现、本侧验收提交：它只改 `frontend/`，不 commit；验收方亲自跑 lint/test/build + 读源码 + 看真实后端截图后才提交。`.handoff/claude-to-kimi.md` / `frontend-status.md` 是当时的交接文件（已不在 git 内）。这个分工效果不错，但非必需——随新团队习惯调整。

## 10. 历史前端交接补充（原 Kimi 记录，保留供追溯）

> 下文的测试数量和待办反映本轮审计之前的状态。当前 MPS 后端、SSE 连接指示、TOML 导入导出、字段正确类型和产物续训已实现；以 §2、修复报告和源码为准。拖动布局、命令面板等深度交互仍未实现。

### 10.1 前端当前状态（实测）
- `npm run lint` 0 警告；`npm run test` **55 通过（10 文件）**；`npm run build` 通过（页面级 React.lazy chunk + echarts 独立 chunk 547KB）；`npm audit` 0 漏洞。
- 两份 locales（zh-CN 默认 / en）约 380 key，**中英已对齐**（之前 §6.3 提到的"95 个 key 靠中文默认兜底"已回填 en.json，此条作废）。i18next 插值已全局改为**单花括号 `{name}`**（与 locales 一致；新加文案保持同风格）。
- 图表已不用 echarts-for-react，是自研 `components/EChart.tsx`（ResizeObserver 自适应 + echarts/core 按需引入）；jsdom 下的 ResizeObserver/canvas polyfill 在 `frontend/tests/setup.ts`。

### 10.2 前端运行与验证
```bash
cd frontend
npm run dev                          # v0.4 起默认直连真实后端，/api 代理到 127.0.0.1:8765
VITE_USE_MOCK=true npm run dev       # 仅开发演示：显式启用 MSW 与模拟 SSE
```
验收脚本（puppeteer-core 驱动本机 Chrome，截图到 `frontend/screenshots/`）：
`test-real-flow.js`（训练全流程）/ `test-fe-m4.js`（数据集流）/ `test-cache-verify.js` / `test-jobdetail-verify.js` / `test-mock-sse.js` / `test-polish-shots.js`（8 页面批量截图）。

### 10.3 前端未完成项（优先级从高到低）
1. **Apple Silicon MPS 后端适配（用户点名的优先项，后端活，前端已承接完毕）**：前端已支持 `gpus[].kind='mps'` 渲染（显示"Apple GPU · MPS"、统一内存、util/temp/power 为 null 时 `--` 兜底）、macOS-arm64 平台识别（`isAppleSilicon`，兼容 `macOS-arm64` 与 `Darwin-arm64` 两种串）、无 GPU 时的蓝色"适配开发中"信息卡。精确契约已写入 `.handoff/kimi-to-claude.md`：`gpus[].kind="cuda"|"mps"`、`mem_*_mb` 用统一内存、训练 `device: mps`、`Plan.gpu_total_mb` 传统一内存。后端实现后前端零改动直接生效，回归 = 真机 toy 训练 + 仪表盘截图。
2. **训练配置页深度交互**（对照 AnimaLoraStudio）：右侧 scroll-spy 分组锚点导航、YAML/解析后配置实时预览抽屉、字段级"恢复默认"链接。
3. **数据集页深度交互**：三栏可拖布局（宽度持久化）、tag 联想输入（danbooru 词表）、批量 tag 操作的影响张数确认弹窗、主导色缩略图 placeholder。
4. **全局**：⌘K 命令面板、设计 token 全面 CSS 变量化（当前是 Tailwind slate/blue 直写，暗色可用但非 token 驱动）、SSE 断线状态的 UI 提示（连接/断开指示灯）。
5. **小项**：数据集图片后端补 `size` 字段后 caption meta 行自动显示（前端防御式渲染已就绪）；mock SSE 只模拟 running 状态的 mock 任务。

### 10.4 前端协作契约（沿用则有效）
- 只改 `frontend/`，不改 `ypuddin/`、`docs/`、不 git commit（验收方统一提交）。
- 每轮交付更新 `.handoff/frontend-status.md`（Done / Not done / How to run / Tests / Questions for backend，如实写）。
- 跨会话通信：`.handoff/kimi-to-claude.md`（前端→后端的需求/问题，当前挂起的就是 10.3.1 的 MPS 适配）与 `.handoff/claude-to-kimi.md`（后端→前端的任务包）。
