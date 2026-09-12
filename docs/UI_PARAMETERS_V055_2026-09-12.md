# v0.5.5 参数、预设、JSON 标签与采样变更记录

> 日期：2026-09-12。本文记录本轮实际实现、全量回归和已有浏览器证据。正式 8876 服务已按原数据根升级并核对数据保留；不以测试通过代替完整权重、Windows/NVIDIA 或训练质量验收。v0.5.4 的项目导航、封面和版本模型族记录保留在[上一版报告](UI_WORKSPACE_V054_2026-09-12.md)。

## 用户问题与实施结果

| 本轮问题 | 已实施的行为 | 验证边界 |
|---|---|---|
| 只有项目内保存/加载，无法单独管理参数预设 | 新增全局“参数预设”页面，按模型族分组，支持真实表单创建、编辑、复制、删除；内置只读但可查看帮助和复制 | 预设不携带另一个项目的图片、底模文件或运行目录；跨模型族加载被阻止 |
| 加载预设缺少说明，容易覆盖当前项目配置 | 显示描述和修改前后参数预览，确认后才应用，取消不修改草稿 | 当前项目的模型族及受保护路径保持原值；加载不等于配置已通过训练检查 |
| 问号说明被侧栏/容器剪裁，点击空白不关闭 | 共用 Help 改为门户浮层，避让视窗和滚动容器边界，支持外部点击、Escape、焦点移出、滚动和路由离开关闭，多个提示互斥 | 几何单测和实际 1004/390 截图分开记录，不将模拟矩形当作浏览器验收 |
| JSON 标签不能真正参与导入、查看和训练 | 新来源默认 auto，JSON 优先于 TXT；解析结构化标签和描述，贯通索引、逐图编辑、缓存、训练和版本复制 | 仅支持明确列出的 JSON 结构；坏 JSON 显示错误，不回退为 TXT 或把原始 JSON 当训练文本 |
| 优化器选择和参数含义不清 | 下拉展示实际注册项，保留显式自定义类入口，补齐精度、调度、正则损失权重和重复次数说明 | 依赖或设备缺失报错；下拉可选不等于所有优化器已在 NVIDIA 验收 |
| LoKr Full、低秩和 Alpha 容易混淆 | LoKr 使用明确模式选择；Full 隐藏不生效的 Alpha，切回低秩恢复数值 | LoKr 完整因子不是完整底模训练，也不等于目标层预设 full-linear |
| 采样缺少 ER-SDE 和噪声调度器 | 实际接入 Euler、Heun、ER-SDE，以及 uniform/simple/sgm_uniform/normal 网格；预览和本地正则生成共用分发 | 保留旧 Euler+uniform 默认；没有声称与参考应用逐像素一致或新增 DPM++ |

## 独立预设与加载保护

全局 `/presets` 使用动态训练 Schema 和实际字段控件，提供名称、说明、适用模型、参数摘要、参数搜索及高级选项。新建时读取目标族的默认配置；内置预设可查看说明、展开分组和复制，只有真正可写字段禁用。自定义预设保存错误显示字段原因并保留输入，离开未保存页面时可保存、放弃或继续编辑；保存失败不会当作成功导航。

项目参数页加载时先展示描述及参数差异。跨族预设显示适用模型并禁用，应用函数也保留当前 `model.family`。删除预设不删除或改写已保存的项目配置。

[trainingPresets.ts](../frontend/src/utils/trainingPresets.ts) 在保存可复用预设和应用预设时都过滤以下字段，应用时保留当前项目对应值：

| 分类 | 受保护字段 |
|---|---|
| 模型文件 | `model.dit_path`、`text_encoder_path`、`vae_path`、`tokenizer_path` |
| 数据与缓存 | `dataset.sources`、`dataset.cache_dir`、`validation.sources` |
| 产物与恢复 | `checkpoint.output_dir`、`checkpoint.resume`、`adapter.resume_weights` |
| 采样与日志文件 | `sampling.output_dir`、`sampling.prompts_file`、`logging.events_path` |

后端 `/presets` CRUD 校验名称、重复项、模型族和参数内容，拒绝修改内置预设。新增与修改语义分开，更新不存在项不会静默创建；写入串行化并采用临时文件原子替换，失败保留旧文件。预设验证不要求当前机器已经具有图片或模型权重，实际训练仍需项目配置与 Plan 检查。

主要实现：[Presets](../frontend/src/pages/Presets/Presets.tsx)、[presetEditor](../frontend/src/utils/presetEditor.ts)、[PresetPreview](../frontend/src/components/PresetPreview.tsx)、[routes_core](../ypuddin/server/routes_core.py)。

## Help 浮层与参数可读性

[ConfigHelp](../frontend/src/components/ConfigHelp.tsx) 将浮层挂到 document body，避免被字段卡片裁剪。位置根据触发点、视窗和所在滚动容器计算，空间不足时翻到上方；长说明可在浮层内部滚动。点击说明内部不会误关，点击空白处、Escape、焦点移出或外层滚动会关闭；打开另一个问号会关闭前一个，分区卸载和路由离开也会清理。

浮层使用现有主题变量。内置预设只读不再通过整块禁用字段集禁用帮助和分组按钮。数据导入中的重复次数输入有独立 label/id，帮助按钮不抢输入框的可访问名称；标签和问号同排，输入保持下一行。

[训练参数说明](TRAINING_PARAMETERS.md) 补充了训练噪声分布、采样噪声调度和学习率调度的区别，以及分桶/原生尺寸、正则图、精度、缓存与续训的实际含义。说明基于当前实现，不提供一套保证训练效果的通用配方。

## JSON 标签的数据链路

新来源默认 `caption_ext="auto"`，同名 JSON 优先于 TXT，扩展名大小写不影响识别。显式 `.txt`、`.json` 或自定义后缀仍仅读取指定格式；旧 `.txt` 配置不自动改写。所选 JSON 损坏、字段类型不符或包含非有限数值时，报告文件名和原因，不静默换成另一个文件。

支持顶层标签数组/字符串、分类标签对象，以及 `fixed/ai_output/from_path` 的完整结构。解析后的固定标签、可变标签、自然语言 `nl` 和触发词按规则组合；未知元数据不进入训练文本。JSON 必须是对象，不能把任意 JSON 文件改名后就当作支持的 caption。具体结构和优先级见 [JSON 标签说明](JSON_CAPTIONS.md)。

标签查看器展示解析文本。逐图编辑器将标签与只读自然语言描述分开；保存 JSON 时更新权威标签列表，保留描述、`meta` 和其他字段，原分类对象按需保留备份。清空标签不会让旧分类重新出现，也不会删除仍保留的自然语言描述。JSON 的格式化和字节布局可能改变，字段内容保留不等于文件逐字节不变。

上传、ZIP、本机导入、索引、数据流水线、缓存、训练和版本复制使用同一解析契约。流水线的可恢复操作保留原始字节备份，并拒绝覆盖外部修改；逐图保存和直接批量标签接口不自动新增流水线撤销记录。批量写入失败会回滚本次已写文件，成功后的手动恢复仍需自己的备份。

未编辑的分类 JSON 保留固定组和描述，逐标签打乱/丢弃作用于可变组；整条 caption dropout 仍可置空全部条件。手动扁平化编辑后，标签进入一般组，不再承诺原分类的固定保护。文本缓存按最终文本和编码器指纹复用；数据身份仍包含标签文件内容，因此只改元数据也可能使完整状态的精确续训拒绝恢复。旧 TXT 身份算法保持原契约。

主要实现：[caption_json](../ypuddin/data/caption_json.py)、[captions](../ypuddin/data/captions.py)、[数据索引](../ypuddin/data/index.py)、[CaptionViewer](../frontend/src/components/datasets/CaptionViewer.tsx)、[逐图编辑器](../frontend/src/pages/Dataset/Dataset.tsx)。本轮没有重新引入自动打标或云端图片标注。

## 优化器、LoKr 与数据来源参数

优化器默认仍为 AdamW。表单列出实际注册的 `adamw`、`adam`、`sgd`、`adamw8bit`、`lion`、`lion8bit`、`prodigy`、`prodigy_plus_sf`、`adafactor`、`came`、`adamw_sf`，高级用户可显式输入已安装的 `module.Class` 和额外参数。缺依赖或设备不支持时给出错误，不自动换用其他算法。8-bit 依赖 CUDA/bitsandbytes；`optimizer.fused_backward=true` 仍未实现并明确拒绝。

LoKr 的 `rank="full"` 使用完整 Kronecker 因子 W1/W2，仍是 LoKr 增量。`adapter.algo="full"` 则训练选定线性层的完整权重；目标预设 `full-linear` 只控制层范围。这三者不能混用。LoKr Full 时缩放固定为 1，Alpha 不参与该模式，界面隐藏但保留原值；切回低秩时再次显示并可编辑。没有把隐藏字段偷偷写成另一个参数。

数据来源的 `repeats` 默认 1，只影响训练项展开，不复制磁盘文件。目录前缀如 `5_character` 只产生可明确应用的重复次数建议。正则来源的 `prior_weight` 显示为“正则损失权重”，不是主体/正则出现比例；值为 0 的图仍占批次，不等同于删除图片。原有加权批次平均语义没有改成独立两项损失。

## 采样器与噪声调度

| 参数 | 当前选项 / 默认 | 行为 |
|---|---|---|
| `sampling.sampler` | `euler`（默认）、`heun`、`er_sde` | 选择预览的数值求解方法 |
| `sampling.scheduler` | `uniform`（默认）、`simple`、`sgm_uniform`、`normal` | 选择噪声时间网格；与学习率调度分开 |
| `sampling.er_sde_order` | 1 / 2 / 3，默认 3 | 最高阶数，历史不足时从一阶开始 |
| `sampling.er_sde_s_noise` | 0–1，默认 1 | 沿途追加噪声强度；0 不会取消初始噪声或改成 ODE 求解器 |

`uniform` 保留旧连续网格；`simple` 从离散网格取点；`sgm_uniform` 与 `normal` 的低噪声端点处理不同。分发器验证步数 1–1000、正且有限的 shift 以及严格递减的有限网格。无效配置报错，不默默改回默认。Euler+uniform 保留原调用路径；Heun 在非末步增加一次预测修正。

[dispatch.py](../ypuddin/sampling/dispatch.py) 已接入训练预览与本地正则图生成。ER-SDE 根据作者 MIT 许可实现的 Taylor 更新独立适配 rectified flow，保留 [来源说明](../ypuddin/sampling/ER_SDE_PROVENANCE.md) 与 [MIT 许可](../ypuddin/sampling/ER_SDE_LICENSE.txt)，没有复制 AnimaLoraStudio/ComfyUI 的 GPL 求解器。

ER-SDE 使用有限信噪比起点，历史阶数逐步升高，以稳定比值和有界数值积分计算高阶系数；终点返回当前去噪估计。所有随机数来自独立生成器，不推进训练的全局 RNG。数值测试覆盖作者方程的独立对照、非均匀网格、端点、CFG、取消及 RNG 隔离；这些不证明正式底模画质优于 Euler，也不保证与其他应用逐位相同。

当前没有接入 DPM++ 3M SDE；参考界面的候选列表不是本项目支持列表。采样选项没有增加新的训练模型族：Anima/Krea 2 已接入，Toy 用于测试，Flux/SDXL 仍禁选。

## 本轮测试与构建证据

| 检查 | 本轮结果 | 证据与范围 |
|---|---|---|
| 前端全量 | **57 个文件、331/331 通过** | `/private/tmp/ypuddin-v055-frontend-final.json`，直接读取 `testResults.length` 与结果计数；不复用 v0.5.4 的 309 项 |
| 后端全量 | **774 项：771 通过、3 跳过，134.143 秒** | `/private/tmp/ypuddin-v055-backend-final.xml` 和同名 JSON/log；0 failures、0 errors，3 项均为 CUDA unavailable |
| Ruff / ESLint | **退出码 0** | 本轮全量检查；不等于外部依赖或全部硬件已验证 |
| TypeScript / Vite 生产构建 | **退出码 0** | 本轮构建成功；正式运行目录更新与数据保留另列 |
| 正式 8876 服务升级 | **已运行 0.5.5，数据与静态资源核对通过** | `/private/tmp/ypuddin-v055-service-proof.json`；六张业务表、192 个文件、settings 和凭据配置状态不变，128/50 个输入/输出哈希匹配 |

新增/更新回归包括 [Help 交互](../frontend/tests/configHelp.test.tsx)、[独立预设](../frontend/tests/presetManagement.test.tsx)、[预设路径保护](../frontend/tests/trainingPresets.test.ts)、[同族预览与跨族禁用](../frontend/tests/families.test.tsx)、[LoKr 模式](../frontend/tests/lokrParameterMode.test.tsx)、[导入来源参数](../frontend/tests/projectWorkflow.test.tsx)、[预设 API](../tests/unit/test_preset_management.py)、[JSON 解析](../tests/unit/test_caption_json.py)、[JSON 文件流程](../tests/unit/test_json_caption_workflow.py)、[调度分发](../tests/unit/test_sampling_schedule_dispatch.py) 和 [ER-SDE 数值](../tests/unit/test_er_sde.py)。

## 正式服务升级与数据保留

本轮按原数据根重启 `http://127.0.0.1:8876`，health 返回 0.5.5，`frontend_stale=false`。`/`、`/presets` 与访问密钥设置路由均返回 HTTP 200；静态前端包含独立预设页面资源，API 静态描述的版本也已更新为 0.5.5。

升级前后对比六张业务表：projects、project_versions、datasets、jobs、artifacts、models 的内容保持；192 个业务文件哈希不变，settings 与四个来源的凭据配置状态保持，操作记录及队列状态未被重置。128 个源码输入和 50 个构建输出通过哈希核对。证明位于 `/private/tmp/ypuddin-v055-service-proof.json`，引用了前后快照、数据对比和静态资源明细。

这是既有本机数据根的升级与 HTTP 资源证明，不是新机器安装、Windows 部署或源码包独立解包验收。浏览器视觉操作见下一节；本文不记录任何凭据内容。

完整机器记录（含服务保留和截图哈希）已纳入 [v0.5.5.json](validation/v0.5.5.json)。

## 已有浏览器证据

以下文件已存在于 `docs/screenshots/v0.5.5/`。预设预览和 Help 覆盖 1004/390 宽度；JSON 已在真实服务中查看、打开编辑器并保存，另读文件确认描述 `nl` 与测试元数据 `quality_meta` 保留。截图展示具体页面状态，不代替该功能所有分支和设备验收。

| 页面 / 操作 | 已保存证据 |
|---|---|
| 采样器与调度器选项 | [sampler-scheduler.png](screenshots/v0.5.5/sampler-scheduler.png) |
| 实际优化器选择 | [optimizer-options.png](screenshots/v0.5.5/optimizer-options.png) |
| 独立预设保存 | [preset-saved.png](screenshots/v0.5.5/preset-saved.png) |
| 加载前参数对比，1004 / 390 | [preset-review.png](screenshots/v0.5.5/preset-review.png)、[preset-review-390.png](screenshots/v0.5.5/preset-review-390.png) |
| Help 边界，1004 / 390 | [help-1004.png](screenshots/v0.5.5/help-1004.png)、[help-390.png](screenshots/v0.5.5/help-390.png) |
| JSON 解析后标签查看与编辑 | [json-caption-view.png](screenshots/v0.5.5/json-caption-view.png)、[json-caption-editor.png](screenshots/v0.5.5/json-caption-editor.png) |
| 自动格式、重复次数建议与正则字段 | [source-auto-repeats.png](screenshots/v0.5.5/source-auto-repeats.png) |
| 正式预设页，1004 / 390 | [presets-formal.png](screenshots/v0.5.5/presets-formal.png)、[presets-390.png](screenshots/v0.5.5/presets-390.png) |

## 仍未完成的验收

Windows/NVIDIA 实机训练、官方完整 Anima/Krea 2 权重的效果和性能、实际 ComfyUI 加载、全新机器安装仍不能由本轮测试推出通过。3 项 CUDA 跳过是未验收边界，不是 CUDA 通过。QA 生成图片、小型权重和数值方程夹具不能作为生产训练质量证明；高阶采样或更多优化器选项也不意味着训练效果保证。正式服务升级证据不替代独立源码包校验；发布包应另行检查所需内容和完整性。当前不宣称整个产品已生产就绪。
