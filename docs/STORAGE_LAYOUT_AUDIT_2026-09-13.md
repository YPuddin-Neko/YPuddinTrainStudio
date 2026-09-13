# 存储目录与迁移边界审计（2026-09-13）

## 范围与结论

审计对象是 `/Volumes/Service/Dev/YPuddinTrainStudio/xiangmuyuanma` 当前源码；比较对象是同级 `AnimaLoraStudio` 工作副本。本文的目录是源码规定的落点，**不是对用户磁盘上全部数据的盘点**。`R` 表示本项目源码根，`D` 表示本次服务的 `data_root`，`P` 表示项目 ID，`VID` 表示数据库版本 ID，`vN` 表示用户可读版本编号，`J` 表示任务 ID。

当前新项目采用“项目 → 版本 → 数据、缓存、任务输出”的结构，训练图与正则图分开，任务输出与采样按任务隔离，基础模型共享。这些边界有明确用途，不需要因为目录数量多就整体推翻。`project/projects`、`traindata/datasets`、`output/runs/outputs` 同时出现，主要来自旧布局兼容和服务/CLI 的不同入口。

真正需要处理的是：**整套数据换根目录没有配套的引用重定位；路径设置只改变后续落点；项目删除遗漏部分索引和自定义缓存；任务删除先删数据库再删文件；缓存设置的名称与覆盖范围不直观。** 下文区分已经确认的代码行为、由代码可推导的条件性影响，以及尚未执行的验证。

本项审计只读源码并新增本文；没有移动、删除、改写用户数据或运行配置，没有启动迁移、训练或依赖安装，也没有进行实机换盘/迁移验证。文档整理只检查引用与 Markdown 差异，不运行全量测试。

## 1. 代码、环境与数据根

| 范围 | 当前落点与职责 | 依据 |
| --- | --- | --- |
| 源码 | `R/ypuddin`、`R/frontend/src`、`R/scripts`、`R/docs`；前端产物为 `R/frontend/dist` | [bootstrap.py:43–45](../scripts/bootstrap.py#L43)；[app.py:42](../ypuddin/server/app.py#L42) |
| Python 环境 | `R/venv`，与服务数据分开；不是模型或训练缓存目录 | [bootstrap.py:43–44](../scripts/bootstrap.py#L43) |
| 官方脚本启动 | bootstrap 使用源码根作为服务子进程 `cwd`；默认相对 `studio_data` 因而落在 `R/studio_data` | [bootstrap.py:573–579](../scripts/bootstrap.py#L573) |
| 直接启动服务 | `ypuddin serve --data-root` 默认 `studio_data`；应用用 `expanduser().resolve()` 解析，因此相对于启动进程工作目录 | [cli.py:314](../ypuddin/cli.py#L314)；[app.py:42–52](../ypuddin/server/app.py#L42) |
| 自定义数据根 | `--data-root /absolute/path` 决定本次 `D`；设置接口拒绝切换 `data_root` | [context.py:89–92](../ypuddin/server/context.py#L89) |

因此，不能笼统说“启动路径全靠 cwd，官方脚本不可靠”。官方 bootstrap 已锚定源码根；直接 CLI/API 入口及相对配置路径仍保留 cwd 语义。源码根与数据根可以同盘相邻，也可以分开，结构本身合理。

## 2. 当前新项目的目录地图

下图表示默认服务布局；目录按功能使用时创建，不保证所有目录同时存在。外部注册模型、自定义输出与高级配置引用的数据集可能在 `D` 之外。

```text
R/
├── ypuddin/、frontend/、scripts/、docs/   代码与前端构建产物
├── venv/                               Python 环境
└── studio_data/                         默认 D；可由 --data-root 改为别处
    ├── studio.db                       服务数据库；运行中可能有 WAL/SHM 伴随文件
    ├── settings.json                   全局设置，含缓存/模型/输出路径
    ├── secrets.json                    本地模型下载凭据
    ├── presets/                        用户参数预设
    ├── models/                         默认共享模型下载与扫描目录
    │   ├── <family>/<kind>/<identity12>/<filename>
    │   └── .downloads/<download-id>/... 下载暂存
    ├── datasets/<dataset-id>.json      数据集记录索引；不是训练图片副本
    ├── thumbs/<hash>_<size>.jpg         共享缩略图
    ├── cache/
    │   ├── index.sqlite                服务数据集扫描索引
    │   └── shared/                     默认无项目任务训练缓存
    ├── environment/
    │   ├── cache/、installer/          依赖安装缓存和必要时创建的 pip helper
    │   └── <operation-id>/...          安装工作目录、下载的 wheels 等
    ├── runs/<J>/                       默认无项目任务运行目录
    └── project/<P>/
        ├── .studio/                   项目封面
        └── v<N>/
            ├── config.json            本版本可编辑草稿
            ├── traindata/             版本拥有的训练图片/caption/mask
            ├── reg/                   正则数据；生成期间有 .staging-<operation-id>
            ├── pipeline/              数据处理流水线操作资料
            ├── cache/                 同一版本的预缓存与训练任务共享
            │   ├── index.sqlite
            │   ├── fingerprints/file-hashes.sqlite
            │   ├── latents/<shard>/<key>.safetensors
            │   └── text/<shard>/<key>.safetensors
            ├── samples/<J>/           本任务生成的采样图
            └── output/<J>/            本任务运行档案
                ├── job-config.toml、config.toml
                ├── run.log、events.jsonl
                ├── control/           pause/stop/save 等控制文件
                ├── *.safetensors      LoRA/LoKr 权重，可能包含 EMA 权重
                ├── state-*/           完整续训状态
                └── tensorboard/       启用时创建；W&B 也以运行目录为本地根
```

| 目录/数据 | 精确源码入口 | 设计判断 |
| --- | --- | --- |
| 数据库与设置 | [app.py:49–52](../ypuddin/server/app.py#L49)、[db.py:48–57](../ypuddin/server/db.py#L48)、[context.py:42–68](../ypuddin/server/context.py#L42) | 数据与代码分离；SQLite 开启 WAL。备份不能只凭目录名字判断运行中数据库是否已一致复制。 |
| 凭据、预设 | [model_credentials.py:81–84](../ypuddin/server/model_credentials.py#L81)、[routes_core.py:347–349](../ypuddin/server/routes_core.py#L347) | 属于持久用户状态，不应当作可重建缓存。 |
| 模型注册与下载 | [routes_core.py:510–565](../ypuddin/server/routes_core.py#L510)、[model_downloads.py:264–269](../ypuddin/server/model_downloads.py#L264)、[暂存:357](../ypuddin/server/model_downloads.py#L357)、[发布:443](../ypuddin/server/model_downloads.py#L443) | 注册保存绝对路径，不复制权重；下载才写受管理的模型根。按族、组件类型与身份分目录，避免同名覆盖，暂存后发布。 |
| 项目与版本 | [context.py:111–117](../ypuddin/server/context.py#L111)、[137–165](../ypuddin/server/context.py#L137)、[routes_work.py:218–234](../ypuddin/server/routes_work.py#L218) | 路径由项目/版本绑定决定；新版本可读编号降低人工定位成本。 |
| 数据集与版本复制 | [versions.py:103–130](../ypuddin/server/versions.py#L103)、[346–373](../ypuddin/server/versions.py#L346) | 图片资料通过真实复制隔离，避免 caption/mask 编辑串版本；新版本重置续训引用和缓存落点，不复制旧输出和大模型。代价是副本占空间。 |
| 封面、正则、流水线 | [project_covers.py:205](../ypuddin/server/project_covers.py#L205)、[regularization.py:113–117](../ypuddin/server/regularization.py#L113)、[dataset_pipeline.py:69–73](../ypuddin/server/dataset_pipeline.py#L69) | 分别归属项目或版本，避免放进模型根或训练输出根。 |
| 数据记录与服务索引 | [routes_work.py:914–965](../ypuddin/server/routes_work.py#L914)、[缩略图:1361–1374](../ypuddin/server/routes_work.py#L1361) | 共享元数据/展示缓存与实际图片分开；需要独立说明清理规则。 |
| 版本训练缓存 | [context.py:221–231](../ypuddin/server/context.py#L221)、[trainer.py:174–175](../ypuddin/train/trainer.py#L174)、[dataset.py:515–526](../ypuddin/data/dataset.py#L515)、[trainer.py:373–374](../ypuddin/train/trainer.py#L373)、[fingerprints.py:103](../ypuddin/models/fingerprints.py#L103) | 同版本多任务复用，跨版本默认隔离；索引、latent、text、模型指纹不是同一种内容。张量缓存按键分片并临时文件替换，[cache.py:27–53](../ypuddin/data/cache.py#L27)。 |
| 运行目录与采样 | [routes_work.py:1675–1692](../ypuddin/server/routes_work.py#L1675)、[1739–1741](../ypuddin/server/routes_work.py#L1739) | 创建任务时固定输出、事件、采样、缓存，保存任务配置快照，避免后来修改草稿改变已建任务。 |
| 日志、权重、续训 | [supervisor.py:190–214](../ypuddin/server/supervisor.py#L190)、[控制文件:477–480](../ypuddin/server/supervisor.py#L477)、[trainer.py:483–540](../ypuddin/train/trainer.py#L483)、[logging.py:14–39](../ypuddin/train/logging.py#L14) | 运行记录跟随任务；完整 `state-*` 与推理权重职责不同。日志不必再单独集中到一个全局 `logs` 目录。 |
| 环境缓存 | [environment.py:386](../ypuddin/server/environment.py#L386)、[289–316](../ypuddin/server/environment.py#L289)、[897–899](../ypuddin/server/environment.py#L897) | 依赖安装工作区，不属于 dataset 的训练缓存；与 `R/venv` 也不是两个等价训练环境。 |

## 3. 同类名字为什么有多套

| 名称 | 实际含义 | 是否应直接合并 |
| --- | --- | --- |
| `project` / `projects` | 新项目 `layout_version >= 2` 用单数；旧项目继续用复数。旧行升级为兼容版本时不搬文件、不重写历史快照。 | 不应直接改名；需要有计划的布局迁移。依据：[context.py:111](../ypuddin/server/context.py#L111)、[db.py:59–100](../ypuddin/server/db.py#L59)。 |
| `vN` / `versions/VID` | 新布局版本为 `project/P/vN`；旧布局版本为 `projects/P/versions/VID`。兼容版本的草稿还可能留在旧项目根。 | 是读取兼容分支，不是随机选择目录。依据：[context.py:137](../ypuddin/server/context.py#L137)、[200–207](../ypuddin/server/context.py#L200)。 |
| `traindata` / `datasets` | 新版本图片在 `traindata`，旧版本在 `datasets`；全局 `D/datasets/*.json` 则是记录索引。 | 图片与 JSON 不能合并；UI/文档应明确两种 `datasets` 的区别。依据：[context.py:153–155](../ypuddin/server/context.py#L153)、[routes_work.py:914](../ypuddin/server/routes_work.py#L914)。 |
| `output` / `runs` / `outputs/run` | 新项目默认 `vN/output/J`；旧版本默认 `runs/J`；无项目服务任务默认 `D/runs/J`；独立 CLI 配置默认 `outputs/run`。 | 入口不同可以保留；文档不能只给其中一条作为所有任务的路径。依据：[context.py:167–219](../ypuddin/server/context.py#L167)、[schema.py:675](../ypuddin/config/schema.py#L675)。 |
| 多处 `cache` | 版本 tensor 缓存、服务扫描索引、缩略图、依赖下载各自生命周期不同。 | 不应机械合并为一个目录；应提供覆盖范围与占用/清理入口。 |

设置中默认 `cache_dir=D/cache` 实际充当“使用版本缓存布局”的判定值：有项目时写 `version/cache`，无项目时写 `D/cache/shared`。自定义缓存写 `<cache_dir>/P/VID`；自定义输出写 `<output_dir>/P/vN/J`。`VID` 与 `vN` 不一致不会直接造成缓存串用，但人工找目录不够统一。[context.py:209–231](../ypuddin/server/context.py#L209)

## 4. 已确认的缺口与影响

### 4.1 整套数据换根没有完整的引用重定位

绝对路径本身有必要：服务先按提交进程 cwd 固定数据/模型/缓存/输出/续训等路径，再在任务运行目录启动子进程。这样任务换 cwd 后仍能找到同一批文件。[config/io.py:105–127](../ypuddin/config/io.py#L105)、[supervisor.py:190–214](../ypuddin/server/supervisor.py#L190)

但是，这些路径同时被持久化到 `settings.json`、版本 `config.json`、数据库数据源/模型/任务/产物行、数据集记录 JSON、任务配置快照等位置。数据库定义与任务入库可见 [db.py:18–38](../ypuddin/server/db.py#L18)、[routes_work.py:1739–1741](../ypuddin/server/routes_work.py#L1739)。目前设置读取只强制刷新 `data_root`；已保存的 `cache_dir/models_dir/output_dir` 仍覆盖新根下的默认值。[context.py:45–68](../ypuddin/server/context.py#L45)

**条件性影响，由代码推导，未实机复现：** 若将 `D1` 完整复制到 `D2`，再仅用 `--data-root D2` 启动，且原设置/草稿已保存 `D1` 下的绝对路径：

- 已保存 `D1/models` 仍是模型根，模型登记和历史任务引用也不会因复制自动改成 `D2`。
- `D1/cache` 不再等于新默认 `D2/cache`，因此会被当成自定义缓存根，新项目任务可能写回 `D1/cache/P/VID`。
- 旧草稿内的 `D1/project/P/vN/output` 不等于 `D2` 的默认运行根，可能被解释为用户自定义输出，再追加 `P/vN/J`。原盘还可写时表现为继续写旧盘，原盘不存在时可能失败；实际结果取决于权限和路径是否存在。

这不是设置页已承诺自动搬迁却失效；当前接口明确拒绝改数据根，部署文档也要求另行迁移。缺口是**没有覆盖这些引用的迁移工具/操作流程**，所以“复制整个目录后改启动参数”不能视为已支持的完整搬家。[context.py:89–92](../ypuddin/server/context.py#L89)、[deploy.md:114–118](deploy.md#L114)

后续方案需要区分“数据根内受管理路径”与“用户外部引用”，生成重定位预览；只重写前者，保留并报告后者。任务快照的历史意义、续训引用与模型登记都必须纳入，不能对 JSON/数据库字符串盲目全局替换。

### 4.2 修改路径设置只改变后续落点，不移动现有文件

保存设置会原子替换 `settings.json`，没有复制目录或改写历史任务。模型下载从当前模型根选目标，原模型登记仍指原文件；新建项目任务使用当时的缓存/输出设置。它是合理且已文档说明的行为，但不是迁移功能。[context.py:74–109](../ypuddin/server/context.py#L74)、[model_downloads.py:264](../ypuddin/server/model_downloads.py#L264)、[deploy.md:116](deploy.md#L116)

相对配置路径也不会自动以配置文件所在目录为基准：`load_config` 只合并内容，服务在 `absolute_paths` 阶段按 cwd 解析。应保持明确说明或未来新增显式的解析基准；不能在升级中悄悄改语义。[config/io.py:80–96](../ypuddin/config/io.py#L80)

### 4.3 项目“连文件删除”仍留下部分项目相关数据

项目删除已做归档、活动任务、版本复制和工作进程检查；文件删除异常会保留归档项目记录，方便重试。这是正确保护。[routes_work.py:351–397](../ypuddin/server/routes_work.py#L351)

确认遗漏：该分支删除任务输出/采样和项目树，然后删除数据库记录；不会遍历删除 `D/datasets/<dataset-id>.json`，也不会清理项目树外的自定义版本缓存。数据集行由项目外键级联删除，绕过单个数据集删除时的 JSON 清理。[routes_work.py:374–400](../ypuddin/server/routes_work.py#L374)、[db.py:18–22](../ypuddin/server/db.py#L18)、[单数据集清理:1207](../ypuddin/server/routes_work.py#L1207)

影响是残留索引文件和可能较大的 tensor 缓存；不能据此断言仍在使用的项目已损坏。共享 `thumbs` 与扫描 `index.sqlite` 的留存可能是复用取舍，但应有垃圾回收/占用说明。清理自定义缓存必须验证项目/版本归属，不能删除用户设置的整个缓存父目录。

### 4.4 单任务删除的失败处理弱于项目删除

`delete_job` 先执行 `c.db.delete("jobs", jid)`，再 `shutil.rmtree` 运行/采样目录。文件删除一旦因权限、占用或 I/O 失败，任务记录已经删除，不能像项目删除那样保留原入口重试。此处是明确的操作顺序问题；本审计没有对真实任务执行破坏性复现。[routes_work.py:1774–1788](../ypuddin/server/routes_work.py#L1774)

建议与项目删除采用一致的可重试流程，或保存清理状态；文件系统与 SQLite 无法组成同一个原子事务，需要处理部分成功。

### 4.5 缓存设置覆盖范围与日志文档需要说清

`paths.cache_dir` 主要控制训练张量缓存的根；服务数据集扫描索引固定在 `D/cache/index.sqlite`，缩略图固定在 `D/thumbs`，依赖缓存固定在 `D/environment/cache`。因此“把缓存改到另一块盘”并不会迁移或重定向所有缓存。这是实际的命名/容量管理缺口，不表示缓存键计算错误。[context.py:221](../ypuddin/server/context.py#L221)、[routes_work.py:931](../ypuddin/server/routes_work.py#L931)、[1366](../ypuddin/server/routes_work.py#L1366)、[environment.py:300](../ypuddin/server/environment.py#L300)

部署文档故障表还把完整日志写成 `studio_data/projects/<pid>/runs/<jid>/run.log`，未覆盖新默认 `project/P/vN/output/J/run.log`。任务真实路径应以持久化的 `run_dir` 为准。该文档行是已确认的过时说明；本文未顺带修改其他文件。[deploy.md:262](deploy.md#L262)

## 5. 与 AnimaLoraStudio 的对应关系

本节仅作源码机制比较，不复制实现。链接指向本地同级参考仓库，随本项目单独分发时该参考仓库可能不存在。

| 边界 | 本项目 | AnimaLoraStudio |
| --- | --- | --- |
| 代码与环境 | `R/venv`；bootstrap 锚定源码根启动 | `studio.sh` 先切到脚本目录，再使用/创建 `venv`，也兼容 `.venv`。[studio.sh:43](../../AnimaLoraStudio/studio.sh#L43)、[126–151](../../AnimaLoraStudio/studio.sh#L126) |
| 默认模型根 | `D/models`，可设置外部目录 | 默认 `<参考源码根>/models`；`models_root()` 每次读取模型根配置。[models/paths.py:209–225](../../AnimaLoraStudio/studio/services/models/paths.py#L209) |
| 默认服务数据 | 启动参数选 `D`；官方脚本默认 `R/studio_data` | 从模块 `__file__` 求源码根，默认 `<参考源码根>/studio_data`。[infrastructure/paths.py:10–30](../../AnimaLoraStudio/studio/infrastructure/paths.py#L10) |
| 数据根位置配置 | `--data-root`；没有根外 pointer 和 UI 复制迁移服务 | 源码根 `studio_data_location.json` 保存绝对目标目录；导入模块时读取，改 pointer 后需重启。[paths.py:24–60](../../AnimaLoraStudio/studio/infrastructure/paths.py#L24) |
| 项目/版本资料 | `project/P/vN`，训练数据、正则、缓存和任务档案按版本归属 | `projects/<id>-<slug>/versions/<label>`，版本创建 `train/reg/output`。[versions.py:184](../../AnimaLoraStudio/studio/services/projects/versions.py#L184)、[382–387](../../AnimaLoraStudio/studio/services/projects/versions.py#L382) |
| 任务日志/采样 | 默认在版本的 `output/J` 与 `samples/J`，项目删除可连带删除 | `studio_data/tasks/<id>` 集中任务快照、监控、采样和日志，使任务历史与版本目录解耦；旧 `logs` 等保留读兼容。[paths.py:64–78](../../AnimaLoraStudio/studio/infrastructure/paths.py#L64) |

Anima 的三块默认目录 `venv/models/studio_data` 是“可重建依赖 / 可共享的大模型 / 用户状态”的划分；本项目把默认模型放在 `D/models` 是另一种合理边界。前者方便单独换模型盘，后者让默认数据根更集中。两者都仍需要处理外部模型和自定义输出，不能仅按顶层目录数量判优劣。

Anima 已有两套真实复制入口，成熟度比本项目当前“只改路径”更进一步：

1. **数据根迁移：** 路由先检查无运行任务；校验目标必须是绝对路径、源目标互不嵌套，实际目标 `<所选父目录>/studio_data` 必须空或不存在；后台复制，SQLite 用 backup API，其余文件 `copy2`；完成后写根外 pointer，保留源目录，重启后生效。[路由:40–57](../../AnimaLoraStudio/studio/api/routers/studio_data.py#L40)、[目标校验:134–161](../../AnimaLoraStudio/studio/services/studio_data.py#L134)、[复制及 pointer:204–295](../../AnimaLoraStudio/studio/services/studio_data.py#L204)
2. **模型根迁移：** 后台复制，单文件先写 `.studio-migrate.part` 再替换，支持已有目标的冲突策略；复制成功后更新模型根配置，后续 `models_root()` 读取立即生效。已有目标数据时失败不会清空整个目标目录。[models_storage.py:245–349](../../AnimaLoraStudio/studio/services/models_storage.py#L245)

但不能把“有复制迁移”表述成“所有历史路径都已完整可迁移”：审阅到的数据迁移函数负责复制与写 pointer，没有遍历重写配置/数据库的绝对引用；参考项目的版本配置创建时也会把 `train/reg/output` 路径写入配置，并明确消费时不回填这些用户字段。[studio_data.py:204–257](../../AnimaLoraStudio/studio/services/studio_data.py#L204)、[version_config.py:53–79](../../AnimaLoraStudio/studio/services/version_config.py#L53)

此外，pointer 目标无效时参考实现会回退默认数据目录并记 warning。它提供了可继续启动的退路，也意味着挂载缺失时可能进入另一份旧数据，不能理解为新旧目录实时同步。[paths.py:33–57](../../AnimaLoraStudio/studio/infrastructure/paths.py#L33)

以上是代码可见的机制与边界；未执行 Anima 实机迁移，未验证迁移过程中所有写入者的协调、迁移后历史任务续训或旧盘卸载后的完整行为。参考项目的迁移入口值得借鉴，但其存在不是端到端迁移正确性的证明。

## 6. 建议处理顺序与验收边界

| 顺序 | 建议交付 | 应验证的实际结果 |
| --- | --- | --- |
| 先处理 | 修正任务删除顺序/失败记录；补齐项目 JSON 索引和受管理自定义缓存清理 | 注入文件删除失败后仍有可重试记录；外部用户目录不被误删；清理可说明残留。 |
| 随后处理 | 统一展示每个项目/版本/任务的实际数据、缓存、输出、日志位置；区分训练缓存与其他缓存；更新日志文档 | 用户从任务能定位真实 `run_dir`；自定义缓存时明确 `VID` 子目录与设置覆盖范围。 |
| 独立实现 | 全数据迁移预览、受管理引用重定位、数据库一致备份、复制校验、切换与恢复流程 | 先用隔离数据根演练，再做获授权的实机验证；旧盘不可用时能打开资料、读日志/采样、找模型、续训和创建新任务，且不意外写回旧盘。 |
| 后续取舍 | 若仍需要统一 `project/projects` 或历史目录，另做可回退的布局迁移 | 保留历史任务绑定，明确哪些档案随项目删除，不能直接批量重命名目录冒充完成迁移。 |

本文不代表上述修复已实施。现有版本隔离、共享模型和任务快照可以保留；需要补齐的是目录归属、生命周期与迁移能力之间的接口。
