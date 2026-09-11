# 最新项目版本状态：v0.5.1

更新：2026-09-12。当前产品行为以本节和 [UI_SIMPLIFICATION_2026-09-12.md](../UI_SIMPLIFICATION_2026-09-12.md) 为准。下方 v0.5.0 及更早记录保留作历史证据，其中的 WD14 自动打标和旧环境包管理页面不再是当前功能。最终测试数量与浏览器证据由本轮验证记录统一列出，本页不复用历史总数。

- **项目与版本目录**：新项目分别填写显示名称和稳定的项目 ID。ID 允许 1–64 位 ASCII 字母、数字、下划线，拒绝 Windows 保留设备名、大小写重复与已有目录冲突。新布局为 `studio_data/project/<project_id>/v1/`、`v2/`；版本目录内分为 `traindata/<dataset_batch>/`、`reg/<regularization_batch>/`、`samples/<job_id>/`、`output/<job_id>/`、`cache/` 和 `config.json`。显示名称修改不改变项目 ID 或物理目录。版本 API ID 仍独立且全局唯一，v1/v2 由版本编号决定。
- **旧数据与输出归属**：既有项目、数据和任务不自动搬到新目录；旧任务继续使用已保存路径。默认输出跟随项目版本，自定义输出根继续追加项目、版本和任务层级。新任务采样独立归属版本下的 `samples/<job_id>`；完整训练状态与权重留在该任务输出目录。全局 `studio.db`、`datasets/<dataset_id>.json` 索引记录及 `cache/index.sqlite` 不属于单个版本的模型缓存。自定义缓存根仍按项目 ID 和版本 API ID 隔离；界面/API 返回路径是实际归属依据。
- **标签与界面精简**：WD14 自动打标、专用模型下载入口和公开推理入口已移除；标签步骤提供现有标签的查看、搜索、分页及逐图编辑入口。已有图片、caption、Mask、已下载模型和历史备份不会因移除入口被删除。侧栏底部统一主题/语言控件，真实服务版本移到设置的“界面与服务”；正常连接时顶栏不保留空反馈槽。
- **正则数据准备**：版本内提供已有图片导入、本地底模 AI 生成与 Danbooru/Gelbooru 收集入口。AI 仅使用当前版本的底模与用户填写的类别提示词，不创建训练适配器、不自动下载模型。网络仅访问限定站点和媒体域名，固定一般/全年龄评级，并保存站点标签；站点凭据仅本次使用，不回显、不落任务配置或日志，也不转发到媒体请求。
- **正则任务和训练消费**：每次独立批次，支持进度、取消、解码检查、像素内容去重和 manifest。完整图片/标签验证后才注册 `is_reg=true` 来源；清理须验证目录所有权，失败、取消或目录冲突不会覆盖旧批次。已完成批次在重启恢复时保留。AI 使用独立进程，与训练和环境维护共享准入锁；网络收集只锁所属版本。正则默认不继承主体触发词、不进入自动验证集切分，但保留显式 caption 配置；损失仍按混合样本的各自权重取批次平均，不能解释为两组损失分别归一化再相加。
- **依赖和运行环境**：启动器默认安装模型支持、常用优化器和本地日志依赖，NVIDIA 主机额外补齐 NVML；常规包不再要求用户逐项点击安装。环境页保留 Python、PyTorch、计算后端/设备及 xFormers、FlashAttention；Sage 旧值标明仅采样。普通包列表和常驻操作历史移除，进行中的安装、待确认计划和失败日志仍可见。W&B 不再提供前端选项或默认安装依赖，本地日志与曲线继续保留。
- **升级契约**：更新本地源码后重启 `studio.bat` / `studio.sh`，启动器检查安装指纹、实际依赖和前端内容指纹，缺项时补齐或重建；已匹配源码的发布包页面可直接使用。补普通依赖时保护既有 Torch/CUDA/Numpy 栈，不因检测到驱动变化自动替换。此流程是启动时检查，不是运行中的热更新或联网自动拉取项目源码。

**当前验收边界**：已在本机 macOS 实际通过浏览器提交 Toy 正则生成，并验证任务完成、图片/标签及版本数据源连通；取消、恢复、目录冲突与两站响应/凭据边界由聚焦自动化测试验证。受控网络响应不代表站点当下可用或特定账号权限已验证。完整 Anima/Krea 权重的训练与正则生成、Windows NVIDIA 环境、CUDA 注意力扩展实际安装与性能、ComfyUI 实际加载仍待目标机器验收；不宣称全部完成或保证显存需求。

# 历史项目版本状态：v0.5.0

2026-09-11：补齐版本内数据准备流水线、可视裁剪与可撤销修改、HF/魔搭下载来源和独立令牌设置；原生分辨率保留独立尺寸，通过有界前向分组累积同一逻辑批次梯度，兼容 Mask 与在线 VAE。高级参数的大块空白、百分比控件、监控布局及服务重启后的事件连接已修正。当时验证与硬件边界见 [UI_PIPELINE_2026-09-11.md](../UI_PIPELINE_2026-09-11.md)，算法规则见 [native-resolution.md](../native-resolution.md)。

# 历史项目版本状态：v0.4.0

2026-09-11：实际参考 AnimaLoraStudio 0.27.0 后统一项目/训练页布局，加入真实版本数据副本、空白/仅参数版本、参数比较、归档和目录展示。旧项目关联兼容 v1，旧文件与任务快照不移动；数据、缓存、任务、采样和权重按明确版本归属管理。设置为四类宽抽屉，产物回到项目结果；顶部硬件四组读数和 GPU 四项读数均使用真实接口，开发 mock 改为显式 opt-in。最终数量与浏览器/打包证据见 [UI_VERSIONS_2026-09-11.md](../UI_VERSIONS_2026-09-11.md)，尚未补入前不沿用历史测试数。高级 TOML 外部路径与可编辑版本数据的边界同见该报告。

# 历史工作区状态：v0.3.0

2026-09-11 新增紧凑配置、真实分桶可视化、手绘训练遮罩与环境依赖管理。当时模型和训练产物归入设置，产物入口已在 v0.4 回到项目结果。历史使用/验证边界见 [UI_REDESIGN_2026-09-11.md](../UI_REDESIGN_2026-09-11.md)。下面保留上一轮工作流和训练核心记录。

# 历史工作流状态：v0.2.0

2026-09-11 后续改造已接通浏览器数据上传、项目四步工作区、常用参数/首屏启动、模型组件下载与默认路径、功率采集和内容指纹打包。下方原训练核心记录保持作历史依据，当时 UI 与交付结果见 [UI_WORKFLOW_2026-09-11.md](../UI_WORKFLOW_2026-09-11.md)。Windows NVIDIA 官方完整模型训练仍待实机验收。

# 实现状态（对照 00-architecture.md 里程碑）

更新：2026-09-12。下表区分训练核心代码接通、自动化回归与完整模型硬件验收；历史核心修复证据见[修复报告](../FIX_REPORT_2026-09-11.md)，本轮产品变更见 [v0.5.1 报告](../UI_SIMPLIFICATION_2026-09-12.md)。设计文档中的目标不能替代本表的验证边界。

| 里程碑 | 状态 | 证据 |
|---|---|---|
| M0 骨架 / 配置 / 事件 / toy 族 / 最小循环 | 已实现；状态格式升级为 v2 | `tests/unit/test_config.py`、`tests/e2e/test_toy_training.py`；精确续训仅能在测试覆盖的配置与设备条件下作出结论 |
| M1 适配器引擎（LoRA / LoKr / LoHa / Full / DoRA）+ 存取转换 + 工具 | 已实现注入、导出、转换、merge / extract / resize；实际 ComfyUI 加载待验收；LyCORIS 加载器交叉测试已通过 | `tests/unit/test_adapters.py`、`tests/unit/test_tools.py`：factorization、scale/scalar、bypass、冻结层和存取往返 |
| M2 数据流水线 + 验证集 + Plan | 已实现；本轮修复切分、分桶、数据与资产指纹、mask 缓存和预检错误 | `tests/unit/test_data.py`、`test_plan.py`、`test_fingerprints.py`；Plan 与训练共享数据布局计算 |
| M3 Anima 族 | 缩小版真实组件的 CPU 训练链路已有测试；官方完整权重的 NVIDIA 训练待验收 | `ypuddin/models/anima/`、`tests/unit/test_anima_vendor.py`、`test_anima_family.py`、`tests/e2e/test_anima_pipeline.py`；包含 online/cached 文本、验证、采样、导出与键转换 |
| M3b Krea 2 族 | 缩小版真实组件的 CPU 训练链路已有测试；官方完整权重的 NVIDIA 训练待验收 | `ypuddin/models/krea2/`、`tests/unit/test_krea2_family.py`、`tests/e2e/test_krea2_pipeline.py`；包含 Qwen3-VL 文本管线、raw / ComfyUI 前缀 / fp8_scaled 加载与转换 |
| M4 内存与执行后端 | CPU 分阶段加载、Block Swap、激活检查点、fp8 冻结层等已有实现与局部回归；CUDA 性能与完整模型峰值待实测 | `tests/unit/test_block_swap.py`、`test_optim.py`；MPS 当前强制 FP32，按统一内存保守估算，不能套用 CUDA 显存节省比例 |
| M5 服务 API + 队列 + SSE + Web | 主要工作流已接通，包含动态配置、TOML、任务进度、产物、版本目录及正则任务；设置页已按 v0.5.1 精简；完整权重的浏览器验收待完成 | `tests/e2e/test_service.py` 与 `frontend/` 测试；真实 uvicorn + 子进程 toy 训练、预缓存、暂停/恢复、队列与 API 回归不等于真实 GPU 验收 |
| M6 文档 / 预设 / 基准对比 | 部署说明与内置预设已有；基准未完成 | 本目录、`docs/deploy.md`、`docs/reference/`；无已验证的最低显存或优于参考项目的速度结论 |

## 当前实现契约

1. **完整状态与推理权重分开**：state v2 保存原始可训练参数（含 scalar）、优化器、调度器、采样器位置、训练进度、RNG 与 EMA；恢复不会把为推理导出的参数误当作原始训练参数。DataLoader 使用独立随机生成器，验证、预览与导出会恢复训练模式和 RNG。旧数据指纹与排序规则不兼容时明确拒绝续训，不能通过忽略错误声称精确恢复。
2. **数据布局一致**：Plan 与训练共用索引和 train/val 切分逻辑。各 source 的分辨率合并进分桶集合；显式验证源保留独立 source 索引，并按图片内容从训练集排除重复数据。正则来源不进入自动切分，显式验证来源仍优先排除内容重叠。步数按实际训练桶、batch size、梯度累积、epochs 与 max_steps 计算；无图片、切分后训练集为空及无效配置返回带字段位置的错误。
3. **缓存依据实际内容**：VAE 与文本缓存指纹包含实际权重、配置和 tokenizer 内容，文件哈希由校验文件元数据的索引复用。latent 与 mask 分离：开启或修改 mask 不要求重编码 VAE，也不读取旧 latent 文件内附带的 mask。caption 和 mask 内容进入新版数据指纹；移动或重命名相同数据不依赖旧绝对路径维持身份。缓存采用唯一临时文件后原子替换。
4. **分阶段加载与设备约束**：主干先在 CPU 加载，准备数据缓存后再按设备和 Block Swap 配置放置；cached 文本与 latent 编码器在相应阶段结束后释放。MPS 首版按 FP32、关闭 autocast 执行，Plan 对权重、文本编码器与激活按 FP32 估算，CPU/MPS 共享内存不计作独立显存释放。显式 fp8、SageAttention 和 8-bit 优化器受设备门禁约束，compile 与 Block Swap 不能同时开启。
5. **Plan 是预检和估算**：Python `plan(..., device=None)` 可离线规划目标配置；传入实际设备时增加硬件能力检查，返回设备和有效精度说明。服务创建任务前按本机设备预检。估算不包含所有后端临时张量与碎片，不构成最低显存或不会 OOM 的承诺。
6. **优化器与日志**：Schedule-Free 优化器接通 train/eval 切换，评估后恢复训练参数；TensorBoard 依赖由启动器默认补齐，是否写入日志仍由配置控制。本地日志、训练/验证曲线和采样事件可用；W&B 仅保留历史配置/底层兼容，不再提供前端选项或默认安装依赖。`optimizer.fused_backward = true` 当前明确拒绝，不能作为已实现功能启用。
7. **事件和任务生命周期**：训练子进程写 `events.jsonl`，监督器尾读后更新 SQLite 并发布 SSE，stdout 仅作日志。训练阶段在优化器步边界暂停并保存完整状态；准备阶段的暂停保留已完成缓存，不伪造尚不存在的训练状态。SSE 回放历史有界且在内存中，重启后的当前状态应从 HTTP 查询恢复。
8. **队列与设置**：任务按优先级、创建时间调度，`max_concurrent` 是总上限，每个加速设备同一时间独占一个任务。默认 `queue.settings.memory_admission = true`，估算峰值超过设备可用内存的 95% 时等待；可关闭估算门禁，但仍保持设备独占。AI 正则生成也占用设备维护准入，网络收集仅锁所属版本。路径设置影响新任务，旧任务保留创建时路径；host/port 下次启动生效，`data_root` 由 CLI 指定，不在设置页迁移。
9. **Web 配置与产物**：配置表单从后端 JSON Schema 获取字段和条件显示规则，模型族能力参与选项展示；配置支持 TOML 导入导出（API 也支持 JSON）。任务页接入 step、phase、checkpoint 等 SSE 更新，提供产物下载与按完整断点续训。EMA 启用时另存 EMA 推理权重；它与普通权重、完整训练断点用途不同。
10. **采样与验证**：固定验证集、时间步和噪声种子用于可比较的验证损失；cached 文本支持有界、确定性的 caption 变体，并预编码采样提示词。采样阶段临时使用 VAE，并在退出时恢复训练资源和模式；需要在完整模型设备上继续验证数值和图像质量。
11. **适配器与格式工具**：LoKr 的 scale/scalar、factorization 和 bypass 路径有回归；`extract` / `merge` 处理模型前缀，`convert` 提供 kohya/ComfyUI 键转换。格式与数值单元测试不能代替实际 ComfyUI 加载验收。
12. **可重复自检入口**：`ypuddin smoke` 使用真实 trainer 执行短训练、预览、保存和回读，写出检查结果、耗时与可用的设备内存指标。toy 和缩小版组件降低了回归成本；历史通过情况见[核心修复报告](../FIX_REPORT_2026-09-11.md)，当前结果以 v0.5.1 验证记录为准。

## 待办（按优先级）

1. NVIDIA GPU 机器上用官方完整 Anima / Krea 2 权重验证：加载、bf16 / fp8 训练、暂停续训、验证、采样出图，以及 LoRA/LoKr 实际载入 ComfyUI 后的效果。MPS 完整模型可用性和数值质量也待实测。
2. fp8 `_scaled_mm` 路径；bitsandbytes 8-bit 优化器验证；unsloth 卸载 / sage / compile 在 CUDA 上的实测。
3. 多 GPU 协同训练（torchrun DDP）尚未接通：采样器的 rank 切分不能替代 DDP 包装、同步和 rank 0 保存；队列可调度多个独立设备任务也不等于单任务多卡训练。
4. Windows/Linux 新环境完整安装和真实模型浏览器操作验收；按实测反馈完善显存曲线、ETA、错误提示与 i18n。
5. 同一数据集/配置下与 sd-scripts、diffusion-pipe 的速度、峰值内存、验证损失基准。现阶段不宣称整体完成或性能超越参考项目。

## 运行方式

部署与日常使用见 `docs/deploy.md`（`studio.sh` / `studio.bat` 一键安装启动）。以下是开发者视角的命令：

```bash
cd xiangmuyuanma
uv venv --python 3.12 venv && uv pip install --python venv/bin/python -e ".[dev,models,server,optim,logging]"
venv/bin/pytest -q                         # 全部 CPU 测试
venv/bin/ypuddin plan config.toml          # 预检；读取权重几何信息，不执行模型训练
venv/bin/ypuddin plan config.toml --device cuda  # 按指定设备增加能力检查和预算
venv/bin/ypuddin train config.toml         # 训练（事件写到 <output_dir>/events.jsonl）
venv/bin/ypuddin serve --port 8765 --data-root ./studio_data   # 服务（前端 dist 存在时同域托管）
```

**GPU 机器首次验证（一条命令）**：真实跑 3 步 + 出一张 512 预览 + 保存/回读适配器，输出时序 / 峰值显存 / loss / 键格式，报告写到 `outputs/smoke/smoke-report.json`（失败时含完整 traceback，直接贴给我即可）：

```bash
venv/bin/ypuddin smoke \
  --set model.family=anima \
  --set model.dit_path=/models/anima-base-v1.0.safetensors \
  --set model.text_encoder_path=/models/Qwen3-0.6B-Base \
  --set model.vae_path=/models/qwen_image_vae.safetensors \
  --set adapter.algo=lokr --set adapter.rank=full --set adapter.factor=8 \
  --set memory.activation_checkpointing=block
# 显存紧张时追加：--set memory.blocks_to_swap=10 --set dataset.text_encoding=cached --resolution 512
# 也可直接给配置文件：ypuddin smoke my.toml --steps 5
```

Anima 配置最小示例：

```toml
[model]
family = "anima"
dit_path = "/models/anima-base-v1.0.safetensors"
text_encoder_path = "/models/Qwen3-0.6B-Base"        # HF 目录或单文件 safetensors
vae_path = "/models/qwen_image_vae.safetensors"
dtype = "bf16"

[dataset]
sources = [{ path = "/data/chara", repeats = 2 }]
resolutions = [1024]
batch_size = 1
caption = { trigger_word = "chara_name", shuffle = true, keep_tokens = 1 }

[adapter]
algo = "lokr"
rank = "full"
alpha = 1.0
factor = 8
preset = "attn-mlp"

[optimizer]
type = "adamw"
lr = 1e-4

[memory]
activation_checkpointing = "block"
blocks_to_swap = 0

[loop]
epochs = 10
grad_accum = 1
mixed_precision = "bf16"

[checkpoint]
output_dir = "outputs/chara-lokr"
name = "chara"
save_every_epochs = 1

[sampling]
enabled = true
every_epochs = 1
prompts = [{ prompt = "chara_name, 1girl, smile" }]

[validation]
enabled = true
split_ratio = 0.1
```

此示例面向 CUDA 配置。MPS 可使用 `--device mps --set model.dtype=fp32 --set loop.mixed_precision=no`；当前执行器即使收到 bf16 配置也会告警并使用 FP32。显式 fp8、SageAttention 和 8-bit 优化器不能用于当前 MPS 路径。
