# 2026-09-11 核心修复与验收

本轮在 `xiangmuyuanma` 内完成修复，保留接手时已有的未提交修改；同级参考仓库未改动，未执行 Git 提交。修复前的问题证据保留在 [完成度审计](COMPLETION_AUDIT_2026-09-11.md)。当前仍是未发布的 0.1.0 集成验证版本。

## 已完成

| 范围 | 修复后的行为 |
|---|---|
| 训练设备 | 自动 CUDA → MPS → CPU；噪声使用可保存的 CPU RNG，生成后送到执行设备，修复 CPU generator 与 GPU randn 不匹配。MPS 强制 FP32、关闭 autocast，保存/恢复 MPS RNG。 |
| 完整续训 | format 2 保存原始训练参数，修复 scalar 折叠、module dropout、DataLoader RNG、Kahan 状态精度。数据与模型身份在恢复前校验；同路径换底模会拒绝，同内容改名可继续。 |
| 数据与缓存 | mask 与 latent 分离，添加/修改/删除 mask 即时生效且无需重编码 VAE；caption/mask/验证源纳入数据指纹。按内容稳定排序，修复 source 分辨率集合、验证源索引、训练/验证重复内容和空训练集。 |
| 编码器身份 | VAE、文本权重、tokenizer、配置、有效 dtype 进入实际内容指纹；持久化文件 hash 索引避免每任务重读大文件。缓存使用独立临时文件与原子替换。 |
| 加载顺序 | DiT 先留 CPU，缓存所需 VAE/文本编码器分阶段加载和释放，再注入/搬运训练主干；换出层不会先被整块搬上 GPU。该路径的 CUDA 全尺寸内存收益仍待测。 |
| 预览与日志 | 接通 `sampling.at_start`、Schedule-Free train/eval 模式和 TensorBoard/W&B sink。预览结束或失败时恢复模式并卸载 cached 模式的 VAE。 |
| 检查点 | 普通与 EMA 权重分别上报，按 step 成组保留；已轮换删除的文件不再出现在下载/续训列表。LoRA smoke 误判已修。 |
| 服务生命周期 | 准备阶段支持文件暂停/停止，缓存可复用；恢复准备中再次暂停保留原恢复目标。终态不被 shutdown 覆盖，失败原因保留，取消计时器只作用于原进程。SSE 连接不再无限阻塞 CLI 关闭。 |
| 队列与设置 | 各加速器独占，多个独立任务可分配到不同设备；可选内存估算准入。新任务冻结绝对路径、独立输出与事件文件；缓存/输出/模型设置影响后续任务，监听地址重启生效，data_root 由启动参数控制。 |
| Web | 动态 Schema/默认值、正确 nullable 字段、TOML 导入导出、预设保存、名称/优先级/排期、字段级错误、真实 EMA/epoch 曲线、SSE 状态/重连/进度、分页、权重下载和完整状态续训。采样每条提示词的空值正确继承全局，不显示伪造的 42/1024。 |
| 安装与文档 | 补 Python 3.10 的 tomli、Schedule-Free/Lion 可选依赖；构建要求修正为 Node 20.19+ 或 22.12+。开发代理端口跟随实际后端设置。HANDOVER、状态表、架构、部署和 API 快照已同步。 |

`fused_backward=true` 尚未实现，现明确拒绝；Kahan 与 Schedule-Free 的不受支持组合也明确拒绝，不再静默接受无效配置。

## 验证

- 后端：最终完整 `venv/bin/python -m pytest`：**278 passed / 3 CUDA skipped**，116.73 秒。包含 SSE 退出和 MPS 指标/轮换接口回归。
- Python lint 与格式检查：`ruff check ypuddin scripts tests`、`ruff format --check ypuddin scripts tests` 通过。
- 前端：**72 tests / 15 files 通过**；lint 0 warnings；TypeScript 与 Vite production build 通过。
- 实际设备：macOS 15.7.9 arm64，32 GB 统一内存，PyTorch 2.14.0、transformers 5.17.0。实际 MPS 测试涵盖训练、swap 开/关、逐位续训和预览；两族真实缩小架构在 CPU 做完整加载/训练/验证/采样/导出回归。
- 可选依赖：真实 Schedule-Free 1.4.1 与 TensorBoard 2.21.0 测过训练、评估、采样、保存与恢复；TensorBoard 文件重新读取确认含 loss、验证与图片。W&B 仅模拟 sink 契约，未连接远程账户。
- 警告边界：现有 vendor CPU autocast 提示、PyTorch backward hook 提示和 TestClient 依赖弃用提示不影响本轮通过；未借此声称 CUDA 路径已验证。

### 浏览器实测

使用新建的临时数据目录与本机 8876 端口，无 mock。所有图片为合成测试数据。

1. 网页创建项目，导入 TOML，Plan 显示总 12 步、每轮 7 步、13.3K 可训练参数。
2. 网页提交任务 `j_fa629f0b8057`，实际 MPS 执行 1–12 步；第 0/6/7/12 步生成预览，写出权重、完整状态和 TensorBoard 日志。
3. 网页最终权重下载触发成功，服务返回 HTTP 200。
4. 点击第 6 步“从此状态继续训练”，新任务 `j_c8e11c8937fd` 实际仅执行 7–12 步并完成。
5. 从磁盘读回两个最终文件，**60 个权重张量逐位相同**，并核对事件序列；这是实际网页 → 服务 → MPS 子进程的恢复验证。
6. 最终构建正确显示 MPS“当前分配”与“当前训练分配量”。无效 TOML 显示 `checkpoint.save_full_state: Extra inputs are not permitted`，原草稿保留。

MPS 的训练分配量取 `torch.mps.current_allocated_memory()`，CUDA 取指定设备的 `max_memory_allocated()`，通过 `vram_metric` 区分。系统统一内存、MPS 当前训练分配量与 CUDA 分配峰值是三种不同口径。

## 升级与保留

- 原有源码改动、权重、数据及旧 `ypuddin-src.zip` 均保留。新包另命名，含源代码、测试、文档、模型结构/tokenizer 资源、编译前端与逐文件 SHA-256 清单；不含 venv/node_modules、训练模型权重、运行数据库/缓存、Git 历史或本机会话目录。
- 完整状态从推理导出权重改为保存训练原始参数。旧数据指纹/文件名排序不能兼容新版时应使用旧版本跑完，或用 `adapter.resume_weights` 热启动新任务；不能跳过校验宣称精确续训。旧 v2 缺少模型身份字段会明确警告无法验证身份。
- 精确恢复依赖兼容的数据、资产、优化配置和执行环境。配置哈希变化仍会警告，不能把任意超参数修改或跨平台恢复解释为逐位保证。
- 新编码器指纹会让旧缓存重新构建一次；旧缓存文件保留。迁移/备份应包含设置指向的外部输出目录，更新源码包不覆盖运行数据。

## 尚未完成的验收与功能

官方 Anima/Krea 2 完整权重的 NVIDIA 训练、CUDA fp8/8-bit/Sage/compile/异步换块峰值与速度、实际 ComfyUI 载入及图像质量、完整模型 MPS 可用性、Windows/Linux 全新安装、与参考项目的同配置 benchmark 均未完成。单任务 DDP、多卡协同训练、fused backward 和拖动布局/命令面板等深度前端交互未实现。部分字段标题仍来自英文 Schema。

下一步应提供官方权重路径与 NVIDIA 设备，使用 `studio.sh smoke` 的报告完成真机验收；不能用 toy 测试替代这一阶段。
