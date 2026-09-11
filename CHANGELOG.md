# Changelog

## 0.5.2（2026-09-12 工作区设计复审与中央凭据）

设置增加独立“访问密钥”页，集中保存/清除 Hugging Face、ModelScope、Danbooru 与 Gelbooru 凭据；GET 只返回配置状态，正则任务默认从同一本机存储读取。表单反馈就近显示，错误不反射密钥。旧模型令牌和一次性站点凭据 API 保留兼容。

新增官方模型组件推荐目录，明确两平台仓库/文件映射并固定字节数和 SHA-256。推荐下载验证后才发布/登记，重试保留原校验；两来源目标目录互斥，本地候选使用前验 SHA，Anima/Krea 共用 VAE 可复用。自定义下载与本地模型登记保留。

重做全局队列、版本结果和任务详情的职责与紧凑布局，后端增加队列过滤/SQL 分页及有界日志读取；项目/训练导航与标签分页保持可达。配置与辅助请求隔离，失败重试不覆盖用户草稿。实际参考 AnimaLoraStudio 0.27.0 的页面组织与服务边界后独立实现，未移植参考代码。

训练草稿按版本保存会话兜底，Back 后恢复并解决旧保存请求覆盖新修改的竞态。Dialog 独立导入样式，修复直接进入冷路由时的弹窗布局。

本版不搬动既有项目/数据、不改完整状态格式。后端 562 通过 / 3 CUDA 跳过，前端 45 文件 / 263 测试通过，lint、类型检查和生产构建通过。当前验证与未覆盖的正式模型、Windows/CUDA 和外部账号权限见 [设计审查与验收记录](docs/UI_DESIGN_REVIEW_2026-09-12.md)；机器证据统一在 `docs/validation/v0.5.2.json`，提交及交付包校验保存在发布包旁的 verification JSON。

## 0.5.1（2026-09-12 界面精简与正则图）

项目显示名与 ASCII ID 分离，新目录采用 `project/<id>/vN`，traindata/reg/samples/output 按职责和任务隔离。加入正则 AI/限定站点收集，移除 WD14 自动打标及 W&B 前端入口；常规依赖改由启动器补齐。历史验收见 [v0.5.1 报告](docs/UI_SIMPLIFICATION_2026-09-12.md)。

## 0.5.0（2026-09-11 数据准备与原生尺寸）

版本数据准备、可视裁剪和可撤销处理、原生尺寸逻辑批次训练、HF/ModelScope 下载来源与独立令牌。该版曾包含 WD14 自动打标，后续 0.5.1 已移除；历史验收见 [v0.5.0 报告](docs/UI_PIPELINE_2026-09-11.md)。

## 0.4.0（2026-09-11 项目版本）

引入真实版本数据副本、空白/仅参数版本、比较归档和任务输出归属；旧项目兼容迁移不搬文件，设置改为抽屉。历史验收见 [v0.4.0 报告](docs/UI_VERSIONS_2026-09-11.md)。

## 0.3.0（2026-09-11 工作区与环境管理）

训练配置改为四个紧凑分区，取消重复常用表单；真实分桶图/明细与固定启动栏同时可见，预检错误中文化并可定位到字段。添加逐图遮罩编辑、灰度 PNG 保存、撤销重做、已有 alpha / sidecar 读取和直接启用训练。模型权重、下载、默认路径和产物统一进入系统设置的环境管理，旧链接继续跳转。

新增环境依赖探测、版本与 wheel 兼容预检、安装计划/日志、安装/修复/卸载及重启门禁；保护当前 Torch/CUDA/NumPy，避免可选扩展升级替换训练基础环境。Anima/Krea 共用注意力层接 xFormers/FlashAttention；Sage 仅明确的无梯度采样使用，训练与 checkpoint 反向保持 SDPA。实际对照 AnimaLoraStudio 0.27.0（3d9d2e8）页面独立实现；未复制参考前端代码或资源。


## 0.2.0（2026-09-11 工作流交付）

项目内贯通数据上传/目录导入、模型选择、常用训练参数、首屏启动按钮与结果；新增 Hugging Face 模型组件下载、默认模型路径、环境诊断和 Windows NVML/nvidia-smi 功率采集。前端构建改为内容指纹校验，发布包包含已编译页面。完整使用、验收与未验证边界见 `docs/UI_WORKFLOW_2026-09-11.md`。


## 0.1.0（未发布，2026-09）

首个可用版本：训练核心（LoRA / LoKr / LoHa / DoRA / 全量差分，精确暂停恢复，确定性验证，block swap，
unsloth 式激活卸载，fp8 冻结底模，Kahan bf16 优化器），Anima 模型族（vendored Cosmos-Predict2 DiT +
Qwen-Image VAE + Qwen3 文本管线），数据流水线（内容哈希缓存、caption 变体缓存、可恢复分桶采样器），
服务 API（项目 / 数据集 / 任务队列 / SSE / 产物 / 模型注册）与 Web 界面，`ypuddin smoke` 自检，
`studio.sh` / `studio.bat` 一键部署。

Krea 2 模型族（`model.family = "krea2"`）：vendored musubi-tuner `SingleStreamDiT`（Apache-2.0），Qwen3-VL-4B
文本管线（官方提示词模板、12 层隐状态堆叠、去 padding 缓存），支持官方 bf16 / ComfyUI 前缀 / Comfy-Org
**fp8_scaled** 检查点（fp8 层沿用文件 scale 直接冻结），分辨率自适应时间步 shift（训练与预览一致），4 个目标预设，
内置预设 `krea2-lokr-default` / `krea2-lora-32`；`ypuddin merge` 兼容 ComfyUI 的 `scale_weight` 键。

### 2026-09-11 审计修复

修复训练恢复、mask/编码器缓存、验证分桶、MPS 与分阶段加载、任务生命周期、设置与队列、前端动态配置/TOML/实时图表/下载续训。新增本机实际 MPS、Schedule-Free、TensorBoard 验证；详见 `docs/FIX_REPORT_2026-09-11.md`。

**兼容变化**：完整状态升级 format 2，保存原始适配器参数及实际模型身份；数据指纹纳入 caption/mask/验证数据且采用内容排序。旧完整状态可能被拒绝恢复，可用旧版本跑完或将旧权重作为 `adapter.resume_weights` 热启动，不能跳过校验假装精确恢复。旧文件不被迁移或删除。`fused_backward=true` 从静默无效变为显式拒绝；Kahan 与 Schedule-Free 组合拒绝。Node 构建要求修正为 20.19+ 或 22.12+（不含 21）。
