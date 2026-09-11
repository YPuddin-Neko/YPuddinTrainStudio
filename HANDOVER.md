# YPuddin Train Studio — 项目交接报告

> 写给接手本项目的模型/工程师。本文自洽：读完这一份 + 点开的几个文件，就能不需要前任任何上下文地继续开发。
> 日期：2026-09-11 · 仓库：`xiangmuyuanma/` · 本文在原交接资料基础上按源码审计与修复验收更新；当前源码与前端交付版本为 0.2.0。

## 0. 先读这三个文件

1. 本文。
2. `docs/UI_WORKFLOW_2026-09-11.md` —— 最新 v0.2.0 前端工作流、上传、模型下载、功率采集、Windows 更新方式与本轮验收。`docs/FIX_REPORT_2026-09-11.md` 记录此前训练核心修复与升级边界；`docs/COMPLETION_AUDIT_2026-09-11.md` 保留修复前审计证据。
3. `docs/design/03-status.md` —— 逐组件状态表与运行方式。

## 1. 项目定位

一个桌面化的 diffusion **LoRA / LoKr 训练器**，目标是比 sd-scripts、diffusion-pipe、AnimaLoraStudio 做得更好（对它们的逐项短板分析见 `docs/reference/*.md`）。不是 GUI 套壳：训练核心、适配器、数据流水线、服务 API、CLI 全部自研。

- **模型族**：`anima`（Anima 2B，Cosmos-Predict2 DiT + Qwen3-0.6B + Qwen-Image VAE）与 `krea2`（Krea 2 Raw 12.9B 单流 MMDiT + Qwen3-VL-4B + 同款 VAE）是一等公民；另有 `toy` 族供 CPU 测试。新族 = 实现 `ModelFamily` 协议（`ypuddin/models/`）。
- **适配器**：LoKr（自研，与 LyCORIS 文件格式兼容）、LoRA、LoHa、Full、DoRA 包装。目标选择用 preset + 有序 rules（`ypuddin/adapters/rules.py`）。
- **形态**：Python 包 `ypuddin`（CLI + FastAPI 服务）+ `frontend/`（React/Vite 界面，可选）。一键脚本 `studio.sh` / `studio.bat`。
- 许可证 Apache-2.0（参考项目里 diffusion-pipe 与 AnimaLoraStudio 是 GPL——只读不抄；sd-scripts / musubi-tuner 是 Apache-2.0，vendor 的代码见 §7）。

## 2. 当前状态（v0.2.0 工作流改造后）

当前是具备实际训练、数据上传到训练启动的 Web 工作流的集成验证版本；官方 Anima / Krea 2 全尺寸权重和 NVIDIA 路径仍待验收，不能称为所有功能已完成。

- v0.2.0 新增项目四步工作区、浏览器图片/目录/ZIP 上传自动同步配置、常用参数与首屏启动、模型组件下载和默认路径、环境诊断、NVML/nvidia-smi 功率采集与前端内容指纹。此前后端测试通过不代表前端体验已经完成；最新验收见 UI_WORKFLOW 文档。
- 前一轮修复了设备 RNG、scalar/dropout/Kahan 续训、mask 缓存、实际编码器指纹、验证源与分桶、分阶段模型加载、准备阶段暂停、队列设备分配、保存设置不生效、前端配置及实时数据断链。
- 前一轮训练核心回归 **278 passed / 3 CUDA skipped**；前端 **72 tests**、lint、TypeScript、production build 通过，包含本机实际 MPS 运算。浏览器已完成 TOML 导入、12 步 MPS 训练、初始/周期预览、权重下载、从第 6 步续训至第 12 步；60 个最终权重张量逐位相同。
- 新完整状态为 format 2，保存原始训练参数与模型资产身份；新数据指纹包含 caption / mask / 验证数据。**旧版完整状态可能不兼容**，不得跳过检查强行恢复。已有权重可通过 `adapter.resume_weights` 热启动新训练；不要删除旧状态或数据。
- Schedule-Free 模式、TensorBoard、W&B sink 和初始采样已接通；真实 Schedule-Free/TensorBoard 已测，W&B 仅模拟契约测试。`optimizer.fused_backward=true` 当前显式拒绝，尚未实现。
- MPS 使用 FP32、不启用 autocast。系统仪表盘显示统一内存；训练指标显示当前 PyTorch 分配量。CUDA 的训练指标是 PyTorch 分配峰值，两者不等同。
- 队列按设备独占运行多个单设备任务，**未实现 DDP**。显存估算用于准入，可关闭 `memory_admission`；估算不是容量保证。

## 3. 仓库布局

```
ypuddin/            后端包本体
  adapters/         LoKr/LoRA/LoHa/Full/DoRA、注入、目标规则、IO/格式转换
  models/           ModelFamily 协议；anima/（含 vendor/）、krea2/（含 vendor/）、toy/
  data/             数据集注册、分桶、caption 增强、内容哈希缓存
  train/            训练循环、暂停/恢复、验证、采样预览
  memory/           block swap、fp8 与激活卸载
  server/           FastAPI：项目/任务/数据集/模型权重/SSE 事件
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
- **暂停/恢复**：任意步边界存全套状态（适配器/优化器/调度器/采样器位置/RNG/EMA），在相同资产、数据与兼容配置/运行环境下，恢复后与不间断训练**逐位一致**（CPU/MPS 回归证明；CUDA 待验收）。这是相对三个参考项目的核心优势，别破坏。
- **缓存**：latent/文本缓存键 = 内容哈希（图片内容 × 桶 × 编码器指纹 × 翻转），跨任务共享；`POST /jobs` 默认把 `dataset.cache_dir` 指到项目共享缓存目录。文本缓存存的是**增强后**的 caption 变体（有界、确定性），所以卸载文本编码器后 shuffle/tag_dropout 仍可用。
- **LoKr 自研而非依赖 LyCORIS**：LyCORIS 4.0.0 的 `merge_to`/`get_diff_weight` 在 `alpha≠rank` 且 w2 低秩时把 scale 乘两次（实测，合并结果错一半）。我们的实现与之**文件格式兼容已实测**（`tests/unit/test_lycoris_compat.py` 用真 LyCORIS 加载我们的文件，输出逐位一致；需 `pip install lycoris-lora` 才跑，否则跳过）。细节：`docs/design/02-adapters-lokr.md` §7。
- **Block swap**：前后向双钩子 + 推迟释放（修掉了"块输入无梯度时 backward hook 提前触发"）；开/关 swap 结果逐位一致（有测试）。
- **配置**：分组 pydantic 模型 → JSON Schema（带 x-ui 提示）→ 前端零手写表单；新增配置项的完整链路见 00 架构文档。
- **部署**：包源镜像优先（中科大→清华→阿里→官方兜底，逐源回退）；torch CUDA 版本按显卡计算能力+驱动选（RTX 50 系强制 cu128）；uv 缓存与项目跨盘时自动 `UV_LINK_MODE=copy`。`studio.bat` 必须纯 ASCII + CRLF（.gitattributes 强制），不要往里写中文或 chcp。

## 5. 测试与验证体系

- `venv/bin/python -m pytest tests/ -q`（全量）；前端 `cd frontend && npm run lint && npm run test && npm run build && npm audit`。
- 四条铁律级测试：暂停/恢复逐位一致、swap 开/关逐位一致、`forward_bypass ≡ merged ≡ base+F.linear(x,ΔW)`（含 alpha≠rank）、LyCORIS 交叉加载。
- Anima/Krea2 各有一个 e2e：用**缩小版真实架构**组件在 CPU 跑完整链路（加载→文本→缓存→训练→验证→采样→导出→ComfyUI 键转换→合并回带前缀底模），fp8_scaled 底模也在其列。
- 前端测试用 MSW mock + 真实后端截图验收（`frontend/screenshots/`）。

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
npm run dev                          # MSW mock 模式（含本地 mock SSE：曲线自增、采样进度、日志流）
VITE_USE_MOCK=false npm run dev      # 直连真实后端（vite 代理 /api → 127.0.0.1:8765）
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
