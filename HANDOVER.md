# YPuddin Train Studio — 项目交接报告

> 写给接手本项目的模型/工程师。本文自洽：读完这一份 + 点开的几个文件，就能不需要前任任何上下文地继续开发。
> 日期：2026-09-11 · 仓库：`xiangmuyuanma/`（本地 git，61 个提交） · 报告由当时的构建代理（Claude/Kimi）撰写。

## 0. 先读这三个文件

1. 本文。
2. `docs/design/00-architecture.md` —— 架构契约（事件流、配置、模型族协议）。设计文档与实现**同步维护**，可信。
3. `docs/design/03-status.md` —— 逐组件状态表与运行方式。

## 1. 项目定位

一个桌面化的 diffusion **LoRA / LoKr 训练器**，目标是比 sd-scripts、diffusion-pipe、AnimaLoraStudio 做得更好（对它们的逐项短板分析见 `docs/reference/*.md`）。不是 GUI 套壳：训练核心、适配器、数据流水线、服务 API、CLI 全部自研。

- **模型族**：`anima`（Anima 2B，Cosmos-Predict2 DiT + Qwen3-0.6B + Qwen-Image VAE）与 `krea2`（Krea 2 Raw 12.9B 单流 MMDiT + Qwen3-VL-4B + 同款 VAE）是一等公民；另有 `toy` 族供 CPU 测试。新族 = 实现 `ModelFamily` 协议（`ypuddin/models/`）。
- **适配器**：LoKr（自研，与 LyCORIS 文件格式兼容）、LoRA、LoHa、Full、DoRA 包装。目标选择用 preset + 有序 rules（`ypuddin/adapters/rules.py`）。
- **形态**：Python 包 `ypuddin`（CLI + FastAPI 服务）+ `frontend/`（React/Vite 界面，可选）。一键脚本 `studio.sh` / `studio.bat`。
- 许可证 Apache-2.0（参考项目里 diffusion-pipe 与 AnimaLoraStudio 是 GPL——只读不抄；sd-scripts / musubi-tuner 是 Apache-2.0，vendor 的代码见 §7）。

## 2. 状态快照（数字均为本报告撰写时实测）

- 后端：Python 约 16.5k 行（`ypuddin/` + `scripts/`），测试约 4k 行，**pytest 全套通过**（211 个，CPU，约 100 秒）。
- 前端：`frontend/src` 约 10.4k 行，**lint 0 警告 / vitest 52 通过 / build 无警告 / npm audit 0 漏洞**。
- 里程碑：后端训练核心、Anima 族、Krea 2 族、服务 API、工具链全部完成；前端 FE-M1～M7（框架→数据集页→真实后端对接→Krea 2 接入）+ 全 UI 中文化完成。
- **最大缺口：两个模型族都没有在 GPU 上用官方权重跑过**（开发机是 Mac）。这是接手后的第一件事，见 §6。

## 3. 仓库布局

```
ypuddin/            后端包本体
  adapters/         LoKr/LoRA/LoHa/Full/DoRA、注入、目标规则、IO/格式转换
  models/           ModelFamily 协议；anima/（含 vendor/）、krea2/（含 vendor/）、toy/
  data/             数据集注册、分桶、caption 增强、内容哈希缓存
  trainer/          训练循环、暂停/恢复、block swap、验证、采样预览
  service/          FastAPI：项目/任务/数据集/模型权重/SSE 事件
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
- **暂停/恢复**：任意步边界存全套状态（适配器/优化器/调度器/采样器位置/RNG/EMA），恢复后与不间断训练**逐位一致**（有测试证明）。这是相对三个参考项目的核心优势，别破坏。
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

1. **GPU 真机验证（唯一关键路径）**。在有 N 卡的机器上：
   ```bash
   ./studio.sh          # 或 studio.bat；一键装环境起服务
   ./studio.sh smoke --set model.dit_path=<官方权重> --set model.text_encoder_path=... --set model.vae_path=...
   ```
   Anima 与 krea2 各跑一遍（Krea 2 的权重清单见 `docs/deploy.md` §5.1）。产出 `outputs/smoke/smoke-report.json`。最可能出问题：官方权重键名、Qwen3 单文件格式、bf16/fp8 显存行为——CPU 测试覆盖不到这些。
2. 真机数字出来后：block swap / unsloth 卸载 / sage / fp8 的收益实测，与 sd-scripts、diffusion-pipe 的基准对比（速度、显存、出图）。
3. 然后才轮到：多卡（采样器已按 rank 切分，训练器未包 DDP）、更多模型族、英文语言包长尾词条补全（约 95 个 key 目前靠代码内中文默认值兜底）。

## 7. 已知风险 / 坑

- **官方权重加载是最大未验证点**（兼容 `net.` / `model.diffusion_model.` / 裸键 / ComfyUI fp8_scaled 的代码都有，但只测过构造的键名）。
- LyCORIS 生态兼容只验证到"它的加载器读我们的文件"；ComfyUI 本体还没挂过我们训出的文件（同一套 kohya 键约定，风险低）。
- 文本管线对空 caption 已修（每行至少保留一个有效 token，防融合注意力 NaN）——改动时保持这个不变量。
- 前端 `JobListResponse` = openapi 的 `JobPage` 别名（分页信封），别再当成 `list[Job]`（出过一次 500 回归）。
- `docs/` 下 4 个参考分析文件在用户机器上被格式化工具碰过（纯格式 diff），如碍事可 `git checkout -- docs/reference/`。
- `.handoff/` 曾是本机多 agent 交接目录，已从 git 移除（`.gitignore`）；如果新工作流不需要，直接删除目录即可。

## 8. 第三方代码与许可证

- 本仓库 Apache-2.0。vendor 的代码（均 Apache-2.0，各目录有 `NOTICE.md` 记录来源 commit 与改动）：`ypuddin/models/anima/vendor/`（sd-scripts：Cosmos-Predict2 DiT、Qwen-Image VAE，已去掉 block swap 与 sd-scripts 依赖）、`ypuddin/models/krea2/vendor/`（musubi-tuner 8934cfb：SingleStreamDiT，同样处理）。
- 同级目录的四个参考仓库（AnimaLoraStudio/、diffusion-pipe/、sd-scripts/、LyCORIS/）**不属于本项目**，运行时不依赖；其中 diffusion-pipe 与 AnimaLoraStudio 是 GPL-3.0——可读可参考，不要把代码搬进本仓库。

## 9. 协作模式备注（可选继承）

此前前端由独立 agent 会话（Kimi）实现、本侧验收提交：它只改 `frontend/`，不 commit；验收方亲自跑 lint/test/build + 读源码 + 看真实后端截图后才提交。`.handoff/claude-to-kimi.md` / `frontend-status.md` 是当时的交接文件（已不在 git 内）。这个分工效果不错，但非必需——随新团队习惯调整。
