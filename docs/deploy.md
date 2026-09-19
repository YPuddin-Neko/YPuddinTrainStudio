# 部署与运行指南

面向要在自己机器（本地工作站或租用的 GPU 机器）上跑 YPuddin Train Studio 的人。开发者相关的内容（测试、导出 API 文档）在 `docs/design/03-status.md`。

## 1. 系统要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Linux、Windows、macOS；macOS 是否启用 MPS 取决于硬件、系统与 PyTorch，先运行 `doctor` 检查。各平台完整权重与全新环境验收仍待完成 |
| Python | 3.10 – 3.12（启动入口会自动查找；装了 [uv](https://docs.astral.sh/uv/) 时可自动下载 3.12） |
| GPU / 内存 | CUDA 训练需要匹配的 NVIDIA 驱动与 PyTorch。MPS 当前按 FP32 执行，预算使用统一内存，Block Swap 不会等额释放物理内存。需求随模型、精度、分辨率、优化器和缓存方式变化；尚无经过完整权重实测的最低容量保证 |
| Node.js | Node 20.x 至少 20.19，或 22.12+；须满足当前 Vite 的 engines，Node 18/21 不支持。仅**构建 Web 界面**时需要，没有 Node 仍可使用 CLI 和 API |
| 磁盘 | 为 PyTorch 和模型依赖预留数 GB；Anima 权重约 4 GB（DiT）+ 1.2 GB（Qwen3-0.6B）+ 0.25 GB（VAE）。另外预留数据、latent/文本缓存、采样图与完整训练断点空间，大小随配置变化 |
| 网络 | 首次安装需要下载依赖（PyTorch 约 2.5 GB）。默认镜像优先：中科大 → 清华 → 阿里 → 官方兜底（某个源缺包/报错/探测不通就自动换下一个）；境外网络可用 `--index=official` 官方优先 |

## 2. 一键启动（推荐）

```bash
git clone <本仓库> YPuddinTrainStudio && cd YPuddinTrainStudio/xiangmuyuanma
studio-windows-cuda.bat      # Windows + NVIDIA CUDA，x86_64（双击或在 PowerShell 里 .\studio-windows-cuda.bat）
studio-cpu.bat               # Windows 仅 CPU，x86_64 / arm64
./studio-linux-cuda.sh       # Linux + NVIDIA CUDA，x86_64
./studio-linux-dtk.sh        # Linux + 海光 DTK，x86_64
./studio-macos.command       # macOS Apple 芯片（MPS），arm64
./studio-cpu.sh              # Linux / macOS 仅 CPU，x86_64 / arm64
```

首次部署会按平台创建独立的 `environment/<平台>/venv`，再安装对应 PyTorch、训练依赖并构建前端。CUDA、CPU、macOS MPS 的依赖环境不共用，各环境只能用上面对应的入口启动；共用阶段在 `scripts/launch.sh` / `scripts/launch.bat`，不是入口。下文用 `<启动入口>` 指代你所在环境的那一个。三个加速入口只在 x86_64 上验证过，在 arm64 上启动会直接拒绝并提示改用 CPU 入口；MPS 入口要求 Apple Silicon。安装标记记录架构，同一个环境目录不会被两种架构共用。直接运行共用阶段（不带 `--profile`）会落到根目录 `venv` 的旧部署，不搬移或重建。平台目录和 Torch 切换关系见 [环境说明](ENVIRONMENT_LIFECYCLE_2026-09-14.md)。

CUDA 环境安装 `ypuddin[models,server,optim,logging,nvidia]`，CPU/MPS 不安装 NVIDIA 依赖，也不自动安装注意力扩展。初始地址为 `http://127.0.0.1:8765/`；保存过 host/port 设置后，下次启动使用保存值，命令行参数优先。后续运行只对选中环境增量补齐依赖，保留已有 Torch/CUDA/NumPy 原生栈。

安装完成后，启动器会校验 `venv` 内的独立安装信息，再自动删除根目录的 `ypuddin.egg-info` 构建副本。已有部署更新代码后正常启动即可清理旧残留，无需删除环境或数据。即使依赖安装被跳过，也会执行此清理；Windows 文件被占用时会提示并在下次启动重试。运行所需的 `venv` 内 `.dist-info` 保留，旧式安装先更新为现代 editable 安装再清理。

常用参数（`.sh` 与 `.bat` 一致）：

| 参数 | 作用 |
|---|---|
| `--port 8800` / `--host 0.0.0.0` / `--data-root /data/studio` | 覆盖服务端口 / 绑定地址 / 数据目录。初始默认 `127.0.0.1`、`8765`、`./studio_data`；host/port 可从设置读取，data-root 始终由本次启动参数决定 |
| `--torch=cu128` | 首次安装或 `--reinstall` 时选择 PyTorch 来源：`cu128` `cu126` `cu124` `cu118` `cpu`。默认 `auto`：脚本识别到 Blackwell 时选 cu128 并检查驱动；其余按驱动主版本 ≥570→cu128、≥560→cu126、≥550→cu124、≥450→cu118，否则 cpu。macOS 自动使用 PyPI 的 CPU/MPS 轮子。这是安装选择规则，安装后仍应通过 `doctor` 和真实 smoke 验证 |
| `--index=auto\|cn\|official` | 包源。`auto` / `cn`（默认）：镜像优先——中科大 → 清华 → 阿里 → 官方兜底，某个源缺包或报错就自动换下一个，探测不通的源先排到后面；CUDA 轮子走阿里 → 上交 → 官方。`official`：普通依赖官方优先、镜像兜底，CUDA 轮子使用官方索引。`--mirror` 等价于 `--index=cn` |
| `--profile=legacy` | 明确使用旧的根目录 `venv`；平台专用入口默认使用独立环境 |
| `--reinstall` | 只重建当前入口选中的基础环境；其他平台、服务数据、模型、数据集和准备好的 Torch 环境保留 |
| `--no-browser` / `--no-frontend` | 不自动开浏览器 / 不构建前端（只要 API） |

子命令：

| 命令 | 作用 |
|---|---|
| `<启动入口>` 或 `<启动入口> run` | 安装/更新 + 构建 + 启动服务 |
| `<启动入口> doctor` | 打印本机情况：Python、torch/CUDA/GPU、可选依赖、Node、前端构建状态。**排障先跑这个** |
| `<启动入口> smoke --set model.dit_path=… --set model.text_encoder_path=… --set model.vae_path=…` | 真实跑 3 步训练 + 出一张预览 + 保存/回读 LoRA，输出报告 `outputs/smoke/smoke-report.json`（见 §6） |
| `<启动入口> build` | 只构建前端 |
| `<启动入口> dev` | 后端 + Vite 热更新前端，默认浏览器开 `http://127.0.0.1:3000/`；自定义后端 `--port` 会传给前端代理，`--fe-port` 可改开发前端端口 |
| `<启动入口> test` | 跑后端 pytest（有 Node 时再跑前端 vitest） |
| `<启动入口> shell` | 打印如何激活 `venv`（之后可直接用 `ypuddin …` 命令） |

## 3. 手动安装（不用脚本时）

以下以 Linux 上安装 CUDA 版 PyTorch 为例；具体 CUDA 来源按驱动与显卡选择：

```bash
cd xiangmuyuanma
uv venv --python 3.12 venv                    # 或 python3.12 -m venv venv
# CUDA 示例：先确认驱动和显卡适用 cu128，再安装
uv pip install --python venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
#   国内镜像（任选其一）：
#   uv pip install --python venv/bin/python torch --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu128
#   uv pip install --python venv/bin/python --no-index --no-deps --find-links https://mirrors.aliyun.com/pytorch-wheels/cu128 torch \
#     && uv pip install --python venv/bin/python torch --index-url https://mirrors.ustc.edu.cn/pypi/simple   # 再补依赖
#   其余依赖可加 --index-url https://mirrors.ustc.edu.cn/pypi/simple（或 pypi.tuna.tsinghua.edu.cn/simple、mirrors.aliyun.com/pypi/simple）
uv pip install --python venv/bin/python -e ".[models,server,optim,logging]"           # 训练 + 服务
uv pip install --python venv/bin/python -e ".[cuda,nvidia]"              # CUDA 可选：bitsandbytes 8-bit、Prodigy 等
uv pip install --python venv/bin/python sageattention                   # 可选：仅无梯度采样使用 Sage，训练反向保持 SDPA
cd frontend && npm ci && npm run build && cd ..                          # 可选：Web 界面
venv/bin/ypuddin serve --host 127.0.0.1 --port 8765 --data-root studio_data
```

macOS 将上面的 PyTorch 安装行改为 `uv pip install --python venv/bin/python torch`，不要安装 CUDA 专属依赖。Windows 将 `venv/bin/python` 换成 `venv\Scripts\python.exe`，其他可执行文件也使用 `Scripts` 下对应路径。pip 用户把 `uv pip install --python venv/bin/python` 换成 `venv/bin/pip install` 即可。extras 含义：

| extra | 内容 |
|---|---|
| `models` | transformers / huggingface-hub / sentencepiece —— 加载 Qwen3 文本编码器与分词器（Anima 必需） |
| `server` | fastapi / uvicorn / psutil —— Web 服务（只用 CLI 训练可不装） |
| `cuda` | bitsandbytes（`optimizer.type = "adamw8bit"`），可选 |
| `nvidia` | nvidia-ml-py（NVIDIA GPU 监控，启动器按平台选择） |
| `optim` | schedulefree、lion-pytorch、prodigyopt、prodigy-plus-schedule-free、pytorch-optimizer |
| `logging` | tensorboard；前端实时日志不依赖云端服务 |
| `dev` | pytest、ruff、httpx（开发） |

启动器默认安装本地日志和常用优化器依赖，无需从环境页逐项安装。W&B 不再提供安装与前端入口，历史配置仍可读取。`optimizer.fused_backward = true` 尚未实现，配置会明确拒绝。

直接运行 `pip/uv pip install -e` 时，setuptools 仍可能生成根目录 `ypuddin.egg-info`；上述自动校验与清理由启动入口执行，不修改包管理器本身的构建行为。

## 4. 目录与数据

```
xiangmuyuanma/
├── venv/                旧部署环境，共用阶段在没有 --profile 时继续兼容
├── environment/<平台>/venv/  新部署各平台独立依赖
├── studio_data/          服务数据目录（--data-root 可改），包含：
│   ├── studio.db         SQLite：项目 / 版本 / 数据集 / 任务 / 产物 / 模型注册表
│   ├── settings.json     「系统设置」页保存的设置
│   ├── project/<project_id>/
│   │   └── v1/（新版本依次 v2、v3…）
│   │       ├── config.json   本版本配置草稿
│   │       ├── traindata/    本版本训练图片、caption 和 Mask 独立副本
│   │       ├── reg/          本版本正则数据，按批次分目录
│   │       ├── samples/<jid>/ 每次训练的采样图
│   │       ├── cache/        同版本任务共享的编码缓存
│   │       └── output/<jid>/ 每次训练：配置快照、事件、日志、权重、state-*/
│   ├── runs/             不属于任何项目的任务
│   ├── datasets/         数据集索引（图片哈希、尺寸、caption 路径）
│   ├── cache/            全局图片索引与无项目任务的共享缓存
│   ├── thumbs/           数据集页缩略图缓存（可删）
│   ├── presets/          用户保存的预设
│   └── models/           `POST /models/scan` 默认扫描的权重目录
├── outputs/              CLI 直接训练时的默认输出（配置里 checkpoint.output_dir）
└── frontend/dist/        构建好的前端（`ypuddin serve` 存在即托管）
```

- **备份**包括 `studio_data/`、自定义输出目录以及外部原始数据集的图片/caption/mask；模型权重也应另行保存或记录可重获来源。缓存可重建，完整训练断点不能用推理权重代替。
- 模型权重放哪都行，在界面「模型权重」页注册或在配置里填绝对路径；建议一个固定目录（如 `/models`），并在「系统设置」里把 `paths.models_dir` 指向它。
- 数据集是一个图片目录（递归），每张图旁边同名 `.txt` 是 caption；`.mask.png` 或 alpha 通道可做遮罩 loss。v0.4 的上传与服务器目录导入保存为当前版本的独立副本，caption/Mask 编辑修改该副本。旧迁移数据或高级 TOML 直接引用的外部目录仍使用原文件，不会仅因升级自动复制。
- 旧项目迁移成兼容 v1 时不移动文件：原 `projects/<pid>/config.json` 继续作为该版本草稿，历史任务读取原运行/缓存/断点路径；新任务采用版本目录。新项目采用手填 ID 和上述 vN 目录；既有 projects 布局不移动。详见 [v0.5.1 目录说明](UI_SIMPLIFICATION_2026-09-12.md)。

### 路径设置何时生效

`paths.cache_dir` 与 `paths.output_mode=custom` 下的 `paths.output_dir` 影响**新建任务**；默认保留上面的版本目录结构。自定义缓存根同时用于后续生成的扫描索引（`service/index`）、缩略图（`service/thumbnails`）和按平台隔离的软件包缓存（`packages/<平台>`）；启动器的 pip/uv 下载也使用这一设置，旧文件不会自动迁移。新项目自定义输出按 `<项目 ID>/vN/<任务 ID>` 分目录，无项目任务使用共享缓存或独立运行目录。旧任务保存自己的配置和运行路径，修改设置不会搬迁旧缓存、权重或断点。`paths.models_dir` 是默认模型扫描目录。相对数据/模型路径在服务接收任务时按服务工作目录解析，建议使用绝对路径。

“存储路径 → 基础环境目录”对应 `paths.bootstrap_env_dir`。留空保留默认环境位置；填写后，下次从启动脚本启动时在 `<基础环境目录>/<平台>/venv` 创建或复用该平台环境。也可用 `<启动入口> --env-root <目录>` 临时覆盖。已有环境不会搬迁或删除，新目录首次启动需要安装依赖；不接受未经本启动器登记的已有环境，也不接受符号链接或目录联接。这个设置只改变基础 Python 环境，界面内准备的其他 PyTorch 版本仍归所属数据根的 `environment/<平台>/runtimes` 管理。


`server.host` / `server.port` 保存后在下次启动生效，`--host` / `--port` 优先覆盖保存值。`paths.data_root` 展示本次启动的数据目录；设置接口拒绝把它改成另一目录。切换数据目录应先迁移所需数据，再用 `--data-root` 启动，不能靠设置页完成迁移。

### 数据布局与缓存

各数据源可单独配置分辨率；分桶集合包含全局与各源分辨率。显式验证源拥有独立源索引，并按图片内容排除训练集内的重复图；Plan 与训练使用同一套切分和步数计算。无图、切分后训练集为空等情况在预检时返回字段位置和原因。

latent 缓存使用图片内容、桶尺寸、预处理与实际 VAE 指纹；mask 在读取样本时独立处理，先缓存后开 mask、修改 mask 都不会复用旧遮罩，也无需为此重编码 VAE。文本缓存包含实际编码器和 tokenizer 指纹。指纹覆盖所提供单文件的字节内容、所传 HF 目录内的非隐藏文件，以及实际使用的内置 tokenizer；哈希通过文件元数据索引复用，缓存文件以唯一临时文件原子替换。caption/mask 内容还会进入训练数据指纹，用来防止数据已变却继续旧训练状态。

## 5. Anima 权重

需要三个文件（配置字段 → 来源）：

| 字段 | 文件 | 说明 |
|---|---|---|
| `model.dit_path` | `anima-base-*.safetensors`（或 ComfyUI 格式 `model.diffusion_model.` 前缀的文件） | DiT 主干 + LLM adapter，约 4 GB bf16 |
| `model.text_encoder_path` | Qwen3-0.6B（HF 目录，或单文件 safetensors） | 文本编码器；单文件时分词器用内置副本 |
| `model.vae_path` | `qwen_image_vae.safetensors` | Qwen-Image VAE（16 通道 / 8 倍） |

训练得到的 `.safetensors` 使用 kohya 键名（`lora_unet_*`）；`ypuddin convert --to comfyui` 提供对应转换，LoRA 可转成 PEFT 键名。键转换和读写往返已有测试，完整权重训练后在实际 ComfyUI 中加载和验证效果仍待验收。

### 5.1 Krea 2 权重（`model.family = "krea2"`）

Krea 2 是 12.9B 参数的单流 MMDiT，文本编码器是 Qwen3-VL-4B-Instruct，VAE 与 Anima 相同（Qwen-Image VAE）：

| 字段 | 文件 | 说明 |
|---|---|---|
| `model.dit_path` | `krea2_raw_bf16.safetensors`（官方 raw 权重，约 26 GB）或 Comfy-Org 的 `krea2_fp8_scaled.safetensors`（约 13 GB） | 键名可带或不带 `model.diffusion_model.` 前缀；CUDA 的 fp8_scaled 路径保留冻结 fp8 权重和文件 scale，执行时仍可能产生临时反量化张量，峰值内存需实测 |
| `model.text_encoder_path` | `Qwen/Qwen3-VL-4B-Instruct` HF 目录（推荐），或 ComfyUI 的单文件 `qwen_3vl_4b*.safetensors`（bf16 或 fp8_scaled 均可） | 只加载语言模型部分（视觉塔不参与）；单文件时把模型的 `config.json` 放在同一目录（没有则按 4B 几何假定），分词器缺失时用内置的 Qwen3 分词器 |
| `model.vae_path` | `qwen_image_vae.safetensors` | 与 Anima 使用相同 VAE 架构；仅当实际权重指纹、预处理和缓存精度等条件一致时可复用 latent 缓存 |

Krea 2 的文本编码只支持 **cached 模式**（`dataset.text_encoding = "auto"` 会自动选 cached）。启用图片缓存时，程序先用 VAE 缓存图片并卸载 VAE，再缓存描述和采样提示词并卸载文本编码器，最后才加载 DiT 主模型的真实权重。全部缓存命中时无需加载对应编码器；“只构建缓存”不会加载 DiT 权重张量。训练预览会按需重新加载 VAE，结束后再次卸载。

CUDA 分块换出任务还会检查当前空闲显存：能够放下所选文件的实际权重，并额外留下 **2 GiB** 时，先临时上传完整 DiT，释放 CPU 文件映射后，再建立需要换出的主机缓存；余量不足时保持原 CPU 加载路径。它不会修改换出块数或训练参数。FP8 文件的缩放值保留原值，同时解除过去会拖住整份文件映射的多余引用。CPU RAM 与 GPU 显存仍需分别留出空间，准备阶段的搬运峰值也要观察，不能仅靠降低分辨率判断是否够用。加载顺序、暂存条件和 FP8 修复详见 [Krea 2 缓存与内存说明](KREA2_MEMORY.md)。

文本缓存默认每批最多编码 16 条文字，与训练 batch 无关。内置预设 `krea2-lokr-default` / `krea2-lora-32` 提供配置起点，应先运行 Plan 和短程 smoke；分阶段加载与显存估算都不是容量保证。

下面是 CUDA fp8 路径的自检示例；MPS 当前强制 FP32、关闭 autocast，并用系统统一内存作保守预算。CPU/MPS 不支持显式 `memory.base_precision=fp8_*`、SageAttention 或 8-bit 优化器；不能把 CPU 与 MPS 之间的 Block Swap 当作独立显存和内存之间的等额腾挪。

```bash
<启动入口> smoke \
  --set model.family=krea2 \
  --set model.dit_path=/models/krea2_fp8_scaled.safetensors \
  --set model.text_encoder_path=/models/Qwen3-VL-4B-Instruct \
  --set model.vae_path=/models/qwen_image_vae.safetensors \
  --set memory.base_precision=fp8_e4m3 --set memory.activation_checkpointing=block \
  --set objective.timestep_sampling=resolution_shift --resolution 512
```

## 6. 第一次在 GPU 机器上验证

```bash
<启动入口> smoke \
  --set model.dit_path=/models/anima-base-v1.0.safetensors \
  --set model.text_encoder_path=/models/Qwen3-0.6B-Base \
  --set model.vae_path=/models/qwen_image_vae.safetensors \
  --set adapter.algo=lokr --set adapter.rank=full --set adapter.factor=8 \
  --set memory.activation_checkpointing=block
# 显存紧张：追加 --set memory.blocks_to_swap=10 --set dataset.text_encoding=cached --resolution 512
```

它会用真实训练器跑 3 步、出一张 512 预览、保存并回读适配器，最后打印每项检查的通过情况、耗时、可用的设备内存指标，并写 `outputs/smoke/smoke-report.json`（失败时含完整 traceback）。这是本机配置的短程验收入口：通过表示这组加载、训练、预览与保存流程能够完成，长时间训练稳定性、其他分辨率和图像质量仍需分别验证。历史自动化修复记录见 [2026-09-11 修复报告](FIX_REPORT_2026-09-11.md)。

MPS 可先用 `--device mps --set model.dtype=fp32 --set loop.mixed_precision=no` 自检。默认配置中的 bf16 在 MPS 上会告警并按 FP32 执行；这条路径仍需完整模型的耗时、内存和图像质量验证。

## 7. 作为常驻服务

### Linux（systemd）

`/etc/systemd/system/ypuddin.service`：

```ini
[Unit]
Description=YPuddin Train Studio
After=network.target

[Service]
User=trainer
WorkingDirectory=/opt/YPuddinTrainStudio/xiangmuyuanma
Environment=PYTHONUTF8=1
ExecStart=/opt/YPuddinTrainStudio/xiangmuyuanma/venv/bin/ypuddin serve --host 127.0.0.1 --port 8765 --data-root /data/studio
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now ypuddin
journalctl -u ypuddin -f
```

先手动跑一次 `<启动入口> --no-browser` 完成安装和前端构建，再交给 systemd。

### Windows

「任务计划程序」→ 创建任务 → 触发器「登录时」→ 操作：程序 `C:\...\xiangmuyuanma\venv\Scripts\ypuddin.exe`，参数 `serve --host 127.0.0.1 --port 8765 --data-root D:\studio_data`，起始于 `C:\...\xiangmuyuanma`。或者直接把 `studio-windows-cuda.bat --no-browser` 的快捷方式放进启动文件夹。

### 远程访问与安全

服务**没有登录认证**，默认只监听 `127.0.0.1`。要从别的机器访问，二选一：

- SSH 端口转发（推荐）：`ssh -L 8765:127.0.0.1:8765 user@gpu-box`，然后本机开 `http://127.0.0.1:8765/`。
- 放在 Nginx/Caddy 之类反向代理后面，由代理做 HTTP Basic Auth / OAuth，再把 `--host` 改成 `0.0.0.0`。**不要**把裸服务直接暴露到公网：文件浏览接口（`/api/fs/list`）与训练任务可以读写 `--data-root` 及配置里写到的任何路径。

SSE（`/api/events`）需要代理关闭响应缓冲（Nginx：`proxy_buffering off; proxy_read_timeout 1h;`）。

### 队列、暂停与恢复

任务创建前会预检数据、配置、设备能力和内存估算；错误包含字段位置。队列按优先级和创建时间调度，定时任务到时进入队列。`max_concurrent` 限制总进程数，每个 CUDA/MPS 设备同一时间只分配一个任务；多设备可执行不同任务，当前不支持单任务 torchrun/DDP 多卡协同训练。

队列设置 `memory_admission` 默认开启：估算峰值超过当前设备可用内存的 95% 时任务等待设备，任务阶段会显示等待原因。可通过 `PUT /api/queue/settings` 设置 `{"memory_admission": false}` 关闭这项估算门禁；设备独占仍然有效。Plan 是启发式估算，包含统一内存的 MPS FP32 预算，不能保证任务不会 OOM。

训练阶段的暂停在优化器步边界保存完整断点；索引/缓存等准备阶段可在进度检查处响应暂停或停止，已完成缓存保留，恢复时继续准备，尚未创建的训练状态不会伪装成可用断点。进程异常退出后任务记录为失败，可从最后一个完整断点恢复。服务正常关闭会尝试让运行任务保存状态，但超时强杀或系统故障不能保证最后一步已落盘。

Web 配置页动态读取后端 Schema，支持 TOML 导入导出（API 也支持 JSON）。任务页通过 step、phase、checkpoint 等 SSE 事件更新进度和产物，可下载权重并从完整断点续训。EMA 开启时另存 EMA 推理权重；下载权重文件不等于下载完整训练状态。SSE 历史回放仅为有界内存记录，刷新或重启后以 HTTP 查询到的任务和指标为准。

## 8. 更新

```bash
git pull
<启动入口>          # 依赖签名（pyproject.toml）或前端源码变了会自动重装 / 重建
```

更新前备份服务数据与外部输出目录。v0.4 启动包含事务式版本迁移：补充 `project_versions` 与归属列，把旧项目/数据源/任务关联到兼容 v1，产物沿原任务归属；不搬迁文件或重写任务快照。这是明确的兼容迁移，不是任意版本都适用的通用数据迁移工具；更换版本同时阅读 `HANDOVER.md` 与 [当前报告](UI_PIPELINE_2026-09-11.md)。

本轮完整训练状态升级为 **state v2**，另存原始可训练参数、scalar、优化器/调度器、采样器、RNG（含 DataLoader 独立生成器）与 EMA。推理 `.safetensors` 的用途仍是加载/分发模型适配器。

“可复现训练”默认关闭（`loop.deterministic=false`），保留现有训练的计算方式。需要对比重复实验时可手动开启：在模型加载前启用 PyTorch 严格确定性算法和确定性卷积，并关闭卷积算法性能试选；NVIDIA CUDA 未指定 cuBLAS workspace 时使用 `:4096:8`。它可能降低速度或增加工作内存，遇到不支持确定性计算的算子会直接停止，不会静默降级。DTK 的原生 SDPA 在此模式下禁用融合实现、使用数学实现，可能明显变慢并增加显存需求；显式选择 FlashAttention/xFormers 时仍走相应扩展，不能把此策略当作对它们的验证。关闭此项后会恢复本进程原来的 SDPA 后端选择。该设置减少相同设备、软件、数据与配置下的重复计算差异，不保证跨设备、跨版本或第三方算子逐位一致；真实模型验收仍需分别比较权重、训练状态和采样结果。

完整状态会记录此计算设置，恢复时不能更改。旧版本状态未记录此项，需显式设置 `loop.deterministic=false` 恢复，以沿用历史计算方式；这不会抹除其他参数的配置指纹变化。若要为旧任务开启确定性计算，应从已有权重新建训练。关闭此项时，相同随机种子和完整 RNG 状态也不能保证 GPU 浮点结果逐位一致。

新版的数据指纹和排序语义与旧版不同。旧格式文件即使能被解析，也不能跳过指纹或参数兼容检查继续宣称精确续训；发生旧指纹不兼容时应使用旧版本完成原任务，或用 `adapter.resume_weights` 从已有适配器权重启动**新训练**，重新建立优化器、调度器和步数。修改 caption/mask 后的数据指纹变化也会阻止沿用旧数据状态；确需保持原训练应恢复当时的数据内容。新版本对内容相同的数据移动/改名不依赖绝对路径作为身份。

## 9. 排障

| 现象 | 处理 |
|---|---|
| Windows 双击 `studio-windows-cuda.bat` 后窗口一直没有任何输出 | `.bat` 必须是 CRLF 换行：编辑器另存过或 `core.autocrlf=false` 的克隆会把换行变成 LF，cmd.exe 会解析错乱。仓库 `.gitattributes` 已强制 `*.bat` 为 CRLF——`git pull` 后 `git checkout -- studio-windows-cuda.bat scripts/launch.bat`（或删掉重新 `git checkout`）即可；不要用手动保存的副本。也可以先开一个 `cmd` 窗口，手动运行 `python scripts\bootstrap.py doctor` 看真实报错 |
| `doctor` 显示 `cuda_available: false` 但机器有 NVIDIA 卡 | 驱动太旧或装了 CPU 版 torch：`nvidia-smi` 看驱动版本，`<启动入口> --reinstall --torch=cu124`（驱动 ≥550）或 `cu118` |
| Windows 上 `ModuleNotFoundError: bitsandbytes` / 训练启动就失败 | 配置里 `optimizer.type` 改回 `adamw`，或在 CUDA 环境安装对应依赖：`.\venv\Scripts\pip install "bitsandbytes>=0.43"` |
| 页面能开但任务列表 / 数据集为空、控制台 404 | 前端是旧构建：`<启动入口> build`；开发态的 mock 数据请用 `dev` 模式而不是 `run` |
| 端口被占用 | `--port 8800`，或找出占用者：`lsof -i :8765`（Linux/macOS）、`netstat -ano \| findstr 8765`（Windows） |
| 前端构建提示 Node 不支持 | 升级到 Node 20.19+（20.x）或 22.12+，确认 `node --version`，再运行 `<启动入口> build` |
| 首次安装很慢 / 超时 | 默认镜像优先（中科大 → 清华 → 阿里 → 官方兜底）；境外网络用 `--index=official` 官方优先。PyTorch 轮子约 2.5 GB；某个镜像缺最新版会自动回退到下一个源，日志里有 `source … failed, trying the next one` |
| RTX 50 系报 `no kernel image is available for execution on the device` / `sm_120 is not compatible` | 装到了旧 CUDA 构建：`<启动入口> --reinstall --torch=cu128`；驱动需 ≥ 570。`<启动入口> doctor` 会对比显卡计算能力与 torch 内核列表并直接给出警告 |
| Apple GPU 没有被使用 | 先看 `doctor` 的 MPS 可用性并检查 PyTorch/硬件，再用 `--device mps` 自检。当前 MPS 执行 FP32，不能启用显式 fp8、SageAttention 或 8-bit 优化器 |
| 训练 OOM | 降分辨率与 batch，启用逐块激活重算、cached 文本；CUDA 可进一步尝试适当数量的 Block Swap、已安装且支持的 8-bit 优化器。MPS 的 CPU 换出不等于释放统一物理内存。Plan 可提供估算，实际峰值以设备自检为准 |
| 任务一直等待设备 | 查看任务等待原因：设备可能被另一任务独占，或估算峰值超过可用内存的 95%。关闭 `memory_admission` 仅跳过内存估算门禁，不解除单设备独占 |
| 恢复提示 dataset fingerprint 不一致 | 检查图片、caption、mask、数据源与验证设置是否变化；旧版指纹不能与新版精确续训混用，见 §8 |
| `fused_backward is not implemented` | 将 `optimizer.fused_backward` 设为 `false`，当前不支持此功能 |
| 采样阶段看起来"卡住" | 任务详情页 header 有「生成预览 第 k/n 张 · 步 x/y」进度；分辩率填错（如 10240）会被配置校验拒绝 |
| 任务状态 `failed`，error 是 `process exited with code …` | 打开任务详情「Logs」看最后几行；`studio_data/projects/<pid>/runs/<jid>/run.log` 是完整日志 |
| 重建 Python 环境 | 在对应平台入口后使用 `--reinstall`，只重建该入口的基础环境；其他环境和服务数据不动 |

## 10. CLI 速查（激活 `venv` 后）

```bash
ypuddin plan     config.toml                 # 本机预检 + 实际切分步数 / 分桶 / 内存估算
ypuddin plan     config.toml --device cuda    # 按指定设备检查；不会执行训练
ypuddin train    config.toml                 # 训练（Ctrl+C = 暂停并保存断点；state-*/ 可 --set checkpoint.resume=…）
ypuddin cache    config.toml                 # 只预编码 latent / 文本缓存
ypuddin smoke    --set model.dit_path=…      # 机器自检
ypuddin inspect  lora.safetensors            # 看元数据与模块
ypuddin convert  lora.safetensors --to comfyui -o out.safetensors
ypuddin merge    --base anima-base.safetensors --adapter lora.safetensors --family anima -o merged.safetensors
ypuddin extract  --base a.safetensors --tuned b.safetensors --algo lokr -o diff-lokr.safetensors
ypuddin serve    --port 8765 --data-root studio_data
```

配置文件支持 TOML 与 JSON（`ypuddin schema` 打印带说明的 JSON Schema）；`--set a.b=c` 可覆盖字段，CLI 的 `--preset preset.toml` 叠加**预设文件**，Web 内置预设通过界面选择。配置校验/导入导出无需加载模型；Plan 还会检查数据和权重几何信息，CLI 默认检测本机设备，也可用 `--device` 指定目标。Python API 的 `plan(..., device=None)` 可只作离线预检，不启用实际设备门禁。最小 Anima 配置见 `docs/design/03-status.md`。


## 项目版本与设置升级（当前 v0.5.2）

先停止旧服务，在源码目录更新 Git 后重新运行本环境的 `<启动入口>`。启动器在重启时检查依赖与前端内容指纹，必要时补依赖/重建前端；已有项目、配置、模型文件保留。没有联网程序自更新或生产热更新。匹配的已构建前端无需 Node，需要重建时则必须满足 Node 版本要求。

项目内按版本进行“训练数据 → 模型准备 → 训练参数 → 任务与结果”。新版本可复制图片、caption、Mask 和验证源，也可只继承参数或使用默认空白配置；任务、采样与产物不复制。新导入目录会生成独立副本，原始目录保留；版本归档也不删除文件。任务仍保存配置快照而非独立图片快照，编辑旧版本数据前应先复制新版本，精确续训继续检查数据指纹。

“设置”从工作区打开为宽抽屉，包含“运行环境 / 模型权重 / 访问密钥 / 软件下载源 / 存储路径 / 界面与服务”六类。全局任务队列管理跨项目任务，训练产物在“项目 → 版本 → 训练结果”查看，采样按所属任务读取。开发前端默认访问真实服务，仅显式 `VITE_USE_MOCK=true` 启用演示模式。

运行环境页显示 Python、PyTorch、平台计算后端和 xFormers/FlashAttention，旧 Sage 配置仅说明采样用途；NVML、TensorBoard 与 Schedule-Free 等普通依赖由启动器按平台补齐，W&B 和 WD14 不再提供入口。扩展操作先生成 wheel / 版本变更计划，确认应用后显示安装日志；进行中和失败信息保留，普通包清单与常驻历史不再占据默认页面。基础 Torch/CUDA/NumPy 受保护；没有兼容预编译 wheel 时显示原因，不隐式启动源码编译。依赖实际修改后需重启，维护状态在重启前阻止训练与预缓存任务启动。

Windows 上“包已安装”不等于它的 CUDA 内核支持当前显卡。例如 RTX 5070 Ti（SM120）使用官方 xFormers 0.0.35 CUDA 12.8 wheel 时，可能遇到 `No operator found` / `GPU ... too new`。此时环境页保留安装版本，分别报告导入状态和实际正反向检测结果；只有 SDPA 自检也通过才提示可继续使用 SDPA。不要为此直接降级受保护的 PyTorch。

xFormers 可调用单独安装的 FlashAttention 2。Windows 可在运行环境中上传同时匹配 Python、PyTorch、CUDA 和 GPU 架构的 FA2 wheel，再执行安装和重新检测。社区预编译 wheel 不属于 FlashAttention 官方 Windows 发布包；应核对发布来源与 SHA256，并以当前机器上的正反向检测为准。没有可用 wheel 时继续使用已通过检测的 SDPA，无须在日常使用的电脑上启动源码编译。

本机使用 Apple GPU，CUDA 扩展安装及真实 NVIDIA 功率读数未在本机验证。需要 CUDA 的选项显示原因并禁用；Apple MPS 不提供功率数据，不显示虚构的瓦数。

### 数据准备与中央访问密钥

停止训练器服务后更新源码，再运行本环境的 `<启动入口>`。启动器沿用依赖与前端源码指纹检测：依赖变化时补安装，前端过期时重建；已启动的旧 Python 进程仍需重启。保留 `studio_data`、已有模型目录和自定义配置，更新包不包含这些运行数据。

新增原生分辨率是可选模式，旧分桶配置不会自动切换。到「训练参数 → 数据与分桶」选择原生模式并查看实际尺寸分组、缩小图片数和训练步数。更换分辨率模式或修改素材后重新检查准备状态。

自动打标已移除；在项目版本“标签查看”中选择图片读取已有标签，必要时进入逐图编辑器。正则数据在“正则图”步骤导入或生成，每批文件保存在本版本 reg 目录；生成/收集成功后自动登记为正则训练来源。

在“设置 → 访问密钥”保存/清除 Hugging Face、ModelScope、Danbooru 与 Gelbooru 凭据。原有模型令牌继续从运行数据根的 `secrets.json` 读取，无需重新填写；页面和 GET API 只显示配置状态，保存的账号与密钥不回显。正则任务默认使用对应站点已保存凭据，运行中的任务保留启动时凭据，清除仅影响后续任务。HF-Mirror 匿名下载；清除模型令牌也会禁用同来源环境变量/CLI 回退。

该文件是本机敏感配置，不是加密保险箱。更新时保留，分享源码或打包时排除。不要把它放进项目 TOML、日志、截图或公开问题报告。前端保存/清除错误显示在相应平台表单内，状态读取失败可在页顶重试。

“模型权重 → 准备模型”按模型族列出主模型、文本编码器和 VAE，可选择 Hugging Face 或魔搭官方来源。推荐文件下载时核验字节数、SHA-256 和组件类型，完成后才登记；重新下载从头开始并保留原校验。共享 VAE 可复用已有文件。本地候选第一次设为推荐默认时需要读取内容验 SHA，大文件会有等待时间，之后按文件状态复用校验；列表刷新不反复读取全部权重。受限仓库仍需在发布平台取得权限，自定义下载不自动具有官方推荐身份。

完整用法、设计来源与验证边界见 [v0.5.2 说明](UI_DESIGN_REVIEW_2026-09-12.md)，原有目录规则见 [v0.5.1 说明](UI_SIMPLIFICATION_2026-09-12.md)。
