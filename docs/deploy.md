# 部署与运行指南

在本地工作站或租用的 GPU 机器上安装和运行 YPuddin Train Studio。测试、导出 API 文档等开发内容见 `docs/design/03-status.md`。

## 1. 系统要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Linux、Windows、macOS。macOS 的 MPS 需要 Apple 芯片，可运行 `doctor` 检查是否可用 |
| Python | 3.10 – 3.12（启动入口会自动查找；装了 [uv](https://docs.astral.sh/uv/) 时可自动下载 3.12） |
| GPU / 内存 | CUDA 训练需要匹配的 NVIDIA 驱动。MPS 按 FP32 执行，与系统共用统一内存，Block Swap 不能腾出物理内存。所需容量随模型、精度、分辨率、优化器和缓存方式变化，可先用 Plan 估算 |
| Node.js | 20.19+（20.x）或 22.12+，Node 18/21 不支持。仅**构建 Web 界面**时需要，没有 Node 仍可使用 CLI 和 API |
| 磁盘 | 为 PyTorch 和模型依赖预留数 GB；Anima 权重约 4 GB（DiT）+ 1.2 GB（Qwen3-0.6B）+ 0.25 GB（VAE）。另外预留数据、latent/文本缓存、采样图与完整训练断点空间，大小随配置变化 |
| 网络 | 首次安装需要下载依赖（PyTorch 约 2.5 GB）。默认依次使用中科大 → 清华 → 阿里 → 官方源，某个源安装失败时换下一个；可在设置中切换，境外网络可用 `--index=official` |

## 2. 一键启动（推荐）

```bash
git clone https://github.com/YPuddin-Neko/YPuddinTrainStudio.git
cd YPuddinTrainStudio
studio-windows-cuda.bat      # Windows + NVIDIA CUDA，x86_64（双击或在 PowerShell 里 .\studio-windows-cuda.bat）
studio-cpu.bat               # Windows 仅 CPU，x86_64 / arm64
./studio-linux-cuda.sh       # Linux + NVIDIA CUDA，x86_64
./studio-linux-dtk.sh        # Linux + 海光 DTK，x86_64
./studio-macos.command       # macOS Apple 芯片（MPS），arm64
./studio-cpu.sh              # Linux / macOS 仅 CPU，x86_64 / arm64
```

首次启动会为当前平台创建独立的 `environment/<平台>/venv`，安装对应的 PyTorch 和训练依赖，并构建前端。各平台环境互不共用，请始终使用对应的启动入口；下文用 `<启动入口>` 指代它。CUDA 与 DTK 入口仅支持 x86_64（arm64 请用 CPU 入口），MPS 入口需要 Apple 芯片。平台目录与 PyTorch 切换见 [环境说明](ENVIRONMENT_LIFECYCLE_2026-09-14.md)。

注意力加速扩展不会自动安装，可在“设置 → 运行环境”中安装；Apple 的 Metal FlashAttention 见 [Apple 注意力加速](METAL_ATTENTION.md)。服务默认地址为 `http://127.0.0.1:8123/`，在设置中保存的地址和端口下次启动生效，命令行参数优先。之后每次启动只补齐缺失的依赖，保留已安装的 PyTorch、CUDA 与 NumPy。

常用参数（`.sh` 与 `.bat` 一致）：

| 参数 | 作用 |
|---|---|
| `--port 8800` / `--host 0.0.0.0` / `--data-root /data/studio` | 指定服务端口 / 绑定地址 / 数据目录，默认 `8123`、`127.0.0.1`、`./studio_data`。在设置中修改过的地址、端口和数据目录下次启动生效，命令行指定的地址和端口优先 |
| `--torch=cu128` | 首次安装或 `--reinstall` 时选择 PyTorch 类型：`cu128` `cu126` `cu124` `cu118` `cpu`。默认 `auto`：RTX 50 系（Blackwell）选 cu128；其余按驱动主版本 ≥570→cu128、≥560→cu126、≥550→cu124、≥450→cu118，否则 cpu。macOS 使用 PyPI 的 PyTorch（含 MPS） |
| `--index=auto\|cn\|official` | 包源；未指定时使用设置中保存的下载源，没有保存时为 `auto`。`auto` / `cn`：普通依赖依次使用中科大 → 清华 → 阿里 → 官方，PyTorch 依次使用上交 → 阿里 → 官方，某个源安装失败时换下一个。`official`：普通依赖官方优先、镜像兜底，PyTorch 只用官方索引。`--mirror` 等价于 `--index=cn` |
| `--profile=legacy` | 使用根目录 `venv`（旧部署）；平台入口默认使用独立环境 |
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
cd YPuddinTrainStudio
uv venv --python 3.12 venv                    # 或 python3.12 -m venv venv
# CUDA 示例：先确认驱动和显卡适用 cu128，再安装
uv pip install --python venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
#   国内镜像（任选其一）：
#   uv pip install --python venv/bin/python torch torchvision --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu128
#   uv pip install --python venv/bin/python --no-index --no-deps --find-links https://mirrors.aliyun.com/pytorch-wheels/cu128 torch torchvision \
#     && uv pip install --python venv/bin/python torch torchvision --index-url https://mirrors.ustc.edu.cn/pypi/simple   # 再补依赖
#   其余依赖可加 --index-url https://mirrors.ustc.edu.cn/pypi/simple（或 pypi.tuna.tsinghua.edu.cn/simple、mirrors.aliyun.com/pypi/simple）
uv pip install --python venv/bin/python -e ".[models,server,optim,logging]"           # 训练 + 服务
uv pip install --python venv/bin/python -e ".[cuda,nvidia]"              # CUDA 可选：bitsandbytes 8-bit 优化器、NVIDIA 监控
uv pip install --python venv/bin/python sageattention                   # 可选：仅无梯度采样使用 Sage，训练反向保持 SDPA
cd frontend && npm ci && npm run build && cd ..                          # 可选：Web 界面
venv/bin/ypuddin serve --host 127.0.0.1 --port 8123 --data-root studio_data
```

macOS 将上面的 PyTorch 安装行改为 `uv pip install --python venv/bin/python torch torchvision`，不要安装 CUDA 专属依赖。Windows 将 `venv/bin/python` 换成 `venv\Scripts\python.exe`，其他可执行文件也使用 `Scripts` 下对应路径。pip 用户把 `uv pip install --python venv/bin/python` 换成 `venv/bin/pip install` 即可。extras 含义：

| extra | 内容 |
|---|---|
| `models` | transformers / diffusers / huggingface-hub / sentencepiece 等 —— 加载模型与文本编码器（训练必需） |
| `server` | fastapi / uvicorn / psutil —— Web 服务（只用 CLI 训练可不装） |
| `cuda` | bitsandbytes（`optimizer.type = "adamw8bit"`）与 nvidia-ml-py，可选 |
| `nvidia` | nvidia-ml-py（NVIDIA GPU 监控，启动器按平台选择） |
| `optim` | schedulefree、lion-pytorch、prodigyopt、prodigy-plus-schedule-free、pytorch-optimizer |
| `logging` | tensorboard |
| `dev` | pytest、ruff、httpx（开发） |

启动器默认安装日志和常用优化器依赖。`optimizer.fused_backward = true` 尚未实现，配置校验会拒绝。

## 4. 目录与数据

```
YPuddinTrainStudio/
├── venv/                旧部署环境（--profile=legacy）
├── environment/<平台>/venv/  各平台独立环境
├── studio_data/          服务数据目录（--data-root 可改），包含：
│   ├── studio.db         SQLite：项目 / 版本 / 数据集 / 任务 / 产物 / 模型注册表
│   ├── settings.json     「设置」中保存的内容
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

- **备份**包括 `studio_data/`、自定义输出目录以及外部原始数据集的图片/caption/mask；模型权重也应另行保存或记录下载来源。缓存可重建，完整训练断点不能用推理权重代替。
- 模型权重放哪都行，在界面「模型权重」页注册或在配置里填绝对路径；建议使用固定目录（如 `/models`），并在「设置 → 存储路径」中把 `paths.models_dir` 指向它。
- 数据集是一个图片目录（递归），每张图旁边同名 `.txt` 是 caption；`.mask.png` 或 alpha 通道可做遮罩 loss。上传和导入的图片保存为当前版本的独立副本，编辑 caption 和 Mask 只修改副本；高级 TOML 直接引用的外部目录使用原文件。
- 新版本可以复制图片、caption、Mask 和验证源，也可以只继承参数或使用空白配置；任务、采样与产物不复制。任务保存配置快照而不是图片快照，修改旧版本数据前建议先复制出新版本。
- 从旧版本升级的项目不会搬移文件：原 `projects/<pid>/config.json` 作为 v1 草稿，历史任务继续使用原来的运行、缓存和断点路径，新任务使用版本目录。

### 路径设置何时生效

`paths.cache_dir` 和自定义输出目录（`paths.output_mode=custom` 时的 `paths.output_dir`）只影响**新建任务**，已有任务继续使用原来的缓存、权重和断点路径，不会搬迁。自定义缓存目录同时存放扫描索引（`service/index`）、缩略图（`service/thumbnails`）和按平台隔离的软件包缓存（`packages/<平台>`），启动器下载依赖也使用它。自定义输出按 `<项目 ID>/vN/<任务 ID>` 分目录。`paths.models_dir` 是默认的模型扫描目录。数据与模型路径建议填写绝对路径（相对路径按服务工作目录解析）。

“存储路径 → 基础环境目录”（`paths.bootstrap_env_dir`）在下次从启动入口启动时生效：在 `<基础环境目录>/<平台>/venv` 创建或复用该平台环境，也可用 `<启动入口> --env-root <目录>` 临时指定。已有环境不会搬迁或删除，新目录首次启动需要重新安装依赖；目录不能是符号链接或目录联接，也不能包含其他程序创建的环境。界面中准备的其他 PyTorch 版本仍保存在数据目录的 `environment/<平台>/runtimes`。

`server.host` / `server.port` 保存后下次启动生效，`--host` / `--port` 优先。在设置中修改数据目录后，服务重启时切换到新目录；已有数据不会自动迁移，需要先自行复制。

### 数据布局与缓存

各数据源可单独配置分辨率；分桶集合包含全局与各源分辨率。显式验证源拥有独立源索引，并按图片内容排除训练集内的重复图；Plan 与训练使用同一套切分和步数计算。无图、切分后训练集为空等情况在预检时返回字段位置和原因。

latent 缓存按图片内容、桶尺寸、预处理和 VAE 指纹区分；遮罩在读取样本时单独处理，修改遮罩不需要重新编码 latent。文本缓存按编码器和分词器指纹区分。caption 和遮罩内容也计入训练数据指纹，数据变化后不能继续旧的训练状态。

## 5. Anima 权重

需要三个文件（配置字段 → 来源）：

| 字段 | 文件 | 说明 |
|---|---|---|
| `model.dit_path` | `anima-base-*.safetensors`（或 ComfyUI 格式 `model.diffusion_model.` 前缀的文件） | DiT 主干 + LLM adapter，约 4 GB bf16 |
| `model.text_encoder_path` | Qwen3-0.6B（HF 目录，或单文件 safetensors） | 文本编码器；单文件时分词器用内置副本 |
| `model.vae_path` | `qwen_image_vae.safetensors` | Qwen-Image VAE（16 通道 / 8 倍） |

训练得到的 `.safetensors` 使用 kohya 键名（`lora_unet_*`）；`ypuddin convert --to comfyui` 可转换为 ComfyUI 键名，LoRA 也可转换为 PEFT 键名。转换结果尚未用完整权重在 ComfyUI 中做加载测试。

### 5.1 Krea 2 权重（`model.family = "krea2"`）

Krea 2 是 12.9B 参数的单流 MMDiT，文本编码器是 Qwen3-VL-4B-Instruct，VAE 与 Anima 相同（Qwen-Image VAE）：

| 字段 | 文件 | 说明 |
|---|---|---|
| `model.dit_path` | `krea2_raw_bf16.safetensors`（官方 raw 权重，约 26 GB）或 Comfy-Org 的 `krea2_fp8_scaled.safetensors`（约 13 GB） | 键名可带或不带 `model.diffusion_model.` 前缀；CUDA 的 fp8_scaled 路径保留冻结 fp8 权重和文件 scale，执行时仍可能产生临时反量化张量 |
| `model.text_encoder_path` | `Qwen/Qwen3-VL-4B-Instruct` HF 目录（推荐），或 ComfyUI 的单文件 `qwen_3vl_4b*.safetensors`（bf16 或 fp8_scaled 均可） | 只加载语言模型部分（视觉塔不参与）；单文件时把模型的 `config.json` 放在同一目录（没有则按 4B 几何假定），分词器缺失时用内置的 Qwen3 分词器 |
| `model.vae_path` | `qwen_image_vae.safetensors` | 与 Anima 使用相同 VAE 架构；仅当实际权重指纹、预处理和缓存精度等条件一致时可复用 latent 缓存 |

冻结文本编码器时，Krea 2 使用 **cached 模式**（`dataset.text_encoding = "auto"` 自动选择）；训练文本编码器时使用在线编码并保留编码器及其反向图。以下加载顺序适用于冻结文本编码器的缓存模式。启用图片缓存时，程序先用 VAE 缓存图片并卸载 VAE，再缓存描述和采样提示词并卸载文本编码器，最后才加载 DiT 主模型的真实权重。全部缓存命中时无需加载对应编码器；“只构建缓存”不会加载 DiT 权重张量。训练预览会按需重新加载 VAE，结束后再次卸载。

使用 CUDA 分块换出时，如果空闲显存能放下所选文件的实际权重并额外留出 **2 GiB**，会先临时上传完整 DiT，释放 CPU 文件映射后再建立换出缓存；余量不足时保持 CPU 加载路径，不会改动换出块数或训练参数。CPU 内存与显存需要分别留出空间，准备阶段也会出现搬运峰值。详见 [Krea 2 缓存与内存说明](KREA2_MEMORY.md)。

文本缓存默认每批最多编码 16 条文字，与训练 batch 无关。内置预设 `krea2-lokr-default` / `krea2-lora-32` 可作为起点，建议先运行 Plan 和 smoke 检查显存。

下面是 CUDA FP8 路径的自检示例。CPU/MPS 不支持 `memory.base_precision=fp8_*`、SageAttention 和 8-bit 优化器；MPS 按 FP32 执行，Block Swap 不能腾出统一内存。

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

它会用真实训练器跑 3 步、出一张 512 预览、保存并回读适配器，最后打印每项检查的通过情况、耗时、可用的设备内存指标，并写 `outputs/smoke/smoke-report.json`（失败时含完整 traceback）。通过表示这组配置能完成加载、训练、预览和保存。

MPS 可先用 `--device mps --set model.dtype=fp32 --set loop.mixed_precision=no` 自检。配置为 bf16 时，MPS 会给出警告并按 FP32 执行。

## 7. 作为常驻服务

### Linux（systemd）

`/etc/systemd/system/ypuddin.service`：

```ini
[Unit]
Description=YPuddin Train Studio
After=network.target

[Service]
User=trainer
WorkingDirectory=/opt/YPuddinTrainStudio
Environment=PYTHONUTF8=1
ExecStart=/opt/YPuddinTrainStudio/environment/linux-cuda/venv/bin/ypuddin serve --host 127.0.0.1 --port 8123 --data-root /data/studio
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now ypuddin
journalctl -u ypuddin -f
```

先手动跑一次 `<启动入口> --no-browser` 完成安装和前端构建，再交给 systemd。其他平台把 `linux-cuda` 换成对应的环境目录（如 `linux-dtk`、`linux-cpu`）。

### Windows

「任务计划程序」→ 创建任务 → 触发器「登录时」→ 操作：程序 `C:\...\YPuddinTrainStudio\environment\windows-cuda\venv\Scripts\ypuddin.exe`，参数 `serve --host 127.0.0.1 --port 8123 --data-root D:\studio_data`，起始于 `C:\...\YPuddinTrainStudio`。或者直接把 `studio-windows-cuda.bat --no-browser` 的快捷方式放进启动文件夹。

### 远程访问与安全

服务**没有登录认证**，默认只监听 `127.0.0.1`。要从别的机器访问，二选一：

- SSH 端口转发（推荐）：`ssh -L 8123:127.0.0.1:8123 user@gpu-box`，然后本机开 `http://127.0.0.1:8123/`。
- 放在 Nginx/Caddy 之类反向代理后面，由代理做 HTTP Basic Auth / OAuth，再把 `--host` 改成 `0.0.0.0`。**不要**把裸服务直接暴露到公网：文件浏览接口（`/api/fs/list`）与训练任务可以读写 `--data-root` 及配置里写到的任何路径。

SSE（`/api/events`）需要代理关闭响应缓冲（Nginx：`proxy_buffering off; proxy_read_timeout 1h;`）。

### 队列、暂停与恢复

任务创建前会预检数据、配置、设备能力和内存估算；错误包含字段位置。队列按优先级和创建时间调度，定时任务到时进入队列。`max_concurrent` 限制总进程数；每张显卡同一时间只运行一个任务，不同显卡可以同时运行不同任务，单个任务也可以使用多张卡，见 [单任务多卡训练](TRAINING_DDP.md) 与 [按显卡运行独立任务](GPU_JOB_SCHEDULING.md)。

队列设置 `memory_admission` 默认开启：估算峰值超过当前设备可用内存的 95% 时任务等待设备，任务阶段会显示等待原因。可通过 `PUT /api/queue/settings` 设置 `{"memory_admission": false}` 关闭这项估算门禁；设备独占仍然有效。Plan 是估算值，不能保证不会显存不足。

训练阶段的暂停会在优化器步之间保存完整断点；缓存准备阶段也可以暂停或停止，已完成的缓存保留，恢复后继续准备。进程异常退出时任务记为失败，可从最后一个完整断点恢复。正常关闭服务时会尝试让运行中的任务保存状态，但强制结束或系统故障时最后一步可能没有保存。

训练参数页支持 TOML 导入导出（API 也支持 JSON）。任务页实时显示进度和产物，可下载权重或从完整断点续训；开启 EMA 时另存 EMA 权重。下载的权重文件不包含完整训练状态，不能用于精确续训。

## 8. 更新

```bash
git pull
<启动入口>          # 依赖签名（pyproject.toml）或前端源码变了会自动重装 / 重建
```

更新前保存训练状态、停止服务，并备份服务数据与外部输出目录。启动器会检查依赖与前端，必要时补装依赖或重建前端；已构建的前端与源码一致时不需要 Node。已有项目、配置和模型文件保留。从 v0.4 之前的版本升级时，启动会自动把旧项目、数据源和任务归入兼容的 v1 版本，不搬迁文件，也不改写任务快照。

完整训练状态（state v2）保存可训练参数、优化器与调度器、采样器、随机数状态（含 DataLoader 生成器）和 EMA；推理用的 `.safetensors` 只用于加载和分发适配器。

“可复现训练”（`loop.deterministic`）默认关闭。开启后在模型加载前启用 PyTorch 确定性算法和确定性卷积，NVIDIA CUDA 未设置 cuBLAS workspace 时使用 `:4096:8`；这可能降低速度、增加显存，遇到不支持确定性计算的算子会直接报错停止。DTK 的原生 SDPA 会改用数学实现，速度明显变慢、显存需求增加；显式选择 FlashAttention/xFormers 时仍使用相应扩展。它只减少相同设备、软件、数据和配置下的差异，不保证跨设备、跨版本或第三方算子逐位一致。

该设置记入完整训练状态，恢复时不能更改；缺少此记录的旧状态需要设置 `loop.deterministic=false` 才能恢复。要为已有任务开启，请从已有权重新建训练。

恢复训练会检查数据指纹：修改图片、caption 或遮罩后不能继续原训练状态，仅移动或改名不受影响。指纹不兼容时，可以恢复原来的数据，或用 `adapter.resume_weights` 从已有权重开始**新训练**（优化器、调度器和步数重新开始）。旧版本创建且指纹不兼容的训练状态，请用旧版本完成。

## 9. 排障

| 现象 | 处理 |
|---|---|
| Windows 双击 `studio-windows-cuda.bat` 后窗口一直没有任何输出 | `.bat` 需要 CRLF 换行；编辑器另存或 `core.autocrlf=false` 的克隆可能改成 LF，导致 cmd.exe 解析错乱。执行 `git checkout -- studio-windows-cuda.bat scripts/launch.bat` 恢复；也可以在 `cmd` 中运行 `python scripts\bootstrap.py doctor` 查看报错 |
| `doctor` 显示 `cuda_available: false` 但机器有 NVIDIA 卡 | 驱动太旧或装了 CPU 版 torch：`nvidia-smi` 看驱动版本，`<启动入口> --reinstall --torch=cu124`（驱动 ≥550）或 `cu118` |
| Windows 上 `ModuleNotFoundError: bitsandbytes` / 训练启动就失败 | 配置里 `optimizer.type` 改回 `adamw`，或在 CUDA 环境安装 `bitsandbytes>=0.43` |
| 页面能开但任务列表 / 数据集为空、控制台 404 | 前端是旧构建：`<启动入口> build` |
| 端口被占用 | `--port 8800`，或找出占用者：`lsof -i :8123`（Linux/macOS）、`netstat -ano \| findstr 8123`（Windows） |
| 前端构建提示 Node 不支持 | 升级到 Node 20.19+（20.x）或 22.12+，确认 `node --version`，再运行 `<启动入口> build` |
| 首次安装很慢 / 超时 | PyTorch 安装包约 2.5 GB。某个源安装失败会自动换下一个（日志显示“包源 … 失败，换下一个源重试”）；境外网络可用 `--index=official`，或在设置中切换下载源 |
| RTX 50 系报 `no kernel image is available for execution on the device` / `sm_120 is not compatible` | 装到了旧 CUDA 构建：`<启动入口> --reinstall --torch=cu128`；驱动需 ≥ 570。`<启动入口> doctor` 会对比显卡计算能力与 torch 内核列表并直接给出警告 |
| Apple GPU 没有被使用 | 运行 `doctor` 查看 MPS 是否可用，再用 `--device mps` 自检。MPS 不支持 FP8、SageAttention 和 8-bit 优化器 |
| 训练 OOM | 降低分辨率和 batch，开启逐块重算和缓存文本编码；CUDA 还可以适当增加 Block Swap 或使用 8-bit 优化器。MPS 的 Block Swap 不能腾出统一内存。Plan 提供估算，实际峰值以 smoke 自检为准 |
| 任务一直等待设备 | 查看任务等待原因：设备可能被另一任务独占，或估算峰值超过可用内存的 95%。关闭 `memory_admission` 仅跳过内存估算门禁，不解除单设备独占 |
| 恢复提示 dataset fingerprint 不一致 | 检查图片、caption、mask、数据源与验证设置是否变化；旧版指纹不能与新版精确续训混用，见 §8 |
| `fused_backward is not implemented` | 将 `optimizer.fused_backward` 设为 `false`，当前不支持此功能 |
| 采样阶段看起来“卡住” | 任务详情页会显示「生成预览 第 k/n 张 · 步 x/y」进度 |
| 任务状态 `failed`，error 是 `process exited with code …` | 在任务详情的日志中查看最后几行；完整日志是任务输出目录下的 `run.log`（默认 `studio_data/project/<项目 ID>/vN/output/<任务 ID>/run.log`） |
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
ypuddin serve    --port 8123 --data-root studio_data
```

配置文件支持 TOML 与 JSON（`ypuddin schema` 打印带说明的 JSON Schema）；`--set a.b=c` 可覆盖字段，CLI 的 `--preset preset.toml` 叠加**预设文件**，Web 内置预设通过界面选择。配置校验和导入导出无需加载模型；Plan 还会检查数据和权重几何信息，默认检测本机设备，也可用 `--device` 指定目标。最小 Anima 配置见 `docs/design/03-status.md`。

## 11. 设置、扩展与模型下载

“设置”包含运行环境、模型权重、访问密钥、软件下载源、存储路径、界面与服务六类。全局任务队列管理跨项目任务，训练产物在“项目 → 版本 → 训练结果”中查看。

在“运行环境”中安装或更换扩展时，会先列出将要安装的 wheel 与版本变化，确认后显示安装日志。基础的 Torch/CUDA/NumPy 受保护；没有兼容的预编译 wheel 时显示原因，不会自动从源码编译。依赖实际发生变化后需要重启 Studio，重启前不能启动训练和缓存任务。

Windows 上“包已安装”不等于它的 CUDA 内核支持当前显卡。例如 RTX 5070 Ti（SM120）使用官方 xFormers 0.0.35 CUDA 12.8 wheel 时，可能遇到 `No operator found` / `GPU ... too new`；运行环境页会分别显示导入状态和实际正反向检测结果，SDPA 检测通过时可继续使用 SDPA。不要为此降级受保护的 PyTorch。

xFormers 可以调用单独安装的 FlashAttention 2。Windows 可在运行环境中上传同时匹配 Python、PyTorch、CUDA 和 GPU 架构的 FA2 wheel，再安装并重新检测。社区预编译 wheel 不是 FlashAttention 官方发布的 Windows 包，请核对发布来源与 SHA256，以本机的正反向检测结果为准。

### 访问密钥

在“设置 → 访问密钥”保存或清除 Hugging Face、ModelScope、Danbooru 与 Gelbooru 凭据。模型令牌保存在数据目录的 `secrets.json`；页面和 GET API 只显示是否已配置，不回显账号与密钥。正则任务默认使用对应站点已保存的凭据，运行中的任务保留启动时的凭据，清除只影响之后的任务。HF-Mirror 匿名下载；清除模型令牌也会停用同来源的环境变量/CLI 回退。

`secrets.json` 是本机敏感配置，没有加密。更新时保留，分享源码或打包时排除；不要把它放进项目 TOML、日志、截图或公开的问题报告。

### 模型下载

“模型权重 → 准备模型”按模型族列出主模型、文本编码器和 VAE，可选择 Hugging Face 或魔搭的官方来源。推荐文件下载完成后会核验字节数、SHA-256 和组件类型，通过后才登记；重新下载从头开始。共享 VAE 可复用已有文件。本地文件第一次设为推荐默认时需要读取整个文件校验 SHA，大文件需要等待，之后按文件状态复用校验结果。受限仓库需要先在发布平台取得权限；自定义下载不会被标记为官方推荐文件。
