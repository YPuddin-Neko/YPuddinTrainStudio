# 部署与运行指南

面向要在自己机器（本地工作站或租用的 GPU 机器）上跑 YPuddin Train Studio 的人。开发者相关的内容（测试、导出 API 文档）在 `docs/design/03-status.md`。

## 1. 系统要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Linux（推荐 Ubuntu 22.04+）、Windows 10/11、macOS 13+（仅 CPU/MPS，能跑通流程但训练 Anima 不实用） |
| Python | 3.10 – 3.12（`studio.sh/.bat` 会自动查找；装了 [uv](https://docs.astral.sh/uv/) 时可自动下载 3.12） |
| GPU | 训练 Anima 2B：NVIDIA 显卡 ≥ 12 GB（开 block swap + cached 文本可在 8 GB 上尝试）；驱动 ≥ 550 建议（决定安装哪个 CUDA 版本的 PyTorch，见 §3） |
| Node.js | ≥ 18，仅在需要**构建 Web 界面**时需要（没有 Node 也能用 CLI 和 API） |
| 磁盘 | 依赖约 6 GB（含 PyTorch）；Anima 权重约 4 GB（DiT）+ 1.2 GB（Qwen3-0.6B）+ 0.25 GB（VAE）；每张训练图的 latent 缓存几十 KB |
| 网络 | 首次安装需要访问 PyPI / download.pytorch.org / npm；国内可加 `--mirror` |

## 2. 一键启动（推荐）

```bash
git clone <本仓库> YPuddinTrainStudio && cd YPuddinTrainStudio/xiangmuyuanma
./studio.sh            # Linux / macOS
studio.bat             # Windows（双击或在 PowerShell 里 .\studio.bat）
```

第一次运行会依次：创建 `.venv` → 按 NVIDIA 驱动版本安装对应 CUDA 的 PyTorch → 安装 `ypuddin[models,server]`（Linux+GPU 再加 `cuda,optim`：bitsandbytes 8-bit 优化器、Prodigy 等）→ 有 Node 则构建前端 → 启动服务 → 打开浏览器 `http://127.0.0.1:8765/`。之后再运行只做增量检查（依赖签名不变就跳过安装，前端源码没变就不重新构建），几秒内起服务。

常用参数（`.sh` 与 `.bat` 一致）：

| 参数 | 作用 |
|---|---|
| `--port 8800` / `--host 0.0.0.0` / `--data-root /data/studio` | 服务端口 / 绑定地址 / 数据目录（默认 `127.0.0.1` `8765` `./studio_data`） |
| `--torch=cu128` | 强制 PyTorch 版本：`cu128` `cu126` `cu124` `cu118` `cpu`（默认 `auto`：驱动 ≥570→cu128，≥560→cu126，≥550→cu124，否则 cu118；无 NVIDIA 驱动→cpu） |
| `--mirror` | 用清华 PyPI 镜像装依赖（PyTorch 仍从官方索引装） |
| `--reinstall` | 删掉 `.venv` 重装（`studio_data/` 不受影响） |
| `--no-browser` / `--no-frontend` | 不自动开浏览器 / 不构建前端（只要 API） |

子命令：

| 命令 | 作用 |
|---|---|
| `./studio.sh` 或 `./studio.sh run` | 安装/更新 + 构建 + 启动服务 |
| `./studio.sh doctor` | 打印本机情况：Python、torch/CUDA/GPU、可选依赖、Node、前端构建状态。**排障先跑这个** |
| `./studio.sh smoke --set model.dit_path=… --set model.text_encoder_path=… --set model.vae_path=…` | 真实跑 3 步训练 + 出一张预览 + 保存/回读 LoRA，输出报告 `outputs/smoke/smoke-report.json`（见 §6） |
| `./studio.sh build` | 只构建前端 |
| `./studio.sh dev` | 后端 + Vite 热更新前端（前端开发用，浏览器开 `http://127.0.0.1:3000/`） |
| `./studio.sh test` | 跑后端 pytest（有 Node 时再跑前端 vitest） |
| `./studio.sh shell` | 打印如何激活 `.venv`（之后可直接用 `ypuddin …` 命令） |

## 3. 手动安装（不用脚本时）

```bash
cd xiangmuyuanma
uv venv --python 3.12 .venv                    # 或 python3.12 -m venv .venv
# PyTorch：Windows 必须从官方索引装 CUDA 版；Linux 的 PyPI 轮子已带 CUDA，但显式指定更稳
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -e ".[models,server]"           # 训练 + 服务
uv pip install --python .venv/bin/python -e ".[cuda,optim]"              # 可选：bitsandbytes 8-bit、Prodigy 等（Windows 上 bitsandbytes 需 ≥0.43 的官方 wheel）
uv pip install --python .venv/bin/python sageattention                   # 可选：model.attention = "sage"
cd frontend && npm ci && npm run build && cd ..                          # 可选：Web 界面
.venv/bin/ypuddin serve --host 127.0.0.1 --port 8765 --data-root studio_data
```

pip 用户把 `uv pip install --python .venv/bin/python` 换成 `.venv/bin/pip install` 即可。extras 含义：

| extra | 内容 |
|---|---|
| `models` | transformers / huggingface-hub / sentencepiece —— 加载 Qwen3 文本编码器与分词器（Anima 必需） |
| `server` | fastapi / uvicorn / psutil —— Web 服务（只用 CLI 训练可不装） |
| `cuda` | bitsandbytes（`optimizer.type = "adamw8bit"`）、nvidia-ml-py（GPU 监控） |
| `optim` | prodigyopt、prodigy-plus-schedulefree、pytorch-optimizer |
| `logging` | tensorboard、wandb |
| `dev` | pytest、ruff、httpx（开发） |

## 4. 目录与数据

```
xiangmuyuanma/
├── .venv/                Python 环境（可随时 --reinstall 重建）
├── studio_data/          服务数据目录（--data-root 可改），包含：
│   ├── studio.db         SQLite：项目 / 数据集 / 任务 / 产物 / 模型注册表 的元数据
│   ├── settings.json     「系统设置」页保存的设置
│   ├── projects/<pid>/   每个项目：config.json（配置草稿）、cache/（latent + 文本缓存，任务间共享）、runs/<jid>/（每次训练：events.jsonl、run.log、权重、samples/、state-*/ 断点）
│   ├── runs/             不属于任何项目的任务
│   ├── datasets/         数据集索引（图片哈希、尺寸、caption 路径）
│   ├── cache/            全局图片索引与无项目任务的共享缓存
│   ├── thumbs/           数据集页缩略图缓存（可删）
│   ├── presets/          用户保存的预设
│   └── models/           `POST /models/scan` 默认扫描的权重目录
├── outputs/              CLI 直接训练时的默认输出（配置里 checkpoint.output_dir）
└── frontend/dist/        构建好的前端（`ypuddin serve` 存在即托管）
```

- **备份**只需 `studio_data/`（元数据 + 产物）；缓存可随时重建。
- 模型权重放哪都行，在界面「模型权重」页注册或在配置里填绝对路径；建议一个固定目录（如 `/models`），并在「系统设置」里把 `paths.models_dir` 指向它。
- 数据集就是一个图片目录（递归），每张图旁边同名 `.txt` 是 caption；`.mask.png` 或 alpha 通道可做遮罩 loss。服务只读它并原地修改 `.txt`，不会复制图片。

## 5. Anima 权重

需要三个文件（配置字段 → 来源）：

| 字段 | 文件 | 说明 |
|---|---|---|
| `model.dit_path` | `anima-base-*.safetensors`（或 ComfyUI 格式 `model.diffusion_model.` 前缀的文件） | DiT 主干 + LLM adapter，约 4 GB bf16 |
| `model.text_encoder_path` | Qwen3-0.6B（HF 目录，或单文件 safetensors） | 文本编码器；单文件时分词器用内置副本 |
| `model.vae_path` | `qwen_image_vae.safetensors` | Qwen-Image VAE（16 通道 / 8 倍） |

训练得到的 `.safetensors` 用 kohya 键名（`lora_unet_*`），ComfyUI 的 LoRA 加载器可直接读；`ypuddin convert --to comfyui` 可转成 PEFT 键名。

## 6. 第一次在 GPU 机器上验证

```bash
./studio.sh smoke \
  --set model.dit_path=/models/anima-base-v1.0.safetensors \
  --set model.text_encoder_path=/models/Qwen3-0.6B-Base \
  --set model.vae_path=/models/qwen_image_vae.safetensors \
  --set adapter.algo=lokr --set adapter.rank=full --set adapter.factor=8 \
  --set memory.activation_checkpointing=block
# 显存紧张：追加 --set memory.blocks_to_swap=10 --set dataset.text_encoding=cached --resolution 512
```

它会用真实训练器跑 3 步、出一张 512 预览、保存并回读 LoRA，最后打印每项检查的通过情况、耗时、峰值显存，并写 `outputs/smoke/smoke-report.json`（失败时含完整 traceback）。把这个文件发给开发者就能定位问题。

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
ExecStart=/opt/YPuddinTrainStudio/xiangmuyuanma/.venv/bin/ypuddin serve --host 127.0.0.1 --port 8765 --data-root /data/studio
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now ypuddin
journalctl -u ypuddin -f
```

先手动跑一次 `./studio.sh --no-browser` 完成安装和前端构建，再交给 systemd。

### Windows

「任务计划程序」→ 创建任务 → 触发器「登录时」→ 操作：程序 `C:\...\xiangmuyuanma\.venv\Scripts\ypuddin.exe`，参数 `serve --host 127.0.0.1 --port 8765 --data-root D:\studio_data`，起始于 `C:\...\xiangmuyuanma`。或者直接把 `studio.bat --no-browser` 的快捷方式放进启动文件夹。

### 远程访问与安全

服务**没有登录认证**，默认只监听 `127.0.0.1`。要从别的机器访问，二选一：

- SSH 端口转发（推荐）：`ssh -L 8765:127.0.0.1:8765 user@gpu-box`，然后本机开 `http://127.0.0.1:8765/`。
- 放在 Nginx/Caddy 之类反向代理后面，由代理做 HTTP Basic Auth / OAuth，再把 `--host` 改成 `0.0.0.0`。**不要**把裸服务直接暴露到公网：文件浏览接口（`/api/fs/list`）与训练任务可以读写 `--data-root` 及配置里写到的任何路径。

SSE（`/api/events`）需要代理关闭响应缓冲（Nginx：`proxy_buffering off; proxy_read_timeout 1h;`）。

## 8. 更新

```bash
git pull
./studio.sh          # 依赖签名（pyproject.toml）或前端源码变了会自动重装 / 重建
```

`studio_data/` 里的 SQLite 结构变更由服务启动时自动迁移（`CREATE TABLE IF NOT EXISTS`；破坏性变更会在 CHANGELOG 里说明）。

## 9. 排障

| 现象 | 处理 |
|---|---|
| `doctor` 显示 `cuda_available: false` 但机器有 NVIDIA 卡 | 驱动太旧或装了 CPU 版 torch：`nvidia-smi` 看驱动版本，`./studio.sh --reinstall --torch=cu124`（驱动 ≥550）或 `cu118` |
| Windows 上 `ModuleNotFoundError: bitsandbytes` / 训练启动就失败 | 配置里 `optimizer.type` 改回 `adamw`，或 `.\.venv\Scripts\pip install bitsandbytes>=0.43` |
| 页面能开但任务列表 / 数据集为空、控制台 404 | 前端是旧构建：`./studio.sh build`；开发态的 mock 数据请用 `dev` 模式而不是 `run` |
| 端口被占用 | `--port 8800`，或找出占用者：`lsof -i :8765`（Linux/macOS）、`netstat -ano \| findstr 8765`（Windows） |
| 首次安装很慢 / 超时 | 国内加 `--mirror`；PyTorch 轮子约 2.5 GB，请耐心或先用 `--torch=cpu` 把流程跑通 |
| 训练 OOM | 依次：`memory.activation_checkpointing = "block"` → `dataset.text_encoding = "cached"` → `memory.blocks_to_swap = 8…20` → 降分辩率 / batch 1 → `optimizer.type = "adamw8bit"`；界面的 Plan 面板会给出估算与建议 |
| 采样阶段看起来"卡住" | 任务详情页 header 有「生成预览 第 k/n 张 · 步 x/y」进度；分辩率填错（如 10240）会被配置校验拒绝 |
| 任务状态 `failed`，error 是 `process exited with code …` | 打开任务详情「Logs」看最后几行；`studio_data/projects/<pid>/runs/<jid>/run.log` 是完整日志 |
| 想彻底重来 | 删除 `.venv/`（`--reinstall`）；数据只在 `studio_data/`，删它才会丢项目 |

## 10. CLI 速查（激活 `.venv` 后）

```bash
ypuddin plan     config.toml                 # 不加载权重：校验 + 步数 / 分桶 / 显存估算
ypuddin train    config.toml                 # 训练（Ctrl+C = 暂停并保存断点；state-*/ 可 --set checkpoint.resume=…）
ypuddin cache    config.toml                 # 只预编码 latent / 文本缓存
ypuddin smoke    --set model.dit_path=…      # 机器自检
ypuddin inspect  lora.safetensors            # 看元数据与模块
ypuddin convert  lora.safetensors --to comfyui -o out.safetensors
ypuddin merge    --base anima-base.safetensors --adapter lora.safetensors --family anima -o merged.safetensors
ypuddin extract  --base a.safetensors --tuned b.safetensors --algo lokr -o diff-lokr.safetensors
ypuddin serve    --port 8765 --data-root studio_data
```

配置文件是 TOML，字段与界面完全一致（`ypuddin schema` 打印带说明的 JSON Schema）；`--set a.b=c` 可覆盖任意字段，`--preset name` 叠加预设。最小 Anima 配置见 `docs/design/03-status.md`。
