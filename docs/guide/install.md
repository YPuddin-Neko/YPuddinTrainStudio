# 安装与启动

## 系统要求

| 项目 | 要求 |
| --- | --- |
| 操作系统 | Windows、Linux、macOS。CUDA 与 DTK 入口只支持 x86_64；MPS 需要 Apple 芯片 |
| Python | 3.10–3.12，推荐 3.12。启动器会自动查找；装了 [uv](https://docs.astral.sh/uv/) 时可以自动下载 3.12 |
| 显卡 | CUDA 训练需要匹配的 NVIDIA 驱动；海光需要匹配的 DTK 运行库和厂商 PyTorch，见 [海光 DTK](dtk.md)。MPS 按 FP32 训练，与系统共用统一内存 |
| Node.js | 20.19+（20.x）或 22.12+，只在构建网页界面时需要；不支持 18、21 |
| 磁盘 | PyTorch 与依赖需要数 GB；另外要为模型权重、训练数据、缓存、预览图和完整断点留出空间 |
| 网络 | 首次安装要下载依赖。默认使用国内镜像，失败后自动换下一个来源 |

显存和内存需求随模型、精度、分辨率、优化器和缓存方式变化，可以先看训练计划里的估算。

## 一键启动

```bash
git clone https://github.com/YPuddin-Neko/YPuddinTrainStudio.git
cd YPuddinTrainStudio
```

然后运行与机器对应的启动脚本：

| 机器 | 启动脚本 |
| --- | --- |
| Windows + NVIDIA | `studio-windows-cuda.bat`（双击，或在 PowerShell 中运行 `.\studio-windows-cuda.bat`） |
| Windows 仅 CPU | `studio-cpu.bat` |
| Linux + NVIDIA | `./studio-linux-cuda.sh` |
| Linux + 海光 DTK | `./studio-linux-dtk.sh`（先准备厂商运行库，见 [海光 DTK](dtk.md)） |
| macOS Apple 芯片 | `./studio-macos.command` |
| Linux／macOS 仅 CPU | `./studio-cpu.sh` |

首次启动会为当前平台创建独立的 Python 环境，安装 PyTorch 和训练依赖，构建前端，然后打开 `http://127.0.0.1:8123/`。以后启动会检查依赖、修复已识别的兼容问题，并按需重新构建前端。各平台环境互不共用，请一直使用同一个启动脚本；下文用 `<启动脚本>` 指代它。平台环境和 PyTorch 版本切换见 [运行环境](environment.md)。

注意力加速扩展不会自动安装，需要时在 **设置 → 运行环境** 中安装，见 [注意力加速](attention.md)。

### 常用参数

`.sh` 和 `.bat` 的参数相同：

| 参数 | 作用 |
| --- | --- |
| `--port 8800`、`--host 0.0.0.0`、`--data-root /data/studio` | 端口、绑定地址、数据目录，默认 `8123`、`127.0.0.1`、`./studio_data`。设置中保存的地址、端口和数据目录下次启动生效，命令行指定的优先 |
| `--torch=cu128` | 首次安装或 `--reinstall` 时的 PyTorch 类型：`cu128`、`cu126`、`cu124`、`cu118`、`cpu`。默认 `auto`：RTX 50 系选 cu128，其余按驱动版本 ≥570→cu128、≥560→cu126、≥550→cu124、≥450→cu118，否则 cpu |
| `--index=auto\|cn\|official` | 依赖下载源。未指定时读取已保存的设置；`auto`／`cn` 优先国内镜像，`official` 优先官方源。`--mirror` 等于 `--index=cn` |
| `--env-root <目录>` | 这次启动使用的基础环境根目录 |
| `--profile=legacy` | 使用根目录的旧 `venv/` |
| `--reinstall` | 只重建当前平台的基础环境；其他平台、数据、模型和界面准备的 PyTorch 版本保留 |
| `--no-browser`、`--no-frontend` | 不自动打开浏览器；不构建前端（只使用 API） |

### 子命令

| 命令 | 作用 |
| --- | --- |
| `<启动脚本>` 或 `<启动脚本> run` | 安装或更新依赖、构建前端、启动服务 |
| `<启动脚本> doctor` | 打印 Python、PyTorch、CUDA／显卡、可选依赖、Node 和前端构建状态。**遇到问题先运行这个** |
| `<启动脚本> smoke --set …` | 用真实权重跑几步训练自检，见下文 |
| `<启动脚本> build` | 只构建前端 |
| `<启动脚本> dev` | 同时启动后端和 Vite 热更新前端，浏览器打开 `http://127.0.0.1:3000/`；`--fe-port` 修改前端端口 |
| `<启动脚本> shell` | 打印激活 Python 环境的方法，之后可以直接使用 `ypuddin` 命令 |

## 首次运行自检

在 GPU 机器上第一次训练前，可以用真实权重跑一次自检：

```bash
<启动脚本> smoke \
  --set model.dit_path=/models/anima-base-v1.0.safetensors \
  --set model.text_encoder_path=/models/Qwen3-0.6B-Base \
  --set model.vae_path=/models/qwen_image_vae.safetensors \
  --set adapter.algo=lokr --set adapter.rank=full --set adapter.factor=8 \
  --set memory.activation_checkpointing=block
# 显存紧张时追加：--set memory.blocks_to_swap=10 --set dataset.text_encoding=cached --resolution 512
```

它用真实训练器跑 3 步、生成一张 512 预览图、保存并重新读取适配器，打印每一项的结果、耗时和显存峰值，并写出 `outputs/smoke/smoke-report.json`（失败时包含完整错误信息）。通过说明这组配置能完成加载、训练、预览和保存。

Krea 2 的 FP8 示例：

```bash
<启动脚本> smoke \
  --set model.family=krea2 \
  --set model.dit_path=/models/krea2_fp8_scaled.safetensors \
  --set model.text_encoder_path=/models/Qwen3-VL-4B-Instruct \
  --set model.vae_path=/models/qwen_image_vae.safetensors \
  --set memory.base_precision=fp8_e4m3 --set memory.activation_checkpointing=block \
  --set objective.timestep_sampling=resolution_shift --resolution 512
```

Mac 可以用 `--device mps --set model.dtype=fp32 --set loop.mixed_precision=no` 自检。MPS 不支持 FP8、SageAttention 和 8-bit 优化器，分块换出也不能腾出统一内存。

## 手动安装

不用启动脚本时，以 Linux + CUDA 为例：

```bash
cd YPuddinTrainStudio
uv venv --python 3.12 venv
uv pip install --python venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python venv/bin/python -e ".[models,server,optim,logging]"
uv pip install --python venv/bin/python -e ".[cuda,nvidia]"     # 可选：8-bit 优化器、NVIDIA 监控
cd frontend && npm ci && npm run build && cd ..                 # 可选：网页界面
venv/bin/ypuddin serve --host 127.0.0.1 --port 8123 --data-root studio_data
```

- PyTorch 的 CUDA 版本按驱动和显卡选择。国内可以把 `--index-url` 换成 `https://mirror.sjtu.edu.cn/pytorch-wheels/cu128`；其余依赖可加 `--index-url https://mirrors.ustc.edu.cn/pypi/simple`。
- macOS 把 PyTorch 安装行改为 `uv pip install --python venv/bin/python torch torchvision`，不要安装 CUDA 依赖。
- Windows 把 `venv/bin/python` 换成 `venv\Scripts\python.exe`。
- 用 pip 时把 `uv pip install --python venv/bin/python` 换成 `venv/bin/pip install`。

| extra | 内容 |
| --- | --- |
| `models` | transformers、diffusers、huggingface-hub、sentencepiece 等，加载模型必需 |
| `server` | fastapi、uvicorn、psutil，Web 服务需要；只用命令行训练可以不装 |
| `optim` | schedulefree、lion-pytorch、prodigyopt、prodigy-plus-schedule-free、pytorch-optimizer |
| `logging` | tensorboard |
| `cuda` | bitsandbytes（8-bit 优化器）与 nvidia-ml-py |
| `nvidia` | nvidia-ml-py（NVIDIA 显卡监控） |

## 作为常驻服务

### Linux（systemd）

先手动运行一次 `<启动脚本> --no-browser` 完成安装和前端构建，再创建 `/etc/systemd/system/ypuddin.service`：

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

其他平台把 `linux-cuda` 换成对应的环境目录，例如 `linux-dtk`、`linux-cpu`。

### Windows

在“任务计划程序”中创建任务：触发器选“登录时”，程序填 `...\YPuddinTrainStudio\environment\windows-cuda\venv\Scripts\ypuddin.exe`，参数填 `serve --host 127.0.0.1 --port 8123 --data-root D:\studio_data`，起始位置填源码目录。也可以把 `studio-windows-cuda.bat --no-browser` 的快捷方式放进启动文件夹。

### 远程访问与安全

服务**没有登录认证**，默认只监听 `127.0.0.1`。从其他机器访问时二选一：

- SSH 端口转发（推荐）：`ssh -L 8123:127.0.0.1:8123 user@gpu-box`，然后在本机打开 `http://127.0.0.1:8123/`。
- 放在 Nginx、Caddy 等反向代理后面，由代理负责认证，再把 `--host` 改成 `0.0.0.0`。

**不要**把服务直接暴露到公网：文件浏览接口和训练任务可以读写数据目录以及配置中写到的任何路径。事件流 `/api/events` 需要代理关闭响应缓冲（Nginx：`proxy_buffering off; proxy_read_timeout 1h;`）。

## 队列、暂停与恢复

- 创建任务前会检查数据、配置、设备能力和显存估算，错误会指出具体字段。
- 队列按优先级和创建时间调度，定时任务到时进入队列。每张显卡同一时间只运行一个任务，不同显卡可以同时运行不同任务，见 [按显卡运行多个任务](gpu-scheduling.md)；一个任务也可以使用多张卡，见 [多卡训练](multi-gpu.md)。
- 训练中的暂停会在两次参数更新之间保存完整断点；缓存准备阶段也可以暂停或停止，已完成的缓存保留。
- 进程异常退出时任务记为失败，可以从最后一个完整断点恢复。正常关闭服务时会尽量让运行中的任务先保存；强制结束或系统故障时，最后一步可能没有保存。
- 训练参数页支持 TOML 导入导出（API 也支持 JSON）。下载的权重文件不含训练状态，不能用来精确续训。

## 更新

```bash
git pull
<启动脚本>
```

更新前先让训练保存状态、停止服务，并备份数据目录和外部输出目录。启动器会检查依赖和前端，必要时补装依赖或重新构建前端；已构建的前端与源码一致时不需要 Node。已有项目、配置和模型文件都会保留。从 v0.4 之前的版本升级时，旧项目会自动归入兼容的 v1 版本，不移动文件。

### 精确续训的条件

- 完整训练状态保存可训练参数、优化器、学习率调度器、数据位置、随机数状态和 EMA。
- 恢复时会检查数据指纹：修改了图片、标签或遮罩后不能继续原来的训练状态；只移动或改名文件不受影响。可以恢复原来的数据，或用 `adapter.resume_weights` 从已有权重开始**新训练**（优化器、调度器和步数重新开始）。
- **可复现训练**（`loop.deterministic`）默认关闭。开启后使用 PyTorch 的确定性算法，可能更慢、占用更多显存，遇到不支持确定性计算的算子会报错停止；DTK 的原生 SDPA 会改用数学实现，速度明显变慢。它只减少同一设备、软件、数据和配置下的差异，不保证跨设备或跨版本逐位一致。这个设置记录在训练状态中，恢复时不能更改。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| 双击 Windows 启动脚本后窗口关闭 | 在 `cmd` 中运行该脚本的 `doctor` 子命令，查看具体错误。项目的 `.gitattributes` 要求 `.bat` 使用 CRLF；手动编辑过脚本时也应检查换行格式 |
| `doctor` 显示 `cuda_available: false`，但机器有 NVIDIA 显卡 | 驱动太旧或装成了 CPU 版 PyTorch：用 `nvidia-smi` 查看驱动版本，然后 `<启动脚本> --reinstall --torch=cu124`（驱动 ≥550）或 `cu118` |
| RTX 50 系报 `no kernel image is available` 或 `sm_120 is not compatible` | 装到了旧的 CUDA 构建：`<启动脚本> --reinstall --torch=cu128`，驱动需要 ≥570。`doctor` 会对比显卡和 PyTorch 的内核并给出警告 |
| `torchvision::nms does not exist` | Torch 与 TorchVision 可能安装了不同的 CUDA 构建。先用原启动脚本重新启动，让启动检查修复匹配关系；仍失败时保留完整日志并核对这两个包的版本和构建后缀 |
| `ModuleNotFoundError: bitsandbytes` | 把优化器改回 `adamw`，或在 CUDA 环境安装 `bitsandbytes>=0.43` |
| 页面能打开，但列表为空或控制台出现 404 | 前端是旧构建：`<启动脚本> build` |
| 前端构建提示 Node 版本不支持 | 升级到 Node 20.19+（20.x）或 22.12+，再运行 `<启动脚本> build` |
| 端口被占用 | 改用 `--port 8800`，或查找占用的程序：`lsof -i :8123`（Linux／macOS）、`netstat -ano \| findstr 8123`（Windows） |
| 首次安装很慢或超时 | PyTorch 安装包较大；开启自动换源时，某个源失败会尝试下一个。境外网络可用 `--index=official`，或在设置中更换下载源 |
| Apple GPU 没有被使用 | 运行 `doctor` 查看 MPS 是否可用，再用 `--device mps` 自检 |
| 训练显存不足 | 降低分辨率和批大小，开启逐块梯度检查点和文字缓存；CUDA 还可以增加分块换出或使用 8-bit 优化器。训练计划的数字是估算，实际峰值以自检结果为准 |
| 任务一直等待设备 | 查看任务的等待原因：显卡可能被其他任务占用，或估算峰值超过可用显存的 95% |
| 恢复时提示数据指纹不一致 | 检查图片、标签、遮罩、数据来源和验证设置是否改动，见上文的续训条件 |
| 预览阶段看起来卡住 | 任务详情页会显示“生成预览 第 k/n 张 · 步 x/y”的进度 |
| 任务失败，错误为 `process exited with code …` | 在任务详情的日志中查看最后几行；完整日志是任务输出目录下的 `run.log` |

## 命令行

激活 Python 环境后可以直接使用 `ypuddin` 命令（运行 `<启动脚本> shell` 查看激活方法）：

```bash
ypuddin plan     config.toml                   # 检查配置，估算步数、分桶和显存，不执行训练
ypuddin plan     config.toml --device cuda     # 按指定设备检查
ypuddin train    config.toml                   # 训练；Ctrl+C 暂停并保存断点
ypuddin cache    config.toml                   # 只预先编码图片和文字缓存
ypuddin validate config.toml                   # 只校验配置
ypuddin schema                                 # 打印带说明的配置 JSON Schema
ypuddin smoke    --set model.dit_path=…        # 真实权重自检
ypuddin inspect  lora.safetensors              # 查看适配器的元数据和模块
ypuddin convert  lora.safetensors --to comfyui -o out.safetensors
ypuddin merge    --base anima-base.safetensors --adapter lora.safetensors --family anima -o merged.safetensors
ypuddin extract  --base a.safetensors --tuned b.safetensors --algo lokr -o diff-lokr.safetensors
ypuddin serve    --port 8123 --data-root studio_data
```

配置文件支持 TOML 和 JSON。`--set a.b=c` 覆盖单个字段，`--preset preset.toml` 叠加一个预设文件。从断点继续训练：`--set checkpoint.resume=<输出目录>/state-…`。

一个最小的 Anima 配置：

```toml
[model]
family = "anima"
dit_path = "/models/anima-base-v1.0.safetensors"
text_encoder_path = "/models/Qwen3-0.6B-Base"
vae_path = "/models/qwen_image_vae.safetensors"
dtype = "bf16"

[dataset]
sources = [{ path = "/data/chara", repeats = 2 }]
resolutions = [1024]
batch_size = 1
image_fit = "pad"
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

[loop]
epochs = 10
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

各参数的含义见 [训练说明](training.md)，也可以在网页界面中点每个参数旁的“?”查看。
