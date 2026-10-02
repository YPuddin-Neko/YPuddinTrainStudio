# 安装与部署

## 环境要求

| 项目 | 要求 |
| --- | --- |
| Python | 3.10–3.12 |
| Node.js | 20.19+（20.x）、22.12+（22.x）或更新主版本；仅前端构建需要 |
| NVIDIA | x86_64 Windows / Linux，安装与 PyTorch CUDA 构建兼容的驱动 |
| Apple | Apple Silicon macOS，使用 MPS；当前训练路径使用 FP32 |
| 海光 | x86_64 Linux，匹配的驱动和 DTK 运行库；海光版 PyTorch 可使用已装好的，或由启动器下载 |

模型权重、编码缓存和完整训练状态分别占用磁盘空间。内存与显存需求取决于模型、分辨率、精度和训练方式。

## 启动

```bash
git clone https://github.com/YPuddin-Neko/YPuddinTrainStudio.git
cd YPuddinTrainStudio
```

| 平台 | 命令 |
| --- | --- |
| Windows CUDA | 在 PowerShell 运行 `.\studio-windows-cuda.bat`，或双击该文件 |
| Linux CUDA | `./studio-linux-cuda.sh` |
| macOS Apple Silicon | `./studio-macos.command` |
| Windows CPU | `.\studio-cpu.bat` |
| Linux / macOS CPU | `./studio-cpu.sh` |
| Linux DTK | `./studio-linux-dtk.sh`，海光版 PyTorch 的来源见 [DTK 部署](RUNTIME_DTK.md) |

脚本自动查找 Python，创建当前平台的虚拟环境，安装依赖并按需构建前端。系统没有适用的 Python、但已安装 `uv` 时，常规启动入口可通过 `uv` 下载 Python 3.12。

所选平台环境首次安装时，终端会询问下载源，可选择自动、官方、中科大、清华或阿里，回车默认自动检测可用站点。已有环境后续启动或重建时直接跳过，即使尚未保存下载源设置也不再询问。选择保存在当前数据目录，之后可在“设置 → 软件下载源”中修改。已保存选择、显式传入 `--index` / `--mirror`，或从脚本、服务等非交互环境启动时，也不询问。

普通 Python 包和 PyTorch 使用不同仓库：中科大、清华选项只指定 PyPI，PyTorch 仍自动选择；官方和阿里选项分别优先使用各自的两类仓库。下载失败后继续尝试其他来源。DTK 的基础框架使用厂商包，不受这个 PyTorch 下载源选择影响。

需要构建前端时，从 [Node.js 官网](https://nodejs.org/en/download) 下载 LTS 安装包，保留 npm 和添加到 `PATH` 的选项。安装后重新打开终端，再运行启动脚本。缺少 Node.js 或 npm、依赖安装失败或构建失败时，默认启动会报错并停止；已有与当前源码一致且校验通过的前端构建时，可直接复用，无需 Node.js。仅运行 API 的部署可显式使用 `--no-frontend` 跳过构建。

默认地址为 `http://127.0.0.1:8123/`。注意力扩展、8-bit 优化器依赖的 bitsandbytes 和打标与遮罩所需的 ONNX Runtime 分别在“设置 → 运行环境”的“注意力加速”“LoRA 环境”“打标与遮罩”中单独安装。ONNX Runtime 在 NVIDIA 显卡上使用 GPU 版，装好后不需要重启服务。

## 首次设置

新安装首次打开网页时，会依次引导设置语言、软件下载源、查看存储位置、添加访问密钥、下载模型和查看运行环境。界面默认中文；首页可选择“跳过引导”，访问密钥和模型下载也可单独跳过。完成或跳过后不再自动打开，可从“设置 → 页面设置 → 重新打开引导”再次进入。已有项目或任务的部署不会因更新而强制进入引导。运行环境页可管理 xFormers 和 FlashAttention；安装后需重启训练器生效。

软件下载源默认自动选择，支持手动检测服务器到各来源的响应延迟，也可分别指定 Python 依赖包和 PyTorch 的来源。开启自动换源时，当前来源下载失败会继续尝试其他镜像和官方站点。

## 启动参数

以下示例使用 Linux CUDA 入口。其他入口采用相同参数。

```bash
./studio-linux-cuda.sh --port 8800 --data-root /data/studio --no-browser
```

| 参数 | 作用 |
| --- | --- |
| `--host <地址>` | 监听地址，默认 `127.0.0.1` |
| `--port <端口>` | 监听端口，默认 `8123` |
| `--data-root <目录>` | 数据目录，默认 `studio_data`；在“设置 → 存储路径”保存了新的数据根目录时，下次启动改用该目录 |
| `--env-root <目录>` | 基础 Python 环境的根目录 |
| `--no-browser` | 本次启动不打开浏览器 |
| `--no-frontend` | 跳过前端构建，用于仅运行 API 的部署 |
| `--index=auto`、`official`、`ustc`、`tuna`、`aliyun`、`cn` | 本次使用指定来源：自动、官方、中科大、清华、阿里，或国内镜像优先；未指定时使用已保存的来源 |
| `--mirror` | 等同于 `--index=cn` |
| `--torch=auto`、`cu128`、`cu126`、`cu124`、`cu118`、`cpu` | 首次安装或重建环境时使用的 PyTorch 构建 |
| `--reinstall` | 删除并重建当前入口的基础环境，保留其他环境与数据 |
| `--profile=legacy` | 使用旧版根目录 `venv/` |

显式的地址、端口和环境根目录参数优先于保存的设置。DTK 使用独立的厂商安装流程，不使用 CUDA / CPU 包源安装基础框架。

| 子命令 | 功能 |
| --- | --- |
| `run` | 默认命令，准备环境并启动服务 |
| `doctor` | 输出 Python、PyTorch、设备、可选依赖和前端构建状态 |
| `smoke <配置文件>` | 用配置中的真实权重做短程训练自检：默认训练 3 步，生成一张预览，保存并重新读取训练权重，报告写入 `outputs/smoke/` |
| `build` | 重新构建前端 |
| `dev` | 启动后端和 Vite 开发服务，前端默认端口为 3000 |
| `shell` | 输出当前环境的激活命令 |

## 远程访问

服务没有内置登录认证。远程访问可使用 SSH 转发：

```bash
ssh -L 8123:127.0.0.1:8123 user@training-server
```

随后在本机打开 `http://127.0.0.1:8123/`。使用反向代理时，应由代理提供认证与访问控制；文件浏览和训练接口可以读写配置指定的路径，不应直接开放到公网。

事件流接口为 `/api/events`。Nginx 代理需关闭响应缓冲并设置较长的读取超时，例如 `proxy_buffering off` 和 `proxy_read_timeout 1h`。

## 常驻服务

Linux 可在完成首次安装后使用 systemd。以下路径和账户需替换为实际部署值：

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

Windows 可通过任务计划程序启动对应环境的 `ypuddin.exe`，参数使用 `serve --host 127.0.0.1 --port 8123 --data-root D:\studio_data`，起始目录设为源码目录。

## 更新

在“设置 → 训练器更新”点击“检查更新”，查看当前版本和 GitHub `main` 的最新源码版本及最新提交信息。无法识别当前提交时，不判断是否已是最新。检查使用训练器的网络代理设置。

发现新版本后，点击“更新并重启”。训练器会下载本次检查对应的提交、构建前端，随后停止服务、替换源码并检查依赖。新版本启动成功后，页面自动刷新。更新沿用当前 Python 环境和已保存的下载源设置，保留现有 PyTorch 计算依赖。

Git 安装和源码包安装均支持网页更新；需要通过项目启动脚本或 `ypuddin serve` 启动。训练、数据处理或模型下载进行中时不能更新。有本地源码改动时，需要先保存并处理改动。源码包通过提交标记或 `SOURCE_MANIFEST.json` 识别版本；缺少提交信息的旧包需要手动更新一次。

更新准备失败时继续运行原版；替换源码后安装或启动失败，会尝试恢复原版源码并检查原版依赖。依赖恢复失败时停止启动，按更新日志处理后重新运行启动脚本。更新记录和源码备份保存在数据目录的当前环境 `service/trainer-updates/` 下；Python 依赖安装不提供完整环境回滚。

需要手动更新时：

1. 暂停训练并保存完整状态，停止服务。
2. 备份数据目录和自定义输出目录。
3. Git 安装在源码目录执行 `git pull --ff-only`；压缩包安装将新源码解压到新目录，继续使用原数据目录和模型路径。
4. 重新运行原启动脚本。

启动器会检查依赖及前端源码指纹。构建已过期时需要 Node.js；路径变更不会自动迁移已有数据，迁移步骤见 [存储与备份](STORAGE.md)。

## 常见故障

| 现象 | 检查项 |
| --- | --- |
| CUDA 不可用 | 执行对应入口的 `doctor`，核对 NVIDIA 驱动与 PyTorch 构建 |
| `torchvision::nms does not exist` | 检查 Torch 与 TorchVision 的版本及 CUDA 构建后缀；重新运行原启动脚本执行兼容检查 |
| `no kernel image is available` | 检查 PyTorch CUDA 构建是否包含当前显卡架构 |
| 页面资源 404 或更新后显示旧界面 | 执行启动入口的 `build` 子命令 |
| 找不到 Node.js 或 npm | 安装上方链接的 Node.js LTS，保留 npm 和 `PATH` 选项，重新打开终端后运行启动脚本 |
| 前端构建拒绝 Node.js 版本 | 使用表中支持的 Node.js 版本后重新构建 |
| 下载失败 | 核对下载源、服务器代理、访问密钥及模型仓库权限 |
| 端口被占用 | 使用 `--port` 更换端口，或检查已有服务进程 |
| 训练显存不足 | 降低分辨率或批大小，检查缓存、梯度检查点和分块换出设置 |
| 恢复时报数据指纹不一致 | 核对图片、标签、遮罩及数据配置是否改动；从权重开始新任务与断点恢复的区别见 [训练配置](TRAINING.md#保存与恢复) |

任务错误和运行日志在任务详情页查看。
