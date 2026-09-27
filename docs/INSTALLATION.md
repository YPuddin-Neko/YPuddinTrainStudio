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

默认地址为 `http://127.0.0.1:8123/`。注意力扩展和 8-bit 优化器依赖的 bitsandbytes 通过“设置 → 运行环境”单独安装。

## 启动参数

以下示例使用 Linux CUDA 入口。其他入口采用相同参数。

```bash
./studio-linux-cuda.sh --port 8800 --data-root /data/studio --no-browser
```

| 参数 | 作用 |
| --- | --- |
| `--host <地址>` | 监听地址，默认 `127.0.0.1` |
| `--port <端口>` | 监听端口，默认 `8123` |
| `--data-root <目录>` | 数据目录，默认 `studio_data` |
| `--env-root <目录>` | 基础 Python 环境的根目录 |
| `--no-browser` | 本次启动不打开浏览器 |
| `--no-frontend` | 跳过前端构建，用于仅运行 API 的部署 |
| `--index=auto`、`cn`、`official` | 指定依赖包下载源；未指定时读取保存的下载设置 |
| `--torch=auto`、`cu128`、`cu126`、`cu124`、`cu118`、`cpu` | 首次安装或重建环境时使用的 PyTorch 构建 |
| `--reinstall` | 删除并重建当前入口的基础环境，保留其他环境与数据 |
| `--profile=legacy` | 使用旧版根目录 `venv/` |

显式的地址、端口和环境根目录参数优先于保存的设置。DTK 使用独立的厂商安装流程，不使用 CUDA / CPU 包源安装基础框架。

| 子命令 | 功能 |
| --- | --- |
| `run` | 默认命令，准备环境并启动服务 |
| `doctor` | 输出 Python、PyTorch、设备、可选依赖和前端构建状态 |
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

1. 暂停训练并保存完整状态，停止服务。
2. 备份数据目录和自定义输出目录。
3. 在源码目录执行 `git pull`。
4. 重新运行原启动脚本。

启动器会检查依赖及前端源码指纹。构建已过期时需要 Node.js；路径变更不会自动迁移已有数据，迁移步骤见 [存储与备份](STORAGE.md)。

## 常见故障

| 现象 | 检查项 |
| --- | --- |
| CUDA 不可用 | 执行对应入口的 `doctor`，核对 NVIDIA 驱动与 PyTorch 构建 |
| `torchvision::nms does not exist` | 检查 Torch 与 TorchVision 的版本及 CUDA 构建后缀；重新运行原启动脚本执行兼容检查 |
| `no kernel image is available` | 检查 PyTorch CUDA 构建是否包含当前显卡架构 |
| 页面资源 404 或更新后显示旧界面 | 执行启动入口的 `build` 子命令 |
| 前端构建拒绝 Node.js 版本 | 使用表中支持的 Node.js 版本后重新构建 |
| 下载失败 | 核对下载源、服务器代理、访问密钥及模型仓库权限 |
| 端口被占用 | 使用 `--port` 更换端口，或检查已有服务进程 |
| 训练显存不足 | 降低分辨率或批大小，检查缓存、梯度检查点和分块换出设置 |
| 恢复时报数据指纹不一致 | 核对图片、标签、遮罩及数据配置是否改动；从权重开始新任务与断点恢复的区别见 [训练配置](TRAINING.md#保存与恢复) |

任务错误和运行日志在任务详情页查看。
