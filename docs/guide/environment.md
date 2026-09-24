# 运行环境

## 每个平台一套独立环境

每个启动脚本对应一个平台，各自使用独立的 Python 环境，互不共用依赖：

| 机器 | 启动脚本 | 环境目录 |
| --- | --- | --- |
| Windows + NVIDIA | `studio-windows-cuda.bat` | `environment/windows-cuda/venv` |
| Linux + NVIDIA | `studio-linux-cuda.sh` | `environment/linux-cuda/venv` |
| Linux + 海光 DTK | `studio-linux-dtk.sh` | `environment/linux-dtk/venv` |
| Apple Silicon Mac | `studio-macos.command` | `environment/macos-mps/venv` |
| 仅 CPU | Windows `studio-cpu.bat`，Linux／macOS `studio-cpu.sh` | `environment/<windows-cpu 或 linux-cpu 或 macos-cpu>/venv` |

- 只创建实际启动过的平台目录。CUDA 入口找不到 NVIDIA 驱动时停止；CPU 入口不安装 NVIDIA 依赖，训练和队列固定使用 CPU。
- CUDA 与 DTK 入口只支持 x86_64，arm64 机器请用 CPU 入口；MPS 入口需要 Apple 芯片。
- 环境目录里的安装标记记录平台和 CPU 架构。用另一个平台的入口操作这个目录，或目录是符号链接时，启动器会拒绝并说明原因。
- 更新代码后检查当前环境的依赖，必要时补装或修复已识别的兼容问题；扩展安装不会擅自替换基础运行时。`--reinstall` 只重建当前入口的基础环境，不影响其他平台、界面准备的 PyTorch 版本和训练数据。
- 服务启动时不继承外部的 `PYTHONPATH`／`PYTHONHOME`，也不使用用户级 Python 包。
- 旧部署的根目录 `venv/` 可以用 `--profile=legacy` 继续使用。
- 基础环境默认放在源码目录的 `environment/` 下；可在 **设置 → 存储路径 → 基础环境目录** 改到别处（下次从启动脚本启动时生效），或临时用 `--env-root <目录>`。已有环境不会被搬走或删除，新目录首次启动需要重新安装依赖。

多个平台可以共用同一份项目数据，但同一个数据目录同时只能运行一个服务。切换平台前先停止原来的服务。

macOS 使用 PyTorch 自带的 MPS 与 SDPA，也可安装匹配的 Metal FlashAttention。页面按启动平台显示相应选项。

## 切换 PyTorch 版本

**设置 → 运行环境 → PyTorch 版本** 可以准备并切换到其他 PyTorch 版本，流程是“查看计划 → 下载并准备 → 重启切换”：

- 候选版本来自 [PyTorch 官方版本表](https://pytorch.org/get-started/previous-versions/)，按当前平台、系统、NVIDIA 驱动和显卡架构过滤。CUDA 环境只切换 CUDA 构建，CPU 环境只切换 CPU 构建，MPS 环境只切换 MPS 构建。
- 新版本装在数据目录的 `environment/<平台>/runtimes/<操作 ID>/`，原环境一直保留。下载或检查失败不会影响正在使用的 PyTorch。
- 准备完成前要通过依赖检查、服务模块导入，以及在所选设备上的一次小型前向和反向计算。失败的环境不会出现在可切换列表里，日志和原因会保留。
- xFormers、FlashAttention、SageAttention、bitsandbytes 不会复制到新环境，切换后需要重新安装匹配的版本。
- 切换后可以用“恢复原环境并重启”回到原来的环境。

## 重启服务

- 修改监听地址或端口后先保存，再点击“重启服务”，服务会使用已保存的地址；未保存的输入不会生效。切换 PyTorch 环境时的重启保留当前连接地址。
- 重启由服务自己的启动器完成：先结束当前服务进程，再启动新的进程，不会按进程名查找或结束其他 Python 程序。
- 训练、数据处理、版本复制、模型下载、扩展安装或 PyTorch 准备进行中时不能重启，界面会说明原因。
- 新环境启动失败时，启动器自动回到原来的解释器和地址，界面不会把回退显示成切换成功。

“启动时打开浏览器”也在设置中保存，在下次从启动脚本启动时生效；命令行 `--no-browser` 可以临时关闭。路径设置的生效时机见 [目录结构](folders.md#路径设置何时生效)。

## 软件下载源

**设置 → 软件下载源** 选择依赖的下载来源：

- Python 依赖包默认中国科学技术大学，可以换成清华大学、阿里云或 PyPI 官方。
- PyTorch 可选国内镜像自动顺序、阿里云、上海交通大学或官方源。CUDA／CPU 使用 PyTorch 专用安装包源，macOS 使用 Python 包源。
- 设置保存在数据目录的 `settings.json`，界面准备 PyTorch、安装扩展依赖和启动器安装都读取同一设置；启动命令显式指定 `--index` 或 `--mirror` 时以命令行为准。
- 可以关闭自动换源。各来源依次尝试，不混用多个索引；日志记录实际使用的地址。取消安装会立即停止，不会继续尝试下一个来源。
- CUDA 环境的版本约束带 `+cu` 后缀，普通依赖解析不会把 PyTorch 换成 CPU 版本。
- 海光的本地厂商 wheel、模型权重和 FlashAttention 社区发布文件使用各自的下载来源，不受这项设置影响。

## 网络代理

**设置 → 网络代理** 可以选择沿用启动环境、直接连接或自定义 HTTP／HTTPS 代理。模型下载、运行环境中的扩展和依赖安装、Danbooru／Gelbooru 请求都使用这项设置，保存后对新请求生效，不修改系统代理。

- 地址必须是训练服务器能访问到的地址；远程部署时 `127.0.0.1` 指的是训练服务器本身。
- 账号和密码分开填写，保存的密码不会显示出来；需要移除时使用“清除已保存密码”。
- 首次启动、服务还没运行时的依赖安装仍使用启动终端里的代理环境变量。

## 扩展安装

在 **设置 → 运行环境** 中安装或更换扩展（xFormers、FlashAttention、SageAttention、bitsandbytes 等）时，会先列出要安装的 wheel 和版本变化，确认后显示安装日志：

- Torch、CUDA、NumPy 等基础运行时受保护，不会被扩展安装替换。
- 没有兼容的预编译 wheel 时显示原因，不会自动从源码编译。
- 依赖发生变化后需要重启服务，重启前不能启动训练和缓存任务。
- “已安装”和“当前显卡可用”是两回事：安装后会在本机执行前向和反向检测，以检测结果为准。

各平台的注意力扩展见 [注意力加速](attention.md)，海光环境见 [海光 DTK](dtk.md)。
