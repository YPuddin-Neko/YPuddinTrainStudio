# 环境切换与服务重启

设置里现在可以选择 PyTorch 版本和计算后端。操作分成“查看计划 → 下载并准备 → 重启切换”：旧环境一直保留，下载失败或验证失败不会把当前服务的 PyTorch 卸载掉。

## 启动入口

| 机器 | 启动脚本 |
|---|---|
| Windows + NVIDIA | `studio-windows-cuda.bat` |
| Apple Silicon Mac | `studio-macos.command` |
| Linux + NVIDIA | `studio-linux-cuda.sh` |
| CPU 机器 | Windows 用 `studio-cpu.bat`，Linux 用 `studio-cpu.sh` |
| 自动检测或旧部署 | 原来的 `studio.bat` / `studio.sh` 继续可用 |

这些入口复用标准库安装逻辑，各自使用独立的依赖环境：

```text
源码目录/
  environment/profiles/
    windows-cuda/venv/
    linux-cuda/venv/
    macos-mps/venv/
    windows-cpu/venv/
    linux-cpu/venv/
    macos-cpu/venv/
  venv/                         旧部署原样保留
当前数据根/
  environment/profiles/<平台>/
    runtimes/<操作ID>/          本平台准备的其他 Torch 版本
    service/selected.json       本平台当前激活的解释器
    cache/、installer/           本平台的扩展安装工作目录
```

只创建实际启动的平台目录。CUDA 入口找不到 NVIDIA 驱动时停止；CPU 入口不安装 NVIDIA 依赖，自动训练和任务队列固定使用 CPU。macOS 的 CPU 与 MPS 环境也各自独立。

普通 `studio.bat` / `studio.sh` 发现根目录已有 `venv` 时继续使用旧部署，不搬移或重建。没有旧环境时按机器创建对应平台环境。显式平台入口总是使用自己的目录，不能拿旧 `venv` 混装。`--profile=legacy` 可明确使用旧目录。各入口的 `doctor` 和 `shell` 都指向自己选中的环境。

更新代码只给选中环境补齐依赖，保留其已有 Torch、CUDA、NumPy。`--reinstall` 只重建该入口的基础环境，不删除其他平台、准备好的 Torch 版本或训练数据；检测到目录链接或不同平台的安装标记会拒绝操作。扩展安装、修复和卸载只允许执行当前平台、当前解释器生成的计划。启动服务时不继承外部 `PYTHONPATH` / `PYTHONHOME`，不使用用户级 Python 包。

各平台可以共享同一份项目数据，但同一数据根同时只启动一个服务。切换平台先停止原服务，再运行对应入口；共享数据不代表共享依赖。

海光 BWGPU / DTK 当前仅保留“待接入”能力状态，不提供安装脚本、下载或多卡训练可用承诺。后续拿到设备后再验证厂商运行时与分布式训练链路。

macOS 的 PyTorch 安装包本身带有 MPS 支持，所以它的下载源是 PyPI，不是 CUDA wheel 源。CUDA 是 NVIDIA 的计算后端；NCCL 用于 NVIDIA GPU 之间通信。二者在 Mac 上不适用，不表示环境损坏。是否支持单任务多卡训练，由训练器自己的执行能力决定，不能只看 `torch.distributed` 是否存在。

## 可切换版本

本轮收录 PyTorch 2.13.0、2.11.0、2.10.0 的官方组合。每个版本的 CUDA 渠道不同，不把任意 Torch 版本与任意 CUDA 标签拼接。来源为 [PyTorch 官方历史版本表](https://pytorch.org/get-started/previous-versions/)；2.13.0 的 Python 要求另以 [PyPI 元数据](https://pypi.org/pypi/torch/2.13.0/json)核对。

界面按当前部署平台、操作系统、NVIDIA 驱动和显卡架构过滤候选。CUDA 环境只切换 CUDA 构建，CPU 环境只切换 CPU 构建，MPS 环境只切换 MPS 构建；跨平台计划和激活请求会被拒绝。2.11.0 + CUDA 12.8 的“小型 CUDA 训练已验证”仅引用此前记录的小型训练范围；其余候选是官方提供的构建，不宣称本项目所有训练模型都已在这些组合上测试。

准备的环境放在当前数据根的 `environment/profiles/<平台>/runtimes/<操作 ID>/`；旧 legacy 部署仍使用原来的 `environment/runtimes/`，旧记录只属于 legacy。新环境固定 Torch/TorchVision 的组合，保留已安装普通依赖和 NumPy 的版本约束，再解析所需训练依赖。xFormers、FlashAttention、SageAttention、bitsandbytes 不跨版本复制；这些扩展要与新环境重新匹配安装。

准备完成必须通过依赖检查、服务模块导入以及选定设备上的微型前向/反向运算。源码通过 `.pth` 关联，不在源码根创建新的 `ypuddin.egg-info`。失败结果保留原因和日志，可以重新创建准备计划；失败环境不会进入可激活列表。安装过程只报告实际阶段和日志，不用虚构百分比表示大文件下载速度。

## 重启的行为

“重启服务”默认保留当前实际监听地址，包括命令行指定的端口。修改了保存地址后，另一个明确的“应用已保存地址并重启”操作才会改变地址。

服务由自己的启动器管理。重启请求写给自己的 worker，先结束当前 HTTP worker，再由原启动器启动替代 worker。不会按进程名称查找或终止其他 Python 进程。训练、数据处理、版本复制、模型下载、扩展安装或 Torch 环境准备期间，后端拒绝重启，并提供具体原因。文件上传从接收请求体开始保护，随后覆盖登记和后台索引；重启已经开始时，新上传在读取文件前被拒绝。

新环境的解释器无法启动或启动失败时，启动器回退原解释器和原监听地址。前端会检查新 worker 的运行标识和实际选中的环境；回退不显示为“新环境切换成功”。已激活环境可以通过“恢复原环境并重启”返回原环境。退出启动器时也会清理它自己的 worker。

扩展或 Torch 操作的“关闭结果”现在保存 `dismissed_at`。再次打开设置不会重弹同一条已关闭错误；历史入口仍能查看原状态和日志。当前探测结果与历史安装错误是两份独立信息，不相互伪造覆盖。

## 本轮验证

- 后端生命周期、环境管理、启动器以及上传、目录同步、导入进度共 218 项测试通过，包括下载失败保留原环境、验证后才能激活、任务占用拒绝、地址保留、解释器启动失败回退、关闭历史持久化和 SIGTERM 清理。上传保护覆盖没有进度编号的请求、取消和失败后释放、接收与重启竞争、后台索引接续保护。
- 环境管理与重启前端共 27 项测试通过，包括 7 种部署环境显示、计划里的旧/新版本显示、必须另点准备、失败关闭后重开、运行中禁用、等待新 worker、回退提示和显式地址变更。
- 独立本地服务实际完成一次重启：worker ID 改变，命令行端口和数据根保持。向启动器发送 SIGTERM 后两个 owned worker 均退出，返回码 143；没有操作现有 8877 服务。
- 本轮未在用户的实际 venv 安装或切换 PyTorch，未把模拟安装测试写成真实下载安装已通过。真实重启证据在源码外 `remote-testing/environment-lifecycle-20260914/service-process-report.json`。
- 平台隔离使用两个实际创建的标准库 venv 和子解释器验证：每个环境能导入自己的测试包，无法导入另一个环境的测试包，不下载任何依赖。另验证选中环境重建不改变其他平台和 legacy 的文件、跨平台计划/卸载/激活拒绝、直接运行解释器仍能识别归属，以及 CPU 自动调度不误用可见 MPS/GPU。
