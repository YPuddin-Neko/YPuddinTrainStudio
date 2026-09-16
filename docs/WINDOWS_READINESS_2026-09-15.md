# Windows 兼容性预检与真机验收

当前没有可用的 Windows 真机。本次在 macOS 上检查 Windows 分支，并运行可移植的真实训练进程；这可以提前发现应用层问题，不能证明 Windows 驱动、CUDA 算子或双卡通信已经通过。没有启动云端 CI、安装 Windows 虚拟机或改动现有训练数据。

可随源码查看的 [验证摘要](validation/WINDOWS_READINESS_2026-09-15.json) 包含测试计数、源码散列及原始证据散列；训练产物保存在仓库外。

## 两种多卡用途

| 用途 | 当前产品路径 | 验证边界 |
| --- | --- | --- |
| 一张卡训练 A，另一张卡训练 B 或测试模型 | 原生 Windows CUDA，每个任务选择一张卡，队列按设备分别调度 | 本地模拟覆盖 Windows 启动与调度分支；未来真机仍需检查 CUDA 与进程释放。 |
| 同一个大模型由两张卡共同容纳、训练 | Linux CUDA／DTK 的 FSDP2；Windows 主机可评估 WSL2 Linux CUDA 路线 | 原生 Windows 单任务多卡入口当前明确禁用；本次没有解除，也没有实测 WSL2 的 NVIDIA 双卡。 |

原生 Windows 的限制来自当前训练器只接入 NCCL 兼容的 CUDA 分布式路径。PyTorch 2.11 的官方文档仍将 Windows distributed 标为 prototype，且 Windows 不提供 NCCL；该版本已列出 Gloo 的部分 GPU 集体通信支持，因此不能把产品限制解释成“Gloo 永远不能做 GPU 分片”。原生 Windows＋Gloo 的 FSDP2、DTensor、HF Adafactor 和保存恢复组合需要单独移植与验收。[PyTorch 分布式后端](https://docs.pytorch.org/docs/2.11/distributed.html#backends)

NVIDIA 提供 WSL2 CUDA 支持。优先评估 WSL2 是因为它与现有 Linux CUDA 训练路径一致，并不代表本项目已通过 WSL2 实测。[CUDA on WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

## 本次发现与修正

环境能力报告原先只按操作系统、PyTorch 和可见显卡判断多卡训练能力。若选择了 CPU 环境而 CUDA 构建仍可发现显卡，会误报支持多卡训练。现在同时检查当前运行环境：CPU 环境不会声明支持 GPU 多卡训练；底层 PyTorch 构建与驱动信息仍如实显示。

测试使用模块局部的 Windows 平台替身、模拟显卡清单和子进程接口；数据库、任务快照和控制文件使用真实的临时目录。调用 `CREATE_NEW_PROCESS_GROUP` 或 `taskkill` 的模拟断言不等于在 Windows 内核执行过这些调用。

| 本地回归 | 结果 | 实际范围 |
| --- | --- | --- |
| 引导、环境管理、wheel 兼容性与注意力后端 | 240 通过 | 包括 Windows 平台选择、环境隔离、兼容 wheel、版本/校验和拒绝、代理失败等；未实际安装 Windows 包。 |
| Windows 执行分支与相关队列回归 | 66 通过、3 项未选择 | 含新增 Windows 启动/队列 12 项及显式运行环境矩阵 13 项；其余是相关已有回归。未选择的 3 项是已有 POSIX 真进程/torchrun 用例。 |
| 服务启动、重启与退出 | 34 通过 | 含 Windows 虚拟环境转发进程的身份校验、重启凭据更新等；仍是本地回归，不是 Windows 内核实测。 |
| 新增运行检查工具 | 12 通过 | 含真实双 CPU 训练、两层 Python 转发启动、CUDA 不可用时拒绝回退、超时清理、另一任务失败、损坏报告和已有目录保护。 |

上述测试部分交叉，不能把它们相加为独立覆盖数量。Windows 队列回归检查了中文和空格路径作为独立命令参数、进程组标志、继承的数字/UUID 显卡掩码、跳过等待另一张卡的任务、进程实际退出前不释放占用、保存/暂停/取消只控制当前任务，以及过期取消回调不能杀掉新的进程。三个 `.bat` 入口的 ASCII 与 CRLF 静态检查通过；没有执行 Windows 的 `cmd.exe`。

原始证据：[兼容性日志](../../remote-testing/windows-preflight-20260915/existing-compatibility.log)、[JUnit](../../remote-testing/windows-preflight-20260915/existing-compatibility.xml)、[Windows 分支回归摘要](../../remote-testing/windows-preflight-20260915/windows-execution-contract-summary.json)、[对应日志](../../remote-testing/windows-preflight-20260915/windows-execution-contract.log)。

服务启动/重启证据：[日志](../../remote-testing/windows-preflight-20260915/service-lifecycle.log)、[JUnit](../../remote-testing/windows-preflight-20260915/service-lifecycle.xml)。

新工具另外通过完整命令行执行：两个独立 CPU 进程各训练 3 步，首步屏障检查通过，预览可读取、权重发生更新且与内存中的训练产物一致，两个进程均正常退出。运行宿主为 macOS，报告明确记录 `windows_cuda_validated=false`。工具回归还检查了 Windows 虚拟环境可能增加一层转发进程的情形；所属进程识别按祖先 PID 和创建时间核验，不依赖直接父进程编号。

对应证据：[12 项工具回归](../../remote-testing/windows-preflight-20260915/runtime-check-tests-final.log)、[双 CPU 命令行报告](../../remote-testing/windows-preflight-20260915/runtime-tool-cpu-01/report.json)。

## 有 Windows 机器后的运行方式

先用项目的 `studio-windows-cuda.bat` 建立并检查 Windows CUDA 环境。在项目目录打开 PowerShell，用该环境的 Python 执行独立检查：

```powershell
.\environment\profiles\windows-cuda\venv\Scripts\python.exe -m ypuddin.tools.runtime_check --device cuda:0 --device cuda:1 --out ".\studio_data\runtime-check-首次双卡"
```

若设置中切换过 PyTorch 环境，使用“运行环境”显示的当前 Python 路径替换上面的路径。输出目录必须尚不存在；再次运行请换一个新目录名。工具只在该目录生成合成小数据、日志、预览和权重，不安装依赖、不下载模型，也不连接或修改正在运行的服务。

工具运行两个独立训练进程，各自用真实产品训练器训练 Toy 小模型。两边完成首步后再同时继续，以确认两个进程都已实际训练；随后检查预览、导出的权重及退出状态。请求 CUDA 时不可回退 CPU。它验证设备与独立进程执行，不额外验证服务 API 队列、正式模型容量、训练画质或 FSDP 通信。

本地开发可使用 `--device cpu --device cpu` 跑相同的进程和文件流程。报告会记录实际宿主与设备；CPU 通过不能记为 Windows／CUDA 通过。

需要双卡分片时，在 WSL2 内使用 Linux Python 和 `studio-linux-cuda.sh`，另建 Linux CUDA 环境，模型路径使用 WSL 可见的 Linux 路径。不要复用 Windows 的虚拟环境。先按 [FSDP 指南](FSDP_TRAINING_2026-09-15.md) 检查通信和容量，再以正式模型执行连续训练、冷进程恢复、导出重载和取消退出验收。早期 Krea2 海光 BF16 用例曾有逐位恢复差异；后续正式产品配方已通过，详见 [2026-09-16 海光验证](DTK_BF16_2026-09-16.md)。这不替代 Windows 或 WSL2 的独立显卡验证。

## 真机仍需补验

- 原生 Windows 的 CUDA 驱动、显卡编号隔离、BF16／FP16 运算、扩展 wheel 加载和真实进程树回收。
- 两个不同任务通过服务队列并行训练，以及训练与模型测试并行、分别暂停或取消。
- WSL2 的两卡 NCCL 集体通信、FSDP 正式模型容量、冷进程恢复、产物重载、退出后显存释放和长期训练稳定性。

本次预检不会把以上项目标为通过。已有海光验收见 [交付报告](DELIVERY_REPORT_2026-09-15.md)。
