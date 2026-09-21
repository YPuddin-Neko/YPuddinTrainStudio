# 海光 DTK 独立环境

海光 Linux 服务器使用 `studio-linux-dtk.sh`。依赖目录固定为源码下的 `environment/linux-dtk/venv`，扩展记录、安装缓存和服务状态放在当前数据根的 `environment/linux-dtk`。它不会使用根目录的旧 `venv`，也不会修改 CUDA、MPS 或 CPU 环境。

## 当前能用到哪一步

以下是 2026-09-14 在 Ubuntu 22.04、两张海光 BW GPU 上的实际结果。当前验证组合是 **DTK 26.04 / 厂商 PyTorch 2.7.1**。详细范围与新增遥测进度见 DTK 验收记录（本地验收记录），正式训练数据见 六组精简 JSON（本地验收记录）。

| 检查项目 | 当前结果 |
| --- | --- |
| Krea2 Raw 正式权重 LoKr | BF16、512×512、8 步；SDPA / FlashAttention / xFormers 各测单卡与双卡，6/6 训练、保存、采样和严格恢复通过 |
| 四个模型族的微型权重 | SDPA、FP32、64×64、单卡，20/20 通过；包含全量微调和适配器训练，范围见下节 |
| FlashAttention / xFormers 运算 | 两张卡、FP16 / BF16、head dimension 64 / 128，16/16 前向、反向和有限值检查通过 |
| 默认独立启动环境 | 从干净 venv 安装、doctor、依赖检查通过；补齐厂商 Triton 及 DTK 库路径 |
| 扩展管理 | Flash 官方下载与修复安装、xFormers 浏览器上传安装均完成真实内核核验，保留原有 Torch / TorchVision / Triton |
| Python 缺少 ensurepip | 已在该服务器实测从本地 pip wheel 创建独立环境；宿主 pip 版本和安装文件未改变 |
| 双卡服务控制 | 小模型经产品 API 完成反向设备顺序、暂停、恢复、取消，所属进程退出且队列配置保持不变 |

正式 Krea2 六组均比较了全部可训练权重、预览图、优化器、调度器、进度、采样顺序及每个进程的随机状态，恢复结果一致。最高张量分配约 45.01 GiB，缓存预留最高约 51.64 GiB，两者不同。这个短训矩阵不代表全参数大模型训练或 DDP 速度优势。

旧 DTK 25.04.1 / Torch 2.4.1 仍存在 Hugging Face Hub / Diffusers 冲突，且 xFormers 缺少需要的 `_symmetric_memory` API；早期单卡恢复不一致也已在记录中保留。不能把旧环境状态、新环境正式训练和下面的微型权重结果混用。Klein 的当前厂商 Flash 缺少 Diffusers 所需接口，会提前给出明确错误；请选择 SDPA。

## DTK 26.04 小模型实卡验证

独立环境使用 PyTorch 2.7.1、HIP 6.3.26093、Diffusers 0.40.0、Transformers 5.17.0。测试只使用第二张 BW，采用 SDPA、FP32、64×64 合成图片、批量 1，每例训练 2 步。模型是实际架构的缩小版本，Qwen Image VAE 也使用缩小测试实例；没有使用正式模型权重。

| 模型族 | 全量微调主模型 | 全量微调文本编码器 | 两者同时微调 | LoRA | LoKr |
| --- | --- | --- | --- | --- | --- |
| Anima | 通过 | 通过 | 通过 | 通过 | 通过 |
| Krea2 | 通过 | 通过 | 通过 | 通过 | 通过 |
| SDXL | 通过 | 通过（CLIP-L/G） | 通过 | 通过 | 通过 |
| Klein（Flux2） | 通过 | 通过 | 通过 | 通过 | 通过 |

20 例均检查了训练参数实际更新、需要冻结的底模保持不变、产物重新加载后逐元素一致、从第 1 步状态恢复后与连续训练逐元素一致，以及训练预览和所选产物的 XYZ 图生成。续训最大绝对差均为 0。最高张量显存分配约 314 MiB、最高缓存预留 430 MiB；这是缩小模型的占用，不能用于估算正式模型的显存需求。

脚本为源码外测试目录中的 `remote-testing/dtk-20260914/full-training-gpu-smoke.py`。原始报告与日志位于同目录下的 `tiny-matrix-gpu1-20260914085100812/`，报告明确记录 `synthetic_reduced_architectures=true`、`production_weight_validation=false`。运行进程已结束，第二张卡空闲显存恢复至运行前的 65198 MiB。此矩阵没有验证 DDP，也没有选择 FlashAttention 或 xFormers。

## 准备厂商运行时

使用机器已经安装并与驱动匹配的 DTK 运行时。启动脚本默认读取 `/opt/dtk`；其他位置通过 `DTK_ROOT` 指定，并将同一路径传给子进程的 `DTKROOT`、`ROCM_PATH` 和 `HIP_PATH`（后者指向 `DTK_ROOT/hip`）。不会调用 `ldconfig`、安装驱动或改系统配置。

脚本为当前进程及其子进程设置运行库、链接库、编译器和头文件搜索路径，只加入实际存在的目录，并把调用者已有路径保留在后面。运行库包括 `lib`、`lib64`、`hip/lib`、`llvm/lib`、`dcc/lib`、`dcc/gcvm/lib`、`dcc/comgr/lib` 及相应的顶层组件目录，还有 `dushmem/lib`、`opencl/lib`、`.hyhal` 和 `/opt/hyhal` 的库目录。DTK 26.04 的 `libomp.so` 位于 `dcc/lib`，只设置顶层 `lib` 会导致 Torch 无法导入；旧版不存在的目录会自动跳过。编译器使用该运行时的 `bin`、`llvm/bin`、`dcc/bin`、`hip/bin` 等现有目录，原有环境变量不会写回终端配置或系统文件。

PyTorch 必须使用匹配 DTK、CPU 架构及 Python 版本的厂商构建。普通 PyPI 的 CUDA wheel 不能用于 DTK。当前项目要求 Python 3.10–3.12、PyTorch ≥ 2.4、TorchVision ≥ 0.19；必须同时准备厂商 Torch 与 TorchVision，不能只替换其中一个。

已准备好的独立 DTK venv 可以直接启动。没有环境时，把匹配厂商 wheel 及其依赖放入一个本地目录，然后执行：

```bash
DTK_ROOT=/opt/dtk \
YPUDDIN_DTK_PYTHON=/path/to/matching/python3 \
./studio-linux-dtk.sh --dtk-wheelhouse=/path/to/vendor-wheels --no-browser
```

本地集合必须包含匹配的 `torch-*.whl`、`torchvision-*.whl` 及它们需要的 wheel。原生包安装使用 `--no-index`，不从其他来源寻找替代品。安装完成后校验 Torch 是 HIP 构建；CPU 或 CUDA 构建会停止后续安装。

需要 FlashAttention 时，还应在同一目录放入配套的厂商 `triton-*.whl`。启动器核对其文件名、包名、版本和 DTK 厂商标签后，一并离线安装；已有 Torch / TorchVision 但缺少 Triton 的环境，也可用相同启动参数补齐。已安装的原生版本会保留，不因目录中放了新版 wheel 就升级。未提供 Triton 时启动器不会强装它，FlashAttention 管理页会提示缺少配套依赖。仅把 FlashAttention 或 xFormers wheel 放进这个目录不会安装该扩展，需要随后在设置中选择官方包或使用“上传安装”。

**选择 SDPA 也要验证厂商运行库。** DTK 26.04 / 厂商 Torch 2.7.1 的部分 BF16 SDPA 路径会使用 FlashAttention 动态库；只通过框架导入或 FP32 小样本检查，不能保证这些路径可用。缺少配套扩展时可能出现 `no matching libraries found for flash_attn_2_cuda`。设置中的 SDPA 检测会分别检查 FP16、受支持的 BF16 及两组注意力形状的前向、反向和有限值；训练路径也会将这个特定错误说明为缺少厂商 FlashAttention 库，并提示安装匹配包、重启和重新检测。它不会据此切换精度或把所有 GPU 错误当成依赖问题。检测通过的范围以实际列出的精度和形状为准。

常规训练器依赖仍从所选 Python 包源获取，并对已有 Torch、TorchVision、Triton、NumPy 和厂商原生依赖使用精确版本约束。依赖冲突会报错，不会自动替换原生环境。DTK Torch 2.4 使用 `transformers<5`，避免安装成功却被 Transformers 关闭 Torch 后端；已有成功标记也会复查这项兼容性。

这项 Transformers 约束只修复其中一个已知冲突。旧环境的 Hugging Face Hub / Diffusers 依赖及 Torch 2.4 / Diffusers 0.40 接口问题仍需处理。各环境已通过的训练范围见文首；不能仅凭依赖安装完成判定。当前界面会展示环境限制，并提供需要核对驱动与具体训练范围的配套参考。

`--reinstall` 会删除当前选中平台的 venv，再创建新环境。它要求预先提供本地厂商 wheel 集合，但文件预检不能保证后续依赖一定安装成功。测试新版 DTK / Torch 时，应使用另一份源码与环境目录，保留当前可运行环境；不要把重建参数当作版本切换按钮。环境目录是符号链接或安装标记属于其他平台时，会拒绝修改。

## Ubuntu 厂商 Python 缺少 ensurepip 时

有些服务器的 Python 可以运行，但没有 `ensurepip`，因此普通 `python -m venv` 无法完成创建。DTK 启动入口按以下顺序处理：已有 `uv` 时优先使用它；否则检查 Python 自带的 `ensurepip`。两者都没有时，使用宿主现有 pip 的 `--python` 功能，把本地 pip wheel 安装进新建的独立 venv。

这种情况下，在 `--dtk-wheelhouse` 指定的目录里额外放一个支持当前 Python 的 `pip-*.whl`。可在能联网的另一台电脑上执行下方下载命令，再把该 wheel 复制过来；它只下载 pip，不安装到那台电脑。

```bash
python -m pip download --only-binary=:all: --no-deps pip --dest pip-wheel
```

目标服务器的宿主 pip 需要支持 `--python`（pip 22.3 或更新）。以本次服务器的 `/usr/bin/python3` 为例，准备好本地 wheel 后直接运行：

```bash
DTK_ROOT=/opt/dtk \
YPUDDIN_DTK_PYTHON=/usr/bin/python3 \
./studio-linux-dtk.sh --dtk-wheelhouse="$PWD/environment/vendor-wheels" --no-browser
```

入口会使用 `venv --without-pip` 创建新环境，再以 `--no-index --no-deps` 安装本地 pip，并验证 pip 位于目标环境中。缺少本地 pip wheel 或宿主 pip 太旧时，会在创建前说明缺少什么；不会执行 apt、升级宿主 pip、安装驱动或触碰其他平台的 venv。已有 `environment/linux-dtk/venv` 仍按原路径启动，不需要重新创建它。

## 检查与使用

```bash
DTK_ROOT=/opt/dtk ./studio-linux-dtk.sh doctor
DTK_ROOT=/opt/dtk ./studio-linux-dtk.sh --host 127.0.0.1 --port 8123 --no-browser
```

上面的 `/opt/dtk` 应替换为创建该环境时选定的运行时目录。后续启动也要沿用匹配的 `DTK_ROOT`；改变这个变量不会转换 venv 里的 Torch、Triton 或其他原生包。新环境创建时可用 `YPUDDIN_DTK_PYTHON` 指定基础 Python；已有默认 venv 时，启动器优先使用该 venv，不会切换到另一个解释器。

HIP PyTorch 沿用 `torch.cuda` Python 接口，所以代码里的 `cuda:0` 可以表示第一张海光卡；判断实际后端要看 HIP 构建和实测设备，不能只根据这个接口名称判断为 NVIDIA。环境接口会分别报告 HIP 版本和实际计算后端。

设置中的 CUDA / CPU Torch 切换以及 NVIDIA 扩展构建不适用于这个环境。DTK 的 xFormers 与 FlashAttention 管理界面改用 [SourceFind 官方厂商目录](https://download.sourcefind.cn:65024/4/main/)：显示当前版本匹配结果，提供官方包下载与经校验的离线上传。下载显示实际字节、速度和剩余时间，可在安装前取消。安装计划会展示选中的扩展及需要补齐的普通 Python 依赖，保留已有包版本；厂商 Torch、Triton 不会被重新安装或替换。具体包及校验规则见 [DTK 注意力扩展](DTK_ATTENTION.md)。

“版本匹配”只表示可以准备安装。包完成安装后仍要在当前卡上执行注意力前向、反向与有限数值检测；只有通过后才显示可用。失败记录可以关闭，运行状态不把历史安装结果当成当前能力。默认注意力仍由训练参数明确选择。

需要代理时，在“设置 → 网络代理”选择沿用启动环境、直接连接或自定义 HTTP / HTTPS 代理。模型下载、环境管理中的厂商包与 pip 依赖、Danbooru / Gelbooru 正则集请求共用该设置，保存后对新请求生效，不修改系统代理。地址必须是训练服务器能够访问的地址；远程部署中的 `127.0.0.1` 指训练服务器。账号和密码分别填写，已保存的密码不会回显；需要移除时使用“清除已保存密码”并保存。TLS、下载来源和文件校验仍会执行。

首次启动、服务尚未运行时的依赖安装仍使用启动终端的代理环境变量；完成启动后，再通过上述设置管理服务内的下载请求。

启动隔离、框架导入、单卡运算、双卡通信、训练器多卡执行是不同的验证项目。真实硬件的训练、保存、续训、采样与通信结果以对应验收报告为准，不由启动脚本是否成功推断。

## 三类安装包分别做什么

| 安装层 | 用途 | 官方来源 | 安装方式 |
| --- | --- | --- | --- |
| 海光驱动 | 让 Linux 内核识别 GPU，负责设备与内核通信 | [驱动目录](https://download.sourcefind.cn:65024/6/main) | 由机器维护人员按发行版、内核、卡型和兼容表手动安装；训练器不执行驱动安装或重启 |
| DTK 用户态运行库 | HIP、数学库、编译器等运行文件；不是 Python 包 | [DTK 版本目录](https://download.sourcefind.cn:65024/1/main) | 使用匹配系统的包，放在独立路径，通过 `DTK_ROOT` 选择 |
| 厂商 Python wheel | PyTorch、TorchVision、Triton、FlashAttention、xFormers | [AI 软件包目录](https://download.sourcefind.cn:65024/4/main/) | 基础框架放进独立 venv；已核查的注意力扩展可在前端自动下载或上传安装 |

先核对 [DTK 与驱动配套表](https://download.sourcefind.cn:65024/file/1/DTK%E4%B8%8E%E9%A9%B1%E5%8A%A8%E7%89%88%E6%9C%AC%E9%85%8D%E5%A5%97%E5%85%B3%E7%B3%BB%E8%A1%A8.md)。这个表列出的 DTK 26.04 驱动要求为 `>=6.3.30-V1.4.1a`，卡型包含 BW。本机 `6.3.31-V1.5.3.beta` 已完成上面的运行验证，但这不代替厂商对 beta 驱动的配套确认，也不能只按版本数字为其他机器推荐。DTK 26.04.1 的配套要求不同，不能把 26.04 与 26.04.1 混用。

## Ubuntu 22.04 / x86_64 / Python 3.11 的已测组合

已核对官方包内部 METADATA 和版本文件的组合如下。它已完成上面的正式 Krea2 六组矩阵、20 例微型权重测试和独立默认环境安装；其他模型的正式权重、其他驱动及参数组合仍需分别验证。

| 组件 | 配套版本 |
| --- | --- |
| DTK | 26.04，Ubuntu 22.04 x86_64 用户态包 |
| PyTorch | `2.7.1+das.opt1.dtk2604` |
| TorchVision | `0.22.0+das.opt1.dtk2604.torch271` |
| Triton | `3.1.0+das.opt1.dtk2604.torch271` |
| FlashAttention | wheel 版本 `2.8.3+das.opt1.dtk2604.torch271` |

后三个 wheel 的元数据均明确要求 Torch 2.7.1。FlashAttention 的包内公开版本仍为 2.6.1，并缺少 Diffusers 使用的 `_wrapped_flash_attn_forward/backward` 接口；Krea2 Flash 通过不代表 Klein Flash 可用。界面只在系统发行版、架构、Python ABI 对应上述组合时显示这些具体下载链接，其他机器显示各类官方目录供匹配选取；驱动仍按厂商配套要求核对。

手动准备用户态运行库时，可以保留现有 `/opt/dtk`，在另一份源码目录内进行：

1. 下载 [Ubuntu 22.04 的 DTK 26.04 压缩包](https://download.sourcefind.cn:65024/file/1/DTK-26.04/Ubuntu22.04/DTK-26.04-Ubuntu22.04-x86_64.tar.gz) 和 [配套 MD5 文件](https://download.sourcefind.cn:65024/file/1/DTK-26.04/Ubuntu22.04/DTK-26.04-Ubuntu22.04-x86_64.tar.gz.md5)，按厂商校验文件核对下载完整性。
2. 将压缩包解到这份源码的 `environment/toolkits/`，得到 `environment/toolkits/dtk-26.04/`。官方包是运行文件目录，厂商安装手册支持解压后配置环境，并允许使用 `/opt` 以外的目录。`environment/` 属于本地运行目录，不打进源码包。
3. 下载表中 Torch、TorchVision、Triton 的同组 `cp311-cp311-manylinux_2_28_x86_64.whl`，连同其依赖放进 `environment/vendor-wheels/`。不要把其他 DTK 或 Torch 构建放入这组目录；本地 wheel 集合应由自己按官方来源准备，启动器不会代替你确认驱动配套。
4. 用下方命令让启动器创建 Python 3.11 的独立 venv，并从本地集合安装厂商 Torch、TorchVision 与 Triton。需要 FlashAttention 时，随后在扩展管理页选择同组官方包，或上传下载好的同一个 wheel。

```bash
DTK_ROOT="$PWD/environment/toolkits/dtk-26.04" \
YPUDDIN_DTK_PYTHON=/path/to/python3.11 \
./studio-linux-dtk.sh --dtk-wheelhouse="$PWD/environment/vendor-wheels" --no-browser
```

已有可运行环境时，在另一份源码和环境目录内准备这组版本，避免覆盖原 venv。驱动不符合要求时，先由机器维护人员处理；上述启动命令不安装驱动。下载匹配 wheel 后，前端的“上传安装”会继续检查文件来源、元数据和实卡能力，不能用于强装尚未纳入核查范围的其他构建。
