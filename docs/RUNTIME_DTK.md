# 海光 DTK 部署

海光环境使用 `studio-linux-dtk.sh`，通过厂商 HIP 版 PyTorch 运行。基础环境默认位于 `environment/linux-dtk/venv`，可用 `--env-root` 指定其他根目录。

## 安装前提

| 组件 | 要求与来源 |
| --- | --- |
| GPU 驱动 | 按服务器发行版、内核、卡型及 DTK 版本安装；[驱动目录](https://download.sourcefind.cn:65024/6/main) |
| DTK 运行库 | 与驱动匹配，默认从 `/opt/dtk` 加载；[DTK 目录](https://download.sourcefind.cn:65024/1/main) |
| Python | 3.10–3.12，与厂商 wheel 的 Python ABI 一致 |
| Torch、TorchVision | 同组 DTK 构建；[厂商软件包目录](https://download.sourcefind.cn:65024/4/main/) |
| Triton | 与所选 Torch / DTK 构建匹配 |

驱动和 DTK 运行库由服务器维护人员安装。训练器不执行系统驱动安装。不同 DTK 小版本按厂商配套关系选择，不能仅凭相近版本号混用。

## DTK 安装指南

“设置 → 运行环境 → DTK 安装指南”显示当前系统、DTK、Python、PyTorch 和驱动版本。服务器为 Ubuntu 22.04 x86_64、Python 3.11 时推荐 DTK 26.04（要求驱动 6.3.30-V1.4.1a），并给出 DTK 安装包、校验文件和 PyTorch、TorchVision、Triton、FlashAttention 的下载链接；其他系统从页面上的 DTK 版本目录、驱动下载目录和驱动配套表中选择。页面只提供下载，DTK 和驱动需在服务器上安装。

## 厂商包组合

项目内置目录包含以下 DTK 26.04 组合，适用于对应的 Linux x86_64 / Python 3.11 环境：

| 组件 | 构建 |
| --- | --- |
| Torch | `2.7.1+das.opt1.dtk2604` |
| TorchVision | `0.22.0+das.opt1.dtk2604.torch271` |
| Triton | `3.1.0+das.opt1.dtk2604.torch271` |
| FlashAttention | `2.8.3+das.opt1.dtk2604.torch271` |
| xFormers | `0.0.33+das.opt1.dtk2604.torch251`（纯 Python 包，需要 Torch 2.5 及以上和 FlashAttention） |

Torch、TorchVision、Triton 和其依赖放在同一 wheel 目录。其他版本需使用相应的厂商配套构建。普通 PyPI CUDA 包不能替代 HIP 版 Torch。

## 首次安装

启动器按以下顺序决定海光版 PyTorch 的来源：

1. **所选 Python 已装好海光版 PyTorch**（官方容器镜像、conda 环境或已有虚拟环境）：新环境直接使用这份安装，不再下载。装在系统 Python 或 conda 环境里的，新环境通过 `--system-site-packages` 继承其已装的包；装在另一个虚拟环境里的，新环境链接到该环境的包目录。
2. **没有预装**：DTK 26.04 + Python 3.11（x86_64，glibc 2.28 及以上）从光合社区下载已核对的 Torch 2.7.1、TorchVision 0.22.0 和 Triton 3.1.0（约 640 MB），大小与 SHA-256 一致才安装。文件保存在环境目录的 `vendor-wheels/`，重建时不重复下载。
3. **其他 DTK / Python 组合，或服务器不能联网**：用 `--dtk-wheelhouse` 提供本地 wheel 目录。

```bash
# 官方镜像里，或 Python 已装好海光版 PyTorch
DTK_ROOT=/opt/dtk ./studio-linux-dtk.sh --no-browser

# 指定 Python
DTK_ROOT=/opt/dtk YPUDDIN_DTK_PYTHON=/path/to/python3.11 ./studio-linux-dtk.sh --no-browser

# 离线或未核对的组合
DTK_ROOT=/opt/dtk ./studio-linux-dtk.sh --dtk-wheelhouse=/data/dtk-wheels --no-browser
```

使用已装的 PyTorch 时，厂商 Torch、TorchVision、Triton 的完整安装包版本保持不变，训练器依赖只装进自己的环境，不修改原环境。运行时的 `torch.__version__` 可能显示 `2.5.1`，安装包版本则是 `2.5.1+das.opt1.dtk2604`；启动器按安装包版本校验和锁定，并检查 HIP 构建。原环境里的 PyTorch 之后被替换，启动时会提示；被删除则停止启动。

训练器要求 NumPy `>=1.26`。共享环境中的 NumPy 1.x 低于 1.26 时，启动器只在项目虚拟环境中安装 `numpy>=1.26,<2`，保留宿主的 NumPy，不跨到 NumPy 2。项目内的包优先于宿主包加载；满足要求的现有版本继续锁定。安装后检查 NumPy 与 PyTorch 的双向数据转换，通过后才继续启动。

`--dtk-wheelhouse` 必须包含匹配的 Torch、TorchVision wheel 及其依赖。基础厂商包从本地目录安装，不从普通网络索引寻找替代包；配套 Triton 可在同一步安装。

其他训练依赖从所选 Python 包源获取。启动器保留厂商 Torch、TorchVision、Triton 的版本约束，依赖冲突时停止安装。FlashAttention、xFormers 通过运行环境页安装，不因文件出现在 wheel 目录中而自动启用。

已有环境继续使用原启动入口和同一 DTK 路径。`--reinstall` 会删除并重建当前 DTK 基础环境，使用已装 PyTorch 的环境仍从原来的 Python 重建；准备其他版本时应使用独立环境目录。

### 旧启动脚本的依赖报错

| 日志 | 原因与处理 |
| --- | --- |
| `新环境里没有读到 … PyTorch 2.5.1（读到 2.5.1+das.opt1.dtk2604）` | 旧脚本混用了运行时版本与安装包版本。更新项目源码，再执行原部署命令，无需替换厂商 PyTorch。 |
| `numpy>=1.26` 与 `numpy==1.25.0` 冲突 | 旧脚本锁住了低于训练器要求的宿主 NumPy。更新项目源码后，原命令会在项目虚拟环境中补齐 NumPy；不需要卸载宿主包或加 `--reinstall`。更换镜像源不能解决版本约束冲突。 |

通过 Git 安装的项目可在源码目录运行 `git pull --ff-only`；通过源码压缩包安装的，更新项目源码文件并保留原来的 `environment/`、`studio_data/` 和自定义数据目录。重试时继续使用原来的 `DTK_ROOT`、`--env-root`、`--data-root` 等参数。

### 缺少 ensurepip

Python 不含 `ensurepip` 时，启动器可使用 `uv`，或借助支持 `--python` 的宿主 pip 引导新环境。后者需要 pip 22.3+；pip 取自 wheel 目录中的 `pip-*.whl`，没有 wheel 目录时从包源下载。

可在联网机器下载后复制到服务器：

```bash
python -m pip download --only-binary=:all: --no-deps pip --dest pip-wheel
```

## 运行库路径

`DTK_ROOT` 默认是 `/opt/dtk`。启动脚本把 `DTKROOT`、`ROCM_PATH` 设为该路径，`HIP_PATH` 设为其下的 `hip` 目录，并将已存在的运行库、编译器及头文件目录加入当前进程环境。

DTK 26.04 的 OpenMP 库位于 `dcc/lib`，启动脚本会同时处理该目录。环境变量只作用于训练器及其子进程，不修改系统库配置。

```bash
DTK_ROOT=/opt/dtk ./studio-linux-dtk.sh doctor
DTK_ROOT=/opt/dtk ./studio-linux-dtk.sh --host 127.0.0.1 --port 8123 --no-browser
```

HIP 版 PyTorch 沿用 `torch.cuda` 接口，`cuda:0` 在此环境中表示第一张海光卡。

## 注意力扩展

在“设置 → 运行环境”选择厂商构建，下载或上传匹配的 wheel，检查安装计划后安装。文件名、包元数据、大小、SHA-256 和运行时条件均需匹配。基础 Torch / Triton 不通过该入口替换。

| 后端 | 依赖 |
| --- | --- |
| SDPA | 厂商 PyTorch；部分 BF16 路径还需匹配的 FlashAttention 动态库 |
| FlashAttention | 对应 Torch / DTK 的厂商包 |
| xFormers | 对应厂商包及其 FlashAttention 依赖 |

缺少 SDPA 所需的厂商动态库时，可能出现 `no matching libraries found for flash_attn_2_cuda`。应安装对应厂商包后重新检测。安装后的设备检测与导入状态分别显示。

Klein 显式选择 FlashAttention 时，通过专用处理器调用厂商 `flash_attn_func`，要求 HIP GPU 和一致的 FP16 / BF16 输入，不接受注意力 mask。该后端记录在训练状态中，不能与 SDPA 状态互相精确恢复。

## 可复现训练

开启“可复现训练”后，Anima、SDXL、Krea 2、Klein 的部分组合（主模型 LoRA / LoKr、文本编码器 LoRA、全量微调）使用固定计算配方：关闭 TF32，注意力改用 SDPA 的数学实现（Klein 选择 FlashAttention 时保留该后端），部分线性层和卷积改用 FP32 计算。Anima、SDXL、Krea 2 的主模型全量微调不符合 BF16 配方时，改为 FP32 计算并关闭混合精度。开启后固定的取值显示在对应字段。

配方记入训练状态，续训时须一致。“训练层类型”为“线性层和卷积层”的任务和开启 DoRA 的任务不使用这些配方；开启 DoRA 时“可复现训练”关闭并置灰。

## 网络与多卡

模型下载、扩展下载和普通依赖安装使用设置中的网络代理。服务启动前的安装过程仍读取终端代理环境变量。

单任务多卡支持 DDP 和 FSDP。卡数、精度、优化器和训练对象要求见 [多卡训练](MULTI_GPU.md)。
