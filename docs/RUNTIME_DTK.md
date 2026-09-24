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

## 厂商包组合

项目内置目录包含以下 DTK 26.04 组合，适用于对应的 Linux x86_64 / Python 3.11 环境：

| 组件 | 构建 |
| --- | --- |
| Torch | `2.7.1+das.opt1.dtk2604` |
| TorchVision | `0.22.0+das.opt1.dtk2604.torch271` |
| Triton | `3.1.0+das.opt1.dtk2604.torch271` |
| FlashAttention | `2.8.3+das.opt1.dtk2604.torch271` |

Torch、TorchVision、Triton 和其依赖放在同一 wheel 目录。其他版本需使用相应的厂商配套构建。普通 PyPI CUDA 包不能替代 HIP 版 Torch。

## 首次安装

以下示例使用已安装的 DTK 26.04 和 Python 3.11：

```bash
DTK_ROOT=/opt/dtk \
YPUDDIN_DTK_PYTHON=/path/to/python3.11 \
./studio-linux-dtk.sh --dtk-wheelhouse=/data/dtk-wheels --no-browser
```

`--dtk-wheelhouse` 必须包含匹配的 Torch、TorchVision wheel 及其依赖。基础厂商包从本地目录安装，不从普通网络索引寻找替代包；配套 Triton 可在同一步安装。

其他训练依赖从所选 Python 包源获取。启动器保留厂商 Torch、TorchVision、Triton 的版本约束，依赖冲突时停止安装。FlashAttention、xFormers 通过运行环境页安装，不因文件出现在 wheel 目录中而自动启用。

已有环境继续使用原启动入口和同一 DTK 路径。`--reinstall` 会删除并重建当前 DTK 基础环境；准备其他版本时应使用独立环境目录。

### 缺少 ensurepip

Python 不含 `ensurepip` 时，启动器可使用 `uv`，或借助支持 `--python` 的宿主 pip 引导新环境。后者需要 pip 22.3+，并在 wheel 目录中准备 `pip-*.whl`。

可在联网机器下载后复制到服务器：

```bash
python -m pip download --only-binary=:all: --no-deps pip --dest pip-wheel
```

## 运行库路径

`DTK_ROOT` 默认是 `/opt/dtk`。启动脚本将该路径传给 `DTKROOT`、`ROCM_PATH` 和 `HIP_PATH`，并将已存在的运行库、编译器及头文件目录加入当前进程环境。

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

## 网络与多卡

模型下载、扩展下载和普通依赖安装使用设置中的网络代理。服务启动前的安装过程仍读取终端代理环境变量。

单任务多卡支持 DDP 和 FSDP。卡数、精度、优化器和训练对象要求见 [多卡训练](MULTI_GPU.md)。
