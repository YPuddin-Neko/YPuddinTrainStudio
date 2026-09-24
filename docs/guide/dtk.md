# 海光 DTK

海光 Linux 服务器使用 `studio-linux-dtk.sh`。基础环境默认在源码下的 `environment/linux-dtk/venv`，也可通过 `--env-root` 指定根目录。扩展记录和服务状态放在数据目录的 `environment/linux-dtk`。它不使用根目录的旧 `venv`，也不会改动 CUDA、MPS 或 CPU 环境。

## 需要准备的三层软件

| 层 | 用途 | 官方来源 | 安装方式 |
| --- | --- | --- | --- |
| 海光驱动 | 让系统识别 GPU | [驱动目录](https://download.sourcefind.cn:65024/6/main) | 由机器维护人员按发行版、内核和卡型安装；训练器不安装驱动 |
| DTK 运行库 | HIP、数学库、编译器等运行文件，不是 Python 包 | [DTK 版本目录](https://download.sourcefind.cn:65024/1/main) | 放在独立路径，用 `DTK_ROOT` 指定 |
| 厂商 Python 包 | PyTorch、TorchVision、Triton、FlashAttention、xFormers | [AI 软件包目录](https://download.sourcefind.cn:65024/4/main/) | 基础框架从本地 wheel 目录装进独立环境；注意力扩展在界面中安装 |

先核对 [DTK 与驱动配套表](https://download.sourcefind.cn:65024/file/1/DTK%E4%B8%8E%E9%A9%B1%E5%8A%A8%E7%89%88%E6%9C%AC%E9%85%8D%E5%A5%97%E5%85%B3%E7%B3%BB%E8%A1%A8.md)。例如 DTK 26.04 要求驱动 `>=6.3.30-V1.4.1a`；26.04 和 26.04.1 的配套要求不同，不能混用。

## 推荐版本组合

Ubuntu 22.04 / x86_64 / Python 3.11：

| 组件 | 版本 |
| --- | --- |
| DTK | 26.04，Ubuntu 22.04 x86_64 用户态包 |
| PyTorch | `2.7.1+das.opt1.dtk2604` |
| TorchVision | `0.22.0+das.opt1.dtk2604.torch271` |
| Triton | `3.1.0+das.opt1.dtk2604.torch271` |
| FlashAttention | `2.8.3+das.opt1.dtk2604.torch271` |

后三个包都要求 Torch 2.7.1。系统、架构和 Python 版本与上表一致时，界面会显示这些包的具体下载链接；其他机器显示官方目录，供自行匹配。

旧版 DTK 25.04.1 / Torch 2.4.1 存在 Hugging Face Hub 与 Diffusers 的版本冲突，建议使用 DTK 26.04。

## 首次安装

厂商 PyTorch 必须与 DTK、CPU 架构和 Python 版本匹配；普通 PyPI 的 CUDA 包不能用于 DTK。项目要求 Python 3.10–3.12、PyTorch ≥ 2.4、TorchVision ≥ 0.19，Torch 和 TorchVision 必须同时准备。

1. 下载 DTK 运行库压缩包和配套的 MD5 文件，核对完整性，例如 [Ubuntu 22.04 的 DTK 26.04](https://download.sourcefind.cn:65024/file/1/DTK-26.04/Ubuntu22.04/DTK-26.04-Ubuntu22.04-x86_64.tar.gz)（[MD5](https://download.sourcefind.cn:65024/file/1/DTK-26.04/Ubuntu22.04/DTK-26.04-Ubuntu22.04-x86_64.tar.gz.md5)）。可以解压到源码的 `environment/toolkits/`，不需要覆盖已有的 `/opt/dtk`。
2. 把同一组的 Torch、TorchVision、Triton wheel 及其依赖放进一个本地目录，例如 `environment/vendor-wheels/`。不要混入其他 DTK 或 Torch 版本的包。
3. 运行启动脚本，从本地目录安装：

```bash
DTK_ROOT="$PWD/environment/toolkits/dtk-26.04" \
YPUDDIN_DTK_PYTHON=/path/to/python3.11 \
./studio-linux-dtk.sh --dtk-wheelhouse="$PWD/environment/vendor-wheels" --no-browser
```

- 本地安装使用 `--no-index`，不会从其他来源寻找替代包。装完会检查 Torch 是否为 HIP 构建，CPU 或 CUDA 构建会停止安装。
- 目录里有配套的厂商 Triton 时一并安装；已有 Torch／TorchVision 但缺少 Triton 的环境，用同样的参数也能补齐。已安装的版本保留，不会因为目录里有新版 wheel 就升级。
- 只把 FlashAttention 或 xFormers 的 wheel 放进这个目录不会安装它们，需要之后在界面中安装。
- 常规训练依赖仍从所选的 Python 包源下载，并用精确版本锁住已有的 Torch、TorchVision、Triton、NumPy 等厂商包；出现冲突会报错，不会替换厂商环境。
- 已有可用环境时，建议在另一份源码和环境目录里尝试新的 DTK／Torch 版本，保留原环境。`--reinstall` 会删除当前的 DTK 环境再重建，不是版本切换按钮。

### Python 缺少 ensurepip

有些服务器的 Python 没有 `ensurepip`，普通 `python -m venv` 无法创建环境。启动脚本会依次尝试：已有的 `uv` → Python 自带的 `ensurepip` → 宿主 pip 的 `--python` 功能（需要 pip 22.3 或更新），把本地 pip wheel 装进新环境。

这种情况下在 `--dtk-wheelhouse` 目录里额外放一个支持当前 Python 的 `pip-*.whl`。可以在能联网的电脑上下载后复制过来：

```bash
python -m pip download --only-binary=:all: --no-deps pip --dest pip-wheel
```

缺少 pip wheel 或宿主 pip 太旧时，脚本会在创建环境前说明缺什么，不会执行 apt、升级宿主 pip 或安装驱动。

## 运行库路径

启动脚本默认读取 `/opt/dtk`，其他位置用 `DTK_ROOT` 指定，并把同一路径传给子进程的 `DTKROOT`、`ROCM_PATH` 和 `HIP_PATH`（指向 `DTK_ROOT/hip`）。

脚本只为当前进程和子进程设置运行库、编译器和头文件的搜索路径，只加入实际存在的目录，原有路径排在后面；不会调用 `ldconfig`、写终端配置或修改系统文件。DTK 26.04 的 `libomp.so` 位于 `dcc/lib`，只设置顶层 `lib` 会导致 Torch 无法导入，脚本会一并处理。

## 检查与启动

```bash
DTK_ROOT=/opt/dtk ./studio-linux-dtk.sh doctor
DTK_ROOT=/opt/dtk ./studio-linux-dtk.sh --host 127.0.0.1 --port 8123 --no-browser
```

- 之后每次启动都要使用创建环境时的 `DTK_ROOT`；改变这个变量不会转换环境里的 Torch 等原生包。
- HIP 版 PyTorch 沿用 `torch.cuda` 接口，所以 `cuda:0` 可以表示第一张海光卡。运行环境页会分别显示 HIP 版本和实际计算后端。
- 设置里的 CUDA／CPU PyTorch 切换和 NVIDIA 扩展不适用于 DTK 环境。

## 注意力扩展

| 选项 | 说明 |
| --- | --- |
| SDPA | 使用厂商 PyTorch；部分 BF16 路径还需要匹配的厂商 FlashAttention 动态库 |
| FlashAttention 2.8.3 厂商包 | 在运行环境页下载或上传安装，安装后做本机检测 |
| xFormers 0.0.33 厂商包 | 需要先安装厂商 FlashAttention；Torch 2.4.1 缺少所需接口，不能使用 |

**选择 SDPA 也需要匹配的厂商运行库。** DTK 26.04 / Torch 2.7.1 的部分 BF16 SDPA 路径会调用 FlashAttention 动态库，缺少时会报 `no matching libraries found for flash_attn_2_cuda`。运行环境页的 SDPA 检测会分别检查 FP16、BF16 和两组注意力形状的前向与反向；训练中遇到这个错误时会提示安装匹配的厂商包。

xFormers 与 FlashAttention 的管理页使用 [SourceFind 官方厂商目录](https://download.sourcefind.cn:65024/4/main/)：

- 只列出已核对内容与元数据的构建，下载后检查完整大小和 SHA256；目录中的其他 wheel 不会显示为可安装。
- 在线下载与离线上传执行相同的文件名、元数据、依赖、ABI、大小和散列检查。下载只接受官方 HTTPS 来源，支持环境变量 `HTTPS_PROXY`，证书验证保持开启。
- 安装计划列出所选扩展和需要补齐的普通依赖，保留已有包版本；厂商 Torch、Triton 不会被重新安装或替换。
- 厂商 FlashAttention 导入时会加载引用 `pytest` 的模块，扩展管理会在安装计划里自动补上 `pytest` 及其依赖。
- “上传安装”只接收已核对的注意力扩展 wheel，不接收驱动、DTK 压缩包或基础 Torch／Triton。

| 扩展 | 构建 | 安装条件 |
| --- | --- | --- |
| FlashAttention | `2.8.3+das.opt1.dtk2604.torch271`（DAS1.8） | Linux x86_64、Python 3.11、DTK 26.04、Torch 2.7.1，已有 einops 和厂商 Triton `3.1.0+das.opt1.dtk2604.torch271` |
| FlashAttention | `2.6.1+das.opt1.dtk25041`（DAS1.6） | Linux x86_64、Python 3.11、DTK 25.04.1、Torch 2.4.1，已有 einops 和厂商 Triton `3.0.0+das.opt1.dtk25041` |
| xFormers | `0.0.33+das.opt1.dtk2604.torch251`（DAS1.8） | DTK 环境、Torch ≥ 2.5、NumPy、FlashAttention ≥ 2.6.1；纯 Python 包，以安装后的本机检测为准 |

### Klein 的 FlashAttention

厂商 FlashAttention 2.8.3 的公开版本号仍是 2.6.1，缺少 Diffusers 原生 Flash 路径需要的 `_wrapped_flash_attn_forward/backward` 接口。Klein 在海光上显式选择 FlashAttention 时，使用专用的注意力处理器直接调用厂商的 `flash_attn_func`，前向和反向都由厂商实现，不修改第三方包和其他模型。

- 需要 HIP GPU 和一致的 FP16／BF16 输入；不支持注意力 mask。接口不满足或内核失败会报错，不会自动改用 SDPA。
- Diffusers 导入时可能打印缺少 `_wrapped_flash_attn_backward`、改用原生注意力的警告；Klein 的处理器不经过那条路径，实际使用的后端以训练记录为准。
- 这个选择会写入训练状态，不能与 SDPA 的训练状态互相精确续训。

## 下载文件

| 文件 | 大小 | SHA256 |
| --- | ---: | --- |
| [FlashAttention 2.8.3（DTK 26.04 / Torch 2.7.1）](https://download.sourcefind.cn:65024/file/4/flash_attn/DAS1.8/flash_attn-2.8.3%2Bdas.opt1.dtk2604.torch271-cp311-cp311-manylinux_2_28_x86_64.whl) | 658,880,343 | `d2cdd700de8622b2473bbac57328ca6682eb4025c274a6b7f7f50c687a3c554b` |
| [FlashAttention 2.6.1（DTK 25.04.1）](https://download.sourcefind.cn:65024/file/4/flash_attn/DAS1.6/flash_attn-2.6.1%2Bdas.opt1.dtk25041-cp311-cp311-manylinux_2_28_x86_64.whl) | 389,084,275 | `5ce0a673a64fc7fead3289a6268932200f09d4d1e0903a09231bf765f649e7f9` |
| [xFormers 0.0.33](https://download.sourcefind.cn:65024/file/4/xformers/DAS1.8/xformers-0.0.33%2Bdas.opt1.dtk2604.torch251-py3-none-any.whl) | 251,694 | `bbcf795c71ff248e261a56ba43a9a9dd8b14f944872c9d9ee8efd260b1bdb915` |
| [Triton 3.1.0（DTK 26.04 / Torch 2.7.1）](https://download.sourcefind.cn:65024/file/4/triton/DAS1.8/triton-3.1.0%2Bdas.opt1.dtk2604.torch271-cp311-cp311-manylinux_2_28_x86_64.whl) | 128,058,626 | `71cc667b28bc326d888959a9695ddded3ea888ca8ad5fe541fefbf2061d4e4b9` |

Triton 属于受保护的基础环境，需要放进 `--dtk-wheelhouse` 由启动脚本安装，注意力扩展安装器不会替换它。
