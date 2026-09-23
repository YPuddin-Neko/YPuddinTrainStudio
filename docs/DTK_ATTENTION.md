# DTK 注意力扩展

## 现在应该选哪个

| 当前环境 | 注意力选项 | 说明 |
| --- | --- | --- |
| DTK 26.04 / Torch 2.7.1 | SDPA | 使用厂商 PyTorch；部分 BF16 路径还需要匹配的厂商 FlashAttention 动态库 |
| DTK 26.04 / Torch 2.7.1 | FlashAttention 2.8.3 厂商包 | 可在运行环境中下载或上传安装，安装后执行本机内核检测 |
| DTK 26.04 / Torch 2.7.1 | xFormers 0.0.33 厂商包 | 可在运行环境中下载或上传安装，需要已安装的厂商 FlashAttention |
| DTK 26.04 / Torch 2.7.1 | Klein 显式 Flash | Base 4B／9B 的 LoRA／LoKr，双卡 DDP／FSDP，见下文 |
| 旧 DTK 25.04.1 / Torch 2.4.1 | xFormers 0.0.33 厂商包 | 不可用：Torch 2.4.1 缺少 `torch.distributed._symmetric_memory` |

FlashAttention 与 xFormers 已在 Krea2 正式权重的单卡与双卡训练中测试；Anima、SDXL 的正式权重尚未在海光上用这两个扩展测试。旧版 DTK 25.04.1 还存在 Hub / Diffusers 版本冲突，建议使用 DTK 26.04。完整启动与环境准备见 [海光 DTK 独立环境](RUNTIME_DTK.md)。

## 官方包与版本匹配

构建列表来自 [SourceFind 官方包合集](https://download.sourcefind.cn:65024/4/main/) 的公开目录 API `GET /api-static/file/ListFile?CategoryID=4&Path=...`（2026-09-14 核对）。产品只列出核对过文件内容与元数据的构建，下载后再检查完整大小和 SHA256；目录中的其他 wheel 不会显示为可安装版本。

| 扩展 | 官方构建 | 安装匹配条件 |
| --- | --- | --- |
| FlashAttention | `2.6.1+das.opt1.dtk25041`，DAS1.6 | Linux x86_64、Python 3.11、厂商 DTK 25.04.1、Torch 2.4.1；已具备 einops 和厂商 Triton `3.0.0+das.opt1.dtk25041` |
| FlashAttention | `2.8.3+das.opt1.dtk2604.torch271`，DAS1.8 | Linux x86_64、Python 3.11、厂商 DTK 26.04、Torch 2.7.1；已具备 einops 和厂商 Triton `3.1.0+das.opt1.dtk2604.torch271` |
| xFormers | `0.0.33+das.opt1.dtk2604.torch251`，DAS1.8 | Linux x86_64 的 DTK 环境、Torch ≥ 2.5、NumPy、FlashAttention ≥ 2.6.1；此包仅含 Python，能否使用以安装后的本机检测为准 |

xFormers 的 `dtk2604.torch251` 是厂商发布标签。这个 wheel 为 `py3-none-any`，没有原生动态库，METADATA 声明 `torch>=2.1.0`；但在 Torch 2.4.1 上导入会因缺少 `torch.distributed._symmetric_memory` 失败，因此产品要求 Torch ≥ 2.5。更高版本的 Torch 也以安装后的本机检测为准。

DAS1.6 的 FlashAttention 2.6.1 包包含原生动态库，且源码记录的编译 Torch 为 2.4。其 `Requires-Dist: torch` 没有限制版本，不能代替二进制兼容检查；产品同时核对 DTK、Torch 与 Python ABI。其导入路径还需要 Triton，虽然 METADATA 未完整列出这一项，目录规则会额外检查厂商 Triton 版本。不同 DTK 或更高 Torch 环境不会沿用这个原生包。

DAS1.8 的 FlashAttention 包 METADATA 为 2.8.3，要求 Torch 2.7.1；公开 `__version__` 仍为 2.6.1，提供 `flash_attn_func`，但没有 Diffusers 原生 Flash 路径导入的 wrapped 前后向接口，因此 Klein 使用下文的专用处理器。

这两个厂商 Flash 包还有未声明的运行依赖：导入 `flash_attn` 时会经过 `flash_attn_triton_interface` 加载直接引用 `pytest` 的模块，干净的训练环境即使已安装 Torch、Triton，也可能提示 `No module named pytest`。扩展管理会在安装或修复计划中补上缺失的 `pytest` 及其普通 Python 依赖；xFormers 使用该厂商 Flash 时也采用同一规则。新增的包先列入计划，再按固定下载地址和散列安装；依赖解析失败时停在计划阶段，已有的 Torch、Triton 及其他包保持原版本。

旧 DAS1.3 的 xFormers `0.0.25+das.opt1.dtk24043` 不提供安装：文件名与 METADATA 的 DTK 标签不一致，编译 Torch 为 2.1，并要求旧的 NumPy 范围。

## Klein 的海光 FlashAttention

Klein 在 HIP 环境显式选择 `model.attention = "flash_attn"` 时，使用专用注意力处理器直接调用厂商 `flash_attn_func`，前向与反向均由厂商实现。处理器保留 Diffusers 0.40 的投影、归一化、旋转位置编码、文本／图像合并及输出顺序，不修改第三方包、全局后端注册表或其他模型。普通 CUDA 环境继续使用 Diffusers 原生后端。

启动前会检查公开函数及 `deterministic` 参数。运行时要求 HIP GPU 和一致的 FP16／BF16 Q、K、V；不支持注意力 mask 或上下文并行（DDP／FSDP 不属于上下文并行）。接口不满足或内核失败会明确报错，不自动改用 SDPA。

开启可复现训练、并满足 [Klein 双卡 BF16 策略](FSDP_ADAPTERS_2026-09-18.md) 时，显式 Flash 选择会被保留，主模型调用厂商的确定性反向。其他组件的 SDPA 调用继续使用数学实现；线性层与冻结文本编码器沿用各自的计算策略。完整状态记录 `klein-dtk-public-flash-v1`，对应训练策略增加 `-flash-v1` 后缀，不能与旧 SDPA 状态混作严格续训。前端显示服务器确认的实际后端。

Diffusers 导入时仍可能输出缺少 `_wrapped_flash_attn_backward`、改用原生注意力的上游警告。它描述的是 Diffusers 自己的注册路径；Klein 的专用处理器直接调用公开的 `flash_attn_func`，绕过缺失的 wrapped 接口。实际后端以训练记录的计算策略为准。

已在 DTK 26.04／厂商 Torch 2.7.1／Diffusers 0.40.0 上以双卡 DDP／FSDP 测试 Klein Base 4B／9B 的 LoRA／LoKr 训练、断点恢复与导出（BF16 底模、FP32 适配器、冻结文本编码器）。

## 下载文件校验

- [FlashAttention 官方 wheel](https://download.sourcefind.cn:65024/file/4/flash_attn/DAS1.6/flash_attn-2.6.1%2Bdas.opt1.dtk25041-cp311-cp311-manylinux_2_28_x86_64.whl)：389,084,275 字节，SHA256 `5ce0a673a64fc7fead3289a6268932200f09d4d1e0903a09231bf765f649e7f9`。
- [FlashAttention DTK 26.04 / Torch 2.7.1 官方 wheel](https://download.sourcefind.cn:65024/file/4/flash_attn/DAS1.8/flash_attn-2.8.3%2Bdas.opt1.dtk2604.torch271-cp311-cp311-manylinux_2_28_x86_64.whl)：658,880,343 字节，SHA256 `d2cdd700de8622b2473bbac57328ca6682eb4025c274a6b7f7f50c687a3c554b`。
- [xFormers 官方 wheel](https://download.sourcefind.cn:65024/file/4/xformers/DAS1.8/xformers-0.0.33%2Bdas.opt1.dtk2604.torch251-py3-none-any.whl)：251,694 字节，SHA256 `bbcf795c71ff248e261a56ba43a9a9dd8b14f944872c9d9ee8efd260b1bdb915`。
- [Triton 官方 wheel](https://download.sourcefind.cn:65024/file/4/triton/DAS1.6/triton-3.0.0%2Bdas.opt1.dtk25041-cp311-cp311-manylinux_2_28_x86_64.whl)：厂商目录提供匹配构建；Triton 属于受保护的原生环境，需要预先准备，不由注意力扩展安装器替换。
- [Triton DTK 26.04 / Torch 2.7.1 官方 wheel](https://download.sourcefind.cn:65024/file/4/triton/DAS1.8/triton-3.1.0%2Bdas.opt1.dtk2604.torch271-cp311-cp311-manylinux_2_28_x86_64.whl)：128,058,626 字节，SHA256 `71cc667b28bc326d888959a9695ddded3ea888ca8ad5fe541fefbf2061d4e4b9`。放入匹配环境的 `--dtk-wheelhouse` 后，独立启动器可以离线安装或补齐；已有 Triton 不会自动升级。

官方在线下载与离线上传执行相同的文件名、METADATA、包依赖、ABI、大小及散列检查。下载只接受官方 HTTPS 来源，限制重定向，完整文件通过后才进入安装计划。支持进程环境中的 `HTTPS_PROXY`，证书验证保持开启。

扩展页的“上传安装”仅接收已经核查的注意力扩展 wheel，不接收驱动、DTK 压缩包或基础 Torch / Triton。找不到匹配包时，按官方目录和当前环境信息准备配套版本；不要用未验证的构建或普通 CUDA 包代替。
