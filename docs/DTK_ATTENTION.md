# DTK 注意力扩展来源与验证

## 现在应该选哪个

2026-09-14，**DTK 26.04 / 厂商 Torch 2.7.1** 已完成 Krea2 Raw 正式 BF16 权重、512×512、8 步 LoKr 的六组测试：SDPA、FlashAttention、xFormers 各测单卡与双卡，训练、保存、采样及严格恢复全部通过。详见 [DTK 验收记录](DTK_ACCEPTANCE_2026-09-14.md) 和 [六组精简数据](validation/DTK_KREA2_2026-09-14.json)。

| 当前环境 | 注意力选项 | 验收结论 |
| --- | --- | --- |
| DTK 26.04 / Torch 2.7.1 | SDPA | Krea2 正式权重单卡 / 双卡通过；四个模型族微型权重单卡矩阵 20/20 通过 |
| DTK 26.04 / Torch 2.7.1 | FlashAttention 2.8.3 厂商包 | Krea2 正式权重单卡 / 双卡通过，实际调用所选扩展；干净默认环境的官方下载安装及修复核验通过 |
| DTK 26.04 / Torch 2.7.1 | xFormers 0.0.33 厂商包 | Krea2 正式权重单卡 / 双卡通过，实际调用所选扩展；浏览器上传安装和实卡内核核验通过 |
| DTK 26.04 / Torch 2.7.1 | Klein 显式 Flash | 当前厂商包缺少 Diffusers 所需接口；缓存及正式权重加载前明确报错，选择 SDPA |
| 旧 DTK 25.04.1 / Torch 2.4.1 | xFormers 0.0.33 厂商包 | 运行缺少 `_symmetric_memory`，不可用；此旧环境限制不等于新版也不可用 |

FlashAttention / xFormers 在两张卡上另有 FP16 / BF16、head dimension 64 / 128 的 16 项运算检查全部通过。正式 Krea2 的训练和采样调用统计确认使用了所选扩展，没有退回 SDPA。六组恢复分别比较全部权重、预览图以及优化器、调度器、进度、采样顺序和随机状态，结果完全一致；不要求不同后端或卡数之间逐位相同，也不据此宣称某个后端更快。

完整启动与环境准备见 [海光 DTK 独立环境](RUNTIME_DTK.md)。旧 DTK 25.04.1 仍有 Hub / Diffusers 冲突及早期恢复不一致记录；当前结论以新版组合的最终报告为准，Krea2 通过不能代表所有模型均支持同一注意力后端。

新环境的 20 例使用 SDPA、FP32、64×64 合成图，每例 2 步：Anima、Krea2、SDXL、Klein 各覆盖全量微调主模型、文本编码器、两者同时训练，以及 LoRA、LoKr。保存产物重新加载、逐元素一致的续训、训练预览和 XYZ 均通过。实际环境同时装有厂商 FlashAttention 与 xFormers，但“已安装”不表示这些用例选择了它们；本轮明确选择 SDPA，未改用其他注意力后端。

这项微型权重检查证明新 Diffusers/Transformers 与 DTK 栈的相应执行流程可用。它与正式 Krea2 六组矩阵分开，也不代表 Anima、SDXL、Klein 的正式权重、大分辨率或 Klein 显式 Flash 已通过。原始记录为源码外 `remote-testing/dtk-20260914/tiny-matrix-gpu1-20260914085100812/report.json`，并明确标注 `production_weight_validation=false`。

## 官方包与版本匹配

目录核查日期：2026-09-14。数据来源是 [SourceFind 官方包合集](https://download.sourcefind.cn:65024/4/main/) 的公开目录 API `GET /api-static/file/ListFile?CategoryID=4&Path=...`。产品列出经过文件内容与元数据核查的构建，下载后再检查完整大小和 SHA256；不会把目录中的所有 wheel 都当作可安装版本。

| 扩展 | 官方构建 | 安装匹配条件 |
| --- | --- | --- |
| FlashAttention | `2.6.1+das.opt1.dtk25041`，DAS1.6 | Linux x86_64、Python 3.11、厂商 DTK 25.04.1、Torch 2.4.1；已具备 einops 和厂商 Triton `3.0.0+das.opt1.dtk25041` |
| FlashAttention | `2.8.3+das.opt1.dtk2604.torch271`，DAS1.8 | Linux x86_64、Python 3.11、厂商 DTK 26.04、Torch 2.7.1；已具备 einops 和厂商 Triton `3.1.0+das.opt1.dtk2604.torch271` |
| xFormers | `0.0.33+das.opt1.dtk2604.torch251`，DAS1.8 | Linux x86_64 的 DTK 环境、Torch ≥ 2.5、NumPy、FlashAttention ≥ 2.6.1；此包仅含 Python，安装后还须实卡验证 |

xFormers 的 `dtk2604.torch251` 是厂商发布标签。这个 wheel 为 `py3-none-any`，没有原生动态库，METADATA 声明 `torch>=2.1.0`。但 2026-09-14 的真实海光环境安装后探测失败，原因是 Torch 2.4.1 缺少 `torch.distributed._symmetric_memory`。产品据此把实际 API 要求提高到 Torch ≥ 2.5，保留厂商原始声明供技术核对；Torch 2.4 不再显示可安装匹配。更高 Torch 仍要重新做当前卡的探测与训练，不能只按版本判定已经可用。

DAS1.6 的 FlashAttention 2.6.1 包包含原生动态库，且源码记录的编译 Torch 为 2.4。其 `Requires-Dist: torch` 没有限制版本，不能代替二进制兼容检查；产品同时核对 DTK、Torch 与 Python ABI。其导入路径还需要 Triton，虽然 METADATA 未完整列出这一项，目录规则会额外检查厂商 Triton 版本。不同 DTK 或更高 Torch 环境不会沿用这个原生包。

新的 DAS1.8 FlashAttention 包已完成完整下载散列与内部元数据核查，纳入同一套自动下载和上传安装校验。它的 METADATA 为 2.8.3，明确要求 Torch 2.7.1；公开 `__version__` 仍为 2.6.1，提供 `flash_attn_func`，但没有 Diffusers 所需的 wrapped 前后向接口。安装后的探测只建立常规 FlashAttention 内核可用性，不能据此宣称 Klein / Diffusers 的显式 Flash 后端可用。

两个已核查的厂商 Flash 包还存在未声明的运行依赖：导入 `flash_attn` 时会经过 `flash_attn_triton_interface` 加载直接引用 `pytest` 的模块。因此，干净训练环境即使已安装 Torch、Triton，也可能提示 `No module named pytest`。扩展管理会在安装或修复计划中补上缺失的 `pytest` 及其普通 Python 依赖；xFormers 使用该厂商 Flash 时也采用同一规则。所有新增包先列入计划，再按固定下载地址和散列安装。依赖解析失败时停在计划阶段，已有 Torch、Triton 及其他包保持原版本；无需让用户另开终端安装开发环境。

这次干净默认环境确实先遇到了该错误。修复后的实际操作 `env_ff3c6357e90d` 将 Pytest 9.1.1、Pluggy 1.6.0、Iniconfig 2.3.0 列入同一计划，完成 658,880,343 字节官方 Flash 包下载、repair 和内核检测；随后 xFormers 通过浏览器上传，操作 `env_b483cb21f39c` 安装并检测成功。Torch、TorchVision、Triton 的厂商版本均未改变。

已排除旧 DAS1.3 的 xFormers `0.0.25+das.opt1.dtk24043`：文件名与 METADATA 的 DTK 标签不一致，内容中编译 Torch 为 2.1，并要求旧 NumPy 范围；不能用于当前 Torch 2.4 / DTK 25.04.1 环境。

## 下载文件校验

- [FlashAttention 官方 wheel](https://download.sourcefind.cn:65024/file/4/flash_attn/DAS1.6/flash_attn-2.6.1%2Bdas.opt1.dtk25041-cp311-cp311-manylinux_2_28_x86_64.whl)：389,084,275 字节，SHA256 `5ce0a673a64fc7fead3289a6268932200f09d4d1e0903a09231bf765f649e7f9`。
- [FlashAttention DTK 26.04 / Torch 2.7.1 官方 wheel](https://download.sourcefind.cn:65024/file/4/flash_attn/DAS1.8/flash_attn-2.8.3%2Bdas.opt1.dtk2604.torch271-cp311-cp311-manylinux_2_28_x86_64.whl)：658,880,343 字节，SHA256 `d2cdd700de8622b2473bbac57328ca6682eb4025c274a6b7f7f50c687a3c554b`。
- [xFormers 官方 wheel](https://download.sourcefind.cn:65024/file/4/xformers/DAS1.8/xformers-0.0.33%2Bdas.opt1.dtk2604.torch251-py3-none-any.whl)：251,694 字节，SHA256 `bbcf795c71ff248e261a56ba43a9a9dd8b14f944872c9d9ee8efd260b1bdb915`。
- [Triton 官方 wheel](https://download.sourcefind.cn:65024/file/4/triton/DAS1.6/triton-3.0.0%2Bdas.opt1.dtk25041-cp311-cp311-manylinux_2_28_x86_64.whl)：厂商目录提供匹配构建；Triton 属于受保护的原生环境，需要预先准备，不由注意力扩展安装器替换。
- [Triton DTK 26.04 / Torch 2.7.1 官方 wheel](https://download.sourcefind.cn:65024/file/4/triton/DAS1.8/triton-3.1.0%2Bdas.opt1.dtk2604.torch271-cp311-cp311-manylinux_2_28_x86_64.whl)：128,058,626 字节，SHA256 `71cc667b28bc326d888959a9695ddded3ea888ca8ad5fe541fefbf2061d4e4b9`。放入匹配环境的 `--dtk-wheelhouse` 后，独立启动器可以离线安装或补齐；已有 Triton 不会自动升级。

官方在线下载与离线上传执行相同的文件名、METADATA、包依赖、ABI、大小及散列检查。下载只接受官方 HTTPS 来源，限制重定向，完整文件通过后才进入安装计划。支持进程环境中的 `HTTPS_PROXY`，证书验证保持开启。

本次服务器访问 SourceFind / PyPI 使用限定官方域名的 TLS 转发，证书验证和完整文件校验均保留；没有据此认定远程服务器能直接连接源站。手动上传流程同样已在真实浏览器完成。

扩展页的“上传安装”仅接收已经核查的注意力扩展 wheel，不接收驱动、DTK 压缩包或基础 Torch / Triton。找不到匹配包时，按官方目录和当前环境信息准备配套版本；不会用未验证的构建或普通 CUDA 包代替。

自动化回归覆盖版本不匹配、缺少厂商 Triton、下载损坏或中断、取消、来源跳转、安装计划、基础依赖保护及安装后探测失败。文首列出的下载和运行结果来自真实海光机器上的产品 API 与训练验收；版本规则和模拟测试只用于防止回归，不代替实卡结果。
