# Windows FlashAttention 2 安装

Windows 页面提供 **mjun0812 社区预编译版本**，不是 Dao-AILab 官方 Windows wheel。当前查询范围固定为 [维护者 v0.9.6 发布](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/tag/v0.9.6)，不查询任意仓库或推测不存在的构建。

该发布的 15 个 FA2 构建覆盖 Python 3.10–3.14、Torch 2.11，以及 CUDA 12.6 / 12.8 / 13.0，平台为 Windows x64。页面按服务实际 Python、Torch 和 Torch CUDA 构建筛选；不按 NVIDIA 驱动显示的最高 CUDA 版本选择。每个组合当前只有一个 FA2 版本。其他组合可手动上传兼容 wheel，再检查计划。

流程为选择构建 → 下载并校验 SHA256 / 大小 → 检查 wheel 和依赖计划 → 用户确认 → 安装并执行本机内核检测。下载和查询使用全局网络代理设置；下载失败不会安装，页面保留手动下载和上传。查询失败时显示原因，使用已缓存或内置的已核实发布元数据。安装计划不能替换 Torch、Triton 等受保护运行时，Windows 禁止隐式源码编译。

候选匹配只说明版本条件符合。2026-09-15 新增流程的验证使用模拟网络、pip 和 GPU；没有在 macOS 执行 Windows 二进制。CP312 / Torch2.11 / cu128 构建已有独立的 [Windows 实机验收记录](WINDOWS_ACCEPTANCE_2026-09-14.md)，其他构建仍以安装后的目标机器检测为准。
