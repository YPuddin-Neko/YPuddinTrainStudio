# Windows FlashAttention 2 安装

Windows 页面提供 **mjun0812 社区预编译版本**，不是 Dao-AILab 官方 Windows wheel。页面从[维护者发布列表](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases)查询构建；查询失败时使用缓存或内置的 v0.9.6 发布元数据。

内置的 v0.9.6 元数据包含 15 个 FA2 构建，覆盖 Python 3.10–3.14、Torch 2.11，以及 CUDA 12.6 / 12.8 / 13.0，平台为 Windows x64。页面按服务实际 Python、Torch 和 Torch CUDA 构建筛选；不按 NVIDIA 驱动显示的最高 CUDA 版本选择。在线查询可返回其他匹配构建；没有匹配项时可手动上传兼容 wheel，再检查计划。

流程为选择构建 → 下载并校验 SHA256 / 大小 → 检查 wheel 和依赖计划 → 用户确认 → 安装并执行本机内核检测。下载和查询使用全局网络代理设置；下载失败不会安装，页面保留手动下载和上传。查询失败时显示原因，使用已缓存或内置的已核实发布元数据。安装计划不能替换 Torch、Triton 等受保护运行时，Windows 禁止隐式源码编译。

版本匹配只说明构建与当前环境的版本相符，能否使用以安装后本机的内核检测为准。其中 CP312 / Torch 2.11 / cu128 构建已在 Windows 实机上测试。
