# macOS GPU 利用率修复 · v0.5.7

旧版 `hardware.gpu_info()` 的 MPS 分支把 `util_pct` 固定为 `None`，因此顶栏始终显示“—”。这不是页面刷新失败，也不表示 GPU 没有工作。旧说明将利用率、功率、温度都说成 MPS 不提供，表述过于绝对。

## 实际读取与含义

新增独立实现的只读 IORegistry 查询，使用 macOS 自带 `/usr/sbin/ioreg -r -c IOAccelerator -d 1 -a`，通过 `plistlib` 读取唯一 Apple AGX 加速器的 `PerformanceStatistics / Device Utilization %`。正常服务不需要 sudo 或安装特权 helper。本机普通用户 UID 501 查询成功；代理沙箱中的查询失败不等于用户需要管理员权限。

该字段已经是 0–100 的百分数，合法 0 和小于 1 的百分数直接保留。不会把 renderer 与 tiler 利用率相加，也不会计算两次差值。它表示**全系统 GPU 活动**，包括桌面和其他应用，不代表训练进程独占的利用率。驱动的统计窗口没有在此接口中公开，不把它描述为精确的一秒平均值。

采集在进程内加锁并缓存两秒，每次命令超时为两秒。缺失、错误数值、损坏输出、查询失败或多个无法对应的 Apple 设备均返回未知；失败也短暂缓存，且不会永久沿用上一次读数。只读取一种类匹配结果，避免将同一 GPU 的父类/子类查询结果重复计算。非 macOS 不启动此命令，CUDA 逻辑保持原样。

GPU 利用率沿用现有 `util_pct` 字段，来源标记为 `ioreg`。统一内存仍使用系统可用内存计算，与 RAM 顶栏共享同一快照。功率与温度没有在本次查询中取得可靠值，分别显示未知，不从利用率估算。

参考证据：[Apple ioreg 手册](https://github.com/apple-oss-distributions/IOKitTools/blob/main/ioreg.tproj/ioreg.8)、[Apple IOKit 属性接口](https://github.com/apple-oss-distributions/IOKitUser/blob/main/IOKitLib.h)、[Stats 对该驱动百分比的使用](https://github.com/exelban/stats/blob/master/Modules/GPU/reader.swift)。没有复制第三方实现代码。

## 验证

- 后端 36 项定向回归通过：Apple 读数与未知值、0%、小数、非法值、超时、失败过期和缓存；以及既有 NVML、nvidia-smi、统一内存快照测试。Ruff 通过。
- 前端三个相关测试文件 18 项通过；TypeScript、ESLint、生产构建通过。包括真实 0/46 值、来源说明、未知功率与温度的独立提示。
- 本机 `gpu_info()` 实际读到 31%，真实 HTTP API 随后读到 59%。CUA 浏览器先显示 GPU 41%、统一内存 91%，后续截图显示 GPU 54%、统一内存 91%，确认读数经实时链路更新。
- 正式 8876 服务已升级到 0.5.7。刷新临时验证标签页后，GPU 显示 52%、统一内存 91%，来源与全系统范围提示正确；194 个业务文件与六张业务表的升级前后快照一致，项目、设置和凭据保留。测试标签页和临时 QA 服务已关闭。
- 本轮没有重跑完整训练套件或进行 GPU 压力测试；上一版的完整回归见 `validation/v0.5.6.json`，不能视为本版全量验证。未执行 npm audit。

最终服务保留证明、构建指纹和浏览器记录见 [v0.5.7 验证记录](validation/v0.5.7.json)。
