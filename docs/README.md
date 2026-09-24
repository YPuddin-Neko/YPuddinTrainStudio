# 使用文档

第一次使用先看 [安装与启动](guide/install.md)，再按 [README 的训练流程](../README.md#开始一次训练) 准备模型和数据。

## 安装与环境

- [安装与启动](guide/install.md)：平台入口、启动参数、远程访问、命令行和常见故障。
- [运行环境](guide/environment.md)：切换 PyTorch、重启、下载源、代理和扩展安装。
- [注意力加速](guide/attention.md)：SDPA、CUDA 扩展与 Apple Metal FlashAttention。
- [海光 DTK](guide/dtk.md)：厂商运行库、独立环境和配套安装包。
- [目录结构与备份](guide/folders.md)：源码、数据、模型、缓存和训练产物分别放在哪里。

## 数据与训练

- [模型与权重](guide/models.md)：各模型所需组件、下载、登记和结果格式。
- [训练说明](guide/training.md)：训练方式、分桶、缓存、精度、LoKr Full 和断点。
- [原生尺寸训练](guide/native-resolution.md)：面积与最长边上限、对齐、补边和缩放。
- [JSON 标签](guide/json-captions.md)：文件格式、分类编辑、训练文本和缓存。
- [Krea 2](guide/krea2.md)：Raw／Turbo、FP8 权重与内存使用。
- [多卡训练](guide/multi-gpu.md)：DDP／FSDP 的配置与限制。
- [按显卡调度任务](guide/gpu-scheduling.md)：不同任务使用不同显卡，以及等待原因。
- [模型测试](guide/xyz-sampling.md)：XYZ 网格、权重比较与采样设置。

## 开发与参考

- [开发说明](development.md)
- [支持范围](SUPPORT.md)
- [OpenAPI 定义](api/openapi.json) · [配置 Schema](api/train-schema.example.json)
- [架构设计](design/00-architecture.md) · [实现参考](reference/)
- [更新记录](../CHANGELOG.md)
