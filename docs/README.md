# 文档

## 部署

| 文档 | 内容 |
| --- | --- |
| [安装与部署](INSTALLATION.md) | 环境要求、启动参数、远程访问、更新与排错 |
| [运行环境](RUNTIME.md) | 平台环境、PyTorch 版本、扩展与 LoRA 环境、打标与遮罩、下载源、代理与重启、硬件监控 |
| [海光 DTK](RUNTIME_DTK.md) | DTK 安装指南、运行库、海光版 PyTorch 的来源、注意力扩展和可复现训练 |
| [注意力后端](ATTENTION.md) | SDPA、CUDA 扩展和 Apple Metal FlashAttention |
| [存储与备份](STORAGE.md) | 目录用途、路径变更、空间占用和备份内容 |

## 训练

| 文档 | 内容 |
| --- | --- |
| [模型配置](MODELS.md) | 模型组件、权重格式、精度与登记 |
| [数据集管理](DATASETS.md) | 导入、图站下载、筛选、检查、预处理、打标、标签编辑、遮罩和正则数据 |
| [JSON 标签](JSON_CAPTIONS.md) | 支持的结构、字段含义与编辑规则 |
| [训练配置](TRAINING.md) | 训练方式与算法、训练层与卷积层、分桶、缓存、精度与显存、优化器、保存续训、预览验证和日志 |
| [原生分辨率](NATIVE_RESOLUTION.md) | 面积和边长上限、对齐、补边及批次 |
| [多卡训练](MULTI_GPU.md) | DDP、FSDP、显卡调度和恢复条件 |
| [模型测试](MODEL_TESTING.md) | XYZ 对比、采样底模、生成结果和模型常驻 |

## 开发

- [开发指南](DEVELOPMENT.md)
- [架构](ARCHITECTURE.md)
- [内置适配器](ADAPTERS.md)：算法、卷积层、导出、恢复与 LyCORIS 的边界
- [OpenAPI](api/openapi.json)
- [训练配置 Schema](api/train-schema.example.json)
