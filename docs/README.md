# 文档

## 部署

| 文档 | 内容 |
| --- | --- |
| [安装与部署](INSTALLATION.md) | 环境要求、启动参数、远程访问、更新与排错 |
| [运行环境](RUNTIME.md) | 平台环境、PyTorch 版本、扩展与 LoRA 环境、下载源、代理与重启 |
| [海光 DTK](RUNTIME_DTK.md) | 运行库、海光版 PyTorch 的来源和注意力扩展 |
| [注意力后端](ATTENTION.md) | SDPA、CUDA 扩展和 Apple Metal FlashAttention |
| [存储与备份](STORAGE.md) | 目录用途、路径变更和备份内容 |

## 训练

| 文档 | 内容 |
| --- | --- |
| [模型配置](MODELS.md) | 模型组件、权重格式、精度与登记 |
| [数据集管理](DATASETS.md) | 导入、筛选、重复次数、标签和遮罩 |
| [JSON 标签](JSON_CAPTIONS.md) | 支持的结构、字段含义与编辑规则 |
| [训练配置](TRAINING.md) | 训练算法、优化器、缓存、保存和续训 |
| [原生分辨率](NATIVE_RESOLUTION.md) | 面积和边长上限、对齐、补边及批次 |
| [多卡训练](MULTI_GPU.md) | DDP、FSDP、显卡调度和恢复条件 |
| [模型测试](MODEL_TESTING.md) | XYZ 对比、采样底模和生成结果 |

## 开发

- [开发指南](DEVELOPMENT.md)
- [架构](ARCHITECTURE.md)
- [OpenAPI](api/openapi.json)
- [训练配置 Schema](api/train-schema.example.json)
