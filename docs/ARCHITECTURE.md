# 架构

## 模块划分

| 目录 | 职责 |
| --- | --- |
| [`frontend/src/`](../frontend/src/) | React 页面、参数表单、API 客户端与事件订阅 |
| [`ypuddin/server/`](../ypuddin/server/) | FastAPI 服务、项目版本、数据管理、模型下载、环境和任务监督 |
| [`ypuddin/config/`](../ypuddin/config/) | 配置结构、加载、兼容规则与计算策略 |
| [`ypuddin/models/`](../ypuddin/models/) | 模型系列、组件加载、文本编码与 VAE |
| [`ypuddin/data/`](../ypuddin/data/) | 图片索引、标签、几何处理、分桶与缓存 |
| [`ypuddin/adapters/`](../ypuddin/adapters/) | LoRA / LoKr 等附加权重的注入、计算与导入导出 |
| [`ypuddin/train/`](../ypuddin/train/) | 训练循环、预览、分布式执行与完整状态 |
| [`ypuddin/sampling/`](../ypuddin/sampling/) | 采样器与调度 |
| [`scripts/`](../scripts/) | 部署引导、接口导出与源码打包 |

## 服务与任务

[`create_app`](../ypuddin/server/app.py) 组装数据库、事件总线、任务监督器、数据流水线及环境管理器。前端通过 HTTP 请求读取或修改状态，通过 `/api/events` 订阅进度与任务事件。

训练、缓存和模型测试由监督器启动独立工作进程。任务保存配置快照及设备请求；监督器分配设备后设置进程可见 GPU，并在进程及子进程退出后释放设备。

服务重启由独立启动器管理，PyTorch 环境切换通过重启更换解释器。环境安装与版本数据修改受任务状态约束。

## 配置与训练计划

[`TrainConfig`](../ypuddin/config/schema.py) 定义字段类型、默认值、范围和界面元数据。模型系列提供组件、训练对象、精度和后端能力；平台计算策略进一步限制可用组合。

[`train/plan.py`](../ypuddin/train/plan.py) 计算数据计划和资源估算。网页数据预览复用图片索引，文件状态变化时重新读取对应内容。

前端的 [`SchemaForm`](../frontend/src/schema/SchemaForm/) 使用同一配置定义，参数分组和界面短说明由前端组织。提交请求仍由后端校验。

## 数据身份与缓存

图片身份、标签内容、遮罩、来源和几何配置参与数据指纹。训练参与状态以来源中的排除列表保存，移出和重新加入不移动图片或关联文件。

图像特征缓存基于图像内容、几何处理和 VAE 身份；文本缓存基于最终文本及编码器、分词器身份。缓存采用临时文件写入与原子替换。缓存复用与完整状态恢复分别校验。

## 模型加载

模型系列封装主模型、文本组件、VAE 和采样差异。Krea 2 与 Klein 可先建立主模型结构，完成图像、文本编码并卸载编码器后再加载主模型权重，避免组件同时驻留。

平台专用原生库按后端加载。常规状态读取与显式计算检测分开处理，运行环境页的计算检测使用隔离进程。

## 适配器算法

参数结构、导出与加速边界见[内置适配器](ADAPTERS.md)。

各算法以不同结构表示权重增量：

```text
LoRA:          ΔW = scale × B × A
LoKr:          ΔW = scale × (W1 ⊗ W2)
LoHa:          ΔW = scale × (B1 A1) ⊙ (B2 A2)
OrthoLoRA:     ΔW = scale × U (C − I) diag(s) Vᵀ
T-LoRA:        ΔW = scale × B diag(m(t)) A
LyCORIS Full:  ΔW = W − W0
```

LoKr 的因子可以进一步低秩拆分。`rank = "full"` 保留完整因子矩阵，不解冻底模。训练层范围单独控制模块匹配。

OrthoLoRA（[`adapters/ortho.py`](../ypuddin/adapters/ortho.py)）的 U、s、V 取自底模权重的主要奇异方向并保持冻结，C 由正交旋转和两组缩放组成，起步时 ΔW 为零。T-LoRA（[`adapters/tlora.py`](../ypuddin/adapters/tlora.py)）的 m(t) 按每个样本的噪声强度保留前 r(t) 个秩，只在训练时生效，预览与导出使用全部秩；正交初始化时另减去冻结的起点。两者都按普通 LoRA 导出。LyCORIS Full 直接训练层权重，导出为 LyCORIS 差值。

线性层包装器根据计算模式直接应用增量，或重建合并权重。导出负责处理缩放和键名约定，加载时恢复相应参数结构。全量微调沿用独立的模型组件保存流程。

## 状态与产物

[`train/state.py`](../ypuddin/train/state.py) 保存可训练参数、优化器、调度器、进度和随机状态。多卡状态包含各进程信息，并校验卡数与训练方式。

推理权重与完整状态分开管理。LoRA / LoKr 输出为 safetensors，全量微调输出为包含组件、配置及清单的 `.model` 目录。项目、版本和任务分别绑定输出路径，避免覆盖其他任务产物。
