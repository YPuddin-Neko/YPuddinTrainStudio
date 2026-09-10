# Changelog

## 0.1.0（未发布，2026-09）

首个可用版本：训练核心（LoRA / LoKr / LoHa / DoRA / 全量差分，精确暂停恢复，确定性验证，block swap，
unsloth 式激活卸载，fp8 冻结底模，Kahan bf16 优化器），Anima 模型族（vendored Cosmos-Predict2 DiT +
Qwen-Image VAE + Qwen3 文本管线），数据流水线（内容哈希缓存、caption 变体缓存、可恢复分桶采样器），
服务 API（项目 / 数据集 / 任务队列 / SSE / 产物 / 模型注册）与 Web 界面，`ypuddin smoke` 自检，
`studio.sh` / `studio.bat` 一键部署。

破坏性变更会记录在这里（数据目录布局、SQLite 结构、配置字段改名）。目前没有。
