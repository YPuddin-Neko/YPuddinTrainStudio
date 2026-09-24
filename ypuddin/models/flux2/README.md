# FLUX.2 Klein 后端

本模块使用 Diffusers 的 `Flux2Transformer2DModel`、`AutoencoderKLFlux2` 和 Transformers 的 Qwen3 实现，训练范围为 Klein Base 4B / 9B。

## 权重与文本组件

| 主模型 | 文本编码器 | 选取的隐藏层 |
| --- | --- | --- |
| Klein Base 4B | Qwen3-4B | 9、18、27 |
| Klein Base 9B | Qwen3-8B | 9、18、27 |

支持原始 BFL 单文件及本地 Diffusers 组件目录。单独的主模型文件需指定文本编码器和 VAE。单文件加载所需的分词器及模型配置随模块提供，加载过程不下载权重。

Base 与蒸馏权重可能形状相同。自动识别依赖 `model_index.json` 或经校验的 `.ypuddin.json`；信息不足时由 `model.flux2_variant` 指定。蒸馏、KV、dev 和量化权重不进入当前训练路径。

## 编码与加载

VAE 使用后验 mode，将 2×2 空间位置打包为 128 通道，并应用权重中的固定归一化参数。解码执行对应的逆变换，缓存身份包含 VAE 权重和归一化配置。

文本条件保留 512 个 token，使用关闭 thinking 的 Qwen 模板。模型使用 CFG，不使用内嵌 guidance 输入。

`load()` 建立主模型结构和延迟加载的编码器。`materialize_backbone()` 在编码器卸载后读取主模型权重，随后注入 LoRA / LoKr。文本编码仍需容纳完整文本编码器及其激活。

分块换出要求逐块梯度检查点。Diffusers 检查点路径按位置参数传递张量，使换出器能跟踪反向传播所需的输入梯度。

## 导出

默认导出使用 `lora_transformer_` 前缀及下划线连接的 Diffusers 模块名。Diffusers 的 Kohya 解析路径采用另一套 BFL 模块名称，不能将两种命名格式混用。

LoRA 可通过库内的 PEFT 转换路径处理缩放和模块名。LoKr 不使用该 PEFT 加载路径。导出格式与使用方法见主项目的权重转换接口。

## 上游参考

- [BFL model geometry](https://github.com/black-forest-labs/flux2/blob/main/src/flux2/model.py)
- [BFL text conditioning](https://github.com/black-forest-labs/flux2/blob/main/src/flux2/text_encoder.py)
- [BFL VAE normalization](https://github.com/black-forest-labs/flux2/blob/main/src/flux2/autoencoder.py)
- [Diffusers Klein pipeline](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py)
- [ComfyUI adapter key mapping](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/lora.py)
- [ComfyUI transformer key mapping](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/utils.py)
