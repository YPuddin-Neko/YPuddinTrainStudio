"""Typed training configuration (single source of truth for CLI, service and UI)."""

from __future__ import annotations

from pathlib import PureWindowsPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .ui import F, ui

Algo = Literal["lora", "lokr", "loha", "full"]
RuleAlgo = Literal["lora", "lokr", "loha", "full", "none"]
DType = Literal["bf16", "fp16", "fp32"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------- model
class ModelConfig(_Strict):
    family: Literal["anima", "krea2", "toy"] = F(
        "anima",
        help="模型族（anima：Anima 2B；krea2：Krea 2 Raw 12.9B）",
        ui_=ui("model", order=0, control="select"),
    )
    dit_path: str | None = F(
        None,
        help="DiT 主干权重（safetensors；支持官方 bf16 与 ComfyUI fp8_scaled 文件）",
        ui_=ui("model", order=10, control="path"),
    )
    text_encoder_path: str | None = F(
        None,
        help="文本编码器（Anima：Qwen3-0.6B；Krea 2：Qwen3-VL-4B-Instruct）——HF 目录或单文件 safetensors",
        ui_=ui("model", order=20, control="path"),
    )
    vae_path: str | None = F(None, help="VAE 权重", ui_=ui("model", order=30, control="path"))
    tokenizer_path: str | None = F(
        None,
        help="附加分词器目录（Anima: 旧版 T5 spiece；留空用内置）",
        ui_=ui("model", order=40, control="path", advanced=True),
    )
    dtype: DType = F(
        "bf16",
        help="CUDA 上加载模型时使用的精度，默认 bf16；应与显卡和权重兼容。CPU/MPS 实际按 fp32 加载。训练算子的混合精度另由训练设置控制，冻结权重存储精度另由显存设置控制。",
        ui_=ui("model", order=50, control="select"),
    )
    attention: Literal["auto", "sdpa", "sage", "xformers", "flash_attn"] = F(
        "auto",
        help="默认 auto 使用当前模型的自动选择。SDPA 是 PyTorch 内置注意力；xFormers/FlashAttention 需匹配的 CUDA 扩展，Sage 仅用于无梯度推理。通常先用 auto/SDPA，仅在环境检查确认支持后切换扩展。",
        ui_=ui("model", order=60, control="select", advanced=True),
    )


# --------------------------------------------------------------------------- dataset
class CaptionConfig(_Strict):
    prefix: str = F("", help="加在 caption 前的文本", ui_=ui("caption", order=0))
    suffix: str = F("", help="加在 caption 后的文本", ui_=ui("caption", order=10))
    trigger_word: str | None = F(None, help="触发词：放在最前且不参与洗牌/丢弃", ui_=ui("caption", order=20))
    keep_tokens: int = F(0, ge=0, help="前 N 个 tag 固定不洗牌", ui_=ui("caption", order=30))
    shuffle: bool = F(False, help="随机打乱 tag 顺序", ui_=ui("caption", order=40, control="switch"))
    tag_dropout: float = F(
        0.0,
        ge=0,
        le=1,
        help="每个 tag 被丢弃的概率",
        ui_=ui("caption", order=50, control="slider", step=0.01),
    )
    caption_dropout: float = F(
        0.0,
        ge=0,
        le=1,
        help="整条 caption 置空的概率（无条件训练，服务于 CFG）",
        ui_=ui("caption", order=60, control="slider", step=0.01),
    )
    separator: str = F(",", help="tag 分隔符", ui_=ui("caption", order=70, advanced=True))
    wildcard: bool = F(
        False, help="支持 {a|b} 通配符随机选择", ui_=ui("caption", order=80, control="switch", advanced=True)
    )
    cache_variants: int = F(
        8,
        ge=1,
        le=256,
        help="文本编码为 cached 时，每张图预缓存的 caption 随机变体数（仅 shuffle / tag_dropout / wildcard 生效时有意义）",
        ui_=ui("caption", order=90, advanced=True, show_when="dataset.text_encoding != 'online'"),
    )


class DatasetSourceConfig(_Strict):
    path: str = F(..., help="图片目录（递归）", ui_=ui(control="path"))
    repeats: int = F(
        1,
        ge=1,
        help="该来源每张图在每轮出现的次数，默认 1；增大会提高该来源的出现频率，并增加每轮训练项，不会复制磁盘文件。多基准分辨率还会按分辨率数展开。",
    )
    caption_ext: str = F(
        "auto",
        help="标签格式：auto 自动查找同名标签文件，JSON 优先于 TXT；指定 .txt、.json 或自定义后缀时只读取该格式。已有显式后缀配置保持原行为。",
    )
    is_reg: bool = F(
        False,
        help="将该来源作为类别先验保持数据，默认关闭。启用后使用独立标签设置，不自动加入主体触发词，且不会被自动切入验证集；它仍参与训练。",
    )
    prior_weight: float = F(
        1.0,
        ge=0,
        help="仅正则来源生效的单图损失倍率，默认 1 与普通图等权，0 不提供该图的损失梯度；改变它不会改变正则图出现次数。出现频率由图片数和 repeats 决定。",
    )
    class_prompt: str | None = F(None, help="缺少 caption 文件时使用的默认 caption")
    caption: CaptionConfig | None = F(None, help="覆盖数据集级 caption 设置")
    resolutions: list[int] | None = F(None, help="覆盖数据集级分辩率列表")


class DatasetConfig(_Strict):
    sources: list[DatasetSourceConfig] = F(
        default_factory=list, help="数据源列表", ui_=ui("dataset", order=0)
    )
    resolutions: list[int] = F(
        [1024],
        help="分桶的基准面积：1024 表示每桶约 1024×1024 像素，并非把所有图片裁成正方形。填写多个值会让每张图在每个基准分辨率各训练一次；通常先用一个值。",
        ui_=ui("dataset", order=10, control="tags", show_when="dataset.resolution_mode == 'bucket'"),
    )
    resolution_mode: Literal["bucket", "native"] = F(
        "bucket",
        help="分桶按接近原图的长宽比选择训练尺寸，等比缩放后中心裁剪；原生保留每图尺寸，仅裁去模型对齐所需的边缘，超预算时按策略缩小或报错，不放大小图。",
        ui_=ui("dataset", order=5, control="select"),
    )
    native_max_pixels: int = F(
        1_048_576,
        ge=1024,
        le=67_108_864,
        help="原生模式单图及一次前向的像素上限；1048576 = 1024²。不同尺寸分组前向后按图片数累积梯度，像素预算不保证整体显存不会溢出",
        ui_=ui("dataset", order=11, show_when="dataset.resolution_mode == 'native'"),
    )
    native_max_side: int = F(
        4096,
        ge=32,
        le=8192,
        help="原生模式单边上限；超限时等比缩小后裁去尺寸对齐边缘，或按策略报错",
        ui_=ui("dataset", order=12, show_when="dataset.resolution_mode == 'native'", advanced=True),
    )
    native_overflow: Literal["downscale", "error"] = F(
        "downscale",
        help="超出像素或单边预算：等比缩小，或报错要求调整；不会悄悄跳过图片",
        ui_=ui(
            "dataset",
            order=13,
            control="select",
            show_when="dataset.resolution_mode == 'native'",
            advanced=True,
        ),
    )
    aspect_ratio_limit: float = F(
        2.0,
        ge=1.0,
        help="分桶最大长边/短边比，默认 2 对应最宽 2:1、最高 1:2。更狭长的原图仍会进入最近的桶并裁剪；大量长图可提高此值或改用原生模式。",
        ui_=ui("dataset", order=20, show_when="dataset.resolution_mode == 'bucket'"),
    )
    area_tolerance: float = F(
        0.10,
        ge=0,
        le=0.5,
        help="候选桶面积相对基准面积的允许偏差，默认 0.10 即 ±10%；用于容纳尺寸对齐后的不同长宽比，不是裁剪比例。通常保持默认。",
        ui_=ui("dataset", order=30, advanced=True, show_when="dataset.resolution_mode == 'bucket'"),
    )
    bucket_step: int | None = F(
        None,
        ge=8,
        help="候选桶宽高的间隔，留空通常为 64 像素，且必须是模型尺寸对齐值的倍数。较小间隔可产生更多形状，也会使同尺寸批次更分散；先用默认并查看数据计划。",
        ui_=ui("dataset", order=35, advanced=True, show_when="dataset.resolution_mode == 'bucket'"),
    )
    bucket_no_upscale: bool = F(
        False,
        help="默认允许小图放大到所选桶；启用后缩小该桶以容纳原图，再按模型尺寸要求对齐。想逐图保留原始大小可选原生模式。",
        ui_=ui("dataset", order=40, control="switch", show_when="dataset.resolution_mode == 'bucket'"),
    )
    batch_size: int = F(
        1,
        ge=1,
        help="每个逻辑批次的目标图片数。分桶按相同尺寸组批，不足一批的尾部不会复制图片补齐；原生模式会按尺寸和像素预算拆成多次前向。实际更新还受梯度累积影响。",
        ui_=ui("dataset", order=50),
    )
    caption: CaptionConfig = F(
        default_factory=CaptionConfig, help="caption 处理", ui_=ui("dataset", order=60)
    )
    flip: bool = F(
        False, help="随机水平翻转（缓存双份 latent）", ui_=ui("dataset", order=70, control="switch")
    )
    masked_loss: bool = F(
        False, help="用 .mask.png 或 alpha 通道加权损失", ui_=ui("dataset", order=80, control="switch")
    )
    num_workers: int = F(
        2, ge=0, help="DataLoader 进程数（Windows 建议 0）", ui_=ui("dataset", order=90, advanced=True)
    )
    cache_dir: str | None = F(
        None,
        help="缓存目录（默认 <output_dir>/cache）",
        ui_=ui("dataset", order=100, control="path", advanced=True),
    )
    cache_latents: bool = F(True, help="预编码并缓存 latents", ui_=ui("dataset", order=110, control="switch"))
    text_encoding: Literal["auto", "online", "cached"] = F(
        "auto",
        help="文本编码：online 每步在线编码（支持 caption 增强），cached 预缓存后卸载编码器",
        ui_=ui("dataset", order=120, control="select"),
    )

    @field_validator("resolutions")
    @classmethod
    def _res(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("resolutions must not be empty")
        out = sorted({int(r) for r in v}, reverse=True)
        for r in out:
            if r < 32 or r > MAX_SIDE:
                raise ValueError(f"resolution {r} out of range [32, {MAX_SIDE}]")
        return out


# --------------------------------------------------------------------------- adapter
class AdapterRule(_Strict):
    match: str = F(..., help="模块名匹配：glob，或 're:' 前缀正则")
    algo: RuleAlgo | None = F(None, help="覆盖算法；none 表示排除")
    rank: int | Literal["full"] | None = None
    alpha: float | None = None
    factor: int | None = None
    lr: float | None = F(None, help="该组参数的学习率（覆盖 optimizer.lr）")
    dropout: float | None = None
    rank_dropout: float | None = None


class AdapterConfig(_Strict):
    algo: Algo = F(
        "lokr",
        help="选择目标层的训练方式。LoRA/LoKr/LoHa 学习附加权重；full 直接训练目标预设选中的线性层完整权重，并导出差分。full 不会自动选中整个模型。",
        ui_=ui("adapter", order=0, control="select"),
    )
    rank: int | Literal["full"] = F(
        16,
        help="低秩分解的大小，默认 16。LoKr 的 full 表示保留完整的两个 Kronecker 因子 W1/W2，不做低秩拆分，仍是 LoKr 适配器；它不同于算法 full 的目标层完整权重训练。整数秩过大时 LoKr 也会自动保留对应完整因子。",
        ui_=ui("adapter", order=10),
    )
    alpha: float = F(
        16.0,
        gt=0,
        help="适配器更新的缩放系数，通常按 alpha/rank 缩放；默认 alpha=rank=16 时为 1。rsLoRA 使用 alpha/√rank。LoKr 两个因子都为完整矩阵时缩放固定为 1，此值不生效。",
        ui_=ui("adapter", order=20),
    )
    factor: int = F(
        -1,
        help="LoKr 的整数分解方式：默认 -1 在每个层维度中选接近平方根的一对因子；正数优先按该因子分解，不能整除时取不大于它的可用因子并给出提示。它决定 W1/W2 的形状，不是训练秩。",
        ui_=ui("adapter", order=30, show_when="adapter.algo == 'lokr'"),
    )
    decompose_both: bool = F(
        False,
        help="在秩足够小时，也把 LoKr 较小的 W1 因子做低秩拆分。默认关闭；rank=full 时无效，整数秩不够小时也会保留完整 W1。",
        ui_=ui("adapter", order=40, control="switch", advanced=True, show_when="adapter.algo == 'lokr'"),
    )
    rs_lora: bool = F(
        False,
        help="rsLoRA：scale = alpha / sqrt(rank)",
        ui_=ui("adapter", order=50, control="switch", advanced=True),
    )
    dora: bool = F(False, help="DoRA 权重分解（幅度/方向）", ui_=ui("adapter", order=60, control="switch"))
    init: Literal["default", "scalar"] = F(
        "default",
        help="初始化：default（一侧置零）/ scalar（全随机 + 可训练标量从 0 起）",
        ui_=ui("adapter", order=70, control="select", advanced=True),
    )
    dropout: float = F(
        0.0, ge=0, le=1, help="对适配器输出的 dropout", ui_=ui("adapter", order=80, advanced=True)
    )
    rank_dropout: float = F(0.0, ge=0, le=1, help="秩轴 dropout", ui_=ui("adapter", order=90, advanced=True))
    module_dropout: float = F(
        0.0, ge=0, le=1, help="整模块跳过概率", ui_=ui("adapter", order=100, advanced=True)
    )
    preset: str = F(
        "attn-mlp",
        help="选择哪些线性层参与训练：选项和层数由当前模型族提供。attn-mlp 通常包含注意力和 MLP；full-linear 表示扩大目标层范围，不会把适配器算法改为 full。",
        ui_=ui("adapter", order=110, control="select"),
    )
    rules: list[AdapterRule] = F(
        default_factory=list, help="按序匹配的覆盖规则", ui_=ui("adapter", order=120, control="rules")
    )
    mode: Literal["auto", "bypass", "merged"] = F(
        "auto",
        help="执行路径（auto 按算法与底模精度选择）",
        ui_=ui("adapter", order=130, control="select", advanced=True),
    )
    param_dtype: Literal["fp32", "bf16"] = F(
        "fp32", help="适配器参数精度（主权重）", ui_=ui("adapter", order=140, control="select", advanced=True)
    )
    lr_scale: dict[str, float] = F(
        default_factory=dict,
        help="按参数类别的学习率倍率，如 {'up': 16}（LoRA+）或 {'w1': 0.5}",
        ui_=ui("adapter", order=150, advanced=True),
    )
    resume_weights: str | None = F(
        None, help="从已有适配器权重热启动", ui_=ui("adapter", order=160, control="path", advanced=True)
    )


# --------------------------------------------------------------------------- objective
class ObjectiveConfig(_Strict):
    timestep_sampling: Literal["uniform", "logit_normal", "shift", "resolution_shift", "mode", "cosmap"] = F(
        "shift",
        help="决定训练时抽到哪些噪声强度：t 越大噪声越多。默认 shift 先做 logit-normal 抽样再应用 shift=3；uniform 均匀抽样，resolution_shift 按图像 token 数调整。它不同于生成预览图的噪声调度器，修改前应固定数据与种子做对比。",
        ui_=ui("objective", order=0, control="select"),
    )
    logit_mean: float = F(
        0.0,
        help="logit-normal 在 sigmoid 变换前的均值，默认 0；提高会偏向较高噪声，降低会偏向较低噪声。仅对应训练时间步分布，通常先保持默认。",
        ui_=ui(
            "objective",
            order=10,
            show_when="objective.timestep_sampling in ['logit_normal','shift','resolution_shift']",
        ),
    )
    logit_std: float = F(
        1.0,
        gt=0,
        help="logit-normal 在 sigmoid 变换前的标准差，默认 1；提高会增加靠近低噪声和高噪声端点的样本。仅在有分布对比目标时调整。",
        ui_=ui(
            "objective",
            order=20,
            show_when="objective.timestep_sampling in ['logit_normal','shift','resolution_shift']",
        ),
    )
    res_shift_tokens: tuple[int, int] = F(
        (256, 4096),
        help="按 token 数插值 shift 的两个参考位置，通用默认 256 和 4096；它们不是图像尺寸上限，区间外仍会外推。修改时须与下方 mu 成对理解；Krea 2 的族推理默认参考上端为 6400。",
        ui_=ui(
            "objective",
            order=25,
            advanced=True,
            show_when="objective.timestep_sampling == 'resolution_shift'",
        ),
    )
    res_shift_mu: tuple[float, float] = F(
        (0.5, 1.15),
        help="上述两个 token 参考位置对应的 mu，默认 0.5 和 1.15；先线性插值 mu，再用 exp(mu) 得到 shift。通常保留当前模型的配置，仅在研究分辨率相关噪声分布时调整。",
        ui_=ui(
            "objective",
            order=26,
            advanced=True,
            show_when="objective.timestep_sampling == 'resolution_shift'",
        ),
    )
    shift: float = F(
        3.0,
        gt=0,
        help="训练时间步偏移，默认 3；大于 1 将样本推向高噪声，1 不偏移。公式为 t'=s·t/(1+(s-1)·t)。该字段只影响训练 shift 分布，预览另有采样 shift。",
        ui_=ui("objective", order=30, show_when="objective.timestep_sampling == 'shift'"),
    )
    mode_scale: float = F(
        1.29,
        help="mode 训练时间步分布的形状系数，默认 1.29；仅在选择 mode 时生效。通常保持默认，并通过时间步诊断观察实际抽样分布。",
        ui_=ui("objective", order=40, show_when="objective.timestep_sampling == 'mode'"),
    )
    stratified: bool = F(
        True,
        help="默认开启，将同一次前向批次的随机分位数分散到不同区间，再打乱顺序；批内只有 1 张图时与普通抽样一致。可关闭以对照独立随机抽样。",
        ui_=ui("objective", order=50, control="switch", advanced=True),
    )
    t_min: float = F(
        0.0,
        ge=0,
        lt=1,
        help="训练时间步裁剪下限，默认 0；实际计算避开精确端点。提高会把更低噪声样本裁到此值，而不是重新抽样；应小于上限。",
        ui_=ui("objective", order=60, advanced=True),
    )
    t_max: float = F(
        1.0,
        gt=0,
        le=1,
        help="训练时间步裁剪上限，默认 1；实际计算避开精确端点。降低会把更高噪声样本裁到此值，而不是重新抽样；应大于下限。",
        ui_=ui("objective", order=70, advanced=True),
    )
    loss: Literal["mse", "huber", "pseudo_huber"] = F(
        "mse",
        help="预测速度与目标速度的误差度量，默认 MSE 平方误差。Huber/pseudo-Huber 改变大误差的惩罚方式；更换后损失数值不可直接与 MSE 比较，也不保证样图更好。",
        ui_=ui("objective", order=80, control="select"),
    )
    huber_c: float = F(
        0.1,
        gt=0,
        help="Huber/pseudo-Huber 从小误差区域过渡到大误差区域的尺度，默认 0.1；MSE 不使用此值。只在使用对应损失并有误差分布依据时调整。",
        ui_=ui("objective", order=90, show_when="objective.loss != 'mse'"),
    )
    weighting: Literal["none", "sigma_sqrt", "cosmap", "snr_like", "cosmos"] = F(
        "none",
        help="给不同噪声时间步的损失乘权重，默认 none 等权；它不改变时间步抽样概率。其他方案会改变优化重点和损失量级，建议先保留默认建立对照。",
        ui_=ui("objective", order=100, control="select"),
    )
    snr_gamma: float = F(
        5.0,
        gt=0,
        help="snr_like 权重中 SNR 的截断值，默认 5；仅选择 snr_like 时生效。它调节不同噪声区间的损失倍率，通常保持默认。",
        ui_=ui("objective", order=110, show_when="objective.weighting == 'snr_like'"),
    )
    ip_noise_gamma: float = F(
        0.0,
        ge=0,
        help="仅给训练输入额外叠加噪声，目标仍使用原始噪声；默认 0 关闭。启用会改变训练任务，应通过固定验证和样图对比，而非将其当作通用提质开关。",
        ui_=ui("objective", order=120, advanced=True),
    )

    @model_validator(mode="after")
    def _range(self) -> ObjectiveConfig:
        if self.t_min >= self.t_max:
            raise ValueError("objective.t_min must be < objective.t_max")
        return self


# --------------------------------------------------------------------------- optimizer / scheduler
class OptimizerConfig(_Strict):
    type: str = F(
        "adamw",
        help="默认 AdamW。内置选项对应真实优化器实现；8-bit 选项需要 CUDA 和 bitsandbytes，其他扩展优化器需要对应依赖。缺失时会报错，不会自动换用其他优化器。高级用户也可填已安装的 module.Class；其额外参数填写在优化器参数中。",
        ui_={
            "x-ui": {
                **ui("optimizer", order=0, control="select")["x-ui"],
                "options": [
                    "adamw",
                    "adam",
                    "sgd",
                    "adamw8bit",
                    "lion",
                    "lion8bit",
                    "prodigy",
                    "prodigy_plus_sf",
                    "adafactor",
                    "came",
                    "adamw_sf",
                ],
                "allow_custom": True,
            }
        },
    )
    lr: float = F(
        1e-4,
        gt=0,
        help="每次更新的基础学习率，默认 0.0001。训练振荡或参数变化过快时可降低；换优化器时应按该优化器要求设置，不同算法的数值不能直接比较。目标层规则和分组倍率可覆盖它。",
        ui_=ui("optimizer", order=10),
    )
    weight_decay: float = F(
        0.01,
        ge=0,
        help="对适用参数的权重衰减强度，默认 0.01，0 表示关闭。LoKr 的 W1 与 DoRA 幅度参数在分组时不应用衰减。",
        ui_=ui("optimizer", order=20),
    )
    betas: tuple[float, float] = F(
        (0.9, 0.99),
        help="动量统计的两个衰减系数，默认 0.9 和 0.99。仅传给界面支持此字段的内置优化器；通常保持默认，自定义优化器可在额外参数中指定。",
        ui_=ui(
            "optimizer",
            order=30,
            advanced=True,
            show_when="optimizer.type in ['adamw','adam','adamw8bit','prodigy','prodigy_plus_sf','adamw_sf','came']",
        ),
    )
    eps: float = F(
        1e-8,
        gt=0,
        help="Adam 系列分母中的数值稳定项，默认 0.00000001。仅传给 Adam/AdamW/AdamW8bit/AdamW Schedule-Free；其他算法的不同 eps 格式使用额外参数。",
        ui_=ui(
            "optimizer",
            order=40,
            advanced=True,
            show_when="optimizer.type in ['adamw','adam','adamw8bit','adamw_sf']",
        ),
    )
    args: dict[str, Any] = F(
        default_factory=dict,
        help="原样传给所选优化器的构造参数；同名值会覆盖上方通用参数。仅填写该优化器文档支持的参数，名称或类型错误会明确报错。",
        ui_=ui("optimizer", order=50, advanced=True),
    )
    grad_clip_norm: float = F(
        1.0,
        ge=0,
        help="更新前将整体梯度范数限制到此值，默认 1，0 关闭；不是逐个参数的数值上限。频繁触发时可结合梯度曲线、学习率和数据检查原因。",
        ui_=ui("optimizer", order=60),
    )
    kahan: bool = F(
        False,
        help="为低精度训练参数保留 fp32 副本并补偿舍入误差，默认关闭；会增加内存。仅在确实用低精度适配器参数时考虑，不支持与 Schedule-Free 优化器组合。",
        ui_=ui("optimizer", order=70, control="switch", advanced=True),
    )
    fused_backward: bool = F(
        False,
        help="预留的反向即时更新选项，当前尚未实现，必须保持关闭；开启会明确拒绝启动，不会静默改成普通训练。",
        ui_=ui("optimizer", order=80, control="switch", advanced=True),
    )
    group_lr: dict[str, float] = F(
        default_factory=dict,
        help="按模块分组的学习率，如 {'llm_adapter': 5e-5, 'te': 2e-5}",
        ui_=ui("optimizer", order=90, advanced=True),
    )

    @field_validator("fused_backward")
    @classmethod
    def _fused_backward_supported(cls, value: bool) -> bool:
        if value:
            raise ValueError("fused_backward is not implemented; use false for normal optimizer steps")
        return value

    @model_validator(mode="after")
    def _schedule_free_kahan(self) -> OptimizerConfig:
        if self.kahan and (
            self.type.lower() in {"prodigy_plus_sf", "adamw_sf"} or "schedulefree" in self.type.lower()
        ):
            raise ValueError("kahan cannot be combined with a schedule-free optimizer")
        return self


class SchedulerConfig(_Strict):
    type: Literal[
        "constant", "linear", "cosine", "cosine_restarts", "polynomial", "warmup_stable_decay", "rex"
    ] = F(
        "cosine",
        help="控制学习率随优化步骤变化的曲线，默认 cosine 逐渐衰减；constant 保持基础倍率。它不控制图像生成的噪声时间步。换优化器时应确认该算法对外部学习率调度的要求。",
        ui_=ui("scheduler", order=0, control="select"),
    )
    warmup_steps: float = F(
        0,
        ge=0,
        help="开头逐步提高学习率的时长，默认 0 不预热；0 到 1 之间表示总优化步数比例，例如 0.05 为 5%，≥1 表示步数。可用于观察和缓和起始更新。",
        ui_=ui("scheduler", order=10),
    )
    min_lr_ratio: float = F(
        0.0,
        ge=0,
        le=1,
        help="衰减下限相对基础学习率的倍率，默认 0；0.1 表示最低为基础值的 10%。constant 不使用此下限，周期重启会重新提高学习率。",
        ui_=ui("scheduler", order=20),
    )
    num_cycles: int = F(
        1,
        ge=1,
        help="余弦重启调度的周期数，默认 1；更多周期会反复降再升学习率，仅 cosine_restarts 使用。需要多次重启实验时才调整。",
        ui_=ui("scheduler", order=30, show_when="scheduler.type == 'cosine_restarts'"),
    )
    power: float = F(
        1.0,
        gt=0,
        help="多项式衰减的幂，默认 1 为线性衰减；更大值会更早降低学习率。仅 polynomial 使用，通常先保留默认。",
        ui_=ui("scheduler", order=40, show_when="scheduler.type == 'polynomial'"),
    )
    decay_steps: float | None = F(
        None,
        help="预热—稳定—衰减调度最后的衰减时长，留空为总步数的约 10%；0 到 1 之间表示比例，≥1 表示步数。仅 warmup_stable_decay 使用。",
        ui_=ui("scheduler", order=50, show_when="scheduler.type == 'warmup_stable_decay'"),
    )


# --------------------------------------------------------------------------- memory
class MemoryConfig(_Strict):
    base_precision: Literal["auto", "bf16", "fp16", "fp32", "fp8_e4m3", "fp8_e5m2"] = F(
        "auto",
        help="冻结线性层权重的存储精度，默认 auto 保留加载后的精度；不会将适配器参数自动变为该精度。FP8 带逐张量缩放且需要 CUDA，应在显存受限时结合样图与训练检查使用。",
        ui_=ui("memory", order=0, control="select"),
    )
    blocks_to_swap: int = F(
        0,
        ge=0,
        help="把多少个模型块的冻结权重暂放 CPU、在执行前移回设备，默认 0 不换出。提高可减少设备驻留权重，但增加内存占用和传输等待；超过实际块数时按全部块处理，不能与 compile 同用。",
        ui_=ui("memory", order=10),
    )
    activation_checkpointing: Literal["none", "block", "unsloth"] = F(
        "none",
        help="减少反向前保存的中间激活，默认 none 全程保留；block 在反向时重算，unsloth 还把块输入卸载到 CPU。会增加重算或传输工作，显存不足时再按模型支持情况选择。",
        ui_=ui("memory", order=20, control="select"),
    )
    offload_text_encoder: bool = F(
        False,
        help="在线编码标签时，在每次编码后把文本编码器移到 CPU，默认关闭；可减少驻留显存但增加传输。cached 文本模式已预编码并卸载编码器，无需依靠此开关。",
        ui_=ui("memory", order=30, control="switch"),
    )
    compile: bool = F(
        False,
        help="对模型块使用 torch.compile，默认关闭；首次会编译，实际收益取决于设备与输入形状。当前仅 CUDA 路径启用，不能与块换出同时使用，先确认普通训练可运行。",
        ui_=ui("memory", order=40, control="switch", advanced=True),
    )
    allow_tf32: bool = F(
        True,
        help="默认允许支持此格式的 NVIDIA GPU 使用 TF32 矩阵乘法；仅设置 PyTorch CUDA 后端许可，不影响 CPU/MPS。严格对照数值精度时可关闭。",
        ui_=ui("memory", order=50, control="switch", advanced=True),
    )


# --------------------------------------------------------------------------- loop
class LoopConfig(_Strict):
    max_steps: int | None = F(
        None,
        ge=1,
        help="最多执行多少次优化器更新，默认留空；它不是图片数或采样步数。与轮数至少设置一个，同时设置时先到者结束。需要固定更新预算时填写。",
        ui_=ui("loop", order=0),
    )
    epochs: int | None = F(
        10,
        ge=1,
        help="遍历展开后训练项的次数，默认 10；repeats 和多分辨率都已计入每轮训练项。留空时必须设置最大步数，先通过数据计划确认实际更新总数。",
        ui_=ui("loop", order=10),
    )
    grad_accum: int = F(
        1,
        ge=1,
        help="累计多少个批次再更新一次参数，默认 1；可增加一次更新覆盖的图片数，而不同时放入更大的批次。尾批会按实际累积量处理，不能简单把每步都视为固定图片数。",
        ui_=ui("loop", order=20),
    )
    mixed_precision: Literal["bf16", "fp16", "no"] = F(
        "bf16",
        help="CUDA 训练算子的自动混合精度，默认 bf16；no 关闭 autocast，fp16 需设备与模型数值兼容。CPU/MPS 当前关闭 autocast；该字段不同于冻结权重存储精度与导出文件精度。",
        ui_=ui("loop", order=30, control="select"),
    )
    seed: int = F(
        42,
        help="训练随机种子，默认 42，影响数据顺序、标签变体、噪声等。对比参数时保持一致；不同设备、依赖版本或数据仍可能产生不同结果，预览图另有自己的种子。",
        ui_=ui("loop", order=40),
    )
    ema: bool = F(
        False,
        help="在 CPU 维护适配器权重的指数移动平均，默认关闭；开启后额外保存 EMA 权重用于对比，会增加内存和文件占用。当前训练预览仍使用当时的普通权重。",
        ui_=ui("loop", order=50, control="switch", advanced=True),
    )
    ema_decay: float = F(
        0.999,
        gt=0,
        lt=1,
        help="EMA 对历史权重的保留比例，默认 0.999；越接近 1，平均权重变化越慢。仅启用 EMA 时有效，不保证平均权重一定优于当前权重。",
        ui_=ui("loop", order=60, advanced=True, show_when="loop.ema == true"),
    )
    nan_skip_limit: int = F(
        50,
        ge=1,
        help="连续出现 NaN/Inf 损失或梯度时的容忍上限，默认 50，达到后报错停止。它用于防止无限跳过；遇到问题应检查数据、精度和学习率，不宜只提高上限。",
        ui_=ui("loop", order=70, advanced=True),
    )
    log_every: int = F(
        1,
        ge=1,
        help="每多少次优化更新记录一条训练指标，默认 1 每步记录。增大会减少曲线点数和日志量，不改变实际更新次数；细查训练过程时保持 1。",
        ui_=ui("loop", order=80, advanced=True),
    )

    @model_validator(mode="after")
    def _stop(self) -> LoopConfig:
        if self.max_steps is None and self.epochs is None:
            raise ValueError("loop.max_steps or loop.epochs must be set")
        return self


# --------------------------------------------------------------------------- checkpoint
class CheckpointConfig(_Strict):
    output_dir: str = F("outputs/run", help="输出目录", ui_=ui("checkpoint", order=0, control="path"))
    name: str = F(
        "lora",
        min_length=1,
        max_length=150,
        pattern=r'^[^/\\:*?"<>|\x00-\x1f\x7f]+$',
        help="产物文件名前缀，不含目录或路径分隔符",
        ui_=ui("checkpoint", order=10),
    )
    save_every_steps: int | None = F(
        None,
        ge=1,
        help="每多少次优化更新导出权重，默认留空不按步保存。适合较长任务保留中间结果；权重文件不含优化器与随机状态。",
        ui_=ui("checkpoint", order=20),
    )
    save_every_epochs: int | None = F(
        1,
        ge=1,
        help="每多少轮导出权重，默认 1 每轮保存；留空关闭此触发器。与按步保存分别生效，可能在同一步产生不同标签的文件。",
        ui_=ui("checkpoint", order=30),
    )
    save_state_every_steps: int | None = F(
        None,
        ge=1,
        help="每多少次优化更新保存完整恢复状态，默认留空。状态含原始训练参数、优化器、数据位置与随机状态，体积通常大于导出权重；需要中断续训保障时设置。",
        ui_=ui("checkpoint", order=40, advanced=True),
    )
    keep_last_n: int | None = F(
        None,
        ge=1,
        help="仅保留最近 N 组按步保存的权重（普通/EMA 成组；轮次与最终产物保留）",
        ui_=ui("checkpoint", order=50, advanced=True),
    )
    save_dtype: DType = F(
        "bf16",
        help="导出权重文件的精度，默认 bf16；不会改变当前训练参数或完整恢复状态的精度。选择 fp32 会增大文件，可用于减少导出舍入。",
        ui_=ui("checkpoint", order=60, control="select"),
    )
    save_on_finish: bool = F(
        True, help="结束时保存最终权重", ui_=ui("checkpoint", order=70, control="switch")
    )
    resume: str | None = F(
        None,
        help="从完整 state 目录恢复训练，默认留空从头开始；会校验数据和模型身份。只有 .safetensors 权重时应使用适配器热启动，无法据此恢复优化器和数据进度。",
        ui_=ui("checkpoint", order=80, control="path"),
    )

    @field_validator("name")
    @classmethod
    def _filename_prefix(cls, value: str) -> str:
        if not value.strip() or value.endswith((".", " ")) or PureWindowsPath(value).is_reserved():
            raise ValueError(
                "checkpoint.name must be a filename prefix; trailing dots/spaces and Windows device names are not allowed"
            )
        return value


# --------------------------------------------------------------------------- sampling / validation / logging
MAX_SIDE = 8192  # pixels; also the ceiling for dataset resolutions


class SamplePrompt(_Strict):
    prompt: str
    negative: str = ""
    seed: int | None = None
    width: int | None = F(None, ge=32, le=MAX_SIDE)
    height: int | None = F(None, ge=32, le=MAX_SIDE)
    steps: int | None = F(None, ge=1, le=1000)
    cfg: float | None = F(None, ge=0, allow_inf_nan=False)


class SamplingConfig(_Strict):
    # Service-owned destination. Omitted for CLI compatibility (<run_dir>/samples).
    output_dir: str | None = None
    enabled: bool = F(
        False,
        help="按下面的时机生成训练预览图，默认关闭；开启后至少填写一条提示词或提示词文件。预览会占用生成时间，不参与梯度更新。",
        ui_=ui("sampling", order=0, control="switch"),
    )
    every_steps: int | None = F(
        None,
        ge=1,
        help="每多少次优化更新生成预览，默认留空不按步触发。设置较大间隔可减少频繁生成；不影响训练最大步数。",
        ui_=ui("sampling", order=10, show_when="sampling.enabled == true"),
    )
    every_epochs: int | None = F(
        1,
        ge=1,
        help="每多少轮生成预览，默认 1 每轮一次；留空关闭按轮触发。它与按步触发独立，同时到期会各生成一组。",
        ui_=ui("sampling", order=20, show_when="sampling.enabled == true"),
    )
    at_start: bool = F(
        False,
        help="默认关闭；开启后在首次训练更新前生成基线图，便于与后续结果对比。从已完成该阶段的完整状态恢复时不会重复生成。",
        ui_=ui("sampling", order=30, control="switch", show_when="sampling.enabled == true"),
    )
    prompts: list[SamplePrompt] = F(
        default_factory=list,
        help="固定预览内容，默认空列表；每条可单独覆盖种子、尺寸、步数和 CFG，未填时继承下方设置。对比训练进度时保持提示词和种子一致。",
        ui_=ui("sampling", order=40, control="prompts", show_when="sampling.enabled == true"),
    )
    prompts_file: str | None = F(
        None,
        help="可选的额外提示词文件，.txt 每行一条，.toml 可携带逐条参数；文件内容会追加到上方列表。留空只使用列表，避免两处重复填写。",
        ui_=ui("sampling", order=50, control="path", show_when="sampling.enabled == true"),
    )
    steps: int | None = F(
        None,
        ge=1,
        le=1000,
        help="生成一张预览的积分步数，留空使用模型族默认（Anima 25、Krea 2 为 28）。更多步通常增加生成耗时，不等于训练更多步，也不保证更好；单条提示词设置优先。",
        ui_=ui("sampling", order=60, show_when="sampling.enabled == true"),
    )
    cfg: float | None = F(
        None,
        ge=0,
        allow_inf_nan=False,
        help="提示词引导强度，留空使用模型族默认（Anima 4、Krea 2 为 5.5）。1 只用正向条件，0 使用负向/空条件；更高值会放大条件差异，不保证效果更好。单条提示词设置优先。",
        ui_=ui("sampling", order=70, show_when="sampling.enabled == true"),
    )
    shift: float | None = F(
        None,
        gt=0,
        allow_inf_nan=False,
        help="调整预览积分时间步的噪声分布，留空使用模型族默认（Anima 为 3，Krea 2 按图像 token 数计算）。1 不偏移，更大值偏向高噪声时间段；不同于训练分布里的 shift。",
        ui_=ui("sampling", order=80, show_when="sampling.enabled == true"),
    )
    width: int = F(
        1024,
        ge=32,
        le=MAX_SIDE,
        help="默认预览宽度，1024 像素；实际会向下对齐到模型要求的倍数。与训练图分辨率独立，增大会增加生成内存和计算需求，单条提示词可覆盖。",
        ui_=ui("sampling", order=90, show_when="sampling.enabled == true"),
    )
    height: int = F(
        1024,
        ge=32,
        le=MAX_SIDE,
        help="默认预览高度，1024 像素；实际会向下对齐到模型要求的倍数。可配合宽度比较横图或竖图，单条提示词可覆盖。",
        ui_=ui("sampling", order=100, show_when="sampling.enabled == true"),
    )
    seed: int = F(
        0,
        help="预览初始噪声种子，默认 0；未单独设种子的第 i 条提示词使用基础种子+i（从 0 计）。固定种子便于比较权重变化，与训练随机种子独立。",
        ui_=ui("sampling", order=110, show_when="sampling.enabled == true"),
    )
    sampler: Literal["euler", "heun", "er_sde"] = F(
        "euler",
        help="预览图的数值求解算法，默认 Euler 保持旧行为；Heun 在非末步增加一次预测做修正，ER-SDE 使用带历史项的随机求解。改变它不改变训练目标，应固定种子与调度器做效果对比。",
        ui_=ui("sampling", order=120, control="select", show_when="sampling.enabled == true"),
    )
    scheduler: Literal["uniform", "simple", "sgm_uniform", "normal"] = F(
        "uniform",
        help="决定每个预览积分步骤经过的噪声时间点。默认 uniform 保留旧版连续均匀网格再应用 shift；simple、sgm_uniform、normal 使用不同网格或端点。它与采样算法、学习率调度是独立设置。",
        ui_=ui("sampling", order=130, control="select", show_when="sampling.enabled == true"),
    )
    er_sde_order: Literal[1, 2, 3] = F(
        3,
        help="ER-SDE 允许的最高阶数，默认 3；开始时历史不足会从低阶逐步升阶。可选 1/2 做对照，阶数更高不保证每个模型或步数下都更好。",
        ui_=ui(
            "sampling",
            order=140,
            control="select",
            advanced=True,
            show_when="sampling.enabled == true && sampling.sampler == 'er_sde'",
        ),
    )
    er_sde_s_noise: float = F(
        1.0,
        ge=0,
        le=1,
        allow_inf_nan=False,
        help="ER-SDE 沿途追加噪声的倍率，默认 1，范围 0～1；0 关闭沿途随机项，但初始噪声仍由种子生成。只影响 ER-SDE，调整时保持其他预览参数一致。",
        ui_=ui(
            "sampling",
            order=150,
            advanced=True,
            show_when="sampling.enabled == true && sampling.sampler == 'er_sde'",
        ),
    )


class ValidationConfig(_Strict):
    enabled: bool = F(
        False,
        help="在固定验证图片和噪声上计算损失，默认关闭；开启需设置验证划分比例或单独来源。用于比较训练变化，不会对验证图片反向更新。",
        ui_=ui("validation", order=0, control="switch"),
    )
    split_ratio: float = F(
        0.0,
        ge=0,
        lt=1,
        help="按图片内容哈希从普通训练图固定划出验证集的比例，默认 0 不自动划分；正则图不参与该划分。少量数据可能分不到足够图片，应查看实际数据计划。",
        ui_=ui("validation", order=10, show_when="validation.enabled == true"),
    )
    sources: list[DatasetSourceConfig] = F(
        default_factory=list,
        help="显式验证数据源",
        ui_=ui("validation", order=20, show_when="validation.enabled == true"),
    )
    every_steps: int | None = F(
        None, ge=1, help="每 N 步验证", ui_=ui("validation", order=30, show_when="validation.enabled == true")
    )
    every_epochs: int | None = F(
        1, ge=1, help="每 N 轮验证", ui_=ui("validation", order=40, show_when="validation.enabled == true")
    )
    timesteps: list[float] = F(
        [0.1, 0.3, 0.5, 0.7, 0.9],
        help="训练时间步分布的固定分位数，默认 0.1/0.3/0.5/0.7/0.9；会经过当前训练分布和 shift 转换，不一定等于同名实际 t。更改后验证损失不可直接与旧设置比较。",
        ui_=ui("validation", order=50, control="tags", show_when="validation.enabled == true"),
    )
    max_images: int | None = F(
        64,
        ge=1,
        help="每次验证最多使用的图数，默认 64；留空使用全部验证图。增加可覆盖更多样本，也会延长验证时间；实际图数不会超过验证集大小。",
        ui_=ui("validation", order=60, show_when="validation.enabled == true"),
    )
    seed: int = F(
        1234,
        help="验证噪声种子",
        ui_=ui("validation", order=70, advanced=True, show_when="validation.enabled == true"),
    )

    @field_validator("timesteps")
    @classmethod
    def _ts(cls, v: list[float]) -> list[float]:
        if not v or any(t <= 0 or t >= 1 for t in v):
            raise ValueError("validation.timesteps must be within (0, 1)")
        return sorted(v)


class WandbConfig(_Strict):
    project: str = "ypuddin"
    run_name: str | None = None
    entity: str | None = None


class LoggingConfig(_Strict):
    tensorboard: bool = F(
        False,
        help="额外写入本地 TensorBoard 格式的训练指标与样图，默认关闭；前端任务日志和曲线不依赖此开关。需要用 TensorBoard 查看时启用。",
        ui_=ui("logging", order=0, control="switch"),
    )
    wandb: WandbConfig | None = F(None, help="Weights & Biases", ui_=ui("logging", order=10, advanced=True))
    events_path: str | None = F(
        None,
        help="事件 JSONL 文件（默认 <output_dir>/events.jsonl）",
        ui_=ui("logging", order=20, control="path", advanced=True),
    )
    level: Literal["debug", "info", "warning"] = F(
        "info", help="日志级别", ui_=ui("logging", order=30, control="select", advanced=True)
    )


# --------------------------------------------------------------------------- root
class TrainConfig(_Strict):
    model: ModelConfig = F(default_factory=ModelConfig)
    dataset: DatasetConfig = F(default_factory=DatasetConfig)
    adapter: AdapterConfig = F(default_factory=AdapterConfig)
    objective: ObjectiveConfig = F(default_factory=ObjectiveConfig)
    optimizer: OptimizerConfig = F(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = F(default_factory=SchedulerConfig)
    memory: MemoryConfig = F(default_factory=MemoryConfig)
    loop: LoopConfig = F(default_factory=LoopConfig)
    checkpoint: CheckpointConfig = F(default_factory=CheckpointConfig)
    sampling: SamplingConfig = F(default_factory=SamplingConfig)
    validation: ValidationConfig = F(default_factory=ValidationConfig)
    logging: LoggingConfig = F(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def _cross(self) -> TrainConfig:
        if self.optimizer.fused_backward and self.loop.grad_accum != 1:
            raise ValueError("optimizer.fused_backward requires loop.grad_accum == 1")
        if self.sampling.enabled and not self.sampling.prompts and not self.sampling.prompts_file:
            raise ValueError("sampling.enabled requires sampling.prompts or sampling.prompts_file")
        if self.validation.enabled and self.validation.split_ratio == 0 and not self.validation.sources:
            raise ValueError("validation.enabled requires split_ratio > 0 or explicit sources")
        if self.adapter.algo != "lokr" and self.adapter.rank == "full":
            raise ValueError("adapter.rank='full' is only meaningful for lokr")
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        schema = cls.model_json_schema()
        schema["x-ui-groups"] = list(__import__("ypuddin.config.ui", fromlist=["GROUPS"]).GROUPS)
        return schema
