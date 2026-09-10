"""Typed training configuration (single source of truth for CLI, service and UI)."""

from __future__ import annotations

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
    family: Literal["anima", "toy"] = F("anima", help="模型族", ui_=ui("model", order=0, control="select"))
    dit_path: str | None = F(None, help="DiT 主干权重（safetensors）", ui_=ui("model", order=10, control="path"))
    text_encoder_path: str | None = F(
        None, help="文本编码器（HF 目录或单文件 safetensors）", ui_=ui("model", order=20, control="path")
    )
    vae_path: str | None = F(None, help="VAE 权重", ui_=ui("model", order=30, control="path"))
    tokenizer_path: str | None = F(
        None, help="附加分词器目录（Anima: 旧版 T5 spiece；留空用内置）", ui_=ui("model", order=40, control="path", advanced=True)
    )
    dtype: DType = F("bf16", help="计算精度（autocast）", ui_=ui("model", order=50, control="select"))
    attention: Literal["auto", "sdpa", "flash", "xformers", "sage"] = F(
        "auto", help="注意力后端", ui_=ui("model", order=60, control="select", advanced=True)
    )


# --------------------------------------------------------------------------- dataset
class CaptionConfig(_Strict):
    prefix: str = F("", help="加在 caption 前的文本", ui_=ui("caption", order=0))
    suffix: str = F("", help="加在 caption 后的文本", ui_=ui("caption", order=10))
    trigger_word: str | None = F(None, help="触发词：放在最前且不参与洗牌/丢弃", ui_=ui("caption", order=20))
    keep_tokens: int = F(0, ge=0, help="前 N 个 tag 固定不洗牌", ui_=ui("caption", order=30))
    shuffle: bool = F(False, help="随机打乱 tag 顺序", ui_=ui("caption", order=40, control="switch"))
    tag_dropout: float = F(0.0, ge=0, le=1, help="每个 tag 被丢弃的概率", ui_=ui("caption", order=50, control="slider", step=0.01))
    caption_dropout: float = F(
        0.0, ge=0, le=1, help="整条 caption 置空的概率（无条件训练，服务于 CFG）", ui_=ui("caption", order=60, control="slider", step=0.01)
    )
    separator: str = F(",", help="tag 分隔符", ui_=ui("caption", order=70, advanced=True))
    wildcard: bool = F(False, help="支持 {a|b} 通配符随机选择", ui_=ui("caption", order=80, control="switch", advanced=True))


class DatasetSourceConfig(_Strict):
    path: str = F(..., help="图片目录（递归）", ui_=ui(control="path"))
    repeats: int = F(1, ge=1, help="重复次数")
    caption_ext: str = F(".txt", help="caption 文件扩展名")
    is_reg: bool = F(False, help="正则集（先验保持）")
    prior_weight: float = F(1.0, ge=0, help="正则集损失权重")
    class_prompt: str | None = F(None, help="缺少 caption 文件时使用的默认 caption")
    caption: CaptionConfig | None = F(None, help="覆盖数据集级 caption 设置")
    resolutions: list[int] | None = F(None, help="覆盖数据集级分辩率列表")


class DatasetConfig(_Strict):
    sources: list[DatasetSourceConfig] = F(default_factory=list, help="数据源列表", ui_=ui("dataset", order=0))
    resolutions: list[int] = F([1024], help="训练分辩率（基准边长，可多个）", ui_=ui("dataset", order=10, control="tags"))
    aspect_ratio_limit: float = F(2.0, ge=1.0, help="分桶允许的最大长宽比", ui_=ui("dataset", order=20))
    area_tolerance: float = F(0.10, ge=0, le=0.5, help="分桶面积容差（相对基准面积）", ui_=ui("dataset", order=30, advanced=True))
    bucket_step: int | None = F(None, ge=8, help="分桶网格步长（默认 64；必须是模型对齐值的倍数）", ui_=ui("dataset", order=35, advanced=True))
    bucket_no_upscale: bool = F(False, help="不放大小图（按原尺寸就近分桶）", ui_=ui("dataset", order=40, control="switch"))
    batch_size: int = F(1, ge=1, help="每个微批的图片数", ui_=ui("dataset", order=50))
    caption: CaptionConfig = F(default_factory=CaptionConfig, help="caption 处理", ui_=ui("dataset", order=60))
    flip: bool = F(False, help="随机水平翻转（缓存双份 latent）", ui_=ui("dataset", order=70, control="switch"))
    masked_loss: bool = F(False, help="用 .mask.png 或 alpha 通道加权损失", ui_=ui("dataset", order=80, control="switch"))
    num_workers: int = F(2, ge=0, help="DataLoader 进程数（Windows 建议 0）", ui_=ui("dataset", order=90, advanced=True))
    cache_dir: str | None = F(None, help="缓存目录（默认 <output_dir>/cache）", ui_=ui("dataset", order=100, control="path", advanced=True))
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
            if r < 32 or r > 8192:
                raise ValueError(f"resolution {r} out of range [32, 8192]")
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
    algo: Algo = F("lokr", help="适配器算法", ui_=ui("adapter", order=0, control="select"))
    rank: int | Literal["full"] = F(
        16, help="秩（LoKr 中为 W2 的低秩；'full' 表示 W2 满矩阵）", ui_=ui("adapter", order=10)
    )
    alpha: float = F(16.0, gt=0, help="缩放 alpha（scale = alpha / rank）", ui_=ui("adapter", order=20))
    factor: int = F(
        -1, help="LoKr 因子：-1 平衡分解；f>0 时小因子 W1 的边长为 f", ui_=ui("adapter", order=30, show_when="adapter.algo == 'lokr'")
    )
    decompose_both: bool = F(
        False, help="LoKr：W1 也做低秩分解", ui_=ui("adapter", order=40, control="switch", advanced=True, show_when="adapter.algo == 'lokr'")
    )
    rs_lora: bool = F(False, help="rsLoRA：scale = alpha / sqrt(rank)", ui_=ui("adapter", order=50, control="switch", advanced=True))
    dora: bool = F(False, help="DoRA 权重分解（幅度/方向）", ui_=ui("adapter", order=60, control="switch"))
    init: Literal["default", "scalar"] = F(
        "default", help="初始化：default（一侧置零）/ scalar（全随机 + 可训练标量从 0 起）", ui_=ui("adapter", order=70, control="select", advanced=True)
    )
    dropout: float = F(0.0, ge=0, le=1, help="对适配器输出的 dropout", ui_=ui("adapter", order=80, advanced=True))
    rank_dropout: float = F(0.0, ge=0, le=1, help="秩轴 dropout", ui_=ui("adapter", order=90, advanced=True))
    module_dropout: float = F(0.0, ge=0, le=1, help="整模块跳过概率", ui_=ui("adapter", order=100, advanced=True))
    preset: str = F("attn-mlp", help="族内目标预设（attn-mlp / attn-only / full-linear / ...）", ui_=ui("adapter", order=110, control="select"))
    rules: list[AdapterRule] = F(default_factory=list, help="按序匹配的覆盖规则", ui_=ui("adapter", order=120, control="rules"))
    mode: Literal["auto", "bypass", "merged"] = F(
        "auto", help="执行路径（auto 按算法与底模精度选择）", ui_=ui("adapter", order=130, control="select", advanced=True)
    )
    param_dtype: Literal["fp32", "bf16"] = F("fp32", help="适配器参数精度（主权重）", ui_=ui("adapter", order=140, control="select", advanced=True))
    lr_scale: dict[str, float] = F(
        default_factory=dict, help="按参数类别的学习率倍率，如 {'up': 16}（LoRA+）或 {'w1': 0.5}", ui_=ui("adapter", order=150, advanced=True)
    )
    resume_weights: str | None = F(None, help="从已有适配器权重热启动", ui_=ui("adapter", order=160, control="path", advanced=True))


# --------------------------------------------------------------------------- objective
class ObjectiveConfig(_Strict):
    timestep_sampling: Literal["uniform", "logit_normal", "shift", "resolution_shift", "mode", "cosmap"] = F(
        "logit_normal", help="时间步采样分布", ui_=ui("objective", order=0, control="select")
    )
    logit_mean: float = F(0.0, help="logit-normal 均值", ui_=ui("objective", order=10, show_when="objective.timestep_sampling in ['logit_normal','shift','resolution_shift']"))
    logit_std: float = F(1.0, gt=0, help="logit-normal 标准差", ui_=ui("objective", order=20, show_when="objective.timestep_sampling in ['logit_normal','shift','resolution_shift']"))
    shift: float = F(3.0, gt=0, help="常数 shift：t' = s·t / (1 + (s-1)·t)", ui_=ui("objective", order=30, show_when="objective.timestep_sampling == 'shift'"))
    mode_scale: float = F(1.29, help="SD3 mode 采样的 scale", ui_=ui("objective", order=40, show_when="objective.timestep_sampling == 'mode'"))
    stratified: bool = F(True, help="批内分层抽样（降低梯度方差）", ui_=ui("objective", order=50, control="switch", advanced=True))
    t_min: float = F(0.0, ge=0, lt=1, help="时间步下限", ui_=ui("objective", order=60, advanced=True))
    t_max: float = F(1.0, gt=0, le=1, help="时间步上限", ui_=ui("objective", order=70, advanced=True))
    loss: Literal["mse", "huber", "pseudo_huber"] = F("mse", help="损失函数", ui_=ui("objective", order=80, control="select"))
    huber_c: float = F(0.1, gt=0, help="Huber delta / pseudo-Huber c", ui_=ui("objective", order=90, show_when="objective.loss != 'mse'"))
    weighting: Literal["none", "sigma_sqrt", "cosmap", "snr_like", "cosmos"] = F(
        "none", help="按时间步的损失加权", ui_=ui("objective", order=100, control="select")
    )
    snr_gamma: float = F(5.0, gt=0, help="snr_like 加权的上限 γ", ui_=ui("objective", order=110, show_when="objective.weighting == 'snr_like'"))
    ip_noise_gamma: float = F(0.0, ge=0, help="输入扰动噪声强度", ui_=ui("objective", order=120, advanced=True))

    @model_validator(mode="after")
    def _range(self) -> ObjectiveConfig:
        if self.t_min >= self.t_max:
            raise ValueError("objective.t_min must be < objective.t_max")
        return self


# --------------------------------------------------------------------------- optimizer / scheduler
class OptimizerConfig(_Strict):
    type: str = F(
        "adamw",
        help="adamw / adamw8bit / lion / prodigy / prodigy_plus_sf / adafactor / came，或 'module.Class'",
        ui_=ui("optimizer", order=0, control="select"),
    )
    lr: float = F(1e-4, gt=0, help="学习率", ui_=ui("optimizer", order=10))
    weight_decay: float = F(0.01, ge=0, help="权重衰减", ui_=ui("optimizer", order=20))
    betas: tuple[float, float] = F((0.9, 0.99), help="Adam betas", ui_=ui("optimizer", order=30, advanced=True))
    eps: float = F(1e-8, gt=0, help="Adam eps", ui_=ui("optimizer", order=40, advanced=True))
    args: dict[str, Any] = F(default_factory=dict, help="透传给优化器的额外参数", ui_=ui("optimizer", order=50, advanced=True))
    grad_clip_norm: float = F(1.0, ge=0, help="梯度范数裁剪（0 关闭）", ui_=ui("optimizer", order=60))
    kahan: bool = F(False, help="bf16 参数的 Kahan 补偿累加", ui_=ui("optimizer", order=70, control="switch", advanced=True))
    fused_backward: bool = F(
        False, help="逐参数在反向中即时更新并释放梯度（需 grad_accum=1）", ui_=ui("optimizer", order=80, control="switch", advanced=True)
    )
    group_lr: dict[str, float] = F(
        default_factory=dict, help="按模块分组的学习率，如 {'llm_adapter': 5e-5, 'te': 2e-5}", ui_=ui("optimizer", order=90, advanced=True)
    )


class SchedulerConfig(_Strict):
    type: Literal["constant", "linear", "cosine", "cosine_restarts", "polynomial", "warmup_stable_decay", "rex"] = F(
        "cosine", help="学习率调度", ui_=ui("scheduler", order=0, control="select")
    )
    warmup_steps: float = F(0, ge=0, help="预热步数（<1 视为总步数比例）", ui_=ui("scheduler", order=10))
    min_lr_ratio: float = F(0.0, ge=0, le=1, help="最终学习率相对初始的比例", ui_=ui("scheduler", order=20))
    num_cycles: int = F(1, ge=1, help="cosine_restarts 周期数", ui_=ui("scheduler", order=30, show_when="scheduler.type == 'cosine_restarts'"))
    power: float = F(1.0, gt=0, help="polynomial 幂", ui_=ui("scheduler", order=40, show_when="scheduler.type == 'polynomial'"))
    decay_steps: float | None = F(
        None, help="warmup_stable_decay 的衰减步数（<1 视为比例）", ui_=ui("scheduler", order=50, show_when="scheduler.type == 'warmup_stable_decay'")
    )


# --------------------------------------------------------------------------- memory
class MemoryConfig(_Strict):
    base_precision: Literal["auto", "bf16", "fp16", "fp32", "fp8_e4m3", "fp8_e5m2"] = F(
        "auto", help="冻结底模的存储精度（fp8 带逐张量缩放，需 CUDA）", ui_=ui("memory", order=0, control="select")
    )
    blocks_to_swap: int = F(0, ge=0, help="换出到 CPU pinned 内存的 block 数", ui_=ui("memory", order=10))
    activation_checkpointing: Literal["none", "block", "unsloth"] = F(
        "none", help="激活检查点：block 逐块重算；unsloth 额外把块输入卸载到 CPU", ui_=ui("memory", order=20, control="select")
    )
    offload_text_encoder: bool = F(False, help="不用时把文本编码器放到 CPU", ui_=ui("memory", order=30, control="switch"))
    compile: bool = F(False, help="逐 block torch.compile", ui_=ui("memory", order=40, control="switch", advanced=True))
    allow_tf32: bool = F(True, help="允许 TF32 matmul", ui_=ui("memory", order=50, control="switch", advanced=True))


# --------------------------------------------------------------------------- loop
class LoopConfig(_Strict):
    max_steps: int | None = F(None, ge=1, help="最大优化步数（与 epochs 至少填一个，先到者停止）", ui_=ui("loop", order=0))
    epochs: int | None = F(10, ge=1, help="训练轮数", ui_=ui("loop", order=10))
    grad_accum: int = F(1, ge=1, help="梯度累积微批数", ui_=ui("loop", order=20))
    mixed_precision: Literal["bf16", "fp16", "no"] = F("bf16", help="混合精度", ui_=ui("loop", order=30, control="select"))
    seed: int = F(42, help="随机种子", ui_=ui("loop", order=40))
    ema: bool = F(False, help="维护适配器权重的 EMA（CPU）", ui_=ui("loop", order=50, control="switch", advanced=True))
    ema_decay: float = F(0.999, gt=0, lt=1, help="EMA 衰减", ui_=ui("loop", order=60, advanced=True, show_when="loop.ema == true"))
    nan_skip_limit: int = F(50, ge=1, help="连续非有限损失跳过次数上限，超过报错", ui_=ui("loop", order=70, advanced=True))
    log_every: int = F(1, ge=1, help="每 N 步发一次 step 事件", ui_=ui("loop", order=80, advanced=True))

    @model_validator(mode="after")
    def _stop(self) -> LoopConfig:
        if self.max_steps is None and self.epochs is None:
            raise ValueError("loop.max_steps or loop.epochs must be set")
        return self


# --------------------------------------------------------------------------- checkpoint
class CheckpointConfig(_Strict):
    output_dir: str = F("outputs/run", help="输出目录", ui_=ui("checkpoint", order=0, control="path"))
    name: str = F("lora", help="产物文件名前缀", ui_=ui("checkpoint", order=10))
    save_every_steps: int | None = F(None, ge=1, help="每 N 步保存权重", ui_=ui("checkpoint", order=20))
    save_every_epochs: int | None = F(1, ge=1, help="每 N 轮保存权重", ui_=ui("checkpoint", order=30))
    save_state_every_steps: int | None = F(None, ge=1, help="每 N 步保存完整可恢复状态", ui_=ui("checkpoint", order=40, advanced=True))
    keep_last_n: int | None = F(None, ge=1, help="只保留最近 N 个权重文件", ui_=ui("checkpoint", order=50, advanced=True))
    save_dtype: DType = F("bf16", help="保存精度", ui_=ui("checkpoint", order=60, control="select"))
    save_on_finish: bool = F(True, help="结束时保存最终权重", ui_=ui("checkpoint", order=70, control="switch"))
    resume: str | None = F(None, help="从完整状态目录恢复", ui_=ui("checkpoint", order=80, control="path"))


# --------------------------------------------------------------------------- sampling / validation / logging
class SamplePrompt(_Strict):
    prompt: str
    negative: str = ""
    seed: int | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    cfg: float | None = None


class SamplingConfig(_Strict):
    enabled: bool = F(False, help="训练期间生成预览图", ui_=ui("sampling", order=0, control="switch"))
    every_steps: int | None = F(None, ge=1, help="每 N 步", ui_=ui("sampling", order=10, show_when="sampling.enabled == true"))
    every_epochs: int | None = F(1, ge=1, help="每 N 轮", ui_=ui("sampling", order=20, show_when="sampling.enabled == true"))
    at_start: bool = F(False, help="训练前先出一组基线图", ui_=ui("sampling", order=30, control="switch", show_when="sampling.enabled == true"))
    prompts: list[SamplePrompt] = F(default_factory=list, help="提示词列表", ui_=ui("sampling", order=40, control="prompts", show_when="sampling.enabled == true"))
    prompts_file: str | None = F(None, help="提示词文件（.txt 每行一条 / .toml）", ui_=ui("sampling", order=50, control="path", show_when="sampling.enabled == true"))
    steps: int | None = F(None, ge=1, help="采样步数（空用族默认）", ui_=ui("sampling", order=60, show_when="sampling.enabled == true"))
    cfg: float | None = F(None, ge=0, help="CFG 强度（空用族默认）", ui_=ui("sampling", order=70, show_when="sampling.enabled == true"))
    shift: float | None = F(None, gt=0, help="采样 shift（空用族默认）", ui_=ui("sampling", order=80, show_when="sampling.enabled == true"))
    width: int = F(1024, ge=64, help="宽", ui_=ui("sampling", order=90, show_when="sampling.enabled == true"))
    height: int = F(1024, ge=64, help="高", ui_=ui("sampling", order=100, show_when="sampling.enabled == true"))
    seed: int = F(0, help="基础种子", ui_=ui("sampling", order=110, show_when="sampling.enabled == true"))
    sampler: Literal["euler"] = F("euler", help="采样器", ui_=ui("sampling", order=120, control="select", show_when="sampling.enabled == true"))


class ValidationConfig(_Strict):
    enabled: bool = F(False, help="确定性验证损失", ui_=ui("validation", order=0, control="switch"))
    split_ratio: float = F(0.0, ge=0, lt=1, help="从训练集按内容哈希切出的比例", ui_=ui("validation", order=10, show_when="validation.enabled == true"))
    sources: list[DatasetSourceConfig] = F(default_factory=list, help="显式验证数据源", ui_=ui("validation", order=20, show_when="validation.enabled == true"))
    every_steps: int | None = F(None, ge=1, help="每 N 步验证", ui_=ui("validation", order=30, show_when="validation.enabled == true"))
    every_epochs: int | None = F(1, ge=1, help="每 N 轮验证", ui_=ui("validation", order=40, show_when="validation.enabled == true"))
    timesteps: list[float] = F([0.1, 0.3, 0.5, 0.7, 0.9], help="固定验证时间步（分位数）", ui_=ui("validation", order=50, control="tags", show_when="validation.enabled == true"))
    max_images: int | None = F(64, ge=1, help="最多使用的验证图数", ui_=ui("validation", order=60, show_when="validation.enabled == true"))
    seed: int = F(1234, help="验证噪声种子", ui_=ui("validation", order=70, advanced=True, show_when="validation.enabled == true"))

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
    tensorboard: bool = F(False, help="写 TensorBoard 日志", ui_=ui("logging", order=0, control="switch"))
    wandb: WandbConfig | None = F(None, help="Weights & Biases", ui_=ui("logging", order=10, advanced=True))
    events_path: str | None = F(None, help="事件 JSONL 文件（默认 <output_dir>/events.jsonl）", ui_=ui("logging", order=20, control="path", advanced=True))
    level: Literal["debug", "info", "warning"] = F("info", help="日志级别", ui_=ui("logging", order=30, control="select", advanced=True))


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
