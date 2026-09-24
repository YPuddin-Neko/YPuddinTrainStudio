
const labels: Record<string, string> = {
  'memory': '显存估算',
  'model.family': '模型系列', 'model.dit_path': '主模型 / DiT', 'model.text_encoder_path': '文本编码器',
  'model.text_encoder_2_path': '第二文本编码器', 'model.prediction_type': 'SDXL 预测方式',
  'model.zero_terminal_snr': '零终点信噪比（Zero SNR）',
  'model.sdxl_max_token_length': 'SDXL 文本长度',
  'model.training_guidance': '训练引导值', 'sampling.guidance': '模型引导值',
  'model.flux2_variant': 'Klein 类型', 'model.krea2_variant': 'Krea 2 类型',
  'training.mode': '训练方式', 'training.train_backbone': '训练主模型（UNet / DiT）', 'training.train_text_encoder': '训练文本编码器', 'training.resume_weights': '全量模型起始权重',
  'model.vae_path': 'VAE', 'model.tokenizer_path': '分词器目录', 'model.dtype': '底模加载精度', 'model.attention': '注意力后端',
  'dataset.sources': '训练数据源', 'dataset.resolutions': '训练分辨率', 'dataset.aspect_ratio_limit': '最大长宽比',
  'dataset.resolution_mode': '分辨率模式', 'dataset.image_fit': '图片适配方式', 'dataset.native_max_pixels': '原生像素预算',
  'dataset.native_max_side': '原生最长边', 'dataset.native_overflow': '超出预算时',
  'dataset.area_tolerance': '面积容差', 'dataset.bucket_step': '分桶步长', 'dataset.bucket_no_upscale': '不放大小图',
  'dataset.batch_size': '批大小', 'dataset.flip': '随机水平翻转', 'dataset.masked_loss': '遮罩加权训练',
  'dataset.num_workers': '数据加载线程', 'dataset.cache_dir': '缓存目录', 'dataset.cache_latents': '训练图像缓存',
  'dataset.text_encoding': '标签处理方式', 'dataset.caption.prefix': '标签前缀', 'dataset.caption.suffix': '标签后缀',
  'dataset.caption.trigger_word': '触发词', 'dataset.caption.keep_tokens': '保留前几个标签', 'dataset.caption.shuffle': '打乱标签顺序',
  'dataset.caption.tag_dropout': '单个标签丢弃率', 'dataset.caption.caption_dropout': '整条标签丢弃率',
  'dataset.caption.separator': '标签分隔符', 'dataset.caption.wildcard': '启用通配符',
  'dataset.caption.cache_variants': '预缓存标签变体数',
  'sampling.scheduler': '采样调度器', 'sampling.er_sde_order': 'ER-SDE 求解阶数', 'sampling.er_sde_s_noise': 'ER-SDE 随机噪声强度',
  'adapter.algo': '适配器算法', 'adapter.rank': 'Rank / 秩', 'adapter.alpha': 'Alpha / 缩放', 'adapter.factor': 'LoKr 分解因子',
  'adapter.decompose_both': '双矩阵低秩分解', 'adapter.rs_lora': 'Rank 稳定缩放', 'adapter.dora': '启用 DoRA',
  'adapter.init': '初始化方式', 'adapter.dropout': '输出丢弃率', 'adapter.rank_dropout': '秩丢弃率',
  'adapter.module_dropout': '模块丢弃率', 'adapter.preset': '适配器作用范围', 'adapter.rules': '逐层覆盖规则',
  'adapter.mode': '适配器计算模式', 'adapter.param_dtype': '可训练参数精度', 'adapter.lr_scale': '学习率缩放',
  'adapter.resume_weights': '已有适配器权重', 'objective.timestep_sampling': '时间步采样',
  'objective.logit_mean': 'Logit 均值', 'objective.logit_std': 'Logit 标准差', 'objective.res_shift_tokens': '分辨率偏移基准',
  'objective.res_shift_mu': '分辨率偏移系数', 'objective.shift': '时间步偏移', 'objective.mode_scale': 'Mode 系数',
  'objective.stratified': '分层时间步采样', 'objective.t_min': '时间步下限', 'objective.t_max': '时间步上限',
  'objective.loss': '损失函数', 'objective.huber_c': 'Huber 系数', 'objective.weighting': '损失加权',
  'objective.scale_v_pred_loss_like_noise_pred': '按 ε 预测尺度缩放 v 损失', 'objective.v_pred_like_loss': '附加 v 预测损失系数', 'objective.debiased_estimation_loss': '去偏损失加权',
  'objective.snr_gamma': 'SNR Gamma', 'objective.ip_noise_gamma': '输入扰动强度', 'optimizer.type': '优化器',
  'optimizer.lr': '学习率', 'optimizer.weight_decay': '权重衰减', 'optimizer.betas': '更新平滑程度',
  'optimizer.eps': '数值稳定常量', 'optimizer.args': '优化器附加参数', 'optimizer.grad_clip_norm': '梯度保护阈值',
  'optimizer.kahan': '低精度更新补偿', 'optimizer.fused_backward': '融合反向更新', 'optimizer.group_lr': '参数组学习率',
  'optimizer.d_coef': '自适应步长倍率（D Coef）', 'optimizer.d0': '初始步长估计（D0）',
  'optimizer.prodigy_steps': '自适应估计步数',
  'optimizer.beta3': '步长估计平滑（β3）', 'optimizer.use_bias_correction': '修正初期统计偏差',
  'optimizer.safeguard_warmup': '预热阶段估计保护', 'optimizer.growth_rate': '步长增长上限',
  'optimizer.slice_p': '步长估计取样间隔', 'optimizer.decouple': '独立计算权重衰减',
  'optimizer.d_limiter': '限制步长突然增长', 'optimizer.schedulefree_c': '权重平均速度',
  'optimizer.split_groups': '各参数组独立估计步长', 'optimizer.split_groups_mean': '合并各组步长估计',
  'optimizer.factored': '分解统计以节省显存', 'optimizer.factored_fp32': '分解统计使用 FP32',
  'optimizer.use_stableadamw': '稳定更新（StableAdamW）', 'optimizer.stochastic_rounding': '低精度写回随机舍入',
  'optimizer.weight_decay_by_lr': '衰减强度随学习率变化', 'optimizer.use_schedulefree': '免调度权重平均',
  'optimizer.use_speed': 'SPEED 步长估计', 'optimizer.use_cautious': 'Cautious 更新方向筛选',
  'optimizer.use_grams': 'Grams 更新方向调整', 'optimizer.use_adopt': 'ADOPT 更新方式',
  'optimizer.use_orthograd': 'OrthoGrad 梯度方向处理', 'optimizer.use_focus': 'FOCUS 更新方式',
  'optimizer.beta2': '梯度大小平滑（β2）', 'optimizer.min_lr': '自适应学习率下限',
  'optimizer.max_lr': '自适应学习率上限', 'optimizer.lr_bump': '学习率调整增量',
  'optimizer.clip_threshold': '更新幅度保护阈值',
  'scheduler.type': '学习率调度', 'scheduler.warmup_steps': '预热步数 / 比例', 'scheduler.min_lr_ratio': '最低学习率比例',
  'scheduler.num_cycles': '调度周期数', 'scheduler.power': '多项式幂', 'scheduler.decay_steps': '衰减步数',
  'memory.base_precision': '底模存储精度', 'memory.blocks_to_swap': '换出到 CPU 的层数',
  'memory.activation_checkpointing': '重算中间结果（梯度检查点）', 'memory.offload_text_encoder': '卸载文本编码器',
  'memory.compile': '编译模型', 'memory.allow_tf32': '允许 TF32', 'loop.max_steps': '最大训练步数',
  'loop.epochs': '训练轮数', 'loop.grad_accum': '梯度累积', 'loop.mixed_precision': '混合精度', 'loop.seed': '随机种子',
  'loop.gpu_count': '训练显卡数量',
  'loop.distributed_strategy': '多卡训练方式',
  'loop.deterministic': '可复现训练',
  'loop.ema': '启用 EMA', 'loop.ema_decay': 'EMA 衰减', 'loop.nan_skip_limit': '无效梯度跳过上限', 'loop.log_every': '日志间隔',
  'checkpoint.output_dir': '训练权重保存位置', 'checkpoint.name': '权重文件名', 'checkpoint.save_every_steps': '每隔几步保存',
  'checkpoint.save_every_epochs': '每隔几轮保存', 'checkpoint.save_state_every_steps': '完整状态保存间隔',
  'checkpoint.keep_last_n': '保留最近几个状态', 'checkpoint.save_dtype': '权重保存精度', 'checkpoint.save_on_finish': '结束时保存权重',
  'checkpoint.resume': '恢复完整训练状态', 'sampling.enabled': '生成训练预览', 'sampling.every_steps': '每隔几步预览',
  'sampling.every_epochs': '每隔几轮预览', 'sampling.at_start': '开始前生成预览', 'sampling.prompts': '预览提示词',
  'sampling.prompts_file': '提示词文件', 'sampling.steps': '采样步数', 'sampling.cfg': 'CFG 引导强度',
  'sampling.shift': '采样时间步偏移', 'sampling.width': '预览宽度', 'sampling.height': '预览高度',
  'sampling.seed': '预览种子', 'sampling.sampler': '采样器', 'validation.enabled': '启用验证集',
  'validation.split_ratio': '验证集划分比例', 'validation.sources': '独立验证数据源', 'validation.every_steps': '每隔几步验证',
  'validation.every_epochs': '每隔几轮验证', 'validation.timesteps': '验证时间步', 'validation.max_images': '验证图片上限',
  'validation.seed': '验证种子', 'logging.tensorboard': 'TensorBoard 日志', 'logging.wandb': 'Weights & Biases',
  'logging.events_path': '训练事件文件', 'logging.level': '日志级别',
};

export function configFieldLabel(path: string, fallback: string, english = false) {
  if (english && path === 'model.sdxl_max_token_length') return 'SDXL caption length';
  if (path === 'loop.gpu_count') return english ? 'Training GPU count' : labels[path];
  if (path === 'loop.distributed_strategy') return english ? 'Multi-GPU training strategy' : labels[path];
  if (path === 'loop.deterministic') return english ? 'Reproducible training' : labels[path];
  if (path === 'adapter.preset') return english ? 'Adapter scope' : labels[path];
  if (english && path.startsWith('training.')) return ({mode:'Training mode',train_backbone:'Train main model (UNet / DiT)',train_text_encoder:'Train text encoder',resume_weights:'Initial full-model weights'} as Record<string,string>)[path.slice(9)] || fallback;
  if (english && path.startsWith('optimizer.')) {
    const optimizerLabels: Record<string, string> = {
      type: 'Optimizer', lr: 'Learning rate', weight_decay: 'Weight decay', betas: 'Update smoothing', eps: 'EPS · numerical stability',
      args: 'Extra optimizer arguments', grad_clip_norm: 'Gradient clipping threshold', kahan: 'Low-precision update compensation', group_lr: 'Parameter-group learning rates',
      d_coef: 'Adaptive step multiplier (D Coef)', d0: 'Initial step estimate (D0)', beta3: 'Step estimate smoothing (β3)', prodigy_steps: 'Adaptive estimation steps',
      use_bias_correction: 'Early-statistics bias correction', safeguard_warmup: 'Warmup estimation safeguard', growth_rate: 'Step-growth limit', slice_p: 'Estimation sampling interval',
      decouple: 'Decoupled weight decay', d_limiter: 'Limit sudden step growth', schedulefree_c: 'Weight averaging speed', split_groups: 'Estimate each group independently',
      split_groups_mean: 'Combine group estimates', factored: 'Factored statistics', factored_fp32: 'FP32 factored statistics', use_stableadamw: 'StableAdamW update normalization',
      stochastic_rounding: 'Stochastic low-precision rounding', weight_decay_by_lr: 'Scale decay with the learning rate', use_schedulefree: 'Schedule-Free weight averaging',
      use_speed: 'SPEED step estimation', use_cautious: 'Cautious direction filtering', use_grams: 'Grams direction adjustment', use_adopt: 'ADOPT updates', use_orthograd: 'OrthoGrad directions', use_focus: 'FOCUS updates',
      beta2: 'Gradient-size smoothing (β2)', min_lr: 'Adaptive learning-rate minimum', max_lr: 'Adaptive learning-rate maximum', lr_bump: 'Learning-rate increment', clip_threshold: 'Normalized update clipping threshold',
    };
    return optimizerLabels[path.slice(10)] || fallback;
  }
  return english ? fallback : labels[path] || fallback;
}

const optimizerEnglishHelp: Record<string, string> = {
  type: 'Choose the algorithm used to update trainable parameters. Start with AdamW unless you need another optimizer’s behavior. Optional optimizers require their corresponding installed packages.',
  d_coef: 'Scales the automatically estimated step size. Default: 1. Higher values generally produce stronger updates; use this instead of changing the managed base learning rate.',
  d0: 'Starting estimate for the adaptive step size. Default: 0.000001. This is the initial estimate, not the live D value during training.',
  beta3: 'Smooths the step-size estimate. Select Automatic to use the square root of β2; usually keep this selected.',
  use_bias_correction: 'Prodigy corrects early statistical bias; PPSF uses RAdam rectification and automatic warmup. Off by default; changes early updates.',
  safeguard_warmup: 'Removes the influence of external warmup from step-size estimation. Off by default; consider it when using external warmup.',
  growth_rate: 'Maximum growth multiplier for the D estimate per update. Select Unlimited to remove the limit. 1.02 allows at most about 2% growth per update.',
  slice_p: 'Samples every N elements for step-size estimation. 1 uses all elements; larger values reduce estimator memory and detail.',
  decouple: 'Applies weight decay separately from the gradient update. On by default; turning it off adds decay to the gradient.',
  prodigy_steps: 'Number of optimizer updates used to estimate the step size. 0 keeps estimating throughout training; a positive value freezes the estimate afterward.',
  d_limiter: 'Limits sudden growth in the step-size estimate. On by default; SPEED uses its own estimation method when enabled.',
  schedulefree_c: 'Changes the speed of Schedule-Free weight averaging. 0 uses the author’s default averaging; usually keep 0.',
  split_groups: 'Estimates step sizes independently for parameter groups. On by default; turning it off shares the estimate without enabling manual group rates.',
  split_groups_mean: 'Uses the harmonic mean of the per-group step estimates. Off by default; requires independent group estimation.',
  factored: 'Stores suitable gradient statistics in factored form to reduce optimizer-state memory. On by default; turning it off stores full statistics.',
  factored_fp32: 'Stores factored statistics in FP32 to reduce rounding errors. On by default; applies only when factored statistics are enabled.',
  use_stableadamw: 'Normalizes updates inside the optimizer using StableAdamW. On by default; cannot be combined with the Adam-atan2 option in the EPS field.',
  stochastic_rounding: 'Uses stochastic rounding when writing low-precision parameters. On by default; does not affect FP32 parameters.',
  weight_decay_by_lr: 'Scales weight decay with the current effective learning rate. On by default; turning it off changes the decay scale.',
  use_schedulefree: 'Uses Schedule-Free weight averaging without an external learning-rate curve. On by default; turning it off enables external scheduling.',
  use_speed: 'Uses SPEED step-size estimation. Off by default; changes the estimation path and ignores the D growth limiter.',
  use_cautious: 'Keeps updates aligned with the current gradient direction. Off by default; cannot be combined with Grams.',
  use_grams: 'Sets update directions using the current gradient. Off by default; cannot be combined with Cautious.',
  use_adopt: 'Changes gradient normalization order and adds ADOPT limits. Off by default; affects early updates.',
  use_orthograd: 'Removes the gradient component parallel to the weight direction. Off by default; use to compare alternative update directions.',
  use_focus: 'Selects FOCUS updates. Off by default; requires factored statistics to be off and EPS to be set.',
  beta2: 'Smooths the gradient-size estimate. Default: 0.999. Higher values use longer history and react more slowly to changes.',
  min_lr: 'Lower bound for per-element adaptive learning rates. Default: 0.0000001; must not exceed the upper bound.',
  max_lr: 'Upper bound for per-element adaptive learning rates. Default: 0.001; limits the adaptive rate.',
  lr_bump: 'Increment added to or subtracted from the adaptive learning rate. Larger increments change the rate more rapidly; usually keep the default.',
  clip_threshold: 'Clips normalized updates inside Automagic. Default: 1. This is separate from external gradient clipping.',
};

/** Explain what a choice changes before introducing the implementation term. */
export function configFieldHelp(path: string, fallback: string | undefined, english = false, optimizerType?: string, scheduleFree = false) {
  if (path === 'optimizer.eps' && optimizerType === 'prodigy_plus_sf') return english
    ? 'Prevents division by very small estimates. Select Adam-atan2 to use that mode instead; StableAdamW and FOCUS must be disabled.'
    : '防止梯度大小估计过小时除法不稳定。选择 Adam-atan2 才会切换算法，此时需关闭 StableAdamW 和 FOCUS。';
  if (path === 'optimizer.grad_clip_norm' && optimizerType === 'prodigy_plus_sf') return english
    ? 'Clips gradients before they reach the optimizer. PPSF defaults this external clipping to 0 (off). StableAdamW provides a different normalization inside the optimizer; these controls are not interchangeable.'
    : '在梯度进入优化器前做外部裁剪。PPSF 默认 0 关闭；StableAdamW 是优化器内部的更新归一化，两者并不是同一个保护功能。';
  if (path === 'optimizer.grad_clip_norm' && optimizerType === 'automagic') return english
    ? 'Clips gradients before they reach the optimizer. Automagic defaults this external clipping to 0 (off); its internal clipping threshold applies to normalized updates instead.'
    : '在梯度进入优化器前做外部裁剪。Automagic 默认 0 关闭；其内部保护阈值限制的是归一化后的更新，两者作用位置不同。';
  if (path === 'optimizer.betas' && scheduleFree) return english
    ? 'β1 controls schedule-free weight averaging; β2 smooths the estimate of gradient size. Usually keep this optimizer’s defaults.'
    : 'β1 控制免调度训练中的权重平均，β2 平滑梯度大小的估计。通常保留当前优化器的默认值。';
  const help: Record<string, [string, string]> = {
    'dataset.resolutions': ['单个分辨率填 1024；多个用逗号或空格分隔，如 1024, 1536。填写正整数边长，不写 1024×1024。1024 表示每桶约 1024×1024 像素；每张图会在每个基准分辨率各训练一次，增加总样本和步数。', 'Enter one size as 1024, or separate multiple sizes with commas or spaces, e.g. 1024, 1536. Use positive integer side lengths, not 1024×1024. A base of 1024 gives roughly 1024×1024 pixels per bucket. Each image trains at every base resolution, increasing samples and steps.'],
    'adapter.resume_weights': ['可选。加载已有 LoRA / LoKr 权重作为本次训练起点；优化器和步数重新开始。接着上次任务训练请使用完整训练状态。', 'Optional. Start from existing LoRA / LoKr weights with a fresh optimizer and step count. Use a full training state to resume a previous run.'],
    'loop.deterministic': ['默认关闭。在相同配置、设备和软件环境下提高重复训练的一致性。开启后可能固定部分计算精度和注意力设置，增加显存与耗时；具体值会显示在对应字段。完整续训需保持原设置和环境。', 'Off by default. Improves repeatability with the same configuration, device and software environment. May manage precision and attention settings and increase memory use and runtime; effective values appear in the fields. Keep the same settings and environment when resuming.'],
    'loop.mixed_precision': ['控制训练运算的自动混合精度。关闭只停用自动混合精度，不改变权重本身的精度。底模存储和导出文件的精度分别设置；可复现训练可能调整实际计算精度。', 'Controls automatic mixed precision during training. Disabling it does not change weight precision. Base-weight storage and export precision are configured separately; reproducible training may adjust compute precision.'],
    'memory.base_precision': ['适配器训练时转换其覆盖的未量化冻结线性层，降低存储精度可节省显存，但可能影响训练质量；原模型文件和适配器精度不变。FP8 需要受支持的 CUDA 或海光环境。全量微调使用沿用精度或 FP32。', 'Converts unquantized frozen linear layers covered by adapters. Lower storage precision saves memory but may affect training quality; source files and adapter precision are unchanged. FP8 requires supported CUDA or DTK hardware. Full fine-tuning uses the loaded precision or FP32.'],
    'memory.activation_checkpointing': ['少保存中间结果，在反向传播时重新计算，以额外耗时换取更少显存。可与梯度累积同时使用。', 'Saves fewer intermediate results and recomputes them during backward to reduce memory at the cost of time. Can be combined with gradient accumulation.'],
    'loop.grad_accum': ['累计指定数量的小批次后更新一次参数。有效批大小 = 每卡批大小 × 梯度累积 × 显卡数量。', 'Updates parameters after the specified number of minibatches. Effective batch size = per-GPU batch size × accumulation steps × GPU count.'],
    'loop.gpu_count': ['1 为单卡；多卡任务会等待所需显卡全部空闲后启动。批大小按每张卡计算；可用训练方式取决于当前平台。', '1 uses one GPU. Multi-GPU jobs wait for all required devices to be free. Batch size is per GPU; available strategies depend on the platform.'],
    'loop.distributed_strategy': ['数据并行：每卡保留完整模型并分配训练数据。显存分片：模型参数、梯度和优化器状态分摊到多卡，适合单卡放不下的模型。分片要求冻结文本编码器，支持主模型全量微调或 LoRA／LoKr，优化器限 AdamW、Adafactor、SGD。FSDP 需要 Linux CUDA/DTK；适配器分片不支持 FP8 底模或整层丢弃。', 'Data parallelism keeps a full model per GPU and splits training data. Memory sharding distributes model parameters, gradients and optimizer states across GPUs. Sharding requires frozen text encoders and supports full backbone training or LoRA/LoKr with AdamW, Adafactor or SGD. FSDP requires Linux CUDA/DTK; adapter sharding does not support FP8 base weights or module dropout.'],
    'optimizer.lr': ['控制参数更新的基础步长。过大容易不稳定，过小学习较慢；全量微调需单独设置，自适应优化器会调整实际步长。', 'Sets the base parameter-update step size. Too large can be unstable; too small slows learning. Set it separately for full fine-tuning; adaptive optimizers adjust the effective step size.'],
    'optimizer.weight_decay': ['给权重施加衰减约束，避免权重持续变大。通常保留默认值；过大可能让模型学不到细节，0 表示关闭。', 'Applies a decay constraint to weights. Usually keep the default; too much can prevent learning details. Set to 0 to disable.'],
    'optimizer.betas': ['β1 平滑更新方向，β2 平滑梯度大小的估计。通常保留优化器默认值；调大后反应更平缓，也会更慢适应变化。', 'β1 smooths the update direction; β2 smooths the estimate of gradient size. Usually keep the optimizer defaults. Higher values smooth changes more but respond more slowly.'],
    'optimizer.grad_clip_norm': ['限制异常大的梯度，降低数值失控的风险。通常保留 1；0 表示关闭梯度裁剪。', 'Limits unusually large gradients to reduce numerical instability. Usually keep 1; set to 0 to disable gradient clipping.'],
    'optimizer.kahan': ['使用 Kahan 补偿保留低精度更新中容易丢失的小数值，会增加状态内存。仅在支持的优化器与精度组合下使用；通常保持关闭。', 'Uses Kahan compensation to retain small values that low-precision updates can lose, adding state memory. Use only with supported optimizers and precision modes; usually leave off.'],
    'optimizer.eps': ['防止除以接近零的数，通常保留优化器默认值。', 'Prevents division by values near zero. Usually keep the optimizer default.'],
    'optimizer.beta3': ['步长估计所用的历史平滑系数。留空时使用 β2 的平方根；通常保留自动。', optimizerEnglishHelp.beta3],
    'optimizer.growth_rate': ['限制 D 估计每一步最多增长的倍率。留空时不设上限；1.02 表示最多增加约 2%，通常保留不限。', optimizerEnglishHelp.growth_rate],
    'optimizer.args': ['仅用于当前优化器支持的额外参数。已有专用控件的参数请在对应位置设置；不确定名称和作用时留空。', 'Only for extra parameters supported by the selected optimizer. Use dedicated controls where available; leave empty unless you know the parameter and its effect.'],
    'optimizer.group_lr': ['分别覆盖不同参数组的学习率。通常留空，使用统一学习率；自动管理学习率时不可覆盖。', 'Overrides the learning rate of individual parameter groups. Usually leave empty to use the shared rate; unavailable when the rate is managed automatically.'],
  };
  return help[path]?.[english ? 1 : 0] || (english && path.startsWith('optimizer.') ? optimizerEnglishHelp[path.slice(10)] : undefined) || fallback;
}

export function configFieldHint(path: string, english = false, optimizerType?: string, scheduleFree = false) {
  const hints: Record<string, [string, string]> = {
    'model.family': ['B 表示十亿个主模型参数，不含文本编码器与 VAE。Klein 的实际规模由模型版本决定。', 'B denotes billion backbone parameters, excluding text encoders and the VAE. The Klein variant determines its actual size.'],
    'loop.deterministic': ['DTK 会按模型与训练方式管理计算精度。实际设置在参数检查后显示，可能增加显存和耗时。', 'DTK manages compute precision for the model and training mode. Effective settings appear after configuration validation and may increase memory use and runtime.'],
    'loop.gpu_count': ['1 为单卡；多卡可选择数据并行或显存分片。', '1 uses one GPU. Multiple GPUs can use data parallelism or memory sharding.'],
    'loop.distributed_strategy': ['数据并行分配训练数据；显存分片将模型分摊到多卡。', 'Data parallelism splits training data; memory sharding distributes the model across GPUs.'],
    'dataset.batch_size': ['每张显卡一次处理的图片数；有效批次还会乘以显卡数和梯度累积。', 'Images processed by each GPU per batch. Effective batch size also includes GPU count and gradient accumulation.'],
    'training.mode': ['适配器生成附加权重；全量微调直接更新并保存所选模型组件。', 'Adapters save additional weights; full fine-tuning updates and saves the selected model components.'],
    'memory.base_precision': ['降低底模存储精度可节省显存，但可能影响训练质量。默认不转换。', 'Lower precision saves memory but may affect training quality. Conversion is off by default.'],
    'memory.activation_checkpointing': ['重新计算中间结果，节省显存但增加耗时。', 'Recomputes intermediate results to save memory at the cost of time.'],
    'loop.grad_accum': ['累积多个小批次后再更新一次参数。', 'Accumulate multiple minibatches before one parameter update.'],
    'adapter.mode': ['自动模式分别计算底模和适配器的输出，不降低权重精度。', 'Automatic computes the base model and adapter outputs separately without reducing weight precision.'],

    'optimizer.lr': ['基础更新步长；全量微调需单独设置，自适应优化器按自身规则管理。', 'Base update step size; set it separately for full fine-tuning. Adaptive optimizers manage it by their own rules.'],
    'optimizer.weight_decay': ['约束权重增长；通常保留默认值，0 关闭。', 'Constrains weight growth; usually keep the default. 0 disables it.'],
    'optimizer.betas': scheduleFree
      ? ['分别控制权重平均与梯度大小估计，通常保留默认值。', 'Controls weight averaging and gradient-size estimation; usually keep the defaults.']
      : ['数值越大，反应越平缓；通常保留默认值。', 'Higher values react more smoothly; usually keep the defaults.'],
    'optimizer.grad_clip_norm': optimizerType === 'prodigy_plus_sf'
      ? ['外部梯度裁剪，默认 0 关闭；与 StableAdamW 内部更新保护不同。', 'External gradient clipping defaults to 0 (off); it differs from StableAdamW’s internal protection.']
      : optimizerType === 'automagic'
      ? ['外部梯度裁剪，默认 0 关闭；与优化器内部更新保护不同。', 'External gradient clipping defaults to 0 (off); it differs from internal update clipping.']
      : ['限制异常大梯度；通常保留 1，0 关闭。', 'Limits unusually large gradients; usually keep 1. 0 disables it.'],
    'optimizer.eps': optimizerType === 'prodigy_plus_sf'
      ? ['通常保留默认值；选择 Adam-atan2 才切换算法，需关闭 StableAdamW 和 FOCUS。', 'Usually keep the default. Selecting Adam-atan2 changes the algorithm; StableAdamW and FOCUS must be off.']
      : ['防止除以接近零的数；通常保留默认值。', 'Prevents division by values near zero; usually keep the default.'],
    'optimizer.beta3': ['留空时使用 β2 的平方根；通常保留自动。', optimizerEnglishHelp.beta3],
    'optimizer.growth_rate': ['每一步的最大增长倍率；留空时不设上限。', optimizerEnglishHelp.growth_rate],
    'optimizer.kahan': ['补偿低精度更新中容易丢失的小数值。', 'Compensates for small values lost during low-precision updates.'],
    'optimizer.group_lr': ['留空时统一使用上方学习率。', 'Leave empty to use the learning rate above for every group.'],
  };
  return hints[path]?.[english ? 1 : 0];
}

export function configPresetLabel(name: string, description: string, defaultPreset?: string, english = false) {
  const names: Record<string, [string, string]> = {
    'attn-mlp': ['常规范围', 'Standard scope'],
    'attn-only': ['精简范围', 'Reduced scope'],
    'full-linear': ['全部线性层', 'All linear layers'],
    'all-linear': ['全部线性层', 'All linear layers'],
    'with-adapter': ['常规范围＋文字适配层', 'Standard scope + text adapter'],
    'attn-mlp-text': ['常规范围＋文字融合层', 'Standard scope + text fusion'],
    'adapter-only': ['仅文字适配层', 'Text adapter only'],
  };
  const label = names[name]?.[english ? 1 : 0] || description || name;
  return name === defaultPreset ? `${label}${english ? ' (default)' : '（默认）'}` : label;
}

export function configOptionLabel(path: string, option: string, english = false) {
  const options: Record<string, Record<string, [string, string]>> = {
    'objective.weighting': { min_snr: ['Min-SNR', 'Min-SNR'] },
    'loop.mixed_precision': {bf16:['BF16 · 自动混合精度','BF16 · automatic mixed precision'],fp16:['FP16 · 自动混合精度','FP16 · automatic mixed precision'],no:['关闭自动混合精度','Automatic mixed precision off']},
    'loop.distributed_strategy': {ddp:['数据并行','Data parallelism'],fsdp:['显存分片（大模型）','Memory sharding (large models)']},
    'training.mode': {adapter:['LoRA','LoRA'],full:['全量微调','Full fine-tuning']},
    'memory.base_precision': {auto:['不转换（沿用加载精度）','No conversion (keep loaded precision)'],fp32:['FP32 · 32 位','FP32 · 32-bit'],bf16:['BF16 · 16 位','BF16 · 16-bit'],fp16:['FP16 · 16 位','FP16 · 16-bit'],fp8_e4m3:['FP8 E4M3 · 启动时量化','FP8 E4M3 · quantize at startup'],fp8_e5m2:['FP8 E5M2 · 启动时量化','FP8 E5M2 · quantize at startup']},
    'model.attention': {auto:['PyTorch SDPA（默认）','PyTorch SDPA (default)'],sdpa:['PyTorch SDPA','PyTorch SDPA'],xformers:['xFormers','xFormers'],flash_attn:['FlashAttention 2','FlashAttention 2'],metal_flash:['Metal FlashAttention · Apple','Metal FlashAttention · Apple'],sage:['SageAttention · 仅采样','SageAttention · sampling only']},
    'adapter.init': {default:['默认初始化','Default initialization'],scalar:['随机权重 + 零值缩放','Random weights + zero scale']},
    'adapter.mode': {auto:['自动 · 分开计算','Automatic · separate computation'],bypass:['分开计算适配器','Compute adapter separately'],weight:['合并权重后计算','Compute merged weights']},
    'memory.activation_checkpointing': {none:['关闭','Off'],block:['逐块重算 · 节省显存','Block recomputation · save memory'],unsloth:['重算并卸载中间输入','Recompute and offload block inputs']},
    'model.prediction_type': { epsilon: ['ε 预测（常规模型）', 'Epsilon (standard)'], v_prediction: ['v 预测', 'v-prediction'] },
    'model.sdxl_max_token_length': { '75': ['75 tokens · 默认', '75 tokens · default'], '150': ['150 tokens · 2 段', '150 tokens · 2 chunks'], '225': ['225 tokens · 3 段', '225 tokens · 3 chunks'] },
    'model.dtype': {auto:['跟随模型','Follow model'],bf16:['BF16','BF16'],fp16:['FP16','FP16'],fp32:['FP32','FP32']},
    'model.krea2_variant': {auto:['读取模型记录','Read model record'],raw:['Raw · 训练模型','Raw · training model'],turbo:['Turbo · 仅采样','Turbo · sampling only']},
    'model.flux2_variant': { auto: ['自动读取模型配置', 'Read model configuration'], dev: ['FLUX.2 dev（已停用）', 'FLUX.2 dev (retired)'], 'klein-base-4b': ['Klein 基础版 4B', 'Klein base 4B'], 'klein-base-9b': ['Klein 基础版 9B', 'Klein base 9B'] },
    'sampling.sampler': { euler:['Euler', 'Euler'], heun:['Heun', 'Heun'], er_sde:['ER-SDE', 'ER-SDE'] },
    'sampling.scheduler': { uniform:['Uniform', 'Uniform'], simple:['Simple', 'Simple'], sgm_uniform:['SGM Uniform', 'SGM Uniform'], normal:['Normal', 'Normal'] },
    'optimizer.type': {adamw:['AdamW','AdamW'],adam:['Adam','Adam'],sgd:['SGD','SGD'],adamw8bit:['AdamW 8-bit','AdamW 8-bit'],lion:['Lion','Lion'],lion8bit:['Lion 8-bit','Lion 8-bit'],prodigy:['Prodigy','Prodigy'],prodigy_plus_sf:['Prodigy Plus Schedule-Free','Prodigy Plus Schedule-Free'],automagic:['Automagic','Automagic'],adafactor:['Adafactor','Adafactor'],came:['CAME','CAME'],adamw_sf:['AdamW Schedule-Free','AdamW Schedule-Free']},
    'dataset.resolution_mode': { bucket: ['分桶 · 统一基准面积', 'Buckets · target area'], native: ['原生 · 每图独立尺寸', 'Native · individual image sizes'] },
    'dataset.text_encoding': {auto:['自动','Automatic'],online:['每步在线编码','Encode each step'],cached:['预编码缓存','Cached embeddings']},
    'dataset.image_fit': { pad: ['保留完整画面', 'Preserve the whole image'], crop: ['裁切填满（旧模式）', 'Crop to fill (legacy)'] },
    'dataset.native_overflow': { downscale: ['等比缩小到预算内', 'Downscale to fit budget'], error: ['报错并停止', 'Stop with an error'] },
  };
  return options[path]?.[option]?.[english ? 1 : 0] || option;
}

export type ConfigTab = 'train' | 'data' | 'model' | 'advanced';
export const CONFIG_TAB_GROUPS: Record<ConfigTab, string[]> = {
  train: ['training', 'loop', 'adapter', 'optimizer', 'scheduler', 'memory'],
  data: ['dataset', 'caption'],
  model: ['model', 'checkpoint'],
  advanced: ['objective', 'sampling', 'validation', 'logging'],
};
export function configTabForPath(path: string): ConfigTab {
  if (path === 'model.attention') return 'train';
  if (path === 'dataset.batch_size') return 'train';
  const group = path.split('.')[0];
  return (Object.keys(CONFIG_TAB_GROUPS) as ConfigTab[]).find(tab => CONFIG_TAB_GROUPS[tab].includes(group)) || 'advanced';
}

export type ConfigIssue = { path: string; label: string; message: string; detail: string; tab: ConfigTab };
/** Stands in for a reason Studio cannot phrase in Chinese; only usable where the raw detail can be expanded. */
export const OPAQUE_CONFIG_ISSUE = '此配置未通过检查，展开详情查看具体原因';
export function presentConfigIssues(errors: Array<{loc?: unknown; msg?: unknown}>, english = false): ConfigIssue[] {
  const issues = errors.map(error => {
    const detail = String(error.msg || '').replace(/^Value error, /, '');
    const location = (Array.isArray(error.loc) ? error.loc.map(String).join('.') : String(error.loc || '')).replace(/^body\./, '').replace(/^config\./, '');
    const embedded = detail.match(/\b(model\.[a-z_]+)\b/)?.[1];
    const path = embedded && (!location || location === 'model') ? embedded : location;
    const label = configFieldLabel(path, path || (english ? 'Configuration' : '配置'), english);
    let message = detail;
    if (!english) {
      if (/is required for|field required/i.test(detail)) message = `请填写或选择${label}`;
      else if (/at least one training dataset source is required/i.test(detail)) message = '请先添加训练图片或导入已有数据集';
      else if (/not found|does not exist|file is missing/i.test(detail)) message = '找不到指定文件，请检查训练机上的路径';
      else if (/outside allowed/i.test(detail)) message = '该路径不在允许使用的目录内';
      else if (/greater than or equal to/i.test(detail)) message = `输入值应大于或等于 ${detail.split(/greater than or equal to/i)[1].trim()}`;
      else if (/greater than/i.test(detail)) message = `输入值应大于 ${detail.split(/greater than/i)[1].trim()}`;
      else if (/less than or equal to/i.test(detail)) message = `输入值应小于或等于 ${detail.split(/less than or equal to/i)[1].trim()}`;
      else if (/requires? CUDA/i.test(detail)) message = '此选项需要 CUDA，请选择当前设备支持的配置';
      else if (/Extra inputs are not permitted/i.test(detail)) message = '当前版本不支持此参数，请检查导入的配置';
      else if (/valid (integer|number)/i.test(detail)) message = '请输入有效数字';
      else if (/no (images|training images)|dataset is empty/i.test(detail)) message = '数据源中没有可用的训练图片';
      else if (/unknown preset/i.test(detail)) message = '当前模型不支持这个训练范围，请重新选择';
      else if (Array.from(detail).every(character => character.charCodeAt(0) < 128)) message = OPAQUE_CONFIG_ISSUE;
    }
    return {path, label, message, detail, tab: configTabForPath(path)};
  });
  return issues.filter((issue, index) => issues.findIndex(item => item.path === issue.path && item.detail === issue.detail) === index);
}

export function presentPlanWarning(code: string, fallback: string, english = false) {
  if (english) return fallback;
  if (code === 'images.padding') {
    const count = fallback.match(/^(\d+) images/)?.[1];
    const share = fallback.match(/\(([\d.]+%) of training/)?.[1];
    return `${count ? `${count} 张` : '部分'}图片通过填充保留完整画面${share ? `，填充占训练画布的 ${share}` : ''}。填充区不直接计入损失，但模型仍能看到；原生尺寸或更宽的长宽比分桶可减少填充。`;
  }
  if (code === 'vram.tight') {
    const amounts = fallback.match(/estimated peak ([\d.]+) MB vs ([\d.]+) MB available/);
    return amounts ? `预计峰值显存 ${amounts[1]} MB，设备可用容量 ${amounts[2]} MB，显存余量可能不足。` : '预计显存余量可能不足，请调整分辨率、批量或显存设置。';
  }
  return ({
    'device.mps_fp32': 'Apple GPU 使用 FP32 训练，不启用混合精度。',
    'captions.missing': '部分图片没有标签，可到数据集补充标签或设置类别提示词。',
    'buckets.small': '部分分桶不足一个完整批次，最后一个批次会使用较少的图片。',
    'native.execution': '不同尺寸按像素预算分组前向，按图片数累积梯度；像素上限并非整体显存保证。',
  } as Record<string, string>)[code] || fallback;
}
