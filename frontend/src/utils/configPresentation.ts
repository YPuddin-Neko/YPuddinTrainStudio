import { FIELD_HELP, FIELD_HINTS } from './fieldCopy';
import { describeValidation, type ValidationIssue } from './validationMessages';


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
  'dataset.resolution_mode': '分辨率模式', 'dataset.image_fit': '图片适配方式', 'dataset.crop_anchor': '裁切保留位置', 'dataset.native_max_pixels': '图像面积上限（等效边长 px）', 'dataset.native_max_pixels_mode': '图像面积上限模式',
  'dataset.native_max_side': '最长边上限（px）', 'dataset.native_overflow': '超出尺寸上限时',
  'dataset.area_tolerance': '面积容差', 'dataset.bucket_step': '分桶步长', 'dataset.bucket_no_upscale': '不放大小图',
  'dataset.batch_size': '批大小', 'dataset.flip': '随机水平翻转', 'dataset.masked_loss': '遮罩加权训练',
  'dataset.num_workers': '数据加载线程', 'dataset.cache_dir': '缓存目录', 'dataset.cache_latents': '训练图像缓存',
  'dataset.text_encoding': '标签处理方式', 'dataset.caption.prefix': '标签前缀', 'dataset.caption.suffix': '标签后缀',
  'dataset.caption.trigger_word': '触发词', 'dataset.caption.keep_tokens': '保留前几个标签', 'dataset.caption.shuffle': '打乱标签顺序',
  'dataset.caption.tag_dropout': '单个标签丢弃率', 'dataset.caption.caption_dropout': '整条标签丢弃率',
  'dataset.caption.separator': '标签分隔符', 'dataset.caption.wildcard': '启用通配符',
  'dataset.caption.cache_variants': '预缓存标签变体数',
  'sampling.scheduler': '采样调度器', 'sampling.er_sde_order': 'ER-SDE 求解阶数', 'sampling.er_sde_s_noise': 'ER-SDE 随机噪声强度',
  'adapter.algo': '训练算法', 'adapter.rank': 'Rank / 秩', 'adapter.alpha': 'Alpha / 缩放', 'adapter.factor': 'LoKr 分解因子',
  'adapter.tlora_min_rank': '最小秩', 'adapter.tlora_power': '秩变化曲线', 'adapter.tlora_ortho': '正交初始化',
  'adapter.decompose_both': '双矩阵低秩分解', 'adapter.rs_lora': 'Rank 稳定缩放', 'adapter.dora': '启用 DoRA', 'adapter.dora_axis': 'DoRA 计算方向',
  'adapter.dora_compute_mode': 'DoRA 计算模式', 'adapter.dora_merge_dtype': 'DoRA 融合精度',
  'adapter.init': '初始化方式', 'adapter.dropout': '输出丢弃率', 'adapter.rank_dropout': '秩丢弃率',
  'adapter.module_dropout': '模块丢弃率', 'adapter.preset': '训练层范围', 'adapter.rules': '逐层覆盖规则',
  'adapter.layer_types': '训练层类型', 'adapter.conv_rank': '卷积层 Rank', 'adapter.conv_alpha': '卷积层 Alpha',
  'adapter.mode': '权重计算方式', 'adapter.param_dtype': '可训练参数精度', 'adapter.lr_scale': '学习率缩放',
  'adapter.resume_weights': '继续训练的权重', 'objective.timestep_sampling': '时间步采样',
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
  'memory.activation_checkpointing': '梯度检查点', 'memory.vae_attention_chunking': 'VAE 注意力分块', 'memory.no_half_vae': 'VAE 保持 FP32', 'memory.vae_tiling': 'VAE 分块', 'memory.cache_encode_tiled': '缓存编码分块', 'memory.vae_encode_batch_size': 'VAE 编码批量', 'memory.offload_text_encoder': '卸载文本编码器',
  'memory.compile': '编译模型', 'memory.allow_tf32': '允许 TF32', 'loop.max_steps': '最大训练步数',
  'loop.epochs': '训练轮数', 'loop.grad_accum': '梯度累积', 'loop.mixed_precision': '混合精度', 'loop.seed': '随机种子',
  'loop.gpu_count': '训练显卡数量',
  'loop.distributed_strategy': '多卡训练方式',
  'loop.deterministic': '可复现训练',
  'loop.ema': '启用 EMA', 'loop.ema_decay': 'EMA 衰减', 'loop.nan_skip_limit': '无效梯度跳过上限', 'loop.log_every': '日志间隔',
  'checkpoint.output_dir': '训练权重保存位置', 'checkpoint.name': '权重文件名', 'checkpoint.save_every_steps': '每隔几步保存',
  'checkpoint.save_every_epochs': '每隔几轮保存', 'checkpoint.save_state_every_steps': '恢复点保存间隔', 'checkpoint.save_state_every_epochs': '每隔几轮保存恢复点',
  'checkpoint.keep_last_n': '保留最近几次权重', 'checkpoint.save_dtype': '权重保存精度', 'checkpoint.save_on_finish': '结束时保存权重',
  'checkpoint.save_training_metadata': '保存训练元数据', 'checkpoint.state_dir': '恢复点保存目录', 'checkpoint.resume': '恢复完整训练状态', 'sampling.output_dir': '采样图保存目录', 'sampling.enabled': '生成训练预览', 'sampling.every_steps': '每隔几步预览',
  'sampling.every_epochs': '每隔几轮预览', 'sampling.at_start': '开始前生成预览', 'sampling.prompts': '预览提示词',
  'sampling.prompts_file': '提示词文件', 'sampling.steps': '采样步数', 'sampling.cfg': 'CFG 引导强度',
  'sampling.shift': '采样时间步偏移', 'sampling.width': '预览宽度', 'sampling.height': '预览高度',
  'sampling.seed': '预览种子', 'sampling.noise': '采样兼容模式', 'sampling.adapter_merge_dtype': '适配器融合精度', 'sampling.sampler': '采样器', 'validation.enabled': '启用验证集',
  'validation.split_ratio': '验证集划分比例', 'validation.sources': '独立验证数据源', 'validation.every_steps': '每隔几步验证',
  'validation.every_epochs': '每隔几轮验证', 'validation.timesteps': '验证时间步', 'validation.max_images': '验证图片上限',
  'validation.seed': '验证种子', 'logging.tensorboard': 'TensorBoard 日志', 'logging.wandb': 'Weights & Biases',
  'logging.events_path': '训练事件文件', 'logging.output_dir': '日志保存目录', 'logging.level': '日志级别',
};

export function configFieldLabel(path: string, fallback: string, english = false) {
  if (english && path === 'logging.output_dir') return 'Log directory';
  if (english && path === 'checkpoint.state_dir') return 'Recovery point directory';
  if (english && path === 'sampling.output_dir') return 'Sample image directory';
  if (english && path === 'dataset.crop_anchor') return 'Crop anchor';
  if (english && path === 'adapter.resume_weights') return 'Weights to continue training';
  if (english && path === 'adapter.dora_axis') return 'DoRA axis';
  if (english && path === 'adapter.dora_compute_mode') return 'DoRA compute mode';
  if (english && path === 'adapter.dora_merge_dtype') return 'DoRA merge precision';
  if (english && path === 'sampling.noise') return 'Sampling compatibility';
  if (english && path === 'memory.vae_tiling') return 'VAE tiling';
  if (english && path === 'memory.cache_encode_tiled') return 'Tiled cache encoding';
  if (english && path === 'memory.no_half_vae') return 'Keep VAE in FP32';
  if (english && path === 'memory.vae_encode_batch_size') return 'VAE encode batch';
  if (english && path === 'sampling.adapter_merge_dtype') return 'Adapter merge precision';
  if (english && path === 'adapter.layer_types') return 'Layer types';
  if (english && path === 'adapter.conv_rank') return 'Convolution rank';
  if (english && path === 'adapter.conv_alpha') return 'Convolution alpha';
  if (english && path.startsWith('adapter.tlora_')) return ({ 'adapter.tlora_min_rank': 'Minimum rank', 'adapter.tlora_power': 'Rank curve', 'adapter.tlora_ortho': 'Orthogonal start' } as Record<string, string>)[path] || fallback;
  if (english && path === 'checkpoint.save_training_metadata') return 'Save training metadata';
  if (english && path === 'checkpoint.save_state_every_steps') return 'Recovery save interval';
  if (english && path === 'checkpoint.save_state_every_epochs') return 'Recovery save interval (epochs)';
  if (english && path.startsWith('dataset.native_')) return ({'dataset.native_max_pixels':'Image area limit (equivalent side, px)','dataset.native_max_pixels_mode':'Image area limit mode','dataset.native_max_side':'Longest side limit (px)','dataset.native_overflow':'When a size limit is exceeded'} as Record<string,string>)[path] || fallback;
  if (english && path === 'model.sdxl_max_token_length') return 'SDXL caption length';
  if (english && path === 'memory') return 'Memory estimate';
  if (english && path === 'memory.vae_attention_chunking') return 'VAE attention chunking';
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

export function schedulerTypeHelp(english = false, options?: string[]) {
  const descriptions: Record<string, [string, string]> = {
    constant: ['始终保持设定的学习率。', 'keeps the set learning rate.'],
    linear: ['从设定值匀速降到最低学习率比例。', 'falls at a steady rate to the minimum ratio.'],
    cosine: ['沿余弦曲线降低，开头和结尾变化慢；默认。', 'falls along a cosine curve, slowly at the start and the end; the default.'],
    cosine_restarts: ['按周期数重复余弦下降，每个周期开始时回到设定值。', 'repeats the cosine fall for the set number of cycles, returning to the set rate at each start.'],
    polynomial: ['按多项式的幂降低，幂为 1 时与线性相同。', 'falls along a polynomial curve; a power of 1 equals linear.'],
    warmup_stable_decay: ['预热后保持设定值，最后一段再降低。', 'holds the set rate after warm-up and falls only in the final stretch.'],
    rex: ['预热后按反射指数曲线降低，前期较慢、后期加快，最终降到最低学习率比例。', 'Reflected Exponential decay after warm-up: slower at first and faster near the end, reaching the minimum learning-rate ratio.'],
  };
  return [
    english ? 'How the learning rate changes over the run:' : '学习率随训练步数变化的曲线：',
    ...Object.entries(descriptions).filter(([option]) => !options || options.includes(option))
      .map(([option, description]) => `${configOptionLabel('scheduler.type', option, english)}${english ? ': ' : '：'}${description[english ? 1 : 0]}`),
    english ? 'Schedule-free optimizers always use Constant.' : '免调度优化器固定使用恒定。',
  ].join('\n');
}

/** Explain what a choice changes before introducing the implementation term. */
export function configFieldHelp(path: string, fallback: string | undefined, english = false, optimizerType?: string, scheduleFree = false) {
  if (path === 'scheduler.type') return schedulerTypeHelp(english);
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
    'dataset.native_max_pixels': ['分辨率优先：按训练图片计算，尽量保留原图尺寸。\n显存优先：按所选显卡和训练参数调整面积上限。\n自定义：填写等效边长，面积上限为其平方。最长边限制和模型对齐同时生效。', 'Resolution first calculates the limit from training images to preserve their original size.\nVRAM first adjusts the limit for the selected GPUs and training settings.\nCustom uses the square of the entered equivalent side length. Longest-side limits and model alignment also apply.'],
    'dataset.native_max_side': ['填宽或高允许达到的最大长度，单位为像素。例如 4096 表示宽、高都不得超过 4096。\n面积上限和最长边上限必须同时满足，以先触及的限制为准。面积填 1024、最长边填 4096 时，2048×2048 的图片仍会因面积超限而缩小。\n想保留原图尺寸，两个上限都需要容纳原图及模型对齐补边。', 'Enter the maximum allowed width or height in pixels. 4096 means neither dimension may exceed 4096.\nBoth the area and longest-side limits must be satisfied; the tighter limit determines the size. An area setting of 1024 still downscales a 2048×2048 image even if the longest-side limit is 4096.\nTo retain the original size, both limits must accommodate the image and any model-alignment padding.'],
    'checkpoint.save_state_every_steps': ['Step：按参数更新次数保存，100 Step 为每 100 步保存。\nEpoch：按完整训练轮数保存，2 Epoch 为每完成 2 轮保存。轮中达到最大步数不算完成一轮。\n默认每 100 Step 保存；关闭开关可停用定期保存。暂停时仍会保存当前恢复点；意外退出只能从最近一次成功保存的位置继续。', 'Step: saves by optimizer updates; 100 Step saves every 100 updates.\nEpoch: saves by completed dataset passes; 2 Epoch saves after every two complete epochs. Reaching the step limit partway through an epoch does not complete it.\nDefaults to 100 Step; turn off the switch to disable periodic saving. Pausing still saves a recovery point; crashes can only recover the last successful save.'],
    'dataset.crop_anchor': ['选择裁切后要保留的位置。上中贴住顶部，多余部分从下方裁掉，可避免居中裁切削去头部；左右位置同理。仅裁掉超出训练尺寸的部分，图片与遮罩保持对齐。分桶和原生尺寸模式均适用，保留完整画面时不使用此设置。', 'Choose the part of the image to retain. Top center keeps the top edge and removes excess from the bottom, helping retain heads; left and right work similarly. Only the area outside the training dimensions is removed, and masks stay aligned. Applies to bucket and native cropping; unused when preserving the whole image.'],
    'dataset.resolution_mode': ['Bucket：将图片按长宽比分组，使用下方训练分辨率设定目标面积。\nNative：按原图尺寸训练，并对齐模型要求的尺寸倍数；超出面积或单边上限时，按所选方式等比缩小或报错。', 'Bucket: groups images by aspect ratio at the target areas set below.\nNative: uses original dimensions aligned to the model’s required multiples; images exceeding the area or side limit are scaled down or rejected according to the overflow setting.'],
    'dataset.image_fit': ['保留完整画面：等比缩放后补边，补边区域不计入直接损失，但仍作为模型输入。\n裁切填满尺寸：等比缩放至填满后，按裁切保留位置裁剪。\n新项目默认保留完整画面；旧配置沿用原来的裁切设置。', 'Pad: scales proportionally and adds padding. Padded areas are excluded from direct loss but remain part of the model input.\nCrop: scales proportionally to fill the target and crops at the selected anchor.\nNew projects default to Pad; existing configurations keep their crop setting.'],
    'dataset.native_overflow': ['等比缩小到上限内：图片超过面积或单边上限时等比缩小。\n报错并停止：图片超过上限时停止并提示调整。', 'Downscale to fit limits: scales images down proportionally when they exceed the area or side limit.\nStop with an error: stops and requests an adjustment when an image exceeds a limit.'],
    'training.mode': ['LoRA：生成 LoRA / LoKr 等附加权重。\n全量微调：直接更新所选组件的原始参数，保存模型组件。', 'LoRA: trains additional weights such as LoRA / LoKr.\nFull fine-tuning: updates the original parameters of the selected components and saves model components.'],
    'objective.loss': ['MSE：平方误差，默认选项。\nHuber / pseudo-Huber：调整大误差的惩罚方式，更换后损失数值不能直接与 MSE 比较。', 'MSE: squared error, the default.\nHuber / pseudo-Huber: changes the penalty for large errors; loss values cannot be compared directly with MSE.'],
    'checkpoint.save_training_metadata': ['默认关闭，只写入出图软件识别底模的键、网络结构及继续训练所需的 DoRA 计算设置。开启后额外写入标题、步数、轮数、学习率、优化器、训练尺寸等元数据，适用于 LoRA、LoKr 及其 EMA 权重。不写入本机目录、图片标签、提示词或访问密钥。完整断点恢复仍需恢复点。', 'Off by default; writes only the keys image tools use to recognize the base model, the network structure and the DoRA compute settings needed to continue training. When enabled, also writes the title, steps, epoch, learning rate, optimizer, training dimensions and other metadata in LoRA and LoKr exports, including EMA weights. Local directories, image captions, prompts and access tokens are excluded. Resuming the full training state still requires a recovery point.'],
    'dataset.resolutions': ['单个分辨率填 1024；多个用逗号或空格分隔，如 1024, 1536。填写正整数边长，不写 1024×1024。1024 表示每桶约 1024×1024 像素；每张图会在每个基准分辨率各训练一次，增加总样本和步数。', 'Enter one size as 1024, or separate multiple sizes with commas or spaces, e.g. 1024, 1536. Use positive integer side lengths, not 1024×1024. A base of 1024 gives roughly 1024×1024 pixels per bucket. Each image trains at every base resolution, increasing samples and steps.'],
    'adapter.resume_weights': ['训练结束后仍想继续优化时，可加载上次导出的 LoRA / LoKr 权重，再设置本次新增的训练轮数或步数，也可调整学习率和数据。底模、算法和权重结构需匹配。优化器和步数重新开始；中断后原样继续请使用完整恢复点。', 'To keep improving a finished run, load its exported LoRA / LoKr weights and set the additional epochs or steps for this new run. Learning rate and data may be changed. The base model, algorithm and weight structure must match. Optimizer state and counters restart; use a full recovery point for an interrupted run.'],
    'loop.deterministic': ['默认关闭。在相同配置、设备和软件环境下提高重复训练的一致性。开启后可能固定部分计算精度和注意力设置，增加显存与耗时；具体值会显示在对应字段。完整续训需保持原设置和环境。', 'Off by default. Improves repeatability with the same configuration, device and software environment. May manage precision and attention settings and increase memory use and runtime; effective values appear in the fields. Keep the same settings and environment when resuming.'],
    'loop.mixed_precision': ['控制训练运算的自动混合精度。关闭只停用自动混合精度，不改变权重本身的精度。底模存储和导出文件的精度分别设置；可复现训练可能调整实际计算精度。', 'Controls automatic mixed precision during training. Disabling it does not change weight precision. Base-weight storage and export precision are configured separately; reproducible training may adjust compute precision.'],
    'memory.base_precision': ['适配器训练时转换其覆盖的未量化冻结线性层，降低存储精度可节省显存，但可能影响训练质量；原模型文件和适配器精度不变。FP8 需要受支持的 CUDA 或海光环境。全量微调使用沿用精度或 FP32。', 'Converts unquantized frozen linear layers covered by adapters. Lower storage precision saves memory but may affect training quality; source files and adapter precision are unchanged. FP8 requires supported CUDA or DTK hardware. Full fine-tuning uses the loaded precision or FP32.'],
    'memory.activation_checkpointing': ['开启后每个模块只保留输入，反向传播时逐块重新计算，显存大幅减少，训练会慢一些（多一次前向计算）。\n开启并卸载到内存：再把这些输入暂存到内存，显存再少一些，但占用内存并增加传输；开启后显存仍不够时再用。\n可与梯度累积同时使用。', 'On keeps only each block\'s input and recomputes blocks during backward: much less memory, somewhat slower (one extra forward pass).\nOn + offload also parks those inputs in system memory for a little more saving, at the cost of RAM and transfers; use it when On is not enough.\nWorks with gradient accumulation.'],
    'loop.grad_accum': ['累计指定数量的批次后再更新一次参数。等效批次 = 批大小 × 梯度累积 × 显卡数。', 'Updates parameters after the specified number of minibatches. Effective batch size = per-GPU batch size × accumulation steps × GPU count.'],
    'loop.gpu_count': ['1 为单卡；多卡任务会等待所需显卡全部空闲后启动。批大小按每张卡计算；可用训练方式取决于当前平台。', '1 uses one GPU. Multi-GPU jobs wait for all required devices to be free. Batch size is per GPU; available strategies depend on the platform.'],
    'loop.distributed_strategy': ['数据并行（DDP）：每张卡保留完整模型，并行处理不同数据，通常比单卡每秒处理更多图片；每卡仍需容纳完整模型，加速幅度取决于卡间通信和负载。\n显存分片（FSDP）：同样并行处理数据，将参数、梯度和优化器状态分摊到多卡以节省显存。相比 DDP 增加参数通信，同等卡数和批量下通常更慢，具体取决于模型和卡间带宽；显存够用时优先选 DDP。需要至少两张 Linux CUDA/DTK 显卡，并冻结文本编码器。', 'Data parallel (DDP): each GPU holds the whole model and processes different data in parallel, usually handling more images per second than one GPU. Each GPU must still fit the whole model; speedup depends on communication and workload.\nSharded (FSDP): also processes data in parallel, while distributing parameters, gradients and optimizer states across GPUs to save memory. Extra parameter communication usually makes it slower than DDP with the same GPU count and batch size, depending on the model and interconnect bandwidth. Prefer DDP when memory is sufficient. Requires at least two Linux CUDA/DTK GPUs and frozen text encoders.'],
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
  return help[path]?.[english ? 1 : 0] || (english && path.startsWith('optimizer.') ? optimizerEnglishHelp[path.slice(10)] : undefined) || FIELD_HELP[path]?.[english ? 1 : 0] || fallback;
}

export function configFieldHint(path: string, english = false, optimizerType?: string, scheduleFree = false, dataset?: {resolution_mode?:string;native_overflow?:string;crop_anchor?:string}) {
  const cropHints: Record<string,[string,string]> = {
    top_left:['保留左上方，从右侧和下方裁掉多余部分。','Keep the top left; trim excess from the right and bottom.'],
    top:['保留顶部，从下方裁掉多余部分。','Keep the top; trim excess from the bottom.'],
    top_right:['保留右上方，从左侧和下方裁掉多余部分。','Keep the top right; trim excess from the left and bottom.'],
    left:['保留左侧，从右侧裁掉多余部分。','Keep the left edge; trim excess from the right.'],
    center:['从四周均匀裁切，保留画面中央。','Trim evenly around the edges, keeping the center.'],
    right:['保留右侧，从左侧裁掉多余部分。','Keep the right edge; trim excess from the left.'],
    bottom_left:['保留左下方，从右侧和上方裁掉多余部分。','Keep the bottom left; trim excess from the right and top.'],
    bottom:['保留底部，从上方裁掉多余部分。','Keep the bottom; trim excess from the top.'],
    bottom_right:['保留右下方，从左侧和上方裁掉多余部分。','Keep the bottom right; trim excess from the left and top.'],
  };
  const dynamic: Record<string, [string, string]> = {
    'dataset.crop_anchor': cropHints[dataset?.crop_anchor || 'center'],
    'dataset.resolution_mode': dataset?.resolution_mode === 'native'
      ? dataset.native_overflow === 'error'
        ? ['使用原图尺寸；超过下方上限时停止并报错。', 'Uses original image sizes; stops with an error if a limit below is exceeded.']
        : ['使用原图尺寸；超过下方上限时等比缩小。', 'Uses original image sizes; scales down proportionally if a limit below is exceeded.']
      : ['按训练分辨率设定目标尺寸，图片按长宽比分组。', 'Groups images by aspect ratio at the configured training resolutions.'],
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
  };
  return (dynamic[path] || FIELD_HINTS[path])?.[english ? 1 : 0];
}

export function configPresetLabel(name: string, description: string, defaultPreset?: string, english = false) {
  const names: Record<string, [string, string]> = {
    'attn-mlp': ['常规范围', 'Standard scope'],
    'attn-only': ['精简范围', 'Reduced scope'],
    'full-linear': ['主模块全部线性层', 'All block linear layers'],
    'all-linear': ['全部线性层', 'All linear layers'],
    'all-layers': ['全部层', 'All layers'],
    'with-adapter': ['常规范围＋文字适配层', 'Standard scope + text adapter'],
    'attn-mlp-text': ['常规范围＋文字融合层', 'Standard scope + text fusion'],
    'adapter-only': ['仅文字适配层', 'Text adapter only'],
  };
  const pair = names[name];
  const label = pair ? (english ? pair[1] : `${pair[1]} (${pair[0]})`) : description || name;
  return name === defaultPreset ? `${label}${english ? ' (default)' : '（默认）'}` : label;
}

export function configOptionLabel(path: string, option: string, english = false) {
  const options: Record<string, Record<string, [string, string]>> = {
    'objective.weighting': { none: ['不加权', 'None'], sigma_sqrt: ['噪声尺度平方根', 'Sigma square root'], cosmap: ['余弦映射', 'CosMap'], snr_like: ['类信噪比加权', 'SNR-like'], cosmos: ['Cosmos', 'Cosmos'], min_snr: ['最小信噪比加权', 'Min-SNR'] },
    'objective.timestep_sampling': { uniform: ['均匀采样', 'Uniform'], logit_normal: ['逻辑正态分布', 'Logit-Normal'], shift: ['偏移采样', 'Shift'], resolution_shift: ['按分辨率偏移', 'Resolution shift'], mode: ['模式分布', 'Mode'], cosmap: ['余弦映射', 'CosMap'] },
    'objective.loss': { mse: ['均方误差', 'MSE'], huber: ['平滑绝对误差', 'Huber'], pseudo_huber: ['伪 Huber 损失', 'Pseudo-Huber'] },
    'adapter.algo': { lora: ['LoRA', 'LoRA'], lokr: ['LoKr', 'LoKr'], loha: ['LoHa', 'LoHa'], ortho: ['OrthoLoRA', 'OrthoLoRA'], tlora: ['T-LoRA', 'T-LoRA'], full: ['LyCORIS Full', 'LyCORIS Full'] },
    'adapter.param_dtype': { fp32: ['FP32', 'FP32'], bf16: ['BF16', 'BF16'] },
    'checkpoint.save_dtype': { bf16: ['BF16', 'BF16'], fp16: ['FP16', 'FP16'], fp32: ['FP32', 'FP32'] },
    'scheduler.type': { constant: ['恒定', 'Constant'], linear: ['线性衰减', 'Linear'], cosine: ['余弦衰减', 'Cosine'], cosine_restarts: ['余弦重启', 'Cosine restarts'], polynomial: ['多项式衰减', 'Polynomial'], warmup_stable_decay: ['预热-稳定-衰减', 'WSD'], rex: ['反射指数衰减', 'REX'] },
    'logging.level': { debug: ['记录调试信息', 'Debug'], info: ['常规进度', 'Info'], warning: ['仅警告和错误', 'Warning'] },
    'loop.mixed_precision': {bf16:['自动混合精度','BF16'],fp16:['自动混合精度','FP16'],no:['关闭自动混合精度','Off']},
    'loop.distributed_strategy': {ddp:['数据并行','DDP'],fsdp:['显存分片','FSDP']},
    'training.mode': {adapter:['LoRA','LoRA'],full:['全量微调','Full fine-tuning']},
    'memory.base_precision': {auto:['沿用加载精度','Auto'],fp32:['32 位','FP32'],bf16:['16 位','BF16'],fp16:['16 位','FP16'],fp8_e4m3:['启动时量化','FP8 E4M3'],fp8_e5m2:['启动时量化','FP8 E5M2']},
    'model.attention': {auto:['默认','PyTorch SDPA'],sdpa:['PyTorch SDPA','PyTorch SDPA'],xformers:['xFormers','xFormers'],flash_attn:['FlashAttention 2','FlashAttention 2'],metal_flash:['Metal FlashAttention · Apple','Metal FlashAttention · Apple'],sage:['仅采样','SageAttention']},
    'adapter.init': {default:['默认初始化','Default'],scalar:['随机权重，零值缩放','Scalar']},
    'adapter.dora_axis': {output:['按输出通道','Output'],input:['按输入通道','Input']},
    'adapter.dora_compute_mode': {standard:['标准模式','Standard mode'],comfyui:['兼容模式','Compatibility mode']},
    'adapter.dora_merge_dtype': {auto:['跟随底模','Auto'],bf16:['BF16','BF16'],fp16:['FP16','FP16'],fp32:['FP32','FP32']},
    'adapter.layer_types': {linear:['只训练线性层','Linear only'],linear_conv:['线性层和卷积层','Linear + convolution']},
    'adapter.mode': {auto:['自动','Automatic'],bypass:['分开计算','Bypass'],merged:['合并权重后计算','Merged']},
    'memory.activation_checkpointing': {none:['关闭','Off'],block:['开启','On'],unsloth:['开启并卸载到内存','On + offload']},
    'model.prediction_type': { epsilon: ['噪声预测', 'Epsilon'], v_prediction: ['速度预测', 'v-prediction'] },
    'model.sdxl_max_token_length': { '75': ['默认', '75 tokens'], '150': ['2 段', '150 tokens'], '225': ['3 段', '225 tokens'] },
    'model.dtype': {auto:['跟随模型','Follow model'],bf16:['BF16','BF16'],fp16:['FP16','FP16'],fp32:['FP32','FP32']},
    'model.krea2_variant': {auto:['读取模型记录','Read model record'],raw:['训练模型','Raw'],turbo:['仅采样','Turbo']},
    'model.flux2_variant': { auto: ['自动读取模型配置', 'Read model configuration'], dev: ['已停用', 'FLUX.2 dev'], 'klein-base-4b': ['Klein 基础版 4B', 'Klein base 4B'], 'klein-base-9b': ['Klein 基础版 9B', 'Klein base 9B'] },
    'sampling.sampler': { euler:['欧拉', 'Euler'], euler_ancestral:['欧拉祖先采样', 'Euler a'], heun:['二阶修正', 'Heun'], er_sde:['随机微分方程', 'ER-SDE'] },
    'sampling.adapter_merge_dtype': { auto:['自动（跟随底模）', 'Auto (follow base model)'], bf16:['BF16', 'BF16'], fp16:['FP16', 'FP16'], fp32:['FP32', 'FP32'] },
    'sampling.noise': { comfyui:['ComfyUI', 'ComfyUI'], a1111:['A1111 WebUI', 'A1111 WebUI'] },
    'sampling.scheduler': { uniform:['均匀', 'Uniform'], simple:['简单', 'Simple'], sgm_uniform:['SGM 均匀', 'SGM Uniform'], normal:['常规', 'Normal'] },
    'optimizer.type': {adamw:['AdamW','AdamW'],adam:['Adam','Adam'],sgd:['SGD','SGD'],adamw8bit:['AdamW 8-bit','AdamW 8-bit'],lion:['Lion','Lion'],lion8bit:['Lion 8-bit','Lion 8-bit'],prodigy:['Prodigy','Prodigy'],prodigy_plus_sf:['Prodigy Plus Schedule-Free','Prodigy Plus Schedule-Free'],automagic:['Automagic','Automagic'],adafactor:['Adafactor','Adafactor'],came:['CAME','CAME'],adamw_sf:['AdamW Schedule-Free','AdamW Schedule-Free']},
    'dataset.native_max_pixels_mode': {auto: ['自动分辨率优先', 'Auto, resolution first'], auto_vram: ['自动显存优先', 'Auto, VRAM first'], custom: ['自定义', 'Custom']},
    'dataset.resolution_mode': { bucket: ['按设定分辨率训练', 'Bucket'], native: ['按原图尺寸训练', 'Native'] },
    'dataset.text_encoding': {auto:['自动','Auto'],online:['每步编码文本','Online'],cached:['训练前缓存文本特征','Cached']},
    'dataset.crop_anchor': {top_left:['左上','Top left'],top:['上中','Top center'],top_right:['右上','Top right'],left:['左中','Middle left'],center:['居中','Center'],right:['右中','Middle right'],bottom_left:['左下','Bottom left'],bottom:['下中','Bottom center'],bottom_right:['右下','Bottom right']},
    'dataset.image_fit': { pad: ['保留完整画面', 'Pad'], crop: ['裁切填满', 'Crop'] },
    'dataset.native_overflow': { downscale: ['等比缩小到上限内', 'Downscale to fit limits'], error: ['报错并停止', 'Stop with an error'] },
  };
  const label = options[path]?.[option];
  if (!label) return option;
  if (path === 'adapter.dora_compute_mode') return label[english ? 1 : 0];
  if (english || label[0] === label[1]) return label[1];
  return `${label[1]} (${label[0]})`;
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

/**
 * Probabilities and fractions the parameter form shows as percentages, unlike EMA decay, timesteps and
 * warmup's mixed steps-or-ratio contract, which keep their native units.
 */
export const PERCENTAGE_FIELDS = ['adapter.dropout', 'adapter.rank_dropout', 'adapter.module_dropout', 'dataset.area_tolerance', 'dataset.caption.tag_dropout', 'dataset.caption.caption_dropout', 'scheduler.min_lr_ratio', 'validation.split_ratio'];

export type ConfigIssue = { path: string; label: string; message: string; detail: string; tab: ConfigTab; type?: string; ctx?: Record<string, unknown> };
/** Stands in for a reason Studio cannot phrase in Chinese; only usable where the raw detail can be expanded. */
export const OPAQUE_CONFIG_ISSUE = '此配置未通过检查，展开详情查看具体原因';
/**
 * Field, label and message for each failed setting. A value that failed validation is phrased from its kind and bound,
 * in the field's unit; a validator's own reason is translated from its wording.
 */
export function presentConfigIssues(errors: ValidationIssue[], english = false): ConfigIssue[] {
  const issues = errors.map(error => {
    const detail = String(error.msg || '').replace(/^Value error, /, '');
    const memory = detail.match(/^estimated peak memory ([\d.]+ GB) exceeds (.+) capacity ([\d.]+ GB)$/);
    const location = (Array.isArray(error.loc) ? error.loc.map(String).join('.') : String(error.loc || '')).replace(/^body\./, '').replace(/^config\./, '');
    const embedded = detail.match(/\b(model\.[a-z_]+)\b/)?.[1];
    const path = embedded && (!location || location === 'model') ? embedded : location;
    const label = configFieldLabel(path, path || (english ? 'Configuration' : '配置'), english);
    const type = typeof error.type === 'string' ? error.type : undefined;
    const ctx = error.ctx && typeof error.ctx === 'object' ? error.ctx as Record<string, unknown> : undefined;
    let message = describeValidation(error, { label, percent: PERCENTAGE_FIELDS.includes(path), english }) ?? detail;
    if (memory) message = english
      ? `Estimated peak memory ${memory[1]} exceeds ${memory[2]} (${memory[3]}), so training cannot start. Turn on gradient checkpointing, reduce the batch size or lower the training size.`
      : `预计显存峰值 ${memory[1]}，超过 ${memory[2]} 的 ${memory[3]} 容量，无法开始训练。开启梯度检查点、减小批大小或降低训练尺寸都能减少显存占用。`;
    else if (message === detail && !english) {
      if (/is required for/i.test(detail)) message = `请填写或选择${label}`;
      else if (/not found|does not exist|file is missing/i.test(detail)) message = '找不到指定文件，请检查训练机上的路径';
      else if (/outside allowed/i.test(detail)) message = '该路径不在允许使用的目录内';
      else if (/bucket step (\d+) must be a multiple of align (\d+)/.test(detail)) { const [, step, align] = detail.match(/bucket step (\d+) must be a multiple of align (\d+)/)!; message = `分桶步长 ${step} 不是 ${align} 的倍数；当前模型要求 ${align} 的倍数，如 ${Math.max(Number(align), Math.round(Number(step) / Number(align)) * Number(align))}`; }
      else if (/requires? CUDA/i.test(detail)) message = '此选项需要 CUDA，请选择当前设备支持的配置';
      else if (/marked v-prediction/i.test(detail)) message = '这个 SDXL 模型是 v 预测模型（文件带 v_pred 标记），请把 SDXL 预测方式改为 v_prediction';
      else if (/marked zero terminal SNR/i.test(detail)) message = '这个 SDXL 模型按零终点信噪比训练（文件带 ztsnr 标记），请开启零终点信噪比（Zero SNR）';
      else if (/no (images|training images)|dataset is empty/i.test(detail)) message = '数据源中没有可用的训练图片';
      else if (/unknown preset/i.test(detail)) message = '当前模型不支持这个训练范围，请重新选择';
      else if (Array.from(detail).every(character => character.charCodeAt(0) < 128)) message = OPAQUE_CONFIG_ISSUE;
    }
    return { path, label, message, detail, tab: configTabForPath(path), type, ctx };
  });
  return issues.filter((issue, index) => issues.findIndex(item => item.path === issue.path && item.detail === issue.detail) === index);
}

type ValueAdvice = { code: string; msg: string; loc?: string | null; step?: number | null; values?: number[] | null; suggestion?: number | number[] | null };
const ADVICE_FIELDS: Record<string, [string, string]> = {
  'dataset.resolutions': ['训练分辨率', 'Training resolution'],
  'dataset.native_max_side': ['最长边上限', 'Longest side limit'],
  'sampling.width': ['预览宽度', 'Preview width'],
  'sampling.height': ['预览高度', 'Preview height'],
};

/** A size the trainer rounds to the model's grid: `msg` names the field, `inline` sits under it; `fix` is the replacement. */
export function presentValueAdvice(warning: ValueAdvice, english = false): { loc: string; msg: string; inline: string; fix?: { label: string; value: number | number[] } } | null {
  if (!warning.loc || warning.step == null || !warning.values?.length || warning.suggestion == null) return null;
  const loc = warning.loc;
  const values = warning.values.join(', ');
  const suggestion = Array.isArray(warning.suggestion) ? warning.suggestion.join(', ') : String(warning.suggestion);
  const prompt = loc.match(/^sampling\.prompts\.(\d+)\.(width|height)$/);
  const source = loc.match(/^dataset\.sources\.(\d+)\.resolutions$/);
  const field = ADVICE_FIELDS[loc]?.[english ? 1 : 0]
    ?? (prompt ? (english ? `Prompt ${Number(prompt[1]) + 1} ${prompt[2]}` : `第 ${Number(prompt[1]) + 1} 条提示词的${prompt[2] === 'width' ? '宽度' : '高度'}`)
      : source ? (english ? `Source ${Number(source[1]) + 1} resolution` : `第 ${Number(source[1]) + 1} 个数据源的分辨率`) : loc);
  const effect = warning.code === 'dataset.resolution_step'
    ? (english ? `bucket sides snap to multiples of ${warning.step}, so training sizes differ from the value entered.` : `分桶边长会取整到 ${warning.step} 的倍数，实际训练尺寸与填写值不同。`)
    : warning.code === 'dataset.native_side_step'
      ? (english ? `sides actually stay within ${suggestion} px.` : `实际最长边不超过 ${suggestion} px。`)
      : warning.code === 'sampling.size_step'
        ? (english ? `previews are generated at ${suggestion} px.` : `预览会按 ${suggestion} px 生成。`)
        : null;
  if (!effect) return null;
  const offGrid = english ? `${values} is not a multiple of ${warning.step}; ` : `${values} 不是 ${warning.step} 的倍数，`;
  return {
    loc,
    msg: english ? `${field} ${offGrid}${effect}` : `${field} ${offGrid}${effect}`,
    inline: english ? `${offGrid.charAt(0).toUpperCase()}${offGrid.slice(1)}${effect}` : `${offGrid}${effect}`,
    fix: { label: english ? `Use ${suggestion}` : `改为 ${suggestion}`, value: warning.suggestion },
  };
}

export function presentPlanWarning(code: string, fallback: string, english = false, warning?: ValueAdvice) {
  const advice = warning && presentValueAdvice(warning, english);
  if (advice) return advice.msg;
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
    'native.execution': '图像面积上限只限制单次计算的图片面积，不等于显存上限。',
  } as Record<string, string>)[code] || fallback;
}
