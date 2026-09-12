
const labels: Record<string, string> = {
  'model.family': '模型系列', 'model.dit_path': '主模型 / DiT', 'model.text_encoder_path': '文本编码器',
  'model.vae_path': 'VAE', 'model.tokenizer_path': '分词器目录', 'model.dtype': '计算精度', 'model.attention': '注意力后端',
  'dataset.sources': '训练数据源', 'dataset.resolutions': '训练分辨率', 'dataset.aspect_ratio_limit': '最大长宽比',
  'dataset.resolution_mode': '分辨率模式', 'dataset.native_max_pixels': '原生像素预算',
  'dataset.native_max_side': '原生最长边', 'dataset.native_overflow': '超出预算时',
  'dataset.area_tolerance': '面积容差', 'dataset.bucket_step': '分桶步长', 'dataset.bucket_no_upscale': '不放大小图',
  'dataset.batch_size': '批大小', 'dataset.flip': '随机水平翻转', 'dataset.masked_loss': '遮罩加权训练',
  'dataset.num_workers': '数据加载线程', 'dataset.cache_dir': '缓存目录', 'dataset.cache_latents': '缓存图像潜变量',
  'dataset.text_encoding': '文本编码方式', 'dataset.caption.prefix': '标签前缀', 'dataset.caption.suffix': '标签后缀',
  'dataset.caption.trigger_word': '触发词', 'dataset.caption.keep_tokens': '保留前几个标签', 'dataset.caption.shuffle': '打乱标签顺序',
  'dataset.caption.tag_dropout': '单个标签丢弃率', 'dataset.caption.caption_dropout': '整条标签丢弃率',
  'dataset.caption.separator': '标签分隔符', 'dataset.caption.wildcard': '启用通配符',
  'dataset.caption.cache_variants': '预缓存标签变体数',
  'sampling.scheduler': '采样调度器', 'sampling.er_sde_order': 'ER-SDE 求解阶数', 'sampling.er_sde_s_noise': 'ER-SDE 随机噪声强度',
  'adapter.algo': '适配器算法', 'adapter.rank': 'Rank / 秩', 'adapter.alpha': 'Alpha / 缩放', 'adapter.factor': 'LoKr 分解因子',
  'adapter.decompose_both': '双矩阵低秩分解', 'adapter.rs_lora': 'Rank 稳定缩放', 'adapter.dora': '启用 DoRA',
  'adapter.init': '初始化方式', 'adapter.dropout': '输出丢弃率', 'adapter.rank_dropout': '秩丢弃率',
  'adapter.module_dropout': '模块丢弃率', 'adapter.preset': '训练目标层', 'adapter.rules': '逐层覆盖规则',
  'adapter.mode': '适配器计算模式', 'adapter.param_dtype': '可训练参数精度', 'adapter.lr_scale': '学习率缩放',
  'adapter.resume_weights': '已有适配器权重', 'objective.timestep_sampling': '时间步采样',
  'objective.logit_mean': 'Logit 均值', 'objective.logit_std': 'Logit 标准差', 'objective.res_shift_tokens': '分辨率偏移基准',
  'objective.res_shift_mu': '分辨率偏移系数', 'objective.shift': '时间步偏移', 'objective.mode_scale': 'Mode 系数',
  'objective.stratified': '分层时间步采样', 'objective.t_min': '时间步下限', 'objective.t_max': '时间步上限',
  'objective.loss': '损失函数', 'objective.huber_c': 'Huber 系数', 'objective.weighting': '损失加权',
  'objective.snr_gamma': 'SNR Gamma', 'objective.ip_noise_gamma': '输入扰动强度', 'optimizer.type': '优化器',
  'optimizer.lr': '学习率', 'optimizer.weight_decay': '权重衰减', 'optimizer.betas': '动量 β1 / β2',
  'optimizer.eps': '数值稳定常量', 'optimizer.args': '优化器附加参数', 'optimizer.grad_clip_norm': '梯度裁剪',
  'optimizer.kahan': 'Kahan 补偿', 'optimizer.fused_backward': '融合反向更新', 'optimizer.group_lr': '参数组学习率',
  'scheduler.type': '学习率调度', 'scheduler.warmup_steps': '预热步数 / 比例', 'scheduler.min_lr_ratio': '最低学习率比例',
  'scheduler.num_cycles': '调度周期数', 'scheduler.power': '多项式幂', 'scheduler.decay_steps': '衰减步数',
  'memory.base_precision': '底模存储精度', 'memory.blocks_to_swap': '换出到 CPU 的层数',
  'memory.activation_checkpointing': '梯度检查点', 'memory.offload_text_encoder': '卸载文本编码器',
  'memory.compile': '编译模型', 'memory.allow_tf32': '允许 TF32', 'loop.max_steps': '最大训练步数',
  'loop.epochs': '训练轮数', 'loop.grad_accum': '梯度累积', 'loop.mixed_precision': '混合精度', 'loop.seed': '随机种子',
  'loop.ema': '启用 EMA', 'loop.ema_decay': 'EMA 衰减', 'loop.nan_skip_limit': '无效梯度跳过上限', 'loop.log_every': '日志间隔',
  'checkpoint.output_dir': '输出目录', 'checkpoint.name': '权重文件前缀', 'checkpoint.save_every_steps': '每隔几步保存',
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
  return english ? fallback : labels[path] || fallback;
}

export function configOptionLabel(path: string, option: string, english = false) {
  const options: Record<string, Record<string, [string, string]>> = {
    'sampling.sampler': { euler:['Euler · 一阶', 'Euler · first order'], heun:['Heun · 二阶', 'Heun · second order'], er_sde:['ER-SDE · 随机微分方程', 'ER-SDE · stochastic solver'] },
    'sampling.scheduler': { uniform:['Uniform · 连续等间隔', 'Uniform · continuous'], simple:['Simple · 离散网格', 'Simple · discrete grid'], sgm_uniform:['SGM Uniform', 'SGM Uniform'], normal:['Normal · 包含末端', 'Normal · includes low endpoint'] },
    'optimizer.type': {adamw:['AdamW','AdamW'],adam:['Adam','Adam'],sgd:['SGD','SGD'],adamw8bit:['AdamW 8-bit','AdamW 8-bit'],lion:['Lion','Lion'],lion8bit:['Lion 8-bit','Lion 8-bit'],prodigy:['Prodigy','Prodigy'],prodigy_plus_sf:['Prodigy Plus Schedule-Free','Prodigy Plus Schedule-Free'],adafactor:['Adafactor','Adafactor'],came:['CAME','CAME'],adamw_sf:['AdamW Schedule-Free','AdamW Schedule-Free']},
    'dataset.resolution_mode': { bucket: ['分桶 · 统一基准面积', 'Buckets · target area'], native: ['原生 · 每图独立尺寸', 'Native · individual image sizes'] },
    'dataset.native_overflow': { downscale: ['等比缩小到预算内', 'Downscale to fit budget'], error: ['报错并停止', 'Stop with an error'] },
  };
  return options[path]?.[option]?.[english ? 1 : 0] || option;
}

export type ConfigTab = 'train' | 'data' | 'model' | 'advanced';
export const CONFIG_TAB_GROUPS: Record<ConfigTab, string[]> = {
  train: ['loop', 'adapter', 'optimizer', 'scheduler', 'memory'],
  data: ['dataset', 'caption'],
  model: ['model', 'checkpoint'],
  advanced: ['objective', 'sampling', 'validation', 'logging'],
};
export function configTabForPath(path: string): ConfigTab {
  if (path === 'dataset.batch_size') return 'train';
  const group = path.split('.')[0];
  return (Object.keys(CONFIG_TAB_GROUPS) as ConfigTab[]).find(tab => CONFIG_TAB_GROUPS[tab].includes(group)) || 'advanced';
}

export type ConfigIssue = { path: string; label: string; message: string; detail: string; tab: ConfigTab };
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
      else if (/unknown preset/i.test(detail)) message = '当前模型不支持这个目标层预设，请重新选择';
      else if (Array.from(detail).every(character => character.charCodeAt(0) < 128)) message = '此配置未通过检查，展开详情查看具体原因';
    }
    return {path, label, message, detail, tab: configTabForPath(path)};
  });
  return issues.filter((issue, index) => issues.findIndex(item => item.path === issue.path && item.detail === issue.detail) === index);
}

export function presentPlanWarning(code: string, fallback: string, english = false) {
  if (english) return fallback;
  return ({
    'device.mps_fp32': 'Apple GPU 使用 FP32 训练，不启用混合精度。',
    'captions.missing': '部分图片没有标签，可到数据集补充标签或设置类别提示词。',
    'buckets.small': '部分分桶不足一个完整批次，最后一个批次会使用较少的图片。',
    'native.execution': '不同尺寸按像素预算分组前向，按图片数累积梯度；像素上限并非整体显存保证。',
  } as Record<string, string>)[code] || fallback;
}
