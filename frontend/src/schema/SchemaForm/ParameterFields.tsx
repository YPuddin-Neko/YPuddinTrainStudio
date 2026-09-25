import React from 'react';

type Section = { key: string; names: string[]; title?: [string, string]; togglesFirst?: boolean; inlineToggles?: boolean };
const section = (key: string, names: string[], title?: [string, string], options: Pick<Section, 'togglesFirst'|'inlineToggles'> = {}): Section => ({ key, names, title, ...options });

/** Every group uses the same grid; these lists only decide grouping and order. */
const layouts: Record<string, Section[]> = {
  model: [
    section('setup', ['model.family', 'training.mode', 'training.train_backbone', 'training.train_text_encoder']),
    // A variant describes the main model file, so it sits beside that file.
    section('files', ['model.dit_path', 'model.prediction_type', 'model.zero_terminal_snr', 'model.flux2_variant', 'model.krea2_variant', 'model.text_encoder_path', 'model.text_encoder_2_path', 'model.vae_path', 'model.tokenizer_path', 'model.sdxl_max_token_length', 'training.resume_weights', 'model.dtype'], ['模型文件', 'Model files']),
  ],
  dataset: [
    section('sources', ['sources']),
    section('sizing', ['resolution_mode', 'resolutions', 'native_max_pixels', 'native_max_side', 'image_fit', 'crop_anchor', 'aspect_ratio_limit', 'native_overflow', 'area_tolerance', 'bucket_step', 'bucket_no_upscale'], ['尺寸与分桶', 'Sizing and buckets']),
    section('images', ['flip', 'masked_loss'], ['增强与遮罩', 'Augmentation and masks']),
    section('loading', ['text_encoding', 'num_workers', 'cache_latents', 'cache_dir'], ['数据读取与缓存', 'Data loading and caching']),
  ],
  caption: [
    section('text', ['dataset.caption.trigger_word', 'dataset.caption.prefix', 'dataset.caption.suffix', 'dataset.caption.separator'], ['标签内容', 'Caption content']),
    section('ordering', ['dataset.caption.shuffle', 'dataset.caption.keep_tokens'], ['标签顺序', 'Caption ordering'], { togglesFirst: true }),
    section('dropout', ['dataset.caption.tag_dropout', 'dataset.caption.caption_dropout'], ['标签丢弃', 'Caption dropout']),
    section('variants', ['dataset.caption.cache_variants', 'dataset.caption.wildcard'], ['文本变化', 'Text variation']),
  ],
  loop: [
    section('batch', ['gpu_count', 'distributed_strategy', 'dataset.batch_size', 'grad_accum'], ['设备与批量', 'Devices and batches']),
    section('duration', ['epochs', 'max_steps'], ['训练时长', 'Training duration']),
    section('compute', ['mixed_precision', 'seed', 'deterministic'], ['精度与随机性', 'Precision and randomness']),
    section('ema', ['ema', 'ema_decay'], ['权重平均', 'Weight averaging'], { togglesFirst: true }),
    section('monitoring', ['nan_skip_limit', 'log_every'], ['异常处理与记录', 'Failures and logging']),
  ],
  adapter: [
    section('setup', ['algo', 'preset', 'parameter_mode', 'factor', 'rank', 'alpha', 'dora', 'decompose_both', 'rs_lora', 'rules']),
    section('initialization', ['init', 'resume_weights'], ['初始化与继续训练', 'Initialization and weight loading']),
    section('regularization', ['dropout', 'rank_dropout', 'module_dropout'], ['训练正则', 'Training regularization']),
    section('execution', ['mode', 'param_dtype', 'lr_scale'], ['计算与学习率', 'Computation and learning rate']),
  ],
  optimizer: [
    section('basic', ['type', 'lr', 'weight_decay'], ['优化器与学习率', 'Optimizer and learning rate']),
    section('adaptive', ['d_coef', 'd0', 'beta3', 'prodigy_steps', 'growth_rate', 'slice_p', 'min_lr', 'max_lr', 'lr_bump', 'use_speed', 'd_limiter', 'safeguard_warmup', 'split_groups', 'split_groups_mean'], ['自动步长估计', 'Step-size estimation']),
    section('averaging', ['use_schedulefree', 'schedulefree_c', 'weight_decay_by_lr', 'decouple'], ['权重平均与衰减', 'Weight averaging and decay'], { togglesFirst: true }),
    section('stability', ['betas', 'beta2', 'eps', 'grad_clip_norm', 'clip_threshold', 'use_bias_correction', 'use_stableadamw'], ['平滑与稳定性', 'Smoothing and stability']),
    section('precision', ['factored', 'factored_fp32', 'stochastic_rounding', 'kahan'], ['状态占用与精度', 'State memory and precision']),
    section('variants', ['use_cautious', 'use_grams', 'use_adopt', 'use_orthograd', 'use_focus'], ['更新方式', 'Update methods']),
    section('options', ['group_lr', 'args'], ['优化器选项', 'Optimizer options']),
  ],
  scheduler: [section('curve', ['type', 'warmup_steps', 'min_lr_ratio', 'num_cycles', 'power', 'decay_steps'])],
  memory: [section('resources', ['model.attention', 'base_precision', 'blocks_to_swap', 'activation_checkpointing', 'offload_text_encoder', 'compile', 'allow_tf32'])],
  objective: [
    section('noise', ['timestep_sampling', 'logit_mean', 'logit_std', 'shift', 'mode_scale', 'res_shift_tokens', 'res_shift_mu', 't_min', 't_max', 'stratified'], ['噪声与时间步', 'Noise and timesteps']),
    section('loss', ['loss', 'huber_c', 'weighting', 'snr_gamma', 'ip_noise_gamma', 'v_pred_like_loss', 'scale_v_pred_loss_like_noise_pred', 'debiased_estimation_loss'], ['损失与加权', 'Loss and weighting']),
  ],
  sampling: [
    section('switches', ['enabled', 'at_start']),
    section('cadence', ['every_steps', 'every_epochs'], ['生成频率', 'Preview frequency']),
    section('prompts', ['prompts', 'prompts_file']),
    section('image', ['width', 'height', 'seed'], ['预览图像', 'Preview images']),
    section('sampling', ['sampler', 'scheduler', 'steps', 'cfg', 'shift', 'guidance', 'er_sde_order', 'er_sde_s_noise'], ['采样设置', 'Sampling settings']),
  ],
  checkpoint: [
    section('files', ['name', 'save_dtype'], ['权重文件', 'Weight files']),
    section('cadence', ['save_every_steps', 'save_every_epochs', 'keep_last_n', 'save_on_finish', 'save_training_metadata'], ['权重保存', 'Weight saving']),
    section('state', ['save_state_every_steps', 'resume'], ['断点恢复', 'Training state']),
  ],
  validation: [
    section('switches', ['enabled']),
    section('sources', ['split_ratio', 'sources'], ['验证数据', 'Validation data']),
    section('cadence', ['every_steps', 'every_epochs'], ['验证频率', 'Validation frequency']),
    section('evaluation', ['max_images', 'timesteps', 'seed'], ['评估设置', 'Evaluation settings']),
  ],
  logging: [section('options', ['level', 'tensorboard'], undefined, {inlineToggles:true})],
};

const isToggle = (node: React.ReactNode) => React.isValidElement(node) && (node.props as any)['data-control-kind'] === 'toggle';
const isWide = (node: React.ReactNode) => React.isValidElement(node) && (node.props as any)['data-field-span'] === 'wide';

/** Conditional controls stay below their switches, so enabling them does not move the trigger. */
export function FieldSection({ fields, title, className = '', togglesFirst = false, inlineToggles = false }: {
  fields: React.ReactNode[]; title?: string; className?: string; togglesFirst?: boolean; inlineToggles?:boolean;
}) {
  const toggles = fields.filter(isToggle);
  const plain = fields.filter(field => !isToggle(field) && !isWide(field));
  const wide = fields.filter(isWide);
  const toggleRow = toggles.length>0 ? <div className="config-toggle-row">{toggles}</div> : null;
  return <div className={`config-field-section ${className}`} data-field-count={fields.length} data-only-toggles={toggles.length === fields.length || undefined}>
    {title && <h3>{title}</h3>}
    {inlineToggles ? [...plain,...toggles] : togglesFirst ? <>{toggleRow}{plain}</> : <>{plain}{toggleRow}</>}
    {wide}
  </div>;
}

export default function ParameterFields({ group, fields, english }: { group: string; fields: React.ReactNode[]; english: boolean }) {
  const sections = layouts[group];
  if (!sections) return <FieldSection fields={fields}/>;
  const path = (node: React.ReactNode) => String((node as React.ReactElement).key);
  const fullPath = (name: string) => name.includes('.') ? name : `${group}.${name}`;
  const assigned = new Set(sections.flatMap(item => item.names.map(fullPath)));
  const remaining = fields.filter(field => !assigned.has(path(field)));
  const visible = sections.map(item => {
    const content = item.names.flatMap(name => fields.filter(field => path(field) === fullPath(name)));
    if (group === 'model' && item.key === 'setup') {
      const targets = content.filter(field => ['training.train_backbone', 'training.train_text_encoder'].includes(path(field)));
      if (targets.length) {
        const other = content.filter(field => !targets.includes(field));
        const label = english ? 'Training components' : '训练对象';
        other.splice(Math.min(2, other.length), 0, <div key="training-components" className="config-field config-training-targets" role="group" aria-label={label}>
          <div className="config-field-heading"><span>{label}</span></div>
          <div className="config-field-control config-training-target-controls">{targets}</div>
          <div className="config-field-footer"/>
        </div>);
        return { item, content: other };
      }
    }
    return { item, content };
  }).filter(entry => entry.content.length > 0);
  // A lone section needs no subtitle: the group heading already names it.
  const titled = visible.length + (remaining.length > 0 ? 1 : 0) > 1;
  return <>{visible.map(({ item, content }) => <FieldSection key={item.key} fields={content} title={titled ? item.title?.[english ? 1 : 0] : undefined}
    className={`config-${group}-${item.key}`} togglesFirst={item.togglesFirst} inlineToggles={item.inlineToggles}/>)}
    {remaining.length > 0 && <FieldSection fields={remaining}/>}</>;
}
