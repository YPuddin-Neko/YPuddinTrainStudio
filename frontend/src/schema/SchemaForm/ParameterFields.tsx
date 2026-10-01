import React from 'react';

// afterToggles: settings a switch of the section reveals, placed under the switch row.
type Section = { key: string; names: string[]; title?: [string, string]; togglesFirst?: boolean; inlineToggles?: boolean; beside?: string; afterToggles?: string[] };
const section = (key: string, names: string[], title?: [string, string], options: Pick<Section, 'togglesFirst'|'inlineToggles'|'beside'|'afterToggles'> = {}): Section => ({ key, names, title, ...options });

/** Every group uses the same grid; these lists only decide grouping and order. */
const layouts: Record<string, Section[]> = {
  model: [
    // Full fine-tuning shows its locked layer types here, beside what it trains.
    section('setup', ['model.family', 'training.mode', 'training.train_backbone', 'training.train_text_encoder', 'adapter.layer_types']),
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
    // Two short sections share a row; their fields stack in each half.
    section('dropout', ['dataset.caption.tag_dropout', 'dataset.caption.caption_dropout'], ['标签丢弃', 'Caption dropout'], { beside: 'ordering' }),
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
    section('setup', ['algo', 'preset', 'layer_types', 'parameter_mode', 'factor', 'rank', 'alpha', 'conv_rank', 'conv_alpha', 'tlora_min_rank', 'tlora_power', 'tlora_ortho', 'dora', 'dora_axis', 'decompose_both', 'rs_lora', 'rules'], undefined, { afterToggles: ['dora_axis'] }),
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
  memory: [section('resources', ['model.attention', 'base_precision', 'blocks_to_swap', 'activation_checkpointing', 'vae_attention_chunking', 'offload_text_encoder', 'compile', 'allow_tf32'])],
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
export function FieldSection({ fields, title, className = '', togglesFirst = false, inlineToggles = false, after = [] }: {
  fields: React.ReactNode[]; title?: string; className?: string; togglesFirst?: boolean; inlineToggles?:boolean; after?: React.ReactNode[];
}) {
  const toggles = fields.filter(isToggle);
  const plain = fields.filter(field => !isToggle(field) && !isWide(field));
  const wide = fields.filter(isWide);
  const toggleRow = toggles.length>0 ? <div className="config-toggle-row">{toggles}</div> : null;
  const count = fields.length + after.length;
  return <div className={`config-field-section ${className}`} data-field-count={count} data-only-toggles={toggles.length === count || undefined}>
    {title && <h3>{title}</h3>}
    {inlineToggles ? [...plain,...toggles,...after] : togglesFirst ? <>{toggleRow}{after}{plain}</> : <>{plain}{toggleRow}{after}</>}
    {wide}
  </div>;
}

export default function ParameterFields({ group, fields, english, renderToggleSection }: { group: string; fields: React.ReactNode[]; english: boolean; renderToggleSection?:(path:string,children:React.ReactNode)=>React.ReactNode }) {
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
  const render = ({ item, content }: typeof visible[number]) => {
    const revealed = new Set((item.afterToggles ?? []).map(fullPath));
    const togglePath=group==='loop' && item.key==='ema' ? 'loop.ema' : group==='caption' && item.key==='ordering' ? 'dataset.caption.shuffle' : undefined;
    return togglePath && renderToggleSection
      ? <div key={item.key} className={`config-field-section config-${group}-${item.key}`}>{renderToggleSection(togglePath,<FieldSection fields={content.filter(field=>path(field)!==togglePath)}/>)}</div>
      : <FieldSection key={item.key} fields={content.filter(field => !revealed.has(path(field)))} after={content.filter(field => revealed.has(path(field)))}
        title={titled ? item.title?.[english ? 1 : 0] : undefined} className={`config-${group}-${item.key}`} togglesFirst={item.togglesFirst} inlineToggles={item.inlineToggles}/>;
  };
  const rendered: React.ReactNode[] = [];
  visible.forEach((entry, index) => {
    const previous = visible[index - 1];
    if (entry.item.beside && previous?.item.key === entry.item.beside) {
      rendered[rendered.length - 1] = <div key={`${previous.item.key}+${entry.item.key}`} className="config-section-pair">{render(previous)}{render(entry)}</div>;
    } else rendered.push(render(entry));
  });
  return <>{rendered}
    {remaining.length > 0 && <FieldSection fields={remaining}/>}</>;
}
