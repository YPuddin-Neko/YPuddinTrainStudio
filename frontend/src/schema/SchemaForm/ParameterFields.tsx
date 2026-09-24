import React from 'react';

type Section = { key: string; title?: [string, string]; names: string[] };
const section = (key: string, names: string[], title?: [string, string]): Section => ({ key, names, title });
const layouts: Record<string, Section[]> = {
  dataset: [
    section('sources', ['sources']),
    section('sizing', ['resolution_mode', 'resolutions', 'native_max_pixels', 'native_max_side', 'native_overflow', 'image_fit', 'aspect_ratio_limit', 'area_tolerance', 'bucket_step'], ['尺寸与分桶', 'Sizing and buckets']),
    section('images', ['bucket_no_upscale', 'flip', 'masked_loss'], ['图像处理与权重', 'Image processing and weighting']),
    section('loading', ['text_encoding', 'num_workers', 'cache_dir', 'cache_latents'], ['数据读取与缓存', 'Data loading and caching']),
  ],
  caption: [
    section('text', ['dataset.caption.trigger_word', 'dataset.caption.prefix', 'dataset.caption.suffix', 'dataset.caption.separator'], ['标签内容', 'Caption content']),
    section('ordering', ['dataset.caption.keep_tokens', 'dataset.caption.shuffle'], ['标签顺序', 'Caption ordering']),
    section('dropout', ['dataset.caption.tag_dropout', 'dataset.caption.caption_dropout'], ['标签丢弃', 'Caption dropout']),
    section('variants', ['dataset.caption.cache_variants', 'dataset.caption.wildcard'], ['文本变化', 'Text variation']),
  ],
  loop: [
    section('batch', ['gpu_count', 'distributed_strategy', 'dataset.batch_size', 'grad_accum'], ['设备与批量', 'Devices and batches']),
    section('duration', ['epochs', 'max_steps'], ['训练时长', 'Training duration']),
    section('compute', ['mixed_precision', 'seed', 'deterministic'], ['精度与随机性', 'Precision and randomness']),
    section('ema', ['ema_decay', 'ema'], ['权重平均', 'Weight averaging']),
    section('monitoring', ['nan_skip_limit', 'log_every'], ['异常处理与记录', 'Failures and logging']),
  ],
  scheduler: [section('curve', ['type', 'warmup_steps', 'min_lr_ratio', 'num_cycles', 'power', 'decay_steps'])],
  memory: [
    section('storage', ['model.attention', 'base_precision', 'blocks_to_swap'], ['计算与显存', 'Computation and memory']),
    section('execution', ['activation_checkpointing', 'offload_text_encoder', 'compile', 'allow_tf32']),
  ],
  objective: [
    section('noise', ['timestep_sampling', 'logit_mean', 'logit_std', 'res_shift_tokens', 'res_shift_mu', 'shift', 'mode_scale', 't_min', 't_max', 'stratified'], ['噪声与时间步', 'Noise and timesteps']),
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
    section('files', ['name', 'save_dtype', 'output_dir'], ['权重文件', 'Weight files']),
    section('cadence', ['save_every_steps', 'save_every_epochs', 'keep_last_n', 'save_on_finish'], ['权重保存', 'Weight saving']),
    section('state', ['save_state_every_steps', 'resume'], ['断点恢复', 'Training state']),
  ],
  validation: [
    section('switches', ['enabled']),
    section('sources', ['sources', 'split_ratio'], ['验证数据', 'Validation data']),
    section('cadence', ['every_steps', 'every_epochs'], ['验证频率', 'Validation frequency']),
    section('evaluation', ['max_images', 'timesteps', 'seed'], ['评估设置', 'Evaluation settings']),
  ],
  logging: [section('options', ['tensorboard', 'level', 'events_path'])],
};

export function FieldSection({ fields, title, className = '', separateSwitches = true, switchesFirst = false }: {
  fields: React.ReactNode[]; title?: string; className?: string; separateSwitches?: boolean; switchesFirst?: boolean;
}) {
  const isToggle = (node: React.ReactNode) => React.isValidElement(node) && node.props['data-control-kind'] === 'toggle';
  const toggles = separateSwitches ? fields.filter(isToggle) : [];
  const switches = toggles.length > 0 && <div className="config-toggle-list">{toggles}</div>;
  return <div className={`config-field-section ${className}`}>
    {title && <h3>{title}</h3>}
    {switchesFirst && switches}
    {separateSwitches ? fields.filter(field => !isToggle(field)) : fields}
    {!switchesFirst && switches}
  </div>;
}

export default function ParameterFields({ group, fields, english }: { group: string; fields: React.ReactNode[]; english: boolean }) {
  const sections = layouts[group];
  if (!sections) return <>{fields}</>;
  const path = (node: React.ReactNode) => String((node as React.ReactElement).key);
  const fullPath = (name: string) => name.includes('.') ? name : `${group}.${name}`;
  const assigned = new Set(sections.flatMap(item => item.names.map(fullPath)));
  const remaining = fields.filter(field => !assigned.has(path(field)));
  return <>{sections.map(item => {
    const content = item.names.flatMap(name => fields.filter(field => path(field) === fullPath(name)));
    return content.length > 0 && <FieldSection key={item.key} fields={content} title={item.title?.[english ? 1 : 0]}
      className={`config-${group}-${item.key}`} separateSwitches={group !== 'logging'} switchesFirst={group === 'loop' && item.key === 'ema'}/>;
  })}{remaining.length > 0 && <FieldSection fields={remaining}/>}</>;
}
