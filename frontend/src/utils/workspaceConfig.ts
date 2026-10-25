import type { ModelAsset, FamilyInfo, DatasetInfo } from '../api/types';
import { familyParameterOptions, inactiveTrainingReason, modelAssetUnsupportedReason } from './trainingFamilies';

/** Windows paths are case-insensitive; preserve case for POSIX server paths. */
export function normalizeDatasetPath(path: string) {
  const trimmed = path.trim();
  const windows = /^[a-z]:[\\/]/i.test(trimmed) || /^[\\/]{2}/.test(trimmed);
  const normalized = trimmed.replace(/\\/g, '/').replace(/\/+/g, '/').replace(/\/$/, '') || (trimmed ? '/' : '');
  return windows ? normalized.toLowerCase() : normalized;
}

export function matchingTrainingDatasets(config: Record<string, any>, datasets: DatasetInfo[]) {
  const sources = Array.isArray(config.dataset?.sources) ? config.dataset.sources : [];
  const paths = new Set(sources.flatMap((source: { path?: unknown }) => typeof source?.path === 'string' && source.path.trim() ? [normalizeDatasetPath(source.path)] : []));
  return datasets.filter(dataset => typeof dataset.source?.path === 'string' && paths.has(normalizeDatasetPath(dataset.source.path)));
}

export const MODEL_PATH_FIELDS = {
  dit_path: 'dit', text_encoder_path: 'text_encoder', text_encoder_2_path: 'text_encoder_2', vae_path: 'vae', tokenizer_path: 'tokenizer',
} as const;

/** Registry defaults fill empty paths only; explicit project paths always win. */
export function fillDefaultModels<T extends Record<string, any>>(config: T, assets: ModelAsset[]) {
  if (inactiveTrainingReason(config)) return { ...config, model: { ...config.model } };
  const model = { ...(config.model || {}), family: config.model?.family || 'anima' };
  let defaultTextProcessor: string | null = null;
  for (const [field, kind] of Object.entries(MODEL_PATH_FIELDS)) {
    if (!model[field]) {
      const asset = assets.find((asset) => asset.family === model.family && asset.kind === kind && asset.is_default && asset.exists && !modelAssetUnsupportedReason(asset));
      model[field] = asset?.path ?? null;
      if (field === 'text_encoder_path' && asset && 'processor_path' in asset && typeof asset.processor_path === 'string' && asset.processor_path.trim()) {
        defaultTextProcessor = asset.processor_path;
      }
    }
  }
  if (!model.tokenizer_path && defaultTextProcessor) model.tokenizer_path = defaultTextProcessor;
  const selected = assets.find(asset => asset.path === model.dit_path && asset.family === model.family && asset.kind === 'dit');
  if (model.family === 'krea2' && selected?.variant) model.krea2_variant = selected.variant;
  return { ...config, model };
}

/** Only call on an explicit family change, never on imported configurations. */
export function changeModelFamily(config: Record<string, any>, family: FamilyInfo, assets: ModelAsset[]) {
  const model = { ...config.model, family: family.name, prediction_type: 'epsilon', zero_terminal_snr: false, training_guidance: 1, flux2_variant: 'auto', krea2_variant: 'auto', dtype: 'auto' };
  for (const field of Object.keys(MODEL_PATH_FIELDS)) model[field] = null;
  const attention = familyParameterOptions(family, 'model.attention');
  if (attention?.length && !attention.includes(model.attention)) model.attention = 'auto';
  const capabilities = new Set(family.capabilities);
  const memory = { ...config.memory };
  if (!capabilities.has('block_swap')) memory.blocks_to_swap = 0;
  if (!capabilities.has('fp8_base') && String(memory.base_precision).startsWith('fp8')) memory.base_precision = 'auto';
  if (!capabilities.has('compile')) memory.compile = false;
  const checkpointing = familyParameterOptions(family, 'memory.activation_checkpointing');
  if (!capabilities.has('activation_checkpointing')) memory.activation_checkpointing = 'none';
  // Block checkpointing is the shared mode; offloading only where the family implements it.
  else if (checkpointing?.length ? !checkpointing.includes(memory.activation_checkpointing) : memory.activation_checkpointing === 'unsloth') memory.activation_checkpointing = 'block';
  if (memory.blocks_to_swap > 0 && capabilities.has('activation_checkpointing')) {
    memory.activation_checkpointing = 'block';
    memory.compile = false;
  }
  const next: Record<string, any> = { ...config, model,
    memory,
    adapter: { ...config.adapter, preset: family.default_preset },
    dataset: { ...config.dataset, text_encoding: family.capabilities.includes('online_text') ? 'auto' : 'cached' },
    sampling: { ...config.sampling, ...family.sampling, guidance: family.sampling.guidance ?? null },
    objective: { ...config.objective },
    training: { ...config.training },
  };
  for (const [group, key] of [['sampling', 'sampler'], ['sampling', 'scheduler'], ['objective', 'timestep_sampling'], ['objective', 'weighting']] as const) {
    const options = familyParameterOptions(family, `${group}.${key}`);
    if (options?.length && !options.includes(next[group][key])) next[group][key] = options[0];
  }
  if (family.name !== 'sdxl') {
    next.dataset.caption = {...next.dataset.caption, weighted:false};
    for (const section of ['dataset', 'validation']) {
      if (Array.isArray(next[section]?.sources)) next[section] = {...next[section], sources:next[section].sources.map((source:Record<string,any>) => source.caption ? {...source,caption:Object.fromEntries(Object.entries(source.caption).filter(([key]) => key !== 'weighted'))} : source)};
    }
    next.training.text_encoder_2_lr = null;
  }
  if (!['anima', 'krea2', 'sdxl', 'flux2', 'flux3'].includes(family.name)) {
    next.objective.noise_offset = 0;
    next.objective.multires_noise_iterations = 0;
  }
  if (family.name !== 'anima') {
    for (const key of ['llm_adapter_lr','self_attn_lr','cross_attn_lr','mlp_lr','modulation_lr']) next.training[key] = null;
    next.objective.infonoise_enabled = false;
    if (next.objective.timestep_sampling === 'resolution_sqrt') next.objective.timestep_sampling = family.objective === 'ddpm' ? 'uniform' : family.sampling.shift == null ? 'resolution_shift' : 'shift';
  }
  if (family.name === 'flux2') {
    next.sampling.steps = null; next.sampling.cfg = null; next.sampling.guidance = null;
  }
  if (family.name === 'flux3' || family.name === 'moss_image') {
    const keepFull = next.training.mode === 'full' && familyParameterOptions(family, 'training.mode')?.includes('full');
    next.training = { ...next.training, mode: keepFull ? 'full' : 'adapter', train_backbone: true, train_text_encoder: false, resume_weights: null, backbone_lr: keepFull ? next.training.backbone_lr : null, text_encoder_lr: null };
    if (!keepFull) next.optimizer = { ...next.optimizer, cpu_offload: false, exclude_bias_norm_from_weight_decay: false };
    next.adapter = { ...next.adapter, layer_types: 'linear', resume_weights: null };
    const algorithms = familyParameterOptions(family, 'adapter.algo');
    if (algorithms?.length && !algorithms.includes(next.adapter.algo)) next.adapter.algo = algorithms[0];
    if (algorithms?.length && Array.isArray(next.adapter.rules)) next.adapter.rules = next.adapter.rules.map((rule: Record<string, any>) =>
      rule.algo && rule.algo !== 'none' && !algorithms.includes(rule.algo) ? { ...rule, algo: null } : rule);
    if (next.adapter.algo !== 'lokr' && next.adapter.rank === 'full') next.adapter.rank = 16;
    const basePrecisions = familyParameterOptions(family, 'memory.base_precision');
    if (basePrecisions?.length && !basePrecisions.includes(next.memory.base_precision)) next.memory.base_precision = 'auto';
    const mixedPrecisions = familyParameterOptions(family, 'loop.mixed_precision');
    next.loop = { ...next.loop, gpu_count: 1, distributed_strategy: 'ddp' };
    if (mixedPrecisions?.length && !mixedPrecisions.includes(next.loop.mixed_precision)) next.loop.mixed_precision = mixedPrecisions.includes('bf16') ? 'bf16' : mixedPrecisions[0];
  }
  if (family.objective !== 'ddpm') {
    next.objective.scale_v_pred_loss_like_noise_pred = false;
    next.objective.v_pred_like_loss = 0;
    next.objective.debiased_estimation_loss = false;
    if (family.name === 'flux3') {
      next.objective = { ...next.objective, timestep_sampling: 'logit_normal', weighting: 'none', logit_mean: 0, logit_std: 1, shift: 3 };
    } else if (family.name === 'moss_image') {
      next.objective = { ...next.objective, timestep_sampling: 'moss_image', weighting: 'none', shift: 1 };
    } else {
      next.objective.timestep_sampling = family.sampling.shift == null ? 'resolution_shift' : 'shift';
      next.objective.shift = family.sampling.shift ?? 3;
    }
    next.objective.res_shift_tokens = [256, family.name === 'krea2' ? 6400 : 4096];
  }
  if (family.objective === 'ddpm') {
    next.sampling.shift = 1;
    next.dataset.text_encoding = 'cached';
  }
  return fillDefaultModels(next, assets);
}
