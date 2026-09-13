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
  for (const [field, kind] of Object.entries(MODEL_PATH_FIELDS)) {
    if (!model[field]) model[field] = assets.find((asset) => asset.family === model.family && asset.kind === kind && asset.is_default && asset.exists && !modelAssetUnsupportedReason(asset))?.path ?? null;
  }
  return { ...config, model };
}

/** Only call on an explicit family change, never on imported configurations. */
export function changeModelFamily(config: Record<string, any>, family: FamilyInfo, assets: ModelAsset[]) {
  const model = { ...config.model, family: family.name, prediction_type: 'epsilon', zero_terminal_snr: false, training_guidance: 1, flux2_variant: 'auto' };
  for (const field of Object.keys(MODEL_PATH_FIELDS)) model[field] = null;
  const attention = familyParameterOptions(family, 'model.attention');
  if (attention?.length && !attention.includes(model.attention)) model.attention = 'auto';
  const capabilities = new Set(family.capabilities);
  const memory = { ...config.memory };
  if (!capabilities.has('block_swap')) memory.blocks_to_swap = 0;
  if (!capabilities.has('fp8_base') && String(memory.base_precision).startsWith('fp8')) memory.base_precision = 'auto';
  if (!capabilities.has('compile')) memory.compile = false;
  if (!capabilities.has('activation_checkpointing')) memory.activation_checkpointing = 'none';
  // Block checkpointing is the shared mode; unsloth support is family-specific.
  else if (memory.activation_checkpointing === 'unsloth') memory.activation_checkpointing = 'block';
  if (memory.blocks_to_swap > 0 && capabilities.has('activation_checkpointing')) {
    memory.activation_checkpointing = 'block';
    memory.compile = false;
  }
  const next = { ...config, model,
    memory,
    adapter: { ...config.adapter, preset: family.default_preset },
    dataset: { ...config.dataset, text_encoding: family.capabilities.includes('online_text') ? 'auto' : 'cached' },
    sampling: { ...config.sampling, ...family.sampling, guidance: family.sampling.guidance ?? null },
    objective: { ...config.objective },
  };
  for (const [group, key] of [['sampling', 'sampler'], ['sampling', 'scheduler'], ['objective', 'timestep_sampling'], ['objective', 'weighting']] as const) {
    const options = familyParameterOptions(family, `${group}.${key}`);
    if (options?.length && !options.includes(next[group][key])) next[group][key] = options[0];
  }
  if (family.name === 'flux2') {
    next.sampling.steps = null; next.sampling.cfg = null; next.sampling.guidance = null;
  }
  if (family.objective !== 'ddpm') {
    next.objective.timestep_sampling = family.sampling.shift == null ? 'resolution_shift' : 'shift';
    next.objective.shift = family.sampling.shift ?? 3;
    next.objective.res_shift_tokens = [256, family.name === 'krea2' ? 6400 : 4096];
  }
  if (family.objective === 'ddpm') {
    next.sampling.shift = 1;
    next.dataset.text_encoding = 'cached';
  }
  return fillDefaultModels(next, assets);
}
