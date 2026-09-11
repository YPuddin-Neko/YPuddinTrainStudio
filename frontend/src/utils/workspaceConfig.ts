import type { ModelAsset, FamilyInfo, DatasetInfo } from '../api/types';

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
  dit_path: 'dit', text_encoder_path: 'text_encoder', vae_path: 'vae', tokenizer_path: 'tokenizer',
} as const;

/** Registry defaults fill empty paths only; explicit project paths always win. */
export function fillDefaultModels(config: Record<string, any>, assets: ModelAsset[]) {
  const model = { ...(config.model || {}), family: config.model?.family || 'anima' };
  for (const [field, kind] of Object.entries(MODEL_PATH_FIELDS)) {
    if (!model[field]) model[field] = assets.find((asset) => asset.family === model.family && asset.kind === kind && asset.is_default && asset.exists)?.path ?? null;
  }
  return { ...config, model };
}

/** Only call on an explicit family change, never on imported configurations. */
export function changeModelFamily(config: Record<string, any>, family: FamilyInfo, assets: ModelAsset[]) {
  const model = { ...config.model, family: family.name };
  for (const field of Object.keys(MODEL_PATH_FIELDS)) model[field] = null;
  return fillDefaultModels({ ...config, model,
    adapter: { ...config.adapter, preset: family.default_preset },
    dataset: { ...config.dataset, text_encoding: 'auto' },
  }, assets);
}
