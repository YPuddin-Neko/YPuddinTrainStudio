import { mergeConfig } from './config';

const VERSION_FIELDS = [
  ['dataset', 'sources'],
  ['dataset', 'cache_dir'],
  ['validation', 'sources'],
  ['checkpoint', 'output_dir'],
  ['checkpoint', 'resume'],
  ['adapter', 'resume_weights'],
  ['training', 'resume_weights'],
  ['sampling', 'output_dir'],
  ['sampling', 'prompts_file'],
  ['logging', 'events_path'],
] as const;

export const PRESET_MODEL_FIELDS = ['dit_path', 'text_encoder_path', 'text_encoder_2_path', 'vae_path', 'tokenizer_path'] as const;

/** Old DoRA recipes must not inherit a newly created configuration's compute mode. */
export function preservePresetDoraMode(config: Record<string, any>): Record<string, any> {
  const result = structuredClone(config);
  if (typeof result.adapter?.dora === 'boolean' && !Object.prototype.hasOwnProperty.call(result.adapter, 'dora_compute_mode')) {
    result.adapter.dora_compute_mode = 'standard';
  }
  return result;
}

/**
 * Reusable hyperparameters must not redirect another version's data or output.
 * Chosen model files travel with the preset; empty ones are dropped so applying
 * the preset keeps the configuration's own files.
 */
export function reusableTrainingPreset(config: Record<string, any>): Record<string, any> {
  const result = preservePresetDoraMode(config);
  for (const [group, field] of VERSION_FIELDS) {
    if (result[group] && typeof result[group] === 'object') delete result[group][field];
  }
  if (result.model && typeof result.model === 'object') {
    for (const field of PRESET_MODEL_FIELDS) if (!result.model[field]) delete result.model[field];
  }
  return result;
}

export function applyTrainingPreset(current: Record<string, any>, preset: Record<string, any>) {
  const reusable = reusableTrainingPreset(preset);
  // Model files only fit the family they were chosen for.
  if (reusable.model && (!current.model?.family || reusable.model.family !== current.model.family)) {
    for (const field of PRESET_MODEL_FIELDS) delete reusable.model[field];
  }
  const merged = mergeConfig(current, reusable);
  if (current.model?.family) merged.model = { ...merged.model, family: current.model.family };
  return merged;
}
