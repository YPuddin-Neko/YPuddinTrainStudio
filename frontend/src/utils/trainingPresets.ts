import { mergeConfig } from './config';
import { normalizeLegacyIntervals } from './intervals';

const VERSION_FIELDS = [
  ['dataset', 'sources'],
  ['dataset', 'cache_dir'],
  ['dataset', 'native_max_pixels_resolved'],
  ['dataset', 'native_batching_version'],
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
  const result = normalizeLegacyIntervals(preservePresetDoraMode(config));
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
  // A preset cannot change the policy belonging to the retained recovery point.
  if (current.checkpoint?.resume && reusable.dataset) {
    for (const field of ['native_batch_mode', 'native_max_pixels_mode', 'native_max_pixels']) delete reusable.dataset[field];
  } else if (!current.checkpoint?.resume && (current.dataset || reusable.dataset)) {
    reusable.dataset = { ...reusable.dataset, native_batch_mode: reusable.dataset?.native_batch_mode ?? 'physical' };
  }
  // Missing opt-ins in older recipes must not inherit an enabled adaptive run.
  reusable.objective = { ...reusable.objective };
  if (!Object.prototype.hasOwnProperty.call(reusable.objective, 'infonoise_enabled')) reusable.objective.infonoise_enabled = false;
  if (!Object.prototype.hasOwnProperty.call(reusable.objective, 'timestep_sampling') && current.objective?.timestep_sampling === 'resolution_sqrt') {
    reusable.objective.timestep_sampling = 'shift';
  }
  if (current.model?.family && current.model.family !== 'anima') {
    reusable.objective.infonoise_enabled = false;
    if (reusable.objective.timestep_sampling === 'resolution_sqrt') {
      // Applying another family's recipe cannot opt the current model into Anima-only sampling.
      delete reusable.objective.timestep_sampling;
    }
    if (current.objective?.timestep_sampling === 'resolution_sqrt' && (!Object.prototype.hasOwnProperty.call(preset.objective || {}, 'timestep_sampling') || preset.objective?.timestep_sampling === 'resolution_sqrt')) {
      reusable.objective.timestep_sampling = current.model.family === 'flux3' ? 'logit_normal' : current.model.family === 'sdxl' ? 'uniform' : ['krea2', 'flux2'].includes(current.model.family) ? 'resolution_shift' : 'shift';
    }
  }
  // Model files only fit the family they were chosen for.
  if (reusable.model && (!current.model?.family || reusable.model.family !== current.model.family)) {
    for (const field of PRESET_MODEL_FIELDS) delete reusable.model[field];
  }
  const merged = mergeConfig(current, reusable);
  if (!current.checkpoint?.resume && merged.dataset) {
    merged.dataset = { ...merged.dataset };
    delete merged.dataset.native_max_pixels_resolved;
    delete merged.dataset.native_batching_version;
  }
  if (current.model?.family) merged.model = { ...merged.model, family: current.model.family };
  return merged;
}
