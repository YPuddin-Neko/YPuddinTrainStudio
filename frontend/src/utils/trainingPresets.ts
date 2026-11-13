import type { FamilyInfo } from '../api/types';
import { mergeConfig } from './config';
import { normalizeLegacyIntervals } from './intervals';
import { normalizeOptimizerConfig, selectOptimizer } from './optimizerCapabilities';
import { defaultTimestepSampling, supportsResolutionCorrectedShift } from './trainingFamilies';

const VERSION_FIELDS = [
  ['dataset', 'sources'],
  ['dataset', 'cache_dir'],
  ['dataset', 'native_max_pixels_resolved'],
  ['dataset', 'native_batching_version'],
  ['objective', 'resolution_reference_tokens_resolved'],
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
  if (Array.isArray(result.sampling?.prompts)) {
    result.sampling.prompts = result.sampling.prompts.map((prompt: Record<string, unknown>) => {
      const reusable = { ...prompt };
      delete reusable.reference_images;
      return reusable;
    });
  }
  if (result.model && typeof result.model === 'object') {
    for (const field of PRESET_MODEL_FIELDS) if (!result.model[field]) delete result.model[field];
  }
  return result;
}

export function applyTrainingPreset(current: Record<string, any>, preset: Record<string, any>, schema?: any, family?: FamilyInfo) {
  const reusable = reusableTrainingPreset(preset);
  // A preset cannot change the policy belonging to the retained recovery point.
  if (current.checkpoint?.resume && reusable.dataset) {
    for (const field of ['native_batch_mode', 'native_max_pixels_mode', 'native_max_pixels']) delete reusable.dataset[field];
  } else if (!current.checkpoint?.resume && (current.dataset || reusable.dataset)) {
    reusable.dataset = { ...reusable.dataset, native_batch_mode: reusable.dataset?.native_batch_mode ?? 'physical' };
  }
  // Missing opt-ins in older recipes must not inherit an enabled adaptive run.
  reusable.objective = { ...reusable.objective };
  const modelFamily = current.model?.family ?? preset.model?.family;
  if (!Object.prototype.hasOwnProperty.call(reusable.objective, 'infonoise_enabled')) reusable.objective.infonoise_enabled = false;
  if (modelFamily && modelFamily !== 'anima') reusable.objective.infonoise_enabled = false;
  if (modelFamily && reusable.objective.timestep_sampling === 'resolution_sqrt'
    && !supportsResolutionCorrectedShift(family, current.model?.family ? current : preset)) {
    delete reusable.objective.timestep_sampling;
  }
  if (reusable.objective.timestep_sampling === 'moss_image' && modelFamily !== 'moss_image') delete reusable.objective.timestep_sampling;
  if (!Object.prototype.hasOwnProperty.call(reusable.objective, 'timestep_sampling') && current.objective?.timestep_sampling === 'resolution_sqrt') {
    reusable.objective.timestep_sampling = defaultTimestepSampling(family, modelFamily);
  }
  if (reusable.objective.timestep_sampling === 'resolution_sqrt') reusable.objective.resolution_reference_mode ??= 'manual';
  if (modelFamily && modelFamily !== 'moss_image' && current.objective?.timestep_sampling === 'moss_image'
    && !Object.prototype.hasOwnProperty.call(reusable.objective, 'timestep_sampling')) {
    reusable.objective.timestep_sampling = defaultTimestepSampling(family, modelFamily);
  }
  // Model files only fit the family they were chosen for.
  if (reusable.model && (!current.model?.family || reusable.model.family !== current.model.family)) {
    for (const field of PRESET_MODEL_FIELDS) delete reusable.model[field];
  }
  const type = reusable.optimizer?.type;
  const normalizedType = typeof type === 'string' ? normalizeOptimizerConfig(schema, { optimizer: { type } }).optimizer.type : undefined;
  const changedOptimizer = normalizedType !== undefined && normalizedType !== current.optimizer?.type;
  const values = changedOptimizer ? normalizeOptimizerConfig(schema, reusable) : reusable;
  // An algorithm named by the preset starts from its own defaults before explicit values are applied.
  const base = changedOptimizer
    ? selectOptimizer(schema, schema ? current : { ...current, optimizer: {} }, normalizedType, { optimizer: values.optimizer })
    : current;
  const merged = mergeConfig(base, values);
  const explicitEmptyArgs = reusable.optimizer?.args && typeof reusable.optimizer.args === 'object' && !Array.isArray(reusable.optimizer.args) && Object.keys(reusable.optimizer.args).length === 0;
  if (merged.optimizer && (changedOptimizer || explicitEmptyArgs)) {
    merged.optimizer.args = values.optimizer?.args === undefined ? {} : values.optimizer.args;
  }
  if (!current.checkpoint?.resume && merged.dataset) {
    merged.dataset = { ...merged.dataset };
    delete merged.dataset.native_max_pixels_resolved;
    delete merged.dataset.native_batching_version;
  }
  if (current.model?.family) merged.model = { ...merged.model, family: current.model.family };
  return changedOptimizer ? normalizeOptimizerConfig(schema, merged) : merged;
}
