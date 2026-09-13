import { mergeConfig } from './config';

const VERSION_FIELDS = [
  ['model', 'dit_path'],
  ['model', 'text_encoder_path'],
  ['model', 'text_encoder_2_path'],
  ['model', 'vae_path'],
  ['model', 'tokenizer_path'],
  ['dataset', 'sources'],
  ['dataset', 'cache_dir'],
  ['validation', 'sources'],
  ['checkpoint', 'output_dir'],
  ['checkpoint', 'resume'],
  ['adapter', 'resume_weights'],
  ['sampling', 'output_dir'],
  ['sampling', 'prompts_file'],
  ['logging', 'events_path'],
] as const;

/** Reusable hyperparameters must not redirect another version's data or output. */
export function reusableTrainingPreset(config: Record<string, any>): Record<string, any> {
  const result = structuredClone(config);
  for (const [group, field] of VERSION_FIELDS) {
    if (result[group] && typeof result[group] === 'object') delete result[group][field];
  }
  return result;
}

export function applyTrainingPreset(current: Record<string, any>, preset: Record<string, any>) {
  const merged = mergeConfig(current, reusableTrainingPreset(preset));
  if (current.model?.family) merged.model = { ...merged.model, family: current.model.family };
  return merged;
}
