import { mergeConfig } from './config';

const VERSION_FIELDS = [
  ['dataset', 'sources'],
  ['dataset', 'cache_dir'],
  ['validation', 'sources'],
  ['checkpoint', 'output_dir'],
  ['checkpoint', 'resume'],
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
  return mergeConfig(current, reusableTrainingPreset(preset));
}
