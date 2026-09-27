import { reusableTrainingPreset } from './trainingPresets';

const HIDDEN_FIELDS: Record<string, string[]> = {
  training: ['resume_weights'],
  // The family is chosen above the form; model files stay editable and optional.
  model: ['family'],
  dataset: ['sources', 'cache_dir'],
  validation: ['sources'],
  checkpoint: ['output_dir', 'resume'],
  adapter: ['resume_weights'],
  sampling: ['output_dir', 'prompts_file'],
  logging: ['events_path'],
};

export function presetEditorSchema(schema: any) {
  const result = structuredClone(schema);
  for (const [group, fields] of Object.entries(HIDDEN_FIELDS)) {
    let section = result.properties?.[group];
    if (section?.$ref) section = section.$ref.slice(2).split('/').reduce((item: any, key: string) => item?.[key], result);
    for (const field of fields) if (section?.properties) delete section.properties[field];
  }
  return result;
}

export function presetFamily(config: Record<string, unknown>): string {
  const model = config.model;
  if (!model || typeof model !== 'object' || !('family' in model)) return '';
  return typeof model.family === 'string' ? model.family : '';
}

export function presetPayload(draft: { name: string; description: string; config: Record<string, any> }) {
  return { name: draft.name.trim(), description: draft.description.trim(), config: reusableTrainingPreset(draft.config) };
}
