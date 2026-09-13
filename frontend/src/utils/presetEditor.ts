import { reusableTrainingPreset } from './trainingPresets';

const HIDDEN_FIELDS: Record<string, string[]> = {
  model: ['family', 'dit_path', 'text_encoder_path', 'text_encoder_2_path', 'vae_path', 'tokenizer_path'],
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

export function presetSummary(config: Record<string, any>, english = false): string {
  const pieces = [
    config.adapter?.algo?.toUpperCase(),
    config.adapter?.rank != null ? `Rank ${config.adapter.rank}` : null,
    config.optimizer?.lr != null ? `LR ${config.optimizer.lr}` : null,
    config.dataset?.batch_size != null ? `${english ? 'Batch' : '批大小'} ${config.dataset.batch_size}` : null,
    config.loop?.epochs != null ? `${config.loop.epochs} ${english ? 'epochs' : '轮'}` : null,
    config.sampling?.steps != null ? `${english ? 'Preview' : '采样'} ${config.sampling.steps} ${english ? 'steps' : '步'}` : null,
  ];
  return pieces.filter(Boolean).join(' · ');
}

export function presetFamily(config: Record<string, unknown>): string {
  const model = config.model;
  if (!model || typeof model !== 'object' || !('family' in model)) return '';
  return typeof model.family === 'string' ? model.family : '';
}

export function presetPayload(draft: { name: string; description: string; config: Record<string, any> }) {
  return { name: draft.name.trim(), description: draft.description.trim(), config: reusableTrainingPreset(draft.config) };
}
