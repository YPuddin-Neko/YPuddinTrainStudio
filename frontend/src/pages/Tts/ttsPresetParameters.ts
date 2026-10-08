import type { TtsPresetConfig, TtsPresetDocument } from '../../api/ttsPresets';
import type { TtsScopedConfig } from '../../api/tts';
import { fields, fieldSchemas, toDraft, type TtsDraft } from './ttsVersionFields';
import { gptSovitsFields, gptSovitsFieldSchemas, toGptSovitsDraft, type GptSovitsDraft } from './gptSovitsVersionFields';

const environmentFields = new Set(['python_path', 'trainer_path', 'model_path', 'pretrained_gpt', 'pretrained_sovits', 'variant']);
export function portableTtsParameters(config: TtsScopedConfig): TtsPresetConfig {
  if (config.engine === 'gpt-sovits-v5') return structuredClone({ engine: config.engine, stage: config.stage, gpt: config.gpt, sovits: config.sovits });
  return structuredClone(Object.fromEntries(fields.filter(field => !environmentFields.has(field)).map(field => [field, config[field]]))) as TtsPresetConfig;
}
export function rawTtsDraft(draft: TtsDraft | GptSovitsDraft, normalizeNumbers = true): Record<string, unknown> {
  const gsv = draft.engine === 'gpt-sovits-v5';
  const result: Record<string, unknown> = {};
  for (const field of gsv ? gptSovitsFields : fields) {
    const property = gsv ? gptSovitsFieldSchemas[field as keyof GptSovitsDraft] : fieldSchemas[field as keyof TtsDraft];
    const rule = 'anyOf' in property ? property.anyOf?.find(item => item.type !== 'null') || property : property;
    const value = draft[field as keyof typeof draft];
    const numeric = rule.type === 'number' || rule.type === 'integer';
    const converted = !normalizeNumbers ? value : !gsv && field === 'valid_interval' && value === '' ? null : numeric && typeof value === 'string' && value.trim() && Number.isFinite(Number(value)) ? Number(value) : value;
    const [group, name] = field.split('.');
    if (gsv && name) ((result[group] ||= {}) as Record<string, unknown>)[name] = structuredClone(converted);
    else result[field] = structuredClone(converted);
  }
  return result;
}
export function presetEditingConfig(value: Record<string, unknown>): TtsScopedConfig {
  return { python_path: '', trainer_path: '', model_path: '', ...(value.engine === 'gpt-sovits-v5' ? { variant: 'v5dev', pretrained_gpt: '', pretrained_sovits: '' } : {}), ...value } as TtsScopedConfig;
}
export function comparablePresetParameters(value: Record<string, unknown>): Record<string, unknown> {
  const config = presetEditingConfig(value);
  const result = rawTtsDraft(config.engine === 'gpt-sovits-v5' ? toGptSovitsDraft(config) : toDraft(config));
  for (const key of environmentFields) delete result[key];
  return result;
}
export function downloadTtsPreset(document: TtsPresetDocument) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(document, null, 2) + '\n'], { type: 'application/json' }));
  const link = window.document.createElement('a');
  link.href = url; link.download = `${document.name.replace(/[^\p{L}\p{N}_-]+/gu, '-').slice(0, 128) || 'tts-preset'}.json`;
  window.document.body.append(link); link.click(); link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
