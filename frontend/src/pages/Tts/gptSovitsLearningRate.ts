import type { TtsTrainSchema } from '../../api/tts';
import type { GptSovitsField } from './gptSovitsVersionFields';

export const ignoredGptLearningRateFields: readonly GptSovitsField[] = [
  'gpt.learning_rate', 'gpt.initial_learning_rate', 'gpt.final_learning_rate', 'gpt.warmup_steps', 'gpt.decay_steps',
];
export type GptSovitsFixedLearningRate = { initial_optimizer_lr: number; after_first_scheduler_step_lr: number };
const record = (value: unknown): Record<string, unknown> | undefined => value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : undefined;

export function resolveGptSovitsFixedLearningRate(schema?: TtsTrainSchema): GptSovitsFixedLearningRate | undefined {
  if (schema?.engine !== 'gpt-sovits-v5' || schema.schema_version !== 1) return undefined;
  const definitions = record(schema.json_schema.$defs);
  const settings = record(definitions?.GptSettings);
  const properties = record(settings?.properties);
  if (!ignoredGptLearningRateFields.every(field => {
    const property = record(properties?.[field.slice(4)]);
    return property?.readOnly === true && property.deprecated === true && property['x-training-effect'] === 'ignored';
  })) return undefined;
  const matches = schema.fixed?.filter(item => item.key === 'gpt_learning_rate') || [];
  if (matches.length !== 1) return undefined;
  const value = record(matches[0].value);
  const initial = value?.initial_optimizer_lr, afterFirst = value?.after_first_scheduler_step_lr;
  if (typeof initial !== 'number' || !Number.isFinite(initial) || initial <= 0
    || typeof afterFirst !== 'number' || !Number.isFinite(afterFirst) || afterFirst <= 0) return undefined;
  return { initial_optimizer_lr: initial, after_first_scheduler_step_lr: afterFirst };
}
