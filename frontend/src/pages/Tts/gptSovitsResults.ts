import type { GptSovitsSampleOptions, TtsCheckpoint } from '../../api/tts';

type Text = (zh: string, en: string) => string;
export const gptSovitsLanguages = [
  ['zh', '中文', 'Chinese'], ['en', '英语', 'English'], ['ja', '日语', 'Japanese'],
  ['ko', '韩语', 'Korean'], ['yue', '粤语', 'Cantonese'], ['auto', '自动识别', 'Automatic'],
] as const;
export const gptSovitsNumbers = [
  { key: 'top_k', zh: 'Top K 候选数', en: 'Top K', min: 1, max: 100, integer: true, hint: ['限制每步考虑的候选数量。', 'Limit the number of candidates at each step.'] },
  { key: 'top_p', zh: 'Top P 概率范围', en: 'Top P', min: 0, max: 1, exclusive: true, hint: ['按累计概率筛选候选。', 'Filter candidates by cumulative probability.'] },
  { key: 'temperature', zh: '采样温度', en: 'Temperature', min: 0, max: 2, exclusive: true, hint: ['控制生成时的随机程度。', 'Control randomness during generation.'] },
  { key: 'speed', zh: '语速', en: 'Speed', min: 0.5, max: 2, hint: ['1 为原始语速。', '1 keeps the original speed.'] },
  { key: 'repetition_penalty', zh: '重复惩罚', en: 'Repetition penalty', min: 1, max: 2, hint: ['抑制重复的语音内容。', 'Reduce repeated speech content.'] },
  { key: 'fragment_interval', zh: '片段间隔（秒）', en: 'Fragment interval (s)', min: 0, max: 2, hint: ['控制生成片段之间的停顿。', 'Set the pause between generated fragments.'] },
  { key: 'cfg_scale', zh: 'CFG 引导强度', en: 'CFG scale', min: 0, max: 2, nullable: true, hint: ['留空使用变体默认值；0 关闭引导。', 'Leave empty for the variant default; 0 disables guidance.'] },
] as const;
export type GptSovitsNumberKey = typeof gptSovitsNumbers[number]['key'];
export function gptSovitsDefaults(variant: string): Required<GptSovitsSampleOptions> {
  return { text_language: 'zh', reference_language: 'zh', top_k: 15, top_p: 1, temperature: 1, speed: 1, repetition_penalty: 1.35, fragment_interval: 0.3, sample_steps: variant === 'v5turbo' ? 4 : 32, cfg_scale: variant === 'v5turbo' ? 0 : 1.3 };
}
export function validGptSovitsOptions(value: unknown): value is GptSovitsSampleOptions {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const entries = value as Record<string, unknown>;
  return Object.entries(entries).every(([key, number]) => {
    if (key === 'text_language' || key === 'reference_language') return gptSovitsLanguages.some(([language]) => language === number);
    if (key === 'sample_steps') return number === null || [4, 8, 16, 32].includes(number as number) && typeof number === 'number';
    const field = gptSovitsNumbers.find(item => item.key === key);
    if (!field) return false;
    if ('nullable' in field && field.nullable && number === null) return true;
    return typeof number === 'number' && Number.isFinite(number) && number <= field.max
      && ('exclusive' in field ? number > field.min : number >= field.min)
      && (!('integer' in field) || Number.isInteger(number));
  });
}
export function gptSovitsStage(stage: string, text: Text) {
  return stage === 'both' ? text('SoVITS + GPT', 'SoVITS + GPT') : stage === 'gpt' ? text('仅 GPT', 'GPT only') : text('仅 SoVITS', 'SoVITS only');
}
export function checkpointProgress(checkpoint: TtsCheckpoint, text: Text) {
  const value = checkpoint.gpt_sovits;
  if (!value) return `${text('更新步数', 'Update step')} ${checkpoint.step ?? text('未知', 'Unknown')}`;
  return (['gpt', 'sovits'] as const).map(stage => {
    const point = value[stage], name = stage === 'gpt' ? 'GPT' : 'SoVITS';
    return value.stage !== 'both' && value.stage !== stage
      ? text(`${name} · 基础权重（未训练）`, `${name} · base weights (not trained)`)
      : `${name} · ${text('轮次', 'Epoch')} ${point.epoch ?? text('未知', 'Unknown')} · ${text('步数', 'Step')} ${point.global_step ?? text('未知', 'Unknown')}`;
  }).join('\n');
}
