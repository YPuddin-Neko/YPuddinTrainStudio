import type { TtsSource } from '../../api/tts';

type Text = (zh: string, en: string) => string;
export const sourceComplete = (source: TtsSource | undefined) => !!source && (source.state === 'valid' || source.state === 'invalid') && !!source.snapshot_id;
export const sourceName = (path: string) => path.replace(/\\/g, '/').split('/').filter(Boolean).pop() || path;
export const sourceState = (source: TtsSource | undefined, text: Text) => !source ? text('未登记', 'Not registered') : ({
  unchecked: text('未检查', 'Unchecked'), checking: text('正在检查', 'Checking'), valid: text('检查通过', 'Passed'),
  invalid: text('存在无效条目', 'Invalid rows'), stale: text('内容已变化', 'Content changed'), error: text('检查失败', 'Check failed'),
})[source.state];
