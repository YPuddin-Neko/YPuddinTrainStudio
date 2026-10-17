import type { TtsEngine } from '../api/tts';

export const TTS_ENGINES: TtsEngine[] = ['voxcpm1.5', 'gpt-sovits-v5'];

export function ttsEngineLabel(engine: string | null | undefined): string {
  if (engine === 'voxcpm1.5') return 'VoxCPM 1.5';
  if (engine === 'gpt-sovits-v5') return 'GPT-SoVITS v5';
  return engine || '—';
}
