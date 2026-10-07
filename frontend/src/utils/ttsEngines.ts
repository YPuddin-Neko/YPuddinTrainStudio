export function ttsEngineLabel(engine: string | null | undefined): string {
  if (engine === 'voxcpm1.5') return 'VoxCPM 1.5';
  if (engine === 'gpt-sovits-v5') return 'GPT-SoVITS v5';
  return engine || '—';
}
