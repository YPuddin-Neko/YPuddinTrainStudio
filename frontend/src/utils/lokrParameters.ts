import type { FamilyInfo } from '../api/types';
import { presetHasConvolutions, selectedPreset } from './fieldContext';

export function usesOnlyFullLoKr(config: Record<string, any>, family?: FamilyInfo) {
  const adapter = config.adapter;
  if ((config.training?.mode ?? 'adapter') !== 'adapter' || adapter?.algo !== 'lokr' || adapter.rank !== 'full'
    || !Array.isArray(adapter.rules) || !adapter.rules.every((rule: {algo?: string; rank?: number | string | null} | null) => rule && (rule.algo === 'none'
      || ((rule.algo || 'lokr') === 'lokr' && (rule.rank == null || rule.rank === 'full'))))) return false;
  if (adapter.layer_types === 'linear' || config.training?.train_backbone === false
    || (adapter.layer_types === 'linear_conv' && adapter.conv_rank === null)) return true;
  const preset = selectedPreset(family, config);
  return !!preset && typeof preset.conv_layers === 'number' && !presetHasConvolutions(preset);
}
