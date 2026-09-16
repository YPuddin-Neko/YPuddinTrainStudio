import type { FamilyInfo } from '../api/types';
import type { StudioSelectOption } from '../components/StudioSelect';

/** Keep retired engines out of entry points even when connected to an older service. */
export function availableTrainingFamilies(families: FamilyInfo[]): FamilyInfo[] {
  return families.filter(family => family.name !== 'flux').map(family =>
    family.name === 'flux2' ? { ...family, label: 'FLUX.2 Klein 4B / 9B' }
      : family.name === 'sdxl' && (!family.label || family.label === 'SDXL') ? {...family, label:'SDXL 2.6B'} : family);
}

export function inactiveTrainingReason(config: Record<string, any> | null | undefined, english = false): string | undefined {
  const name = config?.model?.family;
  const variant = config?.model?.flux2_variant;
  if (name !== 'flux' && !(name === 'flux2' && variant === 'dev')) return undefined;
  const model = name === 'flux' ? 'FLUX.1' : 'FLUX.2 dev';
  return english
    ? `${model} is retired. This configuration remains available to view; create a new configuration with supported model weights to train. Existing files are preserved.`
    : `${model} 已停用。此配置仍可查看；如需训练，请使用受支持的模型权重新建配置。已有文件保持原样。`;
}

/** The server inspects actual weights; filenames cannot identify a retired variant. */
export function modelAssetUnsupportedReason(asset: { family: string; unsupported_reason?: string | null }): string | undefined {
  return asset.unsupported_reason || (asset.family === 'flux' ? 'FLUX.1 已停用' : undefined);
}

/** Unsupported engines cannot be selected for new configurations. */
export function trainingFamilyOptions(families: FamilyInfo[], _english = false, current = ''): StudioSelectOption[] {
  const available = availableTrainingFamilies(families).filter(family => family.name !== 'toy' || current === 'toy');
  return available.map(family => ({ value: family.name, label: family.label || family.name }));
}

/** Older services omit kind/required; their existing weight fields remain required. */
export function modelFamilyWeights(family?: FamilyInfo) {
  return (family?.name === 'flux' ? [] : family?.weights || []).filter(weight =>
    weight.field !== 'text_encoder_2_path' || family?.name === 'sdxl').map(weight => ({
    field: weight.field,
    kind: typeof weight.kind === 'string' ? weight.kind : weight.field.replace(/_path$/, ''),
    label: weight.label,
    hint: weight.hint,
    required: weight.required !== false,
    downloadable: weight.downloadable !== false,
  }));
}

const capabilityFields = {
  'model.attention': 'attention_backends',
  'sampling.sampler': 'sampling_samplers',
  'sampling.scheduler': 'sampling_schedulers',
  'objective.timestep_sampling': 'objective_timestep_sampling',
  'objective.weighting': 'objective_weighting',
} as const;

/** Missing capabilities preserve the schema choices for older services. */
export function familyParameterOptions(family: FamilyInfo | undefined, path: string): string[] | undefined {
  const key = capabilityFields[path as keyof typeof capabilityFields];
  const options = key && family?.[key];
  return Array.isArray(options) ? options.filter((option): option is string => typeof option === 'string') : undefined;
}
