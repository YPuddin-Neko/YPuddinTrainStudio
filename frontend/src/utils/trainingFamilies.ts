import type { FamilyInfo } from '../api/types';
import type { StudioSelectOption } from '../components/StudioSelect';

/** The server registry is authoritative; unsupported engines cannot be selected. */
export function trainingFamilyOptions(families: FamilyInfo[], _english = false, current = ''): StudioSelectOption[] {
  const available = families.filter(family => family.name !== 'toy' || current === 'toy');
  return available.map(family => ({ value: family.name, label: family.label || family.name }));
}

/** Older services omit kind/required; their existing weight fields remain required. */
export function modelFamilyWeights(family?: FamilyInfo) {
  return (family?.weights || []).map(weight => ({
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
