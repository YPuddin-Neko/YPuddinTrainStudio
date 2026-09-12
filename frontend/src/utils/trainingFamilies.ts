import type { FamilyInfo } from '../api/types';
import type { StudioSelectOption } from '../components/StudioSelect';

/** The server registry is authoritative; unsupported engines cannot be selected. */
export function trainingFamilyOptions(families: FamilyInfo[], english = false, current = ''): StudioSelectOption[] {
  const available = families.filter(family => family.name !== 'toy' || current === 'toy');
  return [
    ...available.map(family => ({ value: family.name, label: family.label || family.name })),
    ...['flux', 'sdxl'].filter(name => !families.some(family => family.name === name)).map(name => ({
      value: name, label: `${name === 'flux' ? 'Flux' : 'SDXL'} · ${english ? 'Not integrated yet' : '暂未接入'}`, disabled: true,
    })),
  ];
}
