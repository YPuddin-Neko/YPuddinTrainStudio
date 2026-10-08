import React from 'react';
import DecimalNumberInput from '../../schema/SchemaForm/DecimalNumberInput';
import { scientificText } from '../../utils/numberText';

export function TtsScientificBadge({ value, english }: { value: unknown; english: boolean }) {
  const scientific = typeof value === 'string' && value.trim() ? scientificText(Number(value)) : scientificText(value);
  return scientific ? <span className="config-field-badge" aria-hidden="true" title={english ? 'The same value in scientific notation' : '同一数值的科学计数法'}>{scientific}</span> : null;
}

type DecimalProps = Omit<React.ComponentProps<typeof DecimalNumberInput>, 'value' | 'onChange'> & {
  value: string; onValueChange: (value: string) => void;
};
export function TtsDecimalInput({ value, onValueChange, ...props }: DecimalProps) {
  const number = value.trim() ? Number(value) : NaN;
  return <DecimalNumberInput {...props} step="any" value={Number.isFinite(number) ? number : value}
    onChange={event => onValueChange(event.target.value)}/>;
}
