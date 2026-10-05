import { useEffect, useState } from 'react';
import { optionalValueLabel } from './optionalValues';

export default function FrequencyInput({ value, enabled, name, english, invalid, onChange }: {
  value: number | null;
  enabled: boolean;
  name: string;
  english: boolean;
  invalid?: boolean;
  onChange: (value: number | null, enabled: boolean) => void;
}) {
  const [draft, setDraft] = useState<number | string | null>(enabled ? value : null);
  useEffect(() => setDraft(enabled ? value : null), [value, enabled]);
  const emptyLabel = optionalValueLabel(name, english);
  return <input id={`config-${name}`} aria-label={name} aria-invalid={invalid || undefined}
    className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-600"
    type="number" min={1} step={1} value={draft ?? ''} placeholder={emptyLabel}
    title={`${english ? 'Leave blank: ' : '留空：'}${emptyLabel}`}
    onChange={event => {
      const next = event.target.value === '' ? '' : Number(event.target.value);
      setDraft(next);
      if (next !== '') onChange(next, true);
    }} onBlur={event => {
      if (draft === '' && event.currentTarget.value === '' && !event.currentTarget.validity.badInput) {
        setDraft(null);
        onChange(value, false);
      }
    }}/>
}
