interface NumericControlProps {
  id: string;
  label: string;
  value: number | null | undefined;
  onChange: (value: number | undefined) => void;
  min: number;
  max: number;
  step: number;
  percentage?: boolean;
  unit?: string;
  invalid?: boolean;
  sliderLabel: string;
}

/** Display percentages at the control boundary; the configuration always keeps its original units. */
export default function NumericControl({ id, label, value, onChange, min, max, step, percentage = false, unit, invalid, sliderLabel }: NumericControlProps) {
  const scale = percentage ? 100 : 1;
  const toDisplay = (number: number) => Number((number * scale).toPrecision(12));
  const display = typeof value === 'number' && Number.isFinite(value) ? toDisplay(value) : '';
  // A precise typed value must not make the companion range invalidate the form
  // (e.g. 12.5% with a normal 1% slider increment).
  const increments = display === '' ? 0 : (display - toDisplay(min)) / toDisplay(step);
  const rangeStep = Math.abs(increments - Math.round(increments)) > 1e-8 ? 'any' : toDisplay(step);
  const suffix = percentage ? '%' : unit;
  const change = (raw: string) => onChange(raw === '' ? undefined : Number((Number(raw) / scale).toPrecision(12)));
  return <div className="config-number-control">
    <input type="range" aria-label={`${label} ${sliderLabel}`} aria-valuetext={display === '' ? '—' : `${display}${suffix || ''}`}
      min={toDisplay(min)} max={toDisplay(max)} step={rangeStep} value={display === '' ? toDisplay(min) : display}
      onChange={event => change(event.target.value)} />
    <div className="config-number-entry">
      <input id={id} type="number" aria-label={label} aria-describedby={suffix ? `${id}-unit` : undefined} aria-invalid={invalid}
        min={toDisplay(min)} max={toDisplay(max)} step="any" value={display} onChange={event => change(event.target.value)} />
      {suffix && <span id={`${id}-unit`} className="config-number-unit">{suffix}</span>}
    </div>
  </div>;
}
