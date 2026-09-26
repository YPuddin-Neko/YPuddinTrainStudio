import React from 'react';
import { decimalText } from '../../utils/numberText';

type Props = Omit<React.InputHTMLAttributes<HTMLInputElement>, 'type' | 'value'> & { value: number | string | '' };

/** A number input that shows plain decimals: typed 1e-6 becomes 0.000001 once the field is left. */
export default function DecimalNumberInput({ value, onBlur, ...props }: Props) {
  const input = React.useRef<HTMLInputElement>(null);
  // Values set from outside (defaults, presets, imports) never appear in exponent form.
  React.useLayoutEffect(() => {
    const node = input.current;
    if (node && document.activeElement !== node && typeof value === 'number' && /e/i.test(node.value)) node.value = decimalText(value);
  }, [value]);
  return <input {...props} ref={input} type="number" value={value} onBlur={event => {
    const text = event.currentTarget.value;
    if (/e/i.test(text) && Number.isFinite(Number(text))) event.currentTarget.value = decimalText(Number(text));
    onBlur?.(event);
  }}/>;
}
