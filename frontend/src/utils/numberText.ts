/** Plain decimal text for a number, e.g. 1e-7 → 0.0000001. */
export function decimalText(value: number): string {
  const text = String(value);
  if (!/e/i.test(text) || !Number.isFinite(value)) return text;
  const [mantissa, power] = value.toExponential().split('e');
  const exponent = Number(power);
  const sign = mantissa.startsWith('-') ? '-' : '';
  const digits = mantissa.replace('-', '').replace('.', '');
  return exponent < 0 ? `${sign}0.${'0'.repeat(-exponent - 1)}${digits}` : `${sign}${digits.padEnd(exponent + 1, '0')}`;
}

/** Short scientific form for small values, e.g. 0.0001 → 1e-4; empty when plain decimals read well. */
export function scientificText(value: unknown): string {
  return typeof value === 'number' && Number.isFinite(value) && value !== 0 && Math.abs(value) < 0.01
    ? value.toExponential().replace('e+', 'e') : '';
}
