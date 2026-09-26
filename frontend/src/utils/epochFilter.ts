/** Epoch search for previews and saved outputs: "3", "2-5", "3, 7" or "8+". */

export type EpochRanges = Array<[number, number]>;

/** Epochs completed when something was saved; older events only carry the step. */
export function epochAt(item: { epoch?: number | null; step: number }, stepsPerEpoch?: number | null): number | null {
  if (typeof item.epoch === 'number' && Number.isFinite(item.epoch)) return item.epoch;
  return stepsPerEpoch ? item.step / stepsPerEpoch : null;
}

/** The epoch a value belongs to: after 2.4 epochs training is in epoch 3; the end of epoch 3 is 3.0. */
export function epochOrdinal(epoch: number | null): number | null {
  if (epoch == null) return null;
  return epoch <= 1e-6 ? 0 : Math.ceil(epoch - 1e-6);
}

export function epochText(epoch: number | null): string {
  if (epoch == null) return '—';
  return Math.abs(epoch - Math.round(epoch)) < 0.005 ? String(Math.round(epoch)) : epoch.toFixed(2);
}

/** ``null`` for an empty query; ``'invalid'`` when it cannot be read. */
export function parseEpochQuery(query: string): EpochRanges | null | 'invalid' {
  const normalized = query.trim().replace(/\s*[-~～–—]\s*/g, '-');
  if (!normalized) return null;
  const ranges: EpochRanges = [];
  for (const part of normalized.split(/[\s,，、]+/).filter(Boolean)) {
    const single = part.match(/^(\d+)$/), span = part.match(/^(\d+)-(\d+)$/), open = part.match(/^(\d+)\+$/);
    if (single) ranges.push([Number(single[1]), Number(single[1])]);
    else if (span) ranges.push([Math.min(Number(span[1]), Number(span[2])), Math.max(Number(span[1]), Number(span[2]))]);
    else if (open) ranges.push([Number(open[1]), Infinity]);
    else return 'invalid';
  }
  return ranges;
}

export function inEpochs(epoch: number | null, ranges: EpochRanges | null | 'invalid'): boolean {
  if (ranges == null || ranges === 'invalid') return true;
  const ordinal = epochOrdinal(epoch);
  return ordinal != null && ranges.some(([low, high]) => ordinal >= low && ordinal <= high);
}
