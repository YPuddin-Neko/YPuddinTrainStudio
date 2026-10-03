import type { JobMetrics } from '../api/types';

export type GpuMetricDevice = NonNullable<JobMetrics['gpu_devices']>[number];
export const GPU_READING_KEYS = ['power_w', 'temp_c', 'util_pct', 'mem_used_mb', 'mem_total_mb'] as const;
export type GpuReadingKey = typeof GPU_READING_KEYS[number];
export const GPU_SENSOR_METRICS = ['gpu_power', 'gpu_temp', 'gpu_util', 'gpu_memory'] as const;
const validReading = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0;

/** Whether a job recorded several GPUs, so a chart can choose which of them it shows. */
export const hasSeveralGpus = (metrics: JobMetrics) => (metrics.gpu_devices?.length ?? 0) > 1;

/**
 * The GPU a chart reads. A saved choice applies to jobs that recorded several GPUs; a job on one GPU, or from before
 * per-GPU readings, shows that GPU whatever the layout names.
 */
export function resolveGpuMetricSource(metrics: JobMetrics, source: string): string {
  return hasSeveralGpus(metrics) ? source : 'primary';
}

/** How a GPU source is named when it is not one of the listed devices. */
export function gpuSourceLabel(source: string, english = false): string {
  if (source === 'primary') return english ? 'Primary training GPU' : '主训练 GPU';
  if (source === 'average') return english ? 'Training GPU average' : '训练 GPU 平均';
  if (source === 'mps') return 'Apple GPU';
  return source.replace('cuda:', english ? 'Training GPU ' : '训练 GPU ');
}

/** Each live reading occupies the same step as the training metrics, including missing devices. */
export function appendGpuDeviceStep(previous: JobMetrics, incoming: unknown): GpuMetricDevice[] {
  const rows = new Map<string, Record<string, unknown>>();
  if (Array.isArray(incoming)) for (const row of incoming) {
    if (!row || typeof row !== 'object' || typeof row.id !== 'string' || row.id.length > 24 || !/^(cuda:(0|[1-9][0-9]*)|mps)$/.test(row.id) || rows.has(row.id)) continue;
    rows.set(row.id, row);
  }
  const existing = previous.gpu_devices || [];
  const ids = [...new Set([...existing.map(device => device.id), ...rows.keys()])];
  return ids.map(id => {
    const prior = existing.find(device => device.id === id);
    const row = rows.get(id);
    const result = prior ? { ...prior } : {
      id, name: typeof row?.name === 'string' ? row.name : id,
      index: Number(id.split(':')[1] || 0),
      kind: typeof row?.kind === 'string' && ['cuda', 'dtk', 'rocm', 'mps'].includes(row.kind) ? row.kind : id === 'mps' ? 'mps' : 'cuda',
    } as GpuMetricDevice;
    for (const key of GPU_READING_KEYS) {
      const values = prior?.[key] || [];
      result[key] = [...Array.from({ length: previous.steps.length }, (_, index) => values[index] ?? null), validReading(row?.[key]) ? row[key] as number : null];
    }
    return result;
  });
}

/** Averages omit unavailable sensors; a missing reading is never interpreted as zero. */
export function gpuDeviceValues(metrics: JobMetrics, source: string, key: GpuReadingKey): Array<number | null> {
  const devices = metrics.gpu_devices || [];
  if (source === 'primary') {
    const legacy = key === 'power_w' ? metrics.gpu_power_w : key === 'temp_c' ? metrics.gpu_temp_c : key === 'util_pct' ? metrics.gpu_util_pct : undefined;
    if (legacy?.some(validReading)) return legacy.map(value => validReading(value) ? value : null);
  }
  const selected = source === 'average' ? devices : source === 'primary' ? devices.slice(0, 1) : devices.filter(device => device.id === source);
  return metrics.steps.map((_, index) => {
    const values = selected.map(device => device[key]?.[index]).filter(validReading);
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  });
}

export function gpuMetricDeviceLabel(device: Pick<GpuMetricDevice, 'id' | 'index' | 'name'>, english = false): string {
  return `${english ? 'Training GPU' : '训练 GPU'}${device.id === 'mps' ? '' : ` ${device.index}`} · ${device.name}`;
}
