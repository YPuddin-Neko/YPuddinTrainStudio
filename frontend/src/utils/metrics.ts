import { JobMetrics, ValidationPoint } from '../api/types';
import { appendGpuDeviceStep } from './gpuMetricSeries';

export function smoothLoss(values: Array<number | null>, alpha: number): Array<number | null> {
  let ema: number | null = null;
  return values.map((value) => {
    if (value == null || !Number.isFinite(value)) return null;
    ema = ema == null ? value : alpha * ema + (1 - alpha) * value;
    return ema;
  });
}

export function appendMetricStep(previous: JobMetrics, event: Record<string, any>): JobMetrics {
  if (typeof event.step !== 'number' || event.step <= (previous.steps[previous.steps.length - 1] ?? -1)) return previous;
  const lr = Object.fromEntries([...new Set([...Object.keys(previous.lr), ...Object.keys(event.lr || {})])].map((group) => [
    group, [...(previous.lr[group] || Array(previous.steps.length).fill(null)), event.lr?.[group] ?? null],
  ]));
  return { ...previous, lr, steps: [...previous.steps, event.step], loss: [...previous.loss, event.loss ?? null],
    loss_ema: [...previous.loss_ema, event.loss_ema ?? null], grad_norm: [...previous.grad_norm, event.grad_norm ?? null],
    vram_mb: [...previous.vram_mb, event.vram_mb ?? null],
    vram_metric: event.vram_metric ?? previous.vram_metric,
    it_s: [...previous.it_s, event.it_s ?? null],
    ...gpuSeries(previous, event), gpu_devices: appendGpuDeviceStep(previous, event.gpu_devices) };
}

const GPU_KEYS = ['gpu_power_w', 'gpu_temp_c', 'gpu_util_pct'] as const;

/** Driver readings start empty; a series appears with the first step that reports it. */
function gpuSeries(previous: JobMetrics, event: Record<string, any>): Pick<JobMetrics, typeof GPU_KEYS[number]> {
  return Object.fromEntries(GPU_KEYS.map(key => {
    const values = previous[key] || [];
    const value = typeof event[key] === 'number' ? event[key] : null;
    if (!values.length && value == null) return [key, values];
    return [key, [...(values.length ? values : Array(previous.steps.length).fill(null)), value]];
  })) as Pick<JobMetrics, typeof GPU_KEYS[number]>;
}

export interface NamedSeries {
  name: string;
  data: Array<[number, number]>;
}

/**
 * 把 validation 点列表整形成 ECharts 序列：
 * 每个固定时间步 t 一条线 + 均值线，x 轴为 step。
 * 输入乱序/重复 step 时按 step 升序去重（保留最后一个）。
 */
export function shapeValidationSeries(validation: ValidationPoint[]): {
  steps: number[];
  series: NamedSeries[];
} {
  const byStep = new Map<number, ValidationPoint>();
  for (const v of validation) {
    byStep.set(v.step, v);
  }
  const steps = Array.from(byStep.keys()).sort((a, b) => a - b);

  const tKeys = new Set<string>();
  for (const s of steps) {
    for (const k of Object.keys(byStep.get(s)!.per_t || {})) {
      tKeys.add(k);
    }
  }
  const sortedTKeys = Array.from(tKeys).sort((a, b) => Number(a) - Number(b));

  const series: NamedSeries[] = sortedTKeys.map((t) => ({
    name: `t=${t}`,
    data: steps
      .filter((s) => byStep.get(s)!.per_t[t] != null)
      .map((s) => [s, byStep.get(s)!.per_t[t]] as [number, number]),
  }));

  series.push({
    name: 'mean',
    data: steps.map((s) => [s, byStep.get(s)!.mean] as [number, number]),
  });

  return { steps, series };
}

/** SSE job.validation 增量合并：按 step 去重追加，保持升序 */
export function mergeValidationPoint(
  validation: ValidationPoint[],
  point: ValidationPoint
): ValidationPoint[] {
  const exists = validation.some((v) => v.step === point.step);
  if (exists) return validation;
  return [...validation, point].sort((a, b) => a.step - b.step);
}

/** 日志环形追加：保持最多 cap 行，超出丢弃最旧的 */
export function appendCapped<T>(lines: T[], incoming: T[], cap = 50000): T[] {
  const next = lines.concat(incoming);
  if (next.length <= cap) return next;
  return next.slice(next.length - cap);
}
