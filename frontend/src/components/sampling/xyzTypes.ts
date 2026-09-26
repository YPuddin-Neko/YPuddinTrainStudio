export type AxisKey = 'steps' | 'cfg' | 'seed' | 'sampler' | 'scheduler' | 'shift' | 'adapter_scale' | 'checkpoint';
export type AxisValue = number | string;
export interface XyzAxis { key: AxisKey; values: AxisValue[] }
export interface SamplingValues {
  prompt: string; negative: string; width: number; height: number; steps: number; cfg: number;
  seed: number; sampler: string; scheduler: string; shift: number | null; guidance: number | null;
  adapter_scale: number; checkpoint_id: string | null; sampling_model_id: string | null;
}
export interface XyzRequest extends SamplingValues { gpu_devices?: string[]; name?: string; x: XyzAxis; y?: XyzAxis | null; z?: XyzAxis | null }
export interface XyzOptions {
  source_job_id?: string | null; family: string; training_mode?: 'adapter' | 'full'; defaults: SamplingValues;
  axes: { key: AxisKey; label: string; values?: AxisValue[] }[];
  /** Products of every training run in the version that samples on the same base model, the source's first. */
  checkpoints: { id: string; name: string; step: number; job_id?: string; job_name?: string }[];
  /** Runs of that version left out, and why. */
  excluded_jobs?: { id: string; name: string; reason: string }[];
  sampling_models: { id: string; name: string; variant?: string }[];
  limits: { max_cells: number; max_axis_values?: number; max_pixels?: number };
}
export interface XyzCell {
  index: number; x: number; y: number; z: number; x_value: AxisValue; y_value: AxisValue | null; z_value: AxisValue | null;
  seed: number; steps: number; cfg: number; sampler: string; scheduler: string; shift: number | null;
  adapter_scale: number; checkpoint_id: string | null; file: string; url: string;
}
export interface XyzTask {
  id: string; job_id: string; source_job_id: string; status: string; phase: string; done: number; total: number;
  cell_index?: number | null; sample_step?: number | null; sample_steps?: number | null;
  error: string | null; created_at: number; finished_at: number | null; request: XyzRequest; can_cancel: boolean;
  manifest: { cells: XyzCell[]; grids: { z: number; z_value: AxisValue | null; file: string; url: string }[]; complete: boolean };
}

export const axisNames: Record<AxisKey, [string, string]> = {
  steps: ['采样步数', 'Steps'], cfg: ['CFG 引导强度', 'CFG'], seed: ['随机种子', 'Seed'],
  sampler: ['采样器', 'Sampler'], scheduler: ['调度器', 'Scheduler'], shift: ['时间步偏移', 'Shift'],
  adapter_scale: ['LoRA 强度', 'LoRA strength'], checkpoint: ['训练权重', 'Checkpoint'],
};
export function parseAxis(key: AxisKey, raw: string): XyzAxis {
  const tokens = raw.split(/[,，\n]+/).map(value => value.trim()).filter(Boolean);
  const numeric = !['sampler', 'scheduler', 'checkpoint'].includes(key);
  return { key, values: tokens.map(value => numeric ? Number(value) : value) };
}
export function axisCount(axes: (XyzAxis | null | undefined)[]) {
  return axes.reduce((total, axis) => total * (axis ? axis.values.length : 1), 1);
}

/** Runs of one version name their products alike, so a choice spanning several runs names the run too. */
export function checkpointLabel(options: Pick<XyzOptions, 'checkpoints'>, checkpoint: XyzOptions['checkpoints'][number]) {
  const runs = new Set(options.checkpoints.map(cp => cp.job_id));
  return runs.size > 1 && checkpoint.job_name ? `${checkpoint.job_name} · ${checkpoint.name}` : checkpoint.name;
}
