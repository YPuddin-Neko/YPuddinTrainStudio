import type { Job } from '../api/types';

export const ACTIVE_JOB_STATUSES = 'running,pausing,cancelling';

/** Merge partial events, preserving totals and ignoring replayed training steps. */
export function mergeJobEvent(job: Job, event: Record<string, any>): Job {
  if (job.id !== event.job_id) return job;
  const progress = { ...(job.progress || {}), ...(event.progress || {}) };
  if (event.phase !== undefined) progress.phase = event.phase;
  for (const key of ['total_steps', 'steps_per_epoch', 'message', 'wait_reason']) {
    if (event[key] !== undefined) (progress as any)[key] = event[key];
  }
  const currentStep = job.progress?.step ?? -1;
  if (typeof event.done === 'number' && event.done >= (job.progress?.done ?? -1)) {
    for (const key of ['done', 'total', 'cell_index', 'sample_step', 'sample_steps']) {
      if (event[key] !== undefined) (progress as any)[key] = event[key];
    }
  }
  if (typeof event.step === 'number' && event.step >= currentStep) {
    progress.phase = event.phase ?? 'training';
    for (const key of ['step', 'epoch', 'eta_s', 'it_s', 'vram_metric']) {
      if (event[key] !== undefined) (progress as any)[key] = event[key];
    }
    if (event.vram_mb != null) {
      progress.vram_peak_mb = progress.vram_metric === 'current_allocated'
        ? event.vram_mb : Math.max(progress.vram_peak_mb || 0, event.vram_mb);
    }
  }
  const latest = { ...(job.latest || {}) };
  if (event.step == null || event.step >= currentStep) {
    for (const key of ['loss', 'loss_ema', 'lr', 'loss_mean', 'loss_count', 'loss_mean_scope']) {
      if (event[key] !== undefined) (latest as any)[key] = event[key];
    }
  }
  return { ...job, progress, latest, ...(event.status ? { status: event.status } : {}),
    ...(event.error !== undefined ? { error: event.error } : {}),
    ...(event.exit_code !== undefined ? { exit_code: event.exit_code } : {}) };
}

type Text = (zh: string, en: string) => string;

/** The run a project page should describe: the one holding a device, then paused, then waiting, then the newest. */
export function focusJob<T extends Pick<Job, 'status'>>(jobs: T[]): T | undefined {
  return jobs.find(job => ['running', 'pausing', 'cancelling'].includes(job.status)) || jobs.find(job => job.status === 'paused')
    || jobs.find(job => ['queued', 'scheduled'].includes(job.status)) || jobs[0];
}

export function artifactKindLabel(kind: string, text: Text): string {
  return kind === 'model' ? text('完整模型', 'Full model') : kind === 'adapter' || kind === 'lora' ? text('适配器权重', 'Adapter weights') : kind === 'checkpoint' ? text('检查点', 'Checkpoint') : kind;
}

export function jobTypeLabel(job: Pick<Job, 'type' | 'training_mode'>, text: Text): string {
  if (job.type === 'tts_train') return text('语音 LoRA 训练', 'Speech LoRA training');
  if (job.type === 'tts_sample') return text('语音试听', 'Speech preview');
  if (job.type === 'xyz') return text('模型测试', 'Model testing');
  if (job.type === 'cache') return text('缓存', 'Cache');
  if (job.training_mode === 'full') return text('全量微调', 'Full fine-tuning');
  if (job.training_mode === 'adapter') return text('LoRA 训练', 'LoRA training');
  return text('训练', 'Training');
}

/** Month, day and time; the year only when it is not the current one. */
export function shortTime(ts: number | null | undefined): string {
  if (ts == null) return '—';
  const date = new Date(ts * 1000);
  return date.toLocaleString(undefined, { ...(date.getFullYear() !== new Date().getFullYear() ? { year: 'numeric' } : {}), month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false });
}
