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
    for (const key of ['loss', 'loss_ema', 'lr']) {
      if (event[key] !== undefined) (latest as any)[key] = event[key];
    }
  }
  return { ...job, progress, latest, ...(event.status ? { status: event.status } : {}),
    ...(event.error !== undefined ? { error: event.error } : {}) };
}
