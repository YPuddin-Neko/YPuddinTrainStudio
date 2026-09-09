export const EVENT_TYPES = {
  SYSTEM_STATS: 'system.stats',
  JOB_STATE: 'job.state',
  JOB_PHASE: 'job.phase',
  JOB_CACHE_PROGRESS: 'job.cache_progress',
  JOB_STEP: 'job.step',
  JOB_VALIDATION: 'job.validation',
  JOB_SAMPLE: 'job.sample',
  JOB_CHECKPOINT: 'job.checkpoint',
  JOB_WARNING: 'job.warning',
  JOB_LOG: 'job.log',
  QUEUE_CHANGED: 'queue.changed',
  DATASET_CHANGED: 'dataset.changed',
} as const;

export type EventType = typeof EVENT_TYPES[keyof typeof EVENT_TYPES];

export interface EventMessage<T = any> {
  type: EventType;
  data: T;
  id: string;
}
