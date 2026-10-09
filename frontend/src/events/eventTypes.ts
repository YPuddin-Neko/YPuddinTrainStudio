export const EVENT_TYPES = {
  SYSTEM_STATS: 'system.stats',
  JOB_STATE: 'job.state',
  JOB_PHASE: 'job.phase',
  JOB_CACHE_PROGRESS: 'job.cache_progress',
  JOB_STEP: 'job.step',
  JOB_VALIDATION: 'job.validation',
  JOB_SAMPLE: 'job.sample',
  JOB_SAMPLE_PROGRESS: 'job.sample_progress',
  JOB_XYZ_PROGRESS: 'job.xyz_progress',
  XYZ_MODELS: 'xyz.models',
  JOB_CHECKPOINT: 'job.checkpoint',
  JOB_WARNING: 'job.warning',
  JOB_EVENT: 'job.event',
  JOB_LOG: 'job.log',
  QUEUE_CHANGED: 'queue.changed',
  DATASET_CHANGED: 'dataset.changed',
  TTS_SOURCE_CHANGED: 'tts.source.changed',
  ARTIFACT_CREATED: 'artifact.created',
  BACKGROUND_CHANGED: 'background.changed',
  MODEL_DOWNLOAD: 'model.download',
  TTS_MODEL_DOWNLOAD: 'tts.model_download',
} as const;

export type EventType = typeof EVENT_TYPES[keyof typeof EVENT_TYPES];

export interface EventMessage<T = any> {
  type: EventType;
  data: T;
  id: string;
}
