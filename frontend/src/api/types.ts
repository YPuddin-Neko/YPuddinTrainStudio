import type { components } from './generated';

type S = components['schemas'];

// ---- openapi 覆盖的核心实体：直接从 generated 导出别名 ----
export type SystemStats = S['SystemStats'];
export type GpuStats = S['GpuStats'];
export type Project = S['Project'];
export type Job = S['Job'];
export type JobProgress = S['JobProgress'];
export type JobMetrics = S['JobMetrics'];
export type JobSample = S['JobSample'];
export type JobCheckpoint = S['JobCheckpoint'];
export type JobLogResponse = S['JobLog'];
export type JobLogLine = S['LogLine'];
export type QueueSettings = S['QueueSettings'];
export type FsListResponse = S['FsList'];
export type FsListEntry = S['FsEntry'];
export type Artifact = S['Artifact'];
export type ModelAsset = S['ModelAsset'];
export type Settings = S['Settings'];
export type Plan = S['Plan'];
export type Preset = S['Preset'];
export type DatasetSource = S['DatasetSource'];
export type DatasetStats = S['DatasetStats'];

// DatasetInfo.cache 在 openapi 中是宽松 map，收窄为已知形状
export interface DatasetCacheInfo {
  latents?: { cached: number; total: number };
  text?: { cached: number; total: number };
  cache_dir?: string;
}
export type DatasetInfo = S['DatasetInfo'] & { cache: DatasetCacheInfo };

export type DatasetImage = S['DatasetImage'];
export type DatasetImagesPage = S['ImagePage'];
export type ValidationPoint = S['ValidationPoint'];

// ---- openapi 未覆盖（或形状不便引用）的本地类型：手写保留 ----

export type JobStatus =
  | 'queued'
  | 'scheduled'
  | 'running'
  | 'pausing'
  | 'cancelling'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled';

export type JobListResponse = S['JobPage'];

export interface ApiErrorPayload {
  error: {
    code: string;
    message: string;
    trace_id?: string;
    details?: Record<string, any>;
  };
}

export class ApiError extends Error {
  code: string;
  traceId?: string;
  details?: Record<string, any>;
  status: number;

  constructor(status: number, payload: ApiErrorPayload['error']) {
    super(payload.message || 'API request failed');
    this.name = 'ApiError';
    this.status = status;
    this.code = payload.code || 'unknown_error';
    this.traceId = payload.trace_id;
    this.details = payload.details;
  }
}

// ---- SSE 事件 payload（openapi 未覆盖） ----

export interface SampleProgressEvent {
  job_id: string;
  step: number;
  prompt_index: number;
  prompts: number;
  done: number;
  total: number;
}

export interface JobStepEvent {
  job_id: string;
  step: number;
  epoch: number;
  loss: number;
  loss_ema: number;
  lr: Record<string, number>;
  grad_norm: number;
  it_s: number;
  vram_mb: number | null;
  eta_s: number | null;
}

export interface JobStateEvent {
  job_id: string;
  status: JobStatus;
  progress?: Partial<JobProgress>;
  error?: string | null;
}

export interface JobValidationEvent {
  job_id: string;
  step: number;
  per_t: Record<string, number>;
  mean: number;
}

export interface CacheProgressEvent {
  job_id: string;
  kind: 'latents' | 'text' | 'index';
  done: number;
  total: number;
}
