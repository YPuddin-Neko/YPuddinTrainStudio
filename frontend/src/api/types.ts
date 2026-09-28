import type { components } from './generated';

type S = components['schemas'];

// ---- openapi 覆盖的核心实体：直接从 generated 导出别名 ----
export type SystemStats = S['SystemStats'];
export type GpuStats = S['GpuStats'];
export type Project = S['Project'];
export type ProjectPage = S['ProjectPage'];
export type ProjectCategories = S['ProjectCategories'];
export type Job = S['Job'];
export type JobProgress = S['JobProgress'];
export type JobMetrics = Pick<S['JobMetrics'], 'steps' | 'loss' | 'loss_ema' | 'grad_norm' | 'vram_mb' | 'vram_metric' | 'it_s' | 'validation' | 'gpu_power_w' | 'gpu_temp_c' | 'gpu_util_pct'> & {
  lr: Record<string, Array<number | null>>;
};
export type JobSample = S['JobSample'];
export type JobCheckpoint = S['JobCheckpoint'];
export type JobLogResponse = S['JobLog'];
export type JobLogLine = S['LogLine'];
export type QueueSettings = S['QueueSettings'];
export type QueueDevice = S['QueueDevice'];
export type QueueDevices = S['QueueDevices'];
export type FsListResponse = S['FsList'];
export type FsListEntry = S['FsEntry'];
export type Artifact = S['Artifact'];
export type ModelAsset = S['ModelAsset'];
export type ModelDownload = S['ModelDownload'];
export type ModelDownloadRequest = S['ModelDownloadRequest'];
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
export type VisionCatalog = S['VisionCatalog'];
export type VisionModel = S['VisionModel'];
export type TaggingOptions = S['TaggingOptions'];
export type AutoMaskOptions = S['AutoMaskOptions'];
export type ValidationPoint = S['ValidationPoint'];
export type FamilyInfo = S['FamilyInfo'];
export type FamilyPreset = S['FamilyPreset'];

// ---- openapi 未覆盖（或形状不便引用）的本地类型：手写保留 ----

// GPU 统计：在 generated GpuStats 上扩展 Apple MPS 适配字段
export type GpuInfo = S['GpuStats'] & {
  kind?: 'cuda' | 'mps' | string;
  power_w?: number | null;
};

export interface SystemInfo {
  python?: string;
  platform?: string;
  packages?: Record<string, string | null>;
  ypuddin?: string;
  cuda?: string | null;
  cuda_available?: boolean;
  [key: string]: unknown;
}

/** 判断是否 Apple Silicon 平台（后端 /system/info.platform 形如 macOS-15.7.9-arm64-arm-64bit） */
export function isAppleSilicon(info: SystemInfo | null | undefined): boolean {
  const p = (info?.platform || '').toLowerCase();
  const isMac = p.includes('darwin') || p.includes('macos') || p.includes('mac os');
  return isMac && (p.includes('arm64') || p.includes('arm'));
}

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

export type JobSampleEvent = JobSample & { job_id: string };

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
  vram_metric?: string | null;
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
