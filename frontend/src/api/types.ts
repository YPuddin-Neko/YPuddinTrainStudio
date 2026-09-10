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

export interface GpuStats {
  index: number;
  name?: string;
  util_pct: number;
  mem_used_mb: number;
  mem_total_mb: number;
  temp_c: number;
}

export interface SystemStats {
  cpu_pct: number;
  ram: {
    used_mb: number;
    total_mb: number;
  };
  disks: Array<{
    path: string;
    used_gb: number;
    total_gb: number;
  }>;
  gpus: GpuStats[];
}

export interface Project {
  id: string;
  name: string;
  note?: string;
  created_at: string | number;
  updated_at: string | number;
  archived: boolean;
  dataset_ids: string[];
  stats: {
    jobs: number;
    artifacts: number;
  };
}

export interface JobProgress {
  phase: string;
  step: number;
  total_steps: number;
  steps_per_epoch: number;
  epoch: number;
  eta_s: number | null;
  it_s: number;
  vram_peak_mb: number | null;
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

export interface Job {
  id: string;
  type: 'train' | 'cache' | 'sample' | 'convert';
  name: string;
  project_id?: string | null;
  status: JobStatus;
  priority: number;
  scheduled_at: string | number | null;
  created_at: string | number;
  started_at: string | number | null;
  finished_at: string | number | null;
  progress: Partial<JobProgress>;
  latest: {
    loss?: number;
    loss_ema?: number;
    lr?: Record<string, number>;
  };
  error: string | null;
  resume_from: string | null;
  artifact_ids?: string[];
}

export interface JobListResponse {
  items: Job[];
  total: number;
  page: number;
  page_size: number;
}

export interface JobMetrics {
  steps: number[];
  loss: number[];
  loss_ema: number[];
  lr: Record<string, number[]>;
  grad_norm: number[];
  vram_mb: number[];
  it_s: number[];
  validation: Array<{
    step: number;
    per_t: Record<string, number>;
    mean: number;
  }>;
}

export interface JobSample {
  step: number;
  prompt_index: number;
  prompt: string;
  seed: number;
  url: string;
  width: number;
  height: number;
  created_at: string | number;
}

export interface JobCheckpoint {
  step: number;
  kind: 'weights' | 'full';
  path: string;
  size: number;
  created_at: string | number;
  artifact_id?: string;
}

export interface JobLogLine {
  ts: string;
  level: 'info' | 'warn' | 'error' | 'debug';
  msg: string;
}

export interface JobLogResponse {
  lines: JobLogLine[];
  next_offset: number;
}

export interface QueueSettings {
  held: boolean;
  max_concurrent: number;
}

export interface FsListEntry {
  name: string;
  is_dir: boolean;
  size: number;
  mtime: string;
}

export interface FsListResponse {
  path: string;
  parent: string | null;
  entries: FsListEntry[];
}

export interface Artifact {
  id: string;
  project_id: string | null;
  job_id: string;
  name: string;
  path: string;
  size: number;
  kind?: string;
  step?: number;
  algo: string;
  rank: number | string;
  alpha: number;
  factor: number;
  family: string;
  created_at: string | number;
  metadata: Record<string, any>;
}

export interface ModelAsset {
  id: string;
  family: string;
  kind: 'dit' | 'text_encoder' | 'vae' | 'tokenizer';
  path: string;
  size: number;
  dtype: string;
  exists: boolean;
  is_default: boolean;
}

export interface Settings {
  paths: {
    data_root: string;
    cache_dir: string;
    models_dir: string;
    output_dir: string;
  };
  server: {
    host: string;
    port: number;
  };
  ui: {
    language: string;
    theme: string;
  };
}

export interface Plan {
  ok: boolean;
  errors: Array<{ loc?: string; msg: string }>;
  warnings: Array<{ code?: string; msg: string }>;
  images?: number;
  items?: number;
  captioned?: number;
  steps_per_epoch: number;
  total_steps: number;
  epochs: number;
  buckets: Array<{ w: number; h: number; items?: number; images?: number; batches: number }>;
  params: { trainable: number; base: number; adapted_layers?: number; by_algo?: Record<string, number> };
  memory: {
    weights_mb: number;
    adapter_mb: number;
    optimizer_mb: number;
    activations_mb_by_bucket: Array<{ w: number; h: number; mb: number }>;
    peak_mb_estimate: number;
    gpu_total_mb: number | null;
    suggestions: string[];
  };
  text_encoding: 'online' | 'cached';
  eta_estimate_s?: number | null;
}

export interface Preset {
  name: string;
  description: string;
  config: Record<string, any>;
  builtin: boolean;
  updated_at: string | number | null;
}

// ---- Datasets ----

export interface DatasetSource {
  id: string;
  project_id: string;
  path: string;
  repeats: number;
  caption_ext: string;
  is_reg: boolean;
  prior_weight: number;
  class_prompt: string | null;
  created_at: number;
}

export interface DatasetStats {
  images?: number;
  captioned?: number;
  avg_tags?: number;
  resolutions?: Array<{ w: number; h: number; count: number }>;
  ar_hist?: Array<{ ar: string; count: number }>;
  masks?: number;
}

export interface DatasetInfo {
  source: DatasetSource;
  stats: DatasetStats;
  index_status: 'ready' | 'indexing' | 'stale' | 'failed';
  cache: {
    latents?: { cached: number; total: number };
    text?: { cached: number; total: number };
  };
}

export interface DatasetImage {
  hash: string;
  rel_path: string;
  width: number;
  height: number;
  caption: string;
  has_mask: boolean;
}

export interface DatasetImagesPage {
  items: DatasetImage[];
  total: number;
  page: number;
  page_size: number;
}

export interface SampleProgressEvent {
  job_id: string;
  step: number;
  prompt_index: number;
  prompts: number;
  done: number;
  total: number;
}
