import { apiClient, apiUrl } from './client';
import type { Job } from './types';
import type { components } from './generated';

export type TtsVersionConfig = components['schemas']['TtsVersionConfig'];
export type TtsConfigResponse = components['schemas']['TtsConfigResponse'];
export type GptSovitsVersionConfig = components['schemas']['GptSovitsVersionConfig'];
export type GptSovitsSampleOptions = components['schemas']['GptSovitsSampleOptions'];
export type TtsScopedConfig = TtsConfigResponse['config'];
export type TtsEngine = TtsScopedConfig['engine'];
export type TtsGptSovitsDatasetReport = components['schemas']['TtsGptSovitsDatasetReport'];
export type TtsGptSovitsEnvironmentReport = components['schemas']['TtsGptSovitsEnvironmentReport'];
export type TtsConfigSaveBody = components['schemas']['TtsConfigSaveBody'];
export type TtsIssue = components['schemas']['TtsIssue'];
export type TtsCapabilities = components['schemas']['TtsCapabilities'];
export type TtsEngineCapability = components['schemas']['TtsEngineCapability'];
export type TtsTrainSchema = components['schemas']['TtsTrainSchema'];
export type TtsSource = components['schemas']['TtsSource'];
export type TtsSourcesResponse = components['schemas']['TtsSourcesResponse'];
export type TtsSourcePutBody = components['schemas']['TtsSourcePutBody'];
export type TtsSourceCheckBody = components['schemas']['TtsSourceCheckBody'];
export type TtsSourceRow = components['schemas']['TtsSourceRow'];
export type TtsRowsResponse = components['schemas']['TtsRowsResponse'];
export type TtsSourceChanged = components['schemas']['TtsSourceChanged'];
export type TtsValidationBody = components['schemas']['TtsValidationBody'];
export type TtsValidationReport = components['schemas']['TtsValidationReport'];
export type TtsTrainingBody = components['schemas']['TtsTrainingBody'];
export type TtsSplit = TtsSource['split'];
export type TtsCheckpoint = components['schemas']['TtsCheckpoint'];
export type TtsCheckpointPage = components['schemas']['TtsCheckpointPage'];
export type TtsCheckpointFile = components['schemas']['TtsCheckpointFile'];
type GeneratedSampleBody = components['schemas']['TtsSampleBody'];
// Defaults are optional on the wire; GSV requests omit the Vox-specific options.
export type TtsSampleBody = Omit<GeneratedSampleBody, 'cfg_value' | 'inference_timesteps'> & Partial<Pick<GeneratedSampleBody, 'cfg_value' | 'inference_timesteps'>>;
export type TtsSampleJob = components['schemas']['TtsSampleJob'];
export type TtsSampleJobPage = components['schemas']['TtsSampleJobPage'];
export type TtsAudio = components['schemas']['TtsAudio'];
export type TtsSampleRequestSnapshot = components['schemas']['TtsSampleRequestSnapshot'];
export type TtsSampleSource = components['schemas']['TtsSampleSource'];
export type TtsPageParams = { cursor?: string; limit?: number; include_archived?: boolean };

export function ttsResourceUrl(url: string | null | undefined): string | undefined {
  if (!url || !/^\/api\/tts\/jobs\/[^/?#]+\/(?:audio\/[^/?#]+|checkpoints\/[^/?#]+\/files\/[^/?#]+)$/.test(url)) return undefined;
  try {
    if (url.slice(1).split('/').some(segment => {
      const value = decodeURIComponent(segment);
      return !value || value === '.' || value === '..' || value.includes('/') || value.includes('\\') || [...value].some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127);
    })) return undefined;
  } catch { return undefined; }
  return apiUrl(url.slice(4));
}


export const ttsVersionUrl = (projectId: string, versionId: string) => `/tts/projects/${encodeURIComponent(projectId)}/versions/${encodeURIComponent(versionId)}`;
export const ttsVersionConfigUrl = (projectId: string, versionId: string) => `${ttsVersionUrl(projectId, versionId)}/config`;
export const ttsSourceUrl = (projectId: string, versionId: string, sourceId: string) => `${ttsVersionUrl(projectId, versionId)}/sources/${encodeURIComponent(sourceId)}`;
export function ttsAudioUrl(url: string | null): string | undefined {
  if (!url || !url.startsWith('/api/tts/projects/') || !url.split('?')[0].endsWith('/audio')) return undefined;
  return apiUrl(url.slice(4));
}

export interface TtsConfig {
  engine: 'voxcpm1.5';
  python_path: string;
  trainer_path: string;
  model_path: string;
  train_manifest: string;
  val_manifest: string;
  batch_size: number;
  grad_accum_steps: number;
  num_workers: number;
  num_iters: number;
  save_interval: number;
  learning_rate: number;
  warmup_steps: number;
  lora_rank: number;
  lora_alpha: number;
}

export const isTtsJob = (job: { type: string } | null | undefined) => job?.type === 'tts_train' || job?.type === 'tts_sample';

export const ttsApi = {
  capabilities: (signal?: AbortSignal) => apiClient.get<TtsCapabilities>('/tts/capabilities', { signal, silent: true }),
  trainSchema: (signal?: AbortSignal, engine: TtsEngine = 'voxcpm1.5') => apiClient.get<TtsTrainSchema>('/tts/schema/train', { params: { engine }, signal, silent: true }),
  versionConfig: (projectId: string, versionId: string, signal?: AbortSignal) => apiClient.get<TtsConfigResponse>(ttsVersionConfigUrl(projectId, versionId), { signal, silent: true }),
  saveVersionConfig: (projectId: string, versionId: string, body: TtsConfigSaveBody) => apiClient.put<TtsConfigResponse>(ttsVersionConfigUrl(projectId, versionId), body, { silent: true }),
  sources: (pid: string, vid: string, signal?: AbortSignal) => apiClient.get<TtsSourcesResponse>(`${ttsVersionUrl(pid, vid)}/sources`, { signal, silent: true }),
  putSource: (pid: string, vid: string, split: TtsSplit, body: TtsSourcePutBody) => apiClient.put<TtsSourcesResponse>(`${ttsVersionUrl(pid, vid)}/sources/${split}`, body, { silent: true }),
  removeSource: (pid: string, vid: string, split: TtsSplit, expectedDataRevision: number) => apiClient.delete<TtsSourcesResponse>(`${ttsVersionUrl(pid, vid)}/sources/${split}`, { params: { expected_data_revision: expectedDataRevision }, silent: true }),
  source: (pid: string, vid: string, sid: string, signal?: AbortSignal) => apiClient.get<TtsSource>(ttsSourceUrl(pid, vid, sid), { signal, silent: true }),
  checkSource: (pid: string, vid: string, sid: string, body: TtsSourceCheckBody) => apiClient.post<TtsSource>(`${ttsSourceUrl(pid, vid, sid)}/check`, body, { silent: true }),
  rows: (pid: string, vid: string, sid: string, params: { snapshot_id: string; page: number; page_size: number }, signal?: AbortSignal) => apiClient.get<TtsRowsResponse>(`${ttsSourceUrl(pid, vid, sid)}/rows`, { params, signal, silent: true }),
  validateVersion: (pid: string, vid: string, body: TtsValidationBody, signal?: AbortSignal) => apiClient.post<TtsValidationReport>(`${ttsVersionUrl(pid, vid)}/validate`, body, { signal, silent: true }),
  startVersion: (pid: string, vid: string, body: TtsTrainingBody, idempotencyKey: string) => apiClient.post<Job>(`${ttsVersionUrl(pid, vid)}/jobs`, body, { headers: { 'Idempotency-Key': idempotencyKey }, timeout: 120_000, silent: true }),
  config: (signal?: AbortSignal) => apiClient.get<TtsConfig>('/tts/config', { signal, silent: true }),
  checkpoints: (id: string, signal?: AbortSignal) => apiClient.get<TtsCheckpoint[]>(`/tts/jobs/${encodeURIComponent(id)}/checkpoints`, { signal, silent: true }),
  versionCheckpoints: (pid: string, vid: string, params: TtsPageParams = {}, signal?: AbortSignal) => apiClient.get<TtsCheckpointPage>(`${ttsVersionUrl(pid, vid)}/checkpoints`, { params, signal, silent: true }),
  sampleJobs: (jobId: string, params: TtsPageParams = {}, signal?: AbortSignal) => apiClient.get<TtsSampleJobPage>(`/tts/jobs/${encodeURIComponent(jobId)}/sample-jobs`, { params, signal, silent: true }),
  versionSampleJobs: (pid: string, vid: string, params: TtsPageParams = {}, signal?: AbortSignal) => apiClient.get<TtsSampleJobPage>(`${ttsVersionUrl(pid, vid)}/sample-jobs`, { params, signal, silent: true }),
  sampleJob: (id: string, signal?: AbortSignal) => apiClient.get<TtsSampleJob>(`/tts/sample-jobs/${encodeURIComponent(id)}`, { signal, silent: true }),
  createSample: (jobId: string, body: TtsSampleBody, key: string) => apiClient.post<Job>(`/tts/jobs/${encodeURIComponent(jobId)}/samples`, body, { headers: { 'Idempotency-Key': key }, timeout: 120_000, silent: true }),
};
