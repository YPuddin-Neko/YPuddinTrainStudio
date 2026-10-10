import { apiClient, READ_TIMEOUT_MS } from './client';
import type { components } from './generated';
import type { TtsEngine } from './tts';

export type TtsEnvironmentSnapshot = components['schemas']['TtsEnvironmentSnapshot'];
export type TtsEnvironmentOperation = components['schemas']['TtsEnvironmentOperation'];
export type TtsManagedEnvironment = components['schemas']['TtsManagedEnvironment'];
export type TtsPythonCandidate = components['schemas']['TtsPythonCandidate'];
export type TtsEnvironmentCheckRequest = components['schemas']['TtsEnvironmentCheckRequest'];
export type TtsEnvironmentPrepareRequest = components['schemas']['TtsEnvironmentPrepareRequest'];
export type TtsEnvironmentChanged = { operation_id: string | null; engine: TtsEngine | null; status: TtsEnvironmentOperation['status'] | null };

export const ttsEnvironmentKeys = {
  snapshot: ['tts-environments'] as const,
  operation: (id: string) => ['tts-environment-operation', id] as const,
  write: ['tts-environment-write'] as const,
};
export const ttsEnvironmentActive = (operation: TtsEnvironmentOperation) => operation.status === 'queued' || operation.status === 'running';
const operationUrl = (id: string) => `/tts/environments/operations/${encodeURIComponent(id)}`;
export const ttsEnvironmentsApi = {
  snapshot: (signal?: AbortSignal) => apiClient.get<TtsEnvironmentSnapshot>('/tts/environments', { signal, silent: true, timeout: READ_TIMEOUT_MS }),
  check: (body: TtsEnvironmentCheckRequest) => apiClient.post<TtsEnvironmentOperation>('/tts/environments/checks', body, { silent: true, timeout: READ_TIMEOUT_MS }),
  prepare: (body: TtsEnvironmentPrepareRequest) => apiClient.post<TtsEnvironmentOperation>('/tts/environments/preparations', body, { silent: true, timeout: READ_TIMEOUT_MS }),
  operation: (id: string, signal?: AbortSignal) => apiClient.get<TtsEnvironmentOperation>(operationUrl(id), { signal, silent: true, timeout: READ_TIMEOUT_MS }),
  cancel: (id: string) => apiClient.post<TtsEnvironmentOperation>(`${operationUrl(id)}/cancel`, {}, { silent: true, timeout: READ_TIMEOUT_MS }),
  retry: (id: string) => apiClient.post<TtsEnvironmentOperation>(`${operationUrl(id)}/retry`, {}, { silent: true, timeout: READ_TIMEOUT_MS }),
  setDefault: (engine: TtsEngine, environmentId: string) => apiClient.put<TtsEnvironmentSnapshot>(`/tts/environments/defaults/${encodeURIComponent(engine)}`, { environment_id: environmentId }, { silent: true, timeout: READ_TIMEOUT_MS }),
};
