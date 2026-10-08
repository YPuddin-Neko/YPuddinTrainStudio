import type { components } from './generated';
import { apiClient } from './client';
import type { TtsEngine } from './tts';

export type TtsPreset = components['schemas']['TtsPreset'];
export type TtsPresetConfig = TtsPreset['config'];
export type TtsPresetDocument = components['schemas']['TtsPresetDocument'];
export type TtsPresetResolveBody = components['schemas']['TtsPresetResolveBody'];
export type TtsPresetResolved = components['schemas']['TtsPresetResolveResponse'];
// Request defaults allow portable parameters without local environment fields.
export type TtsPresetCreate = Omit<components['schemas']['TtsPresetCreateBody'], 'config'> & { config: TtsPresetConfig };
export type TtsPresetUpdate = TtsPresetCreate & Pick<components['schemas']['TtsPresetUpdateBody'], 'expected_revision'>;
const route = (id: string) => `/tts/presets/${encodeURIComponent(id)}`;
export const ttsPresetsApi = {
  list: (engine: TtsEngine, signal?: AbortSignal) => apiClient.get<TtsPreset[]>('/tts/presets', { params: { engine }, signal, silent: true }),
  defaults: (engine: TtsEngine, signal?: AbortSignal) => apiClient.get<TtsPresetConfig>('/tts/presets/defaults', { params: { engine }, signal, silent: true }),
  get: (id: string, signal?: AbortSignal) => apiClient.get<TtsPreset>(route(id), { signal, silent: true }),
  create: (body: TtsPresetCreate) => apiClient.post<TtsPreset>('/tts/presets', body, { silent: true }),
  update: (id: string, body: TtsPresetUpdate) => apiClient.put<TtsPreset>(route(id), body, { silent: true }),
  remove: (id: string, revision: number) => apiClient.delete<components['schemas']['TtsPresetDeleted']>(route(id), { params: { expected_revision: revision }, silent: true }),
  import: (body: TtsPresetDocument) => apiClient.post<TtsPreset>('/tts/presets/import', body, { silent: true }),
  export: (id: string) => apiClient.get<TtsPresetDocument>(`${route(id)}/export`, { silent: true }),
  resolve: (id: string, body: TtsPresetResolveBody) => apiClient.post<TtsPresetResolved>(`${route(id)}/resolve`, body, { silent: true }),
};
