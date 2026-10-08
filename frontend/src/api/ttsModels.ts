import { apiClient, READ_TIMEOUT_MS } from './client';
import type { components } from './generated';

export type TtsModelPackage = components['schemas']['TtsModelPackage'];
export type TtsInstalledModel = components['schemas']['TtsInstalledModel'];
export type TtsModelDownload = components['schemas']['TtsModelDownload'];
export type TtsModelEngine = TtsModelPackage['engine'];
export type TtsModelVariant = TtsModelPackage['variant'];

export function ttsModelIssueText(issue: components['schemas']['TtsModelIssue'], text: (zh: string, en: string) => string): string {
  const known: Record<string, [string, string]> = {
    'tts.model.changed': ['模型文件已改变，请检查文件，或更换模型目录后重新下载。', 'Model files have changed. Check the files, or choose another model directory and download again.'],
    'tts.model.missing': ['模型文件已移走或丢失。', 'Model files were moved or are missing.'],
    'tts.model.unavailable': ['无法访问模型文件，请检查目录及权限。', 'Model files cannot be accessed. Check the directory and its permissions.'],
  };
  const message = known[issue.code];
  return message && issue.message === message[0] ? text(...message) : issue.message;
}

export const ttsModelKeys = {
  catalog: ['tts-models', 'catalog'] as const,
  installed: ['tts-models', 'installed'] as const,
  downloads: ['tts-models', 'downloads'] as const,
};
const read = <T,>(path: string, signal?: AbortSignal) => apiClient.get<T>(path, { signal, silent: true, timeout: READ_TIMEOUT_MS });
const taskUrl = (id: string) => `/tts/models/downloads/${encodeURIComponent(id)}`;
export const ttsModelsApi = {
  catalog: (signal?: AbortSignal) => read<TtsModelPackage[]>('/tts/models/catalog', signal),
  installed: (signal?: AbortSignal) => read<TtsInstalledModel[]>('/tts/models', signal),
  downloads: (signal?: AbortSignal) => read<TtsModelDownload[]>('/tts/models/downloads', signal),
  download: (id: string, signal?: AbortSignal) => read<TtsModelDownload>(taskUrl(id), signal),
  start: (packageId: string) => apiClient.post<TtsModelDownload>('/tts/models/downloads', { package_id: packageId, provider: 'huggingface' }, { silent: true }),
  cancel: (id: string) => apiClient.post<TtsModelDownload>(`${taskUrl(id)}/cancel`, {}, { silent: true }),
  retry: (id: string) => apiClient.post<TtsModelDownload>(`${taskUrl(id)}/retry`, {}, { silent: true }),
};

export function ttsModelMatches(model: Pick<TtsInstalledModel, 'engine' | 'variant'>, engine: TtsModelEngine, variant?: TtsModelVariant): boolean {
  return model.engine === engine && (engine === 'voxcpm1.5' ? model.variant == null : !!variant && model.variant === variant);
}
export function ttsModelSelectable(model: TtsInstalledModel, engine: TtsModelEngine, variant?: TtsModelVariant): boolean {
  return model.ready && model.status === 'ready' && !!model.bindings.model_path.trim() && ttsModelMatches(model, engine, variant);
}
export const ttsDownloadActive = (task: TtsModelDownload) => ['queued', 'downloading', 'verifying'].includes(task.status);

/** Keep retries from reviving older failures; downloads into different folders remain separate. */
export function latestTtsDownloads(tasks: TtsModelDownload[]): TtsModelDownload[] {
  const seen = new Set<string>();
  return [...tasks].sort((a, b) => b.created_at - a.created_at || b.id.localeCompare(a.id)).filter(task => {
    const key = JSON.stringify([task.package_id, task.package_revision, task.target_path]);
    if (seen.has(key)) return false;
    seen.add(key); return true;
  });
}

export function ttsModelsUrl(engine?: TtsModelEngine, variant?: TtsModelVariant): string {
  const params = new URLSearchParams({ tab: 'models', type: 'tts' });
  if (engine) params.set('engine', engine);
  if (variant) params.set('variant', variant);
  return `/settings/environment?${params}`;
}
