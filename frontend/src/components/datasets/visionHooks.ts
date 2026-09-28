import { useEffect, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import type { DatasetInfo, Settings, VisionCatalog, VisionModel, VlmService, VlmServices } from '../../api/types';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';

export const ACTIVE_DOWNLOAD = ['queued', 'downloading', 'verifying'];

/** Tagging and head-detection models with their download state; polls while one downloads. */
export function useVisionModels() {
  return useQuery({
    queryKey: ['vision-models'],
    queryFn: ({ signal }) => apiClient.get<VisionCatalog>('/vision/models', { signal, silent: true }),
    refetchInterval: query => query.state.data?.models.some(model => ACTIVE_DOWNLOAD.includes(model.download?.status ?? '')) ? 1000 : false,
  });
}

/** Taggers of one series, newest first as the catalog lists them. */
export const TAGGER_SERIES: Record<string, string> = { wd: 'WD', pixai: 'PixAI', cl: 'CL Tagger' };
export const modelName = (model: VisionModel) => model.label.replace(/^(WD|PixAI|CL Tagger) /, '');

export type TaggingSettings = NonNullable<Settings['tagging']>;
export type VlmSettings = NonNullable<TaggingSettings['vlm']>;

/** 设置 → 打标: the vision model service and where models download from. Saving updates every open page. */
export function useTaggingSettings() {
  const client = useQueryClient();
  const query = useQuery({
    queryKey: ['settings'],
    queryFn: ({ signal }) => apiClient.get<Settings>('/settings', { signal, silent: true }),
    staleTime: 30_000,
  });
  useEffect(() => {
    const changed = (event: Event) => { const detail = (event as CustomEvent<Settings>).detail; if (detail) client.setQueryData(['settings'], detail); else void client.invalidateQueries({ queryKey: ['settings'] }); };
    window.addEventListener('studio.settings.changed', changed);
    return () => window.removeEventListener('studio.settings.changed', changed);
  }, [client]);
  const save = async (tagging: TaggingSettings) => {
    const result = await apiClient.put<Settings>('/settings', { tagging }, { silent: true });
    client.setQueryData(['settings'], result);
    window.dispatchEvent(new CustomEvent('studio.settings.changed', { detail: result }));
    return result;
  };
  return { tagging: query.data?.tagging, query, save };
}

export const datasetLabel = (path: string) => path.replace(/\\/g, '/').split('/').filter(Boolean).pop()?.replace(/^(?:d_[0-9a-f]+-)+/, '') || path;

/** Settings kept per browser so the next run starts from the last one. */
export function useRememberedSettings<T extends object>(key: string, defaults: T) {
  const [settings, setSettings] = useState<T>(() => {
    try { return { ...defaults, ...JSON.parse(localStorage.getItem(key) || '{}') }; } catch { return defaults; }
  });
  const update = (patch: Partial<T>) => setSettings(previous => {
    const next = { ...previous, ...patch };
    try { localStorage.setItem(key, JSON.stringify(next)); } catch { /* per-viewer convenience only */ }
    return next;
  });
  return [settings, update] as const;
}

/** The datasets a run can cover: every training dataset together, or one dataset. */
export function useScopeOptions(projectId: string, versionId: string) {
  const text = useWorkspaceText();
  const datasets = useQuery({ queryKey: ['caption-datasets', projectId, versionId], queryFn: ({ signal }) => apiClient.get<DatasetInfo[]>(`/projects/${projectId}/datasets`, { params: { version_id: versionId, include_cache: false }, signal, silent: true }) });
  const rows = datasets.data || [];
  const training = rows.filter(row => !row.source.is_reg);
  // A dataset still being indexed has no counts yet.
  const count = (row: DatasetInfo) => row.stats?.training_images ?? row.stats?.images ?? 0;
  const total = training.reduce((sum, row) => sum + count(row), 0);
  const options = [
    ...(training.length > 1 ? [{ value: 'training', label: text(`全部训练集 · ${total} 张`, `All training sets · ${total} images`) }] : []),
    ...rows.map(row => ({ value: row.source.id, label: `${row.source.is_reg ? text('正则 · ', 'Reg · ') : ''}${datasetLabel(row.source.path)} · ${count(row)} ${text('张', 'images')}` })),
  ];
  const ids = (scope: string) => scope === 'training' ? training.map(row => row.source.id) : [scope];
  return { options, ids, loading: datasets.isPending, first: options[0]?.value || '' };
}

/** Vision model services with whether a key is saved for each. */
export function useVlmServices() {
  return useQuery({
    queryKey: ['vlm-services'],
    queryFn: ({ signal }) => apiClient.get<VlmServices>('/vlm/services', { signal, silent: true }),
  });
}

/** The models a service offers, read by the server with the saved key. */
export function useVlmModels(provider: string, baseUrl: string, enabled: boolean) {
  return useQuery({
    queryKey: ['vlm-models', provider, baseUrl],
    queryFn: () => apiClient.post<{ models: string[] }>(`/vlm/services/${provider}/models`, { base_url: baseUrl || null }, { silent: true }),
    enabled,
    retry: false,
    staleTime: 5 * 60_000,
  });
}

/** The address, model and key state a vision-model run uses; null until the services have loaded. */
export function resolveService(settings: Pick<VlmSettings, 'provider' | 'base_urls' | 'models'> | undefined, services?: VlmService[]) {
  const service = services?.find(item => item.id === settings?.provider) || services?.[0];
  if (!service) return null;
  const baseUrl = service.editable ? (settings?.base_urls?.[service.id] ?? service.base_url) : service.base_url;
  return { service, baseUrl, model: settings?.models?.[service.id] || '' };
}

export const SERVICE_NAMES: Record<string, [string, string]> = {
  openai: ['OpenAI', 'OpenAI'],
  gemini: ['Google Gemini', 'Google Gemini'],
  openrouter: ['OpenRouter', 'OpenRouter'],
  siliconflow: ['SiliconFlow (硅基流动)', 'SiliconFlow'],
  dashscope: ['Model Studio (阿里云百炼)', 'Alibaba Cloud Model Studio'],
  deepseek: ['DeepSeek', 'DeepSeek'],
  ollama: ['Ollama', 'Ollama'],
  lmstudio: ['LM Studio', 'LM Studio'],
  custom: ['Custom (自定义)', 'Custom'],
};

/** The model-list error in the reader's language; the service's own detail follows. */
export function useVlmError() {
  const text = useWorkspaceText();
  return (error: unknown) => {
    const code = (error as { code?: string } | null)?.code;
    const detail = formatApiError(error);
    if (code === 'vlm.base_url') return text('接口地址格式应为 http(s)://主机[:端口]/路径，例如 http://127.0.0.1:11434/v1。', 'Enter the address as http(s)://host[:port]/path, e.g. http://127.0.0.1:11434/v1.');
    if (code === 'vlm.unreachable') return text(`无法连接该服务，请确认服务已启动，且训练服务器能访问这个地址。（${detail}）`, `Cannot reach the service. Check that it is running and reachable from the training server. (${detail})`);
    if (code === 'vlm.models') return text(`服务拒绝列出模型：${detail}`, `The service did not list its models: ${detail}`);
    return detail;
  };
}

/** Label categories a caption can take, in caption order. */
export function useCategoryLabels() {
  const text = useWorkspaceText();
  return {
    general: text('通用', 'General'), character: text('角色', 'Characters'), copyright: text('作品', 'Works'), artist: text('画师', 'Artists'),
    meta: text('元信息', 'Meta'), model: text('生成模型', 'Generators'), quality: text('质量', 'Quality'), rating: text('评级', 'Rating'),
  } as Record<string, string>;
}
