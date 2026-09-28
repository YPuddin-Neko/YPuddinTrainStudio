import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import type { DatasetInfo, VisionCatalog } from '../../api/types';
import { useWorkspaceText } from '../../utils/workspaceText';

export const ACTIVE_DOWNLOAD = ['queued', 'downloading', 'verifying'];

/** Tagging and head-detection models with their download state; polls while one downloads. */
export function useVisionModels() {
  return useQuery({
    queryKey: ['vision-models'],
    queryFn: ({ signal }) => apiClient.get<VisionCatalog>('/vision/models', { signal, silent: true }),
    refetchInterval: query => query.state.data?.models.some(model => ACTIVE_DOWNLOAD.includes(model.download?.status ?? '')) ? 1000 : false,
  });
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
