import { useEffect } from 'react';
import { useQueryClient, type QueryClient } from '@tanstack/react-query';
import { apiClient, READ_TIMEOUT_MS } from '../client';
import type { components } from '../generated';
import type { ModelAsset, ModelDownload } from '../types';
import { IDLE_POLL_MS, invalidateResource, useResourceQuery } from '../resourcePolicy';
import { EVENT_TYPES } from '../../events/eventTypes';
import { useEventStream, useEventStreamStatus } from '../../events/useEventStream';

export type ImageRecommendation = components['schemas']['RecommendedModel'];
export type ImageModelSnapshot = { catalog: ImageRecommendation[]; assets: ModelAsset[]; tasks: ModelDownload[] };
export const imageModelKeys = {
  catalog: ['image-models', 'catalog'], assets: ['image-models', 'assets'], tasks: ['image-models', 'tasks'],
} as const;
export const imageDownloadActive = (task: ModelDownload) => ['queued', 'downloading'].includes(task.status);

const updates = new WeakMap<QueryClient, { revision: number; tasks: Map<string, { revision: number; task: ModelDownload }> }>();
const changesFor = (client: QueryClient) => {
  let changes = updates.get(client);
  if (!changes) { changes = { revision: 0, tasks: new Map() }; updates.set(client, changes); }
  return changes;
};
function refreshInstalled(client: QueryClient, before: ModelDownload[] | undefined, after: ModelDownload[], fromEvent = false) {
  if (after.some(task => !imageDownloadActive(task) && (before?.some(previous => previous.id === task.id && imageDownloadActive(previous))
    || fromEvent && !before?.some(previous => previous.id === task.id)))) {
    invalidateResource(client, imageModelKeys.assets); invalidateResource(client, imageModelKeys.catalog);
  }
}
export function updateImageDownload(client: QueryClient, task: ModelDownload, fromEvent = false) {
  const previous = client.getQueryData<ModelDownload[]>(imageModelKeys.tasks);
  const changes = changesFor(client);
  const current = previous?.find(item => item.id === task.id) ?? changes.tasks.get(task.id)?.task;
  if (current && (imageDownloadActive(task) && !imageDownloadActive(current)
    || task.status === 'queued' && current.status !== 'queued'
    || imageDownloadActive(task) && (task.progress_at ?? 0) < (current.progress_at ?? 0))) return;
  if (current && JSON.stringify(current) === JSON.stringify(task)) return;
  changes.tasks.set(task.id, { revision: ++changes.revision, task });
  const next = [task, ...(previous ?? []).filter(item => item.id !== task.id)];
  // A task event is not a complete list. Keep it until the initial list arrives.
  if (previous) {
    if (!fromEvent) void client.cancelQueries({ queryKey: imageModelKeys.tasks, exact: true });
    client.setQueryData(imageModelKeys.tasks, next);
  }
  refreshInstalled(client, previous, next, fromEvent);
}
async function readTasks(client: QueryClient, signal?: AbortSignal) {
  const revision = changesFor(client).revision;
  const previous = client.getQueryData<ModelDownload[]>(imageModelKeys.tasks);
  const rows = await apiClient.get<ModelDownload[]>('/models/downloads', { signal, silent: true, timeout: READ_TIMEOUT_MS });
  const newer = [...changesFor(client).tasks.values()].filter(item => item.revision > revision).map(item => item.task);
  const next = [...newer, ...rows.filter(row => !newer.some(task => task.id === row.id))];
  if (!signal?.aborted) refreshInstalled(client, previous, next);
  return next;
}

/** The setup step and model manager observe the same lists and task progress. */
export function useImageModelResources(enabled = true) {
  const client = useQueryClient();
  const connection = useEventStreamStatus();
  const catalog = useResourceQuery<ImageRecommendation[]>({ queryKey: imageModelKeys.catalog,
    queryFn: ({ signal }) => apiClient.get('/models/recommendations', { signal, silent: true, timeout: READ_TIMEOUT_MS }), enabled, pollInterval: IDLE_POLL_MS });
  const assets = useResourceQuery<ModelAsset[]>({ queryKey: imageModelKeys.assets,
    queryFn: ({ signal }) => apiClient.get('/models', { signal, silent: true, timeout: READ_TIMEOUT_MS }), enabled, pollInterval: IDLE_POLL_MS });
  const tasks = useResourceQuery<ModelDownload[]>({ queryKey: imageModelKeys.tasks,
    queryFn: ({ signal }) => readTasks(client, signal), enabled,
    pollInterval: rows => connection !== 'connected' && rows?.some(imageDownloadActive) ? 3000 : IDLE_POLL_MS });
  const acceptDownload = (task: ModelDownload) => updateImageDownload(client, task);
  const acceptModel = (model: ModelAsset, recommendationId?: string) => {
    void client.cancelQueries({ queryKey: imageModelKeys.assets, exact: true });
    void client.cancelQueries({ queryKey: imageModelKeys.catalog, exact: true });
    client.setQueryData<ModelAsset[]>(imageModelKeys.assets, previous => [...(previous ?? []).filter(item => item.id !== model.id)
      .map(item => model.is_default && item.family === model.family && item.kind === model.kind ? { ...item, is_default: false } : item), model]);
    client.setQueryData<ImageRecommendation[]>(imageModelKeys.catalog, previous => previous?.map(entry => entry.id === recommendationId || entry.model_id === model.id || entry.available_path === model.path
      ? { ...entry, model_id: model.id, available_path: model.path, is_default: model.is_default }
      : model.is_default && entry.family === model.family && entry.kind === model.kind ? { ...entry, is_default: false } : entry));
  };
  useEventStream<ModelDownload>(EVENT_TYPES.MODEL_DOWNLOAD, task => {
    if (task?.id && task.status) updateImageDownload(client, task, true);
  });
  useEffect(() => {
    const changed = () => Object.values(imageModelKeys).forEach(key => invalidateResource(client, key));
    window.addEventListener('studio-models-changed', changed);
    return () => { window.removeEventListener('studio-models-changed', changed); };
  }, [client]);
  const removeModel = (id: string) => {
    void client.cancelQueries({ queryKey: imageModelKeys.assets, exact: true });
    void client.cancelQueries({ queryKey: imageModelKeys.catalog, exact: true });
    client.setQueryData<ModelAsset[]>(imageModelKeys.assets, previous => previous?.filter(model => model.id !== id));
    client.setQueryData<ImageRecommendation[]>(imageModelKeys.catalog, previous => previous?.map(entry => entry.model_id === id ? { ...entry, model_id: null, is_default: false } : entry));
  };
  const cancelReads = () => Promise.all(Object.values(imageModelKeys).map(queryKey => client.cancelQueries({ queryKey, exact: true })));
  const acceptSnapshot = (snapshot: ImageModelSnapshot) => {
    const previous = client.getQueryData<ModelDownload[]>(imageModelKeys.tasks) ?? [];
    const tasks = snapshot.tasks.map(task => {
      const current = previous.find(item => item.id === task.id);
      return current && (imageDownloadActive(task) && !imageDownloadActive(current)
        || task.status === 'queued' && current.status !== 'queued'
        || imageDownloadActive(task) && (task.progress_at ?? 0) < (current.progress_at ?? 0)) ? current : task;
    });
    const next = { ...snapshot, tasks };
    for (const name of ['catalog', 'assets', 'tasks'] as const) client.setQueryData(imageModelKeys[name], next[name]);
    return next;
  };
  const refresh = async () => { await cancelReads(); return Promise.all([catalog.refetch(), assets.refetch(), tasks.refetch()]); };
  return { catalog, assets, tasks, readTasks: (signal?: AbortSignal) => readTasks(client, signal), acceptDownload, acceptModel, acceptSnapshot, removeModel, cancelReads, refresh };
}
