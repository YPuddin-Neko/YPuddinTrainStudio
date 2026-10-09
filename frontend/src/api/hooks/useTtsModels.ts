import { useEffect } from 'react';
import { useQueryClient, type QueryClient } from '@tanstack/react-query';
import { EVENT_TYPES } from '../../events/eventTypes';
import { useEventStream, useEventStreamStatus } from '../../events/useEventStream';
import { IDLE_POLL_MS, LOCAL_STALE_MS, invalidateResource, useResourceQuery, useSharedSettings } from '../resourcePolicy';
import { ttsDownloadActive, ttsModelKeys, ttsModelsApi, type TtsModelDownload } from '../ttsModels';

const updates = new WeakMap<QueryClient, { revision: number; tasks: Map<string, { revision: number; task: TtsModelDownload }> }>();
const changesFor = (client: QueryClient) => {
  let changes = updates.get(client);
  if (!changes) { changes = { revision: 0, tasks: new Map() }; updates.set(client, changes); }
  return changes;
};

function refreshInstallations(client: QueryClient, before: TtsModelDownload[] | undefined, after: TtsModelDownload[], fromEvent = false) {
  if (after.some(task => !ttsDownloadActive(task) && (before?.some(previous => previous.id === task.id && ttsDownloadActive(previous))
    || (fromEvent || before !== undefined) && !before?.some(previous => previous.id === task.id)))) {
    invalidateResource(client, ttsModelKeys.installed);
    invalidateResource(client, ttsModelKeys.catalog);
  }
}

/** Mutation responses can arrive after a newer progress event for the same task. */
export function updateTtsDownload(client: QueryClient, task: TtsModelDownload, fromEvent = false) {
  const previous = client.getQueryData<TtsModelDownload[]>(ttsModelKeys.downloads);
  const changes = changesFor(client);
  const current = previous?.find(item => item.id === task.id) || changes.tasks.get(task.id)?.task;
  if (current && (ttsDownloadActive(task) && !ttsDownloadActive(current)
    || !fromEvent && task.status === 'queued' && current.status !== 'queued'
    || (task.progress_at ?? 0) < (current.progress_at ?? 0) && ttsDownloadActive(task))) return;
  if (current && JSON.stringify(current) === JSON.stringify(task)) return;
  changes.tasks.set(task.id, { revision: ++changes.revision, task });
  const next = [task, ...(previous || []).filter(item => item.id !== task.id)];
  if (previous) client.setQueryData(ttsModelKeys.downloads, next);
  refreshInstallations(client, previous, next, true);
}

export function useTtsInstalledModels(enabled = true) {
  const client = useQueryClient();
  useEventStream<TtsModelDownload>(EVENT_TYPES.TTS_MODEL_DOWNLOAD, task => {
    if (enabled && task?.id && task.status) updateTtsDownload(client, task, true);
  });
  useEffect(() => {
    const changed = () => invalidateResource(client, ttsModelKeys.installed);
    window.addEventListener('studio-tts-models-changed', changed);
    return () => window.removeEventListener('studio-tts-models-changed', changed);
  }, [client]);
  return useResourceQuery({ queryKey: ttsModelKeys.installed, queryFn: ({ signal }) => ttsModelsApi.installed(signal),
    enabled, staleTime: LOCAL_STALE_MS, pollInterval: IDLE_POLL_MS });
}

export function useTtsModelResources(enabled = true) {
  const client = useQueryClient();
  const connection = useEventStreamStatus();
  const catalog = useResourceQuery({ queryKey: ttsModelKeys.catalog, queryFn: ({ signal }) => ttsModelsApi.catalog(signal),
    enabled, staleTime: 5 * 60_000, refetchOnVisible: false });
  const installed = useTtsInstalledModels(enabled);
  const downloads = useResourceQuery({
    queryKey: ttsModelKeys.downloads,
    queryFn: async ({ signal }) => {
      const revision = changesFor(client).revision;
      const previous = client.getQueryData<TtsModelDownload[]>(ttsModelKeys.downloads);
      const rows = await ttsModelsApi.downloads(signal);
      const newer = [...changesFor(client).tasks.values()].filter(item => item.revision > revision).map(item => item.task);
      const next = newer.length ? [...newer, ...rows.filter(row => !newer.some(task => task.id === row.id))] : rows;
      if (!signal.aborted) refreshInstallations(client, previous, next);
      return next;
    },
    enabled, staleTime: LOCAL_STALE_MS,
    pollInterval: data => connection !== 'connected' && data?.some(ttsDownloadActive) ? 2500 : IDLE_POLL_MS,
  });
  const settings = useSharedSettings(enabled);
  return { catalog, installed, downloads, settings };
}
