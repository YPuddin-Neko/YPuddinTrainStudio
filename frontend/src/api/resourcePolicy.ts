import { createContext, useContext, useEffect, useRef, useSyncExternalStore } from 'react';
import { hashKey, useQuery, useQueryClient, type QueryClient, type QueryKey } from '@tanstack/react-query';
import { apiClient, READ_TIMEOUT_MS } from './client';
import type { Settings } from './types';
import { useEventStream, useEventStreamStatus } from '../events/useEventStream';
import { EVENT_TYPES } from '../events/eventTypes';

export const LOCAL_STALE_MS = 60_000;
export const ONLINE_STALE_MS = 3_600_000;
export const IDLE_POLL_MS = 60_000;
export const ResourceActivityContext = createContext(true);
const subscribeVisibility = (notify: () => void) => {
  document.addEventListener('visibilitychange', notify);
  return () => document.removeEventListener('visibilitychange', notify);
};
const visibleSnapshot = () => document.visibilityState !== 'hidden';
export function usePageVisible() {
  const active = useContext(ResourceActivityContext);
  return useSyncExternalStore(subscribeVisibility, visibleSnapshot, () => true) && active;
}

type ReadOptions<T> = {
  queryKey: QueryKey;
  queryFn: (context: { signal: AbortSignal }) => Promise<T>;
  enabled?: boolean;
  initialData?: T;
  staleTime?: number;
  pollInterval?: number | false | ((data: T | undefined) => number | false);
  refetchOnVisible?: boolean;
};
type PollOwner = { enabled: boolean; interval: () => number | false; read: () => Promise<unknown> };
type PollEntry = { owners: Set<PollOwner>; timer?: ReturnType<typeof setTimeout>; dispose: () => void; schedule: () => void };
const pollers = new WeakMap<QueryClient, Map<string, PollEntry>>();

// A query can appear in a page, drawer and picker at once. They share one clock.
function registerPoll(client: QueryClient, key: QueryKey, owner: PollOwner) {
  let entries = pollers.get(client);
  if (!entries) { entries = new Map(); pollers.set(client, entries); }
  const hash = hashKey(key);
  let entry = entries.get(hash);
  if (!entry) {
    const owners = new Set<PollOwner>();
    const next: PollEntry = { owners, dispose: () => {}, schedule: () => {} };
    next.schedule = () => {
      clearTimeout(next.timer);
      const state = client.getQueryState(key);
      if (!state || state.data === undefined || state.error || state.fetchStatus !== 'idle') return;
      const eligible = [...owners].filter(item => item.enabled).map(item => ({ item, interval: item.interval() }))
        .filter((item): item is { item: PollOwner; interval: number } => typeof item.interval === 'number' && item.interval > 0)
        .sort((a, b) => a.interval - b.interval);
      if (!eligible.length) return;
      const { item, interval } = eligible[0];
      next.timer = setTimeout(() => { void item.read().catch(() => {}); }, Math.max(0, interval - (Date.now() - state.dataUpdatedAt)));
    };
    const unsubscribe = client.getQueryCache().subscribe(event => { if (event.query.queryHash === hash) next.schedule(); });
    next.dispose = () => { clearTimeout(next.timer); unsubscribe(); };
    entry = next; entries.set(hash, entry);
  }
  entry.owners.add(owner); entry.schedule();
  return () => {
    entry.owners.delete(owner);
    if (entry.owners.size) entry.schedule();
    else { entry.dispose(); entries.delete(hash); }
  };
}

type Invalidation = { promise: Promise<void>; running: boolean; dirty: boolean };
const invalidations = new WeakMap<QueryClient, Map<string, Invalidation>>();
export function invalidateResource(client: QueryClient, queryKey: QueryKey): Promise<void> {
  let pending = invalidations.get(client);
  if (!pending) { pending = new Map(); invalidations.set(client, pending); }
  const hash = hashKey(queryKey), existing = pending.get(hash);
  if (existing) { if (existing.running) existing.dirty = true; return existing.promise; }
  const entry: Invalidation = { promise: Promise.resolve(), running: false, dirty: false };
  entry.promise = Promise.resolve().then(async () => {
    entry.running = true;
    do {
      entry.dirty = false;
      await client.invalidateQueries({ queryKey, exact: true }, { cancelRefetch: true });
    } while (entry.dirty);
  })
    .finally(() => { pending.delete(hash); });
  pending.set(hash, entry);
  return entry.promise;
}

export function useResourceQuery<T>({ queryKey, queryFn, enabled = true, initialData, staleTime = LOCAL_STALE_MS, pollInterval = false, refetchOnVisible = true }: ReadOptions<T>) {
  const client = useQueryClient(), visible = usePageVisible();
  const hash = hashKey(queryKey);
  const latest = useRef({ queryKey, queryFn, pollInterval });
  latest.current = { queryKey, queryFn, pollInterval };
  const query = useQuery<T, Error>({
    queryKey, queryFn, initialData, staleTime, gcTime: Math.max(10 * 60_000, staleTime),
    enabled: resource => enabled && visible && !resource.state.error,
    retry: false, retryOnMount: false, networkMode: 'always',
    refetchOnWindowFocus: false, refetchOnReconnect: false,
  });
  const active = enabled && visible;
  useEffect(() => registerPoll(client, latest.current.queryKey, {
    enabled: active,
    interval: () => {
      const { queryKey: key, pollInterval: interval } = latest.current;
      return typeof interval === 'function' ? interval(client.getQueryData<T>(key)) : interval;
    },
    read: () => client.fetchQuery({ queryKey: latest.current.queryKey, queryFn: latest.current.queryFn, staleTime: 0, retry: false, networkMode: 'always' }),
  }), [client, hash, active, pollInterval]);
  const previousVisible = useRef(visible);
  useEffect(() => {
    const key = latest.current.queryKey;
    if (!active) {
      // Query observers update their enabled state before this microtask.
      void Promise.resolve().then(() => {
        if (!client.getQueryCache().find({ queryKey: key, exact: true })?.isActive()) void client.cancelQueries({ queryKey: key, exact: true });
      });
    } else if (!previousVisible.current && refetchOnVisible && !client.getQueryState(key)?.error) {
      // Re-enabling a stale observer may already have started this same read.
      void client.invalidateQueries({ queryKey: key, exact: true }, { cancelRefetch: false });
    }
    previousVisible.current = visible;
  }, [active, visible, client, hash, refetchOnVisible]);
  useEffect(() => {
    if (!active || !refetchOnVisible) return;
    const focus = () => {
      const key = latest.current.queryKey, state = client.getQueryState(key);
      if (state && !state.error && state.fetchStatus === 'idle' && Date.now() - state.dataUpdatedAt >= staleTime)
        void client.invalidateQueries({ queryKey: key, exact: true }, { cancelRefetch: false });
    };
    window.addEventListener('focus', focus);
    return () => window.removeEventListener('focus', focus);
  }, [active, client, staleTime, refetchOnVisible]);
  return query;
}

const settingsSubscribers = new WeakMap<QueryClient, { count: number; dispose: () => void }>();
export function useResourceCacheEvents() {
  const client = useQueryClient();
  useEffect(() => {
    let subscription = settingsSubscribers.get(client);
    if (!subscription) {
      const networkChanged = () => {
        for (const endpoint of ['/environment/latest', '/environment/lora', '/environment/torch', '/environment/windows/wheels', '/environment/dtk/wheels']) {
          const queryKey = ['environment', endpoint];
          void client.cancelQueries({ queryKey, exact: true });
          void client.invalidateQueries({ queryKey, exact: true, refetchType: 'none' });
        }
      };
      const changed = (event: Event) => {
        const detail = (event as CustomEvent<Settings>).detail;
        if (detail) {
          const previous = client.getQueryData<Settings>(['settings']);
          if (JSON.stringify([previous?.network, previous?.downloads]) !== JSON.stringify([detail.network, detail.downloads])) networkChanged();
          if (previous?.paths?.models_dir !== detail.paths?.models_dir) {
            for (const key of [['vision-models'], ['image-models', 'assets'], ['image-models', 'catalog']]) void invalidateResource(client, key);
          }
          void client.cancelQueries({ queryKey: ['settings'], exact: true });
          client.setQueryData(['settings'], detail);
        } else void invalidateResource(client, ['settings']);
      };
      const credentialsChanged = () => {
        void invalidateResource(client, ['credentials']);
        void invalidateResource(client, ['vision-models']);
      };
      window.addEventListener('studio.settings.changed', changed);
      window.addEventListener('studio.network.changed', networkChanged);
      window.addEventListener('credentials.changed', credentialsChanged);
      subscription = { count: 0, dispose: () => {
        window.removeEventListener('studio.settings.changed', changed);
        window.removeEventListener('studio.network.changed', networkChanged);
        window.removeEventListener('credentials.changed', credentialsChanged);
      } };
      settingsSubscribers.set(client, subscription);
    }
    subscription.count++;
    return () => { if (!--subscription.count) { subscription.dispose(); settingsSubscribers.delete(client); } };
  }, [client]);
}
export function useSharedSettings(enabled = true) {
  useResourceCacheEvents();
  const query = useResourceQuery<Settings>({ queryKey: ['settings'], enabled,
    queryFn: ({ signal }) => apiClient.get('/settings', { signal, silent: true, timeout: READ_TIMEOUT_MS }) });
  return query;
}

type ResourceEvent = { id?: string; kind?: string; state?: string; dismissed?: boolean };
export function useLocalResourceSync(queryKeys: QueryKey[], accepts: (event: ResourceEvent) => boolean, onChange?: (event: ResourceEvent) => void) {
  const client = useQueryClient(), connection = useEventStreamStatus();
  const latest = useRef({ queryKeys, accepts, onChange }); latest.current = { queryKeys, accepts, onChange };
  const statuses = useRef(new Map<string, string>());
  useEventStream<ResourceEvent>(EVENT_TYPES.BACKGROUND_CHANGED, event => {
    if (!event.id || !latest.current.accepts(event)) return;
    const signature = `${event.state ?? ''}:${!!event.dismissed}`;
    if (statuses.current.get(event.id) === signature) return;
    statuses.current.set(event.id, signature);
    latest.current.onChange?.(event);
    for (const key of latest.current.queryKeys) void invalidateResource(client, key);
  });
  const wasConnected = useRef(false), previous = useRef(connection);
  useEffect(() => {
    if (connection === 'connected') {
      if (wasConnected.current && previous.current !== 'connected')
        for (const key of latest.current.queryKeys) void invalidateResource(client, key);
      wasConnected.current = true;
    }
    previous.current = connection;
  }, [connection, client]);
}
