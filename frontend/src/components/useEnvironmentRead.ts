import { useCallback, useEffect, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { apiClient, READ_TIMEOUT_MS } from '../api/client';
import { LOCAL_STALE_MS, useResourceQuery } from '../api/resourcePolicy';
import { formatApiError } from '../utils/errors';

type Options<T> = {
  enabled?: boolean;
  interval?: number | false | ((value: T | undefined) => number | false);
  refreshParam?: boolean;
  probe?: boolean;
  initialData?: T;
  staleTime?: number;
  refetchOnVisible?: boolean;
  reconcile?: (value: T, previous: T | undefined) => T;
  onSuccess?: (value: T, previous: T | null) => void;
};

export function useEnvironmentRead<T>(endpoint: string, { enabled = true, interval = false, refreshParam = false, probe = false, initialData, staleTime = LOCAL_STALE_MS, refetchOnVisible = true, reconcile, onSuccess }: Options<T> = {}) {
  const client = useQueryClient();
  const [probing, setProbing] = useState(false);
  const mounted = useRef(true), accepted = useRef(onSuccess), previous = useRef<T | null>(null);
  const merge = useRef(reconcile); merge.current = reconcile;
  accepted.current = onSuccess;
  const fetch = useCallback(async (signal: AbortSignal, refresh = false) => {
    const value = await apiClient.get<T>(endpoint, {
      silent: true, signal, timeout: probe && refresh ? 120_000 : READ_TIMEOUT_MS,
      ...(refreshParam ? { params: { refresh } } : {}),
    });
    return merge.current ? merge.current(value, client.getQueryData<T>(['environment', endpoint])) : value;
  }, [client, endpoint, probe, refreshParam]);
  const query = useResourceQuery<T>({ queryKey: ['environment', endpoint], enabled, initialData, staleTime, pollInterval: interval, refetchOnVisible,
    queryFn: ({ signal }) => fetch(signal) });
  const lastError = useRef('');
  if (query.error) lastError.current = formatApiError(query.error);
  else if (query.status === 'success' && !query.isFetching) lastError.current = '';
  useEffect(() => {
    if (query.data !== undefined) { const before = previous.current; previous.current = query.data; accepted.current?.(query.data, before); }
  }, [query.data, query.dataUpdatedAt]);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  const read = useCallback(async (refresh = false, explicit = false) => {
    if (!enabled || !mounted.current) return;
    const queryKey = ['environment', endpoint];
    if (!explicit && client.getQueryState(queryKey)?.error) return;
    if (probe && refresh) setProbing(true);
    try {
      await client.fetchQuery<T>({ queryKey, queryFn: ({ signal }) => fetch(signal, refresh), staleTime: 0, retry: false, networkMode: 'always' });
    } catch { /* The shared query retains the failure for the row's retry control. */ }
    finally { if (mounted.current) setProbing(false); }
  }, [client, enabled, endpoint, fetch, probe]);
  const reload = useCallback((refresh = false) => read(refresh, true), [read]);
  const updateData = useCallback((update: (current: T | null) => T | null) => {
    const key = ['environment', endpoint];
    void client.cancelQueries({ queryKey: key, exact: true });
    client.setQueryData<T>(key, current => update(current ?? null) ?? undefined);
  }, [client, endpoint]);
  return { data: query.data ?? null, error: lastError.current, loading: query.isFetching, probing, reload, read, updateData };
}
