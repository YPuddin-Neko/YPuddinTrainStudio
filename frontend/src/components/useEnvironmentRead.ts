import { useCallback, useEffect, useRef, useState } from 'react';
import { apiClient, READ_TIMEOUT_MS } from '../api/client';
import { formatApiError } from '../utils/errors';

type Options<T> = {
  enabled?: boolean;
  interval?: number | false;
  refreshParam?: boolean;
  probe?: boolean;
  initialData?: T;
  onSuccess?: (value: T, previous: T | null) => void;
};

export function useEnvironmentRead<T>(endpoint: string, { enabled = true, interval = false, refreshParam = false, probe = false, initialData, onSuccess }: Options<T> = {}) {
  const initial = useRef(initialData);
  const [data, setData] = useState<T | null>(initialData ?? null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(enabled && initialData === undefined);
  const [probing, setProbing] = useState(false);
  const value = useRef<T | null>(initialData ?? null);
  const alive = useRef(false);
  const failed = useRef(false);
  const accepted = useRef(onSuccess);
  accepted.current = onSuccess;
  const pending = useRef<{ controller: AbortController; promise: Promise<void> } | null>(null);
  const read = useCallback((refresh = false, replace = false): Promise<void> => {
    if (!alive.current || !enabled || failed.current && !replace) return Promise.resolve();
    if (pending.current) {
      if (!replace) return pending.current.promise;
      pending.current.controller.abort();
    }
    failed.current = false;
    const controller = new AbortController();
    const current = () => alive.current && pending.current?.controller === controller && !controller.signal.aborted;
    setLoading(true); setProbing(probe && refresh);
    const promise = apiClient.get<T>(endpoint, {
      silent: true, signal: controller.signal, timeout: probe && refresh ? 120_000 : READ_TIMEOUT_MS,
      ...(refreshParam ? { params: { refresh } } : {}),
    }).then(next => {
      if (!current()) return;
      const previous = value.current;
      value.current = next; setData(next); setError(''); accepted.current?.(next, previous);
    }).catch(failure => {
      if (current()) { failed.current = true; setError(formatApiError(failure)); }
    }).finally(() => {
      if (!current()) return;
      pending.current = null; setLoading(false); setProbing(false);
    });
    pending.current = { controller, promise };
    return promise;
  }, [enabled, endpoint, probe, refreshParam]);
  useEffect(() => {
    alive.current = true;
    if (initial.current === undefined) void read();
    return () => { alive.current = false; pending.current?.controller.abort(); pending.current = null; };
  }, [read]);
  useEffect(() => {
    if (!enabled || !interval) return;
    const timer = window.setInterval(() => { void read(); }, interval);
    return () => window.clearInterval(timer);
  }, [enabled, interval, read]);
  const reload = useCallback((refresh = false) => read(refresh, true), [read]);
  const updateData = useCallback((update: (current: T | null) => T | null) => {
    if (!alive.current) return;
    pending.current?.controller.abort(); pending.current = null;
    setLoading(false); setProbing(false);
    const previous = value.current;
    const next = update(previous);
    value.current = next; setData(next);
    if (next !== null) accepted.current?.(next, previous);
  }, []);
  return { data, error, loading, probing, reload, read, updateData };
}
