import React from 'react';
import { apiClient } from '../client';
import type { QueueDevices } from '../types';
import { formatApiError } from '../../utils/errors';

/** Device ownership and readings, refreshed every `seconds`. */
export function useQueueDevices(seconds = 5) {
  const [snapshot, setSnapshot] = React.useState<QueueDevices | null>(null);
  const [error, setError] = React.useState('');
  const [loading, setLoading] = React.useState(true);
  const [revision, setRevision] = React.useState(0);
  React.useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const refresh = async () => {
      try {
        const next = await apiClient.get<QueueDevices>('/queue/devices', { signal: controller.signal, silent: true });
        if (!controller.signal.aborted) { setSnapshot(next); setError(''); }
      } catch (failure) { if (!controller.signal.aborted) setError(formatApiError(failure)); }
      finally { if (!controller.signal.aborted) { setLoading(false); timer = setTimeout(() => void refresh(), seconds * 1000); } }
    };
    void refresh();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [revision, seconds]);
  return { snapshot, error, loading, refresh: () => { setLoading(true); setRevision(value => value + 1); } };
}
