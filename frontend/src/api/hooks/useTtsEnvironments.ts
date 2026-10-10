import { useEffect, useRef } from 'react';
import { useIsMutating, useMutation, useQueryClient, type QueryClient } from '@tanstack/react-query';
import { useEventStream, useEventStreamStatus } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { invalidateResource, useResourceQuery } from '../resourcePolicy';
import { ttsEnvironmentActive, ttsEnvironmentKeys as keys, ttsEnvironmentsApi as api, type TtsEnvironmentChanged, type TtsEnvironmentOperation, type TtsEnvironmentSnapshot } from '../ttsEnvironments';
import type { TtsEngine } from '../tts';

function latestOperation(previous: TtsEnvironmentOperation | undefined, next: TtsEnvironmentOperation) {
  if (previous && (previous.updated_at > next.updated_at || !ttsEnvironmentActive(previous) && ttsEnvironmentActive(next))) return previous;
  return next;
}
function reconcileSnapshot(client: QueryClient, snapshot: TtsEnvironmentSnapshot): TtsEnvironmentSnapshot {
  const previous = client.getQueryData<TtsEnvironmentSnapshot>(keys.snapshot);
  return { ...snapshot, operations: snapshot.operations.map(operation => {
    const before = previous?.operations.find(item => item.id === operation.id);
    const detail = client.getQueryData<TtsEnvironmentOperation>(keys.operation(operation.id));
    return latestOperation(detail, latestOperation(before, operation));
  }) };
}
function rememberOperation(client: QueryClient, operation: TtsEnvironmentOperation) {
  const key = keys.operation(operation.id);
  client.setQueryData<TtsEnvironmentOperation>(key, previous => latestOperation(previous, operation));
  client.setQueryData<TtsEnvironmentSnapshot>(keys.snapshot, previous => previous ? { ...previous,
    operations: [latestOperation(previous.operations.find(item => item.id === operation.id), operation), ...previous.operations.filter(item => item.id !== operation.id)],
  } : previous);
}
async function refreshResource(client: QueryClient, queryKey: readonly string[]) {
  const state = client.getQueryState(queryKey);
  if (state?.fetchStatus === 'fetching' && state.data === undefined) await client.cancelQueries({ queryKey, exact: true });
  await invalidateResource(client, queryKey);
}
const readers = new WeakMap<QueryClient, number>();

export function useTtsEnvironments(enabled = true) {
  const client = useQueryClient(), connection = useEventStreamStatus();
  const query = useResourceQuery({ queryKey: keys.snapshot, enabled,
    queryFn: async ({ signal }) => reconcileSnapshot(client, await api.snapshot(signal)) });
  useEventStream<TtsEnvironmentChanged>(EVENT_TYPES.TTS_ENVIRONMENT_CHANGED, event => {
    if (!enabled) return;
    void refreshResource(client, keys.snapshot);
    if (event.operation_id) void refreshResource(client, keys.operation(event.operation_id));
  });
  useEffect(() => {
    if (!enabled) return;
    readers.set(client, (readers.get(client) || 0) + 1);
    return () => {
      const remaining = (readers.get(client) || 1) - 1;
      readers.set(client, remaining);
      if (!remaining) {
        // No subscriber can observe changes while this cached snapshot is closed.
        void client.cancelQueries({ queryKey: keys.snapshot, exact: true });
        void client.invalidateQueries({ queryKey: keys.snapshot, exact: true, refetchType: 'none' });
      }
    };
  }, [client, enabled]);
  const previousConnection = useRef(connection), connected = useRef(connection === 'connected');
  useEffect(() => {
    if (enabled && connection === 'connected') {
      if (connected.current && previousConnection.current !== 'connected') void refreshResource(client, keys.snapshot);
      connected.current = true;
    }
    previousConnection.current = connection;
  }, [client, connection, enabled]);
  return query;
}

export function useTtsEnvironmentOperation(id: string | undefined, enabled = true) {
  const client = useQueryClient();
  const query = useResourceQuery({ queryKey: keys.operation(id || ''), enabled: enabled && !!id,
    queryFn: async ({ signal }) => latestOperation(client.getQueryData<TtsEnvironmentOperation>(keys.operation(id!)), await api.operation(id!, signal)),
    pollInterval: operation => operation && ttsEnvironmentActive(operation) ? 1500 : false });
  const terminal = useRef('');
  useEffect(() => {
    const operation = query.data;
    if (!operation || ttsEnvironmentActive(operation)) return;
    const identity = `${operation.id}:${operation.updated_at}`;
    if (terminal.current === identity) return;
    terminal.current = identity;
    rememberOperation(client, operation);
    void invalidateResource(client, keys.snapshot);
  }, [client, query.data]);
  return query;
}

type Action = { kind: 'check'; engine: TtsEngine } | { kind: 'prepare'; engine: TtsEngine } | { kind: 'cancel'; id: string } | { kind: 'retry'; id: string } | { kind: 'default'; engine: TtsEngine; environmentId: string };
const writes = new WeakSet<QueryClient>();
export function useTtsEnvironmentActions() {
  const client = useQueryClient();
  const pending = useIsMutating({ mutationKey: keys.write }) > 0;
  const mutation = useMutation({ mutationKey: keys.write, retry: false, mutationFn: async (action: Action) => {
    if (writes.has(client)) return;
    writes.add(client);
    try {
      if (action.kind === 'default') {
        const snapshot = await api.setDefault(action.engine, action.environmentId);
        await client.cancelQueries({ queryKey: keys.snapshot, exact: true });
        client.setQueryData(keys.snapshot, reconcileSnapshot(client, snapshot));
      } else {
        const operation = action.kind === 'check' ? await api.check({ engine: action.engine })
          : action.kind === 'prepare' ? await api.prepare({ engine: action.engine, make_default: true })
            : action.kind === 'cancel' ? await api.cancel(action.id) : await api.retry(action.id);
        await Promise.all([client.cancelQueries({ queryKey: keys.snapshot, exact: true }), client.cancelQueries({ queryKey: keys.operation(operation.id), exact: true })]);
        rememberOperation(client, operation);
      }
    } finally {
      writes.delete(client);
      // A lost response may still have started an operation on the server.
      void invalidateResource(client, keys.snapshot);
    }
  } });
  return { run: mutation.mutate, error: mutation.error, reset: mutation.reset, pending };
}
