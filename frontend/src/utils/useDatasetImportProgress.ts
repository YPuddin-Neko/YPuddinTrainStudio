import React from 'react';
import { apiClient } from '../api/client';
import { ApiError } from '../api/types';
import type { components } from '../api/generated';
import type { DatasetUploadProgress } from './uploadDataset';

export type DatasetImportProgressSnapshot = components['schemas']['DatasetImportProgress'];
export type DatasetImportPhase = DatasetImportProgressSnapshot['phase'];
export interface DatasetImportOperation {
  id: string;
  mode: 'upload' | 'path';
  state: 'active' | 'completed' | 'failed';
  snapshot: DatasetImportProgressSnapshot | null;
  elapsed: number;
  unavailable: boolean;
  upload?: DatasetUploadProgress;
  unconfirmed?: boolean;
  /** Creating the upload session, before any byte is sent. */
  preparing?: boolean;
  /** The busy state the import waits for, such as version.indexing. */
  waiting?: string;
}
interface ProgressRun {
  id: string;
  started: number;
  stopped: boolean;
  controller: AbortController | null;
  nextPoll?: number;
  pollTimeout?: number;
  clock?: number;
}

function stopRun(run: ProgressRun | null) {
  if (!run) return;
  run.stopped = true;
  window.clearTimeout(run.nextPoll);
  window.clearTimeout(run.pollTimeout);
  window.clearInterval(run.clock);
  run.controller?.abort();
}

export function createProgressId(): string | undefined {
  try {
    if (typeof globalThis.crypto?.randomUUID === 'function') return globalThis.crypto.randomUUID();
  } catch { /* Some embedded browsers expose an unavailable secure-context method. */ }
  try {
    // getRandomValues remains available on plain HTTP LAN origins.
    if (typeof globalThis.crypto?.getRandomValues !== 'function') return undefined;
    const bytes = globalThis.crypto.getRandomValues(new Uint8Array(16));
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('');
    return `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20)}`;
  } catch { return undefined; }
}

/** Observe the existing import request; progress failures never replace its actual result. */
export function useDatasetImportProgress(projectId: string) {
  const [operation, setOperation] = React.useState<DatasetImportOperation | null>(null);
  const active = React.useRef<ProgressRun | null>(null);
  const localSequence = React.useRef(0);

  React.useEffect(() => () => { stopRun(active.current); active.current = null; }, [projectId]);

  const start = React.useCallback((mode: 'upload' | 'path') => {
    stopRun(active.current);
    const progressId = createProgressId();
    // The local key only distinguishes React state; it is never sent to the server.
    const run: ProgressRun = { id: progressId || `local:${++localSequence.current}`, started: performance.now(), stopped: false, controller: null };
    active.current = run;
    setOperation({ id: run.id, mode, state: 'active', snapshot: null, elapsed: 0, unavailable: !progressId });
    const current = () => active.current === run && !run.stopped;
    const elapsed = () => Math.max(0, (performance.now() - run.started) / 1000);
    run.clock = window.setInterval(() => {
      if (current()) setOperation(previous => previous?.id === run.id ? { ...previous, elapsed: Math.max(previous.elapsed, elapsed()) } : previous);
    }, 1000);
    let notFound = 0;
    let failures = 0;
    let pollDelay = 500;
    const poll = async () => {
      if (!current()) return;
      const controller = new AbortController();
      run.controller = controller;
      run.pollTimeout = window.setTimeout(() => controller.abort(), 3000);
      let again = false;
      try {
        const snapshot = await apiClient.get<DatasetImportProgressSnapshot>(`/projects/${projectId}/datasets/import-progress/${run.id}`, { silent: true, signal: controller.signal });
        if (!current()) return;
        if (snapshot.id !== run.id) throw new Error('Unexpected import progress id');
        notFound = 0; failures = 0; pollDelay = 500;
        setOperation(previous => previous?.id === run.id ? { ...previous, snapshot, unavailable: false, elapsed: Math.max(elapsed(), Number.isFinite(snapshot.elapsed_seconds) ? snapshot.elapsed_seconds : 0) } : previous);
        again = snapshot.phase !== 'completed' && snapshot.phase !== 'failed';
      } catch (failure) {
        if (!current()) return;
        // Proxies can buffer uploads before the server creates their progress record.
        const exhausted = failure instanceof ApiError && failure.status === 404 ? ++notFound >= 8 : ++failures >= 3;
        again = mode === 'upload' || !exhausted;
        if (exhausted) {
          pollDelay = Math.min(5000, pollDelay * 2);
          setOperation(previous => previous?.id === run.id ? { ...previous, unavailable: true } : previous);
        }
      } finally {
        window.clearTimeout(run.pollTimeout);
        run.controller = null;
        if (again && current()) run.nextPoll = window.setTimeout(() => { void poll(); }, pollDelay);
      }
    };
    if (progressId) void poll();
    return progressId;
  }, [projectId]);

  const updateUpload = React.useCallback((upload: DatasetUploadProgress) => {
    const run = active.current;
    if (!run || run.stopped) return;
    setOperation(previous => previous?.id === run.id ? { ...previous, upload } : previous);
  }, []);

  const finish = React.useCallback((state: 'completed' | 'failed', unconfirmed = false) => {
    const run = active.current;
    if (!run || run.stopped) return;
    stopRun(run);
    setOperation(previous => previous?.id === run.id ? { ...previous, state, unconfirmed, elapsed: Math.max(previous.elapsed, (performance.now() - run.started) / 1000) } : previous);
  }, []);

  const reset = React.useCallback(() => {
    stopRun(active.current); active.current = null;
    setOperation(null);
  }, []);

  return { operation, start, finish, reset, updateUpload };
}
