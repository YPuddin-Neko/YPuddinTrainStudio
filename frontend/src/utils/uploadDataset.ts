import { apiClient } from '../api/client';
import { uploadBlob } from '../api/upload';
import { ApiError, type DatasetInfo } from '../api/types';
import type { components } from '../api/generated';
import type { DatasetUploadFile } from './datasetFiles';

export type DatasetUploadResult = DatasetInfo & { datasets?: DatasetInfo[] };
export type UploadSessionStatus = components['schemas']['DatasetUploadSessionStatus'];
export interface DatasetUploadProgress {
  bytesDone: number;
  bytesTotal: number;
  filesDone: number;
  filesTotal: number;
  bytesPerSecond: number | null;
  complete: boolean;
}
export interface DatasetUploadOptions {
  version_id?: string;
  target_dataset_id?: string;
  progress_id?: string;
  name?: string;
  repeats?: number;
  is_reg?: boolean;
  prior_weight?: number;
  class_prompt?: string;
  caption_ext: string;
}
const MIN_CHUNK_BYTES = 64 * 1024;
const MAX_CHUNK_BYTES = 8 * 1024 ** 2;
/** Files in flight at once: many small images otherwise wait one round trip each behind a proxy. */
export const UPLOAD_CONCURRENCY = 4;
const networkFailure = (error: unknown) => error instanceof TypeError && /fetch|network|load failed/i.test(error.message);
const gatewayFailure = (error: unknown) => error instanceof ApiError && [502, 503, 504].includes(error.status) && error.code === `http_${error.status}`;
export const isNetworkFailure = (error: unknown) => networkFailure(error) || gatewayFailure(error);

function checkAbort(signal: AbortSignal) {
  if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
}
export function waitFor(signal: AbortSignal, milliseconds: number) {
  return new Promise<void>((resolve, reject) => {
    checkAbort(signal);
    const aborted = () => { window.clearTimeout(timer); reject(new DOMException('Aborted', 'AbortError')); };
    const timer = window.setTimeout(() => { signal.removeEventListener('abort', aborted); resolve(); }, milliseconds);
    signal.addEventListener('abort', aborted, { once: true });
  });
}
const retryDelay = (signal: AbortSignal, attempt: number) => waitFor(signal, 300 * attempt);

export const sessionEndpoint = (projectId: string, sessionId: string) => `/projects/${projectId}/datasets/upload-sessions/${encodeURIComponent(sessionId)}`;

/** Register the manifest; nothing is imported until every byte has arrived and "complete" is called. */
export async function createUploadSession(projectId: string, files: DatasetUploadFile[], options: DatasetUploadOptions, signal: AbortSignal): Promise<{ id: string; chunk_bytes: number }> {
  checkAbort(signal);
  try {
    return await apiClient.post(`/projects/${projectId}/datasets/upload-sessions`, { ...options, files: files.map(({ file, relativePath }) => ({ name: relativePath, size: file.size })) }, { signal, silent: true });
  } catch (error) {
    if (error instanceof ApiError && [404, 405].includes(error.status) && ['http_404', 'http_405', 'http.error'].includes(error.code)) {
      throw new ApiError(error.status, { code: 'upload.update_required', message: 'Chunked uploads require an updated training service.' });
    }
    throw error;
  }
}

export function readUploadSession(projectId: string, sessionId: string, signal?: AbortSignal) {
  return apiClient.get<UploadSessionStatus>(sessionEndpoint(projectId, sessionId), { signal, silent: true });
}

/** Release staged files; a fresh signal, because cancelling the upload aborted its own. */
export function deleteUploadSession(projectId: string, sessionId: string) {
  const cleanup = new AbortController();
  const timer = window.setTimeout(() => cleanup.abort(), 3000);
  return apiClient.delete(sessionEndpoint(projectId, sessionId), { signal: cleanup.signal, silent: true }).finally(() => window.clearTimeout(timer));
}

interface SendOptions {
  endpoint: string;
  files: DatasetUploadFile[];
  /** Bytes the server already holds for each file; sending continues from there. */
  received: number[];
  chunkBytes: number;
  signal: AbortSignal;
  onProgress: (progress: DatasetUploadProgress) => void;
  concurrency?: number;
  /** Resend the bytes before each partial offset: the server refuses them if the files changed. */
  verify?: boolean;
}

/** Send every missing byte. Chunks of one file stay in order; several files travel at once. */
export async function sendUploadFiles({ endpoint, files, received, chunkBytes, signal, onProgress, concurrency = UPLOAD_CONCURRENCY, verify = false }: SendOptions): Promise<void> {
  if (!Number.isSafeInteger(chunkBytes) || chunkBytes < 1) throw new Error('Invalid upload chunk size');
  let size = Math.min(MAX_CHUNK_BYTES, chunkBytes);
  const offsets = files.map(({ file }, index) => Math.min(Math.max(0, received[index] || 0), file.size));
  const loading = new Map<number, number>();
  const bytesTotal = files.reduce((sum, { file }) => sum + file.size, 0);
  let filesDone = files.filter(({ file }, index) => offsets[index] >= file.size).length;
  const sum = (values: Iterable<number>) => { let total = 0; for (const value of values) total += value; return total; };
  const startBytes = sum(offsets);
  const started = performance.now();
  const controller = new AbortController();
  const abort = () => controller.abort();
  signal.addEventListener('abort', abort, { once: true });
  let stopped = false;
  const report = () => {
    if (stopped) return;
    const bytesDone = Math.min(bytesTotal, sum(offsets) + sum(loading.values()));
    const elapsed = (performance.now() - started) / 1000;
    // The first seconds say little about the speed; no rate (and no time left) until then.
    onProgress({ bytesDone, bytesTotal, filesDone, filesTotal: files.length, bytesPerSecond: elapsed >= 2 ? Math.max(0, bytesDone - startBytes) / elapsed : null, complete: false });
  };
  const put = async (index: number, offset: number, body: Blob, counted = true) => {
    for (let attempts = 0; ; ) {
      try {
        return await uploadBlob(`${endpoint}/files/${index}?offset=${offset}`, body, controller.signal, loaded => { if (counted) { loading.set(index, loaded); report(); } });
      } catch (error) {
        loading.delete(index);
        if (!isNetworkFailure(error) || ++attempts > 2) throw error;
        await retryDelay(controller.signal, attempts);
      }
    }
  };
  const sendFile = async (index: number) => {
    const { file } = files[index];
    if (verify && offsets[index] > 0) {
      // A replay of received bytes is compared byte for byte; a changed file gets a chunk conflict.
      const start = Math.max(0, offsets[index] - MIN_CHUNK_BYTES);
      const response = await put(index, start, file.slice(start, offsets[index]), false);
      if (response.received !== offsets[index]) throw new Error('Invalid upload offset');
    }
    while (offsets[index] < file.size) {
      checkAbort(controller.signal);
      const offset = offsets[index];
      const length = Math.min(size, file.size - offset);
      let response: { received: number };
      try {
        response = await put(index, offset, file.slice(offset, offset + length));
      } catch (error) {
        if (error instanceof ApiError && error.status === 413 && error.code === 'http_413' && length > MIN_CHUNK_BYTES) {
          // A proxy limit: every file continues with smaller chunks from the same offsets.
          size = Math.min(size, Math.max(MIN_CHUNK_BYTES, Math.floor(length / 2)));
          continue;
        }
        throw error;
      }
      if (response.received < offset + length || response.received > file.size) throw new Error('Invalid upload offset');
      offsets[index] = response.received;
      loading.delete(index);
      report();
    }
    filesDone++;
    report();
  };
  const queue = files.flatMap(({ file }, index) => offsets[index] < file.size ? [index] : []);
  let next = 0;
  const worker = async () => { while (next < queue.length) await sendFile(queue[next++]); };
  report();
  try {
    await Promise.all(Array.from({ length: Math.min(Math.max(1, concurrency), queue.length) }, worker));
    checkAbort(signal);
  } catch (error) {
    controller.abort();
    throw signal.aborted ? new DOMException('Aborted', 'AbortError') : error;
  } finally {
    stopped = true;
    signal.removeEventListener('abort', abort);
  }
  onProgress({ bytesDone: bytesTotal, bytesTotal, filesDone: files.length, filesTotal: files.length, bytesPerSecond: null, complete: true });
}

/** Ask the server how an import it may still be running ended, instead of importing again. */
export async function settleUploadSession(projectId: string, sessionId: string, signal: AbortSignal, onImporting?: () => void): Promise<UploadSessionStatus> {
  let failures = 0;
  for (let delay = 1000; ; delay = Math.min(3000, delay + 500)) {
    try {
      const status = await readUploadSession(projectId, sessionId, signal);
      failures = 0;
      if (status.state !== 'finalizing') return status;
      onImporting?.();
    } catch (error) {
      if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
      if (error instanceof ApiError && error.status === 404) throw new ApiError(404, { code: 'upload.result_unconfirmed', message: 'The import result could not be confirmed.' });
      if (!isNetworkFailure(error)) throw error;
      // About a minute without an answer: report the result as unconfirmed, the session stays.
      if (++failures > 20) throw new ApiError(error instanceof ApiError ? error.status : 0, { code: 'upload.result_unconfirmed', message: 'The import result could not be confirmed.' });
    }
    await waitFor(signal, delay);
  }
}

/** Import the staged files once; a lost response is resolved by reading the session, never by a second import. */
export async function completeUpload(projectId: string, sessionId: string, signal: AbortSignal, onImporting?: () => void): Promise<DatasetUploadResult> {
  const endpoint = sessionEndpoint(projectId, sessionId);
  for (let attempt = 0; ; attempt++) {
    let lost: unknown;
    try {
      return await apiClient.post<DatasetUploadResult>(`${endpoint}/complete`, {}, { signal, silent: true });
    } catch (error) {
      if (signal.aborted || !isNetworkFailure(error)) throw error;
      lost = error;
    }
    const status = await settleUploadSession(projectId, sessionId, signal, onImporting);
    if (status.state === 'completed' && status.result) return status.result as DatasetUploadResult;
    if (status.error) throw new ApiError(409, { code: status.error.code, message: status.error.message, details: { retryable: status.error.retryable } });
    // The service has not imported anything: the request did not get through, so send it again.
    if (attempt >= 2) throw lost;
    await retryDelay(signal, attempt + 1);
  }
}
