import { apiClient } from '../api/client';
import { uploadBlob } from '../api/upload';
import { ApiError, type DatasetInfo } from '../api/types';
import type { DatasetUploadFile } from './datasetFiles';

export interface DatasetUploadProgress {
  bytesDone: number;
  bytesTotal: number;
  filesDone: number;
  filesTotal: number;
  bytesPerSecond: number | null;
  complete: boolean;
}
interface DatasetUploadOptions {
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
const networkFailure = (error: unknown) => error instanceof TypeError && /fetch|network|load failed/i.test(error.message);
const gatewayFailure = (error: unknown) => error instanceof ApiError && [502, 503, 504].includes(error.status) && error.code === `http_${error.status}`;

function checkAbort(signal: AbortSignal) {
  if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
}
function retryDelay(signal: AbortSignal, attempt: number) {
  return new Promise<void>((resolve, reject) => {
    checkAbort(signal);
    const aborted = () => { window.clearTimeout(timer); reject(new DOMException('Aborted', 'AbortError')); };
    const timer = window.setTimeout(() => { signal.removeEventListener('abort', aborted); resolve(); }, 300 * attempt);
    signal.addEventListener('abort', aborted, { once: true });
  });
}

/** Stage every file before committing so captions, masks and folders stay together. */
export async function uploadDataset(projectId: string, files: DatasetUploadFile[], options: DatasetUploadOptions, signal: AbortSignal, onProgress: (progress: DatasetUploadProgress) => void): Promise<DatasetInfo & { datasets?: DatasetInfo[] }> {
  const endpoint = `/projects/${projectId}/datasets/upload-sessions`;
  let session: { id: string; chunk_bytes: number };
  checkAbort(signal);
  try {
    session = await apiClient.post(endpoint, { ...options, files: files.map(({ file, relativePath }) => ({ name: relativePath, size: file.size })) }, { signal, silent: true });
  } catch (error) {
    if (error instanceof ApiError && [404, 405].includes(error.status) && ['http_404', 'http_405', 'http.error'].includes(error.code)) {
      throw new ApiError(error.status, { code: 'upload.update_required', message: 'Chunked uploads require an updated training service.' });
    }
    throw error;
  }
  const sessionEndpoint = `${endpoint}/${encodeURIComponent(session.id)}`;
  let succeeded = false;
  try {
    if (!Number.isSafeInteger(session.chunk_bytes) || session.chunk_bytes < 1) throw new Error('Invalid upload chunk size');
    let chunkBytes = Math.min(MAX_CHUNK_BYTES, session.chunk_bytes);
    const bytesTotal = files.reduce((sum, { file }) => sum + file.size, 0);
    const started = performance.now();
    let confirmed = 0;
    let filesDone = 0;
    const report = (currentBytes = 0, complete = false) => {
      const elapsed = (performance.now() - started) / 1000;
      const bytesDone = confirmed + currentBytes;
      onProgress({ bytesDone, bytesTotal, filesDone, filesTotal: files.length, bytesPerSecond: elapsed > 0 ? bytesDone / elapsed : null, complete });
    };
    report();
    for (let index = 0; index < files.length; index++) {
      const { file } = files[index];
      let offset = 0;
      while (offset < file.size) {
        checkAbort(signal);
        const size = Math.min(chunkBytes, file.size - offset);
        let attempts = 0;
        let response: { received: number } | undefined;
        while (!response) {
          try {
            response = await uploadBlob(`${sessionEndpoint}/files/${index}?offset=${offset}`, file.slice(offset, offset + size), signal, loaded => report(offset + loaded));
          } catch (error) {
            if (error instanceof ApiError && error.status === 413 && error.code === 'http_413' && size > MIN_CHUNK_BYTES) {
              chunkBytes = Math.max(MIN_CHUNK_BYTES, Math.floor(size / 2));
              break;
            }
            if (!networkFailure(error) || ++attempts > 2) throw error;
            await retryDelay(signal, attempts);
          }
        }
        if (!response) continue;
        if (response.received < offset + size || response.received > file.size) throw new Error('Invalid upload offset');
        offset = response.received;
        report(offset);
      }
      confirmed += file.size;
      filesDone++;
      report();
    }
    checkAbort(signal);
    report(0, true);
    for (let attempt = 0; ; attempt++) {
      try {
        const result = await apiClient.post<DatasetInfo & { datasets?: DatasetInfo[] }>(`${sessionEndpoint}/complete`, {}, { signal, silent: true });
        succeeded = true;
        return result;
      } catch (error) {
        if (!networkFailure(error) && !gatewayFailure(error)) throw error;
        if (attempt >= 2) throw new ApiError(error instanceof ApiError ? error.status : 0, { code: 'upload.result_unconfirmed', message: 'The import result could not be confirmed.' });
        await retryDelay(signal, attempt + 1);
      }
    }
  } finally {
    if (!succeeded) {
      // Use a fresh signal: cancellation must also release the staged files.
      const cleanup = new AbortController();
      const timer = window.setTimeout(() => cleanup.abort(), 3000);
      void apiClient.delete(sessionEndpoint, { signal: cleanup.signal, silent: true }).catch(() => {}).finally(() => window.clearTimeout(timer));
    }
  }
}
