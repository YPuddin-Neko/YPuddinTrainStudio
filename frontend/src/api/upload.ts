import { apiUrl } from './client';
import { ApiError, type ApiErrorPayload } from './types';

/** XHR exposes bytes sent while a proxy may still be buffering the request. */
export function uploadBlob(endpoint: string, body: Blob, signal: AbortSignal, onProgress: (loaded: number) => void): Promise<{ received: number }> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) { reject(new DOMException('Aborted', 'AbortError')); return; }
    const xhr = new XMLHttpRequest();
    const abort = () => { finish(new DOMException('Aborted', 'AbortError')); xhr.abort(); };
    const finish = (error?: unknown, result?: { received: number }) => {
      signal.removeEventListener('abort', abort);
      xhr.upload.onprogress = null;
      xhr.onload = xhr.onerror = xhr.onabort = xhr.ontimeout = null;
      if (error) reject(error); else resolve(result!);
    };
    xhr.upload.onprogress = event => onProgress(Math.min(body.size, event.loaded));
    xhr.onload = () => {
      let data: { error?: ApiErrorPayload['error']; received?: number } | undefined;
      try { data = JSON.parse(xhr.responseText); } catch { /* Gateways may return an HTML error. */ }
      if (xhr.status < 200 || xhr.status >= 300) {
        finish(new ApiError(xhr.status, data?.error || { code: `http_${xhr.status}`, message: xhr.statusText || `HTTP ${xhr.status}` }));
      } else if (!Number.isSafeInteger(data?.received) || data!.received! < 0) {
        finish(new Error('Invalid upload response'));
      } else finish(undefined, { received: data!.received! });
    };
    xhr.onerror = () => finish(new TypeError('Failed to fetch'));
    xhr.ontimeout = () => finish(new TypeError('Failed to fetch'));
    xhr.onabort = () => finish(new DOMException('Aborted', 'AbortError'));
    xhr.open('PUT', apiUrl(endpoint));
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    signal.addEventListener('abort', abort, { once: true });
    try { xhr.send(body); } catch (error) { finish(error); }
  });
}
