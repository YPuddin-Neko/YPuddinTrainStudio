import { ApiError, ApiErrorPayload } from './types';
import { formatApiError } from '../utils/errors';

interface RequestOptions extends RequestInit {
  params?: Record<string, string | number | boolean | undefined>;
  silent?: boolean;
  /** Milliseconds to wait for the answer; then the request fails with a RequestTimeout. */
  timeout?: number;
}

/** How long a page section waits for its data before it offers a retry instead of loading on. */
export const READ_TIMEOUT_MS = 60_000;

export class RequestTimeout extends Error {
  constructor() {
    super('The service did not answer in time');
    this.name = 'TimeoutError';
  }
}

/** The caller's signal, plus an abort when the time is up. */
function deadline(signal: AbortSignal | null | undefined, timeout: number) {
  const controller = new AbortController();
  let expired = false;
  const timer = window.setTimeout(() => { expired = true; controller.abort(); }, timeout);
  const forward = () => controller.abort();
  if (signal?.aborted) controller.abort(); else signal?.addEventListener('abort', forward, { once: true });
  return {
    signal: controller.signal,
    expired: () => expired,
    clear: () => { window.clearTimeout(timer); signal?.removeEventListener('abort', forward); },
  };
}

export function apiUrl(endpoint: string): string {
  const base = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '');
  return new URL(`${base}${endpoint}`, window.location.origin).toString();
}

async function request<T>(method: string, endpoint: string, body?: unknown, options: RequestOptions = {}): Promise<T> {
  const { params, silent, timeout, ...init } = options;
  const url = new URL(apiUrl(endpoint));
  Object.entries(params || {}).forEach(([key, value]) => {
    if (value !== undefined) url.searchParams.set(key, String(value));
  });
  const isFormData = body instanceof FormData;
  const limit = timeout ? deadline(init.signal, timeout) : null;
  try {
    const response = await fetch(url, {
      ...init,
      ...(limit ? { signal: limit.signal } : {}),
      method,
      headers: { ...(isFormData ? {} : { 'Content-Type': 'application/json' }), ...init.headers },
      ...(body === undefined ? {} : { body: isFormData ? body : JSON.stringify(body) }),
    });
    if (!response.ok) {
      let payload: ApiErrorPayload['error'] | undefined;
      try { payload = (await response.json())?.error; } catch { /* Non-JSON error. */ }
      throw new ApiError(response.status, payload || {
        code: `http_${response.status}`, message: response.statusText || `HTTP ${response.status}`,
      });
    }
    return response.status === 204 ? undefined as T : await response.json() as T;
  } catch (caught) {
    const error = limit?.expired() ? new RequestTimeout() : caught;
    if (!silent && !(error instanceof Error && error.name === 'AbortError')) {
      window.dispatchEvent(new CustomEvent('api.error', { detail: formatApiError(error) }));
    }
    throw error;
  } finally {
    limit?.clear();
  }
}

export const apiClient = {
  get: <T>(endpoint: string, options?: RequestOptions) => request<T>('GET', endpoint, undefined, options),
  post: <T>(endpoint: string, body: unknown, options?: RequestOptions) => request<T>('POST', endpoint, body, options),
  put: <T>(endpoint: string, body: unknown, options?: RequestOptions) => request<T>('PUT', endpoint, body, options),
  patch: <T>(endpoint: string, body: unknown, options?: RequestOptions) => request<T>('PATCH', endpoint, body, options),
  delete: <T>(endpoint: string, options?: RequestOptions) => request<T>('DELETE', endpoint, undefined, options),
};
