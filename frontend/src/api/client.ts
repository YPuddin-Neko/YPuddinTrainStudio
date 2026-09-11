import { ApiError, ApiErrorPayload } from './types';
import { formatApiError } from '../utils/errors';

interface RequestOptions extends RequestInit {
  params?: Record<string, string | number | boolean | undefined>;
  silent?: boolean;
}

export function apiUrl(endpoint: string): string {
  const base = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '');
  return new URL(`${base}${endpoint}`, window.location.origin).toString();
}

async function request<T>(method: string, endpoint: string, body?: unknown, options: RequestOptions = {}): Promise<T> {
  const { params, silent, ...init } = options;
  const url = new URL(apiUrl(endpoint));
  Object.entries(params || {}).forEach(([key, value]) => {
    if (value !== undefined) url.searchParams.set(key, String(value));
  });
  const isFormData = body instanceof FormData;
  try {
    const response = await fetch(url, {
      ...init,
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
  } catch (error) {
    if (!silent && !(error instanceof Error && error.name === 'AbortError')) {
      window.dispatchEvent(new CustomEvent('api.error', { detail: formatApiError(error) }));
    }
    throw error;
  }
}

export const apiClient = {
  get: <T>(endpoint: string, options?: RequestOptions) => request<T>('GET', endpoint, undefined, options),
  post: <T>(endpoint: string, body: unknown, options?: RequestOptions) => request<T>('POST', endpoint, body, options),
  put: <T>(endpoint: string, body: unknown, options?: RequestOptions) => request<T>('PUT', endpoint, body, options),
  patch: <T>(endpoint: string, body: unknown, options?: RequestOptions) => request<T>('PATCH', endpoint, body, options),
  delete: <T>(endpoint: string, options?: RequestOptions) => request<T>('DELETE', endpoint, undefined, options),
};
