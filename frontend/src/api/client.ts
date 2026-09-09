import { ApiError, ApiErrorPayload } from './types';

interface RequestOptions extends RequestInit {
  params?: Record<string, string | number | boolean | undefined>;
}

function getBaseUrl() {
  return import.meta.env.VITE_API_BASE_URL || '/api';
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let payload: ApiErrorPayload['error'] | undefined;
    try {
      const data = await response.json();
      if (data && data.error) {
        payload = data.error;
      }
    } catch (e) {
      // Ignore json parse error if not json
    }

    if (payload) {
      throw new ApiError(response.status, payload);
    }
    throw new ApiError(response.status, {
      code: `http_${response.status}`,
      message: response.statusText || 'Request failed',
    });
  }
  return response.json() as Promise<T>;
}

export const apiClient = {
  get: async <T>(endpoint: string, options: RequestOptions = {}): Promise<T> => {
    const { params, ...init } = options;
    const url = new URL(`${getBaseUrl()}${endpoint}`, window.location.origin);
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          url.searchParams.append(key, String(value));
        }
      });
    }
    const response = await fetch(url.toString(), {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      ...init,
    });
    return handleResponse<T>(response);
  },

  post: async <T>(endpoint: string, body: any, options: RequestOptions = {}): Promise<T> => {
    const { params, ...init } = options;
    const url = new URL(`${getBaseUrl()}${endpoint}`, window.location.origin);
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          url.searchParams.append(key, String(value));
        }
      });
    }
    const response = await fetch(url.toString(), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      ...init,
    });
    return handleResponse<T>(response);
  },

  put: async <T>(endpoint: string, body: any, options: RequestOptions = {}): Promise<T> => {
    const { params, ...init } = options;
    const url = new URL(`${getBaseUrl()}${endpoint}`, window.location.origin);
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          url.searchParams.append(key, String(value));
        }
      });
    }
    const response = await fetch(url.toString(), {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      ...init,
    });
    return handleResponse<T>(response);
  },

  delete: async <T>(endpoint: string, options: RequestOptions = {}): Promise<T> => {
    const { params, ...init } = options;
    const url = new URL(`${getBaseUrl()}${endpoint}`, window.location.origin);
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          url.searchParams.append(key, String(value));
        }
      });
    }
    const response = await fetch(url.toString(), {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
      ...init,
    });
    return handleResponse<T>(response);
  },

  patch: async <T>(endpoint: string, body: any, options: RequestOptions = {}): Promise<T> => {
    const { params, ...init } = options;
    const url = new URL(`${getBaseUrl()}${endpoint}`, window.location.origin);
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          url.searchParams.append(key, String(value));
        }
      });
    }
    const response = await fetch(url.toString(), {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      ...init,
    });
    return handleResponse<T>(response);
  },
};
