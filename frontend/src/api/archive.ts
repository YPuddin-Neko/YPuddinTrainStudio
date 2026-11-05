import { apiClient } from './client';
import type { components } from './generated';

export type ArchivedVersion = components['schemas']['ArchivedProjectVersion'];

export const archivedVersions = (signal?: AbortSignal) => apiClient.get<ArchivedVersion[]>('/archived-versions', { signal, silent: true });
