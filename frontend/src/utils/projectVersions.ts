import { apiClient } from '../api/client';
import type { Project } from '../api/types';
import type { components } from '../api/generated';

export type VersionedProject = Project & { active_version_id?: string | null; version_count?: number; layout_version?: number };
export type ProjectVersion = components['schemas']['ProjectVersion'];
export function projectUrl(projectId: string, versionId?: string | null, step?: string) {
  const base = `/projects/${encodeURIComponent(projectId)}${versionId ? `/v/${encodeURIComponent(versionId)}` : ''}`;
  return step === 'train' ? `${base}/train` : step ? `${base}?step=${step}` : base;
}
export function versionConfigUrl(projectId: string, versionId?: string | null) {
  return `/projects/${projectId}/config${versionId ? `?version_id=${encodeURIComponent(versionId)}` : ''}`;
}
export function modelConfigUrl(projectId: string, versionId?: string | null) {
  return `${projectUrl(projectId, versionId, 'train')}?tab=model`;
}
export function configDifferences(left: unknown, right: unknown, prefix = ''): { path: string; before: unknown; after: unknown }[] {
  if (JSON.stringify(left) === JSON.stringify(right)) return [];
  const isRecord = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
  if (isRecord(left) && isRecord(right)) return [...new Set([...Object.keys(left), ...Object.keys(right)])].sort().flatMap(key => configDifferences(left[key], right[key], prefix ? `${prefix}.${key}` : key));
  return [{ path: prefix, before: left, after: right }];
}

const activationQueues = new Map<string, Promise<unknown>>();
const latestActivation = new Map<string, string>();
/** Serialize remembered-version writes; explicit route/API scopes remain authoritative. */
export async function activateProjectVersion(projectId: string, versionId: string) {
  latestActivation.set(projectId,versionId);
  const next = (activationQueues.get(projectId) || Promise.resolve()).catch(() => {}).then(async () => {
    if(latestActivation.get(projectId) !== versionId) return;
    return apiClient.patch<VersionedProject>(`/projects/${projectId}`,{active_version_id:versionId},{silent:true});
  });
  activationQueues.set(projectId,next);
  return next;
}
