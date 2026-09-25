import { projectUrl } from './projectVersions';

export function datasetReturnTarget(candidate: unknown, project?: string | null, version?: string | null) {
  if (typeof candidate !== 'string' || !project) return null;
  const base = projectUrl(project, version);
  const pathname = candidate.split(/[?#]/, 1)[0];
  return pathname === base || pathname === `${base}/train` ? candidate : null;
}
