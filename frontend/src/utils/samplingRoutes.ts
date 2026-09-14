export function samplingUrl(sourceJobId?: string | null, taskId?: string | null, projectId?: string | null, versionId?: string | null) {
  const params = new URLSearchParams();
  if (projectId) params.set('project_id', projectId);
  if (versionId) params.set('version_id', versionId);
  if (sourceJobId) params.set('source_job_id', sourceJobId);
  if (taskId) params.set('task_id', taskId);
  return `/sampling${params.size ? `?${params}` : ''}`;
}
