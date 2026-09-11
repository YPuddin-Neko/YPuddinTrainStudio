import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import type { ProjectVersion, VersionedProject } from '../../utils/projectVersions';

export function useProjectVersions(project: VersionedProject | null, versionId?: string) {
  const enabled = !!project && (!!project.active_version_id || !!versionId);
  const query = useQuery({
    queryKey: ['project-versions', project?.id],
    queryFn: () => apiClient.get<ProjectVersion[]>(`/projects/${project!.id}/versions`, { params: { include_archived: true }, silent: true }),
    enabled,
    refetchInterval: query => query.state.data?.some(item => item.status === 'copying') ? 1200 : false,
  });
  const versions = query.data || [];
  const current = versions.find(item => item.id === (versionId || project?.active_version_id));
  return { versions, current, enabled, loading: enabled && query.isPending, error: query.error, refresh: query.refetch };
}
