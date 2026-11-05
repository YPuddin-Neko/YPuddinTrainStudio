import { useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import type { ProjectVersion, VersionedProject } from '../../utils/projectVersions';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';

export function useProjectArchiveEvents() {
  const client = useQueryClient();
  const changed = (event: { project_id?: string }) => {
    if (event.project_id) {
      void client.invalidateQueries({ queryKey: ['project', event.project_id] }, { cancelRefetch: false });
      void client.invalidateQueries({ queryKey: ['project-versions', event.project_id] }, { cancelRefetch: false });
    }
    void client.invalidateQueries({ queryKey: ['archived-versions'] }, { cancelRefetch: false });
    void client.invalidateQueries({ queryKey: ['archive-projects'] }, { cancelRefetch: false });
  };
  useEventStream(EVENT_TYPES.PROJECTS_CHANGED, changed);
  useEventStream(EVENT_TYPES.VERSIONS_CHANGED, changed);
}

export function useProjectVersions(project: VersionedProject | null, versionId?: string) {
  const enabled = !!project && (!!project.active_version_id || !!versionId || !!project.version_count);
  const query = useQuery({
    queryKey: ['project-versions', project?.id],
    queryFn: () => apiClient.get<ProjectVersion[]>(`/projects/${project!.id}/versions`, { params: { include_archived: true }, silent: true }),
    enabled,
    refetchInterval: query => query.state.data?.some(item => item.status === 'copying' || item.deletion?.state === 'deleting') ? 1200 : false,
  });
  const versions = query.data || [];
  const current = versions.find(item => item.id === (versionId || project?.active_version_id));
  return { versions, current, enabled, loading: enabled && query.isPending, error: query.error, refresh: query.refetch };
}
