import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { useLocation } from 'react-router-dom';
import { apiClient } from '../../api/client';
import { ApiError } from '../../api/types';
import type { ProjectVersion, VersionedProject } from '../../utils/projectVersions';
import type { ProjectSidebarSelection } from './ProjectSidebarContext';
import ProjectWorkspaceHeader from './ProjectWorkspaceHeader';

export default function PersistentProjectSidebar({selection,beforeAction}: {
  selection: ProjectSidebarSelection; beforeAction: () => Promise<void>;
}) {
  const location = useLocation();
  const pageOwnsSelection = selection.routeKey === location.key;
  const projectId = selection.project.id;
  const routeProject = location.pathname.match(/^\/projects\/([^/]+)/)?.[1];
  const unresolvedWorkspace = !pageOwnsSelection && (routeProject && routeProject !== encodeURIComponent(projectId)
    || location.pathname.startsWith('/datasets/') && location.pathname !== selection.pathname);
  const refreshWhileAway = !pageOwnsSelection && !unresolvedWorkspace;
  const projectQuery = useQuery({
    queryKey:['project',projectId],
    queryFn:({signal})=>apiClient.get<VersionedProject>(`/projects/${projectId}`,{signal,silent:true}),
    enabled:refreshWhileAway, placeholderData:selection.project, retry:false,
  });
  const versionsQuery = useQuery({
    queryKey:['project-versions',projectId],
    queryFn:({signal})=>apiClient.get<ProjectVersion[]>(`/projects/${projectId}/versions`,{params:{include_archived:true},signal,silent:true}),
    enabled:refreshWhileAway && !!(selection.versionId || selection.project.active_version_id),
    placeholderData:selection.versions, retry:false,
    refetchInterval:query=>refreshWhileAway && query.state.data?.some(item=>item.status==='copying') ? 1200 : false,
  });
  const refreshProject = projectQuery.refetch;
  const refreshVersions = versionsQuery.refetch;
  React.useEffect(()=>{
    if(!refreshWhileAway)return;
    void refreshProject();
    if(selection.versionId || selection.project.active_version_id)void refreshVersions();
  },[location.key,refreshWhileAway,refreshProject,refreshVersions,selection.versionId,selection.project.active_version_id]);
  const project = pageOwnsSelection ? selection.project : projectQuery.data || selection.project;
  const versions = pageOwnsSelection ? selection.versions : versionsQuery.data || selection.versions;
  const versionId = !pageOwnsSelection && versionsQuery.isSuccess && selection.versionId && !versions.some(item=>item.id===selection.versionId)
    ? project.active_version_id || undefined : selection.versionId;
  const current = versions.find(item=>item.id===(versionId || project.active_version_id)) || (pageOwnsSelection ? selection.current : undefined);
  const missing = !pageOwnsSelection && projectQuery.error instanceof ApiError && projectQuery.error.status === 404;
  if(unresolvedWorkspace || missing || !pageOwnsSelection && project.archived)return null;
  return <ProjectWorkspaceHeader sidebarOnly project={project} versionId={versionId} versions={versions} current={current}
    active={selection.active} workflowActive={pageOwnsSelection} beforeAction={beforeAction}
    refresh={async()=>{await Promise.all([projectQuery.refetch(),versionsQuery.refetch()]);}}
    error={pageOwnsSelection ? undefined : projectQuery.error || versionsQuery.error}/>;
}
