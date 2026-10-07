import React from 'react';
import { Navigate, useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import type { Project } from '../../api/types';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import { projectUrl } from '../../utils/projectVersions';
import { LoadingNote } from '../../components/Loading';

const ProjectDetail = React.lazy(() => import('./ProjectDetail'));
const TrainConfig = React.lazy(() => import('../TrainConfig/TrainConfig'));
const DatasetCuration = React.lazy(() => import('../Dataset/DatasetCuration'));
const TtsProjectWorkspace = React.lazy(() => import('../Tts/TtsProjectWorkspace'));

export default function ProjectRoute({ view = 'detail' }: { view?: 'detail' | 'train' | 'curate' }) {
  const { id = '', versionId } = useParams();
  const text = useWorkspaceText();
  const query = useQuery({ queryKey: ['project', id], queryFn: ({ signal }) => apiClient.get<Project>(`/projects/${encodeURIComponent(id)}`, { signal, silent: true }), enabled: !!id });
  if (query.isPending) return <LoadingNote block label={text('正在打开项目…', 'Opening project…')}/>;
  if (!query.data) return <div role="alert" className="workspace-message error">{formatApiError(query.error)}<button className="ui-btn ui-btn-sm" onClick={() => void query.refetch()}>{text('重试', 'Retry')}</button></div>;
  if (query.data.project_type === 'tts') return view === 'curate'
    ? <Navigate replace to={projectUrl(id, versionId, 'data')}/>
    : <TtsProjectWorkspace key={`${id}/${versionId || 'active'}`} project={query.data} versionId={versionId} training={view === 'train'}/>;
  return view === 'train' ? <TrainConfig/> : view === 'curate' ? <DatasetCuration/> : <ProjectDetail/>;
}
