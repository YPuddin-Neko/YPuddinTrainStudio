import { Link, Navigate, useLocation, useParams, useSearchParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { Activity, ArrowRight, Loader2 } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { DatasetInfo, DatasetSource, FamilyInfo, JobListResponse } from '../../api/types';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import ProjectWorkspaceHeader from '../../components/projects/ProjectWorkspaceHeader';
import { useProjectVersions } from '../../components/projects/useProjectVersions';
import VersionResults from '../../components/projects/VersionResults';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import { modelConfigUrl, projectUrl, versionConfigUrl, type VersionedProject } from '../../utils/projectVersions';
import ProjectDataImport from './ProjectDataImport';
import ProjectOverview from './ProjectOverview';
import DatasetPipelinePanel from '../../components/datasets/DatasetPipelinePanel';
import ProjectDatasetCards, { type WorkspaceDataset } from '../../components/datasets/ProjectDatasetCards';
import '../../styles/project-workspace.css';
import './project-data.css';

export default function ProjectDetail() {
  const { id, versionId } = useParams<{ id: string; versionId: string }>();
  return <ProjectDetailContent key={`${id}/${versionId || 'active'}`} projectId={id || ''} versionId={versionId}/>;
}
function ProjectDetailContent({projectId: id, versionId}: {projectId: string; versionId?: string}) {
  const { t } = useTranslation();
  const text = useWorkspaceText();
  const location = useLocation();
  const [params] = useSearchParams();
  const legacyModels = params.get('step') === 'models';
  const step = params.get('step') === 'results' ? 'results' : params.get('step') === 'data' ? 'data' : 'overview';
  const projectQuery = useQuery({ queryKey: ['project', id], queryFn: () => apiClient.get<VersionedProject>(`/projects/${id}`, {silent:true}), enabled: !!id });
  const project = projectQuery.data || null;
  const versions = useProjectVersions(project, versionId);
  const archived = !!versions.current?.archived;
  const scopedReady = !versions.enabled || versions.current?.status === 'ready';
  const configQuery = useQuery({
    queryKey:['project-workspace-config',id,versionId],enabled:!!project && scopedReady,
    queryFn:()=>apiClient.get<Record<string,any>>(versionConfigUrl(id,versionId),{silent:true}),
  });
  const familiesQuery = useQuery({
    queryKey:['families'], queryFn:()=>apiClient.get<FamilyInfo[]>('/families',{silent:true}),
    enabled:!!project && scopedReady && step === 'data',
  });
  const datasetsQuery = useQuery({
    queryKey:['project-workspace-datasets',id,versionId],enabled:!!project && scopedReady,
    queryFn:async()=>{
      const rows=await apiClient.get<Array<DatasetInfo|DatasetSource>>(`/projects/${id}/datasets`,{params:{version_id:versionId,include_cache:false},silent:true});
      return rows.map(item=>'source' in item?item as DatasetInfo:{source:item as DatasetSource}) as WorkspaceDataset[];
    },
    refetchInterval:query=>query.state.data?.some(item=>item.index_status==='indexing')?2000:false,
  });
  const jobsQuery = useQuery({
    queryKey:['project-workspace-active-jobs',id,versionId],enabled:!!project && scopedReady,
    queryFn:()=>apiClient.get<JobListResponse>('/jobs',{params:{project_id:id,version_id:versionId,status:'running,pausing,cancelling',page:1,page_size:5},silent:true}),
    refetchInterval:query=>query.state.data?.items.length?2000:false,
  });
  const imported = () => { void configQuery.refetch(); void datasetsQuery.refetch(); void versions.refresh(); };
  useEventStream(EVENT_TYPES.DATASET_CHANGED, imported);
  useEventStream(EVENT_TYPES.JOB_STATE, () => {void jobsQuery.refetch();void versions.refresh();});
  if (projectQuery.isPending) return <div role="status" className="workspace-loading"><Loader2 size={16} className="animate-spin"/>{text('正在打开项目…','Opening project…')}</div>;
  if (!project) return <div role="alert" className="workspace-message error">{projectQuery.error ? formatApiError(projectQuery.error) : text('项目不存在','Project not found')}<button type="button" className="ui-btn ui-btn-sm" onClick={() => void projectQuery.refetch()}>{t('common.retry')}</button></div>;
  if (legacyModels) return <Navigate replace to={modelConfigUrl(id, versionId || project.active_version_id)} state={location.state}/>;
  if (!versionId && project.active_version_id) {
    const search = params.has('step') ? location.search : `${location.search}${location.search ? '&' : '?'}step=overview`;
    return <Navigate replace to={{ pathname: projectUrl(id, project.active_version_id), search, hash: location.hash }} state={location.state}/>;
  }
  const config = configQuery.data;
  const captionFormats = Array.isArray(familiesQuery.data) ? familiesQuery.data.find(family=>family.name === config?.model?.family)?.caption_formats : undefined;
  const datasets = datasetsQuery.data || [];
  const jobs = jobsQuery.data?.items || [];
  const activeJob = jobs.find(job => ['running','pausing','cancelling'].includes(job.status));
  const imageCount = datasets.reduce((count,dataset) => count + (dataset.stats?.images || 0),0);
  const captionCount = datasets.reduce((count,dataset) => count + (dataset.stats?.captioned || 0),0);
  const maskCount = datasets.reduce((count,dataset) => count + (dataset.stats?.masks || 0),0);
  const indexing = datasets.some(dataset => dataset.index_status === 'indexing');
  const unavailable = versions.enabled && !versions.loading && !versions.current;
  return <div className="project-workspace" data-testid="project-detail-page">
    <ProjectWorkspaceHeader project={project} versionId={versionId} versions={versions.versions} current={versions.current} active={step} refresh={versions.refresh} error={versions.error} status={step === 'data' && datasetsQuery.isSuccess ? <div className="project-data-summary" aria-label={text('本版本数据统计','Version data summary')}>{[{label:text('图片','Images'),value:imageCount},{label:text('标签','Captions'),value:captionCount},{label:text('遮罩','Masks'),value:maskCount}].map(item=><span key={item.label}>{item.label} <strong>{item.value}</strong></span>)}{indexing && <Loader2 size={12} className="animate-spin" aria-label={text('索引中','Indexing')}/>}</div> : undefined}/>
    {unavailable && <div role="alert" className="workspace-message error">{text('该版本不存在或不属于当前项目。','This version does not belong to this project.')}<Link className="ui-link" to={projectUrl(id)}>{text('返回当前版本','Return to current version')}</Link></div>}
    {scopedReady && [{key:'config',query:configQuery,label:text('版本配置读取失败','Version configuration could not be loaded')},{key:'datasets',query:datasetsQuery,label:text('数据集列表读取失败','Dataset list could not be loaded')},{key:'jobs',query:jobsQuery,label:text('活动任务读取失败','Active jobs could not be loaded')}].map(item=>item.query.error && <div key={item.key} role="alert" className="workspace-message error">{item.label}: {formatApiError(item.query.error)}<button type="button" className="ui-btn ui-btn-sm" aria-label={`${text('重试','Retry')} · ${item.label}`} onClick={()=>void item.query.refetch()}>{t('common.retry')}</button></div>)}
    {activeJob && <Link to={`/jobs/${activeJob.id}`} className="version-run-status"><Activity size={16}/><strong>{activeJob.name}</strong><span>{t(`queue.status.${activeJob.status}`,activeJob.status)}</span><span className="tabular-nums">{activeJob.progress?.step ?? 0} / {activeJob.progress?.total_steps ?? '—'}</span><span className="run-status-action">{text('查看训练监控','View training monitor')}<ArrowRight size={14}/></span></Link>}
    {scopedReady && !unavailable && <>
      {step === 'results' ? <VersionResults projectId={id} versionId={versionId} readOnly={archived}/> : configQuery.isPending || datasetsQuery.isPending ? <div className="workspace-loading" role="status"><Loader2 size={16} className="animate-spin"/>{text('正在读取版本数据…','Loading version data…')}</div> : configQuery.isError || !config || datasetsQuery.isError ? null : step === 'overview' ? <ProjectOverview project={project} version={versions.current} versionId={versionId} config={config} datasets={datasets}/> : <section className="version-data-section">

        {versionId ? <DatasetPipelinePanel projectId={id} versionId={versionId} config={config} readOnly={archived} datasets={datasets} onChanged={imported}
          datasetList={<ProjectDatasetCards datasets={datasets} projectId={id} versionId={versionId} onRefresh={imported}/>}
          importPanel={<ProjectDataImport key={`${id}/${versionId}`} projectId={id} versionId={versionId} onImported={imported} captionFormats={captionFormats}/>}/>
          : <div className={archived ? '' : 'version-data-layout'}><ProjectDatasetCards datasets={datasets} projectId={id} versionId={versionId} onRefresh={imported}/>{!archived && <ProjectDataImport key={`${id}/${versionId}`} projectId={id} versionId={versionId} onImported={imported} captionFormats={captionFormats}/>}</div>}
      </section>}
    </>}
  </div>;
}
