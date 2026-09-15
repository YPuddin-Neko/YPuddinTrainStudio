import React from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { ArrowRight, Grid2X2, RefreshCw, Search, X } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { Job, JobListResponse, Project } from '../../api/types';
import StudioSelect from '../../components/StudioSelect';
import XyzSampling from '../../components/sampling/XyzSampling';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import { projectUrl } from '../../utils/projectVersions';
import '../Queue/queue.css';
import './sampling-page.css';

type SourceJob = Job & { project_name?: string | null; version_name?: string | null };
const PAGE_SIZE = 50;

export default function Sampling() {
  const text = useWorkspaceText();
  const [params, setParams] = useSearchParams();
  const projectId = params.get('project_id') || '';
  const versionId = params.get('version_id') || '';
  const sourceId = params.get('source_job_id') || '';
  const query = params.get('q') || '';
  const page = Math.max(1, Number(params.get('page')) || 1);
  const [projects, setProjects] = React.useState<Project[]>([]);
  const [projectsError, setProjectsError] = React.useState('');
  const [projectsRevision, setProjectsRevision] = React.useState(0);
  const [jobs, setJobs] = React.useState<SourceJob[]>([]);
  const [sourceJob, setSourceJob] = React.useState<SourceJob | null>(null);
  const [total, setTotal] = React.useState(0);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState('');
  const [sourceError, setSourceError] = React.useState('');
  const [sourceRevision, setSourceRevision] = React.useState(0);
  const [scopeState, setScopeState] = React.useState<'checking' | 'ready' | 'archived' | 'unavailable' | 'not-ready' | 'busy'>('checking');
  const [scopeError, setScopeError] = React.useState('');
  const [revision, setRevision] = React.useState(0);
  const change = (patch: Record<string, string | null>) => {
    const next = new URLSearchParams(params);
    Object.entries(patch).forEach(([key, value]) => value ? next.set(key, value) : next.delete(key));
    setParams(next);
  };
  React.useEffect(() => {
    const controller = new AbortController(); setProjectsError('');
    void apiClient.get<Project[] | { items: Project[] }>('/projects', { signal: controller.signal, silent: true }).then(data => {
      if (!controller.signal.aborted) setProjects(Array.isArray(data) ? data : data.items);
    }).catch(failure => { if (!controller.signal.aborted) setProjectsError(formatApiError(failure)); });
    return () => controller.abort();
  }, [projectsRevision]);
  React.useEffect(() => {
    const controller = new AbortController(); setLoading(true); setError('');
    void apiClient.get<JobListResponse | SourceJob[]>('/jobs', { params: { type: 'train', project_id: projectId || undefined, version_id: versionId || undefined, q: query || undefined, page, page_size: PAGE_SIZE }, signal: controller.signal, silent: true }).then(data => {
      if (controller.signal.aborted) return;
      const rows = Array.isArray(data) ? data : data.items;
      setJobs(rows.filter(job => job.type === 'train')); setTotal(Array.isArray(data) ? rows.length : data.total);
    }).catch(failure => { if (!controller.signal.aborted) setError(formatApiError(failure)); }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [projectId, versionId, query, page, revision]);
  const invalidSourceMessage = text('请选择训练任务作为对比来源。', 'Choose a training run as the comparison source.');
  React.useEffect(() => {
    setSourceJob(current => current?.id === sourceId ? current : null); setSourceError(''); setScopeState('checking'); setScopeError('');
    if (!sourceId) return;
    const controller = new AbortController();
    void apiClient.get<SourceJob>(`/jobs/${encodeURIComponent(sourceId)}`, { signal: controller.signal, silent: true }).then(async job => {
      if (controller.signal.aborted) return;
      if (job.type !== 'train') { setSourceError(invalidSourceMessage); return; }
      setSourceJob(job);
      if (!job.project_id) { setScopeState('ready'); return; }
      try {
        const [project, version] = await Promise.all([
          apiClient.get<{ archived: boolean }>(`/projects/${encodeURIComponent(job.project_id)}`, { signal: controller.signal, silent: true }),
          job.version_id ? apiClient.get<{ archived: boolean; status: string; busy: boolean }>(`/projects/${encodeURIComponent(job.project_id)}/versions/${encodeURIComponent(job.version_id)}`, { signal: controller.signal, silent: true }) : Promise.resolve(null),
        ]);
        if (!controller.signal.aborted) setScopeState(project.archived || version?.archived ? 'archived' : version?.busy ? 'busy' : version && version.status !== 'ready' ? 'not-ready' : 'ready');
      } catch (failure) { if (!controller.signal.aborted) { setScopeState('unavailable'); setScopeError(formatApiError(failure)); } }
    }).catch(failure => { if (!controller.signal.aborted) setSourceError(formatApiError(failure)); });
    return () => controller.abort();
  }, [sourceId, invalidSourceMessage, sourceRevision]);
  const sources = sourceJob && !jobs.some(job => job.id === sourceJob.id) ? [sourceJob, ...jobs] : jobs;
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  return <section className="sampling-page task-workspace" data-testid="sampling-page">
    <header className="task-page-heading"><div><h1>{text('模型测试', 'Model testing')}</h1><p>{text('选择训练任务的底模与权重，对比采样参数；可指定另一张空闲显卡，与训练分别运行。', 'Compare sampling settings with a training run’s model and checkpoints. Choose a separate free GPU to generate while training.')}</p></div><Link className="task-button" to="/queue?type=xyz">{text('测试队列', 'Test queue')}<ArrowRight size={14}/></Link></header>
    <div className="sampling-source-panel">
      <div className="sampling-source-filters"><label><span>{text('项目', 'Project')}</span><StudioSelect aria-label={text('对比来源项目', 'Comparison project')} value={projectId} onValueChange={value => change({ project_id: value, version_id: null, source_job_id: null, task_id: null, page: null })} options={[{ value: '', label: text('所有项目', 'All projects') }, ...projects.map(project => ({ value: project.id, label: project.name })), ...(projectId && !projects.some(project => project.id === projectId) ? [{ value: projectId, label: projectId }] : [])]}/></label><label className="sampling-source-search"><Search size={14}/><input aria-label={text('搜索来源训练任务', 'Search source training runs')} placeholder={text('搜索任务名称…', 'Search run names…')} value={query} onChange={event => change({ q: event.target.value, page: null })}/></label><button type="button" className="task-button" disabled={loading} onClick={() => { setRevision(value => value + 1); setSourceRevision(value => value + 1); }} aria-label={text('刷新来源任务', 'Refresh source runs')}><RefreshCw size={14} className={loading ? 'animate-spin' : ''}/></button></div>
      {projectsError && <div className="sampling-source-error" role="alert"><span>{text('项目列表读取失败', 'Could not load projects')}: {projectsError}</span><button type="button" className="task-link" onClick={() => setProjectsRevision(value => value + 1)}>{text('重试项目列表', 'Retry projects')}</button></div>}
      {versionId && <div className="sampling-version-filter"><span>{text('当前版本', 'Current version')}: {sourceJob?.version_name || versionId}</span><button type="button" onClick={() => change({ version_id: null, page: null })} aria-label={text('清除版本筛选', 'Clear version filter')}><X size={12}/></button></div>}
      <div className="sampling-source-choice"><label><span>{text('来源训练任务', 'Source training run')}</span><StudioSelect searchable disabled={loading} aria-label={text('来源训练任务', 'Source training run')} value={sourceId} placeholder={text('选择训练任务', 'Choose a training run')} options={sources.map(job => ({ value: job.id, label: [job.name, job.project_name, job.version_name].filter(Boolean).join(' · ') }))} onValueChange={value => change({ source_job_id: value, task_id: null })}/></label>{pages > 1 && <nav className="task-actions" aria-label={text('来源任务分页', 'Source run pages')}><button type="button" className="task-button" disabled={loading || page <= 1} onClick={() => change({ page: String(page - 1) })}>{text('上一页', 'Previous')}</button><span>{page} / {pages}</span><button type="button" className="task-button" disabled={loading || page >= pages} onClick={() => change({ page: String(page + 1) })}>{text('下一页', 'Next')}</button></nav>}</div>
      {sourceJob && <div className="sampling-source-context"><span>{sourceJob.name}</span><Link to={`/jobs/${encodeURIComponent(sourceJob.id)}?tab=metrics`}>{text('查看训练任务', 'View training run')}</Link>{sourceJob.project_id && <Link to={projectUrl(sourceJob.project_id, sourceJob.version_id, 'results')}>{text('返回版本训练结果', 'Back to version results')}</Link>}</div>}
    </div>
    {(error || sourceError) && <div className="task-error" role="alert">{error || sourceError}<button type="button" onClick={() => { setRevision(value => value + 1); if (sourceId) change({ source_job_id: null, task_id: null }); }}>{text('重新选择', 'Choose again')}</button></div>}
    {sourceJob && scopeState !== 'ready' && <div className="task-notice" role="status">{scopeState === 'archived' ? text('来源项目或版本已归档；可以查看对比记录，恢复归档后才能生成。', 'The source project or version is archived. Existing comparisons remain viewable; restore it to generate.') : scopeState === 'busy' ? text('来源版本正在处理，暂时无法生成对比图；已有对比记录仍可查看。', 'The source version is being processed. Existing comparisons remain viewable, but generation is temporarily unavailable.') : scopeState === 'not-ready' ? text('来源版本尚未就绪，暂时无法生成对比图。', 'The source version is not ready for generation.') : scopeState === 'unavailable' ? <>{text('无法确认来源版本状态', 'Could not check source version status')}: {scopeError} <button type="button" className="task-link" onClick={() => setSourceRevision(value => value + 1)}>{text('重新检查来源', 'Recheck source')}</button></> : text('正在检查来源版本状态…', 'Checking source version status…')}</div>}
    {sourceJob ? <XyzSampling readOnly={scopeState !== 'ready'} key={sourceJob.id} sourceJobId={sourceJob.id} initialTaskId={params.get('task_id') || undefined}/> : !sourceError && <div className="task-empty sampling-page-empty"><Grid2X2 size={32}/><strong>{sourceId ? text('正在读取来源任务…', 'Reading source run…') : text('选择要比较的训练任务', 'Choose a training run to compare')}</strong><p>{text('使用该任务的底模和已保存权重。X 轴比较参数或权重，Y 轴增加另一组条件，Z 轴可选。', 'Use the run’s model and saved checkpoints. Compare settings or weights on X, add another condition on Y, and optionally use Z for pages.')}</p>{!loading && !jobs.length && <Link to="/projects">{text('打开项目并配置训练', 'Open a project and configure training')}</Link>}</div>}
  </section>;
}
