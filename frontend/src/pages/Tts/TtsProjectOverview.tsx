import React from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router-dom';
import { Activity, AlertTriangle, ArrowRight, Ban, CheckCircle2, Circle, CircleAlert, Clock, Download, ListChecks, Loader2, Pencil } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { Job, JobListResponse, Project } from '../../api/types';
import { ttsApi, ttsResourceUrl, type TtsScopedConfig, type TtsSource, type TtsSourceChanged } from '../../api/tts';
import { EVENT_TYPES } from '../../events/eventTypes';
import { useEventStream, useEventStreamStatus } from '../../events/useEventStream';
import { formatApiError } from '../../utils/errors';
import { formatBytes, formatEta } from '../../utils/format';
import { focusJob, mergeJobEvent, shortTime } from '../../utils/jobs';
import { projectUrl, type ProjectVersion } from '../../utils/projectVersions';
import { ttsEngineLabel } from '../../utils/ttsEngines';
import { useWorkspaceText } from '../../utils/workspaceText';
import type { ReadinessCheck } from '../ProjectDetail/OverviewBanner';
import { ProjectArtwork } from '../Projects/ProjectCardParts';
import ProjectEditor from '../Projects/ProjectEditor';
import { categoryLabel, type GalleryProject } from '../Projects/projectGallery';
import { JobActions, JobProgressSummary, JobStatus } from '../Queue/jobPresentation';
import { checkpointProgress } from './gptSovitsResults';
import TtsOverviewDataPanel from './TtsOverviewDataPanel';
import { sourceComplete, sourceName, sourceState } from './ttsOverviewData';
import TtsOverviewParameters from './TtsOverviewParameters';
import '../Queue/queue.css';
import '../ProjectDetail/project-overview.css';
import './tts-overview.css';

interface Props { project: Project; version: ProjectVersion; config: TtsScopedConfig; readOnly: boolean }
const liveStates = ['queued', 'scheduled', 'running', 'cancelling'];
type JobEvent = { job_id?: string; project_id?: string; version_id?: string; [key: string]: unknown };

function VersionStatus({ name, job, loading, error, checks, next, archived, trainUrl, resultsUrl, refresh }: {
  name: string; job?: Job; loading: boolean; error: unknown; checks: ReadinessCheck[]; next: { href: string; label: string };
  archived: boolean; trainUrl: string; resultsUrl: string; refresh: () => void;
}) {
  const text = useWorkspaceText();
  if (loading) return <section className="overview-banner overview-banner-loading" data-tone="neutral" role="status" aria-label={text('正在读取版本状态…', 'Loading version status…')}><span className="ui-skeleton"/><span className="ui-skeleton"/></section>;
  if (error) return <section className="overview-banner" data-tone="neutral"><div className="overview-inline-error" role="alert"><span>{text('训练状态读取失败', 'Could not load training status')} · {formatApiError(error)}</span><button type="button" className="ui-btn ui-btn-sm" onClick={refresh}>{text('重试', 'Retry')}</button></div></section>;
  if (!job && archived) return null;
  const status = job?.status;
  const pending = checks.filter(check => check.state !== 'done').length;
  const tone = status === 'failed' ? 'danger' : status === 'completed' ? 'success' : status === 'cancelled' ? 'neutral' : job ? 'active' : pending ? 'prepare' : 'neutral';
  const title = job ? ({ running: text('训练中', 'Training'), cancelling: text('正在取消', 'Cancelling'), queued: text('排队中', 'Queued'), scheduled: text('已排期', 'Scheduled'), failed: text('上次训练失败', 'Last run failed'), completed: text('训练完成', 'Training complete'), cancelled: text('训练已取消', 'Training cancelled') }[job.status] || text('训练状态', 'Training status')) : text('训练准备', 'Training preparation');
  const Icon = status === 'failed' ? AlertTriangle : status === 'completed' ? CheckCircle2 : status === 'cancelled' ? Ban : status === 'queued' || status === 'scheduled' ? Clock : job ? Activity : ListChecks;
  return <section className="overview-banner" data-tone={tone} data-testid="tts-overview-banner" aria-label={`${name} · ${title}`}>
    <div className="overview-banner-head"><span className="overview-banner-icon" aria-hidden="true"><Icon size={18}/></span><div className="overview-banner-title"><h2>{name} · {title}</h2><p>{job ? <><Link className="overview-banner-job" to={`/jobs/${encodeURIComponent(job.id)}`}>{job.name}</Link> · {shortTime(job.finished_at ?? job.started_at ?? job.created_at)}</> : text('检查训练数据、模型与参数后开始训练。', 'Check the training data, model and parameters before starting.')}</p></div>
      <div className="overview-banner-actions">{job ? <>{!archived && <JobActions job={job} onUpdated={refresh}/>}<Link className={`ui-btn ui-btn-sm${liveStates.includes(job.status) ? ' ui-btn-primary' : ''}`} to={liveStates.includes(job.status) ? `/jobs/${encodeURIComponent(job.id)}` : job.status === 'failed' ? `/jobs/${encodeURIComponent(job.id)}?tab=logs` : resultsUrl}>{liveStates.includes(job.status) ? text('打开训练监控', 'Open monitor') : job.status === 'failed' ? text('查看日志', 'View logs') : text('查看训练结果', 'View results')}<ArrowRight size={13}/></Link></> : <Link className="ui-btn ui-btn-sm ui-btn-primary" to={next.href}>{next.label}<ArrowRight size={13}/></Link>}</div>
    </div>
    {job ? <>{!(job.status === 'failed' && job.error) && <div className="tts-overview-job-progress"><JobProgressSummary job={job}/></div>}{job.error && <pre className="overview-banner-error">{job.error}</pre>}{!archived && ['failed', 'cancelled'].includes(job.status) && <p className="overview-banner-note"><Link className="ui-link" to={trainUrl}>{text('检查训练参数', 'Review parameters')}<ArrowRight size={13}/></Link></p>}</> : <ul className="overview-checklist">{checks.map(check => <li key={check.key}><Link to={check.href} data-state={check.state}>{check.state === 'done' ? <CheckCircle2 size={16}/> : check.state === 'warn' ? <CircleAlert size={16}/> : <Circle size={16}/>}<span><strong>{check.label}</strong><small title={check.detail}>{check.detail}</small></span></Link></li>)}</ul>}
  </section>;
}

export default function TtsProjectOverview(props: Props) {
  return <Overview key={`${props.project.id}:${props.version.id}`} {...props}/>;
}

function Overview({ project, version, config, readOnly }: Props) {
  const text = useWorkspaceText(), client = useQueryClient();
  const { i18n } = useTranslation();
  const [editing, setEditing] = React.useState(false);
  const pid = project.id, vid = version.id;
  const dataUrl = projectUrl(pid, vid, 'data'), trainUrl = projectUrl(pid, vid, 'train'), resultsUrl = projectUrl(pid, vid, 'results');
  const sourceKey = ['tts-sources', pid, vid], jobsKey = ['tts-overview-jobs', pid, vid], activeKey = ['tts-overview-active', pid, vid];
  const archived = !!project.archived || !!version.archived;
  const sources = useQuery({ queryKey: sourceKey, queryFn: ({ signal }) => ttsApi.sources(pid, vid, signal), refetchInterval: query => query.state.data?.items.some(item => item.state === 'checking') ? 2000 : false });
  const jobs = useQuery({ queryKey: jobsKey, queryFn: ({ signal }) => apiClient.get<JobListResponse>('/jobs', { params: { project_id: pid, version_id: vid, type: 'tts_train', page: 1, page_size: 4 }, signal, silent: true }), refetchInterval: query => query.state.data?.items.some(item => liveStates.includes(item.status)) ? 5000 : false });
  const active = useQuery({ queryKey: activeKey, queryFn: ({ signal }) => apiClient.get<JobListResponse>('/jobs', { params: { project_id: pid, version_id: vid, type: 'tts_train', group: 'active', page: 1, page_size: 20 }, signal, silent: true }), refetchInterval: query => query.state.data?.items.length ? 5000 : false });
  const checkpoints = useQuery({ queryKey: ['tts-overview-checkpoints', pid, vid], queryFn: ({ signal }) => ttsApi.versionCheckpoints(pid, vid, { limit: 3 }, signal) });
  const refreshJobs = () => { void jobs.refetch(); void active.refetch(); };
  const relevant = (event: JobEvent) => (!event.project_id || event.project_id === pid) && (!event.version_id || event.version_id === vid);
  const updateJob = (event: JobEvent) => {
    if (!relevant(event) || !event.job_id) return;
    for (const key of [jobsKey, activeKey]) client.setQueryData<JobListResponse>(key, data => data && { ...data, items: data.items.map(job => mergeJobEvent(job, event)) });
  };
  useEventStream<JobEvent>(EVENT_TYPES.JOB_STEP, updateJob);
  useEventStream<JobEvent>(EVENT_TYPES.JOB_PHASE, updateJob);
  useEventStream<JobEvent>(EVENT_TYPES.JOB_STATE, event => { if (relevant(event)) { updateJob(event); refreshJobs(); void checkpoints.refetch(); } });
  useEventStream<JobEvent>(EVENT_TYPES.QUEUE_CHANGED, event => { if (relevant(event)) refreshJobs(); });
  useEventStream<JobEvent>(EVENT_TYPES.JOB_CHECKPOINT, event => { if (relevant(event)) void checkpoints.refetch(); });
  useEventStream<TtsSourceChanged>(EVENT_TYPES.TTS_SOURCE_CHANGED, event => {
    if (event.scope.project_id === pid && event.scope.version_id === vid) {
      void client.invalidateQueries({ queryKey: sourceKey, exact: true });
      void client.invalidateQueries({ queryKey: ['project-versions', pid] });
      void client.invalidateQueries({ queryKey: ['project', pid] });
    }
  });
  const connection = useEventStreamStatus(), previousConnection = React.useRef(connection);
  React.useEffect(() => {
    if (connection === 'connected' && previousConnection.current !== 'connected') { void sources.refetch(); refreshJobs(); void checkpoints.refetch(); }
    previousConnection.current = connection;
  });
  const sourceSignature = sources.data ? JSON.stringify(sources.data) : null;
  const lastSourceSignature = React.useRef<string | null>(null);
  React.useEffect(() => {
    if (sourceSignature === null || sourceSignature === lastSourceSignature.current) return;
    if (lastSourceSignature.current !== null) { void client.invalidateQueries({ queryKey: ['project-versions', pid] }); void client.invalidateQueries({ queryKey: ['project', pid] }); }
    lastSourceSignature.current = sourceSignature;
  }, [sourceSignature, client, pid]);
  const sourceItems = (sources.data?.items || []).filter(source => source.scope.project_id === pid && source.scope.version_id === vid);
  const training = sourceItems.find(source => source.split === 'train');
  const trainingComplete = sourceComplete(training);
  const sourceDetail = (source: TtsSource | undefined) => sourceComplete(source) && source?.summary ? text(`${source.summary.clips_count.toLocaleString()} 条音频`, `${source.summary.clips_count.toLocaleString()} clips`) : sourceState(source, text);
  const model = sourceName(config.model_path.trim());
  const checks: ReadinessCheck[] = [
    { key: 'audio', label: text('训练音频', 'Training audio'), detail: sources.isPending ? text('正在读取…', 'Loading…') : sources.error ? text('读取失败', 'Could not load') : sourceDetail(training), state: !sources.error && trainingComplete && !!training?.summary?.valid_clips_count ? 'done' : 'todo', href: `${dataUrl}&tts_data=rows&split=train` },
    { key: 'check', label: text('数据检查', 'Data checks'), detail: sources.error ? text('读取失败', 'Could not load') : sourceState(training, text), state: !sources.error && training?.state === 'valid' ? 'done' : 'todo', href: `${dataUrl}&tts_data=issues&split=train` },
    { key: 'model', label: text('训练底模', 'Base model'), detail: model ? text(`已选择 · ${model}`, `Selected · ${model}`) : text('尚未选择', 'Not selected'), state: model ? 'done' : 'todo', href: trainUrl },
  ];
  const next = sources.error || !training || training.state !== 'valid' ? { href: dataUrl, label: text('整理训练数据', 'Prepare training data') } : !model ? { href: trainUrl, label: text('配置训练模型', 'Configure models') } : { href: trainUrl, label: text('检查参数并开始训练', 'Review and start training') };
  const scopedJob = (job: Job) => job.type === 'tts_train' && job.project_id === pid && job.version_id === vid;
  const combined = [...new Map([...(active.data?.items || []), ...(jobs.data?.items || [])].filter(scopedJob).map(job => [job.id, job])).values()].sort((a, b) => b.created_at - a.created_at);
  const focus = focusJob(combined), runTotal = jobs.data?.total;
  const recent = combined.slice(0, 4);
  const outputs = (checkpoints.data?.items || []).filter(item => item.project_id === pid && item.version_id === vid);
  const counts = sourceItems.map(source => `${source.split === 'train' ? text('训练集', 'Training set') : text('验证集', 'Validation set')} ${sourceDetail(source)}`).join(' · ');
  const date = (value: unknown) => typeof value === 'number' && Number.isFinite(value) ? new Date(value * 1000).toLocaleDateString(i18n.resolvedLanguage || 'zh-CN', { year: 'numeric', month: 'short', day: 'numeric' }) : '—';
  return <section className="project-overview tts-project-overview" data-testid="tts-project-overview">
    <VersionStatus name={version.name} job={focus} loading={jobs.isPending || active.isPending} error={jobs.error || active.error} checks={checks} next={next} archived={archived} trainUrl={trainUrl} resultsUrl={resultsUrl} refresh={refreshJobs}/>
    <div className="overview-layout">
      <section className="overview-main" aria-labelledby="tts-overview-data-title"><header className="overview-section-heading"><div><h2 id="tts-overview-data-title">{text('训练数据', 'Training data')}</h2>{counts && !sources.error && <p>{counts}{trainingComplete && training?.summary && ` · ${formatEta(training.summary.duration_seconds)}`}</p>}</div><Link className="ui-btn ui-btn-sm" to={dataUrl}>{readOnly ? text('查看训练数据', 'View training data') : text('管理训练数据', 'Manage data')}<ArrowRight size={13}/></Link></header>
        {sources.isPending ? <div className="overview-panel"><p className="overview-muted" role="status"><Loader2 size={14} className="animate-spin"/>{text('正在读取音频与检查结果…', 'Loading audio and check results…')}</p></div> : sources.error ? <div className="overview-panel overview-inline-error" role="alert"><span>{formatApiError(sources.error)}</span><button type="button" className="ui-btn ui-btn-sm" onClick={() => void sources.refetch()}>{text('重新读取数据', 'Reload data')}</button></div> : <TtsOverviewDataPanel sources={sourceItems} engine={config.engine} dataUrl={dataUrl} readOnly={readOnly} refresh={() => void sources.refetch()}/>}
      </section>
      <aside className="overview-aside" aria-label={text('版本信息', 'Version details')}>
        <section className="overview-panel overview-configuration" aria-labelledby="tts-overview-config-title"><header className="overview-panel-heading"><h3 id="tts-overview-config-title">{text('关联的训练参数与模型类型', 'Linked parameters and model type')}</h3><Link className="ui-link" to={trainUrl}>{text('训练参数', 'Parameters')}<ArrowRight size={13}/></Link></header><Link to={trainUrl} className="overview-model" data-missing={!model || undefined}><span>{ttsEngineLabel(config.engine)}{config.engine === 'gpt-sovits-v5' ? ` · ${config.variant}` : ' · LoRA'}</span><strong title={config.model_path || undefined}>{model || text('尚未配置训练底模', 'Base model not configured')}</strong></Link><TtsOverviewParameters config={config}/></section>
        <section className="overview-panel overview-runs" aria-labelledby="tts-overview-runs-title"><header className="overview-panel-heading"><div className="overview-panel-title"><h3 id="tts-overview-runs-title">{text('训练记录', 'Training runs')}</h3>{runTotal !== undefined && runTotal > 0 && <span className="overview-count">{runTotal}</span>}</div>{!!runTotal && <Link className="ui-link" to={`${resultsUrl}&result_tab=jobs`}>{text('全部记录', 'All runs')}<ArrowRight size={13}/></Link>}</header>
          {jobs.isPending || active.isPending ? <p className="overview-muted" role="status"><Loader2 size={14} className="animate-spin"/>{text('正在读取训练记录…', 'Loading training runs…')}</p> : jobs.error || active.error ? <div className="overview-inline-error" role="alert"><span>{formatApiError(jobs.error || active.error)}</span><button type="button" className="ui-btn ui-btn-sm" onClick={refreshJobs}>{text('重试', 'Retry')}</button></div> : recent.length ? <ul className="overview-run-list">{recent.map(job => <li key={job.id}><Link to={`/jobs/${encodeURIComponent(job.id)}`}><span className="overview-run-top"><JobStatus status={job.status}/><time>{shortTime(job.created_at)}</time></span><strong title={job.name}>{job.name}</strong><JobProgressSummary job={job}/></Link></li>)}</ul> : <p className="overview-muted">{text('还没有训练记录。开始训练后，这里会显示每次训练的进度和保存的模型权重。', 'No runs yet. Training progress and saved weights will appear here.')}</p>}
        </section>
        {(!!runTotal || !!outputs.length || checkpoints.isPending || !!checkpoints.error) && <section className="overview-panel overview-outputs" aria-labelledby="tts-overview-outputs-title"><header className="overview-panel-heading"><h3 id="tts-overview-outputs-title">{text('最新训练产物', 'Latest outputs')}</h3><Link className="ui-link" to={`${resultsUrl}&result_tab=artifacts`}>{text('全部产物', 'All outputs')}<ArrowRight size={13}/></Link></header>
          {checkpoints.isPending ? <p className="overview-muted" role="status"><Loader2 size={14} className="animate-spin"/>{text('正在读取训练产物…', 'Loading outputs…')}</p> : checkpoints.error ? <div className="overview-inline-error" role="alert"><span>{formatApiError(checkpoints.error)}</span><button type="button" className="ui-btn ui-btn-sm" onClick={() => void checkpoints.refetch()}>{text('重新读取产物', 'Reload outputs')}</button></div> : outputs.length ? <ul className="overview-output-list">{outputs.map(item => <li key={item.id}><span><strong title={item.name}>{item.name}</strong><small title={checkpointProgress(item, text)}>{formatBytes(item.size)} · {item.created_at === null ? text('保存时间未知', 'Save time unknown') : shortTime(item.created_at)}</small><span className="tts-overview-output-files">{item.files.map(file => { const url = ttsResourceUrl(file.download_url); return url ? <a key={file.id} className="ui-link" href={url} download={file.name} aria-label={text(`下载 ${file.name}`, `Download ${file.name}`)}><Download size={12}/>{file.name}</a> : <span key={file.id}>{file.name}</span>; })}</span></span></li>)}</ul> : <p className="overview-muted">{text('训练保存的模型权重会显示在这里。', 'Saved model weights appear here.')}</p>}
        </section>}
        <section className="overview-panel overview-about" aria-labelledby="tts-overview-about-title"><header className="overview-panel-heading"><h3 id="tts-overview-about-title">{text('关于项目', 'About')}</h3>{!project.archived && <button type="button" className="ui-btn ui-btn-sm ui-btn-quiet" onClick={() => setEditing(true)}><Pencil size={13}/>{text('编辑', 'Edit')}</button>}</header><div className="overview-about-body"><div className="overview-about-art"><ProjectArtwork name={project.name} coverUrl={(project as GalleryProject).cover_url}/></div><p data-empty={!project.note?.trim() || undefined}>{project.note?.trim() || text('没有项目备注', 'No project note')}</p></div><dl className="overview-facts"><div><dt>{text('分类', 'Category')}</dt><dd>{project.category ? categoryLabel(project.category, i18n.resolvedLanguage?.startsWith('en') || false) : text('未分类', 'Uncategorized')}</dd></div><div><dt>{text('版本', 'Versions')}</dt><dd>{text(`${project.version_count ?? 1} 个`, `${project.version_count ?? 1}`)}</dd></div>{version.note?.trim() && <div className="overview-fact-wide"><dt>{text('版本说明', 'Version note')}</dt><dd>{version.note.trim()}</dd></div>}<div><dt>{text('创建于', 'Created')}</dt><dd>{date(project.created_at)}</dd></div><div><dt>{text('最后更新', 'Updated')}</dt><dd>{date(version.updated_at || project.updated_at)}</dd></div></dl></section>
      </aside>
    </div>
    {editing && <ProjectEditor project={project as GalleryProject} categories={[]} onClose={() => setEditing(false)} onPartial={updated => client.setQueryData(['project', pid], updated)} onSaved={updated => { client.setQueryData(['project', pid], updated); setEditing(false); }}/> }
  </section>;
}
