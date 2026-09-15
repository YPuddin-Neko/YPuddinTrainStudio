import React from 'react';
import { Link, Navigate, useSearchParams } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Activity, Box, Grid2X2, ChevronLeft, ChevronRight, Download, ExternalLink, Image as ImageIcon, Loader2, RefreshCw, X } from 'lucide-react';
import { apiClient, apiUrl } from '../../api/client';
import type { Job, JobSample, JobListResponse } from '../../api/types';
import { ApiError } from '../../api/types';
import SampleLoss from '../SampleLoss';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { mergeJobEvent } from '../../utils/jobs';
import { formatTime } from '../../utils/format';
import { formatApiError } from '../../utils/errors';
import { projectUrl } from '../../utils/projectVersions';
import { useWorkspaceText } from '../../utils/workspaceText';
import StudioSelect from '../StudioSelect';
import Artifacts from '../../pages/Artifacts/Artifacts';
import { samplingUrl } from '../../utils/samplingRoutes';
import '../../styles/project-results.css';

interface VersionResultsProps { projectId: string; versionId?: string; readOnly?: boolean }
type VersionedJob = Job & { version_id?: string | null };
type ResultTab = 'jobs' | 'samples' | 'artifacts' | 'xyz';
const PAGE_SIZE = 50;
const fileUrl = (url: string) => url.startsWith('/api/') ? apiUrl(url.slice(4)) : url;

export default function VersionResults(props: VersionResultsProps) {
  // A version change immediately discards every selection and pending response.
  return <VersionResultsWorkspace key={`${props.projectId}:${props.versionId || ''}`} {...props}/>;
}

function VersionResultsWorkspace({ projectId, versionId, readOnly = false }: VersionResultsProps) {
  const { t } = useTranslation();
  const text = useWorkspaceText();
  const [params, setParams] = useSearchParams();
  const tab: ResultTab = ['jobs', 'samples', 'artifacts', 'xyz'].includes(params.get('result_tab') || '') ? params.get('result_tab') as ResultTab : 'artifacts';
  const setTab = (value: ResultTab) => { const next = new URLSearchParams(params); next.set('result_tab', value); setParams(next); };
  // Keep only the chosen job, so paging the task list does not erase its label or filter.
  const [artifactJob, setArtifactJob] = React.useState<Pick<VersionedJob, 'id' | 'name'> | null>(null);
  const artifactJobId = artifactJob?.id || '';
  const [samplePage, setSamplePage] = React.useState(1);

  const [page, setPage] = React.useState(1);
  const [jobs, setJobs] = React.useState<VersionedJob[]>([]);
  const [total, setTotal] = React.useState(0);
  const [jobsLoading, setJobsLoading] = React.useState(true);
  const [jobsError, setJobsError] = React.useState('');
  const [selectedJobId, setSelectedJobId] = React.useState('');
  const [samples, setSamples] = React.useState<JobSample[]>([]);
  const [samplesLoading, setSamplesLoading] = React.useState(false);
  const [samplesError, setSamplesError] = React.useState('');
  const [lightbox, setLightbox] = React.useState<JobSample | null>(null);
  const [failedImages, setFailedImages] = React.useState<Set<string>>(new Set());
  const jobsRequest = React.useRef<AbortController | null>(null);
  const samplesRequest = React.useRef<AbortController | null>(null);
  const eventsDuringRequest = React.useRef<Record<string, any>[] | null>(null);
  const refreshTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const closeLightbox = React.useRef<HTMLButtonElement>(null);
  const lightboxOpener = React.useRef<HTMLButtonElement | null>(null);
  const selectedJob = jobs.find(job => job.id === selectedJobId);
  const sampleJobs = jobs.filter(job => job.type !== 'cache');
  const artifactJobs = artifactJob && !sampleJobs.some(job => job.id === artifactJob.id) ? [artifactJob, ...sampleJobs] : sampleJobs;
  const chooseArtifactJob = (id: string) => setArtifactJob(artifactJobs.find(job => job.id === id) || null);

  const fetchJobs = React.useCallback(async () => {
    jobsRequest.current?.abort(); const controller = new AbortController(); jobsRequest.current = controller;
    const pendingEvents: Record<string, any>[] = []; eventsDuringRequest.current = pendingEvents;
    setJobsLoading(true); setJobsError('');
    try {
      const response = await apiClient.get<JobListResponse | VersionedJob[]>('/jobs', { params: { project_id: projectId, ...(versionId ? { version_id: versionId } : {}), page, page_size: PAGE_SIZE, type: 'train' }, signal: controller.signal, silent: true });
      if (controller.signal.aborted) return;
      const rows = (Array.isArray(response) ? response : response.items || []) as VersionedJob[];
      const ownRows = rows.filter(job => job.project_id === projectId && (!versionId || job.version_id === versionId)).map(job => pendingEvents.reduce<VersionedJob>((current, event) => mergeJobEvent(current, event), job));
      const count = Array.isArray(response) ? ownRows.length : response.total;
      setJobs(ownRows); setTotal(count);
      setSelectedJobId(previous => ownRows.some(job => job.id === previous && job.type !== 'cache') ? previous : ownRows.find(job => job.type !== 'cache')?.id || '');
      if (page > Math.max(1, Math.ceil(count / PAGE_SIZE))) setPage(Math.max(1, Math.ceil(count / PAGE_SIZE)));
    } catch (error) { if (!controller.signal.aborted) setJobsError(formatApiError(error)); }
    finally { if (!controller.signal.aborted) setJobsLoading(false); if (eventsDuringRequest.current === pendingEvents) eventsDuringRequest.current = null; }
  }, [projectId, versionId, page]);

  React.useEffect(() => {
    void fetchJobs();
    return () => { jobsRequest.current?.abort(); if (refreshTimer.current) clearTimeout(refreshTimer.current); };
  }, [fetchJobs]);
  React.useEffect(() => {
    if (!artifactJobId || jobsLoading) return;
    const current = jobs.find(job => job.id === artifactJobId);
    if (current) {
      setArtifactJob(previous => previous?.id === current.id && previous.name !== current.name ? { id: current.id, name: current.name } : previous);
      return;
    }
    // Absence from a page is not deletion. Check the single selected record before clearing it.
    const controller = new AbortController();
    void apiClient.get<VersionedJob>(`/jobs/${encodeURIComponent(artifactJobId)}`, { signal: controller.signal, silent: true }).then(job => {
      if (controller.signal.aborted) return;
      const own = job.project_id === projectId && (!versionId || job.version_id === versionId);
      setArtifactJob(previous => previous?.id !== artifactJobId ? previous : !own ? null : previous.name === job.name ? previous : { id: job.id, name: job.name });
    }).catch(error => {
      if (!controller.signal.aborted && error instanceof ApiError && error.status === 404) setArtifactJob(previous => previous?.id === artifactJobId ? null : previous);
    });
    return () => controller.abort();
  }, [artifactJobId, jobs, jobsLoading, projectId, versionId]);
  const scheduleJobsRefresh = () => {
    if (refreshTimer.current) clearTimeout(refreshTimer.current);
    refreshTimer.current = setTimeout(() => { refreshTimer.current = null; void fetchJobs(); }, 250);
  };
  const updateJob = (event: Record<string, any>) => {
    if (typeof event.job_id !== 'string') return;
    eventsDuringRequest.current?.push(event);
    setJobs(previous => previous.map(job => mergeJobEvent(job, event)));
  };
  useEventStream(EVENT_TYPES.JOB_STEP, updateJob);
  useEventStream(EVENT_TYPES.JOB_PHASE, updateJob);
  useEventStream(EVENT_TYPES.JOB_STATE, event => { updateJob(event); scheduleJobsRefresh(); });
  useEventStream(EVENT_TYPES.QUEUE_CHANGED, scheduleJobsRefresh);

  const fetchSamples = React.useCallback(async () => {
    if (tab !== 'samples' || !selectedJobId) return;
    samplesRequest.current?.abort(); const controller = new AbortController(); samplesRequest.current = controller;
    setSamplesLoading(true); setSamplesError('');
    try {
      const rows = await apiClient.get<JobSample[]>(`/jobs/${encodeURIComponent(selectedJobId)}/samples`, { signal: controller.signal, silent: true });
      if (!controller.signal.aborted) setSamples([...rows].sort((a, b) => b.step - a.step || a.prompt_index - b.prompt_index || b.created_at - a.created_at));
    } catch (error) { if (!controller.signal.aborted) setSamplesError(formatApiError(error)); }
    finally { if (!controller.signal.aborted) setSamplesLoading(false); }
  }, [tab, selectedJobId]);
  React.useEffect(() => {
    setSamples([]); setSamplePage(1); setSamplesError(''); setFailedImages(new Set()); setLightbox(null);
    void fetchSamples(); return () => samplesRequest.current?.abort();
  }, [fetchSamples]);
  useEventStream(EVENT_TYPES.JOB_SAMPLE, (event: { job_id?: string }) => {
    if (event.job_id === selectedJobId) void fetchSamples();
  });
  React.useEffect(() => {
    if (!lightbox) return;
    const previous = lightboxOpener.current || document.activeElement as HTMLElement | null; closeLightbox.current?.focus();
    return () => previous?.focus();
  }, [lightbox]);

  const tabs: { id: ResultTab; label: string; icon: typeof Activity }[] = [
    { id: 'artifacts', label: text('模型权重', 'Model weights'), icon: Box },
    { id: 'samples', label: text('采样图', 'Samples'), icon: ImageIcon },
    { id: 'xyz', label: text('模型测试', 'Model testing'), icon: Grid2X2 },
    { id: 'jobs', label: text('训练记录', 'Training records'), icon: Activity },
  ];
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const status = (job: Job) => t(`queue.status.${job.status}`, { defaultValue: job.status });
  const pagination = pages > 1 ? <nav className="results-pagination" aria-label={text('训练任务分页', 'Training run pages')}><span>{text(`共 ${total} 个任务 · 每页 ${PAGE_SIZE} 个`, `${total} jobs · ${PAGE_SIZE} per page`)}</span><div><button type="button" disabled={jobsLoading || page <= 1} onClick={() => setPage(value => value - 1)} aria-label={text('上一页任务', 'Previous jobs page')}><ChevronLeft size={14}/></button><span>{page} / {pages}</span><button type="button" disabled={jobsLoading || page >= pages} onClick={() => setPage(value => value + 1)} aria-label={text('下一页任务', 'Next jobs page')}><ChevronRight size={14}/></button></div></nav> : null;

  return <section className="version-results" data-testid="version-results">
    <div className="results-toolbar"><div className="results-tabs" role="tablist" aria-label={text('版本训练结果', 'Version training results')}>{tabs.map((item, index) => <button type="button" key={item.id} role="tab" id={`results-tab-${item.id}`} aria-controls={`results-panel-${item.id}`} aria-selected={tab === item.id} tabIndex={tab === item.id ? 0 : -1} onClick={() => setTab(item.id)} onKeyDown={event => {
      const next = event.key === 'ArrowRight' ? (index + 1) % tabs.length : event.key === 'ArrowLeft' ? (index + tabs.length - 1) % tabs.length : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : -1;
      if (next >= 0) { event.preventDefault(); setTab(tabs[next].id); document.getElementById(`results-tab-${tabs[next].id}`)?.focus(); }
    }}><item.icon size={14}/>{item.label}</button>)}</div><div className="results-overview"><span>{text(`共 ${total} 次训练`, `${total} training runs`)}</span><Link to={`/queue?project_id=${encodeURIComponent(projectId)}`} title={text('在全局队列管理调度', 'Manage scheduling in the queue')}>{text('全局队列', 'Queue')}<ExternalLink size={12}/></Link></div>{tab !== 'artifacts' && tab !== 'xyz' && <button type="button" className="results-refresh" disabled={tab === 'jobs' ? jobsLoading : samplesLoading} onClick={() => void (tab === 'samples' ? fetchSamples() : fetchJobs())}><RefreshCw size={13} className={(tab === 'jobs' ? jobsLoading : samplesLoading) ? 'animate-spin' : ''}/>{t('common.refresh')}</button>}</div>
    {jobsError && <div className="results-error" role="alert">{jobsError}<button type="button" onClick={() => void fetchJobs()}>{t('common.retry')}</button></div>}

    {tab === 'jobs' && <div role="tabpanel" id="results-panel-jobs" aria-labelledby="results-tab-jobs">{pagination}
      {jobsLoading && jobs.length === 0 ? <p className="results-empty" role="status"><Loader2 size={16} className="animate-spin"/>{t('common.loading')}</p> : jobs.length === 0 && !jobsError ? <div className="results-empty"><Activity size={22}/><p>{text('此版本还没有训练记录', 'This version has no training records yet')}</p>{!readOnly && <Link to={projectUrl(projectId, versionId, 'train')}>{text('配置并启动训练', 'Configure and start training')}</Link>}</div> : <div className="results-table-wrap"><table className="results-job-table" aria-label={text('当前版本训练记录', 'Training records in this version')}><thead><tr><th>{text('任务', 'Job')}</th><th>{t('common.status')}</th><th>{t('queue.progress')}</th><th>{text('创建时间', 'Created')}</th><th>{text('结果', 'Results')}</th></tr></thead><tbody>{jobs.map(job => <tr key={job.id} data-testid={`result-job-${job.id}`}>
        <td><Link to={`/jobs/${encodeURIComponent(job.id)}`}>{job.name}</Link><small>{job.type === 'cache' ? text('缓存任务', 'Cache job') : text('训练任务', 'Training job')} · {job.id}</small></td>
        <td><span className="results-job-status" data-status={job.status}>{status(job)}</span>{job.error && <span className="results-job-error" title={job.error}>{job.error}</span>}</td>
        <td className="results-progress">{job.progress?.step ?? '—'} / {job.progress?.total_steps ?? '—'}{job.latest?.loss != null && <small>Loss {Number(job.latest.loss).toFixed(4)}</small>}</td>
        <td><time>{formatTime(job.created_at)}</time></td>
        <td><div className="results-row-actions"><Link to={`/jobs/${encodeURIComponent(job.id)}`}>{text('监控与日志', 'Monitor & logs')}</Link><button type="button" onClick={() => { setArtifactJob({ id: job.id, name: job.name }); setTab('artifacts'); }}>{text('查看权重', 'View weights')}</button>{job.type !== 'cache' && <button type="button" onClick={() => { setSelectedJobId(job.id); setTab('samples'); }}>{text('查看采样图', 'View samples')}</button>}</div></td>
      </tr>)}</tbody></table></div>}
    </div>}

    {tab === 'samples' && <div role="tabpanel" id="results-panel-samples" aria-labelledby="results-tab-samples">
      <div className="results-selection-controls"><div className="results-sample-toolbar"><label><span>{text('训练任务', 'Training job')}</span><StudioSelect aria-label={text('采样所属任务', 'Sample source job')} value={selectedJobId} disabled={jobsLoading || sampleJobs.length === 0} onValueChange={setSelectedJobId} placeholder={text('本页没有训练任务', 'No training jobs on this page')} options={sampleJobs.map(job => ({ value: job.id, label: `${job.name} · ${status(job)} · ${formatTime(job.created_at)}` }))}/></label>{selectedJob && <Link to={`/jobs/${encodeURIComponent(selectedJob.id)}`}>{text('打开任务详情', 'Open job details')}<ExternalLink size={12}/></Link>}{pagination}</div>
      {samples.length > 24 && <div className="results-pagination"><span>{text(`此任务共 ${samples.length} 张采样图`, `${samples.length} samples in this job`)}</span><div><button aria-label={text('上一页采样图', 'Previous samples page')} disabled={samplePage <= 1} onClick={() => setSamplePage(value => value - 1)}><ChevronLeft size={14}/></button><span>{samplePage} / {Math.ceil(samples.length / 24)}</span><button aria-label={text('下一页采样图', 'Next samples page')} disabled={samplePage >= Math.ceil(samples.length / 24)} onClick={() => setSamplePage(value => value + 1)}><ChevronRight size={14}/></button></div></div>}
      </div>
      {selectedJob && <p className="results-sample-context">{text('任务', 'Job')} <strong>{selectedJob.name}</strong> · {text('创建于', 'Created')} {formatTime(selectedJob.created_at)}</p>}
      {samplesError && <div className="results-error" role="alert">{samplesError}<button type="button" onClick={() => void fetchSamples()}>{t('common.retry')}</button></div>}
      {samplesLoading && samples.length === 0 ? <p className="results-empty" role="status"><Loader2 size={16} className="animate-spin"/>{text('读取此任务的采样图…', 'Loading samples for this job…')}</p> : !samplesError && samples.length === 0 ? <div className="results-empty"><ImageIcon size={22}/><p>{text('此任务暂无采样图', 'This job has no samples yet')}</p><span>{text('训练开启采样后，保存的图片会自动显示在这里。', 'Saved images appear here when sampling is enabled for training.')}</span></div> : <div className="results-sample-grid">{samples.slice((samplePage - 1) * 24, samplePage * 24).map(sample => <article className="results-sample-card" key={`${sample.url}-${sample.step}-${sample.prompt_index}-${sample.seed}`}>
        <button className="results-sample-preview" type="button" onClick={event => { lightboxOpener.current = event.currentTarget; setLightbox(sample); }} aria-label={text(`查看采样图：第 ${sample.step} 步，提示词 ${sample.prompt_index + 1}`, `View sample: step ${sample.step}, prompt ${sample.prompt_index + 1}`)}>{failedImages.has(sample.url) ? <span><ImageIcon size={22}/>{text('图片文件不可用', 'Image file unavailable')}</span> : <img src={fileUrl(sample.url)} alt={sample.prompt} loading="lazy" width={sample.width} height={sample.height} onError={() => setFailedImages(previous => new Set(previous).add(sample.url))}/>}</button>
        <div className="results-sample-description"><div><strong>{text('步数', 'Step')} {sample.step}</strong><span>Seed {sample.seed}</span></div><SampleLoss sample={sample}/><p title={sample.prompt}>{sample.prompt}</p><div><time>{formatTime(sample.created_at)}</time><a href={fileUrl(sample.url)} download aria-label={text(`下载第 ${sample.step} 步采样图`, `Download step ${sample.step} sample`)}><Download size={13}/></a></div></div>
      </article>)}</div>}
    </div>}

    {tab === 'xyz' && <Navigate replace to={samplingUrl(selectedJobId || null, null, projectId, versionId)}/>}

    {tab === 'artifacts' && <div role="tabpanel" id="results-panel-artifacts" aria-labelledby="results-tab-artifacts"><div className="results-sample-toolbar results-output-selector"><label><span>{text('训练任务', 'Training job')}</span><StudioSelect aria-label={text('权重所属任务', 'Weight source job')} value={artifactJobId} onValueChange={chooseArtifactJob} options={[{ value: '', label: text('此版本全部训练', 'All training runs in this version') }, ...artifactJobs.map(job => ({ value: job.id, label: job.name }))]}/></label>{artifactJobId && <Link to={`/jobs/${artifactJobId}?tab=checkpoints`}>{text('查看此任务检查点', 'View job checkpoints')}</Link>}{pagination}</div><Artifacts embedded projectId={projectId} versionId={versionId} jobId={artifactJobId || undefined} readOnly={readOnly}/></div>}

    {lightbox && <div className="results-lightbox" onClick={() => setLightbox(null)}><div role="dialog" aria-modal="true" aria-label={text('采样图预览', 'Sample preview')} className="results-lightbox-content" onClick={event => event.stopPropagation()} onKeyDown={event => { if (event.key === 'Escape') setLightbox(null); }}><header><strong>{selectedJob?.name} · {text('步数', 'Step')} {lightbox.step}</strong><button ref={closeLightbox} type="button" onClick={() => setLightbox(null)} aria-label={t('common.close')}><X size={18}/></button></header><img src={fileUrl(lightbox.url)} alt={lightbox.prompt}/><footer><p>{lightbox.prompt}</p><SampleLoss sample={lightbox}/><span>{lightbox.width} × {lightbox.height} · Seed {lightbox.seed} · {formatTime(lightbox.created_at)}</span><a href={fileUrl(lightbox.url)} download><Download size={14}/>{t('common.download')}</a></footer></div></div>}
  </section>;
}
