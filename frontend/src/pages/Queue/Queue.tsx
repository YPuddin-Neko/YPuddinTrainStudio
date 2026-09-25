import { gpuMemorySummary, gpuMemoryDetails } from '../../utils/gpuMemory';
import React from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Activity, Clock3, History, SlidersHorizontal, PauseCircle, Play, Search, RefreshCw, Trash2, Inbox } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { Job, JobListResponse, Project, QueueSettings } from '../../api/types';
import StudioSelect from '../../components/StudioSelect';
import { useQueueDevices } from '../../api/hooks/useQueueDevices';
import { gpuDeviceLabel } from '../../utils/gpuDevices';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { formatTime } from '../../utils/format';
import { formatApiError } from '../../utils/errors';
import { mergeJobEvent } from '../../utils/jobs';
import { useWorkspaceText } from '../../utils/workspaceText';
import { JobActions, JobContext, JobStatus, type ContextJob } from './jobPresentation';
import './queue.css';
import Switch from '../../components/Switch';
import { SlidingIndicator } from '../../components/motion';

type Group = 'active' | 'waiting' | 'history';
const groups: Group[] = ['active', 'waiting', 'history'];
const statuses = { active: ['running', 'pausing', 'cancelling', 'paused'], waiting: ['queued', 'scheduled'], history: ['completed', 'failed', 'cancelled'] };

export default function Queue() {
  const text = useWorkspaceText();
  const { t } = useTranslation();
  const gpuStatus = useQueueDevices();
  const progressLabel = (job: Job) => {
    const progress = job.progress;
    if (progress?.wait_reason && progress.wait_reason !== 'waiting for a free accelerator with enough memory') return progress.wait_reason;
    return t(`phase.${progress?.phase}`, t('job.phaseInProgress'));
  };
  const [params, setParams] = useSearchParams();
  const group = groups.includes(params.get('view') as Group) ? params.get('view') as Group : 'active';
  const page = Math.max(1, Math.floor(Number(params.get('page')) || 1));
  const pageSize = [20, 50, 100].includes(Number(params.get('size'))) ? Number(params.get('size')) : 20;
  const status = params.get('status') || '', project = params.get('project_id') || '', type = params.get('type') || '', query = params.get('q') || '';
  const scope = JSON.stringify([group, page, pageSize, status, project, type, query]);
  const [loadedScope, setLoadedScope] = React.useState('');
  const [jobs, setJobs] = React.useState<ContextJob[]>([]);
  const [total, setTotal] = React.useState(0);
  const [counts, setCounts] = React.useState<Record<Group, number | null>>({ active: null, waiting: null, history: null });
  const [projects, setProjects] = React.useState<Project[]>([]);
  const [settings, setSettings] = React.useState<QueueSettings>({ held: false, max_concurrent: null, memory_admission: true });
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState('');
  const [settingBusy, setSettingBusy] = React.useState(false);
  const request = React.useRef<AbortController | null>(null);
  const countRequest = React.useRef<AbortController | null>(null);
  const refreshTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const change = (patch: Record<string, string | null>, replace = false) => {
    const next = new URLSearchParams(params); next.delete('page');
    Object.entries(patch).forEach(([key, value]) => value ? next.set(key, value) : next.delete(key)); setParams(next, { replace });
  };
  const fetchJobs = React.useCallback(async () => {
    request.current?.abort(); const controller = new AbortController(); request.current = controller; setLoading(true); setError('');
    try {
      const data = await apiClient.get<JobListResponse | ContextJob[]>('/jobs', { params: { group, page, page_size: pageSize, status: status || undefined, project_id: project || undefined, type: type || undefined, q: query || undefined }, signal: controller.signal, silent: true });
      if (controller.signal.aborted) return;
      setJobs(Array.isArray(data) ? data : data.items); setLoadedScope(JSON.stringify([group, page, pageSize, status, project, type, query])); setTotal(Array.isArray(data) ? data.length : data.total);
    } catch (failure) { if (!controller.signal.aborted) setError(formatApiError(failure)); }
    finally { if (!controller.signal.aborted) setLoading(false); }
  }, [group, page, pageSize, status, project, type, query]);
  const fetchCounts = React.useCallback(async () => {
    countRequest.current?.abort(); const controller = new AbortController(); countRequest.current = controller;
    const results = await Promise.allSettled(groups.map(value => apiClient.get<JobListResponse>('/jobs', { params: { group: value, page_size: 1, project_id: project || undefined, type: type || undefined, q: query || undefined }, signal: controller.signal, silent: true })));
    if (!controller.signal.aborted) setCounts(Object.fromEntries(results.map((result, index) => [groups[index], result.status === 'fulfilled' && !Array.isArray(result.value) ? result.value.total : null])) as Record<Group, number | null>);
  }, [project, type, query]);
  React.useEffect(() => { void fetchJobs(); return () => { request.current?.abort(); if (refreshTimer.current) clearTimeout(refreshTimer.current); }; }, [fetchJobs]);
  React.useEffect(() => { void fetchCounts(); return () => countRequest.current?.abort(); }, [fetchCounts]);
  React.useEffect(() => {
    const controller = new AbortController();
    void apiClient.get<Project[] | { items: Project[] }>('/projects', { signal: controller.signal, silent: true }).then(data => setProjects(Array.isArray(data) ? data : data.items || [])).catch(() => {});
    void apiClient.get<QueueSettings>('/queue/settings', { signal: controller.signal, silent: true }).then(setSettings).catch(failure => { if (!controller.signal.aborted) setError(formatApiError(failure)); });
    return () => { controller.abort(); if (refreshTimer.current) clearTimeout(refreshTimer.current); };
  }, []);
  const refresh = () => { void fetchJobs(); void fetchCounts(); gpuStatus.refresh(); };
  const queueRefresh = () => { if (refreshTimer.current) clearTimeout(refreshTimer.current); refreshTimer.current = setTimeout(refresh, 180); };
  useEventStream(EVENT_TYPES.JOB_STATE, queueRefresh);
  useEventStream(EVENT_TYPES.QUEUE_CHANGED, queueRefresh);
  useEventStream(EVENT_TYPES.JOB_STEP, event => setJobs(rows => rows.map(job => mergeJobEvent(job, event))));
  useEventStream(EVENT_TYPES.JOB_PHASE, event => setJobs(rows => rows.map(job => mergeJobEvent(job, event))));
  useEventStream(EVENT_TYPES.JOB_XYZ_PROGRESS, event => setJobs(rows => rows.map(job => mergeJobEvent(job, event))));
  const updateSettings = async (patch: Partial<QueueSettings>) => {
    setSettingBusy(true); setError('');
    try { setSettings(await apiClient.put<QueueSettings>('/queue/settings', patch, { silent: true })); }
    catch (failure) { setError(formatApiError(failure)); }
    finally { setSettingBusy(false); }
  };
  const updatePriority = async (job: Job, raw: string) => {
    const priority = Number(raw); if (!raw.trim() || !Number.isSafeInteger(priority)) { setError(text('优先级必须是整数。', 'Priority must be an integer.')); return; }
    if (priority === job.priority) return;
    try { await apiClient.patch(`/jobs/${job.id}`, { priority }, { silent: true }); refresh(); }
    catch (failure) { setError(formatApiError(failure)); }
  };
  const remove = async (job: Job) => {
    if (!window.confirm(text(`删除任务记录“${job.name}”？磁盘文件会保留。`, `Delete the record for “${job.name}”? Files stay on disk.`))) return;
    try { await apiClient.delete(`/jobs/${job.id}`, { silent: true }); refresh(); } catch (failure) { setError(formatApiError(failure)); }
  };
  const visibleJobs = scope === loadedScope ? jobs : [];
  const pages = Math.max(1, Math.ceil(total / pageSize));
  const groupLabels = { active: text('运行与暂停', 'Active & paused'), waiting: text('等待调度', 'Waiting'), history: text('训练历史', 'History') };
  const statusLabels: Record<string, string> = { running: text('运行中', 'Running'), pausing: text('暂停中', 'Pausing'), cancelling: text('取消中', 'Cancelling'), paused: text('已暂停', 'Paused'), queued: text('排队中', 'Queued'), scheduled: text('已排期', 'Scheduled'), completed: text('已完成', 'Completed'), failed: text('失败', 'Failed'), cancelled: text('已取消', 'Cancelled') };
  return <section className="queue-workspace task-workspace">
    <header className="task-page-heading"><div><h1 data-testid="queue-title">{text('全局训练队列', 'Training queue')}</h1></div><div className="task-actions"><button type="button" className="ui-btn ui-btn-icon" disabled={loading} onClick={refresh} aria-label={text('刷新队列', 'Refresh queue')} title={text('刷新队列', 'Refresh queue')}><RefreshCw size={14}/></button><button type="button" className="ui-btn" data-testid="toggle-held" disabled={settingBusy} onClick={() => void updateSettings({ held: !settings.held })}>{settings.held ? <Play size={14}/> : <PauseCircle size={14}/>}{settings.held ? text('继续调度', 'Resume scheduling') : text('暂停调度', 'Hold scheduling')}</button></div></header>
    {settings.held && <p className="task-notice" role="status">{text('调度已暂停：正在运行的任务会继续；等待中的任务不会启动。', 'Scheduling is held. Running jobs continue; waiting jobs will not start.')}</p>}
    {gpuStatus.error && <p className="task-notice" role="status">{text('显卡占用暂时无法读取', 'GPU assignments are temporarily unavailable')}: {gpuStatus.error}</p>}
    {!!gpuStatus.snapshot?.devices.length && <div className="queue-device-overview" aria-label={text('显卡占用', 'GPU assignments')}>{gpuStatus.snapshot.devices.map(device => <article key={device.device} data-busy={!!device.job_id}><strong>{gpuDeviceLabel(device.device)}<span>{device.name}</span></strong>{device.job_id ? <Link to={`/jobs/${encodeURIComponent(device.job_id)}`}>{device.job_name || device.job_id}</Link> : <span>{text('空闲', 'Free')}</span>}{gpuMemorySummary(device,text) && <small title={gpuMemoryDetails(device,text)}>{gpuMemorySummary(device,text)}</small>}</article>)}</div>}
    <div className="queue-controls" data-testid="queue-controls">
    <div className="queue-navigation"><div className="task-tabs ui-tabs" role="tablist" aria-label={text('队列分区', 'Queue views')}>{groups.map((value, index) => { const Icon = [Activity, Clock3, History][index]; return <button key={value} id={`queue-tab-${value}`} role="tab" aria-controls="queue-job-view" tabIndex={group === value ? 0 : -1} aria-selected={group === value} onKeyDown={event => { const next = event.key === 'ArrowRight' ? (index + 1) % 3 : event.key === 'ArrowLeft' ? (index + 2) % 3 : event.key === 'Home' ? 0 : event.key === 'End' ? 2 : -1; if (next >= 0) { event.preventDefault(); change({ view: groups[next], status: null }); document.getElementById(`queue-tab-${groups[next]}`)?.focus(); } }} onClick={() => change({ view: value, status: null })}><Icon size={15}/>{groupLabels[value]}<span>{counts[value] ?? '—'}</span></button>; })}<SlidingIndicator className="ui-tabs-indicator"/></div><details className="queue-options" data-popover><summary className="ui-btn ui-btn-quiet"><SlidersHorizontal size={14}/>{text('调度设置', 'Scheduling options')}</summary><div><label>{text('并行方式', 'Concurrency mode')}<StudioSelect aria-label={text('并行方式', 'Concurrency mode')} value={settings.max_concurrent == null ? 'auto' : 'limit'} disabled={settingBusy} onValueChange={value => void updateSettings({ max_concurrent: value === 'auto' ? null : 1 })} options={[{ value: 'auto', label: text('自动 · 按空闲显卡', 'Automatic · available GPUs') }, { value: 'limit', label: text('限制任务数量', 'Limit concurrent jobs') }]}/></label>{settings.max_concurrent != null && <label>{text('最多并行任务', 'Concurrent jobs')}<input aria-label={text('最多并行任务', 'Concurrent jobs')} type="number" min={1} max={64} key={settings.max_concurrent} defaultValue={settings.max_concurrent} disabled={settingBusy} onBlur={event => { const value = Number(event.target.value); if (event.target.value.trim() && Number.isInteger(value) && value >= 1 && value <= 64) void updateSettings({ max_concurrent: value }); else setError(text('并行数量必须为 1–64。', 'Concurrency must be 1–64.')); }}/></label>}<Switch checked={settings.memory_admission !== false} disabled={settingBusy} onCheckedChange={checked => void updateSettings({ memory_admission: checked })}>{text('启动前检查可用显存', 'Check free memory before launch')}</Switch><p>{text('每张加速卡同一时间由一个任务占用；并行数量是上限。', 'Each accelerator runs one job at a time. Concurrency is an upper limit.')}</p></div></details></div>
    <div className="queue-filters"><label className="queue-search"><Search size={15}/><input aria-label={text('搜索任务', 'Search jobs')} placeholder={text('任务、项目或版本名称', 'Job, project or version name')} value={query} onChange={event => change({ q: event.target.value || null }, true)}/></label><StudioSelect aria-label={text('项目筛选', 'Project filter')} value={project} onValueChange={value => change({ project_id: value })} options={[{ value: '', label: text('所有项目', 'All projects') }, ...projects.map(item => ({ value: item.id, label: item.name }))]}/><StudioSelect aria-label={text('任务类型', 'Job type')} value={type} onValueChange={value => change({ type: value })} options={[{ value: '', label: text('所有类型', 'All job types') }, { value: 'train', label: text('训练', 'Training') }, { value: 'cache', label: text('缓存准备', 'Cache preparation') }, { value: 'xyz', label: text('模型测试', 'Model testing') }]}/><StudioSelect aria-label={text('状态筛选', 'Status filter')} value={status} onValueChange={value => change({ status: value })} options={[{ value: '', label: text('所有状态', 'All statuses') }, ...statuses[group].map(value => ({ value, label: statusLabels[value] }))]}/>{(query || project || type || status) && <button type="button" className="ui-link task-link" onClick={() => change({ q: null, project_id: null, type: null, status: null })}>{text('清除筛选', 'Clear filters')}</button>}</div>
    {error && <div className="task-error" role="alert">{error}<button type="button" className="ui-btn ui-btn-sm" onClick={refresh}>{text('重试', 'Retry')}</button></div>}
    {group === 'waiting' && <p className="queue-helper">{text('优先级高的先启动，同级按创建顺序；排期到点后参与调度。', 'Higher priority starts first, then creation order. Scheduled jobs become eligible at their start time.')}</p>}
    <nav className="task-pagination" aria-label={text('任务分页', 'Job pagination')}><span>{text(`共 ${total} 个任务`, `${total} jobs`)}</span><StudioSelect aria-label={text('每页任务数', 'Jobs per page')} value={String(pageSize)} options={[20, 50, 100].map(value => ({ value: String(value), label: text(`${value} 个 / 页`, `${value} / page`) }))} onValueChange={value => change({ size: value })}/><div><button type="button" className="ui-btn" disabled={loading || page <= 1} onClick={() => change({ page: String(page - 1) })}>{text('上一页', 'Previous')}</button><span>{page} / {pages}</span><button type="button" className="ui-btn" disabled={loading || page >= pages} onClick={() => change({ page: String(page + 1) })}>{text('下一页', 'Next')}</button></div></nav>
    </div>
    <div className="task-table-scroll" id="queue-job-view" role="tabpanel" aria-labelledby={`queue-tab-${group}`} aria-busy={loading}><table className="queue-table" data-testid="jobs-table"><thead><tr><th>{text('任务', 'Job')}</th><th>{text('项目 / 版本', 'Project / version')}</th><th>{text('状态与进度', 'Status & progress')}</th>{group === 'waiting' && <th>{text('优先级', 'Priority')}</th>}<th className="queue-time-column">{text('创建 / 排期', 'Created / scheduled')}</th><th>{text('操作', 'Actions')}</th></tr></thead><tbody>{visibleJobs.map(job => <tr key={job.id} data-testid={`job-row-${job.id}`}><td><Link className="task-name" to={`/jobs/${job.id}?from=queue`} state={{ queueReturnTo: `/queue${params.toString() ? `?${params}` : ''}` }}>{job.name}</Link><small>{job.type === 'xyz' ? text('模型测试', 'Model testing') : job.type === 'cache' ? text('缓存', 'Cache') : text('训练', 'Training')} · {job.id}</small></td><td><JobContext job={job}/></td><td><JobStatus status={job.status}/>{(job.progress?.devices?.length || job.gpu_devices?.length) ? <small>{job.progress?.devices?.length ? text('实际显卡', 'Assigned GPUs') : text('申请显卡', 'Requested GPUs')}: {(job.progress?.devices?.length ? job.progress.devices : job.gpu_devices || []).map(gpuDeviceLabel).join(', ')}</small> : ['queued', 'scheduled'].includes(job.status) && <small>{text('自动选择空闲显卡', 'Automatically choose free GPUs')}</small>}{job.type === 'xyz' && job.progress?.total ? <div className="queue-progress"><span>{job.progress.done ?? 0} / {job.progress.total} {text('张', 'images')}</span><progress max={job.progress.total} value={job.progress.done ?? 0}/></div> : job.progress?.total_steps ? <div className="queue-progress"><span>{job.progress.step ?? 0} / {job.progress.total_steps}</span><progress max={job.progress.total_steps} value={job.progress.step ?? 0}/></div> : job.progress?.phase && <small>{progressLabel(job)}</small>}</td>{group === 'waiting' && <td className="queue-priority-cell" data-label={text('优先级', 'Priority')}><input aria-label={`${text('优先级', 'Priority')}: ${job.name}`} type="number" key={`${job.id}-${job.priority}`} defaultValue={job.priority} onBlur={event => void updatePriority(job, event.target.value)}/></td>}<td className="queue-time-column"><time>{formatTime(job.created_at)}</time>{job.scheduled_at != null && <small>{text('排期', 'Scheduled')}: {formatTime(job.scheduled_at)}</small>}</td><td className="queue-action-cell"><JobActions job={job} onUpdated={refresh}/>{group === 'history' && <button type="button" className="ui-btn ui-btn-quiet ui-btn-sm ui-btn-icon ui-btn-danger task-remove" onClick={() => void remove(job)} aria-label={`${text('删除记录', 'Delete record')}: ${job.name}`}><Trash2 size={12}/></button>}</td></tr>)}</tbody></table>{!visibleJobs.length && !error && <div className="task-empty" data-testid="queue-empty"><Inbox size={28}/><strong>{loading ? text('读取任务…', 'Loading jobs…') : text('此分区没有匹配任务', 'No matching jobs in this view')}</strong><Link className="ui-link" to="/projects">{text('打开项目', 'Open projects')}</Link></div>}</div>
    <p className="queue-scope-note"><Link className="ui-link" to="/settings/environment?tab=models">{text('模型管理', 'Models')}</Link></p>
  </section>;
}
