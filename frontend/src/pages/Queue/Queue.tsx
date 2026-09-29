import { gpuMemoryDetails, formatGpuMemory } from '../../utils/gpuMemory';
import React from 'react';
import { Link, useLocation, useSearchParams } from 'react-router-dom';
import { Activity, Archive, ArchiveRestore, CircleAlert, Clock3, History, SlidersHorizontal, PauseCircle, Play, Search, RefreshCw, Trash2, Inbox, X } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { Job, JobListResponse, Project, QueueSettings } from '../../api/types';
import StudioSelect from '../../components/StudioSelect';
import ConfigHelp from '../../components/ConfigHelp';
import ProgressBar from '../../components/ProgressBar';
import { useQueueDevices } from '../../api/hooks/useQueueDevices';
import { useTelemetryInterval } from '../../api/hooks/useTelemetryInterval';
import { gpuDeviceLabel } from '../../utils/gpuDevices';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { formatApiError } from '../../utils/errors';
import { jobTypeLabel, mergeJobEvent, shortTime } from '../../utils/jobs';
import { useWorkspaceText } from '../../utils/workspaceText';
import { JobActions, JobContext, JobProgressSummary, JobStatus, type ContextJob } from './jobPresentation';
import JobDeleteDialog from './JobDeleteDialog';
import './queue.css';
import Switch from '../../components/Switch';
import { SlidingIndicator } from '../../components/motion';
import PageLocation from '../../components/PageLocation';

type Group = 'active' | 'waiting' | 'history' | 'archive';
const groups: Group[] = ['active', 'waiting', 'history', 'archive'];
const statuses = { active: ['running', 'pausing', 'cancelling', 'paused'], waiting: ['queued', 'scheduled'], history: ['completed', 'failed', 'cancelled'], archive: ['completed', 'failed', 'cancelled'] };
const groupIcons = { active: Activity, waiting: Clock3, history: History, archive: Archive };

function DeviceCell({ job }: { job: Job }) {
  const text = useWorkspaceText();
  const assigned = job.progress?.devices?.length ? job.progress.devices : [];
  const devices = assigned.length ? assigned : job.gpu_devices || [];
  if (!devices.length) return <span className="queue-muted">{['queued', 'scheduled'].includes(job.status) ? text('自动分配', 'Automatic') : '—'}</span>;
  return <><span className="queue-device-list">{devices.map(gpuDeviceLabel).join(', ')}</span><small>{assigned.length ? text('已分配', 'Assigned') : text('申请', 'Requested')}</small></>;
}

export default function Queue() {
  const text = useWorkspaceText();
  const gpuStatus = useQueueDevices(useTelemetryInterval());
  const location = useLocation();
  const blocked = gpuStatus.snapshot?.blocked_reason;
  const [params, setParams] = useSearchParams();
  const group = groups.includes(params.get('view') as Group) ? params.get('view') as Group : 'active';
  const page = Math.max(1, Math.floor(Number(params.get('page')) || 1));
  const pageSize = [20, 50, 100].includes(Number(params.get('size'))) ? Number(params.get('size')) : 20;
  const status = params.get('status') || '', project = params.get('project_id') || '', type = params.get('type') || '', query = params.get('q') || '';
  const scope = JSON.stringify([group, page, pageSize, status, project, type, query]);
  const [loadedScope, setLoadedScope] = React.useState('');
  const [jobs, setJobs] = React.useState<ContextJob[]>([]);
  const [total, setTotal] = React.useState(0);
  const [counts, setCounts] = React.useState<Record<Group, number | null>>({ active: null, waiting: null, history: null, archive: null });
  const [archived, setArchived] = React.useState<Job | null>(null);
  const [deleting, setDeleting] = React.useState<Job | null>(null);
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
  // Deleting from history only archives: the files stay until the job is deleted in the archive.
  const archive = async (job: Job, value: boolean) => {
    setError('');
    try { await apiClient.patch(`/jobs/${encodeURIComponent(job.id)}`, { archived: value }, { silent: true }); setArchived(value ? job : null); refresh(); }
    catch (failure) { setError(formatApiError(failure)); }
  };
  const visibleJobs = scope === loadedScope ? jobs : [];
  const pages = Math.max(1, Math.ceil(total / pageSize));
  const filtered = !!(query || project || type || status);
  const groupLabels = { active: text('运行与暂停', 'Active & paused'), waiting: text('等待调度', 'Waiting'), history: text('训练历史', 'History'), archive: text('归档', 'Archive') };
  const statusLabels: Record<string, string> = { running: text('运行中', 'Running'), pausing: text('暂停中', 'Pausing'), cancelling: text('取消中', 'Cancelling'), paused: text('已暂停', 'Paused'), queued: text('排队中', 'Queued'), scheduled: text('已排期', 'Scheduled'), completed: text('已完成', 'Completed'), failed: text('失败', 'Failed'), cancelled: text('已取消', 'Cancelled') };
  const devices = gpuStatus.snapshot?.devices || [];
  const freeDevices = devices.filter(device => !device.job_id && device.status !== 'unavailable').length;
  const summary = [
    counts.active != null && text(`${counts.active} 个运行或暂停`, `${counts.active} active`),
    counts.waiting != null && text(`${counts.waiting} 个等待`, `${counts.waiting} waiting`),
    devices.length > 0 && text(`${freeDevices} / ${devices.length} 张显卡空闲`, `${freeDevices} of ${devices.length} GPUs free`),
  ].filter(Boolean).join(' · ');
  const emptyTitle = loading ? text('读取任务…', 'Loading jobs…') : filtered ? text('没有符合筛选条件的任务', 'No jobs match the filters') : group === 'archive' ? text('归档是空的', 'The archive is empty') : group === 'history' ? text('还没有训练记录', 'No finished jobs yet') : group === 'waiting' ? text('没有等待中的任务', 'No jobs are waiting') : text('没有运行中的任务', 'No jobs are running');

  return <section className="queue-page task-workspace">
    <header className="queue-header">
      <PageLocation trail={[{ label: text('任务队列', 'Job queue') }]}/><div className="queue-heading"><h1 data-testid="queue-title">{text('任务队列', 'Job queue')}</h1>{summary && <p>{summary}</p>}</div>
      <div className="queue-header-actions">
        <button type="button" className="ui-btn ui-btn-icon" disabled={loading} onClick={refresh} aria-label={text('刷新队列', 'Refresh queue')} title={text('刷新队列', 'Refresh queue')}><RefreshCw size={15}/></button>
        <details className="queue-options" data-popover><summary className="ui-btn"><SlidersHorizontal size={14}/>{text('调度设置', 'Scheduling options')}</summary><div>
          <label>{text('并行方式', 'Concurrency mode')}<StudioSelect aria-label={text('并行方式', 'Concurrency mode')} value={settings.max_concurrent == null ? 'auto' : 'limit'} disabled={settingBusy} onValueChange={value => void updateSettings({ max_concurrent: value === 'auto' ? null : 1 })} options={[{ value: 'auto', label: text('自动 · 按空闲显卡', 'Automatic · available GPUs') }, { value: 'limit', label: text('限制任务数量', 'Limit concurrent jobs') }]}/></label>
          {settings.max_concurrent != null && <label>{text('最多并行任务', 'Concurrent jobs')}<input aria-label={text('最多并行任务', 'Concurrent jobs')} type="number" min={1} max={64} key={settings.max_concurrent} defaultValue={settings.max_concurrent} disabled={settingBusy} onBlur={event => { const value = Number(event.target.value); if (event.target.value.trim() && Number.isInteger(value) && value >= 1 && value <= 64) void updateSettings({ max_concurrent: value }); else setError(text('并行数量必须为 1–64。', 'Concurrency must be 1–64.')); }}/></label>}
          <Switch checked={settings.memory_admission !== false} disabled={settingBusy} onCheckedChange={checked => void updateSettings({ memory_admission: checked })}>{text('启动前检查可用显存', 'Check free memory before launch')}</Switch>
          <p>{text('每张显卡同一时间只运行一个任务；并行数量是上限。', 'Each GPU runs one job at a time. Concurrency is an upper limit.')}</p>
        </div></details>
        <button type="button" className={`ui-btn${settings.held ? ' ui-btn-primary' : ''}`} data-testid="toggle-held" disabled={settingBusy} onClick={() => void updateSettings({ held: !settings.held })}>{settings.held ? <Play size={14}/> : <PauseCircle size={14}/>}{settings.held ? text('继续调度', 'Resume scheduling') : text('暂停调度', 'Hold scheduling')}</button>
      </div>
    </header>
    {settings.held && <p className="task-notice" role="status">{text('调度已暂停：正在运行的任务会继续；等待中的任务不会启动。', 'Scheduling is held. Running jobs continue; waiting jobs will not start.')}</p>}
    {gpuStatus.snapshot?.restart_required ? <p className="task-notice queue-restart-notice" role="status" data-testid="queue-restart-notice"><CircleAlert size={15} aria-hidden="true"/><span>{text('一些环境设置发生了变化，需要重启服务后才能生效，队列将在重启后继续。', 'Some environment settings changed and take effect after the service restarts. The queue continues after the restart.')}</span><Link className="ui-btn ui-btn-sm" to="/settings/preferences?section=interface" state={{ backgroundLocation: location }}>{text('去重启', 'Restart')}</Link></p>
      : blocked && <p className="task-notice" role="status" data-testid="queue-blocked-notice">{text(`新任务暂不启动：${blocked}`, `New jobs are not starting: ${blocked}`)}</p>}
    {gpuStatus.error && <p className="task-notice" role="status">{text('显卡占用暂时无法读取', 'GPU assignments are temporarily unavailable')}: {gpuStatus.error}</p>}
    {devices.length > 0 && <section className="queue-devices" aria-label={text('显卡状态', 'GPU status')}>{devices.map(device => {
      const used = device.mem_used_mb, totalMemory = device.mem_total_mb;
      const unavailable = device.status === 'unavailable';
      return <article key={device.device} className="queue-device" data-busy={!!device.job_id || undefined} data-unavailable={unavailable || undefined}>
        <header className="queue-device-head">
          <div className="queue-device-name"><strong>{gpuDeviceLabel(device.device)}</strong><span title={device.name}>{device.name}</span></div>
          <div className="queue-device-state">{unavailable ? <span className="queue-device-free" data-state="unavailable">{text('不可用', 'Unavailable')}</span>
            : device.job_id ? <><JobStatus status={device.status || 'running'}/><Link to={`/jobs/${encodeURIComponent(device.job_id)}`} title={device.job_name || device.job_id}>{device.job_name || device.job_id}</Link></>
              : <span className="queue-device-free"><span className="queue-dot" aria-hidden="true"/>{text('空闲', 'Free')}</span>}</div>
        </header>
        <dl className="queue-device-metrics">
          <div><dt>{text('使用率', 'Load')}</dt><dd>{device.util_pct == null ? '—' : `${Math.round(device.util_pct)}%`}</dd></div>
          <div><dt>{text('功率', 'Power')}</dt><dd>{device.power_w == null ? '—' : `${device.power_w < 10 ? device.power_w.toFixed(1) : Math.round(device.power_w)} W`}{device.power_limit_w ? <small> / {Math.round(device.power_limit_w)} W</small> : null}</dd></div>
          <div><dt>{text('温度', 'Temperature')}</dt><dd>{device.temp_c == null ? '—' : `${Math.round(device.temp_c)} °C`}</dd></div>
          <div className="queue-device-memory" title={gpuMemoryDetails(device, text)}><dt>{device.memory_scope === 'unified_system' ? text('统一内存', 'Unified memory') : text('显存', 'Memory')}</dt><dd><ProgressBar label={`${gpuDeviceLabel(device.device)} ${text('显存占用', 'memory in use')}`} value={used ?? 0} max={totalMemory ?? 0}/><span>{formatGpuMemory(used)} / {formatGpuMemory(totalMemory)}</span></dd></div>
        </dl>
      </article>;
    })}</section>}
    <div className="queue-board">
      <div className="queue-board-head" data-testid="queue-controls">
        <div className="queue-navigation"><div className="task-tabs ui-tabs" role="tablist" aria-label={text('队列分区', 'Queue views')}>{groups.map((value, index) => { const Icon = groupIcons[value]; const last = groups.length - 1; return <button key={value} id={`queue-tab-${value}`} role="tab" aria-controls="queue-job-view" tabIndex={group === value ? 0 : -1} aria-selected={group === value} onKeyDown={event => { const next = event.key === 'ArrowRight' ? (index + 1) % groups.length : event.key === 'ArrowLeft' ? (index + last) % groups.length : event.key === 'Home' ? 0 : event.key === 'End' ? last : -1; if (next >= 0) { event.preventDefault(); change({ view: groups[next], status: null }); document.getElementById(`queue-tab-${groups[next]}`)?.focus(); } }} onClick={() => { setArchived(null); change({ view: value, status: null }); }}><Icon size={15}/>{groupLabels[value]}<span>{counts[value] ?? '—'}</span></button>; })}<SlidingIndicator className="ui-tabs-indicator"/></div></div>
        <div className="queue-filters"><label className="queue-search"><Search size={15}/><input aria-label={text('搜索任务', 'Search jobs')} placeholder={text('任务、项目或版本名称', 'Job, project or version name')} value={query} onChange={event => change({ q: event.target.value || null }, true)}/></label><StudioSelect aria-label={text('项目筛选', 'Project filter')} value={project} onValueChange={value => change({ project_id: value })} options={[{ value: '', label: text('所有项目', 'All projects') }, ...projects.map(item => ({ value: item.id, label: item.name }))]}/><StudioSelect aria-label={text('任务类型', 'Job type')} value={type} onValueChange={value => change({ type: value })} options={[{ value: '', label: text('所有类型', 'All job types') }, { value: 'train', label: text('训练', 'Training') }, { value: 'cache', label: text('缓存准备', 'Cache preparation') }, { value: 'xyz', label: text('模型测试', 'Model testing') }]}/><StudioSelect aria-label={text('状态筛选', 'Status filter')} value={status} onValueChange={value => change({ status: value })} options={[{ value: '', label: text('所有状态', 'All statuses') }, ...statuses[group].map(value => ({ value, label: statusLabels[value] }))]}/>{filtered && <button type="button" className="ui-btn ui-btn-quiet queue-clear" onClick={() => change({ q: null, project_id: null, type: null, status: null })}><X size={14}/>{text('清除筛选', 'Clear filters')}</button>}</div>
      </div>
      {error && <div className="task-error" role="alert">{error}<button type="button" className="ui-btn ui-btn-sm" onClick={refresh}>{text('重试', 'Retry')}</button></div>}
      {archived && <p className="task-notice queue-notice" role="status"><span>{text(`已将“${archived.name}”移到归档，权重和记录仍保留。`, `Moved “${archived.name}” to the archive; its weights and records are kept.`)}</span><button type="button" className="ui-btn ui-btn-sm" onClick={() => void archive(archived, false)}>{text('撤销', 'Undo')}</button><button type="button" className="ui-btn ui-btn-sm ui-btn-quiet ui-btn-icon" onClick={() => setArchived(null)} aria-label={text('关闭提示', 'Dismiss')}><X size={13}/></button></p>}
      {group === 'archive' && <p className="queue-archive-note">{text('归档的任务不再出现在队列和项目结果里，文件保留。恢复后回到训练历史；在这里删除会移除任务的所有文件。', 'Archived jobs leave the queue and project results but keep their files. Restore returns a job to History; deleting here removes all of its files.')}</p>}
      <div className="queue-list" id="queue-job-view" role="tabpanel" aria-labelledby={`queue-tab-${group}`} aria-busy={loading}>
        {visibleJobs.length > 0 && <table className="queue-table" data-testid="jobs-table"><thead><tr><th>{text('任务', 'Job')}</th><th>{text('项目 / 版本', 'Project / version')}</th><th>{text('状态', 'Status')}</th><th>{text('显卡', 'GPU')}</th>{group === 'waiting' && <th><span className="queue-th-help">{text('优先级', 'Priority')}<ConfigHelp label={text('优先级 · 说明', 'Priority help')}>{text('数值高的任务先启动，同级按创建顺序；已排期的任务到点后参与调度。', 'Higher values start first, then creation order. Scheduled jobs become eligible at their start time.')}</ConfigHelp></span></th>}<th>{group === 'waiting' ? text('创建 / 排期', 'Created / scheduled') : text('创建时间', 'Created')}</th><th className="queue-action-head"><span className="sr-only">{text('操作', 'Actions')}</span></th></tr></thead>
          <tbody>{visibleJobs.map(job => <tr key={job.id} data-testid={`job-row-${job.id}`} data-status={job.status}>
            <td className="queue-job-cell"><Link className="queue-job-name" to={`/jobs/${job.id}?from=queue`} state={{ queueReturnTo: `/queue${params.toString() ? `?${params}` : ''}` }}>{job.name}</Link><small><span className="queue-type">{jobTypeLabel(job.type, text)}</span>{job.id}</small></td>
            <td><JobContext job={job}/></td>
            <td className="queue-status-cell"><JobStatus status={job.status}/><JobProgressSummary job={job}/></td>
            <td className="queue-device-cell"><DeviceCell job={job}/></td>
            {group === 'waiting' && <td className="queue-priority-cell" data-label={text('优先级', 'Priority')}><input aria-label={`${text('优先级', 'Priority')}: ${job.name}`} type="number" key={`${job.id}-${job.priority}`} defaultValue={job.priority} onBlur={event => void updatePriority(job, event.target.value)}/></td>}
            <td className="queue-time-cell"><time dateTime={new Date(job.created_at * 1000).toISOString()} title={new Date(job.created_at * 1000).toLocaleString()}>{shortTime(job.created_at)}</time>{job.scheduled_at != null && <small>{text('排期', 'Scheduled')} {shortTime(job.scheduled_at)}</small>}</td>
            <td className="queue-action-cell"><div>{group === 'archive' ? <>
              <button type="button" className="ui-btn ui-btn-sm" onClick={() => void archive(job, false)} aria-label={`${text('恢复', 'Restore')}: ${job.name}`} title={text('移回训练历史', 'Move back to History')}><ArchiveRestore size={13}/>{text('恢复', 'Restore')}</button>
              <button type="button" className="ui-btn ui-btn-sm ui-btn-danger" onClick={() => setDeleting(job)} aria-label={`${text('彻底删除', 'Delete permanently')}: ${job.name}`} title={text('删除任务和它的所有文件', 'Delete the job and all of its files')}><Trash2 size={13}/>{text('删除', 'Delete')}</button>
            </> : <><JobActions job={job} onUpdated={refresh}/>{group === 'history' && <button type="button" className="ui-btn ui-btn-quiet ui-btn-sm ui-btn-icon" onClick={() => void archive(job, true)} aria-label={`${text('归档', 'Archive')}: ${job.name}`} title={text('移到归档，文件保留', 'Move to the archive; files are kept')}><Archive size={13}/></button>}</>}</div></td>
          </tr>)}</tbody></table>}
        {!visibleJobs.length && !error && <div className="queue-empty" data-testid="queue-empty"><Inbox size={26} aria-hidden="true"/><strong>{emptyTitle}</strong>{!loading && (filtered ? <button type="button" className="ui-btn ui-btn-sm" onClick={() => change({ q: null, project_id: null, type: null, status: null })}>{text('清除筛选', 'Clear filters')}</button> : group === 'archive' ? <span>{text('在训练历史里归档的任务会出现在这里。', 'Jobs archived from History appear here.')}</span> : group !== 'history' && <Link className="ui-btn ui-btn-sm" to="/projects">{text('打开项目', 'Open projects')}</Link>)}</div>}
      </div>
      {deleting && <JobDeleteDialog job={deleting} onClose={() => setDeleting(null)} onDeleted={() => { setDeleting(null); refresh(); }}/>}
      {(total > 0 || page > 1) && <nav className="queue-footer" data-testid="queue-pagination" aria-label={text('任务分页', 'Job pagination')}><span>{text(`共 ${total} 个任务`, `${total} jobs`)}</span><div className="queue-pages"><StudioSelect aria-label={text('每页任务数', 'Jobs per page')} value={String(pageSize)} options={[20, 50, 100].map(value => ({ value: String(value), label: text(`${value} 个 / 页`, `${value} / page`) }))} onValueChange={value => change({ size: value })}/><button type="button" className="ui-btn ui-btn-sm" disabled={loading || page <= 1} onClick={() => change({ page: String(page - 1) })}>{text('上一页', 'Previous')}</button><span className="queue-page-number">{page} / {pages}</span><button type="button" className="ui-btn ui-btn-sm" disabled={loading || page >= pages} onClick={() => change({ page: String(page + 1) })}>{text('下一页', 'Next')}</button></div></nav>}
    </div>
  </section>;
}
