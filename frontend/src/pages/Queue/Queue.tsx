import { mergeJobEvent } from '../../utils/jobs';
import React from 'react';
import { useTranslation } from 'react-i18next';
import { apiClient } from '../../api/client';
import { Job, JobListResponse, QueueSettings } from '../../api/types';
import { Link } from 'react-router-dom';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { formatTime } from '../../utils/format';
import { Play, Pause, XCircle, Save, RefreshCcw, Trash2, PauseCircle, Inbox } from 'lucide-react';

// 状态徽标：running 绿 / paused·pausing 琥珀 / cancelling·failed 红 / completed 蓝 / 其余 slate
function statusBadgeClass(status: string): string {
  if (status === 'running') {
    return 'bg-green-100 text-green-700 dark:bg-green-950/40 dark:text-green-400';
  }
  if (status === 'paused' || status === 'pausing' || status === 'pauseing') {
    return 'bg-amber-100 text-amber-700 dark:bg-amber-950/40 dark:text-amber-400';
  }
  if (status === 'cancelling' || status === 'canceling' || status === 'failed') {
    return 'bg-red-100 text-red-700 dark:bg-red-950/40 dark:text-red-400';
  }
  if (status === 'completed') {
    return 'bg-blue-100 text-blue-700 dark:bg-blue-950/40 dark:text-blue-400';
  }
  return 'bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-300';
}

// locales 尚无 queue.status.* 键，经 defaultValue 提供中文（含乐观状态 *ing 变体）
const STATUS_DEFAULTS: Record<string, string> = {
  queued: '排队中',
  scheduled: '已排期',
  running: '运行中',
  pausing: '暂停中',
  pauseing: '暂停中',
  paused: '已暂停',
  resumeing: '恢复中',
  saveing: '保存中',
  retrying: '重试中',
  cancelling: '取消中',
  canceling: '取消中',
  cancelled: '已取消',
  completed: '已完成',
  failed: '失败',
};

export default function Queue() {
  const { t } = useTranslation();
  const [page, setPage] = React.useState(1);
  const [total, setTotal] = React.useState(0);
  const [statusFilter, setStatusFilter] = React.useState('');
  const pageSize = 50;
  const requestRef = React.useRef(0);
  const [jobs, setJobs] = React.useState<Job[]>([]);
  const [settings, setSettings] = React.useState<QueueSettings>({ held: false, max_concurrent: 1, memory_admission: true });
  const [optimisticStates, setOptimisticStates] = React.useState<Record<string, string>>({});

  const statusLabel = (status: string): string =>
    t(`queue.status.${status}`, STATUS_DEFAULTS[status] ?? status);

  const fetchJobs = React.useCallback(() => {
    const requestId = ++requestRef.current;
    apiClient.get<JobListResponse | Job[]>('/jobs', { params: { page, page_size: pageSize, status: statusFilter || undefined } }).then((data) => {
      if (requestId !== requestRef.current) return;
      const items = Array.isArray(data) ? data : data.items;
      const count = Array.isArray(data) ? data.length : data.total;
      setJobs(items); setTotal(count);
      if (page > 1 && items.length === 0) setPage(Math.max(1, Math.ceil(count / pageSize)));
    }).catch(console.error);
  }, [page, statusFilter]);

  const fetchSettings = () => {
    apiClient.get<QueueSettings>('/queue/settings').then(setSettings).catch(console.error);
  };

  React.useEffect(() => {
    fetchSettings();
  }, []);

  React.useEffect(() => { fetchJobs(); }, [fetchJobs]);

  // 监听 job.state 更新
  useEventStream(EVENT_TYPES.JOB_STATE, (data: any) => {
    setJobs((prev) =>
      prev.map((j) =>
        j.id === data.job_id
          ? mergeJobEvent(j, data)
          : j
      )
    );
    if (statusFilter) fetchJobs();
    // 状态已确认，清除乐观状态
    setOptimisticStates((prev) => {
      const next = { ...prev };
      delete next[data.job_id];
      return next;
    });
  });

  useEventStream(EVENT_TYPES.JOB_STEP, (data: any) => setJobs((rows) => rows.map((job) => mergeJobEvent(job, data))));
  useEventStream(EVENT_TYPES.JOB_PHASE, (data: any) => setJobs((rows) => rows.map((job) => mergeJobEvent(job, data))));

  // 监听队列变化刷新
  useEventStream(EVENT_TYPES.QUEUE_CHANGED, () => {
    fetchJobs();
  });

  const handleAction = (jobId: string, action: string) => {
    if (action === 'cancel' && !window.confirm(t('queue.cancelConfirm'))) return;
    // 乐观 UI 状态
    setOptimisticStates((prev) => ({ ...prev, [jobId]: `${action}ing` }));

    apiClient.post<Job>(`/jobs/${jobId}/${action}`, {})
      .then((updatedJob) => {
        if (updatedJob.id === jobId) setJobs((prev) => prev.map((j) => (j.id === jobId ? updatedJob : j)));
        else fetchJobs();
        // 清除乐观状态
        setOptimisticStates((prev) => {
          const next = { ...prev };
          delete next[jobId];
          return next;
        });
      })
      .catch((err) => {
        console.error(err);
        setOptimisticStates((prev) => {
          const next = { ...prev };
          delete next[jobId];
          return next;
        });
      });
  };

  const handleDelete = (jobId: string) => {
    if (window.confirm(t('queue.deleteConfirm'))) {
      apiClient.delete(`/jobs/${jobId}`).then(fetchJobs).catch(console.error);
    }
  };

  const handleToggleHeld = () => {
    const nextHeld = !settings.held;
    apiClient.put<QueueSettings>('/queue/settings', { held: nextHeld })
      .then(setSettings)
      .catch(console.error);
  };

  // ---- 拖拽调整优先级：拖到目标行插入并全局 normalize（自上而下 10 递减步进），逐个 PATCH 变更项 ----
  const dragJobIdRef = React.useRef<string | null>(null);
  const [dropTarget, setDropTarget] = React.useState<{ id: string; position: 'above' | 'below' } | null>(null);

  const handleRowDragStart = (jobId: string) => {
    dragJobIdRef.current = jobId;
  };

  const handleRowDragOver = (e: React.DragEvent<HTMLTableRowElement>, jobId: string) => {
    if (!dragJobIdRef.current || dragJobIdRef.current === jobId) return;
    e.preventDefault();
    const rect = e.currentTarget.getBoundingClientRect();
    const position = e.clientY < rect.top + rect.height / 2 ? 'above' : 'below';
    setDropTarget({ id: jobId, position });
  };

  const handleRowDrop = (e: React.DragEvent<HTMLTableRowElement>, targetJob: Job) => {
    e.preventDefault();
    const draggedId = dragJobIdRef.current;
    const position = dropTarget?.position || 'below';
    dragJobIdRef.current = null;
    setDropTarget(null);
    if (!draggedId || draggedId === targetJob.id) return;

    // 1) 计算新顺序：把被拖行插入目标行上方/下方
    const rest = jobs.filter((j) => j.id !== draggedId);
    const dragged = jobs.find((j) => j.id === draggedId);
    if (!dragged) return;
    const targetIdx = rest.findIndex((j) => j.id === targetJob.id);
    if (targetIdx === -1) return;
    const insertAt = position === 'above' ? targetIdx : targetIdx + 1;
    rest.splice(insertAt, 0, dragged);

    // 2) 全局 normalize：按显示顺序自上而下分配 10 步进递减优先级
    const normalized = rest.map((j, idx) => ({ ...j, priority: (rest.length - idx) * 10 }));
    setJobs(normalized);

    // 3) 仅 PATCH 优先级发生变化的项
    const changed = normalized.filter((j, idx) => jobs.find((o) => o.id === j.id)?.priority !== (rest.length - idx) * 10);
    Promise.all(changed.map((c) => apiClient.patch(`/jobs/${c.id}`, { priority: c.priority })))
      .then(fetchJobs)
      .catch((err) => {
        console.error(err);
        fetchJobs();
      });
  };

  const handleDragEnd = () => {
    dragJobIdRef.current = null;
    setDropTarget(null);
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold" data-testid="queue-title">{t('queue.title')}</h2>
        <button
          onClick={handleToggleHeld}
          data-testid="toggle-held"
          className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium ${
            settings.held
              ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400'
              : 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300'
          }`}
        >
          <PauseCircle className="w-4 h-4" />
          <span>{settings.held ? t('queue.schedulingHeld') : t('queue.pauseScheduling')}</span>
        </button>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 text-sm">
        <select aria-label={t('queue.statusFilter')} value={statusFilter} onChange={(e) => { setStatusFilter(e.target.value); setPage(1); }} className="rounded border px-3 py-2 dark:bg-slate-900 dark:border-slate-600">
          <option value="">{t('queue.allStatuses')}</option>
          {['queued', 'scheduled', 'running', 'pausing', 'cancelling', 'paused', 'completed', 'failed', 'cancelled'].map((status) => <option key={status} value={status}>{statusLabel(status)}</option>)}
        </select>
        <label className="flex items-center gap-2"><input type="checkbox" checked={settings.memory_admission !== false}
          onChange={(e) => apiClient.put<QueueSettings>('/queue/settings', { memory_admission: e.target.checked }).then(setSettings).catch(console.error)} />{t('queue.memoryAdmission')}</label>
        <div className="flex items-center gap-3">
          <span>{t('queue.pagination', { page, pages: Math.max(1, Math.ceil(total / pageSize)), total })}</span>
          <button disabled={page <= 1} onClick={() => setPage((v) => v - 1)} className="rounded border px-3 py-1 disabled:opacity-40">{t('common.previous')}</button>
          <button disabled={page * pageSize >= total} onClick={() => setPage((v) => v + 1)} className="rounded border px-3 py-1 disabled:opacity-40">{t('common.next')}</button>
        </div>
      </div>
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-x-auto">
        <table className="w-full text-left border-collapse" data-testid="jobs-table">
          <thead>
            <tr className="border-b border-slate-200 dark:border-slate-700 text-xs font-semibold text-slate-400">
              <th className="p-4">{t('queue.id')}</th>
              <th className="p-4">{t('queue.name')}</th>
              <th className="p-4">{t('queue.type')}</th>
              <th className="p-4">{t('queue.project')}</th>
              <th className="p-4">{t('common.status')}</th>
              <th className="p-4">{t('queue.progress')}</th>
              <th className="p-4">{t('queue.priority')}</th>
              <th className="p-4">{t('queue.created')}</th>
              <th className="p-4 text-right">{t('queue.actions')}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200 dark:divide-slate-700 text-sm">
            {jobs.length === 0 && (
              <tr>
                <td colSpan={9} className="p-12">
                  <div className="flex flex-col items-center justify-center text-center space-y-2" data-testid="queue-empty">
                    <Inbox className="w-10 h-10 text-slate-300 dark:text-slate-600" />
                    <div className="text-sm font-medium text-slate-500 dark:text-slate-400">{t('queue.empty')}</div>
                    <div className="text-xs text-slate-400 dark:text-slate-500">
                      {t('queue.emptyHint', '从项目详情页发起训练任务后，会显示在这里。')}
                    </div>
                  </div>
                </td>
              </tr>
            )}
            {jobs.map((job) => {
              const displayStatus = optimisticStates[job.id] || job.status;
              const isDropTarget = dropTarget?.id === job.id;
              return (
                <tr
                  key={job.id}
                  className={`hover:bg-slate-50 dark:hover:bg-slate-750 ${
                    isDropTarget
                      ? dropTarget?.position === 'above'
                        ? 'border-t-2 border-t-blue-500'
                        : 'border-b-2 border-b-blue-500'
                      : ''
                  }`}
                  data-testid={`job-row-${job.id}`}
                  draggable
                  onDragStart={() => handleRowDragStart(job.id)}
                  onDragOver={(e) => handleRowDragOver(e, job.id)}
                  onDrop={(e) => handleRowDrop(e, job)}
                  onDragEnd={handleDragEnd}
                >
                  <td className="p-4 font-mono text-xs">{job.id}</td>
                  <td className="p-4 font-medium">
                    <Link to={`/jobs/${job.id}`} className="hover:underline text-blue-500">
                      {job.name}
                    </Link>
                  </td>
                  <td className="p-4 capitalize text-slate-500">{job.type}</td>
                  <td className="p-4 text-slate-500">{job.project_id || '-'}</td>
                  <td className="p-4">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${statusBadgeClass(displayStatus)}`}>
                      {statusLabel(displayStatus)}
                    </span>
                    {job.progress?.phase === 'waiting_for_device' && <p className="mt-1 text-xs text-amber-600">{t('phase.waiting_for_device')}</p>}
                  </td>
                  <td className="p-4">
                    {job.progress && job.progress.step != null && job.progress.total_steps != null && job.progress.total_steps > 0 ? (
                      <div className="space-y-1">
                        <div className="text-xs text-slate-500 font-mono">{job.progress.step} / {job.progress.total_steps}</div>
                        <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
                          <div
                            className="bg-blue-600 h-1.5 rounded-full transition-all duration-300"
                            style={{ width: `${Math.min(100, (job.progress.step / job.progress.total_steps) * 100)}%` }}
                          />
                        </div>
                      </div>
                    ) : '--'}
                  </td>
                  <td className="p-4">
                    <input
                      type="number"
                      key={`${job.id}-${job.priority}`}
                      defaultValue={job.priority}
                      aria-label={t('queue.priority')}
                      onBlur={(e) => {
                        const val = Number(e.target.value);
                        if (val !== job.priority) {
                          apiClient.patch(`/jobs/${job.id}`, { priority: val }).then(fetchJobs).catch(console.error);
                        }
                      }}
                      className="w-16 px-2 py-1 border rounded text-sm dark:bg-slate-900 dark:border-slate-600"
                    />
                  </td>
                  <td className="p-4 text-xs text-slate-400">
                    {formatTime(job.created_at)}
                  </td>
                  <td className="p-4 text-right">
                    <div className="flex items-center justify-end space-x-2">
                      {job.status === 'running' && (
                        <button
                          onClick={() => handleAction(job.id, 'pause')}
                          className="p-1 text-slate-500 hover:text-amber-600"
                          title={t('queue.pause')}
                          data-testid={`job-pause-${job.id}`}
                        >
                          <Pause className="w-4 h-4" />
                        </button>
                      )}
                      {job.status === 'paused' && (
                        <button
                          onClick={() => handleAction(job.id, 'resume')}
                          className="p-1 text-slate-500 hover:text-green-600"
                          title={t('queue.resume')}
                          data-testid={`job-resume-${job.id}`}
                        >
                          <Play className="w-4 h-4" />
                        </button>
                      )}
                      {job.status === 'running' && (
                        <button
                          onClick={() => handleAction(job.id, 'save')}
                          className="p-1 text-slate-500 hover:text-blue-600"
                          title={t('queue.save')}
                          data-testid={`job-save-${job.id}`}
                        >
                          <Save className="w-4 h-4" />
                        </button>
                      )}
                      <button
                        onClick={() => handleAction(job.id, 'retry')}
                        className="p-1 text-slate-500 hover:text-indigo-600"
                        title={t('queue.retry')}
                        data-testid={`job-retry-${job.id}`}
                      >
                        <RefreshCcw className="w-4 h-4" />
                      </button>
                      {(['running', 'queued', 'scheduled', 'paused'].includes(job.status)) && (
                        <button
                          onClick={() => handleAction(job.id, 'cancel')}
                          className="p-1 text-slate-500 hover:text-red-600"
                          title={t('queue.cancel')}
                          data-testid={`job-cancel-${job.id}`}
                        >
                          <XCircle className="w-4 h-4" />
                        </button>
                      )}
                      <button
                        onClick={() => handleDelete(job.id)}
                        className="p-1 text-slate-500 hover:text-red-700"
                        title={t('queue.deleteRecord')}
                        data-testid={`job-delete-${job.id}`}
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
