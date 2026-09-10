import React from 'react';
import { apiClient } from '../../api/client';
import { Job, JobListResponse, QueueSettings } from '../../api/types';
import { Link } from 'react-router-dom';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { Play, Pause, XCircle, Save, RefreshCcw, Trash2, PauseCircle } from 'lucide-react';

export default function Queue() {
  const [jobs, setJobs] = React.useState<Job[]>([]);
  const [settings, setSettings] = React.useState<QueueSettings>({ held: false, max_concurrent: 1 });
  const [optimisticStates, setOptimisticStates] = React.useState<Record<string, string>>({});

  const fetchJobs = () => {
    apiClient.get<JobListResponse | Job[]>('/jobs').then((data) => {
      // 兼容分页响应 {items, total, page, page_size} 与简单数组
      if (Array.isArray(data)) {
        setJobs(data);
      } else if (data && Array.isArray(data.items)) {
        setJobs(data.items);
      }
    }).catch(console.error);
  };

  const fetchSettings = () => {
    apiClient.get<QueueSettings>('/queue/settings').then(setSettings).catch(console.error);
  };

  React.useEffect(() => {
    fetchJobs();
    fetchSettings();
  }, []);

  // 监听 job.state 更新
  useEventStream(EVENT_TYPES.JOB_STATE, (data: any) => {
    setJobs((prev) =>
      prev.map((j) =>
        j.id === data.job_id
          ? { ...j, status: data.status, progress: data.progress || j.progress }
          : j
      )
    );
    // 状态已确认，清除乐观状态
    setOptimisticStates((prev) => {
      const next = { ...prev };
      delete next[data.job_id];
      return next;
    });
  });

  // 监听队列变化刷新
  useEventStream(EVENT_TYPES.QUEUE_CHANGED, () => {
    fetchJobs();
  });

  const handleAction = (jobId: string, action: string) => {
    // 乐观 UI 状态
    setOptimisticStates((prev) => ({ ...prev, [jobId]: `${action}ing` }));

    apiClient.post<Job>(`/jobs/${jobId}/${action}`, {})
      .then((updatedJob) => {
        setJobs((prev) => prev.map((j) => (j.id === jobId ? updatedJob : j)));
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
    if (window.confirm('Are you sure you want to delete this job record?')) {
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
        <h2 className="text-2xl font-bold">Job Queue</h2>
        <button
          onClick={handleToggleHeld}
          className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium ${
            settings.held
              ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400'
              : 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300'
          }`}
        >
          <PauseCircle className="w-4 h-4" />
          <span>{settings.held ? 'Scheduling Held' : 'Pause Scheduling'}</span>
        </button>
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
        <table className="w-full text-left border-collapse" data-testid="jobs-table">
          <thead>
            <tr className="border-b border-slate-200 dark:border-slate-700 text-xs font-semibold text-slate-400">
              <th className="p-4">ID</th>
              <th className="p-4">Name</th>
              <th className="p-4">Type</th>
              <th className="p-4">Project</th>
              <th className="p-4">Status</th>
              <th className="p-4">Progress</th>
              <th className="p-4">Priority</th>
              <th className="p-4">Created</th>
              <th className="p-4 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200 dark:divide-slate-700 text-sm">
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
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      displayStatus === 'running'
                        ? 'bg-green-100 text-green-700 dark:bg-green-950/40 dark:text-green-400'
                        : displayStatus === 'paused'
                        ? 'bg-amber-100 text-amber-700 dark:bg-amber-950/40 dark:text-amber-400'
                        : displayStatus.includes('ing')
                        ? 'bg-blue-100 text-blue-700 dark:bg-blue-950/40 dark:text-blue-400'
                        : 'bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-300'
                    }`}>
                      {displayStatus}
                    </span>
                  </td>
                  <td className="p-4">
                    {job.progress && job.progress.step != null && job.progress.total_steps != null && job.progress.total_steps > 0 ? (
                      <div className="space-y-1">
                        <div className="text-xs text-slate-500">{job.progress.step} / {job.progress.total_steps}</div>
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
                      defaultValue={job.priority}
                      onBlur={(e) => {
                        const val = Number(e.target.value);
                        if (val !== job.priority) {
                          apiClient.patch(`/jobs/${job.id}`, { priority: val }).then(fetchJobs);
                        }
                      }}
                      className="w-16 px-2 py-1 border rounded text-sm dark:bg-slate-900 dark:border-slate-600"
                    />
                  </td>
                  <td className="p-4 text-xs text-slate-400">
                    {(() => {
                      const t = job.created_at;
                      if (typeof t === 'number') return new Date(t * 1000).toLocaleString();
                      return new Date(t).toLocaleString();
                    })()}
                  </td>
                  <td className="p-4 text-right">
                    <div className="flex items-center justify-end space-x-2">
                      {job.status === 'running' && (
                        <button onClick={() => handleAction(job.id, 'pause')} className="p-1 text-slate-500 hover:text-amber-600" title="Pause">
                          <Pause className="w-4 h-4" />
                        </button>
                      )}
                      {job.status === 'paused' && (
                        <button onClick={() => handleAction(job.id, 'resume')} className="p-1 text-slate-500 hover:text-green-600" title="Resume">
                          <Play className="w-4 h-4" />
                        </button>
                      )}
                      {job.status === 'running' && (
                        <button onClick={() => handleAction(job.id, 'save')} className="p-1 text-slate-500 hover:text-blue-600" title="Save Checkpoint">
                          <Save className="w-4 h-4" />
                        </button>
                      )}
                      <button onClick={() => handleAction(job.id, 'retry')} className="p-1 text-slate-500 hover:text-indigo-600" title="Retry">
                        <RefreshCcw className="w-4 h-4" />
                      </button>
                      {(job.status === 'running' || job.status === 'queued') && (
                        <button onClick={() => handleAction(job.id, 'cancel')} className="p-1 text-slate-500 hover:text-red-600" title="Cancel">
                          <XCircle className="w-4 h-4" />
                        </button>
                      )}
                      <button onClick={() => handleDelete(job.id)} className="p-1 text-slate-500 hover:text-red-700" title="Delete">
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
