import React from 'react';
import { Link } from 'react-router-dom';
import { Pause, Play, Save, RotateCcw, XCircle, Loader2 } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { Job } from '../../api/types';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import { projectUrl } from '../../utils/projectVersions';

export type ContextJob = Job & { project_name?: string | null; version_name?: string | null; version_number?: number | null };
export function JobContext({ job }: { job: ContextJob }) {
  const text = useWorkspaceText();
  if (!job.project_id) return <span className="task-muted">{text('独立任务', 'Standalone job')}</span>;
  const number = job.version_number ? `v${job.version_number}` : '';
  const name = job.version_name?.trim() || '';
  const includesNumber = number && new RegExp(`^${number}(?:$|[\\s·:：-])`, 'i').test(name);
  const version = number && name && !includesNumber ? `${number} · ${name}` : name || number || text('所属版本', 'Job version');
  return <Link className="task-context" to={projectUrl(job.project_id, job.version_id, 'results')}>
    <strong>{job.project_name || job.project_id}</strong>
    <small>{version}</small>
  </Link>;
}
export function JobStatus({ status }: { status: string }) {
  const text = useWorkspaceText();
  const labels: Record<string, [string, string]> = { running: ['运行中', 'Running'], paused: ['已暂停', 'Paused'], pausing: ['暂停中', 'Pausing'], cancelling: ['取消中', 'Cancelling'], queued: ['排队中', 'Queued'], scheduled: ['已排期', 'Scheduled'], completed: ['已完成', 'Completed'], failed: ['失败', 'Failed'], cancelled: ['已取消', 'Cancelled'] };
  return <span className="task-status" data-status={status}>{labels[status] ? text(...labels[status]) : status}</span>;
}
export function JobActions({ job, onUpdated }: { job: Job; onUpdated: (updated: Job) => void }) {
  const text = useWorkspaceText();
  const [busy, setBusy] = React.useState('');
  const [error, setError] = React.useState('');
  const current = React.useRef(job.id); current.current = job.id;
  const run = async (action: string) => {
    if (action === 'cancel' && !window.confirm(text(`取消任务“${job.name}”？任务将在安全位置停止。已有产物会保留。`, `Cancel “${job.name}”? It will stop at a safe point. Existing outputs are kept.`))) return;
    const id = job.id; setBusy(action); setError('');
    try { const result = await apiClient.post<Job>(`/jobs/${id}/${action}`, {}, { silent: true }); if (current.current === id) onUpdated(result); }
    catch (failure) { if (current.current === id) setError(formatApiError(failure)); }
    finally { if (current.current === id) setBusy(''); }
  };
  const actions: { key: string; label: string; Icon: typeof Play }[] = [];
  if (['running', 'queued', 'scheduled'].includes(job.status)) actions.push({ key: 'pause', label: text('暂停', 'Pause'), Icon: Pause });
  if (job.status === 'paused') actions.push({ key: 'resume', label: text('继续', 'Resume'), Icon: Play });
  if (job.status === 'running' && job.type !== 'cache') actions.push({ key: 'save', label: text('保存检查点', 'Save checkpoint'), Icon: Save });
  if (['running', 'queued', 'scheduled', 'paused', 'pausing'].includes(job.status)) actions.push({ key: 'cancel', label: text('取消', 'Cancel'), Icon: XCircle });
  if (['failed', 'cancelled', 'completed'].includes(job.status)) actions.push({ key: 'retry', label: job.type === 'cache' ? text('重新准备', 'Prepare again') : text('重新训练', 'Run again'), Icon: RotateCcw });
  return <div className="task-actions-wrap"><div className="task-actions">{actions.map(({ key, label, Icon }) => <button key={key} type="button" className="task-button" disabled={!!busy} data-testid={`job-${key}-${job.id}`} onClick={() => void run(key)}>{busy === key ? <Loader2 size={13} className="animate-spin"/> : <Icon size={13}/>}<span>{label}</span></button>)}</div>{error && <p className="task-inline-error" role="alert">{error}</p>}</div>;
}
