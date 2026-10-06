import React from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Pause, Play, Save, RotateCcw, XCircle, Loader2, Zap } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { Job } from '../../api/types';
import ProgressBar from '../../components/ProgressBar';
import { useConfirmation } from '../../components/useConfirmation';
import { formatApiError } from '../../utils/errors';
import { formatEta } from '../../utils/format';
import { useWorkspaceText } from '../../utils/workspaceText';
import { projectUrl } from '../../utils/projectVersions';
import { shortTime } from '../../utils/jobs';

const DEVICE_WAIT = 'waiting for a free accelerator with enough memory';

/** The line under a job's status: progress while it runs, the reason once it stops. */
export function JobProgressSummary({ job }: { job: Job }) {
  const text = useWorkspaceText();
  const { t } = useTranslation();
  const progress = job.progress || {};
  const active = ['running', 'pausing', 'cancelling', 'paused'].includes(job.status);
  // A pause another run forced; the job waits until someone resumes it.
  const preempted = ['pausing', 'paused'].includes(job.status) && progress.preempted_by ? text(`为「${progress.preempted_by}」暂停`, `Paused for “${progress.preempted_by}”`) : '';
  if (job.type === 'xyz' && progress.total && active) {
    return <div className="queue-progress"><ProgressBar label={text('测试进度', 'Testing progress')} value={progress.done ?? 0} max={progress.total}/><span>{progress.done ?? 0} / {progress.total} {text('张', 'images')}</span></div>;
  }
  if (active && progress.total_steps) {
    const step = progress.step ?? 0;
    const eta = job.status === 'running' && progress.eta_s != null ? ` · ${text('剩余', 'left')} ${formatEta(progress.eta_s)}` : '';
    return <><div className="queue-progress"><ProgressBar label={text('训练进度', 'Training progress')} value={step} max={progress.total_steps}/><span>{step} / {progress.total_steps} · {Math.floor(step / progress.total_steps * 100)}%{eta}</span></div>{preempted && <small className="queue-status-note">{preempted}</small>}</>;
  }
  if (active || job.status === 'queued') {
    const reason = preempted || (progress.wait_reason && progress.wait_reason !== DEVICE_WAIT ? progress.wait_reason : progress.phase ? t(`phase.${progress.phase}`, t('job.phaseInProgress')) : '');
    return reason ? <small className="queue-status-note">{reason}</small> : null;
  }
  if (job.status === 'scheduled' && job.scheduled_at != null) return <small className="queue-status-note">{text(`${shortTime(job.scheduled_at)} 开始`, `Starts ${shortTime(job.scheduled_at)}`)}</small>;
  if (job.status === 'failed') {
    const reason = (job.error || '').trim().split('\n')[0];
    return reason ? <small className="queue-status-note queue-status-error" title={job.error || undefined}>{reason}</small> : null;
  }
  const elapsed = job.started_at != null && job.finished_at != null ? job.finished_at - job.started_at : null;
  const reached = progress.step != null ? (job.type === 'xyz' ? '' : text(`${progress.step} 步`, `${progress.step} steps`)) : '';
  const parts = [reached, elapsed != null ? text(`用时 ${formatEta(elapsed)}`, `took ${formatEta(elapsed)}`) : ''].filter(Boolean);
  return parts.length ? <small className="queue-status-note">{parts.join(' · ')}</small> : null;
}

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
  return <span className="task-status" data-status={status}><span className="task-status-dot" aria-hidden="true"/>{labels[status] ? text(...labels[status]) : status}</span>;
}
export function JobActions({ job, onUpdated }: { job: Job; onUpdated: (updated: Job) => void }) {
  const text = useWorkspaceText();
  const { confirm, confirmation } = useConfirmation();
  const [busy, setBusy] = React.useState('');
  const [error, setError] = React.useState('');
  const pending = React.useRef(false);
  const current = React.useRef(job.id); current.current = job.id;
  const run = async (action: string) => {
    if (pending.current) return;
    pending.current = true;
    const id = job.id; setBusy(action); setError('');
    try {
      if (action === 'cancel' && !await confirm({ title: text('取消任务', 'Cancel job'), message: text(`取消任务“${job.name}”？任务将在安全位置停止。已有产物会保留。`, `Cancel “${job.name}”? It will stop at a safe point. Existing outputs are kept.`), confirmLabel: text('取消任务', 'Cancel job'), danger: true })) return;
      if (action === 'force' && !await confirm({ title: text('强制开始', 'Force start'), message: text(`强制开始“${job.name}”？将跳过显存估算立即开始；没有空闲显卡时，最早开始运行的任务会保存状态并暂停。`, `Force-start “${job.name}”? It starts now without the memory estimate. If no GPU is free, the job that started earliest saves its state and pauses.`), confirmLabel: text('强制开始', 'Force start') })) return;
      if (current.current !== id) return;
      const result = await apiClient.post<Job>(`/jobs/${id}/${action}`, {}, { silent: true }); if (current.current === id) onUpdated(result);
    }
    catch (failure) { if (current.current === id) setError(formatApiError(failure)); }
    finally { pending.current = false; setBusy(''); }
  };
  const actions: { key: string; label: string; Icon: typeof Play }[] = [];
  if (job.type !== 'xyz' && ['running', 'queued', 'scheduled'].includes(job.status)) actions.push({ key: 'pause', label: text('暂停', 'Pause'), Icon: Pause });
  if (job.type !== 'xyz' && job.status === 'paused') actions.push({ key: 'resume', label: text('继续', 'Resume'), Icon: Play });
  if (job.forced_at == null && (['queued', 'scheduled'].includes(job.status) || job.type !== 'xyz' && job.status === 'paused')) actions.push({ key: 'force', label: text('强制开始', 'Force start'), Icon: Zap });
  if (job.status === 'running' && job.type === 'train') actions.push({ key: 'save', label: text('保存检查点', 'Save checkpoint'), Icon: Save });
  if (['running', 'queued', 'scheduled', 'paused', 'pausing'].includes(job.status)) actions.push({ key: 'cancel', label: text('取消', 'Cancel'), Icon: XCircle });
  if (['failed', 'cancelled', 'completed'].includes(job.status)) actions.push({ key: 'retry', label: job.type === 'xyz' ? text('重新生成', 'Generate again') : job.type === 'cache' ? text('重新准备', 'Prepare again') : text('重新训练', 'Run again'), Icon: RotateCcw });
  return <div className="task-actions-wrap"><div className="task-actions">{actions.map(({ key, label, Icon }) => <button key={key} type="button" className="ui-btn ui-btn-sm" disabled={!!busy} data-testid={`job-${key}-${job.id}`} onClick={() => void run(key)}>{busy === key ? <Loader2 size={13} className="animate-spin"/> : <Icon size={13}/>}<span>{label}</span></button>)}</div>{error && <p className="task-inline-error" role="alert">{error}</p>}{confirmation}</div>;
}
