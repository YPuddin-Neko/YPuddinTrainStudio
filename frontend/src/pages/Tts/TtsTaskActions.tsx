import React from 'react';
import { Link } from 'react-router-dom';
import { Cpu, Loader2, RotateCcw, XCircle, Zap } from 'lucide-react';
import { apiClient } from '../../api/client';
import { ApiError, type Job } from '../../api/types';
import Dialog from '../../components/Dialog';
import { useConfirmation } from '../../components/useConfirmation';
import { useWorkspaceText } from '../../utils/workspaceText';
import TtsGpuPicker from './TtsGpuPicker';
import { ttsAction, ttsJobError, type TtsJobAction } from './ttsJobActions';


type Retry = { jobId: string; key: string; body: Record<string, never>; blocked: boolean; message?: string };
const retryStorageKey = (id: string) => `tts-job-retry:v1:${id}`;
function readRetry(id: string): Retry | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(retryStorageKey(id)) || 'null') as Retry | null;
    return value?.jobId === id && /^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i.test(value.key)
      && value.body && typeof value.body === 'object' && !Array.isArray(value.body) && Object.keys(value.body).length === 0 ? value : null;
  } catch { return null; }
}

export function TtsTaskActions({ job, onUpdated }: { job: Job; onUpdated: (updated: Job) => void }) {
  return <TaskActions key={job.id} job={job} onUpdated={onUpdated}/>;
}
function TaskActions({ job, onUpdated }: { job: Job; onUpdated: (updated: Job) => void }) {
  const text = useWorkspaceText();
  const { confirm, confirmation } = useConfirmation();
  const [busy, setBusy] = React.useState(''), [error, setError] = React.useState('');
  const [retry, setRetry] = React.useState(() => readRetry(job.id));
  const [retryOpen, setRetryOpen] = React.useState(false), [gpuOpen, setGpuOpen] = React.useState(false);
  const [devices, setDevices] = React.useState(job.gpu_devices || []), [gpuValid, setGpuValid] = React.useState(false);
  const pending = React.useRef(false), alive = React.useRef(true), latest = React.useRef(job); latest.current = job;
  React.useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  const permit = (action: TtsJobAction) => ttsAction(latest.current, action, text);
  const run = async (action: 'cancel' | 'force') => {
    if (pending.current || !permit(action).allowed) return;
    pending.current = true; setBusy(action); setError('');
    try {
      const approved = await confirm(action === 'cancel' ? {
        title: text('取消任务', 'Cancel job'), confirmLabel: text('取消任务', 'Cancel job'), danger: true,
        message: text(`取消任务“${job.name}”？进程将停止，已保存的产物会保留，尚未保存的进度会丢失。`, `Cancel “${job.name}”? The process stops; saved outputs are kept and unsaved progress is lost.`),
      } : {
        title: text('强制开始', 'Force start'), confirmLabel: text('强制开始', 'Force start'),
        message: text(`强制开始“${job.name}”？将跳过显存估算并优先调度；可暂停的任务会保存状态后让出显卡，语音任务需结束或取消后让出显卡。`, `Force-start “${job.name}”? Skip the memory estimate and prioritize this job. Pausable jobs save and yield their GPU; speech jobs must finish or be cancelled first.`),
      });
      if (!approved || !alive.current) return;
      if (!permit(action).allowed) { setError(permit(action).reason); return; }
      const updated = await apiClient.post<Job>(`/jobs/${encodeURIComponent(job.id)}/${action}`, {}, { silent: true });
      if (alive.current) onUpdated(updated);
    } catch (failure) { if (alive.current) setError(ttsJobError(failure)); }
    finally { pending.current = false; if (alive.current) setBusy(''); }
  };
  const retryJob = async () => {
    if (pending.current || retry?.blocked) return;
    let attempt = retry || readRetry(job.id);
    if (attempt?.blocked || !attempt && !permit('retry').allowed) return;
    pending.current = true; setBusy('retry'); setError('');
    try {
      if (!attempt) {
        attempt = { jobId: job.id, key: crypto.randomUUID(), body: {}, blocked: false };
        try { sessionStorage.setItem(retryStorageKey(job.id), JSON.stringify(attempt)); }
        catch { attempt = null; throw new Error(text('浏览器无法保留重试请求，请允许网站存储后重试。', 'Enable site storage so the retry request can be retained.')); }
        setRetry(attempt);
      }
      const updated = await apiClient.post<Job>(`/jobs/${encodeURIComponent(attempt.jobId)}/retry`, attempt.body, { headers: { 'Idempotency-Key': attempt.key }, timeout: 120_000, silent: true });
      if (!alive.current) return;
      try { sessionStorage.removeItem(retryStorageKey(job.id)); } catch { /* Replaying a retained key returns the same job. */ }
      setRetry(null); setRetryOpen(false); onUpdated(updated);
    } catch (failure) {
      if (!alive.current) return;
      setError(ttsJobError(failure));
      if (attempt && failure instanceof ApiError && failure.status >= 400 && failure.status < 500 && failure.code !== 'tts.request_pending') {
        const blocked = { ...attempt, blocked: true, message: ttsJobError(failure) }; setRetry(blocked);
        try { sessionStorage.setItem(retryStorageKey(job.id), JSON.stringify(blocked)); } catch { /* The original key remains unchanged. */ }
      }
    } finally { pending.current = false; if (alive.current) setBusy(''); }
  };
  const resetRetry = async () => {
    if (pending.current || !await confirm({ title: text('放弃原重试请求', 'Discard retry request'), message: text('原请求可能已创建任务，请先查看队列。放弃后再次重试会创建新的任务。', 'The original request may have created a job. Check the queue first; another retry after discarding it creates a new job.'), confirmLabel: text('放弃原请求', 'Discard original request'), danger: true }) || !alive.current) return;
    try { sessionStorage.removeItem(retryStorageKey(job.id)); setRetry(null); setError(''); }
    catch { setError(text('无法清除原请求，请检查网站存储权限。', 'Cannot clear the request. Check site storage permissions.')); }
  };
  const saveGpu = async () => {
    if (pending.current || !permit('change_gpu').allowed || !gpuValid) return;
    pending.current = true; setBusy('change_gpu'); setError('');
    try {
      const updated = await apiClient.patch<Job>(`/jobs/${encodeURIComponent(job.id)}`, { gpu_devices: devices }, { silent: true });
      if (alive.current) { setGpuOpen(false); onUpdated(updated); }
    } catch (failure) { if (alive.current) setError(ttsJobError(failure)); }
    finally { pending.current = false; if (alive.current) setBusy(''); }
  };
  const waiting = ['queued', 'scheduled'].includes(job.status), terminal = ['completed', 'failed', 'cancelled'].includes(job.status);
  const buttons = [
    { key: 'cancel' as const, label: text('取消', 'Cancel'), Icon: XCircle, show: waiting || ['running', 'cancelling'].includes(job.status) || permit('cancel').allowed, onClick: () => void run('cancel') },
    { key: 'force' as const, label: text('强制开始', 'Force start'), Icon: Zap, show: waiting || permit('force').allowed, onClick: () => void run('force') },
    { key: 'retry' as const, label: retry ? text('确认重试结果', 'Confirm retry result') : job.type === 'tts_sample' ? text('重新生成', 'Generate again') : text('从头重新训练', 'Train again from start'), Icon: RotateCcw, show: terminal || !!retry || permit('retry').allowed, onClick: () => { setError(retry?.message || ''); setRetryOpen(true); } },
    { key: 'change_gpu' as const, label: text('修改显卡', 'Change GPU'), Icon: Cpu, show: waiting || terminal || permit('change_gpu').allowed, onClick: () => { setDevices(job.gpu_devices || []); setError(''); setGpuOpen(true); } },
  ].filter(button => button.show);
  return <div className="task-actions-wrap tts-task-actions"><div className="task-actions">{buttons.map(({ key, label, Icon, onClick }) => {
    const permission = permit(key), replay = key === 'retry' && !!retry;
    return <button key={key} type="button" className="ui-btn ui-btn-sm" disabled={!!busy || !replay && !permission.allowed} title={!replay ? permission.reason || undefined : undefined} aria-describedby={!replay && !permission.allowed ? `tts-action-${job.id}-${key}` : undefined} data-testid={`job-${key}-${job.id}`} onClick={onClick}>{busy === key ? <Loader2 size={13} className="animate-spin"/> : <Icon size={13}/>}<span>{label}</span></button>;
  })}</div>{buttons.map(({ key, label }) => !permit(key).allowed && !(key === 'retry' && retry) ? <small key={key} id={`tts-action-${job.id}-${key}`} className="queue-status-note">{label}：{permit(key).reason}</small> : null)}
    {error && !gpuOpen && !retryOpen && <p className="task-inline-error" role="alert">{error}</p>}
    {retryOpen && <Dialog title={job.type === 'tts_sample' ? text('重新生成试听', 'Generate preview again') : text('从头重新训练', 'Train again from start')} onClose={() => setRetryOpen(false)} closeDisabled={!!busy}>
      <p className="workspace-confirm-message">{retry ? text('继续确认会使用原请求，不会重复创建任务。', 'Confirming reuses the original request without creating another job.') : job.type === 'tts_sample' ? text('使用原任务的参数与来源权重重新生成试听，已有音频保留。', 'Generate another preview with the original parameters and source weights. Existing audio is kept.') : text('使用原任务的参数和数据从头训练，不从检查点恢复。已有权重与记录保留。', 'Train from the beginning with the original parameters and data. Existing weights and records are kept; this does not resume a checkpoint.')}</p>
      {retry?.blocked && <p className="task-inline-error" role="status">{text('原请求不能继续，请核对任务后再决定是否放弃该请求。', 'This request cannot continue. Check the job before deciding whether to discard it.')}</p>}
      {!retry && !permit('retry').allowed && <p className="task-inline-error" role="status">{permit('retry').reason}</p>}
      {error && <p role="alert" className="task-inline-error">{error}</p>}
      <div className="workspace-confirm-actions"><button type="button" className="ui-btn" disabled={!!busy} onClick={() => setRetryOpen(false)}>{text('关闭', 'Close')}</button>{retry && <button type="button" className="ui-btn" disabled={!!busy} onClick={() => void resetRetry()}>{text('放弃原请求', 'Discard original request')}</button>}<button type="button" className="ui-btn ui-btn-primary" disabled={!!busy || retry?.blocked || !retry && !permit('retry').allowed} onClick={() => void retryJob()}>{busy ? text('正在确认…', 'Confirming…') : retry ? text('继续确认原请求', 'Confirm original request') : text('确认重试', 'Confirm retry')}</button></div>
    </Dialog>}
    {gpuOpen && <Dialog title={text('修改运行显卡', 'Change run GPU')} onClose={() => setGpuOpen(false)} closeDisabled={!!busy}>
      <TtsGpuPicker value={devices} onChange={setDevices} onValidityChange={setGpuValid} disabled={!!busy || !permit('change_gpu').allowed} training/>
      {!permit('change_gpu').allowed && <p className="task-inline-error" role="status">{permit('change_gpu').reason}</p>}
      {error && <p role="alert" className="task-inline-error">{error}</p>}
      <div className="workspace-confirm-actions"><button type="button" className="ui-btn" disabled={!!busy} onClick={() => setGpuOpen(false)}>{text('取消', 'Cancel')}</button><button type="button" className="ui-btn ui-btn-primary" disabled={!!busy || !gpuValid || !permit('change_gpu').allowed} onClick={() => void saveGpu()}>{text('保存显卡', 'Save GPU')}</button></div>
    </Dialog>}{confirmation}
  </div>;
}

export function TtsJobLineage({ job }: { job: Job }) {
  const text = useWorkspaceText();
  const [available, setAvailable] = React.useState<Record<string, string>>({});
  React.useEffect(() => {
    const abort = new AbortController(); setAvailable({});
    for (const id of new Set([job.source_job_id, job.retry_of_job_id].filter((value): value is string => !!value))) {
      void apiClient.get<Job>(`/jobs/${encodeURIComponent(id)}`, { signal: abort.signal, silent: true }).then(result => {
        if (!abort.signal.aborted && result.id === id) setAvailable(previous => ({ ...previous, [id]: result.name || id }));
      }).catch(() => { /* Missing source records are described by the result view. */ });
    }
    return () => abort.abort();
  }, [job.id, job.source_job_id, job.retry_of_job_id]);
  return <>{[[job.source_job_id, text('来源训练', 'Source training')], [job.retry_of_job_id, text('重试来源', 'Retried from')]].map(([id, label]) => id && available[id] ? <div key={`${label}:${id}`}><dt>{label}</dt><dd><Link to={`/jobs/${encodeURIComponent(id)}`}>{available[id]}</Link></dd></div> : null)}</>;
}
