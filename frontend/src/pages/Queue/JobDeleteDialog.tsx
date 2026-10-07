import React from 'react';
import { AlertTriangle, FolderX, Loader2 } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { Job } from '../../api/types';
import type { components } from '../../api/generated';
import Dialog from '../../components/Dialog';
import { formatApiError } from '../../utils/errors';
import { formatBytes } from '../../utils/format';
import { useWorkspaceText } from '../../utils/workspaceText';
import { isTtsJob } from '../../api/tts';
import { ttsAction, ttsJobError } from '../Tts/ttsJobActions';

type Storage = components['schemas']['JobStorage'];
const KIND_ORDER = ['products', 'records', 'resume', 'logs', 'samples'];

export default function JobDeleteDialog({ job, onClose, onDeleted }: { job: Job; onClose: () => void; onDeleted: () => void }) {
  return isTtsJob(job) ? <TtsJobDeleteDialog key={job.id} job={job} onClose={onClose} onDeleted={onDeleted}/> : <ImageJobDeleteDialog job={job} onClose={onClose} onDeleted={onDeleted}/>;
}
function ImageJobDeleteDialog({ job, onClose, onDeleted }: { job: Job; onClose: () => void; onDeleted: () => void }) {
  const text = useWorkspaceText();
  const [storage, setStorage] = React.useState<Storage | null>(null);
  const [loadError, setLoadError] = React.useState('');
  const [error, setError] = React.useState('');
  const [busy, setBusy] = React.useState(false);
  const kindLabels: Record<string, string> = {
    products: text('训练产物', 'Training outputs'), records: text('任务记录', 'Job records'), resume: text('恢复点', 'Resume points'),
    logs: text('日志', 'Logs'), samples: text('采样图', 'Previews'),
  };
  const load = React.useCallback(async (signal?: AbortSignal) => {
    setLoadError('');
    try { setStorage(await apiClient.get<Storage>(`/jobs/${encodeURIComponent(job.id)}/storage`, { signal, silent: true })); }
    catch (failure) { if (!signal?.aborted) setLoadError(formatApiError(failure)); }
  }, [job.id]);
  React.useEffect(() => { const controller = new AbortController(); void load(controller.signal); return () => controller.abort(); }, [load]);
  const remove = async () => {
    setBusy(true); setError('');
    try { await apiClient.delete(`/jobs/${encodeURIComponent(job.id)}`, { params: { delete_files: true }, silent: true }); onDeleted(); }
    catch (failure) { setError(formatApiError(failure)); setBusy(false); }
  };
  const folders = storage?.folders.filter(folder => folder.exists) ?? [];
  const separator = text('、', ', ');

  return <Dialog title={text('彻底删除任务', 'Delete job permanently')} onClose={onClose} closeDisabled={busy}>
    <div className="job-delete">
      <p className="job-delete-warning"><AlertTriangle size={16} aria-hidden="true"/><span>{text(`将永久删除“${job.name}”的任务记录和它的所有文件，删除后无法恢复。`, `“${job.name}” and all of its files will be deleted permanently. This cannot be undone.`)}</span></p>
      {loadError ? <div className="task-error" role="alert">{loadError}<button type="button" className="ui-btn ui-btn-sm" onClick={() => void load()}>{text('重试', 'Retry')}</button></div>
        : !storage ? <div className="job-delete-loading" aria-busy="true" aria-label={text('读取任务文件', 'Reading job files')}>{[0, 1, 2].map(index => <span key={index} className="ui-skeleton"/>)}</div>
          : folders.length ? <>
            <p>{text('以下文件夹会连同其中的所有文件一起删除：', 'These folders are deleted with everything in them:')}</p>
            <ul className="job-delete-folders">{folders.map(folder => <li key={folder.path}>
              <div><strong>{[...folder.kinds].sort((a, b) => KIND_ORDER.indexOf(a) - KIND_ORDER.indexOf(b)).map(kind => kindLabels[kind] || kind).join(separator)}</strong>
                <span>{formatBytes(folder.bytes)} · {text(`${folder.files} 个文件`, `${folder.files} files`)}</span></div>
              <code>{folder.path}</code>
            </li>)}</ul>
            <p className="job-delete-total">{text(`共 ${formatBytes(storage.total_bytes)}`, `${formatBytes(storage.total_bytes)} in total`)}{storage.artifacts > 0 && text(`，产物列表中的 ${storage.artifacts} 个产物也会移除。`, `; ${storage.artifacts} entries leave the outputs list.`)}</p>
          </> : <p className="job-delete-empty"><FolderX size={16} aria-hidden="true"/>{text('这个任务的文件夹已经不在磁盘上，只会删除任务记录。', 'This job’s folders are no longer on disk; only its record is deleted.')}</p>}
      {error && <div className="task-error" role="alert">{error}</div>}
      <div className="job-delete-actions"><button type="button" className="ui-btn" disabled={busy} onClick={onClose}>{text('取消', 'Cancel')}</button>
        <button type="button" className="ui-btn ui-btn-primary ui-btn-danger" disabled={busy || !storage} onClick={() => void remove()}>{busy && <Loader2 size={14} className="animate-spin" aria-hidden="true"/>}{text('永久删除', 'Delete permanently')}</button></div>
    </div>
  </Dialog>;
}

function TtsJobDeleteDialog({ job, onClose, onDeleted }: { job: Job; onClose: () => void; onDeleted: () => void }) {
  const text = useWorkspaceText();
  const [files, setFiles] = React.useState(false), [busy, setBusy] = React.useState(false);
  const [storage, setStorage] = React.useState<Storage | null>(null), [loadError, setLoadError] = React.useState(''), [error, setError] = React.useState('');
  const pending = React.useRef(false), alive = React.useRef(true);
  const permission = ttsAction(job, 'delete', text);
  React.useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  const load = React.useCallback(async (signal?: AbortSignal) => {
    setLoadError('');
    try { const value = await apiClient.get<Storage>(`/jobs/${encodeURIComponent(job.id)}/storage`, { signal, silent: true }); if (alive.current && !signal?.aborted) setStorage(value); }
    catch (failure) { if (alive.current && !signal?.aborted) setLoadError(ttsJobError(failure)); }
  }, [job.id]);
  React.useEffect(() => { const abort = new AbortController(); void load(abort.signal); return () => abort.abort(); }, [load]);
  const remove = async () => {
    if (pending.current || !permission.allowed || files && !storage) return;
    pending.current = true; setBusy(true); setError('');
    try { await apiClient.delete(`/jobs/${encodeURIComponent(job.id)}`, { params: { delete_files: files }, silent: true }); if (alive.current) onDeleted(); }
    catch (failure) { if (alive.current) setError(ttsJobError(failure)); }
    finally { pending.current = false; if (alive.current) setBusy(false); }
  };
  return <Dialog title={text('删除语音任务', 'Delete speech job')} onClose={onClose} closeDisabled={busy}><div className="job-delete">
    <p className="job-delete-warning"><AlertTriangle size={16}/><span>{text(`永久删除“${job.name}”的任务记录？删除后无法从训练历史中恢复。`, `Permanently delete the record for “${job.name}”? It cannot be restored to training history.`)}</span></p>
    <fieldset className="tts-job-delete-options" disabled={busy}><legend>{text('文件处理方式', 'Job files')}</legend><label><input type="radio" name={`tts-delete-${job.id}`} checked={!files} onChange={() => setFiles(false)}/>{text('保留文件，仅删除任务记录', 'Keep files and delete only the job record')}</label><label><input type="radio" name={`tts-delete-${job.id}`} checked={files} onChange={() => setFiles(true)}/>{text('同时删除任务文件', 'Also delete job files')}</label></fieldset>
    <p>{files ? text('下列文件夹及其中的权重、音频和日志会一起删除。', 'The folders below and their weights, audio and logs will be deleted.') : text('权重、音频和日志保留在磁盘原位置。', 'Weights, audio and logs remain in their original folders.')}{job.type === 'tts_train' && text(' 已生成的子任务试听音频不会随来源训练一起删除。', ' Existing audio from preview jobs is kept when deleting its source training job.')}</p>
    {files && (loadError ? <div role="alert" className="task-error">{loadError}<button type="button" className="ui-btn ui-btn-sm" onClick={() => void load()}>{text('重试', 'Retry')}</button></div> : !storage ? <p role="status">{text('正在读取任务文件…', 'Reading job files…')}</p> : <><ul className="job-delete-folders">{storage.folders.filter(folder => folder.exists).map(folder => <li key={folder.path}><div><strong>{text('任务文件', 'Job files')}</strong><span>{formatBytes(folder.bytes)} · {text(`${folder.files} 个文件`, `${folder.files} files`)}</span></div><code>{folder.path}</code></li>)}</ul><p className="job-delete-total">{text(`共 ${formatBytes(storage.total_bytes)}`, `${formatBytes(storage.total_bytes)} in total`)}</p></>)}
    {!permission.allowed && <p role="status" className="task-inline-error">{permission.reason}</p>}{error && <p role="alert" className="task-error">{error}</p>}
    <div className="job-delete-actions"><button type="button" className="ui-btn" disabled={busy} onClick={onClose}>{text('取消', 'Cancel')}</button><button type="button" className="ui-btn ui-btn-danger" disabled={busy || !permission.allowed || files && !storage} onClick={() => void remove()}>{busy ? text('正在删除…', 'Deleting…') : files ? text('删除记录与文件', 'Delete record and files') : text('仅删除记录', 'Delete record only')}</button></div>
  </div></Dialog>;
}
