import React from 'react';
import { AlertTriangle, FolderX, Loader2 } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { Job } from '../../api/types';
import type { components } from '../../api/generated';
import Dialog from '../../components/Dialog';
import { formatApiError } from '../../utils/errors';
import { formatBytes } from '../../utils/format';
import { useWorkspaceText } from '../../utils/workspaceText';

type Storage = components['schemas']['JobStorage'];
const KIND_ORDER = ['products', 'records', 'resume', 'logs', 'samples'];

/**
 * Deleting from the archive removes the job for good: every folder listed here goes, with its products,
 * resume points, logs and previews. The dialog lists them with their sizes before anything is removed.
 */
export default function JobDeleteDialog({ job, onClose, onDeleted }: { job: Job; onClose: () => void; onDeleted: () => void }) {
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
