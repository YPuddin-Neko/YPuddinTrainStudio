import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowUpCircle, Check, Download, FolderOpen, PackagePlus, RefreshCw, RotateCcw, Trash2, Upload, X, type LucideIcon } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { components } from '../../api/generated';
import { EVENT_TYPES } from '../../events/eventTypes';
import { useEventStream, useEventStreamStatus } from '../../events/useEventStream';
import { filesFromSelection } from '../../utils/datasetFiles';
import { canChooseFilesAgain, formatUploadBytes, uploadErrorMessage } from '../../utils/datasetUploadCopy';
import { datasetUploads, useDatasetUploads, type UploadEntry } from '../../utils/datasetUploads';
import { formatApiError } from '../../utils/errors';
import { formatBytes, formatEta } from '../../utils/format';
import { useWorkspaceText } from '../../utils/workspaceText';
import ProgressBar from '../ProgressBar';
import './task-center.css';

type ServerTask = components['schemas']['BackgroundTask'] & { session_id?: string };
type TaskState = 'running' | 'completed' | 'failed' | 'cancelled' | 'interrupted';
type Text = (zh: string, en: string) => string;
interface TaskItem {
  id: string;
  kind: string;
  subject: string;
  state: TaskState;
  done: number | null;
  total: number | null;
  unit: string | null;
  rate: number | null;
  phase: string;
  error: string;
  link: string | null;
  /** Local milliseconds; service times are moved onto this clock. */
  started: number;
  finished: number | null;
  cancellable: boolean;
  upload?: UploadEntry;
}

// Work shorter than this never shows, so quick saves and refreshes do not flash the ball.
const SHOW_AFTER_MS = 1000;
// Completed work stays this long; failures stay until they are dismissed.
const KEEP_DONE_MS = 6000;
const POLL_MS = 3000;

const kindIcons: Record<string, LucideIcon> = {
  dataset_upload: Upload, model_download: Download, project_delete: Trash2, dataset_refresh: RefreshCw,
  environment: PackagePlus, trainer_update: ArrowUpCircle,
};
const importPhases: Record<string, [string, string]> = {
  receiving: ['等待导入', 'Waiting to import'],
  extracting: ['正在解压文件', 'Extracting files'],
  validating: ['正在校验文件与标签', 'Checking files and captions'],
  copying: ['正在同步到当前版本', 'Copying into this version'],
  registering: ['正在登记图片目录', 'Registering image folders'],
};

function kindTitle(kind: string, text: Text) {
  switch (kind) {
    case 'dataset_upload': return text('上传数据集', 'Dataset upload');
    case 'model_download': return text('下载模型', 'Model download');
    case 'project_delete': return text('删除项目', 'Project deletion');
    case 'dataset_refresh': return text('刷新数据集索引', 'Dataset index refresh');
    case 'environment': return text('安装运行环境', 'Environment install');
    case 'trainer_update': return text('更新训练器', 'Trainer update');
    default: return kind;
  }
}

function uploadItem(entry: UploadEntry, text: Text): TaskItem {
  const running = ['preparing', 'uploading', 'importing', 'waiting'].includes(entry.phase);
  const server = entry.phase === 'importing' ? entry.server : null;
  const measured = server && server.phase !== 'completed' && server.phase !== 'failed' && server.phase !== 'receiving';
  const bytes = measured && (server.bytes_total ?? 0) > 0;
  const phase = entry.phase === 'preparing' ? text('正在准备上传', 'Preparing the upload')
    : entry.phase === 'uploading' ? entry.remote ? text('正在另一个标签页上传', 'Uploading in another tab') : text(`正在上传 · ${entry.filesDone} / ${entry.filesTotal} 个文件`, `Uploading · ${entry.filesDone} / ${entry.filesTotal} files`)
      : entry.phase === 'waiting' ? entry.waiting === 'version.indexing' ? text('等待数据集索引完成后导入', 'Waiting for dataset indexing to finish') : text('等待当前版本的其他操作完成后导入', 'Waiting for another operation on this version')
        : entry.phase === 'importing' ? server && importPhases[server.phase] ? text(...importPhases[server.phase]) : text('正在导入', 'Importing')
          : entry.phase === 'interrupted' ? entry.needsFiles ? text('上传未完成，选择相同的文件可继续', 'Not finished; choose the same files to continue') : text('文件已上传，尚未导入', 'Uploaded, not imported yet')
            : '';
  return {
    id: `upload:${entry.id}`, kind: 'dataset_upload', subject: entry.subject,
    state: running ? 'running' : entry.phase === 'completed' ? 'completed' : entry.phase === 'interrupted' ? 'interrupted' : 'failed',
    done: measured ? bytes ? server.bytes_done : server.files_done : entry.phase === 'importing' || entry.phase === 'waiting' ? null : entry.bytesDone,
    total: measured ? bytes ? server.bytes_total ?? null : server.files_total ?? null : entry.phase === 'importing' || entry.phase === 'waiting' ? null : entry.bytesTotal,
    unit: measured && !bytes ? 'files' : 'bytes', rate: entry.phase === 'uploading' ? entry.rate : measured ? server.bytes_per_second ?? null : null,
    phase, error: entry.phase === 'failed' || entry.error?.problem === 'mismatch' ? uploadErrorMessage(entry, text) : '',
    link: entry.link, started: entry.startedAt, finished: entry.finishedAt ?? null,
    cancellable: !entry.remote && ['preparing', 'uploading', 'waiting'].includes(entry.phase), upload: entry,
  };
}

function serverItem(task: ServerTask, skew: number, text: Text): TaskItem {
  const detail = task.detail || '';
  const phase = task.state !== 'running' ? ''
    : task.kind === 'dataset_upload' && importPhases[detail] ? text(...importPhases[detail])
      : detail === 'queued' ? text('排队中', 'Queued')
        : detail === 'verifying' ? text('正在校验', 'Verifying')
          : task.kind === 'model_download' && !detail ? text('正在下载', 'Downloading')
            : detail;
  return {
    id: task.id, kind: task.kind, subject: task.subject || '', state: task.state, done: task.done ?? null, total: task.total ?? null,
    unit: task.unit ?? null, rate: task.rate ?? null, phase, error: task.state === 'failed' ? task.error || text('未返回详细原因。', 'No details were returned.') : '',
    link: task.link ?? null, started: task.started_at * 1000 + skew, finished: task.finished_at != null ? task.finished_at * 1000 + skew : null,
    cancellable: task.cancellable && task.state === 'running',
  };
}

/** Uploads count in binary units like the import form; downloads in the units of the model pages. */
const size = (value: number, binary: boolean) => binary ? formatUploadBytes(value) : formatBytes(Math.round(value));

function amounts(done: number, total: number, unit: string | null, text: Text, binary: boolean) {
  switch (unit) {
    case 'bytes': return `${size(done, binary)} / ${size(total, binary)}`;
    case 'files': return text(`${done} / ${total} 个文件`, `${done} / ${total} files`);
    case 'images': return text(`${done} / ${total} 张图片`, `${done} / ${total} images`);
    case 'items': return text(`${done} / ${total} 项`, `${done} / ${total} items`);
    default: return `${done} / ${total}`;
  }
}

function useNow(items: TaskItem[]) {
  const [now, setNow] = React.useState(() => Date.now());
  const active = items.some(item => item.state === 'running'
    || (item.state === 'completed' || item.state === 'cancelled') && item.finished != null
      && item.finished - item.started >= SHOW_AFTER_MS && now - item.finished < KEEP_DONE_MS);
  const timeline = JSON.stringify(items.map(item => [item.id, item.state, item.started, item.finished]));
  React.useEffect(() => {
    setNow(Date.now());
    if (!active) return;
    const timer = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(timer);
  }, [active, timeline]);
  return now;
}

/** Background work outside the job queue: a ball under the hardware status, and the list it opens. */
export default function TaskCenter() {
  const text = useWorkspaceText();
  const panelId = React.useId();
  const [tasks, setTasks] = React.useState<ServerTask[]>([]);
  const [skew, setSkew] = React.useState(0);
  const [open, setOpen] = React.useState(false);
  const [actionError, setActionError] = React.useState('');
  const ball = React.useRef<HTMLButtonElement>(null);
  const panel = React.useRef<HTMLDivElement>(null);
  const folderInput = React.useRef<HTMLInputElement>(null);
  const fileInput = React.useRef<HTMLInputElement>(null);
  const resuming = React.useRef<string | null>(null);
  const uploads = useDatasetUploads();

  const load = React.useCallback(async () => {
    try {
      const body = await apiClient.get<{ tasks: ServerTask[]; now: number }>('/background-tasks', { silent: true });
      setSkew(Date.now() - body.now * 1000);
      setTasks(body.tasks);
    } catch { /* The list keeps what it knew; the next event or poll brings it up to date. */ }
  }, []);
  React.useEffect(() => { void load(); }, [load]);
  useEventStream<ServerTask | { id: string; dismissed: true }>(EVENT_TYPES.BACKGROUND_CHANGED, data => {
    setTasks(current => 'dismissed' in data ? current.filter(task => task.id !== data.id)
      : current.some(task => task.id === data.id) ? current.map(task => task.id === data.id ? data : task) : [data, ...current]);
  });
  // Events missed while disconnected are read again once the stream is back.
  const connection = useEventStreamStatus();
  const previousConnection = React.useRef(connection);
  React.useEffect(() => {
    if (connection === 'connected' && previousConnection.current === 'disconnected') void load();
    previousConnection.current = connection;
  }, [connection, load]);
  const serverRunning = tasks.some(task => task.state === 'running');
  React.useEffect(() => {
    if (!serverRunning) return;
    const timer = window.setInterval(() => void load(), POLL_MS);
    return () => window.clearInterval(timer);
  }, [serverRunning, load]);

  const sessions = new Set(uploads.flatMap(entry => entry.sessionId ? [entry.sessionId] : []));
  const all = [
    ...uploads.map(entry => uploadItem(entry, text)),
    // The page that uploads shows its own import; other browsers see the service's.
    ...tasks.filter(task => !task.session_id || !sessions.has(task.session_id)).map(task => serverItem(task, skew, text)),
  ];
  const now = useNow(all);
  const shown = all.filter(item => item.state === 'failed' || item.state === 'interrupted'
    || item.state === 'running' && now - item.started >= SHOW_AFTER_MS
    || item.finished != null && item.finished - item.started >= SHOW_AFTER_MS && now - item.finished < KEEP_DONE_MS)
    .sort((a, b) => Number(b.state === 'running') - Number(a.state === 'running') || b.started - a.started);
  const running = shown.filter(item => item.state === 'running');
  const attention = shown.filter(item => item.state === 'failed' || item.state === 'interrupted');
  const fractions = running.flatMap(item => item.total && item.total > 0 && item.done != null ? [Math.min(1, item.done / item.total)] : []);
  const progress = fractions.length ? fractions.reduce((sum, value) => sum + value, 0) / fractions.length : null;
  const state = running.length ? 'running' : attention.length ? 'failed' : 'done';
  const visible = shown.length > 0 || open;

  const close = React.useCallback((focus = true) => {
    setOpen(false);
    setActionError('');
    if (focus) ball.current?.focus({ preventScroll: true });
  }, []);
  React.useEffect(() => {
    if (!open) return;
    panel.current?.focus({ preventScroll: true });
    const outside = (event: PointerEvent) => {
      const target = event.target as Node;
      if (!panel.current?.contains(target) && !ball.current?.contains(target)) close(false);
    };
    document.addEventListener('pointerdown', outside);
    return () => document.removeEventListener('pointerdown', outside);
  }, [open, close]);

  const act = async (work: () => Promise<unknown>) => {
    setActionError('');
    try { await work(); await load(); } catch (error) { setActionError(formatApiError(error)); }
  };
  const cancel = (item: TaskItem) => item.upload
    ? void datasetUploads.cancel(item.upload.id)
    : void act(() => apiClient.post(`/background-tasks/${encodeURIComponent(item.id)}/cancel`, {}, { silent: true }));
  const dismiss = (item: TaskItem) => item.upload
    ? datasetUploads.dismiss(item.upload.id)
    : void act(() => apiClient.delete(`/background-tasks/${encodeURIComponent(item.id)}`, { silent: true }));
  const chooseAgain = (entry: UploadEntry, folder: boolean) => {
    resuming.current = entry.id;
    (folder ? folderInput : fileInput).current?.click();
  };
  const resumeWith = (files: File[]) => {
    if (folderInput.current) folderInput.current.value = '';
    if (fileInput.current) fileInput.current.value = '';
    const id = resuming.current;
    resuming.current = null;
    if (!id || !files.length) return;
    try { datasetUploads.resume(id, filesFromSelection(files)); setActionError(''); }
    catch (error) { setActionError(formatApiError(error)); }
  };

  const label = running.length ? text(`${running.length} 项后台任务正在进行`, `${running.length} background tasks running`)
    : attention.length ? text(`${attention.length} 项后台任务需要处理`, `${attention.length} background tasks need attention`)
      : text('后台任务已完成', 'Background tasks finished');
  const percent = progress === null ? null : Math.round(progress * 100);
  const row = (item: TaskItem) => {
    const Icon = kindIcons[item.kind] || RefreshCw;
    const fraction = item.total && item.total > 0 && item.done != null ? Math.min(1, item.done / item.total) : null;
    const remaining = item.state === 'running' && item.rate && item.rate > 0 && item.total != null && item.done != null ? (item.total - item.done) / item.rate : null;
    const meta = [
      item.total != null && item.done != null ? amounts(item.done, item.total, item.unit, text, item.kind === 'dataset_upload') : '',
      item.state === 'running' && item.rate && item.unit === 'bytes' ? `${size(item.rate, item.kind === 'dataset_upload')}/s` : '',
      remaining != null ? text(`剩余 ${formatEta(remaining)}`, `${formatEta(remaining)} left`) : '',
    ].filter(Boolean).join(' · ');
    const status = item.state === 'completed' ? text('已完成', 'Done') : item.state === 'cancelled' ? text('已取消', 'Cancelled')
      : item.state === 'failed' ? text('失败', 'Failed') : item.state === 'interrupted' ? text('已中断', 'Interrupted')
        : fraction != null ? `${Math.floor(fraction * 100)}%` : '';
    const entry = item.upload;
    return <li key={item.id} className="task-center-item" data-state={item.state}>
      <Icon size={16} aria-hidden="true"/>
      <div className="task-center-item-main">
        <div className="task-center-item-title"><strong>{kindTitle(item.kind, text)}</strong>{status && <span>{status}</span>}</div>
        {item.subject && <p className="task-center-subject" title={item.subject}>{item.subject}</p>}
        {item.state === 'running' && <ProgressBar className="task-center-progress" label={`${kindTitle(item.kind, text)} · ${item.subject}`} value={fraction ?? undefined} max={1}/>}
        {item.phase && <p className="task-center-meta">{item.phase}</p>}
        {meta && item.state !== 'completed' && <p className="task-center-meta">{meta}</p>}
        {item.error && <p className="task-center-error" role="alert">{item.error}</p>}
        <div className="task-center-actions">
          {entry && item.state === 'interrupted' && entry.needsFiles && <>
            <button type="button" className="ui-btn ui-btn-sm ui-btn-primary" onClick={() => chooseAgain(entry, true)}><FolderOpen size={13}/>{text('选择文件夹继续', 'Choose folder to continue')}</button>
            <button type="button" className="ui-btn ui-btn-sm" onClick={() => chooseAgain(entry, false)}>{text('选择文件继续', 'Choose files to continue')}</button>
          </>}
          {entry && (item.state === 'interrupted' && !entry.needsFiles || item.state === 'failed' && entry.canRetry) && <button type="button" className="ui-btn ui-btn-sm" onClick={() => datasetUploads.retry(entry.id)}><RotateCcw size={13}/>{item.state === 'interrupted' ? text('导入', 'Import') : entry.error?.problem === 'expired' ? text('重新上传', 'Upload again') : text('重试', 'Retry')}</button>}
          {entry && item.state === 'failed' && canChooseFilesAgain(entry) && <button type="button" className="ui-btn ui-btn-sm" onClick={() => chooseAgain(entry, true)}>{text('重新选择文件夹', 'Choose folder again')}</button>}
          {item.cancellable && <button type="button" className="ui-btn ui-btn-sm" onClick={() => cancel(item)}>{text('取消', 'Cancel')}</button>}
          {item.link && <Link className="ui-btn ui-btn-sm ui-btn-quiet" to={item.link} onClick={() => close(false)}>{text('查看', 'View')}</Link>}
          {(item.state === 'failed' || item.state === 'interrupted') && !entry?.remote && <button type="button" className="ui-btn ui-btn-sm ui-btn-quiet" onClick={() => dismiss(item)}>{entry?.sessionId ? text('放弃', 'Discard') : text('移除', 'Remove')}</button>}
        </div>
      </div>
    </li>;
  };

  return <div className="task-center" data-testid="task-center">
    {visible && <button ref={ball} type="button" className="task-center-ball" data-state={state} data-progress={state === 'running' && percent === null ? 'indeterminate' : undefined}
      aria-label={label} title={label} aria-expanded={open} aria-controls={open ? panelId : undefined} onClick={() => open ? close() : setOpen(true)}>
      <svg className="task-center-ring" viewBox="0 0 36 36" aria-hidden="true">
        <circle className="task-center-track" cx="18" cy="18" r="15.915"/>
        <circle className="task-center-value" cx="18" cy="18" r="15.915" pathLength={100} strokeDasharray={`${state !== 'running' ? 100 : percent ?? 25} 100`}/>
      </svg>
      <span className="task-center-count">{state === 'running' ? running.length : state === 'failed' ? '!' : <Check size={12} strokeWidth={3}/>}</span>
      {state === 'running' && attention.length > 0 && <span className="task-center-alert" aria-hidden="true"/>}
    </button>}
    {open && <div id={panelId} ref={panel} className="task-center-panel" role="dialog" aria-label={text('后台任务', 'Background tasks')} tabIndex={-1}
      onKeyDown={event => { if (event.key === 'Escape') { event.stopPropagation(); close(); } }}>
      <header><h2>{text('后台任务', 'Background tasks')}</h2><button type="button" className="ui-btn ui-btn-quiet ui-btn-sm ui-btn-icon" aria-label={text('关闭', 'Close')} onClick={() => close()}><X size={15}/></button></header>
      {actionError && <p className="task-center-error task-center-action-error" role="alert">{actionError}</p>}
      {shown.length ? <ul className="task-center-list">{shown.map(row)}</ul> : <p className="task-center-empty">{text('没有后台任务', 'No background tasks')}</p>}
    </div>}
    {open && <>
      <input ref={fileInput} hidden type="file" multiple accept="image/*,.txt,.json,.mask,.zip" aria-label={text('选择文件继续后台上传', 'Choose files to continue the background upload')} onChange={event => resumeWith(Array.from(event.target.files || []))}/>
      <input ref={folderInput} hidden type="file" multiple {...{ webkitdirectory: '' }} aria-label={text('选择文件夹继续后台上传', 'Choose a folder to continue the background upload')} onChange={event => resumeWith(Array.from(event.target.files || []))}/>
    </>}
  </div>;
}
