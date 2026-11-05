import React from 'react';
import { AlertTriangle, FolderX, Loader2 } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { components } from '../../api/generated';
import Dialog from '../../components/Dialog';
import { formatApiError } from '../../utils/errors';
import { formatBytes } from '../../utils/format';
import { useWorkspaceText } from '../../utils/workspaceText';
import type { GalleryProject } from './projectGallery';
import type { ProjectVersion } from '../../utils/projectVersions';

type Storage = components['schemas']['ProjectStorage'];
type Location = components['schemas']['ProjectStorageLocation'];
const KIND_ORDER = ['project', 'version', 'config', 'datasets', 'cache', 'products', 'records', 'resume', 'logs', 'samples'];

/**
 * The preview lists files and folders removed in the background, and the reason for keeping any location.
 */
export default function ProjectDeleteDialog({ project, version, onClose, onStarted }: { project: Pick<GalleryProject, 'id' | 'name'>; version?: Pick<ProjectVersion, 'id' | 'name'>; onClose: () => void; onStarted: () => void }) {
  const text = useWorkspaceText();
  const [storage, setStorage] = React.useState<Storage | null>(null);
  const [loadError, setLoadError] = React.useState('');
  const [error, setError] = React.useState('');
  const [busy, setBusy] = React.useState(false);
  const target = `/projects/${encodeURIComponent(project.id)}${version ? `/versions/${encodeURIComponent(version.id)}` : ''}`;
  const kindLabels: Record<string, string> = {
    project: text('项目文件夹', 'Project folder'), version: text('版本文件夹', 'Version folder'), cache: text('训练缓存', 'Training cache'), products: text('训练产物', 'Training outputs'),
    config: text('训练配置文件', 'Training configuration file'), datasets: text('训练数据目录', 'Training data folder'),
    records: text('任务记录', 'Job records'), resume: text('恢复点', 'Resume points'), logs: text('日志', 'Logs'), samples: text('采样图', 'Previews'),
  };
  const problemText: Record<string, string> = {
    unavailable: text('无法访问这个位置，可能是移动硬盘没有连接。', 'This location cannot be reached, for example an unplugged drive.'),
    linked: text('路径经过符号链接，可能指向其他文件，不能自动删除。', 'The path goes through a symbolic link that may lead to other files, so it is not removed.'),
    outside: text('不在数据文件夹或已设置的保存位置内，不能自动删除。', 'It is outside the data folder and every configured location, so it is not removed.'),
  };
  const load = React.useCallback(async (signal?: AbortSignal) => {
    setLoadError('');
    try { setStorage(await apiClient.get<Storage>(`${target}/${version ? 'delete-preview' : 'storage'}`, { signal, silent: true })); }
    catch (failure) { if (!signal?.aborted) setLoadError(formatApiError(failure)); }
  }, [target, version]);
  React.useEffect(() => { const controller = new AbortController(); void load(controller.signal); return () => controller.abort(); }, [load]);
  const locations = storage?.locations.filter(location => location.exists || location.problem) ?? [];
  const kept = locations.filter(location => location.problem);
  const remove = async () => {
    setBusy(true); setError('');
    const query = new URLSearchParams([...(version ? [] : [['delete_files', 'true']]), ...kept.map(location => ['skip', location.path])]);
    try { await apiClient.delete(`${target}${query.size ? `?${query}` : ''}`, { silent: true }); onStarted(); }
    catch (failure) { setError(formatApiError(failure)); setBusy(false); void load(); }
  };
  const separator = text('、', ', ');
  const label = (location: Location) => [...location.kinds].sort((a, b) => KIND_ORDER.indexOf(a) - KIND_ORDER.indexOf(b)).map(kind => kindLabels[kind] || kind).join(separator);

  return <Dialog title={version ? text('彻底删除版本','Delete version permanently') : text('彻底删除项目', 'Delete project permanently')} onClose={onClose} closeDisabled={busy}>
    <div className="project-delete">
      <p className="project-delete-warning"><AlertTriangle size={16} aria-hidden="true"/><span>{version ? text(`将永久删除“${project.name} / ${version.name}”的训练记录和下列文件，删除后无法恢复。`,`Training records and the files below for “${project.name} / ${version.name}” are deleted permanently. This cannot be undone.`) : text(`将永久删除“${project.name}”的所有版本、训练记录和下列文件，删除后无法恢复。`, `All versions and training records of “${project.name}” and the files below are deleted permanently. This cannot be undone.`)}</span></p>
      {storage?.deletion?.state === 'failed' && <p className="project-delete-retry" role="status">{text('上次删除没有完成', 'The last deletion did not finish')}{storage.deletion.error ? `：${storage.deletion.error}` : ''}</p>}
      {loadError ? <div className="project-delete-error" role="alert">{loadError}<button type="button" className="ui-btn ui-btn-sm" onClick={() => void load()}>{text('重试', 'Retry')}</button></div>
        : !storage ? <div className="project-delete-loading" aria-busy="true" aria-label={version ? text('读取版本文件','Reading version files') : text('读取项目文件', 'Reading project files')}>{[0, 1, 2].map(index => <span key={index} className="ui-skeleton"/>)}</div>
          : <>
            {storage.blocked && <div className="project-delete-error" role="alert"><span>{storage.blocked.message}</span><button type="button" className="ui-btn ui-btn-sm" onClick={() => void load()}>{text('重新检查', 'Check again')}</button></div>}
            {locations.length ? <>
              <p>{text('以下文件和文件夹将被删除，文件夹包含其中的所有文件：', 'The following files and folders are deleted, including everything inside the folders:')}</p>
              <ul className="project-delete-folders">{locations.map(location => <li key={location.path} data-problem={location.problem || undefined}>
                <div><strong>{label(location)}{location.custom && <span className="project-delete-custom">{text('自定义路径', 'Custom path')}</span>}</strong>
                  {location.problem ? <span className="project-delete-kept">{text('保留', 'Kept')}</span> : <span>{formatBytes(location.bytes)} · {text(`${location.files} 个文件`, `${location.files} files`)}</span>}</div>
                <code>{location.path}</code>
                {location.problem && <p className="project-delete-problem">{problemText[location.problem]}</p>}
              </li>)}</ul>
              <p className="project-delete-total">{text(`共 ${formatBytes(storage.total_bytes)}`, `${formatBytes(storage.total_bytes)} in total`)}{storage.artifacts > 0 && text(`，产物列表中的 ${storage.artifacts} 个产物也会移除。`, `; ${storage.artifacts} entries leave the outputs list.`)}</p>
            </> : <p className="project-delete-empty"><FolderX size={16} aria-hidden="true"/>{version ? text('这个版本的文件已经不在磁盘上，只会删除版本记录。','This version’s files are no longer on disk; only its records are deleted.') : text('这个项目的文件已经不在磁盘上，只会删除项目记录。', 'This project’s files are no longer on disk; only its records are deleted.')}</p>}
          </>}
      {error && <div className="project-delete-error" role="alert">{error}</div>}
      <div className="project-delete-actions"><button type="button" className="ui-btn" disabled={busy} onClick={onClose}>{text('取消', 'Cancel')}</button>
        <button type="button" className="ui-btn ui-btn-primary ui-btn-danger" disabled={busy || !storage || !!storage.blocked} onClick={() => void remove()}>{busy && <Loader2 size={14} className="animate-spin" aria-hidden="true"/>}{kept.length ? text(`保留 ${kept.length} 个位置，删除其余文件`, `Keep ${kept.length} and delete the rest`) : text('永久删除', 'Delete permanently')}</button></div>
    </div>
  </Dialog>;
}
