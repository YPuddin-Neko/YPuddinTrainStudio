import React from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { Link, useNavigate } from 'react-router-dom';
import { GitBranch, Plus, Settings2, ArrowLeft, GitCompare, FolderOpen, Loader2, Copy, AlertCircle } from 'lucide-react';
import { apiClient } from '../../api/client';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import { activateProjectVersion, configDifferences, projectUrl, versionConfigUrl, type ProjectVersion, type VersionedProject } from '../../utils/projectVersions';
import { type WorkspaceStep, ProjectWorkflow } from '../ProjectWorkflow';
import Dialog from '../Dialog';

interface Props {
  project: VersionedProject; versionId?: string; versions: ProjectVersion[]; current?: ProjectVersion;
  active: WorkspaceStep; refresh: () => Promise<unknown>; beforeAction?: () => Promise<void>;
  status?: React.ReactNode; error?: unknown;
}
export default function ProjectWorkspaceHeader({ project, versionId, versions, current, active, refresh, beforeAction, status, error: loadError }: Props) {
  const text = useWorkspaceText();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [dialog, setDialog] = React.useState<'create' | 'edit' | 'compare' | 'paths' | null>(null);
  const [name, setName] = React.useState('');
  const [note, setNote] = React.useState('');
  const [source, setSource] = React.useState('');
  const [mode, setMode] = React.useState<'copy' | 'empty'>('copy');
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState('');
  const [showArchived, setShowArchived] = React.useState(false);
  const [compareTo, setCompareTo] = React.useState('');
  const [comparison, setComparison] = React.useState<ReturnType<typeof configDifferences> | null>(null);
  const archivedActivations = React.useRef(new Set<string>());
  const supported = !!project.active_version_id || !!versionId;
  const selectedId = versionId || project.active_version_id || undefined;
  React.useEffect(() => {
    if (busy || !versionId || archivedActivations.current.has(versionId) || current?.status !== 'ready' || current.archived || current.busy || project.active_version_id === versionId) return;
    let active = true;
    void activateProjectVersion(project.id,versionId).then(updated => {if(active && updated)queryClient.setQueryData(['project',project.id],updated);}).catch(error => {if(active)setError(formatApiError(error));});
    return () => { active = false; };
  }, [busy,versionId,current?.status,current?.archived,current?.busy,project.id,project.active_version_id,queryClient]);
  const readyVersions = versions.filter(item => item.status === 'ready');
  const copySources = readyVersions.filter(item => !item.archived && !('busy' in item && item.busy));
  const begin = async (kind: typeof dialog) => {
    if (busy) return;
    setError(''); setBusy(true);
    try {
      if (!current?.archived) await beforeAction?.();
      setName(kind === 'edit' ? current?.name || '' : `v${versions.length + 1}`);
      setNote(kind === 'edit' ? current?.note || '' : '');
      setSource(copySources.find(item => item.id === current?.id)?.id || copySources[0]?.id || ''); setMode('copy');
      setCompareTo(readyVersions.find(item => item.id !== selectedId)?.id || ''); setComparison(null);
      setDialog(kind);
    } catch (error) { setError(formatApiError(error)); }
    finally { setBusy(false); }
  };
  const save = async (event: React.FormEvent) => {
    event.preventDefault(); if (!name.trim() || busy) return;
    setBusy(true); setError('');
    try {
      if (dialog === 'create') {
        if (source && !copySources.some(item => item.id === source)) throw new Error(text('来源版本已不可复制，请重新选择。', 'The source version is no longer available to copy. Choose another source.'));
        const next = await apiClient.post<ProjectVersion>(`/projects/${project.id}/versions`, { name: name.trim(), note: note.trim(), source_version_id: source || null, data_mode: source ? mode : 'empty', copy_config: !!source }, { silent: true });
        await refresh(); setDialog(null); navigate(projectUrl(project.id, next.id));
      } else if (current && !current.archived) {
        await apiClient.patch(`/projects/${project.id}/versions/${current.id}`, { name: name.trim(), note: note.trim() }, { silent: true });
        await refresh(); setDialog(null);
      }
    } catch (error) { setError(formatApiError(error)); }
    finally { setBusy(false); }
  };
  const archive = async () => {
    if (!current || busy) return;
    setBusy(true); setError('');
    try {
      const updated = await apiClient.patch<ProjectVersion>(`/projects/${project.id}/versions/${current.id}`, { archived: !current.archived }, { silent: true });
      // The mutation can finish before query observers render the new version.
      if (updated.archived) archivedActivations.current.add(current.id);
      else archivedActivations.current.delete(current.id);
      queryClient.setQueryData<ProjectVersion[]>(['project-versions', project.id], previous => previous?.map(item => item.id === updated.id ? updated : item));
      await Promise.all([refresh(), queryClient.invalidateQueries({ queryKey: ['project', project.id] })]); setDialog(null);
    } catch (error) { setError(formatApiError(error)); }
    finally { setBusy(false); }
  };
  const compare = async () => {
    if (!current || !compareTo || busy) return;
    setBusy(true); setError(''); setComparison(null);
    try {
      const [before, after] = await Promise.all([apiClient.get(versionConfigUrl(project.id, compareTo), { silent: true }), apiClient.get(versionConfigUrl(project.id, current.id), { silent: true })]);
      setComparison(configDifferences(before, after));
    } catch (error) { setError(formatApiError(error)); }
    finally { setBusy(false); }
  };
  const formatValue = (value: unknown) => value === undefined ? '—' : JSON.stringify(value);
  const problem = error || (loadError ? formatApiError(loadError) : '');
  return <>
    <header className="project-identity">
      <div className="project-identity-main"><Link to="/projects" className="project-back"><ArrowLeft size={14}/>{text('项目', 'Projects')}</Link><h1>{project.name}<span>/</span><span className="project-version-name">{current?.name || (supported ? '…' : text('工作区', 'Workspace'))}</span></h1><p>{current?.note || project.note || text('数据、参数与结果在当前版本内管理', 'Manage data, parameters and results in this version')}</p></div>
      <div className="project-heading-status">{status}</div>
    </header>
    {supported && <div className="project-version-strip"><GitBranch size={15}/><span className="version-strip-label">{text('版本', 'Version')}</span><div className="version-list" role="navigation" aria-label={text('项目版本', 'Project versions')}>
      {versions.filter(item => showArchived || !item.archived || item.id === selectedId).map(item => <Link key={item.id} to={projectUrl(project.id, item.id, active)} aria-current={item.id === selectedId ? 'page' : undefined} className={`version-chip ${item.id === selectedId ? 'selected' : ''}`}><span className={`version-status-dot status-${item.status}`}/>{item.name}{item.archived && <small>{text('已归档', 'Archived')}</small>}{item.status === 'copying' && <Loader2 size={12} className="animate-spin"/>}</Link>)}
    </div><div className="version-actions"><button onClick={() => void begin('create')} disabled={busy || versions.some(item => item.status === 'copying')}><Plus size={14}/>{text('新版本', 'New version')}</button><button onClick={() => void begin('compare')} disabled={!current || readyVersions.length < 2 || busy} title={text('比较版本参数', 'Compare version parameters')}><GitCompare size={14}/>{text('比较', 'Compare')}</button><button onClick={() => void begin('paths')} disabled={!current} title={text('查看本版本目录', 'View version folders')} aria-label={text('查看本版本目录', 'View version folders')}><FolderOpen size={15}/></button><button onClick={() => void begin('edit')} disabled={!current || current.status === 'copying' || busy} aria-label={text('版本设置', 'Version settings')}><Settings2 size={15}/></button></div></div>}
    {versions.some(item => item.archived) && <label className="version-archive-toggle"><input type="checkbox" checked={showArchived} onChange={event => setShowArchived(event.target.checked)}/>{text('显示已归档版本', 'Show archived versions')}</label>}
    {problem && !dialog && <div role="alert" className="workspace-message error">{problem}<button onClick={() => {setError(''); void refresh();}}>{text('重试', 'Retry')}</button></div>}
    {current?.archived && <div className="workspace-message" role="status"><AlertCircle size={16}/><div><strong>{text('此版本已归档 · 只读', 'This version is archived · Read only')}</strong><p>{text('可以查看数据、比较参数和下载已有结果；恢复版本后继续编辑与训练。', 'View data, compare configurations and download existing results. Restore the version to edit or train.')}</p></div><button disabled={busy} onClick={() => void archive()}>{busy ? text('正在恢复…', 'Restoring…') : text('恢复版本', 'Restore version')}</button></div>}
    {current?.status === 'copying' && <div className="workspace-message" role="status"><Loader2 size={16} className="animate-spin"/><div><strong>{text('正在建立独立版本', 'Creating an independent version')}</strong><p>{text('复制图片、标签与遮罩，完成后即可编辑；原版本保持不变。', 'Copying images, captions and masks. The original version is preserved.')}</p><progress max={Math.max(1,current.progress?.files_total || 0)} value={current.progress?.files_done || 0}/><span>{current.progress?.files_done || 0} / {current.progress?.files_total || '…'} {text('个文件', 'files')}</span></div></div>}
    {current?.status === 'failed' && <div className="workspace-message error" role="alert"><AlertCircle size={16}/><div><strong>{text('版本准备失败', 'Version preparation failed')}</strong><p>{current.error}</p><p>{text('原版本数据仍然保留。修正原因后，可从原版本重新创建。', 'Original data is preserved. Resolve the issue and create again from the source version.')}</p></div></div>}
    <ProjectWorkflow projectId={project.id} versionId={selectedId} active={active}/>
    {dialog && <Dialog title={dialog === 'create' ? text('新建实验版本', 'New experiment version') : dialog === 'edit' ? text('版本设置', 'Version settings') : dialog === 'compare' ? text('比较版本', 'Compare versions') : text('本版本的文件位置', 'Files in this version')} onClose={() => { if (!busy) {setDialog(null);setError('');} }} wide={dialog === 'compare' || dialog === 'paths'}>
      {error && <div role="alert" className="workspace-message error">{error}</div>}
      {(dialog === 'create' || dialog === 'edit') && <form onSubmit={save} className="version-form"><label>{text('版本名称', 'Version name')}<input value={name} onChange={event => setName(event.target.value)} maxLength={120} required disabled={busy || dialog === 'edit' && !!current?.archived}/></label><label>{text('实验说明', 'Experiment notes')}<textarea value={note} onChange={event => setNote(event.target.value)} placeholder={text('例如：仅训练服装区域，学习率调整为 0.0002', 'For example: train clothing only, learning rate 0.0002')} disabled={busy || dialog === 'edit' && !!current?.archived}/></label>
        {dialog === 'create' && <><label>{text('创建来源', 'Create from')}<select value={source} onChange={event => setSource(event.target.value)} disabled={busy}><option value="">{text('默认配置 · 空白版本', 'Default configuration · blank version')}</option>{copySources.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>{source && <fieldset><legend>{text('继承内容', 'Include')}</legend><label><input type="radio" name="version-mode" checked={mode === 'copy'} onChange={() => setMode('copy')} disabled={busy}/><div><strong>{text('配置和完整数据副本', 'Configuration and a complete data copy')}</strong><p>{text('图片、标签、遮罩、验证集分别复制，后续编辑互不影响。', 'Independent images, captions, masks and validation data. Later edits do not affect the source.')}</p></div></label><label><input type="radio" name="version-mode" checked={mode === 'empty'} onChange={() => setMode('empty')} disabled={busy}/><div><strong>{text('仅配置，重新准备数据', 'Configuration only, prepare new data')}</strong><p>{text('保留实验参数，清空数据来源。', 'Keep experiment parameters and start with no data sources.')}</p></div></label></fieldset>}<p className="version-copy-note"><Copy size={14}/>{text('训练记录、采样图和产物不会复制到新版本。', 'Jobs, generated samples and outputs stay in the original version.')}</p></>}
        <footer>{dialog === 'edit' && <button type="button" onClick={() => void archive()} disabled={busy || versions.filter(item => !item.archived && item.status === 'ready').length < 2 && !current?.archived}>{current?.archived ? text('恢复版本', 'Restore version') : text('归档版本', 'Archive version')}</button>}<span/><button type="button" disabled={busy} onClick={() => setDialog(null)}>{text('取消', 'Cancel')}</button><button type="submit" className="primary" disabled={busy || !name.trim() || dialog === 'edit' && !!current?.archived}>{busy && <Loader2 size={14} className="animate-spin"/>}{dialog === 'create' ? text('创建版本', 'Create version') : text('保存', 'Save')}</button></footer>
      </form>}
      {dialog === 'paths' && current && <div className="version-paths"><p>{text('图片、标签和遮罩位于当前版本的数据目录；每次训练使用独立任务目录。', 'Images, captions and masks belong to this version. Each training job has a separate run folder.')}</p><dl>{([{key:'datasets',label:text('训练数据', 'Training data')},{key:'config',label:text('参数草稿', 'Configuration draft')},{key:'cache',label:text('编码缓存', 'Encoding cache')},{key:'runs',label:text('任务与产物', 'Runs and outputs')}] as const).map(item => <div key={item.key}><dt>{item.label}</dt><dd>{current.paths[item.key]}</dd></div>)}</dl><div className="version-tree">{text('任务目录/', 'run/')}<br/> config.toml · {text('此次训练参数快照', 'configuration snapshot')}<br/> samples/ · {text('训练期间采样图', 'training samples')}<br/> *.safetensors · {text('导出权重', 'saved weights')}<br/> events.jsonl · run.log</div></div>}
      {dialog === 'compare' && current && <div className="version-comparison"><div className="comparison-controls"><label>{text('对比版本', 'Compare from')}<select value={compareTo} onChange={event => { setCompareTo(event.target.value);setComparison(null); }}>{readyVersions.filter(item => item.id !== current.id).map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label><span>→ {current.name}</span><button disabled={busy || !compareTo} onClick={() => void compare()}>{busy ? text('读取中…', 'Loading…') : text('查看差异', 'Show differences')}</button></div><div className="version-comparison-stats">{[versions.find(item => item.id === compareTo),current].filter(Boolean).map(item => <div key={item!.id}><strong>{item!.name}</strong><span>{item!.stats.images} {text('张图片', 'images')} · {item!.stats.jobs} {text('次训练', 'jobs')} · {item!.stats.artifacts} {text('个产物', 'outputs')}</span></div>)}</div>{comparison && (comparison.length ? <div className="comparison-table-wrap"><table><thead><tr><th>{text('参数', 'Parameter')}</th><th>{versions.find(item => item.id === compareTo)?.name}</th><th>{current.name}</th></tr></thead><tbody>{comparison.map(item => <tr key={item.path}><th>{item.path}</th><td>{formatValue(item.before)}</td><td>{formatValue(item.after)}</td></tr>)}</tbody></table></div> : <p>{text('两个版本的参数相同。', 'The configurations are identical.')}</p>)}</div>}
    </Dialog>}
  </>;
}
