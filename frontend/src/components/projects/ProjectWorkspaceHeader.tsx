import React from 'react';
import { createPortal } from 'react-dom';
import { useQueryClient } from '@tanstack/react-query';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { GitBranch, Plus, Settings2, GitCompare, FolderOpen, Loader2, Copy, AlertCircle } from 'lucide-react';
import { apiClient } from '../../api/client';
import { useFamilies } from '../../api/hooks/useFamilies';
import { trainingFamilyOptions } from '../../utils/trainingFamilies';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import { activateProjectVersion, configDifferences, projectUrl, versionConfigUrl, type ProjectVersion, type VersionedProject } from '../../utils/projectVersions';
import { type WorkspaceStep, ProjectWorkflow } from '../ProjectWorkflow';
import Dialog from '../Dialog';
import StudioSelect from '../StudioSelect';
import { useWorkspaceHeight } from './useWorkspaceHeight';
import { ProjectSidebarContext } from './ProjectSidebarContext';
import '../../styles/project-sidebar.css';

interface Props {
  project: VersionedProject; versionId?: string; versions: ProjectVersion[]; current?: ProjectVersion;
  active: WorkspaceStep; refresh: () => Promise<unknown>; beforeAction?: () => Promise<void>;
  status?: React.ReactNode; error?: unknown; title?: string; titleBadge?: React.ReactNode;
}
export default function ProjectWorkspaceHeader({ project, versionId, versions, current, active, refresh, beforeAction, status, error: loadError, title: customTitle, titleBadge }: Props) {
  const text = useWorkspaceText();
  const { data: families = [], isError: familiesError } = useFamilies();
  const navigationRef = useWorkspaceHeight('--workspace-head-height');
  const sidebar = React.useContext(ProjectSidebarContext);
  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();
  const [dialog, setDialog] = React.useState<'create' | 'edit' | 'compare' | 'paths' | null>(null);
  const [name, setName] = React.useState('');
  const [note, setNote] = React.useState('');
  const [source, setSource] = React.useState('');
  const [family, setFamily] = React.useState('');
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
  const sourceFamily = versions.find(item => item.id === source)?.family;
  const familyLabel = (value?: string) => families.find(item => item.name === value)?.label || value || text('沿用配置', 'From configuration');
  const familyOptions = trainingFamilyOptions(families, text('zh', 'en') === 'en', sourceFamily || family);
  const incompatibleFamily = !!sourceFamily && !!family && sourceFamily !== family;
  const copySources = readyVersions.filter(item => !item.archived && !('busy' in item && item.busy));
  const switchVersion = async (nextId: string) => {
    if (busy || nextId === selectedId || !versions.some(item => item.id === nextId)) return;
    setBusy(true); setError('');
    try {
      if (!current?.archived) await beforeAction?.();
      const tab = active === 'train' ? new URLSearchParams(location.search).get('tab') : null;
      navigate(`${projectUrl(project.id, nextId, active)}${tab ? `?tab=${encodeURIComponent(tab)}` : ''}`);
      sidebar?.closeNavigation();
    } catch (error) { setError(formatApiError(error)); }
    finally { setBusy(false); }
  };
  const begin = async (kind: typeof dialog) => {
    if (busy) return;
    setError(''); setBusy(true);
    try {
      if (!current?.archived) await beforeAction?.();
      setName(kind === 'edit' ? current?.name || '' : `v${Math.max(versions.length, ...versions.map(item => item.number || 0)) + 1}`);
      setNote(kind === 'edit' ? current?.note || '' : '');
      const origin = copySources.find(item => item.id === current?.id) || copySources[0];
      setSource(origin?.id || ''); setFamily(origin?.family || ''); setMode('copy');
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
        const next = await apiClient.post<ProjectVersion>(`/projects/${project.id}/versions`, { name: name.trim(), note: note.trim(), source_version_id: source || null, data_mode: source ? mode : 'empty', copy_config: !!source, ...(family ? {family} : {}) }, { silent: true });
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
  const versionState = (item: ProjectVersion) => item.archived ? text('已归档', 'Archived') : item.status === 'copying' ? text('创建中', 'Creating') : item.status === 'failed' ? text('失败', 'Failed') : item.busy ? text('使用中', 'In use') : text('就绪', 'Ready');
  const title = customTitle ?? (active === 'overview' ? text('项目概览', 'Project overview') : active === 'data' ? text('训练数据', 'Training data') : active === 'train' ? text('训练参数', 'Training parameters') : text('训练结果', 'Training results'));
  const projectControls = <section className="project-sidebar" aria-label={text('当前项目工作区', 'Current project workspace')}>
    <Link className="project-sidebar-identity" to={projectUrl(project.id, selectedId, 'overview')} title={project.note || project.name}><strong title={project.name}>{project.name}</strong><small title={project.id}>{project.id}</small></Link>
    {supported && <>
      <div className="project-sidebar-version-heading"><span>{text('当前版本', 'Current version')}</span>{current && <span className={`project-sidebar-version-state status-${current.status}`} title={current.error || current.note || versionState(current)}>{current.status === 'copying' && <Loader2 size={11} className="animate-spin"/>}{versionState(current)}</span>}</div>
      <StudioSelect searchable aria-label={text('项目版本', 'Project versions')} value={selectedId || ''} disabled={busy || !versions.length} icon={<GitBranch size={13}/>} onValueChange={value => void switchVersion(value)} options={versions.filter(item => showArchived || !item.archived || item.id === selectedId).map(item => ({value:item.id,label:`${item.name}${item.archived || item.status !== 'ready' ? ` · ${versionState(item)}` : ''}`}))}/>
      <div className="project-sidebar-model">{text('模型类型', 'Model type')} · {familyLabel(current?.family)}</div>
      <div className="project-sidebar-actions" aria-label={text('版本操作', 'Version actions')}>
        <button type="button" onClick={() => void begin('create')} disabled={busy || versions.some(item => item.status === 'copying')} title={text('新版本', 'New version')} aria-label={text('新版本', 'New version')}><Plus size={14}/></button>
        <button type="button" onClick={() => void begin('compare')} disabled={!current || readyVersions.length < 2 || busy} title={text('比较版本参数', 'Compare version parameters')} aria-label={text('比较', 'Compare')}><GitCompare size={14}/></button>
        <button type="button" onClick={() => void begin('paths')} disabled={!current || busy} title={text('查看本版本目录', 'View version folders')} aria-label={text('查看本版本目录', 'View version folders')}><FolderOpen size={14}/></button>
        <button type="button" onClick={() => void begin('edit')} disabled={!current || current.status === 'copying' || busy} title={text('版本设置', 'Version settings')} aria-label={text('版本设置', 'Version settings')}><Settings2 size={14}/></button>
      </div>
      {versions.some(item => item.archived) && <label className="project-sidebar-archived"><input type="checkbox" checked={showArchived} onChange={event => setShowArchived(event.target.checked)}/>{text('显示已归档版本', 'Show archived versions')}</label>}
    </>}
    <ProjectWorkflow projectId={project.id} versionId={selectedId} active={active} sidebar/>
  </section>;
  return <>
    {sidebar ? sidebar.target && createPortal(projectControls, sidebar.target) : <div className="project-sidebar-fallback">{projectControls}</div>}
    <header className="workspace-navigation workspace-page-heading" ref={navigationRef}><div className="workspace-heading-main"><nav className="workspace-breadcrumb" aria-label={text('当前位置', 'Current location')}><Link to="/projects">{text('项目', 'Projects')}</Link><span aria-hidden="true">/</span><Link to={projectUrl(project.id, selectedId, 'overview')}>{project.name}</Link>{current && <><span aria-hidden="true">/</span><span>{current.name}</span></>}</nav><div className="workspace-heading-title"><h1 title={title}>{title}</h1>{titleBadge}</div></div>{status && <div className="project-heading-status">{status}</div>}</header>
    {problem && !dialog && <div role="alert" className="workspace-message error">{problem}<button onClick={() => {setError(''); void refresh();}}>{text('重试', 'Retry')}</button></div>}
    {current?.archived && <div className="workspace-message" role="status"><AlertCircle size={16}/><div><strong>{text('此版本已归档 · 只读', 'This version is archived · Read only')}</strong><p>{text('可以查看数据、比较参数和下载已有结果；恢复版本后继续编辑与训练。', 'View data, compare configurations and download existing results. Restore the version to edit or train.')}</p></div><button disabled={busy} onClick={() => void archive()}>{busy ? text('正在恢复…', 'Restoring…') : text('恢复版本', 'Restore version')}</button></div>}
    {current?.status === 'copying' && <div className="workspace-message" role="status"><Loader2 size={16} className="animate-spin"/><div><strong>{text('正在建立独立版本', 'Creating an independent version')}</strong><p>{text('复制图片、标签与遮罩，完成后即可编辑；原版本保持不变。', 'Copying images, captions and masks. The original version is preserved.')}</p><progress max={Math.max(1,current.progress?.files_total || 0)} value={current.progress?.files_done || 0}/><span>{current.progress?.files_done || 0} / {current.progress?.files_total || '…'} {text('个文件', 'files')}</span></div></div>}
    {current?.status === 'failed' && <div className="workspace-message error" role="alert"><AlertCircle size={16}/><div><strong>{text('版本准备失败', 'Version preparation failed')}</strong><p>{current.error}</p><p>{text('原版本数据仍然保留。修正原因后，可从原版本重新创建。', 'Original data is preserved. Resolve the issue and create again from the source version.')}</p></div></div>}
    {dialog && <Dialog title={dialog === 'create' ? text('新建实验版本', 'New experiment version') : dialog === 'edit' ? text('版本设置', 'Version settings') : dialog === 'compare' ? text('比较版本', 'Compare versions') : text('本版本的文件位置', 'Files in this version')} onClose={() => { if (!busy) {setDialog(null);setError('');} }} closeDisabled={busy} wide={dialog === 'compare' || dialog === 'paths'}>
      {error && <div role="alert" className="workspace-message error">{error}</div>}
      {(dialog === 'create' || dialog === 'edit') && <form onSubmit={save} className="version-form"><label>{text('版本名称', 'Version name')}<input value={name} onChange={event => setName(event.target.value)} maxLength={120} required disabled={busy || dialog === 'edit' && !!current?.archived}/></label><label>{text('实验说明', 'Experiment notes')}<textarea value={note} onChange={event => setNote(event.target.value)} placeholder={text('例如：仅训练服装区域，学习率调整为 0.0002', 'For example: train clothing only, learning rate 0.0002')} disabled={busy || dialog === 'edit' && !!current?.archived}/></label>
        {dialog === 'create' && <><label>{text('创建来源', 'Create from')}<StudioSelect aria-label={text('创建来源', 'Create from')} value={source} onValueChange={value => { setSource(value);setFamily(versions.find(item => item.id === value)?.family || ''); }} disabled={busy} options={[{value:'',label:text('默认配置 · 空白版本', 'Default configuration · blank version')},...copySources.map(item => ({value:item.id,label:item.name}))]}/></label>
          <label>{text('训练模型类型', 'Training model type')}<StudioSelect aria-label={text('训练模型类型', 'Training model type')} value={family} onValueChange={setFamily} disabled={busy} options={[{value:'',label:source ? text(`沿用来源版本 · ${familyLabel(sourceFamily)}`, `Inherit source · ${familyLabel(sourceFamily)}`) : text('默认类型 · Anima', 'Default type · Anima')},...familyOptions]}/></label>
          <p className="version-family-note">{incompatibleFamily ? text('模型类型不同：按所选类型重建底模组件和训练默认配置，清除原模型路径与恢复权重；数据按下方选项继承。', 'Different model type: initialize the selected engine defaults and clear incompatible model paths and resume weights. Dataset inheritance follows the choice below.') : text('自动使用对应的底模组件、默认参数和模型库默认权重。未下载的组件可在训练参数中补充。', 'Use matching model components, training defaults and default library weights. Missing components can be selected in training parameters.')}</p>
          {familiesError && <p className="version-family-note" role="status">{text('模型类型列表暂时无法读取，仍可沿用来源版本配置。', 'Model types are unavailable. The source configuration can still be inherited.')}</p>}
          {source && <fieldset><legend>{text('继承内容', 'Include')}</legend><label><input type="radio" name="version-mode" checked={mode === 'copy'} onChange={() => setMode('copy')} disabled={busy}/><div><strong>{incompatibleFamily ? text('复制数据，使用新模型配置', 'Copy data with new model defaults') : text('配置和完整数据副本', 'Configuration and a complete data copy')}</strong><p>{text('图片、标签、遮罩、验证集分别复制，后续编辑互不影响。', 'Independent images, captions, masks and validation data. Later edits do not affect the source.')}</p></div></label><label><input type="radio" name="version-mode" checked={mode === 'empty'} onChange={() => setMode('empty')} disabled={busy}/><div><strong>{incompatibleFamily ? text('使用新模型配置，重新准备数据', 'New model defaults and new data') : text('仅配置，重新准备数据', 'Configuration only, prepare new data')}</strong><p>{incompatibleFamily ? text('按所选模型类型重建默认参数，清空数据来源。', 'Initialize defaults for the selected model and start with no data sources.') : text('保留实验参数，清空数据来源。', 'Keep experiment parameters and start with no data sources.')}</p></div></label></fieldset>}<p className="version-copy-note"><Copy size={14}/>{text('训练记录、采样图和产物不会复制到新版本。', 'Jobs, generated samples and outputs stay in the original version.')}</p></>}
        <footer>{dialog === 'edit' && <button type="button" onClick={() => void archive()} disabled={busy || versions.filter(item => !item.archived && item.status === 'ready').length < 2 && !current?.archived}>{current?.archived ? text('恢复版本', 'Restore version') : text('归档版本', 'Archive version')}</button>}<span/><button type="button" disabled={busy} onClick={() => setDialog(null)}>{text('取消', 'Cancel')}</button><button type="submit" className="primary" disabled={busy || !name.trim() || dialog === 'edit' && !!current?.archived}>{busy && <Loader2 size={14} className="animate-spin"/>}{dialog === 'create' ? text('创建版本', 'Create version') : text('保存', 'Save')}</button></footer>
      </form>}
      {dialog === 'paths' && current && <div className="version-paths"><p>{text('显示名称可重命名；项目 ID 与版本目录保持不变。图片、标签及遮罩保存在对应数据目录中。', 'Display names can be renamed; project IDs and version folders stay unchanged. Images, captions and masks remain in their data folders.')}</p><dl>{[
        {key:'root',label:text('本版本目录', 'Version folder'),path:current.paths.root},
        {key:'traindata',label:text('训练数据', 'Training data'),path:current.paths.traindata || current.paths.datasets},
        {key:'reg',label:text('正则图', 'Regularization images'),path:current.paths.reg},
        {key:'samples',label:text('采样图', 'Samples'),path:current.paths.samples},
        {key:'output',label:text('模型产物', 'Model outputs'),path:current.paths.output || current.paths.runs},
        {key:'config',label:text('参数草稿', 'Configuration draft'),path:current.paths.config},
        {key:'cache',label:text('编码缓存', 'Encoding cache'),path:current.paths.cache},
      ].filter(item=>item.path).map(item=><div key={item.key}><dt>{item.label}</dt><dd>{item.path}</dd></div>)}</dl><div className="version-tree">{current.paths.output && current.paths.samples ? <>{text('每次训练按任务 ID 分开保存：','Each training job has its own ID subfolder:')}<br/>samples/{'<job_id>'}/ · {text('采样图片','sample images')}<br/>output/{'<job_id>'}/ · {text('权重、训练配置和日志','weights, configuration and logs')}<br/>{text('使用自定义输出位置时，以上方服务返回的实际路径为准。','When a custom output location is used, the actual paths above apply.')}</> : <>{text('旧项目沿用原有任务目录，不会迁移已保存文件。','Existing projects retain their original run folders and files.')}<br/>config.toml · samples/ · *.safetensors · events.jsonl · run.log</>}</div></div>}
      {dialog === 'compare' && current && <div className="version-comparison"><div className="comparison-controls"><label>{text('对比版本', 'Compare from')}<StudioSelect aria-label={text('对比版本', 'Compare from')} value={compareTo} disabled={busy} onValueChange={value => { setCompareTo(value);setComparison(null); }} options={readyVersions.filter(item => item.id !== current.id).map(item => ({value:item.id,label:item.name}))}/></label><span>→ {current.name}</span><button disabled={busy || !compareTo} onClick={() => void compare()}>{busy ? text('读取中…', 'Loading…') : text('查看差异', 'Show differences')}</button></div><div className="version-comparison-stats">{[versions.find(item => item.id === compareTo),current].filter(Boolean).map(item => <div key={item!.id}><strong>{item!.name}</strong><span>{item!.stats.images} {text('张图片', 'images')} · {item!.stats.jobs} {text('次训练', 'jobs')} · {item!.stats.artifacts} {text('个产物', 'outputs')}</span></div>)}</div>{comparison && (comparison.length ? <div className="comparison-table-wrap"><table><thead><tr><th>{text('参数', 'Parameter')}</th><th>{versions.find(item => item.id === compareTo)?.name}</th><th>{current.name}</th></tr></thead><tbody>{comparison.map(item => <tr key={item.path}><th>{item.path}</th><td>{formatValue(item.before)}</td><td>{formatValue(item.after)}</td></tr>)}</tbody></table></div> : <p>{text('两个版本的参数相同。', 'The configurations are identical.')}</p>)}</div>}
    </Dialog>}
  </>;
}
