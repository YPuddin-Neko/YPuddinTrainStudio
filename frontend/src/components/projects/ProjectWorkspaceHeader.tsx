import React from 'react';
import { createPortal } from 'react-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { GitBranch, Plus, Settings2, GitCompare, FolderOpen, Loader2, Copy, AlertCircle } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { FamilyInfo } from '../../api/types';
import { ttsApi, type TtsConfigResponse, type TtsEngine } from '../../api/tts';
import { useFamilies } from '../../api/hooks/useFamilies';
import { inactiveTrainingReason, trainingFamilyOptions } from '../../utils/trainingFamilies';
import { useWorkspaceText } from '../../utils/workspaceText';
import { ttsEngineLabel } from '../../utils/ttsEngines';
import { formatApiError } from '../../utils/errors';
import { activateProjectVersion, configDifferences, projectUrl, versionConfigUrl, type ProjectVersion, type VersionedProject } from '../../utils/projectVersions';
import { type WorkspaceStep, ProjectWorkflow } from '../ProjectWorkflow';
import Dialog from '../Dialog';
import StudioSelect from '../StudioSelect';
import { useWorkspaceHeight } from './useWorkspaceHeight';
import { ProjectSidebarContext } from './ProjectSidebarContext';
import '../../styles/project-sidebar.css';
import Switch from '../Switch';
import TopbarBreadcrumb from '../TopbarBreadcrumb';

interface Props {
  project: VersionedProject; versionId?: string; versions: ProjectVersion[]; current?: ProjectVersion;
  active: WorkspaceStep; refresh: () => Promise<unknown>; beforeAction?: () => Promise<void>;
  sidebarOnly?: boolean; workflowActive?: boolean;
  status?: React.ReactNode; error?: unknown; title?: string; titleBadge?: React.ReactNode; breadcrumbTrail?: React.ReactNode;
}
export default function ProjectWorkspaceHeader(props: Props) {
  return props.project.project_type === 'tts'
    ? <WorkspaceHeader {...props} families={[]} familiesError={false}/>
    : <ImageWorkspaceHeader {...props}/>;
}

function ImageWorkspaceHeader(props: Props) {
  const { data: families = [], isError: familiesError } = useFamilies();
  return <WorkspaceHeader {...props} families={families} familiesError={familiesError}/>;
}

function WorkspaceHeader({ project, versionId, versions, current, active, refresh, beforeAction, status, error: loadError, title: customTitle, titleBadge, breadcrumbTrail, sidebarOnly = false, workflowActive = true, families, familiesError }: Props & { families: FamilyInfo[]; familiesError: boolean }) {
  const text = useWorkspaceText();
  const speech = project.project_type === 'tts';
  const ownerArchived = speech && !!project.archived;
  const ownerArchivedRef = React.useRef(ownerArchived); ownerArchivedRef.current = ownerArchived;
  const ownerArchivedReason = ownerArchived ? text('项目已归档，请先恢复项目。', 'Restore the archived project before making changes.') : undefined;
  const navigationRef = useWorkspaceHeight('--workspace-head-height', !sidebarOnly);
  const sidebar = React.useContext(ProjectSidebarContext);
  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();
  const registerSidebar = sidebar?.register;
  React.useLayoutEffect(()=>{
    if(sidebarOnly || !registerSidebar)return;
    return registerSidebar({project,versionId,versions,current,active,routeKey:location.key,pathname:location.pathname},beforeAction,refresh);
  },[sidebarOnly,registerSidebar,project,versionId,versions,current,active,location.key,location.pathname,beforeAction,refresh]);
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
  const [engineConfig, setEngineConfig] = React.useState<TtsConfigResponse | null>(null);
  const [nextEngine, setNextEngine] = React.useState<TtsEngine>('voxcpm1.5');
  const [nextVariant, setNextVariant] = React.useState<'v5dev' | 'v5turbo'>('v5dev');
  const [createEngine, setCreateEngine] = React.useState<TtsEngine | ''>('');
  const [createVariant, setCreateVariant] = React.useState<'v5dev' | 'v5turbo' | ''>('');
  const [engineConflict, setEngineConflict] = React.useState(false);
  const capabilities = useQuery({ queryKey: ['tts-capabilities'], queryFn: ({ signal }) => ttsApi.capabilities(signal), enabled: speech && (dialog === 'edit' || dialog === 'create') });
  const engineOptions = (capabilities.data?.engines || []).filter(item => item.id === 'voxcpm1.5' || item.id === 'gpt-sovits-v5').map(item => ({ value: item.id, label: ttsEngineLabel(item.id) }));
  const engineChanged = !!engineConfig && (nextEngine !== engineConfig.config.engine || nextEngine === 'gpt-sovits-v5' && engineConfig.config.engine === 'gpt-sovits-v5' && nextVariant !== engineConfig.config.variant);
  const engineUnavailable = !engineOptions.some(item => item.value === nextEngine);
  const engineWriteBlocked = engineChanged && (engineUnavailable || !!current?.busy || current?.status !== 'ready');
  const archivedActivations = React.useRef(new Set<string>());
  const supported = !!project.active_version_id || !!versionId;
  const selectedId = versionId || project.active_version_id || undefined;
  React.useEffect(() => {
    if (sidebarOnly || busy || ownerArchived || !versionId || archivedActivations.current.has(versionId) || current?.status !== 'ready' || current.archived || current.busy || project.active_version_id === versionId) return;
    let active = true;
    void activateProjectVersion(project.id,versionId).then(updated => {if(active && updated)queryClient.setQueryData(['project',project.id],updated);}).catch(error => {if(active)setError(formatApiError(error));});
    return () => { active = false; };
  }, [sidebarOnly,busy,ownerArchived,versionId,current?.status,current?.archived,current?.busy,project.id,project.active_version_id,queryClient]);
  const readyVersions = versions.filter(item => item.status === 'ready');
  const sourceFamily = versions.find(item => item.id === source)?.family;
  const sourceVersion = versions.find(item => item.id === source);
  const createTargetEngine = createEngine || sourceVersion?.engine || current?.engine || project.active_engine || 'voxcpm1.5';
  const createTargetVariant = createVariant || (sourceVersion?.engine === createTargetEngine ? sourceVersion.variant : null) || 'v5dev';
  const changedSourceEngine = speech && !!sourceVersion?.engine && createTargetEngine !== sourceVersion.engine;
  const changedSourceVariant = speech && !changedSourceEngine && createTargetEngine === 'gpt-sovits-v5' && !!sourceVersion?.variant && createTargetVariant !== sourceVersion.variant;
  const createEngineUnavailable = speech && (!!createEngine || !source) && !engineOptions.some(item => item.value === createTargetEngine);
  const familyLabel = (value?: string | null) => value === 'flux' ? `FLUX.1 · ${text('已停用', 'Retired')}` : trainingFamilyOptions(families, text('zh', 'en') === 'en', value || undefined).find(item => item.value === value)?.label || value || text('沿用配置', 'From configuration');
  const familyOptions = trainingFamilyOptions(families, text('zh', 'en') === 'en', dialog === 'edit' ? family : sourceFamily || family);
  const incompatibleFamily = speech ? changedSourceEngine : !!sourceFamily && !!family && sourceFamily !== family;
  const hasUncopyableAudio = (item?: ProjectVersion) => speech && !!item && (!item.audio_stats || [item.audio_stats.train, item.audio_stats.validation].some(split => split && !['missing', 'valid'].includes(split.state)));
  const audioCopyUnavailable = hasUncopyableAudio(versions.find(item => item.id === source));
  const copySources = readyVersions.filter(item => !item.archived && !('busy' in item && item.busy));
  const switchVersion = async (nextId: string) => {
    if (busy || nextId === selectedId || !versions.some(item => item.id === nextId)) return;
    setBusy(true); setError('');
    try {
      if (!current?.archived && !ownerArchived) await beforeAction?.();
      const tab = active === 'train' && workflowActive ? new URLSearchParams(location.search).get('tab') : null;
      navigate(`${projectUrl(project.id, nextId, active)}${tab ? `?tab=${encodeURIComponent(tab)}` : ''}`);
      sidebar?.closeNavigation();
    } catch (error) { setError(formatApiError(error)); }
    finally { setBusy(false); }
  };
  const begin = async (kind: typeof dialog) => {
    if (busy || ownerArchived && (kind === 'create' || kind === 'edit')) return;
    setError(''); setBusy(true);
    try {
      if (!current?.archived && !ownerArchived) await beforeAction?.();
      if (ownerArchivedRef.current && (kind === 'create' || kind === 'edit')) return;
      setName(kind === 'edit' ? current?.name || '' : `v${Math.max(versions.length, ...versions.map(item => item.number || 0)) + 1}`);
      setNote(kind === 'edit' ? current?.note || '' : '');
      const origin = copySources.find(item => item.id === current?.id) || copySources[0];
      setSource(origin?.id || ''); setFamily(kind === 'edit' ? current?.display_family ?? current?.family ?? '' : origin?.family || ''); setMode(hasUncopyableAudio(origin) ? 'empty' : 'copy');
      setCreateEngine(''); setCreateVariant('');
      setCompareTo(readyVersions.find(item => item.id !== selectedId)?.id || ''); setComparison(null);
      setEngineConfig(null); setEngineConflict(false);
      if (speech && kind === 'edit' && current) {
        try {
          const saved = await ttsApi.versionConfig(project.id, current.id);
          setEngineConfig(saved); setNextEngine(saved.config.engine);
          setNextVariant(saved.config.engine === 'gpt-sovits-v5' ? saved.config.variant : 'v5dev');
        } catch (failure) { setError(formatApiError(failure)); }
      }
      setDialog(kind);
    } catch (error) { setError(formatApiError(error)); }
    finally { setBusy(false); }
  };
  const save = async (event: React.FormEvent) => {
    event.preventDefault(); if (!name.trim() || busy || ownerArchived) return;
    if (speech && dialog === 'edit' && (engineConflict || engineWriteBlocked)) return;
    if (dialog === 'create' && createEngineUnavailable) return;
    setBusy(true); setError('');
    let changedModel = false;
    try {
      if (dialog === 'create') {
        if (source && !copySources.some(item => item.id === source)) throw new Error(text('来源版本已不可复制，请重新选择。', 'The source version is no longer available to copy. Choose another source.'));
        if (mode === 'copy' && audioCopyUnavailable) throw new Error(text('请先检查来源版本的音频清单，或选择仅复制配置。', 'Check the source version’s audio manifests first, or copy configuration only.'));
        const retiredFamily = !speech && inactiveTrainingReason({ model: { family: family || sourceFamily } }, text('zh', 'en') === 'en');
        if (retiredFamily) throw new Error(retiredFamily);
        if (!speech && source && sourceFamily === 'flux2' && (!family || family === sourceFamily)) {
          const sourceConfig = await apiClient.get<Record<string, any>>(versionConfigUrl(project.id, source), { silent: true });
          const retiredConfig = inactiveTrainingReason(sourceConfig, text('zh', 'en') === 'en');
          if (retiredConfig) throw new Error(retiredConfig);
        }
        const next = await apiClient.post<ProjectVersion>(`/projects/${project.id}/versions`, { name: name.trim(), note: note.trim(), source_version_id: source || null, data_mode: source ? mode : 'empty', copy_config: !!source, ...(!speech && family ? {family} : {}), ...(speech && (createEngine || !source) ? { engine: createTargetEngine } : {}), ...(speech && createTargetEngine === 'gpt-sovits-v5' && (createVariant || !source || changedSourceEngine) ? { variant: createTargetVariant } : {}) }, { silent: true });
        await refresh(); setDialog(null); navigate(projectUrl(project.id, next.id));
      } else if (current && !current.archived) {
        if (speech && engineConfig && engineChanged) {
          const saved = await ttsApi.changeEngine(project.id, current.id, { expected_revision: engineConfig.revision, engine: nextEngine, ...(nextEngine === 'gpt-sovits-v5' ? { variant: nextVariant } : {}) });
          changedModel = true; setEngineConfig(saved);
          queryClient.setQueryData(['tts-version-config', project.id, current.id], saved);
          await Promise.all([queryClient.invalidateQueries({ queryKey: ['tts-sources', project.id, current.id] }), refresh(), queryClient.invalidateQueries({ queryKey: ['project', project.id] })]);
        }
        const updated = await apiClient.patch<ProjectVersion>(`/projects/${project.id}/versions/${current.id}`, { name: name.trim(), note: note.trim(), ...(!speech && family !== (current.display_family ?? current.family ?? '') ? { display_family: family || null } : {}) }, { silent: true });
        queryClient.setQueryData<ProjectVersion[]>(['project-versions', project.id], previous => previous?.map(item => item.id === updated.id ? updated : item));
        await Promise.all([refresh(), queryClient.invalidateQueries({ queryKey: ['project', project.id] })]); setDialog(null);
      }
    } catch (error) {
      if (speech && error && typeof error === 'object' && 'code' in error && error.code === 'tts.config_conflict') setEngineConflict(true);
      setError(`${changedModel ? text('模型类型已更新，但版本信息尚未保存。', 'Model type was updated, but version details have not been saved. ') : ''}${formatApiError(error)}`);
    }
    finally { setBusy(false); }
  };
  const reloadEngine = async () => {
    if (!current || busy) return;
    setBusy(true); setError('');
    try {
      const saved = await ttsApi.versionConfig(project.id, current.id);
      setEngineConfig(saved); setEngineConflict(false);
      if (!engineConfig) { setNextEngine(saved.config.engine); setNextVariant(saved.config.engine === 'gpt-sovits-v5' ? saved.config.variant : 'v5dev'); }
    } catch (failure) { setError(formatApiError(failure)); }
    finally { setBusy(false); }
  };
  const archive = async () => {
    if (!current || busy || ownerArchived) return;
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
      const readConfig = (id: string) => speech
        ? ttsApi.versionConfig(project.id, id).then(response => response.config)
        : apiClient.get(versionConfigUrl(project.id, id), { silent: true });
      const [before, after] = await Promise.all([readConfig(compareTo), readConfig(current.id)]);
      setComparison(configDifferences(before, after));
    } catch (error) { setError(formatApiError(error)); }
    finally { setBusy(false); }
  };
  const formatValue = (value: unknown) => value === undefined ? '—' : JSON.stringify(value);
  const problem = error || (loadError ? formatApiError(loadError) : '');
  const versionState = (item: ProjectVersion) => item.archived ? text('已归档', 'Archived') : item.status === 'copying' ? text('创建中', 'Creating') : item.status === 'failed' ? text('失败', 'Failed') : item.busy ? text('使用中', 'In use') : text('就绪', 'Ready');
  const title = customTitle ?? (active === 'overview' ? text('项目概览', 'Project overview') : active === 'data' ? text('训练数据', 'Training data') : active === 'train' ? text('训练参数', 'Training parameters') : text('训练结果', 'Training results'));
  const audioSummary = (version?: ProjectVersion) => {
    const summary = version?.audio_stats?.train;
    if (!summary) return text('训练音频数量未知', 'Training clip count unknown');
    if (summary.state === 'missing') return text('未登记训练音频', 'No training audio registered');
    if (summary.state === 'valid' && summary.clips_count != null) return text(`${summary.clips_count} 段训练音频`, `${summary.clips_count} training clips`);
    const states = {
      unchecked: text('训练音频待检查', 'Training audio not checked'),
      checking: text('正在检查训练音频', 'Checking training audio'),
      valid: text('训练音频数量未知', 'Training clip count unknown'),
      invalid: text('训练音频存在问题', 'Training audio has issues'),
      stale: text('训练音频需重新检查', 'Training audio needs rechecking'),
      error: text('训练音频检查失败', 'Training audio check failed'),
    };
    return states[summary.state];
  };
  const engine = current?.engine || project.active_engine;
  const engineLabel = `${ttsEngineLabel(engine)}${speech && current?.variant ? ` · ${current.variant}` : ''}`;
  const projectControls = <section className="project-sidebar" aria-label={text('当前项目工作区', 'Current project workspace')}>
    <Link className="project-sidebar-identity" to={projectUrl(project.id, selectedId, 'overview')} title={project.note || project.name}><strong title={project.name}>{project.name}</strong><small title={project.id}>{project.id}</small></Link>
    {supported && <>
      <div className="project-sidebar-version-heading"><span>{text('当前版本', 'Current version')}</span>{current && <span className={`project-sidebar-version-state status-${current.status}`} title={current.error || current.note || versionState(current)}>{current.status === 'copying' && <Loader2 size={11} className="animate-spin"/>}{versionState(current)}</span>}</div>
      <StudioSelect searchable aria-label={text('项目版本', 'Project versions')} value={selectedId || ''} disabled={busy || !versions.length} icon={<GitBranch size={13}/>} onValueChange={value => void switchVersion(value)} options={versions.filter(item => showArchived || !item.archived || item.id === selectedId).map(item => ({value:item.id,label:`${item.name}${item.archived || item.status !== 'ready' ? ` · ${versionState(item)}` : ''}`}))}/>
      <div className="project-sidebar-model">{text('模型类型', 'Model type')} · {speech ? engineLabel : familyLabel(current?.display_family ?? current?.family)}</div>
      {speech && <div className="project-sidebar-model">{audioSummary(current)}</div>}
      <div className="project-sidebar-actions" aria-label={text('版本操作', 'Version actions')}>
        <button type="button" className="ui-btn ui-btn-icon" onClick={() => void begin('create')} disabled={busy || ownerArchived || versions.some(item => item.status === 'copying')} title={ownerArchivedReason || text('新版本', 'New version')} aria-label={text('新版本', 'New version')}><Plus size={14}/></button>
        <button type="button" className="ui-btn ui-btn-icon" onClick={() => void begin('compare')} disabled={!current || readyVersions.length < 2 || busy} title={text('比较版本参数', 'Compare version parameters')} aria-label={text('比较', 'Compare')}><GitCompare size={14}/></button>
        <button type="button" className="ui-btn ui-btn-icon" onClick={() => void begin('paths')} disabled={!current || busy} title={text('查看本版本目录', 'View version folders')} aria-label={text('查看本版本目录', 'View version folders')}><FolderOpen size={14}/></button>
        <button type="button" className="ui-btn ui-btn-icon" onClick={() => void begin('edit')} disabled={!current || current.status === 'copying' || busy || ownerArchived} title={ownerArchivedReason || text('版本设置', 'Version settings')} aria-label={text('版本设置', 'Version settings')}><Settings2 size={14}/></button>
      </div>
      {versions.some(item => item.archived) && <Switch className="project-sidebar-archived studio-switch-small" checked={showArchived} onCheckedChange={setShowArchived}>{text('显示已归档版本', 'Show archived versions')}</Switch>}
    </>}
    <ProjectWorkflow projectId={project.id} versionId={selectedId} projectType={project.project_type} active={workflowActive ? active : undefined} sidebar/>
  </section>;
  return <>
    {sidebarOnly ? projectControls : sidebar?.register ? null : sidebar ? sidebar.target && createPortal(projectControls, sidebar.target) : <div className="project-sidebar-fallback">{projectControls}</div>}
    {!sidebarOnly && <header className="workspace-navigation workspace-page-heading" ref={navigationRef}><div className="workspace-heading-main"><TopbarBreadcrumb><nav className="workspace-breadcrumb" aria-label={text('当前位置', 'Current location')}><Link to="/projects">{text('项目', 'Projects')}</Link><span aria-hidden="true">/</span><Link to={projectUrl(project.id, selectedId, 'overview')}>{project.name}</Link>{current && <><span aria-hidden="true">/</span><span>{current.name}</span></>}{breadcrumbTrail && <><span aria-hidden="true">/</span>{breadcrumbTrail}</>}</nav></TopbarBreadcrumb><div className="workspace-heading-title"><h1 title={title}>{title}</h1>{titleBadge}</div></div>{status && <div className="project-heading-status">{status}</div>}</header>}
    {problem && !dialog && <div role="alert" className="workspace-message error">{problem}<button type="button" className="ui-btn ui-btn-sm" onClick={() => {setError(''); void refresh();}}>{text('重试', 'Retry')}</button></div>}
    {!sidebarOnly && ownerArchived && <div className="workspace-message" role="status"><AlertCircle size={16}/><div><strong>{text('此项目已归档 · 只读', 'This project is archived · Read only')}</strong><p>{text('请在项目列表恢复项目后继续修改。', 'Restore the project from the project list to make changes.')}</p></div></div>}
    {!sidebarOnly && !ownerArchived && current?.archived && <div className="workspace-message" role="status"><AlertCircle size={16}/><div><strong>{text('此版本已归档 · 只读', 'This version is archived · Read only')}</strong><p>{text('恢复版本后可继续编辑与训练。', 'Restore the version to edit or train.')}</p></div><button type="button" className="ui-btn ui-btn-sm" disabled={busy} onClick={() => void archive()}>{busy ? text('正在恢复…', 'Restoring…') : text('恢复版本', 'Restore version')}</button></div>}
    {!sidebarOnly && current?.status === 'copying' && <div className="workspace-message" role="status"><Loader2 size={16} className="animate-spin"/><div><strong>{text('正在建立独立版本', 'Creating an independent version')}</strong><p>{speech ? text('正在准备版本文件，完成后即可编辑；原版本保持不变。', 'Preparing version files. The original version is preserved.') : text('复制图片、标签与遮罩，完成后即可编辑；原版本保持不变。', 'Copying images, captions and masks. The original version is preserved.') }</p><progress max={Math.max(1,current.progress?.files_total || 0)} value={current.progress?.files_done || 0}/><span>{current.progress?.files_done || 0} / {current.progress?.files_total || '…'} {text('个文件', 'files')}</span></div></div>}
    {!sidebarOnly && current?.status === 'failed' && <div className="workspace-message error" role="alert"><AlertCircle size={16}/><div><strong>{text('版本准备失败', 'Version preparation failed')}</strong><p>{current.error}</p><p>{text('原版本数据仍然保留。修正原因后，可从原版本重新创建。', 'Original data is preserved. Resolve the issue and create again from the source version.')}</p></div></div>}
    {dialog && <Dialog title={dialog === 'create' ? text('新建实验版本', 'New experiment version') : dialog === 'edit' ? text('版本设置', 'Version settings') : dialog === 'compare' ? text('比较版本', 'Compare versions') : text('本版本的文件位置', 'Files in this version')} onClose={() => { if (!busy) {setDialog(null);setError('');} }} closeDisabled={busy} wide={dialog === 'compare' || dialog === 'paths'}>
      {error && <div role="alert" className="workspace-message error">{error}</div>}
      {ownerArchived && (dialog === 'create' || dialog === 'edit') && <p role="status" className="version-copy-note">{ownerArchivedReason}</p>}
      {(dialog === 'create' || dialog === 'edit') && <form onSubmit={save} className="version-form"><label>{text('版本名称', 'Version name')}<input value={name} onChange={event => setName(event.target.value)} maxLength={120} required disabled={busy || ownerArchived || dialog === 'edit' && !!current?.archived}/></label><label>{text('实验说明', 'Experiment notes')}<textarea value={note} onChange={event => setNote(event.target.value)} placeholder={speech ? text('例如：调整录音数据，学习率设为 1e-4', 'For example: update recordings, learning rate 1e-4') : text('例如：仅训练服装区域，学习率调整为 0.0002', 'For example: train clothing only, learning rate 0.0002')} disabled={busy || ownerArchived || dialog === 'edit' && !!current?.archived}/></label>
        {dialog === 'edit' && !speech && <label>{text('模型类型', 'Model type')}<StudioSelect aria-label={text('模型类型', 'Model type')} value={family} onValueChange={setFamily} disabled={busy || !!current?.archived} options={familyOptions}/></label>}
        {dialog === 'edit' && speech && <>
          <label>{text('模型类型', 'Model type')}<StudioSelect aria-label={text('模型类型', 'Model type')} value={engineConfig ? nextEngine : current?.engine || ''} onValueChange={value => setNextEngine(value as TtsEngine)} disabled={busy || ownerArchived || !!current?.archived || !!current?.busy || current?.status !== 'ready' || !engineConfig || !engineOptions.length} options={engineOptions}/></label>
          {engineConfig && nextEngine === 'gpt-sovits-v5' && <label>{text('模型变体', 'Model variant')}<StudioSelect aria-label={text('模型变体', 'Model variant')} value={nextVariant} onValueChange={value => setNextVariant(value as 'v5dev' | 'v5turbo')} disabled={busy || ownerArchived || !!current?.archived || !!current?.busy || current?.status !== 'ready'} options={[{ value: 'v5dev', label: 'v5dev' }, { value: 'v5turbo', label: 'v5turbo' }]}/></label>}
          {engineChanged && <p className="version-family-note">{nextEngine !== engineConfig?.config.engine ? text('更换模型类型会重建训练参数并清空环境与模型路径。音频文件、数据登记和已有训练记录保留，清单需要重新检查。', 'Changing model type resets training parameters, environment paths and model paths. Audio files, registered sources and existing jobs are preserved. Recheck the manifests afterwards.') : text('更换模型变体会清空基础模型与预训练权重路径，保留训练参数和数据。', 'Changing the variant clears base-model and pretrained-weight paths. Training parameters and data are preserved.')}</p>}
          {(!engineConfig || engineConflict) && <p className="version-family-note" role="status">{engineConflict ? text('版本参数已变化。请重新读取后核对所选模型类型，再保存。', 'Version parameters changed. Reload them, review the selected model type, then save.') : text('读取版本参数后可更换模型类型。', 'Load version parameters to change model type.')} <button type="button" className="ui-link" disabled={busy} onClick={() => void reloadEngine()}>{text('重新读取参数', 'Reload parameters')}</button></p>}
          {capabilities.error && <p className="version-family-note" role="status">{text('无法读取模型类型。', 'Could not load model types.')} <button type="button" className="ui-link" disabled={busy} onClick={() => void capabilities.refetch()}>{text('重试', 'Retry')}</button></p>}
        </>}
        {dialog === 'create' && <><label>{text('创建来源', 'Create from')}<StudioSelect aria-label={text('创建来源', 'Create from')} value={source} onValueChange={value => { setSource(value);setFamily(versions.find(item => item.id === value)?.family || ''); setCreateEngine(''); setCreateVariant(''); if (hasUncopyableAudio(versions.find(item => item.id === value))) setMode('empty'); }} disabled={busy || ownerArchived} options={[{value:'',label:text('默认配置 · 空白版本', 'Default configuration · blank version')},...copySources.map(item => ({value:item.id,label:item.name}))]}/></label>
          {speech && <><label>{text('训练模型类型', 'Training model type')}<StudioSelect aria-label={text('训练模型类型', 'Training model type')} value={source ? createEngine : createTargetEngine} onValueChange={value => { setCreateEngine(value as TtsEngine | ''); setCreateVariant(''); }} disabled={busy || ownerArchived} options={[...(source ? [{ value: '', label: text(`沿用来源版本 · ${ttsEngineLabel(sourceVersion?.engine)}`, `Inherit source · ${ttsEngineLabel(sourceVersion?.engine)}`) }] : []), ...engineOptions]}/></label>
            {createTargetEngine === 'gpt-sovits-v5' && <label>{text('模型变体', 'Model variant')}<StudioSelect aria-label={text('模型变体', 'Model variant')} value={sourceVersion?.engine === 'gpt-sovits-v5' && !changedSourceEngine ? createVariant : createTargetVariant} onValueChange={value => setCreateVariant(value as typeof createVariant)} disabled={busy || ownerArchived} options={[...(sourceVersion?.engine === 'gpt-sovits-v5' && !changedSourceEngine ? [{ value: '', label: text(`沿用来源版本 · ${sourceVersion.variant || 'v5dev'}`, `Inherit source · ${sourceVersion.variant || 'v5dev'}`) }] : []), { value: 'v5dev', label: 'v5dev' }, { value: 'v5turbo', label: 'v5turbo' }]}/></label>}
            {(changedSourceEngine || changedSourceVariant || !source) && <p className="version-family-note">{changedSourceEngine ? text('新版本使用目标模型的默认训练参数，并清空环境与模型路径。复制数据后会按目标模型重新检查，不兼容时新版本会创建失败；也可选择重新准备数据。来源版本保持不变。', 'The new version uses target-model defaults with empty environment and model paths. Copied data is checked for the target model; incompatible data makes creation fail. You can start with empty data instead. The source version is preserved.') : changedSourceVariant ? text('新版本保留训练参数与环境，清空基础模型和预训练权重路径。', 'The new version keeps training parameters and environment paths, and clears base-model and pretrained-weight paths.') : text('使用所选模型的默认配置。', 'Uses the selected model’s defaults.')}</p>}
            {capabilities.error && <p className="version-family-note" role="status">{text('模型类型列表暂时无法读取，仍可沿用来源版本配置。', 'Model types are unavailable. The source configuration can still be inherited.')} <button type="button" className="ui-link" disabled={busy} onClick={() => void capabilities.refetch()}>{text('重试', 'Retry')}</button></p>}
          </>}
          {!speech && <><label>{text('训练模型类型', 'Training model type')}<StudioSelect aria-label={text('训练模型类型', 'Training model type')} value={family || (source ? '' : 'anima')} onValueChange={setFamily} disabled={busy} options={[...(source ? [{value:'',label:text(`沿用来源版本 · ${familyLabel(sourceFamily)}`, `Inherit source · ${familyLabel(sourceFamily)}`)}] : []),...familyOptions]}/></label>
          <p className="version-family-note">{incompatibleFamily ? text('更换模型类型会重置模型路径、恢复权重和训练参数。', 'Changing model type resets model paths, resume weights and training settings.') : text('使用所选模型的默认配置。', 'Uses the selected model’s defaults.')}</p>
          {familiesError && <p className="version-family-note" role="status">{text('模型类型列表暂时无法读取，仍可沿用来源版本配置。', 'Model types are unavailable. The source configuration can still be inherited.')}</p>}</>}
          {source && <fieldset><legend>{text('继承内容', 'Include')}</legend><label><input type="radio" name="version-mode" checked={mode === 'copy'} onChange={() => setMode('copy')} disabled={busy || ownerArchived || audioCopyUnavailable}/><div><strong>{incompatibleFamily ? text('复制数据，使用新模型配置', 'Copy data with new model defaults') : text('配置和完整数据副本', 'Configuration and a complete data copy')}</strong><p>{speech ? audioCopyUnavailable ? text('已登记的音频清单须全部检查通过；也可仅复制配置。', 'All registered audio manifests must pass their checks. Configuration-only copying is also available.') : text('复制清单、音频及参考音频，后续修改互不影响。', 'Copy manifests, audio and reference audio into an independent version.') : text('复制图片、标签、遮罩和验证集，后续修改互不影响。', 'Independent images, captions, masks and validation data. Later edits do not affect the source.')}</p></div></label><label><input type="radio" name="version-mode" checked={mode === 'empty'} onChange={() => setMode('empty')} disabled={busy || ownerArchived}/><div><strong>{incompatibleFamily ? text('使用新模型配置，重新准备数据', 'New model defaults and new data') : text('仅配置，重新准备数据', 'Configuration only, prepare new data')}</strong><p>{incompatibleFamily ? text('按所选模型类型重建默认参数，清空数据来源。', 'Initialize defaults for the selected model and start with no data sources.') : text('保留实验参数，清空数据来源。', 'Keep experiment parameters and start with no data sources.')}</p></div></label></fieldset>}<p className="version-copy-note"><Copy size={14}/>{speech ? text('训练记录、检查点和试听音频不会复制到新版本。', 'Jobs, checkpoints and preview audio stay in the original version.') : text('训练记录、采样图和产物不会复制到新版本。', 'Jobs, generated samples and outputs stay in the original version.')}</p></>}
        <footer>{dialog === 'edit' && <button type="button" className="ui-btn" onClick={() => void archive()} disabled={busy || ownerArchived || versions.filter(item => !item.archived && item.status === 'ready').length < 2 && !current?.archived}>{current?.archived ? text('恢复版本', 'Restore version') : text('归档版本', 'Archive version')}</button>}<span/><button type="button" className="ui-btn" disabled={busy} onClick={() => setDialog(null)}>{text('取消', 'Cancel')}</button><button type="submit" className="ui-btn ui-btn-primary" disabled={busy || ownerArchived || !name.trim() || dialog === 'create' && createEngineUnavailable || dialog === 'edit' && (!!current?.archived || speech && (engineConflict || engineWriteBlocked))}>{busy && <Loader2 size={14} className="animate-spin"/>}{dialog === 'create' ? text('创建版本', 'Create version') : text('保存', 'Save')}</button></footer>
      </form>}
      {dialog === 'paths' && current && <div className="version-paths"><p>{text('重命名不改变项目 ID 或文件目录。', 'Renaming does not change the project ID or folders.')}</p><dl>{[
        {key:'root',label:text('本版本目录', 'Version folder'),path:current.paths.root},
        {key:'traindata',label:text('训练数据', 'Training data'),path:current.paths.traindata || current.paths.datasets},
        {key:'reg',label:text('正则图', 'Regularization images'),path:speech ? null : current.paths.reg},
        {key:'samples',label:speech ? text('试听音频', 'Preview audio') : text('采样图', 'Samples'),path:current.paths.samples},
        {key:'output',label:text('模型产物', 'Model outputs'),path:current.paths.output || current.paths.runs},
        {key:'jobs',label:text('任务记录', 'Job records'),path:current.paths.jobs},
        {key:'config',label:text('参数草稿', 'Configuration draft'),path:current.paths.config},
        {key:'cache',label:text('编码缓存', 'Encoding cache'),path:speech ? null : current.paths.cache},
      ].filter(item=>item.path).map(item=><div key={item.key}><dt>{item.label}</dt><dd>{item.path}</dd></div>)}</dl><div className="version-tree">{text('每次训练按任务 ID 分开保存：','Each training job has its own ID subfolder:')}<br/>{current.paths.output ? 'output' : 'runs'}/{'<job_id>'}/ · {text('模型产物（LoRA 与模型权重）','model outputs (LoRA and model weights)')}<br/>jobs/{'<job_id>'}/ · {speech ? text('任务配置、日志和事件', 'job configuration, logs and events') : text('训练配置、日志和事件；恢复点在其中的 resume/','configuration, logs and events; resume points in its resume/')}<br/>samples/{'<job_id>'}/ · {speech ? text('试听音频', 'preview audio') : text('采样图片','sample images')}</div></div>}
      {dialog === 'compare' && current && <div className="version-comparison"><div className="comparison-controls"><label>{text('对比版本', 'Compare from')}<StudioSelect aria-label={text('对比版本', 'Compare from')} value={compareTo} disabled={busy} onValueChange={value => { setCompareTo(value);setComparison(null); }} options={readyVersions.filter(item => item.id !== current.id).map(item => ({value:item.id,label:item.name}))}/></label><span>→ {current.name}</span><button type="button" className="ui-btn" disabled={busy || !compareTo} onClick={() => void compare()}>{busy ? <><Loader2 size={14} className="animate-spin" aria-hidden="true"/>{text('读取中…', 'Loading…')}</> : text('查看差异', 'Show differences')}</button></div><div className="version-comparison-stats">{[versions.find(item => item.id === compareTo),current].filter(Boolean).map(item => <div key={item!.id}><strong>{item!.name}</strong><span>{speech ? audioSummary(item!) : `${item!.stats.images} ${text('张图片', 'images')}`} · {item!.stats.jobs} {speech ? text('个任务', 'jobs') : text('次训练', 'jobs')}{!speech && <> · {item!.stats.artifacts} {text('个产物', 'outputs')}</>}</span></div>)}</div>{comparison && (comparison.length ? <div className="comparison-table-wrap"><table><thead><tr><th>{text('参数', 'Parameter')}</th><th>{versions.find(item => item.id === compareTo)?.name}</th><th>{current.name}</th></tr></thead><tbody>{comparison.map(item => <tr key={item.path}><th>{item.path}</th><td>{formatValue(item.before)}</td><td>{formatValue(item.after)}</td></tr>)}</tbody></table></div> : <p>{text('两个版本的参数相同。', 'The configurations are identical.')}</p>)}</div>}
    </Dialog>}
  </>;
}
