import { projectUrl, type ProjectVersion, type VersionedProject } from '../../utils/projectVersions';
import React from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { Link, useParams, useNavigate, useLocation, UNSAFE_DataRouterContext } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { apiClient, apiUrl } from '../../api/client';
import { DatasetInfo, DatasetImage } from '../../api/types';
import { useDatasetImages } from '../../api/hooks/useDatasetImages';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { MaskEditor } from '../../components/masks/MaskEditor';
import { formatApiError } from '../../utils/errors';
import { TagChips } from '../../components/TagChips';
import StructuredCaptionEditor from '../../components/datasets/StructuredCaptionEditor';
import { getCaptionStructure, captionFieldChanges, type CaptionFieldDraft } from '../../utils/captionStructure';
import { formatBytes, formatParams } from '../../utils/format';
import '../../styles/project-workspace.css';
import './dataset-workspace.css';
import ProjectDataImport from '../ProjectDetail/ProjectDataImport';
import DatasetImagePane from '../../components/datasets/DatasetImagePane';
import ProgressBar from '../../components/ProgressBar';
import ProjectWorkspaceHeader from '../../components/projects/ProjectWorkspaceHeader';
import DatasetNavigationGuard from '../../components/datasets/DatasetNavigationGuard';
import { useWorkspaceText } from '../../utils/workspaceText';
import { datasetReturnTarget } from '../../utils/datasetReturn';
import TopbarBreadcrumb from '../../components/TopbarBreadcrumb';
import {
  RefreshCcw,
  Trash2,
  Search,
  X,
  Brush,
  ArrowLeft,
} from 'lucide-react';

export default function Dataset() {
  const { id } = useParams<{ id: string }>();
  return <DatasetWorkspace key={id || ''} id={id}/>;
}
export function DatasetWorkspace({id}: {id?:string}) {
  const navigate = useNavigate();
  const location = useLocation();
  const hasDataRouter = !!React.useContext(UNSAFE_DataRouterContext);
  const { t } = useTranslation();
  const text = useWorkspaceText();
  const queryClient = useQueryClient();
  const contextProblem = text('无法确认此数据集所属的项目版本。','The dataset project/version could not be verified.');

  // locales 占位符为单花括号（{n}），i18next 默认插值（{{}}）不处理，需手工替换
  const tt = React.useCallback(
    (key: string, vars: Record<string, string | number>) =>
      Object.entries(vars).reduce((s, [k, v]) => s.split(`{${k}}`).join(String(v)), t(key)),
    [t]
  );

  const [info, setInfo] = React.useState<DatasetInfo | null>(() => queryClient.getQueriesData<DatasetInfo[]>({ queryKey: ['project-workspace-datasets'] }).flatMap(([, rows]) => rows || []).find(row => row.source.id === id && row.stats && row.index_status) || null);
  const [infoError, setInfoError] = React.useState('');
  const [projectContext, setProjectContext] = React.useState<{project:VersionedProject;versions:ProjectVersion[];current?:ProjectVersion}|null>(() => {
    const source = info?.source;
    if (!source?.project_id) return null;
    const project = queryClient.getQueryData<VersionedProject>(['project', source.project_id]);
    const versions = queryClient.getQueryData<ProjectVersion[]>(['project-versions', source.project_id]) || [];
    const current = versions.find(version => version.id === source.version_id && version.project_id === source.project_id);
    return project?.id === source.project_id && (!source.version_id || current) ? { project, versions, current } : null;
  });
  const [contextError, setContextError] = React.useState('');
  const contextRequest = React.useRef<AbortController|null>(null);
  const [indexProgress, setIndexProgress] = React.useState<{ done: number; total: number } | null>(null);
  const [editorImage, setEditorImage] = React.useState<DatasetImage | null>(null);
  const [activeImage, setActiveImage] = React.useState<string | null>(null);
  const [maskImage, setMaskImage] = React.useState<{ hash: string; relPath: string } | null>(null);
  const [actionError, setActionError] = React.useState('');
  const [editCaption, setEditCaption] = React.useState('');
  const [captionFields, setCaptionFields] = React.useState<CaptionFieldDraft>({});
  const [captionPath, setCaptionPath] = React.useState('');
  const [captionBase, setCaptionBase] = React.useState('');
  const [savingCaption, setSavingCaption] = React.useState(false);
  const captionSave = React.useRef<Promise<void>|null>(null);
  const [batchAdd, setBatchAdd] = React.useState('');
  const [batchRemove, setBatchRemove] = React.useState('');
  const [thumbnailWidth, setThumbnailWidth] = React.useState(140);
  const [showBatch, setShowBatch] = React.useState(false);
  const [showAddImages,setShowAddImages] = React.useState(false);
  const [addingImages,setAddingImages] = React.useState(false);
  const [folderName, setFolderName] = React.useState('');
  const [repeats, setRepeats] = React.useState('1');
  const savedFolderName = (info?.source.path || '').replace(/\\/g,'/').replace(/\/+$/,'').split('/').pop() || '';
  const savedRepeats = info?.source.repeats;
  React.useEffect(() => {setFolderName(savedFolderName); if(savedRepeats !== undefined)setRepeats(String(savedRepeats));}, [savedFolderName, savedRepeats]);
  const [busyAction, setBusyAction] = React.useState<string | null>(null);
  const [versionAccess, setVersionAccess] = React.useState<{ key: string; editable: boolean; archived: boolean; error?: string } | null>(null);
  const versionRequest = React.useRef<AbortController | null>(null);
  const versionKey = info?.source.version_id ? `${info.source.project_id}/${info.source.version_id}` : '';
  const canEdit = !!info && !infoError && (!versionKey || versionAccess?.key === versionKey && versionAccess.editable);
  const checkVersionAccess = React.useCallback(async () => {
    if (!versionKey) return;
    versionRequest.current?.abort(); const controller = new AbortController(); versionRequest.current = controller;
    try {
      const version = await apiClient.get<{ status: string; archived: boolean; busy?: boolean }>(`/projects/${encodeURIComponent(info!.source.project_id || '')}/versions/${encodeURIComponent(info!.source.version_id!)}`, { signal: controller.signal, silent: true });
      if (!controller.signal.aborted) setVersionAccess({ key: versionKey, editable: version.status === 'ready' && !version.archived && !version.busy, archived: version.archived });
    } catch (error) { if (!controller.signal.aborted) setVersionAccess({ key: versionKey, editable: false, archived: false, error: formatApiError(error) }); }
  }, [versionKey, info]);
  React.useEffect(() => {
    void checkVersionAccess(); window.addEventListener('focus', checkVersionAccess);
    return () => { versionRequest.current?.abort(); window.removeEventListener('focus', checkVersionAccess); };
  }, [checkVersionAccess]);

  const images = useDatasetImages(id, 60, 'all');
  const activeImg = activeImage ? editorImage : undefined;
  const activeStructure = getCaptionStructure(activeImg ?? undefined);
  const activeJson = activeImg?.caption_format?.toLowerCase().replace(/^\./, '') === 'json';
  const changedFields = captionFieldChanges(activeStructure, captionFields);
  const captionDirty = editCaption !== captionBase || changedFields.length > 0;
  const captionLocked = !canEdit || savingCaption || !!activeImg?.caption_error || activeJson && !activeStructure?.editable;
  const fetchInfo = React.useCallback(() => {
    if (!id) return;
    apiClient.get<DatasetInfo>(`/datasets/${id}`, {silent:true,params:{include_cache:false}}).then((data) => {
      setInfo(data); setInfoError('');
      if (data.index_status !== 'indexing') setIndexProgress(null);
    }).catch(error => {setInfoError(formatApiError(error));setProjectContext(null);});
  }, [id]);

  React.useEffect(() => {
    fetchInfo();
  }, [fetchInfo]);
  const refreshProjectContext = React.useCallback(async () => {
    const source=info?.source;
    if (!source?.project_id || infoError) return;
    contextRequest.current?.abort();const controller=new AbortController();contextRequest.current=controller;
    try {
      const [project,versions]=await Promise.all([
        apiClient.get<VersionedProject>(`/projects/${source.project_id}`,{silent:true,signal:controller.signal}),
        source.version_id ? apiClient.get<ProjectVersion[]>(`/projects/${source.project_id}/versions`,{silent:true,signal:controller.signal,params:{include_archived:true}}) : Promise.resolve([]),
      ]);
      if(controller.signal.aborted)return;
      const current=versions.find(version=>version.id===source.version_id && version.project_id===source.project_id);
      if(project.id!==source.project_id || source.version_id && !current) throw new Error(contextProblem);
      setProjectContext({project,versions,current});setContextError('');
      if(current)setVersionAccess({key:`${source.project_id}/${source.version_id}`,editable:current.status==='ready' && !current.archived && !current.busy,archived:current.archived});
    } catch(error){if(!controller.signal.aborted){setProjectContext(null);setContextError(formatApiError(error));}}
  },[info?.source,infoError,contextProblem]);
  React.useEffect(()=>{void refreshProjectContext();return()=>contextRequest.current?.abort();},[refreshProjectContext]);

  // 索引完成 / caption 修改 / rescan 完成 → 重新拉取数据集信息
  useEventStream(EVENT_TYPES.DATASET_CHANGED, (data: any) => {
    if (data.dataset_id === id) {
      fetchInfo();
      images.refresh();
    }
  });

  // 索引进度（kind=index 时 job_id 实为 dataset_id）
  useEventStream(EVENT_TYPES.JOB_CACHE_PROGRESS, (data: any) => {
    if (data.kind === 'index' && data.job_id === id) {
      setIndexProgress({ done: data.done, total: data.total });
    }
  });

  const updateSettings = async (changes: {name?: string; repeats?: number; masked_loss?: boolean}) => {
    if (!id || !canEdit) return;
    setBusyAction('settings');setActionError('');
    try {const updated=await apiClient.patch<DatasetInfo>(`/datasets/${id}`,changes);setInfo(updated);images.refresh();}
    catch(error){setActionError(formatApiError(error));throw error;}
    finally{setBusyAction(null);}
  };
  const enableMaskedTraining = () => updateSettings({masked_loss:true});

  const openEditor = (hash: string, relPath: string) => {
    const img = images.items.find((i) => i.hash === hash && i.rel_path === relPath);
    if (!img) return;
    setEditorImage(img); setActiveImage(hash); setCaptionPath(relPath); setCaptionFields({});
    setEditCaption(img.caption_tags ?? img.caption ?? '');
    setCaptionBase(img.caption_tags ?? img.caption ?? '');
  };

  const saveCaption = async () => {
    if (captionSave.current) return captionSave.current;
    if (!activeImage || !id || !activeImg || captionLocked) throw new Error(text('当前图片只读，无法保存标签。','The image is read only; its caption cannot be saved.'));
    setSavingCaption(true);setActionError('');
    const pending=apiClient.put(`/datasets/${id}/images/${activeImage}/caption`, activeJson ? { caption_fields: changedFields, caption_revision: activeStructure!.revision } : { caption: editCaption },{params:{rel_path:activeImg.rel_path},silent:true})
      .then(() => {
        if (activeJson) images.refresh(); else images.updateCaption(activeImage, editCaption, activeImg.rel_path);
        setCaptionFields({});
        setCaptionBase(editCaption);
        setActiveImage(null);
      })
      .catch(error=>{setActionError(formatApiError(error));throw error;})
      .finally(() => {setSavingCaption(false);captionSave.current=null;});
    captionSave.current=pending;return pending;
  };
  const closeCaption = () => {if(!savingCaption && (!captionDirty || window.confirm(text('标签尚未保存，确定放弃这些修改？','Discard the unsaved caption changes?'))))setActiveImage(null);};
  const beforeNavigation = async () => {
    if(addingImages)throw new Error(text('图片正在添加，请完成后再离开。','Images are being added. Wait for completion before leaving.'));
    if(maskImage)throw new Error(text('请先在遮罩编辑器中保存或关闭，再切换项目页面。','Save or close the mask editor before switching project pages.'));
    if(activeImage && captionDirty || captionSave.current)await saveCaption();
  };
  const leaveRef=React.useRef({beforeNavigation,dirty:false});
  React.useLayoutEffect(()=>{leaveRef.current={beforeNavigation,dirty:addingImages || !!maskImage || !!activeImage && captionDirty || savingCaption};});
  React.useEffect(()=>{
    let leaving=false;
    const click=(event:MouseEvent)=>{
      if(hasDataRouter)return;
      if(!leaveRef.current.dirty || event.defaultPrevented || event.button!==0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey)return;
      const anchor=event.target instanceof Element?event.target.closest<HTMLAnchorElement>('a[href]'):null;
      if(!anchor || anchor.hasAttribute('download') || anchor.target && anchor.target!=='_self')return;
      const next=new URL(anchor.href,window.location.href);if(next.origin!==window.location.origin)return;
      event.preventDefault();if(leaving)return;leaving=true;
      const settings = next.pathname === '/settings' || next.pathname.startsWith('/settings/');
      void leaveRef.current.beforeNavigation().then(()=>navigate(`${next.pathname}${next.search}${next.hash}`, settings ? {state:{backgroundLocation:location}} : undefined)).catch(error=>setActionError(formatApiError(error))).finally(()=>{leaving=false;});
    };
    const unload=(event:BeforeUnloadEvent)=>{if(leaveRef.current.dirty){event.preventDefault();event.returnValue='';}};
    document.addEventListener('click',click,true);window.addEventListener('beforeunload',unload);
    return()=>{document.removeEventListener('click',click,true);window.removeEventListener('beforeunload',unload);};
  },[navigate,location,hasDataRouter]);

  const applyBatchTags = () => {
    if (!id || !canEdit || images.selected.size === 0) return;
    const add = batchAdd.split(',').map((t) => t.trim()).filter(Boolean);
    const remove = batchRemove.split(',').map((t) => t.trim()).filter(Boolean);
    if (add.length === 0 && remove.length === 0) return;
    setBusyAction('batch');
    apiClient
      .post(`/datasets/${id}/tags/batch`, { hashes: images.items.filter(image=>images.selected.has(image.rel_path)).map(image=>image.hash), rel_paths: [...images.selected], add, remove })
      .then(() => {
        images.refresh();
        images.clearSelection();
        setBatchAdd('');
        setBatchRemove('');
      })
      .catch(error => setActionError(formatApiError(error)))
      .finally(() => setBusyAction(null));
  };

  const handleRescan = () => {
    if (!id || !canEdit) return;
    setBusyAction('rescan');
    apiClient.post(`/datasets/${id}/rescan`, {})
      .then(fetchInfo)
      .catch(error => setActionError(formatApiError(error)))
      .finally(() => setBusyAction(null));
  };

  const handleDelete = () => {
    if (!id || !info || !canEdit) return;
    if (window.confirm(tt('dataset.deleteConfirm', { path: info.source.path }))) {
      setBusyAction('delete');
      apiClient.delete(`/datasets/${id}`)
        .then(() => navigate(returnUrl))
        .catch(error => setActionError(formatApiError(error)))
        .finally(() => setBusyAction(null));
    }
  };

  const statusLabel = (status?: string): string => {
    switch (status) {
      case 'ready':
        return t('dataset.statusReady', '已就绪');
      case 'indexing':
        return t('dataset.statusIndexing', '索引中');
      case 'failed':
        return t('dataset.statusFailed', '索引失败');
      case 'stale':
        return t('dataset.statusStale', '索引已过期');
      default:
        return status || '--';
    }
  };

  const stats = info?.stats;
  const activeSize = activeImg?.size;
  const datasetName = info?.source.path.replace(/[\\/]+$/, '').split(/[\\/]/).pop()?.replace(/^(?:d_[0-9a-f]+-)+/i, '') || id;
  const returnQuery = new URLSearchParams(location.search);
  // Metadata remains authoritative after a direct visit or refresh; query context
  // keeps the return link usable while the dataset request is still loading.
  const returnProject = info?.source.project_id || returnQuery.get('project');
  const returnVersion = info?.source.project_id ? info.source.version_id : returnQuery.get('version');
  const libraryUrl = returnProject ? `${projectUrl(returnProject, returnVersion, 'data')}&data_step=datasets#version-datasets` : '/projects';
  const returnUrl = datasetReturnTarget(location.state?.datasetReturnTo, returnProject, returnVersion) || libraryUrl;
  const loadingWorkspace = !infoError && (!info || !!info.source.project_id && !projectContext && !contextError);
  const curationUrl = info?.source.project_id && info.source.version_id ? `${projectUrl(info.source.project_id, info.source.version_id, 'data')}&data_step=curate&dataset=${encodeURIComponent(id || '')}` : '';

  return (
    <div className="dataset-workspace" data-testid="dataset-page">
      {hasDataRouter && <DatasetNavigationGuard shouldBlock={() => {
        return leaveRef.current.dirty;
      }} beforeLeave={() => leaveRef.current.beforeNavigation()} onError={error => setActionError(formatApiError(error))}/>}
      <div className="dataset-workspace-navigation">
        <Link className="ui-btn dataset-back" to={returnUrl}><ArrowLeft size={16}/>{text('返回', 'Back')}</Link>
        {projectContext && !infoError ? <ProjectWorkspaceHeader project={projectContext.project} versionId={info?.source.version_id || undefined} versions={projectContext.versions} current={projectContext.current} active="data" title={datasetName} breadcrumbTrail={<Link to={libraryUrl}>{text('训练数据', 'Training data')}</Link>} refresh={refreshProjectContext} beforeAction={beforeNavigation}/> : <header className="workspace-navigation workspace-page-heading"><div className="workspace-heading-main"><TopbarBreadcrumb><nav className="workspace-breadcrumb" aria-label={text('当前位置', 'Current location')}><Link to="/projects">{text('项目', 'Projects')}</Link>{returnProject && <><span aria-hidden="true">/</span><Link to={libraryUrl}>{text('训练数据', 'Training data')}</Link></>}</nav></TopbarBreadcrumb><div className="workspace-heading-title"><h1 title={info?.source.path}>{info && !infoError ? datasetName : text('图片、标签与遮罩','Images, captions and masks')}</h1></div></div></header>}
      </div>
      {loadingWorkspace ? <div className="dataset-loading" role="status"><span className="sr-only">{text('正在读取数据集…', 'Loading dataset…')}</span><div className="ui-skeleton"/><div className="ui-skeleton"/></div> : <>
      {(infoError || contextError) && <div role="alert" className="workspace-message error">{infoError || contextError}<button type="button" className="ui-btn ui-btn-sm" onClick={()=>{fetchInfo();void refreshProjectContext();}}>{t('common.retry')}</button></div>}
      {versionKey && versionAccess?.key === versionKey && !canEdit && <div className="workspace-message dataset-readonly-notice" role={versionAccess.error ? 'alert' : 'status'}>
        <span>{versionAccess.error ? `${text('无法确认版本状态，编辑已暂停：', 'Cannot verify version status; editing is paused: ')}${versionAccess.error}` : versionAccess.archived ? text('此版本已归档，图片、标签和遮罩只读。', 'This version is archived. Images, captions and masks are read only.') : text('此版本暂不可编辑，当前为只读查看。', 'This version is not editable yet. Viewing in read-only mode.')}</span>
        <Link className="ui-link" to={projectUrl(info!.source.project_id || '', info!.source.version_id, 'data')}>{text('返回版本工作区', 'Return to version workspace')}</Link>
        <button type="button" className="ui-link" onClick={() => void checkVersionAccess()}>{t('common.refresh')}</button>
      </div>}
      {actionError && !activeImage && <div role="alert" className="workspace-message error">{actionError}</div>}
      <div className="dataset-folder-bar" data-testid="dataset-overview">
        {info && <form className="dataset-folder-settings" onSubmit={event => {
          event.preventDefault();
          void beforeNavigation().then(() => updateSettings({ ...(folderName !== savedFolderName ? { name: folderName.trim() } : {}), repeats: Number(repeats) })).catch(error => setActionError(formatApiError(error)));
        }}>
          <label>{text('文件夹名称','Folder name')}<input aria-label={text('文件夹名称','Folder name')} value={folderName} required disabled={!canEdit || !!busyAction || addingImages || !info.source.can_rename} onChange={event => setFolderName(event.target.value)}/></label>
          <label className="dataset-repeat-field">{text('每轮重复次数','Repeats per epoch')}<input aria-label={text('每轮重复次数','Repeats per epoch')} type="number" min={1} max={1000000} required value={repeats} disabled={!canEdit || !!busyAction || addingImages} onChange={event => setRepeats(event.target.value)}/></label>
          <button type="submit" className="ui-btn ui-btn-primary" aria-label={text('保存目录设置','Save folder settings')} disabled={!canEdit || !!busyAction || addingImages || !folderName.trim() || !Number.isInteger(Number(repeats)) || Number(repeats)<1 || Number(repeats)>1000000 || (folderName === savedFolderName && Number(repeats) === info.source.repeats)}>{busyAction === 'settings' ? text('保存中…','Saving…') : text('保存','Save')}</button>
        </form>}
        <div className="dataset-folder-actions">
          {curationUrl && <Link className="ui-btn" to={curationUrl}>{text('数据集筛选','Curation')}</Link>}
          {canEdit && info?.source.can_append && <button type="button" className="ui-btn" aria-expanded={showAddImages} onClick={()=>setShowAddImages(value=>!value)} disabled={addingImages}>{text('添加图片', 'Add images')}</button>}
          <button type="button" className="ui-btn ui-btn-icon" onClick={handleRescan} disabled={!canEdit || !!busyAction || addingImages} title={t('dataset.rescan')} aria-label={t('dataset.rescan')}><RefreshCcw size={15}/></button>
          <button type="button" className="ui-btn ui-btn-icon ui-btn-danger" onClick={handleDelete} disabled={!canEdit || !!busyAction || addingImages} title={t('dataset.remove')} aria-label={t('dataset.remove')}><Trash2 size={15}/></button>
        </div>
      </div>
      {showAddImages && (canEdit || addingImages) && info?.source.can_append && info.source.project_id && <ProjectDataImport
        projectId={info.source.project_id} versionId={info.source.version_id || undefined} targetDataset={info}
        onBusyChange={setAddingImages} onImported={()=>{fetchInfo();images.refresh();void queryClient.invalidateQueries({queryKey:['caption-datasets']});}}/>}
      <div className="dataset-library-meta">
        <span className={`dataset-index-state status-${info?.index_status || 'unknown'}`}>{statusLabel(info?.index_status)}</span>
        {stats && <span>{text(`共 ${formatParams(stats.images)} 张 · 已有标签 ${formatParams(stats.captioned ?? 0)} 张 · 遮罩 ${formatParams(stats.masks ?? 0)} 张`, `${formatParams(stats.images)} images · ${formatParams(stats.captioned ?? 0)} captioned · ${formatParams(stats.masks ?? 0)} masks`)}</span>}
        <details className="dataset-source-details" data-popover><summary>{text('目录详情', 'Folder details')}</summary><div><code>{info?.source.path}</code><span>{text('标签格式', 'Caption format')}: {info?.source.caption_ext || '—'}</span></div></details>
      </div>
      {/* 索引进度条 */}
      {(info?.index_status === 'indexing' || indexProgress) && (
        <div className="dataset-index-progress" data-testid="index-progress">
          <div><span>{t('dataset.indexing')}</span><span className="tabular-nums">{indexProgress ? `${indexProgress.done} / ${indexProgress.total}` : '…'}</span></div>
          <ProgressBar label={t('dataset.indexing')} value={indexProgress && indexProgress.total > 0 ? indexProgress.done : undefined} max={indexProgress?.total || 100}/>
        </div>
      )}

      <div className="dataset-library-tools">
        <label className="dataset-library-search"><Search size={16}/><input type="search" aria-label={text('搜索文件名或标签','Search filenames or captions')} value={images.q} onChange={event => images.setQ(event.target.value)} placeholder={t('dataset.filterPlaceholder')} data-testid="dataset-search"/>{images.q && <button type="button" className="ui-btn ui-btn-quiet ui-btn-sm ui-btn-icon" aria-label={text('清除图片筛选','Clear image filter')} onClick={() => images.setQ('')}><X size={15}/></button>}</label>
        <label className="dataset-thumbnail-size">{text('缩略图','Thumbnails')}<input type="range" min={120} max={220} step={20} value={thumbnailWidth} onChange={event => setThumbnailWidth(Number(event.target.value))} aria-label={text('缩略图大小','Thumbnail size')}/></label>
        {canEdit && <button type="button" className="ui-btn" aria-expanded={showBatch} onClick={() => setShowBatch(value => !value)} disabled={!images.selected.size}>{text('批量编辑标签','Edit selected captions')}{images.selected.size > 0 ? ` (${images.selected.size})` : ''}</button>}
      </div>
      {showBatch && images.selected.size > 0 && <fieldset className="dataset-browser-batch" disabled={!!busyAction}><legend>{text('批量修改已选图片的标签','Edit captions of selected images')}</legend>
        <label>{text('添加标签','Add tags')}<input value={batchAdd} onChange={event => setBatchAdd(event.target.value)} placeholder={t('dataset.addTagsPlaceholder')} data-testid="batch-add-input"/></label>
        <label>{text('移除标签','Remove tags')}<input value={batchRemove} onChange={event => setBatchRemove(event.target.value)} placeholder={t('dataset.removeTagsPlaceholder')}/></label>
        <button type="button" onClick={applyBatchTags} disabled={!batchAdd.trim() && !batchRemove.trim()} className="ui-btn ui-btn-primary" data-testid="batch-apply-btn">{busyAction === 'batch' ? t('dataset.applying') : t('dataset.applyToSelection')}</button>
      </fieldset>}
      <div className="dataset-library-pane">
        <DatasetImagePane datasetId={id} images={images} training all count={stats?.images ?? 0} canEdit={canEdit} busy={!!busyAction} minWidth={thumbnailWidth} onMove={() => {}} onOpen={openEditor}/>
      </div>

      {canEdit && maskImage && id && <MaskEditor datasetId={id} imageId={maskImage.hash} relPath={maskImage.relPath} onClose={() => setMaskImage(null)} onSaved={() => { fetchInfo(); images.refresh(); }} onEnableTraining={enableMaskedTraining} />}

      {/* 大图 + caption 编辑 */}
      {activeImage && (
        <div className="caption-dialog-backdrop" onClick={closeCaption}>
          <div className="caption-dialog" onClick={(e) => e.stopPropagation()} data-testid="caption-editor" role="dialog" aria-modal="true" aria-label={text('编辑图片标签','Edit image caption')}>
            <header>
              <h3 className="dataset-preview-title">{activeImg?.rel_path}</h3>
              <button type="button" onClick={closeCaption} disabled={savingCaption} className="ui-btn ui-btn-quiet ui-btn-icon" title={t('common.close')} aria-label={t('common.close')}><X size={18}/></button>
            </header>
            <div className="caption-dialog-body">
              <div>
                <img src={apiUrl(`/datasets/${id}/images/${activeImage}/file`)} alt={activeImg?.rel_path || ''} className="caption-dialog-image"/>
                {activeImg && (
                  <div className="caption-dialog-meta" data-testid="caption-meta">
                    <span>{t('dataset.resolution', '分辨率')}: <span className="tabular-nums">{activeImg.width}×{activeImg.height}</span></span>
                    {typeof activeSize === 'number' && <span>{t('dataset.fileSize', '文件大小')}: <span className="tabular-nums">{formatBytes(activeSize)}</span></span>}
                    <span>{t('dataset.masks')}: {activeImg.has_mask ? t('dataset.maskYes', '有') : t('dataset.maskNo', '无')}</span>
                  </div>
                )}
              </div>
              <div className="caption-dialog-editor">
                {canEdit && <button type="button" disabled={savingCaption} onClick={() => { if (activeImg) {void (captionDirty?saveCaption():Promise.resolve()).then(()=>{setMaskImage({ hash: activeImg.hash, relPath: activeImg.rel_path });setActiveImage(null);}).catch(()=>{});} }} className="ui-btn"><Brush size={15}/>{text('编辑这张图片的训练遮罩', 'Edit this image’s training mask')}</button>}
                <div className="caption-dialog-label">{t('dataset.captionEditorTitle')}{activeImg?.caption_format && ` · ${activeImg.caption_format.toUpperCase()}`}</div>
                {activeImg?.caption_error && <p role="alert" className="caption-dialog-error">{activeImg.caption_error}</p>}
                {actionError && <p role="alert" className="caption-dialog-error">{actionError}</p>}
                {activeJson ? <StructuredCaptionEditor key={`${activeImage}/${captionPath}`} structure={activeStructure} draft={captionFields} onChange={setCaptionFields} disabled={captionLocked} readOnly={!canEdit}/> : canEdit ? <fieldset disabled={captionLocked}><TagChips caption={editCaption} onChange={setEditCaption} readOnly={savingCaption}/></fieldset> : <p className="caption-dialog-text">{editCaption || t('dataset.noCaption', '（无 caption）')}</p>}
                <footer>
                  <button type="button" onClick={closeCaption} disabled={savingCaption} className="ui-btn">{t('common.cancel')}</button>
                  <button type="button" onClick={()=>void saveCaption().catch(()=>{})} disabled={captionLocked} className="ui-btn ui-btn-primary" data-testid="caption-save-btn">{savingCaption ? t('dataset.saving') : t('dataset.saveCaption')}</button>
                </footer>
              </div>
            </div>
          </div>
        </div>
      )}
      </>}
    </div>
  );
}
