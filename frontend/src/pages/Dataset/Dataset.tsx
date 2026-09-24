import { projectUrl, type ProjectVersion, type VersionedProject } from '../../utils/projectVersions';
import React from 'react';
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
import './dataset-workspace.css';
import DatasetImagePane from './DatasetImagePane';
import ProjectWorkspaceHeader from '../../components/projects/ProjectWorkspaceHeader';
import DatasetNavigationGuard from '../../components/datasets/DatasetNavigationGuard';
import { useWorkspaceText } from '../../utils/workspaceText';
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
export function DatasetWorkspace({id, curation = false, selector}: {id?:string; curation?:boolean; selector?:React.ReactNode}) {
  const navigate = useNavigate();
  const location = useLocation();
  const hasDataRouter = !!React.useContext(UNSAFE_DataRouterContext);
  const { t } = useTranslation();
  const text = useWorkspaceText();
  const contextProblem = text('无法确认此数据集所属的项目版本。','The dataset project/version could not be verified.');

  // locales 占位符为单花括号（{n}），i18next 默认插值（{{}}）不处理，需手工替换
  const tt = React.useCallback(
    (key: string, vars: Record<string, string | number>) =>
      Object.entries(vars).reduce((s, [k, v]) => s.split(`{${k}}`).join(String(v)), t(key)),
    [t]
  );

  const [info, setInfo] = React.useState<DatasetInfo | null>(null);
  const [infoError, setInfoError] = React.useState('');
  const [projectContext, setProjectContext] = React.useState<{project:VersionedProject;versions:ProjectVersion[];current?:ProjectVersion}|null>(null);
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

  const trainingImages = useDatasetImages(id, 60, curation ? 'training' : 'all');
  const unusedImages = useDatasetImages(curation ? id : undefined, 60, 'unused');
  const images = {
    items: curation ? [...trainingImages.items, ...unusedImages.items] : trainingImages.items,
    selected: new Set([...trainingImages.selected, ...(curation ? unusedImages.selected : [])]),
    q: trainingImages.q,
    setQ: (query: string) => { trainingImages.setQ(query); unusedImages.setQ(query); },
    refresh: () => { trainingImages.refresh(); unusedImages.refresh(); },
    clearSelection: () => { trainingImages.clearSelection(); unusedImages.clearSelection(); },
    updateCaption: (hash: string, caption: string, path: string) => {
      const source = trainingImages.items.some(image => image.hash === hash && image.rel_path === path) ? trainingImages : unusedImages;
      source.updateCaption(hash, caption, path);
    },
  };
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
  const setParticipation = async (included: boolean) => {
    const source = included ? unusedImages : trainingImages;
    if (!id || !canEdit || !source.selected.size) return;
    setBusyAction('membership'); setActionError('');
    try {
      await apiClient.post(`/datasets/${id}/membership`, { paths: [...source.selected], included });
      source.clearSelection(); images.refresh(); fetchInfo();
    } catch (error) { setActionError(formatApiError(error)); }
    finally { setBusyAction(null); }
  };

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
    if(maskImage)throw new Error(text('请先在遮罩编辑器中保存或关闭，再切换项目页面。','Save or close the mask editor before switching project pages.'));
    if(activeImage && captionDirty || captionSave.current)await saveCaption();
  };
  const leaveRef=React.useRef({beforeNavigation,dirty:false});
  React.useLayoutEffect(()=>{leaveRef.current={beforeNavigation,dirty:!!maskImage || !!activeImage && captionDirty || savingCaption};});
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
  const returnUrl = returnProject ? `${projectUrl(returnProject, returnVersion, 'data')}&data_step=import#version-datasets` : '/projects';

  const returnLink = <Link className="dataset-workspace-return" to={returnUrl}><ArrowLeft size={14}/>{returnProject ? text('返回本版本数据集', 'Back to version datasets') : text('返回项目', 'Back to projects')}</Link>;

  return (
    <div className={`dataset-workspace${curation ? ' dataset-curation-workspace' : ''}`} data-testid="dataset-page">
      {hasDataRouter && <DatasetNavigationGuard shouldBlock={() => {
        return leaveRef.current.dirty;
      }} beforeLeave={() => leaveRef.current.beforeNavigation()} onError={error => setActionError(formatApiError(error))}/>}
      <div className="dataset-workspace-navigation">
        {projectContext && !infoError ? <ProjectWorkspaceHeader project={projectContext.project} versionId={info?.source.version_id || undefined} versions={projectContext.versions} current={projectContext.current} active="data" title={curation ? text('训练集筛选','Training set curation') : datasetName} titleBadge={selector} breadcrumbLeading={returnLink} refresh={refreshProjectContext} beforeAction={beforeNavigation}/> : <><nav className="workspace-breadcrumb">{returnLink}</nav><h1 className="text-base font-semibold" title={info?.source.path}>{curation ? text('训练集筛选','Training set curation') : info && !infoError ? datasetName : text('图片、标签与遮罩','Images, captions and masks')}</h1></>}
      </div>
      {(infoError || contextError) && <div role="alert" className="workspace-message error">{infoError || contextError}<button onClick={()=>{fetchInfo();void refreshProjectContext();}}>{t('common.retry')}</button></div>}
      {versionKey && !canEdit && <div className="flex flex-wrap items-center gap-2 rounded border border-slate-300 bg-slate-50 px-3 py-2 text-xs dark:border-slate-700 dark:bg-slate-900" role={versionAccess?.key === versionKey && versionAccess.error ? 'alert' : 'status'}>
        <span>{versionAccess?.key !== versionKey ? text('正在确认版本状态，暂以只读方式查看。', 'Checking version status. Viewing in read-only mode.') : versionAccess.error ? `${text('无法确认版本状态，编辑已暂停：', 'Cannot verify version status; editing is paused: ')}${versionAccess.error}` : versionAccess.archived ? text('此版本已归档，图片、标签和遮罩只读。', 'This version is archived. Images, captions and masks are read only.') : text('此版本暂不可编辑，当前为只读查看。', 'This version is not editable yet. Viewing in read-only mode.')}</span>
        <Link className="text-blue-600" to={projectUrl(info!.source.project_id || '', info!.source.version_id, 'data')}>{text('返回版本工作区', 'Return to version workspace')}</Link>
        <button type="button" className="text-blue-600" onClick={() => void checkVersionAccess()}>{t('common.refresh')}</button>
      </div>}
      {actionError && !activeImage && <div role="alert" className="rounded-lg bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950 dark:text-red-300">{actionError}</div>}
      {!curation && <><div className="dataset-folder-bar" data-testid="dataset-overview">
        {info && <form className="dataset-folder-settings" onSubmit={event => {
          event.preventDefault();
          void beforeNavigation().then(() => updateSettings({ ...(folderName !== savedFolderName ? { name: folderName.trim() } : {}), repeats: Number(repeats) })).catch(error => setActionError(formatApiError(error)));
        }}>
          <label>{text('文件夹名称','Folder name')}<input aria-label={text('文件夹名称','Folder name')} value={folderName} required disabled={!canEdit || !!busyAction || !info.source.can_rename} onChange={event => setFolderName(event.target.value)}/></label>
          <label className="dataset-repeat-field">{text('每轮重复次数','Repeats per epoch')}<input aria-label={text('每轮重复次数','Repeats per epoch')} type="number" min={1} max={1000000} required value={repeats} disabled={!canEdit || !!busyAction} onChange={event => setRepeats(event.target.value)}/></label>
          <button type="submit" aria-label={text('保存目录设置','Save folder settings')} disabled={!canEdit || !!busyAction || !folderName.trim() || !Number.isInteger(Number(repeats)) || Number(repeats)<1 || Number(repeats)>1000000 || (folderName === savedFolderName && Number(repeats) === info.source.repeats)}>{busyAction === 'settings' ? text('保存中…','Saving…') : text('保存','Save')}</button>
        </form>}
        <div className="dataset-folder-actions">
          {info?.source.project_id && info.source.version_id && <Link to={`/projects/${info.source.project_id}/v/${info.source.version_id}/curate?dataset=${id}`}>{text('训练集筛选','Training set curation')}</Link>}
          {canEdit && info?.source.project_id && <Link to={projectUrl(info.source.project_id, info.source.version_id, 'data')}>{text('添加图片', 'Add images')}</Link>}
          <button onClick={handleRescan} disabled={!canEdit || !!busyAction} title={t('dataset.rescan')} aria-label={t('dataset.rescan')}><RefreshCcw size={16}/></button>
          <button onClick={handleDelete} disabled={!canEdit || !!busyAction} title={t('dataset.remove')} aria-label={t('dataset.remove')} className="dataset-remove-action"><Trash2 size={16}/></button>
        </div>
      </div>
      <div className="dataset-library-meta">
        <span className={`dataset-index-state status-${info?.index_status || 'unknown'}`}>{statusLabel(info?.index_status)}</span>
        {stats && <span>{text(`共 ${formatParams(stats.images)} 张 · 已有标签 ${formatParams(stats.captioned ?? 0)} 张 · 遮罩 ${formatParams(stats.masks ?? 0)} 张`, `${formatParams(stats.images)} images · ${formatParams(stats.captioned ?? 0)} captioned · ${formatParams(stats.masks ?? 0)} masks`)}</span>}
        <details className="dataset-source-details"><summary>{text('目录详情', 'Folder details')}</summary><div><code>{info?.source.path}</code><span>{text('标签格式', 'Caption format')}: {info?.source.caption_ext || '—'}</span></div></details>
      </div>

      </>}
      {/* 索引进度条 */}
      {(info?.index_status === 'indexing' || indexProgress) && (
        <div className="p-4 bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800 rounded-xl" data-testid="index-progress">
          <div className="flex justify-between text-xs text-blue-700 dark:text-blue-300 mb-1.5">
            <span>{t('dataset.indexing')}</span>
            <span className="font-mono">{indexProgress ? `${indexProgress.done} / ${indexProgress.total}` : '…'}</span>
          </div>
          <div className="w-full bg-blue-200 dark:bg-blue-900 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all"
              style={{ width: `${indexProgress && indexProgress.total > 0 ? (indexProgress.done / indexProgress.total) * 100 : 20}%` }}
            />
          </div>
        </div>
      )}

      <div className="dataset-library-tools">
        <label className="dataset-library-search"><Search size={16}/><input type="search" aria-label={text('搜索文件名或标签','Search filenames or captions')} value={images.q} onChange={event => images.setQ(event.target.value)} placeholder={t('dataset.filterPlaceholder')} data-testid="dataset-search"/>{images.q && <button type="button" aria-label={text('清除图片筛选','Clear image filter')} onClick={() => images.setQ('')}><X size={16}/></button>}</label>
        <label className="dataset-thumbnail-size">{text('缩略图','Thumbnails')}<input type="range" min={120} max={220} step={20} value={thumbnailWidth} onChange={event => setThumbnailWidth(Number(event.target.value))} aria-label={text('缩略图大小','Thumbnail size')}/></label>
        {canEdit && !curation && <button type="button" aria-expanded={showBatch} onClick={() => setShowBatch(value => !value)} disabled={!images.selected.size}>{text('批量编辑标签','Edit selected captions')}{images.selected.size > 0 ? ` (${images.selected.size})` : ''}</button>}
      </div>
      {showBatch && images.selected.size > 0 && <fieldset className="dataset-browser-batch" disabled={!!busyAction}><legend>{text('批量修改已选图片的标签','Edit captions of selected images')}</legend>
        <label>{text('添加标签','Add tags')}<input value={batchAdd} onChange={event => setBatchAdd(event.target.value)} placeholder={t('dataset.addTagsPlaceholder')} data-testid="batch-add-input"/></label>
        <label>{text('移除标签','Remove tags')}<input value={batchRemove} onChange={event => setBatchRemove(event.target.value)} placeholder={t('dataset.removeTagsPlaceholder')}/></label>
        <button onClick={applyBatchTags} disabled={!batchAdd.trim() && !batchRemove.trim()} className="dataset-primary-action" data-testid="batch-apply-btn">{busyAction === 'batch' ? t('dataset.applying') : t('dataset.applyToSelection')}</button>
      </fieldset>}
      <div className={`dataset-curation-panes${curation ? '' : ' dataset-single-pane'}`}>
        <DatasetImagePane datasetId={id} images={trainingImages} training all={!curation} previewOnly={curation} count={curation ? stats?.training_images ?? stats?.images ?? 0 : stats?.images ?? 0} canEdit={canEdit} busy={!!busyAction} minWidth={thumbnailWidth} onMove={() => void setParticipation(false)} onOpen={openEditor}/>
        {curation && <DatasetImagePane datasetId={id} previewOnly images={unusedImages} training={false} count={stats?.held_out_images ?? 0} canEdit={canEdit} busy={!!busyAction} minWidth={thumbnailWidth} onMove={() => void setParticipation(true)} onOpen={openEditor}/>}
      </div>

      {canEdit && maskImage && id && <MaskEditor datasetId={id} imageId={maskImage.hash} relPath={maskImage.relPath} onClose={() => setMaskImage(null)} onSaved={() => { fetchInfo(); images.refresh(); }} onEnableTraining={enableMaskedTraining} />}

      {/* 大图 + caption 编辑抽屉 */}
      {activeImage && (
        <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-6" onClick={closeCaption}>
          <div
            className="bg-white dark:bg-slate-800 rounded-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto p-6 space-y-4 shadow-xl"
            onClick={(e) => e.stopPropagation()}
            data-testid="caption-editor"
            role="dialog" aria-modal="true" aria-label={curation ? text('图片预览','Image preview') : text('编辑图片标签','Edit image caption')}
          >
            <div className="flex justify-between items-center">
              <h3 className="dataset-preview-title">{activeImg?.rel_path}</h3>
              <button onClick={closeCaption} disabled={savingCaption} className="text-slate-400 hover:text-slate-600" title={t('common.close')}>
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <img
                  src={apiUrl(`/datasets/${id}/images/${activeImage}/file`)}
                  alt={activeImg?.rel_path || ''}
                  className="w-full max-h-[50vh] object-contain rounded-lg bg-slate-100 dark:bg-slate-900"
                />
                {activeImg && (
                  <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs text-slate-400" data-testid="caption-meta">
                    <span>
                      {t('dataset.resolution', '分辨率')}: <span className="font-mono">{activeImg.width}×{activeImg.height}</span>
                    </span>
                    {typeof activeSize === 'number' && (
                      <span>
                        {t('dataset.fileSize', '文件大小')}: <span className="font-mono">{formatBytes(activeSize)}</span>
                      </span>
                    )}
                    <span>
                      {t('dataset.masks')}: {activeImg.has_mask ? t('dataset.maskYes', '有') : t('dataset.maskNo', '无')}
                    </span>
                  </div>
                )}
              </div>
              <div className="space-y-3">
                {canEdit && !curation && <button type="button" disabled={savingCaption} onClick={() => { if (activeImg) {void (captionDirty?saveCaption():Promise.resolve()).then(()=>{setMaskImage({ hash: activeImg.hash, relPath: activeImg.rel_path });setActiveImage(null);}).catch(()=>{});} }} className="flex items-center gap-2 rounded-lg border border-slate-300 px-3 py-2 text-sm dark:border-slate-600"><Brush className="h-4 w-4" />{text('编辑这张图片的训练遮罩', 'Edit this image’s training mask')}</button>}
                <div className="text-xs text-slate-400">{curation ? text('图片标签','Caption') : t('dataset.captionEditorTitle')}{activeImg?.caption_format && ` · ${activeImg.caption_format.toUpperCase()}`}</div>
                {activeImg?.caption_error && <p role="alert">{activeImg.caption_error}</p>}
                {actionError && <p role="alert" className="text-sm text-red-600">{actionError}</p>}
                {curation ? <p className="whitespace-pre-wrap text-sm">{activeImg?.caption || text('暂无标签','No caption')}</p> : activeJson ? <StructuredCaptionEditor key={`${activeImage}/${captionPath}`} structure={activeStructure} draft={captionFields} onChange={setCaptionFields} disabled={captionLocked} readOnly={!canEdit}/> : canEdit ? <fieldset disabled={captionLocked}><TagChips caption={editCaption} onChange={setEditCaption} readOnly={savingCaption}/></fieldset> : <p className="whitespace-pre-wrap text-sm">{editCaption || t('dataset.noCaption', '（无 caption）')}</p>}
                <div className="flex justify-end space-x-2 pt-2">
                  <button
                    onClick={closeCaption} disabled={savingCaption}
                    className="px-4 py-2 text-sm rounded bg-slate-200 dark:bg-slate-700"
                  >
                    {t('common.cancel')}
                  </button>
                  {!curation && <button
                    onClick={()=>void saveCaption().catch(()=>{})}
                    disabled={captionLocked}
                    className="px-4 py-2 text-sm rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
                    data-testid="caption-save-btn"
                  >
                    {savingCaption ? t('dataset.saving') : t('dataset.saveCaption')}
                  </button>}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
