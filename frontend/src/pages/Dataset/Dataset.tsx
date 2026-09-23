import { modelConfigUrl, projectUrl, versionConfigUrl, type ProjectVersion, type VersionedProject } from '../../utils/projectVersions';
import React from 'react';
import { Link, useParams, useNavigate, useLocation, UNSAFE_DataRouterContext } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { apiClient, apiUrl } from '../../api/client';
import { DatasetInfo, Job, Plan } from '../../api/types';
import { useDatasetImages } from '../../api/hooks/useDatasetImages';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { MaskEditor } from '../../components/masks/MaskEditor';
import { formatApiError } from '../../utils/errors';
import { TagChips } from '../../components/TagChips';
import StructuredCaptionEditor from '../../components/datasets/StructuredCaptionEditor';
import { getCaptionStructure, captionFieldChanges, type CaptionFieldDraft } from '../../utils/captionStructure';
import { formatBytes, formatParams, formatPercent } from '../../utils/format';
import './dataset-workspace.css';
import { NextStepLink } from '../../components/ProjectWorkflow';
import ProjectWorkspaceHeader from '../../components/projects/ProjectWorkspaceHeader';
import DatasetNavigationGuard from '../../components/datasets/DatasetNavigationGuard';
import { useWorkspaceHeight } from '../../components/projects/useWorkspaceHeight';
import { useWorkspaceText } from '../../utils/workspaceText';
import {
  RefreshCcw,
  Trash2,
  Layers,
  Image as ImageIcon,
  Search,
  SearchX,
  X,
  CheckSquare,
  Square,
  Zap,
  Brush,
  ArrowLeft,
} from 'lucide-react';

const THUMB_SIZE = 256;
const CARD_W = 200;
const CARD_H = 286;
const GAP = 12;

function Histogram({ data, label, barColor }: { data: Array<{ name: string; count: number }>; label: string; barColor: string }) {
  const max = Math.max(1, ...data.map((d) => d.count));
  return (
    <div>
      <div className="text-xs text-slate-400 mb-1.5">{label}</div>
      <div className="flex items-end space-x-1 h-16">
        {data.map((d, i) => (
          <div key={i} className="flex-1 flex flex-col items-center justify-end min-w-0" title={`${d.name}: ${d.count}`}>
            <div className={`w-full rounded-t ${barColor}`} style={{ height: `${(d.count / max) * 100}%` }} />
            <div className="text-[9px] text-slate-400 mt-0.5 truncate w-full text-center">{d.name}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Dataset() {
  const { id } = useParams<{ id: string }>();
  return <DatasetContent key={id || ''} id={id}/>;
}
function DatasetContent({id}: {id?:string}) {
  const navigate = useNavigate();
  const location = useLocation();
  const hasDataRouter = !!React.useContext(UNSAFE_DataRouterContext);
  const allowedDestination = React.useRef<string | null>(null);
  const { t } = useTranslation();
  const text = useWorkspaceText();
  const navigationRef = useWorkspaceHeight('--workspace-head-height');
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
  const [buckets, setBuckets] = React.useState<Plan['buckets'] | null>(null);
  const [showDistribution, setShowDistribution] = React.useState(false);
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

  const images = useDatasetImages(id);
  const activeImg = activeImage ? images.items.find(i => i.hash === activeImage && i.rel_path === captionPath) : undefined;
  const activeStructure = getCaptionStructure(activeImg);
  const activeJson = activeImg?.caption_format?.toLowerCase().replace(/^\./, '') === 'json';
  const changedFields = captionFieldChanges(activeStructure, captionFields);
  const captionDirty = editCaption !== captionBase || changedFields.length > 0;
  const captionLocked = !canEdit || savingCaption || !!activeImg?.caption_error || activeJson && !activeStructure?.editable;
  const gridRef = React.useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = React.useState(0);
  const [viewportH, setViewportH] = React.useState(600);
  const [viewportW, setViewportW] = React.useState(1200);

  const fetchInfo = React.useCallback(() => {
    if (!id) return;
    apiClient.get<DatasetInfo>(`/datasets/${id}`, {silent:true}).then((data) => {
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

  // viewport 尺寸跟踪（虚拟滚动用）
  React.useEffect(() => {
    const el = gridRef.current;
    if (!el) return;
    const onResize = () => {setViewportH(el.clientHeight || 600);setViewportW(el.clientWidth || 1200);};
    onResize();
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(onResize);
    observer?.observe(el);
    window.addEventListener('resize', onResize);
    return () => { observer?.disconnect(); window.removeEventListener('resize', onResize); };
  }, []);

  // 虚拟滚动窗口计算
  const containerWidth = Math.max(1, viewportW);
  const cols = Math.max(1, Math.floor(containerWidth / (CARD_W + GAP)));
  const cardWidth = Math.min(260, Math.max(1, (containerWidth - GAP - (cols - 1) * GAP) / cols));
  const rows = Math.ceil(images.items.length / cols);
  const rowH = CARD_H + GAP;
  const startRow = Math.max(0, Math.floor(scrollTop / rowH) - 2);
  const endRow = Math.min(rows, Math.ceil((scrollTop + viewportH) / rowH) + 2);
  const visibleItems = images.items.slice(startRow * cols, endRow * cols);

  const enableMaskedTraining = async () => {
    if (!info || !canEdit) return;
    const endpoint = versionConfigUrl(info.source.project_id || '', info.source.version_id);
    const config = await apiClient.get<{ dataset?: Record<string, unknown>; [key: string]: unknown }>(endpoint);
    await apiClient.put(endpoint, { ...config, dataset: { ...config.dataset, masked_loss: true } });
    const destination = projectUrl(info.source.project_id || '', info.source.version_id, 'train');
    // The editor invokes this only after its mask was saved successfully.
    allowedDestination.current = destination;
    navigate(destination);
  };

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(e.currentTarget.scrollTop);
    // 接近底部时加载下一页
    const el = e.currentTarget;
    if (el.scrollTop + el.clientHeight >= el.scrollHeight - rowH * 3) {
      images.loadMore();
    }
  };

  const openEditor = (hash: string, relPath: string) => {
    const img = images.items.find((i) => i.hash === hash && i.rel_path === relPath);
    if (!img) return;
    setActiveImage(hash); setCaptionPath(relPath); setCaptionFields({});
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
      .post(`/datasets/${id}/tags/batch`, { hashes: Array.from(images.selected), add, remove })
      .then(() => {
        images.refresh();
        images.clearSelection();
        setBatchAdd('');
        setBatchRemove('');
      })
      .catch(console.error)
      .finally(() => setBusyAction(null));
  };

  const handleRescan = () => {
    if (!id || !canEdit) return;
    setBusyAction('rescan');
    apiClient.post(`/datasets/${id}/rescan`, {})
      .then(fetchInfo)
      .catch(console.error)
      .finally(() => setBusyAction(null));
  };

  const handleDelete = () => {
    if (!id || !info || !canEdit) return;
    if (window.confirm(tt('dataset.deleteConfirm', { path: info.source.path }))) {
      setBusyAction('delete');
      apiClient.delete(`/datasets/${id}`)
        .then(() => navigate(returnUrl))
        .catch(console.error)
        .finally(() => setBusyAction(null));
    }
  };

  const handlePrecache = () => {
    if (!id || !info || !canEdit) return;
    setBusyAction('precache');
    apiClient.post<Job>(`/jobs`, {
      type: 'cache',
      name: `cache-${info.source.path.split('/').pop()}-${Date.now()}`,
      project_id: info.source.project_id,
      version_id: info.source.version_id,
    })
      .then(() => alert(t('dataset.precacheEnqueued')))
      .catch(console.error)
      .finally(() => setBusyAction(null));
  };

  const handleBucketPreview = () => {
    if (!info) return;
    setBusyAction('buckets');
    apiClient.get<any>(versionConfigUrl(info.source.project_id || '', info.source.version_id))
      .then((config) => apiClient.post<Plan>('/plan', { config, dataset_ids: [id] }))
      .then((plan) => { setBuckets(plan.buckets || []); setShowDistribution(true); })
      .catch(console.error)
      .finally(() => setBusyAction(null));
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
  const coverage = stats?.images ? Math.round(((stats.captioned || 0) / stats.images) * 100) : 0;
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
    <div className="dataset-workspace space-y-3" data-testid="dataset-page">
      {hasDataRouter && <DatasetNavigationGuard shouldBlock={destination => {
        if(allowedDestination.current === `${destination.pathname}${destination.search}${destination.hash}`){allowedDestination.current=null;return false;}
        return leaveRef.current.dirty;
      }} beforeLeave={() => leaveRef.current.beforeNavigation()} onError={error => setActionError(formatApiError(error))}/>}
      <div className="dataset-workspace-navigation" ref={navigationRef}>
        {projectContext && !infoError ? <ProjectWorkspaceHeader project={projectContext.project} versionId={info?.source.version_id || undefined} versions={projectContext.versions} current={projectContext.current} active="data" title={datasetName} breadcrumbLeading={returnLink} refresh={refreshProjectContext} beforeAction={beforeNavigation}/> : <><nav className="workspace-breadcrumb">{returnLink}</nav><h1 className="text-base font-semibold" title={info?.source.path}>{info && !infoError ? datasetName : text('图片、标签与遮罩','Images, captions and masks')}</h1></>}
      </div>
      {(infoError || contextError) && <div role="alert" className="workspace-message error">{infoError || contextError}<button onClick={()=>{fetchInfo();void refreshProjectContext();}}>{t('common.retry')}</button></div>}
      {versionKey && !canEdit && <div className="flex flex-wrap items-center gap-2 rounded border border-slate-300 bg-slate-50 px-3 py-2 text-xs dark:border-slate-700 dark:bg-slate-900" role={versionAccess?.key === versionKey && versionAccess.error ? 'alert' : 'status'}>
        <span>{versionAccess?.key !== versionKey ? text('正在确认版本状态，暂以只读方式查看。', 'Checking version status. Viewing in read-only mode.') : versionAccess.error ? `${text('无法确认版本状态，编辑已暂停：', 'Cannot verify version status; editing is paused: ')}${versionAccess.error}` : versionAccess.archived ? text('此版本已归档，图片、标签和遮罩只读。', 'This version is archived. Images, captions and masks are read only.') : text('此版本暂不可编辑，当前为只读查看。', 'This version is not editable yet. Viewing in read-only mode.')}</span>
        <Link className="text-blue-600" to={projectUrl(info!.source.project_id || '', info!.source.version_id, 'data')}>{text('返回版本工作区', 'Return to version workspace')}</Link>
        <button type="button" className="text-blue-600" onClick={() => void checkVersionAccess()}>{t('common.refresh')}</button>
      </div>}
      {actionError && !activeImage && <div role="alert" className="rounded-lg bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950 dark:text-red-300">{actionError}</div>}
      <div className="dataset-overview-toolbar" data-testid="dataset-overview">
        <div className="dataset-overview-counts">
          <span className={`dataset-index-state status-${info?.index_status || 'unknown'}`}>{statusLabel(info?.index_status)}</span>
          {stats && <>
            <span>{t('dataset.images')} <b>{formatParams(stats.images)}</b></span>
            <span>{t('dataset.captioned')} <b>{formatParams(stats.captioned ?? 0)}</b> <small>{formatPercent(coverage)}</small></span>
            <span>{t('dataset.masks')} <b>{formatParams(stats.masks ?? 0)}</b></span>
          </>}
          <details className="dataset-source-details"><summary>{text('数据与遮罩说明', 'Dataset & mask details')}</summary><div>
            <code>{info?.source.path}</code>
            <p>{t('dataset.repeats')} ×{info?.source.repeats ?? '--'} · {t('dataset.caption')}: {info?.source.caption_ext || '--'}</p>
            {info?.cache?.latents && <p>{t('dataset.latents')}: {info.cache.latents.cached}/{info.cache.latents.total} · {t('dataset.text')}: {info.cache.text?.cached ?? 0}/{info.cache.text?.total ?? 0}</p>}

            <p>{text('白色参与训练，黑色忽略。没有独立遮罩时使用原图 Alpha；没有 Alpha 时全图参与。启用遮罩训练后生效。', 'White trains, black is ignored. Without a sidecar, image alpha is used; without alpha, the whole image participates. Enable masked training to apply these weights.')}</p>
          </div></details>
        </div>
        <div className="dataset-overview-actions">
          {canEdit && info?.source.project_id && <Link to={projectUrl(info.source.project_id, info.source.version_id, 'data')}>{text('添加数据', 'Add data')}</Link>}
          <button disabled={!canEdit || busyAction === 'mask-enable'} onClick={() => { setBusyAction('mask-enable'); setActionError(''); void enableMaskedTraining().catch(error => setActionError(formatApiError(error))).finally(() => setBusyAction(null)); }} className="dataset-primary-action">{text('启用遮罩并前往训练', 'Enable masks and open training')}</button>
          {info?.source.project_id && <NextStepLink to={modelConfigUrl(info.source.project_id, info.source.version_id)}>{text('选择训练模型', 'Choose training model')}</NextStepLink>}
          <button type="button" aria-expanded={showDistribution} aria-controls="dataset-distribution" onClick={() => setShowDistribution(value => !value)}><Layers size={13}/>{text('分布与分桶', 'Distribution & buckets')}</button>
          <button onClick={handlePrecache} disabled={!canEdit || busyAction === 'precache'} title={t('dataset.precache')} aria-label={busyAction === 'precache' ? t('dataset.enqueuing') : t('dataset.precache')}><Zap size={14}/></button>
          <button onClick={handleRescan} disabled={!canEdit || busyAction === 'rescan'} title={t('dataset.rescan')} aria-label={t('dataset.rescan')}><RefreshCcw size={14}/></button>
          <button onClick={handleDelete} disabled={!canEdit || busyAction === 'delete'} title={t('dataset.remove')} aria-label={t('dataset.remove')} className="dataset-remove-action"><Trash2 size={14}/></button>
        </div>
      </div>

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

      {stats && showDistribution && <div id="dataset-distribution" className="dataset-distribution grid gap-3 rounded border border-slate-200 p-3 sm:grid-cols-3 dark:border-slate-700">
          <div className="min-w-0">
            <Histogram
              label={t('dataset.resolutions')}
              barColor="bg-blue-400"
              data={(stats.resolutions || []).map((r) => ({ name: `${r.w}×${r.h}`, count: r.count }))}
            />
          </div>
          <div className="min-w-0">
            <Histogram
              label={t('dataset.aspectRatio')}
              barColor="bg-indigo-400"
              data={(stats.ar_hist || []).map((r) => ({ name: r.ar, count: r.count }))}
            />
          </div>
          <div className="max-h-40 min-w-0 overflow-auto">
            <div className="mb-1.5 flex items-center justify-between gap-2 text-xs text-slate-400"><span>{t('dataset.bucketsTitle')}</span><button onClick={handleBucketPreview} disabled={busyAction === 'buckets'} className="text-blue-500 disabled:opacity-50">{busyAction === 'buckets' ? t('dataset.computing') : text('计算分桶', 'Calculate buckets')}</button></div>
            {buckets ? (
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-slate-400">
                    <th className="text-left">W×H</th>
                    <th className="text-right">{t('dataset.bucketItems', '条目')}</th>
                    <th className="text-right">{t('dataset.bucketBatches', '批数')}</th>
                  </tr>
                </thead>
                <tbody>
                  {buckets.map((b, i) => (
                    <tr key={i}>
                      <td className="font-mono">{b.w}×{b.h}</td>
                      <td className="text-right font-mono">{(b as any).items ?? (b as any).images}</td>
                      <td className="text-right font-mono">{b.batches}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="text-xs text-slate-400">{t('dataset.bucketsHint')}</div>
            )}
          </div>
      </div>}

      <div className="dataset-browser-toolbar">
        <div className="dataset-browser-filter-row" role="search" aria-label={text('筛选当前目录图片', 'Filter images in this folder')}>
          <label className="dataset-browser-search"><span>{text('图片筛选', 'Image filter')}</span><div><Search size={17}/><input type="text" aria-label={text('搜索文件名或标签', 'Search filenames or captions')} value={images.q} onChange={event => images.setQ(event.target.value)} placeholder={t('dataset.filterPlaceholder')} data-testid="dataset-search"/>{images.q && <button type="button" aria-label={text('清除图片筛选', 'Clear image filter')} onClick={() => images.setQ('')}><X size={16}/></button>}</div></label>
          <span className="dataset-browser-result-count">{tt('dataset.imagesTotal', { n: images.total })}</span>
        </div>
        <div className="dataset-browser-selection" role="group" aria-label={text('图片选择', 'Image selection')}>
          <span>{tt('dataset.selected', { n: images.selected.size })}</span>
          <button disabled={!canEdit} onClick={images.selectAll}><CheckSquare size={17}/>{t('dataset.selectAll')}</button>
          <button disabled={!canEdit || !images.selected.size} onClick={images.clearSelection}><Square size={17}/>{t('dataset.selectNone')}</button>
        </div>
        {canEdit && images.selected.size > 0 && <fieldset className="dataset-browser-batch" disabled={busyAction === 'batch'}><legend>{text('批量修改已选图片的标签', 'Edit captions of selected images')}</legend>
          <label>{text('添加标签', 'Add tags')}<input type="text" value={batchAdd} onChange={event => setBatchAdd(event.target.value)} placeholder={t('dataset.addTagsPlaceholder')} data-testid="batch-add-input"/></label>
          <label>{text('移除标签', 'Remove tags')}<input type="text" value={batchRemove} onChange={event => setBatchRemove(event.target.value)} placeholder={t('dataset.removeTagsPlaceholder')}/></label>
          <button onClick={applyBatchTags} disabled={!batchAdd.trim() && !batchRemove.trim()} className="dataset-primary-action" data-testid="batch-apply-btn">{busyAction === 'batch' ? t('dataset.applying') : t('dataset.applyToSelection')}</button>
        </fieldset>}
      </div>

      {/* 虚拟滚动图片网格 */}
      <div
        ref={gridRef}
        onScroll={handleScroll}
        className="dataset-browser-grid relative overflow-y-auto bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700"
        data-testid="image-grid"
      >
        <div style={{ height: rows * rowH, position: 'relative' }}>
          <div
            style={{
              position: 'absolute',
              top: startRow * rowH,
              left: 0,
              right: 0,
              display: 'grid',
              gridTemplateColumns: `repeat(${cols}, ${cardWidth}px)`,
              gap: GAP,
              justifyContent: 'center',
              padding: GAP / 2,
            }}
          >
            {visibleItems.map((img) => {
              const selected = images.selected.has(img.hash);
              return (
                <div
                  key={`${img.hash}:${img.rel_path}`}
                  className={`relative rounded-lg overflow-hidden border cursor-pointer group ${
                    selected ? 'border-blue-500 ring-2 ring-blue-500/40' : 'border-slate-200 dark:border-slate-700'
                  }`}
                  style={{ width: cardWidth, height: CARD_H }}
                  data-testid={`image-card-${img.hash}`}
                >
                  <button type="button" onClick={() => openEditor(img.hash, img.rel_path)} aria-label={`${canEdit ? text('编辑标签', 'Edit caption') : text('查看图片与标签', 'View image and caption')}: ${img.rel_path}`} className="block w-full">
                    <img src={apiUrl(`/datasets/${id}/images/${img.hash}/thumb?size=${THUMB_SIZE}`)} alt={img.rel_path} loading="lazy" className="w-full h-[170px] object-contain bg-slate-100 dark:bg-slate-900" />
                  </button>
                  <button
                    disabled={!canEdit}
                    aria-label={`${text('选择图片', 'Select image')}: ${img.rel_path}`}
                    onClick={() => images.toggleSelect(img.hash)}
                    aria-pressed={selected}
                    className={`dataset-image-select ${selected ? 'is-selected' : ''}`}
                  >
                    {selected ? <CheckSquare size={18}/> : <Square size={18}/>}
                  </button>
                  <div className="dataset-image-caption"><strong title={img.rel_path}>{img.rel_path}</strong><span title={img.caption}>{img.width} × {img.height} · {img.caption || t('dataset.noCaption', '（无 caption）')}</span></div>
                  {canEdit && <button type="button" onClick={() => setMaskImage({ hash: img.hash, relPath: img.rel_path })} className="dataset-image-mask mx-1.5 flex min-h-9 w-[calc(100%-12px)] items-center justify-center gap-1 rounded border border-slate-300 px-2 py-1 text-xs hover:bg-slate-100 dark:border-slate-600 dark:hover:bg-slate-700"><Brush className="h-3.5 w-3.5" />{img.has_mask ? text('编辑遮罩 · 已有文件', 'Edit mask · saved') : text('编辑遮罩', 'Edit mask')}</button>}
                </div>
              );
            })}
          </div>
        </div>
        {images.loading && (
          <div className="sticky bottom-0 text-center text-xs text-slate-400 py-2 bg-white/80 dark:bg-slate-800/80">{t('common.loading')}</div>
        )}
      </div>

      {canEdit && maskImage && id && <MaskEditor datasetId={id} imageId={maskImage.hash} relPath={maskImage.relPath} onClose={() => setMaskImage(null)} onSaved={() => { fetchInfo(); images.refresh(); }} onEnableTraining={enableMaskedTraining} />}

      {/* 大图 + caption 编辑抽屉 */}
      {activeImage && (
        <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-6" onClick={closeCaption}>
          <div
            className="bg-white dark:bg-slate-800 rounded-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto p-6 space-y-4 shadow-xl"
            onClick={(e) => e.stopPropagation()}
            data-testid="caption-editor"
            role="dialog" aria-modal="true" aria-label={text('编辑图片标签','Edit image caption')}
          >
            <div className="flex justify-between items-center">
              <h3 className="font-semibold text-lg font-mono">{activeImg?.rel_path}</h3>
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
                {canEdit && <button type="button" disabled={savingCaption} onClick={() => { if (activeImg) {void (captionDirty?saveCaption():Promise.resolve()).then(()=>{setMaskImage({ hash: activeImg.hash, relPath: activeImg.rel_path });setActiveImage(null);}).catch(()=>{});} }} className="flex items-center gap-2 rounded-lg border border-slate-300 px-3 py-2 text-sm dark:border-slate-600"><Brush className="h-4 w-4" />{text('编辑这张图片的训练遮罩', 'Edit this image’s training mask')}</button>}
                <div className="text-xs text-slate-400">{t('dataset.captionEditorTitle')}{activeImg?.caption_format && ` · ${activeImg.caption_format.toUpperCase()}`}</div>
                {activeImg?.caption_error && <p role="alert">{activeImg.caption_error}</p>}
                {actionError && <p role="alert" className="text-sm text-red-600">{actionError}</p>}
                {activeJson ? <StructuredCaptionEditor key={`${activeImage}/${captionPath}`} structure={activeStructure} draft={captionFields} onChange={setCaptionFields} disabled={captionLocked} readOnly={!canEdit}/> : canEdit ? <fieldset disabled={captionLocked}><TagChips caption={editCaption} onChange={setEditCaption} readOnly={savingCaption}/></fieldset> : <p className="whitespace-pre-wrap text-sm">{editCaption || t('dataset.noCaption', '（无 caption）')}</p>}
                <div className="flex justify-end space-x-2 pt-2">
                  <button
                    onClick={closeCaption} disabled={savingCaption}
                    className="px-4 py-2 text-sm rounded bg-slate-200 dark:bg-slate-700"
                  >
                    {t('common.cancel')}
                  </button>
                  <button
                    onClick={()=>void saveCaption().catch(()=>{})}
                    disabled={captionLocked}
                    className="px-4 py-2 text-sm rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
                    data-testid="caption-save-btn"
                  >
                    {savingCaption ? t('dataset.saving') : t('dataset.saveCaption')}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {images.error && (
        <div className="p-3 bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 rounded text-sm text-red-600 dark:text-red-400">
          {images.error}
        </div>
      )}

      {/* 空态：区分「筛选无结果」与「数据集真空」 */}
      {images.items.length === 0 && !images.loading && (
        images.q ? (
          <div className="p-10 text-center text-slate-400" data-testid="dataset-filter-empty">
            <SearchX className="w-10 h-10 mx-auto mb-2 opacity-40" />
            <div className="font-medium text-slate-500 dark:text-slate-300">{t('dataset.noFilterResults', '筛选无结果')}</div>
            <div className="text-xs mt-1">{t('dataset.noFilterResultsHint', '没有匹配当前过滤条件的图片，试试更换关键词。')}</div>
          </div>
        ) : (
          <div className="p-10 text-center text-slate-400" data-testid="dataset-empty">
            <ImageIcon className="w-10 h-10 mx-auto mb-2 opacity-40" />
            <div className="font-medium text-slate-500 dark:text-slate-300">{t('dataset.noImages')}</div>
            <div className="text-xs mt-1">{t('dataset.noImagesHint', '可点击「重新扫描」刷新索引，或确认目录中包含图片文件。')}</div>
          </div>
        )
      )}
    </div>
  );
}
