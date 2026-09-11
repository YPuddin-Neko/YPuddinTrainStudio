import { projectUrl, versionConfigUrl } from '../../utils/projectVersions';
import React from 'react';
import { Link, useParams, useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { apiClient, apiUrl } from '../../api/client';
import { DatasetInfo, Job, Plan } from '../../api/types';
import { useDatasetImages } from '../../api/hooks/useDatasetImages';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { MaskEditor } from '../../components/masks/MaskEditor';
import { formatApiError } from '../../utils/errors';
import { TagChips } from '../../components/TagChips';
import { formatBytes, formatParams, formatPercent } from '../../utils/format';
import './dataset-workspace.css';
import { useWorkspaceHeight } from '../../components/projects/useWorkspaceHeight';
import { ProjectWorkflow, NextStepLink } from '../../components/ProjectWorkflow';
import { useWorkspaceText } from '../../utils/workspaceText';
import {
  RefreshCcw,
  Trash2,
  Database,
  Layers,
  Image as ImageIcon,
  Search,
  SearchX,
  X,
  CheckSquare,
  Square,
  Zap,
  Brush,
} from 'lucide-react';

const THUMB_SIZE = 256;
const CARD_W = 176;
const CARD_H = 260;
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
  const navigate = useNavigate();
  const { t } = useTranslation();
  const text = useWorkspaceText();

  // locales 占位符为单花括号（{n}），i18next 默认插值（{{}}）不处理，需手工替换
  const tt = React.useCallback(
    (key: string, vars: Record<string, string | number>) =>
      Object.entries(vars).reduce((s, [k, v]) => s.split(`{${k}}`).join(String(v)), t(key)),
    [t]
  );

  const [info, setInfo] = React.useState<DatasetInfo | null>(null);
  const [indexProgress, setIndexProgress] = React.useState<{ done: number; total: number } | null>(null);
  const [activeImage, setActiveImage] = React.useState<string | null>(null);
  const [maskImage, setMaskImage] = React.useState<{ hash: string; relPath: string } | null>(null);
  const [actionError, setActionError] = React.useState('');
  const [editCaption, setEditCaption] = React.useState('');
  const [savingCaption, setSavingCaption] = React.useState(false);
  const [batchAdd, setBatchAdd] = React.useState('');
  const [batchRemove, setBatchRemove] = React.useState('');
  const [buckets, setBuckets] = React.useState<Plan['buckets'] | null>(null);
  const [showDistribution, setShowDistribution] = React.useState(false);
  const [busyAction, setBusyAction] = React.useState<string | null>(null);
  const [versionAccess, setVersionAccess] = React.useState<{ key: string; editable: boolean; archived: boolean; error?: string } | null>(null);
  const versionRequest = React.useRef<AbortController | null>(null);
  const versionKey = info?.source.version_id ? `${info.source.project_id}/${info.source.version_id}` : '';
  const canEdit = !!info && (!versionKey || versionAccess?.key === versionKey && versionAccess.editable);
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
  const gridRef = React.useRef<HTMLDivElement>(null);
  const navigationRef = useWorkspaceHeight('--dataset-navigation-height');
  const [scrollTop, setScrollTop] = React.useState(0);
  const [viewportH, setViewportH] = React.useState(600);
  const [viewportW, setViewportW] = React.useState(1200);

  const fetchInfo = React.useCallback(() => {
    if (!id) return;
    apiClient.get<DatasetInfo>(`/datasets/${id}`).then((data) => {
      setInfo(data);
      if (data.index_status !== 'indexing') setIndexProgress(null);
    }).catch(console.error);
  }, [id]);

  React.useEffect(() => {
    fetchInfo();
  }, [fetchInfo]);

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
  const cols = Math.max(1, Math.floor((containerWidth + GAP) / (CARD_W + GAP)));
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
    navigate(projectUrl(info.source.project_id || '', info.source.version_id, 'train'));
  };

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(e.currentTarget.scrollTop);
    // 接近底部时加载下一页
    const el = e.currentTarget;
    if (el.scrollTop + el.clientHeight >= el.scrollHeight - rowH * 3) {
      images.loadMore();
    }
  };

  const openEditor = (hash: string) => {
    const img = images.items.find((i) => i.hash === hash);
    if (!img) return;
    setActiveImage(hash);
    setEditCaption(img.caption || '');
  };

  const saveCaption = () => {
    if (!activeImage || !id || !canEdit) return;
    setSavingCaption(true);
    apiClient
      .put(`/datasets/${id}/images/${activeImage}/caption`, { caption: editCaption })
      .then(() => {
        images.updateCaption(activeImage, editCaption);
        setActiveImage(null);
      })
      .catch(console.error)
      .finally(() => setSavingCaption(false));
  };

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
        .then(() => navigate(projectUrl(info.source.project_id || '', info.source.version_id)))
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
  const activeImg = activeImage ? images.items.find((i) => i.hash === activeImage) : undefined;
  const activeSize = activeImg?.size;
  const datasetName = info?.source.path.replace(/[\\/]+$/, '').split(/[\\/]/).pop()?.replace(/^(?:d_[0-9a-f]+-)+/i, '') || id;

  return (
    <div className="dataset-workspace space-y-3" data-testid="dataset-page">
      {info?.source.project_id && <div className="dataset-workspace-navigation" ref={navigationRef}><Link to={projectUrl(info.source.project_id,info.source.version_id,'data')} className="dataset-workspace-return">{text('返回版本工作区','Return to version workspace')}</Link><ProjectWorkflow projectId={info.source.project_id} versionId={info.source.version_id} active="data" /></div>}
      {versionKey && !canEdit && <div className="flex flex-wrap items-center gap-2 rounded border border-slate-300 bg-slate-50 px-3 py-2 text-xs dark:border-slate-700 dark:bg-slate-900" role={versionAccess?.key === versionKey && versionAccess.error ? 'alert' : 'status'}>
        <span>{versionAccess?.key !== versionKey ? text('正在确认版本状态，暂以只读方式查看。', 'Checking version status. Viewing in read-only mode.') : versionAccess.error ? `${text('无法确认版本状态，编辑已暂停：', 'Cannot verify version status; editing is paused: ')}${versionAccess.error}` : versionAccess.archived ? text('此版本已归档，图片、标签和遮罩只读。', 'This version is archived. Images, captions and masks are read only.') : text('此版本暂不可编辑，当前为只读查看。', 'This version is not editable yet. Viewing in read-only mode.')}</span>
        <Link className="text-blue-600" to={projectUrl(info!.source.project_id || '', info!.source.version_id, 'data')}>{text('返回版本工作区', 'Return to version workspace')}</Link>
        <button type="button" className="text-blue-600" onClick={() => void checkVersionAccess()}>{t('common.refresh')}</button>
      </div>}
      {actionError && <div role="alert" className="rounded-lg bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950 dark:text-red-300">{actionError}</div>}
      {/* 顶部标题与动作 */}
      <div className="flex flex-wrap justify-between items-center gap-2">
        <div className="min-w-0 flex-1 basis-56">
          <h2 className="flex min-w-0 items-center gap-2 text-base font-semibold" title={info?.source.path}>
            <Database className="h-4 w-4 shrink-0 text-blue-500" />
            <span className="truncate">{datasetName}</span>
          </h2>
          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-slate-500">
            <span>{t('dataset.repeats')} ×{info?.source.repeats ?? '--'}</span>
            <span>{t('dataset.caption')}: {info?.source.caption_ext || '--'}</span>
            <span className={`px-2 py-0.5 rounded ${
              info?.index_status === 'ready'
                ? 'bg-green-100 text-green-700 dark:bg-green-950/40 dark:text-green-400'
                : info?.index_status === 'indexing'
                ? 'bg-blue-100 text-blue-700 dark:bg-blue-950/40 dark:text-blue-400'
                : 'bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-300'
            }`}>
              {statusLabel(info?.index_status)}
            </span>
            {info && <details className="min-w-0 max-w-full"><summary className="cursor-pointer">{text('查看完整路径', 'Full path')}</summary><code className="mt-1 block break-all text-[11px]">{info.source.path}</code></details>}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-1.5 [&>button]:px-2 [&>button]:py-1.5 [&>button]:text-xs">
          <button
            onClick={handleBucketPreview}
            disabled={busyAction === 'buckets'}
            className="flex items-center space-x-1.5 px-3 py-2 text-sm bg-slate-100 dark:bg-slate-800 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-50"
          >
            <Layers className="w-4 h-4" />
            <span>{busyAction === 'buckets' ? t('dataset.computing') : t('dataset.bucketPreview')}</span>
          </button>
          <button
            onClick={handlePrecache}
            disabled={!canEdit || busyAction === 'precache'}
            className="flex items-center space-x-1.5 px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            <Zap className="w-4 h-4" />
            <span>{busyAction === 'precache' ? t('dataset.enqueuing') : t('dataset.precache')}</span>
          </button>
          <button
            onClick={handleRescan}
            disabled={!canEdit || busyAction === 'rescan'}
            className="flex items-center space-x-1.5 px-3 py-2 text-sm bg-slate-100 dark:bg-slate-800 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-50"
          >
            <RefreshCcw className="w-4 h-4" />
            <span>{t('dataset.rescan')}</span>
          </button>
          <button
            onClick={handleDelete}
            disabled={!canEdit || busyAction === 'delete'}
            className="flex items-center space-x-1.5 px-3 py-2 text-sm bg-red-50 text-red-600 dark:bg-red-950/40 dark:text-red-400 rounded-lg hover:bg-red-100 disabled:opacity-50"
          >
            <Trash2 className="w-4 h-4" />
            <span>{t('dataset.remove')}</span>
          </button>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 dark:border-slate-700 dark:bg-slate-800">
        <div className="min-w-0 flex-1 basis-72 text-xs text-slate-500">
          <p>{canEdit ? text('点击图片编辑标签，用“编辑遮罩”绘制训练区域。', 'Click an image to edit captions; choose Edit mask to paint the training area.') : text('点击图片查看原图与标签。', 'Click an image to view the original and its caption.')}</p>
          <details className="mt-1"><summary className="cursor-pointer text-slate-600 dark:text-slate-300">{text('白色参与训练，黑色忽略 · 遮罩规则', 'White trains, black is ignored · Mask rules')}</summary><p className="mt-1 max-w-2xl">{text('没有独立遮罩时使用原图 Alpha；没有 Alpha 时全图参与。只有启用遮罩训练后才会生效。', 'Without a sidecar, image alpha is used; without alpha, the whole image participates. Enable masked training to apply these weights.')}</p></details>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          {canEdit && info?.source.project_id && <Link to={projectUrl(info.source.project_id || '', info.source.version_id, 'data')} className="px-1 py-1.5 text-blue-500">{text('添加数据', 'Add dataset')}</Link>}
          <button disabled={!canEdit || busyAction === 'mask-enable'} onClick={() => { setBusyAction('mask-enable'); setActionError(''); void enableMaskedTraining().catch((error) => setActionError(formatApiError(error))).finally(() => setBusyAction(null)); }} className="shrink-0 rounded-md bg-blue-600 px-2.5 py-1.5 text-white disabled:opacity-50">{text('启用遮罩并前往训练', 'Enable masks and open training')}</button>
          {info?.source.project_id && <NextStepLink to={projectUrl(info.source.project_id || '', info.source.version_id, 'models')}>{text('模型准备', 'Model setup')}</NextStepLink>}
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

      {/* 统计保持一行，分布与分桶按需展开 */}
      {stats && (
        <div className="rounded-lg border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-800" data-testid="dataset-overview">
          <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2">
            <dl className="flex flex-wrap items-center gap-x-5 gap-y-1 text-xs text-slate-500">
              <div className="flex items-baseline gap-2"><dt>{t('dataset.images')}</dt><dd className="font-mono text-sm font-semibold text-slate-800 dark:text-slate-100">{formatParams(stats.images)}</dd></div>
              <div className="flex items-baseline gap-2"><dt>{t('dataset.captioned')}</dt><dd><span className="font-mono text-slate-800 dark:text-slate-100">{formatParams(stats.captioned ?? 0)}</span> · {formatPercent(coverage)}</dd></div>
              <div className="flex items-baseline gap-2"><dt>{t('dataset.masks')}</dt><dd className="font-mono text-slate-800 dark:text-slate-100">{formatParams(stats.masks ?? 0)}</dd></div>
              {info?.cache?.latents && <div>{t('dataset.latents')}: <span className="font-mono">{info.cache.latents.cached}/{info.cache.latents.total}</span> · {t('dataset.text')}: <span className="font-mono">{info.cache.text?.cached ?? 0}/{info.cache.text?.total ?? 0}</span></div>}
            </dl>
            <button type="button" aria-expanded={showDistribution} aria-controls="dataset-distribution" onClick={() => setShowDistribution(value => !value)} className="flex items-center gap-1.5 py-1 text-xs text-blue-500"><Layers size={13}/>{text('分布与分桶', 'Distribution & buckets')}</button>
          </div>
          {showDistribution && <div id="dataset-distribution" className="grid gap-3 border-t border-slate-200 p-3 sm:grid-cols-3 dark:border-slate-700">
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
            <div className="text-xs text-slate-400 mb-1.5">{t('dataset.bucketsTitle')}</div>
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
        </div>
      )}

      {/* 搜索与批量操作条 */}
      <div className="dataset-browser-toolbar flex flex-wrap items-center gap-2 px-3 py-2 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
        <div className="flex items-center space-x-2 flex-1 min-w-0 basis-52">
          <Search className="w-4 h-4 text-slate-400" />
          <input
            type="text"
            value={images.q}
            onChange={(e) => images.setQ(e.target.value)}
            placeholder={t('dataset.filterPlaceholder')}
            className="min-w-0 flex-1 px-2 py-1 text-xs bg-transparent outline-none"
            data-testid="dataset-search"
          />
        </div>
        <div className="text-xs text-slate-400">
          {images.selected.size > 0
            ? tt('dataset.selected', { n: images.selected.size })
            : tt('dataset.imagesTotal', { n: images.total })}
        </div>
        <button disabled={!canEdit} onClick={images.selectAll} className="text-xs px-2 py-1.5 bg-slate-100 dark:bg-slate-700 rounded hover:bg-slate-200 flex items-center space-x-1">
          <CheckSquare className="w-3.5 h-3.5" />
          <span>{t('dataset.selectAll')}</span>
        </button>
        <button disabled={!canEdit} onClick={images.clearSelection} className="text-xs px-2 py-1.5 bg-slate-100 dark:bg-slate-700 rounded hover:bg-slate-200 flex items-center space-x-1">
          <Square className="w-3.5 h-3.5" />
          <span>{t('dataset.selectNone')}</span>
        </button>
        {canEdit && images.selected.size > 0 && (
          <>
            <input
              type="text"
              value={batchAdd}
              onChange={(e) => setBatchAdd(e.target.value)}
              placeholder={t('dataset.addTagsPlaceholder')}
              className="px-2 py-1.5 text-xs border rounded dark:bg-slate-900 dark:border-slate-600 w-36"
              data-testid="batch-add-input"
            />
            <input
              type="text"
              value={batchRemove}
              onChange={(e) => setBatchRemove(e.target.value)}
              placeholder={t('dataset.removeTagsPlaceholder')}
              className="px-2 py-1.5 text-xs border rounded dark:bg-slate-900 dark:border-slate-600 w-36"
            />
            <button
              onClick={applyBatchTags}
              disabled={busyAction === 'batch'}
              className="px-3 py-1.5 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
              data-testid="batch-apply-btn"
            >
              {busyAction === 'batch' ? t('dataset.applying') : t('dataset.applyToSelection')}
            </button>
          </>
        )}
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
              gridTemplateColumns: `repeat(${cols}, ${CARD_W}px)`,
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
                  style={{ width: CARD_W, height: CARD_H }}
                  data-testid={`image-card-${img.hash}`}
                >
                  <button type="button" onClick={() => openEditor(img.hash)} aria-label={`${canEdit ? text('编辑标签', 'Edit caption') : text('查看图片与标签', 'View image and caption')}: ${img.rel_path}`} className="block w-full">
                    <img src={apiUrl(`/datasets/${id}/images/${img.hash}/thumb?size=${THUMB_SIZE}`)} alt={img.rel_path} loading="lazy" className="w-full h-[170px] object-cover bg-slate-100 dark:bg-slate-900" />
                  </button>
                  <button
                    disabled={!canEdit}
                    aria-label={`${text('选择图片', 'Select image')}: ${img.rel_path}`}
                    onClick={() => images.toggleSelect(img.hash)}
                    className={`absolute top-1.5 left-1.5 w-5 h-5 rounded flex items-center justify-center text-[10px] font-bold ${
                      selected ? 'bg-blue-600 text-white' : 'bg-black/40 text-white opacity-0 group-hover:opacity-100'
                    }`}
                  >
                    {selected ? '✓' : ''}
                  </button>
                  <div className="p-1.5 text-[10px] text-slate-500 truncate" title={img.caption}>
                    <span className="font-mono">{img.width}×{img.height}</span> · {img.caption || t('dataset.noCaption', '（无 caption）')}
                  </div>
                  {canEdit && <button type="button" onClick={() => setMaskImage({ hash: img.hash, relPath: img.rel_path })} className="mx-1.5 flex min-h-9 w-[calc(100%-12px)] items-center justify-center gap-1 rounded border border-slate-300 px-2 py-1 text-xs hover:bg-slate-100 dark:border-slate-600 dark:hover:bg-slate-700"><Brush className="h-3.5 w-3.5" />{img.has_mask ? text('编辑遮罩 · 已有文件', 'Edit mask · saved') : text('编辑遮罩', 'Edit mask')}</button>}
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
        <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-6" onClick={() => setActiveImage(null)}>
          <div
            className="bg-white dark:bg-slate-800 rounded-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto p-6 space-y-4 shadow-xl"
            onClick={(e) => e.stopPropagation()}
            data-testid="caption-editor"
          >
            <div className="flex justify-between items-center">
              <h3 className="font-semibold text-lg font-mono">{activeImg?.rel_path}</h3>
              <button onClick={() => setActiveImage(null)} className="text-slate-400 hover:text-slate-600" title={t('common.close')}>
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
                {canEdit && <button type="button" onClick={() => { if (activeImg) { setMaskImage({ hash: activeImg.hash, relPath: activeImg.rel_path }); setActiveImage(null); } }} className="flex items-center gap-2 rounded-lg border border-slate-300 px-3 py-2 text-sm dark:border-slate-600"><Brush className="h-4 w-4" />{text('编辑这张图片的训练遮罩', 'Edit this image’s training mask')}</button>}
                <div className="text-xs text-slate-400">{t('dataset.captionEditorTitle')}</div>
                {canEdit ? <TagChips caption={editCaption} onChange={setEditCaption} /> : <p className="whitespace-pre-wrap text-sm">{editCaption || t('dataset.noCaption', '（无 caption）')}</p>}
                <div className="flex justify-end space-x-2 pt-2">
                  <button
                    onClick={() => setActiveImage(null)}
                    className="px-4 py-2 text-sm rounded bg-slate-200 dark:bg-slate-700"
                  >
                    {t('common.cancel')}
                  </button>
                  <button
                    onClick={saveCaption}
                    disabled={!canEdit || savingCaption}
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
