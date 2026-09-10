import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { apiClient } from '../../api/client';
import { DatasetInfo, Job, Plan } from '../../api/types';
import { useDatasetImages } from '../../api/hooks/useDatasetImages';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { TagChips } from '../../components/TagChips';
import {
  RefreshCcw,
  Trash2,
  Database,
  Layers,
  Image as ImageIcon,
  Search,
  X,
  CheckSquare,
  Square,
  Zap,
} from 'lucide-react';

const THUMB_SIZE = 256;
const CARD_W = 176;
const CARD_H = 224;
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

  const [info, setInfo] = React.useState<DatasetInfo | null>(null);
  const [indexProgress, setIndexProgress] = React.useState<{ done: number; total: number } | null>(null);
  const [activeImage, setActiveImage] = React.useState<string | null>(null);
  const [editCaption, setEditCaption] = React.useState('');
  const [savingCaption, setSavingCaption] = React.useState(false);
  const [batchAdd, setBatchAdd] = React.useState('');
  const [batchRemove, setBatchRemove] = React.useState('');
  const [buckets, setBuckets] = React.useState<Plan['buckets'] | null>(null);
  const [busyAction, setBusyAction] = React.useState<string | null>(null);

  const images = useDatasetImages(id);
  const gridRef = React.useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = React.useState(0);
  const [viewportH, setViewportH] = React.useState(600);

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
    const onResize = () => setViewportH(el.clientHeight || 600);
    onResize();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  // 虚拟滚动窗口计算
  const containerWidth = Math.max(320, gridRef.current?.clientWidth || 1200);
  const cols = Math.max(1, Math.floor((containerWidth + GAP) / (CARD_W + GAP)));
  const rows = Math.ceil(images.items.length / cols);
  const rowH = CARD_H + GAP;
  const startRow = Math.max(0, Math.floor(scrollTop / rowH) - 2);
  const endRow = Math.min(rows, Math.ceil((scrollTop + viewportH) / rowH) + 2);
  const visibleItems = images.items.slice(startRow * cols, endRow * cols);

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
    if (!activeImage || !id) return;
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
    if (!id || images.selected.size === 0) return;
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
    if (!id) return;
    setBusyAction('rescan');
    apiClient.post(`/datasets/${id}/rescan`, {})
      .then(fetchInfo)
      .catch(console.error)
      .finally(() => setBusyAction(null));
  };

  const handleDelete = () => {
    if (!id || !info) return;
    if (window.confirm(`Remove dataset registration for "${info.source.path}"? Files on disk will NOT be deleted.`)) {
      setBusyAction('delete');
      apiClient.delete(`/datasets/${id}`)
        .then(() => navigate(`/projects/${info.source.project_id}`))
        .catch(console.error)
        .finally(() => setBusyAction(null));
    }
  };

  const handlePrecache = () => {
    if (!id || !info) return;
    setBusyAction('precache');
    apiClient.post<Job>(`/jobs`, {
      type: 'cache',
      name: `cache-${info.source.path.split('/').pop()}-${Date.now()}`,
      project_id: info.source.project_id,
    })
      .then(() => alert('Pre-cache job enqueued.'))
      .catch(console.error)
      .finally(() => setBusyAction(null));
  };

  const handleBucketPreview = () => {
    if (!info) return;
    setBusyAction('buckets');
    apiClient.get<any>(`/projects/${info.source.project_id}/config`)
      .then((config) => apiClient.post<Plan>('/plan', { config, dataset_ids: [id] }))
      .then((plan) => setBuckets(plan.buckets || []))
      .catch(console.error)
      .finally(() => setBusyAction(null));
  };

  const stats = info?.stats;
  const coverage = stats?.images ? Math.round(((stats.captioned || 0) / stats.images) * 100) : 0;

  return (
    <div className="space-y-6" data-testid="dataset-page">
      {/* 顶部标题与动作 */}
      <div className="flex flex-wrap justify-between items-center gap-3">
        <div>
          <h2 className="text-2xl font-bold flex items-center space-x-2">
            <Database className="w-6 h-6 text-blue-500" />
            <span className="font-mono text-xl">{info?.source.path || id}</span>
          </h2>
          <div className="flex items-center space-x-3 mt-1 text-xs text-slate-400">
            <span>repeats ×{info?.source.repeats}</span>
            <span>caption: {info?.source.caption_ext}</span>
            <span className={`px-2 py-0.5 rounded ${
              info?.index_status === 'ready'
                ? 'bg-green-100 text-green-700 dark:bg-green-950/40 dark:text-green-400'
                : info?.index_status === 'indexing'
                ? 'bg-blue-100 text-blue-700 dark:bg-blue-950/40 dark:text-blue-400'
                : 'bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-300'
            }`}>
              {info?.index_status}
            </span>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={handleBucketPreview}
            disabled={busyAction === 'buckets'}
            className="flex items-center space-x-1.5 px-3 py-2 text-sm bg-slate-100 dark:bg-slate-800 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-50"
          >
            <Layers className="w-4 h-4" />
            <span>{busyAction === 'buckets' ? 'Computing…' : 'Bucket Preview'}</span>
          </button>
          <button
            onClick={handlePrecache}
            disabled={busyAction === 'precache'}
            className="flex items-center space-x-1.5 px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            <Zap className="w-4 h-4" />
            <span>{busyAction === 'precache' ? 'Enqueuing…' : 'Pre-cache'}</span>
          </button>
          <button
            onClick={handleRescan}
            disabled={busyAction === 'rescan'}
            className="flex items-center space-x-1.5 px-3 py-2 text-sm bg-slate-100 dark:bg-slate-800 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-50"
          >
            <RefreshCcw className="w-4 h-4" />
            <span>Rescan</span>
          </button>
          <button
            onClick={handleDelete}
            disabled={busyAction === 'delete'}
            className="flex items-center space-x-1.5 px-3 py-2 text-sm bg-red-50 text-red-600 dark:bg-red-950/40 dark:text-red-400 rounded-lg hover:bg-red-100 disabled:opacity-50"
          >
            <Trash2 className="w-4 h-4" />
            <span>Remove</span>
          </button>
        </div>
      </div>

      {/* 索引进度条 */}
      {(info?.index_status === 'indexing' || indexProgress) && (
        <div className="p-4 bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800 rounded-xl" data-testid="index-progress">
          <div className="flex justify-between text-xs text-blue-700 dark:text-blue-300 mb-1.5">
            <span>Indexing dataset…</span>
            <span>{indexProgress ? `${indexProgress.done} / ${indexProgress.total}` : '…'}</span>
          </div>
          <div className="w-full bg-blue-200 dark:bg-blue-900 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all"
              style={{ width: `${indexProgress && indexProgress.total > 0 ? (indexProgress.done / indexProgress.total) * 100 : 20}%` }}
            />
          </div>
        </div>
      )}

      {/* 概览卡 */}
      {stats && (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4" data-testid="dataset-overview">
          <div className="p-4 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 space-y-3">
            <div className="text-xs text-slate-400">Images</div>
            <div className="text-2xl font-bold">{stats.images ?? '--'}</div>
            <div className="text-xs text-slate-400">
              Captioned: {stats.captioned ?? 0} ({coverage}%) · Masks: {stats.masks ?? 0}
            </div>
            {info?.cache?.latents && (
              <div className="text-xs text-slate-400">
                Latents: {info.cache.latents.cached}/{info.cache.latents.total} · Text: {info.cache.text?.cached ?? 0}/{info.cache.text?.total ?? 0}
              </div>
            )}
          </div>
          <div className="p-4 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
            <Histogram
              label="Resolutions"
              barColor="bg-blue-400"
              data={(stats.resolutions || []).map((r) => ({ name: `${r.w}×${r.h}`, count: r.count }))}
            />
          </div>
          <div className="p-4 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
            <Histogram
              label="Aspect Ratio"
              barColor="bg-indigo-400"
              data={(stats.ar_hist || []).map((r) => ({ name: r.ar, count: r.count }))}
            />
          </div>
          <div className="p-4 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-auto">
            <div className="text-xs text-slate-400 mb-1.5">Buckets (project draft)</div>
            {buckets ? (
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-slate-400">
                    <th className="text-left">W×H</th>
                    <th className="text-right">Items</th>
                    <th className="text-right">Batches</th>
                  </tr>
                </thead>
                <tbody>
                  {buckets.map((b, i) => (
                    <tr key={i}>
                      <td className="font-mono">{b.w}×{b.h}</td>
                      <td className="text-right">{(b as any).items ?? (b as any).images}</td>
                      <td className="text-right">{b.batches}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="text-xs text-slate-400">Click "Bucket Preview" to compute.</div>
            )}
          </div>
        </div>
      )}

      {/* 搜索与批量操作条 */}
      <div className="flex flex-wrap items-center gap-3 p-3 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
        <div className="flex items-center space-x-2 flex-1 min-w-[220px]">
          <Search className="w-4 h-4 text-slate-400" />
          <input
            type="text"
            value={images.q}
            onChange={(e) => images.setQ(e.target.value)}
            placeholder="Filter by tag / caption…"
            className="flex-1 px-2 py-1.5 text-sm bg-transparent outline-none"
            data-testid="dataset-search"
          />
        </div>
        <div className="text-xs text-slate-400">
          {images.selected.size > 0 ? `${images.selected.size} selected` : `${images.total} images`}
        </div>
        <button onClick={images.selectAll} className="text-xs px-2 py-1.5 bg-slate-100 dark:bg-slate-700 rounded hover:bg-slate-200 flex items-center space-x-1">
          <CheckSquare className="w-3.5 h-3.5" />
          <span>All</span>
        </button>
        <button onClick={images.clearSelection} className="text-xs px-2 py-1.5 bg-slate-100 dark:bg-slate-700 rounded hover:bg-slate-200 flex items-center space-x-1">
          <Square className="w-3.5 h-3.5" />
          <span>None</span>
        </button>
        {images.selected.size > 0 && (
          <>
            <input
              type="text"
              value={batchAdd}
              onChange={(e) => setBatchAdd(e.target.value)}
              placeholder="+ tags (comma)"
              className="px-2 py-1.5 text-xs border rounded dark:bg-slate-900 dark:border-slate-600 w-36"
              data-testid="batch-add-input"
            />
            <input
              type="text"
              value={batchRemove}
              onChange={(e) => setBatchRemove(e.target.value)}
              placeholder="− tags (comma)"
              className="px-2 py-1.5 text-xs border rounded dark:bg-slate-900 dark:border-slate-600 w-36"
            />
            <button
              onClick={applyBatchTags}
              disabled={busyAction === 'batch'}
              className="px-3 py-1.5 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
              data-testid="batch-apply-btn"
            >
              {busyAction === 'batch' ? 'Applying…' : 'Apply to selection'}
            </button>
          </>
        )}
      </div>

      {/* 虚拟滚动图片网格 */}
      <div
        ref={gridRef}
        onScroll={handleScroll}
        className="relative overflow-y-auto bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700"
        style={{ height: 600 }}
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
                  key={img.hash}
                  className={`relative rounded-lg overflow-hidden border cursor-pointer group ${
                    selected ? 'border-blue-500 ring-2 ring-blue-500/40' : 'border-slate-200 dark:border-slate-700'
                  }`}
                  style={{ width: CARD_W, height: CARD_H }}
                  data-testid={`image-card-${img.hash}`}
                >
                  <img
                    src={`/api/datasets/${id}/images/${img.hash}/thumb?size=${THUMB_SIZE}`}
                    alt={img.rel_path}
                    loading="lazy"
                    className="w-full h-[170px] object-cover bg-slate-100 dark:bg-slate-900"
                    onClick={() => openEditor(img.hash)}
                  />
                  <button
                    onClick={() => images.toggleSelect(img.hash)}
                    className={`absolute top-1.5 left-1.5 w-5 h-5 rounded flex items-center justify-center text-[10px] font-bold ${
                      selected ? 'bg-blue-600 text-white' : 'bg-black/40 text-white opacity-0 group-hover:opacity-100'
                    }`}
                  >
                    {selected ? '✓' : ''}
                  </button>
                  <div className="p-1.5 text-[10px] text-slate-500 truncate" title={img.caption}>
                    <span className="font-mono">{img.width}×{img.height}</span> · {img.caption || '(no caption)'}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
        {images.loading && (
          <div className="sticky bottom-0 text-center text-xs text-slate-400 py-2 bg-white/80 dark:bg-slate-800/80">Loading…</div>
        )}
      </div>

      {/* 大图 + caption 编辑抽屉 */}
      {activeImage && (
        <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-6" onClick={() => setActiveImage(null)}>
          <div
            className="bg-white dark:bg-slate-800 rounded-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto p-6 space-y-4 shadow-xl"
            onClick={(e) => e.stopPropagation()}
            data-testid="caption-editor"
          >
            <div className="flex justify-between items-center">
              <h3 className="font-semibold text-lg font-mono">{images.items.find((i) => i.hash === activeImage)?.rel_path}</h3>
              <button onClick={() => setActiveImage(null)} className="text-slate-400 hover:text-slate-600">
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <img
                src={`/api/datasets/${id}/images/${activeImage}/file`}
                alt="preview"
                className="w-full max-h-[50vh] object-contain rounded-lg bg-slate-100 dark:bg-slate-900"
              />
              <div className="space-y-3">
                <div className="text-xs text-slate-400">Caption (comma-separated tags)</div>
                <TagChips caption={editCaption} onChange={setEditCaption} />
                <div className="flex justify-end space-x-2 pt-2">
                  <button
                    onClick={() => setActiveImage(null)}
                    className="px-4 py-2 text-sm rounded bg-slate-200 dark:bg-slate-700"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={saveCaption}
                    disabled={savingCaption}
                    className="px-4 py-2 text-sm rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
                    data-testid="caption-save-btn"
                  >
                    {savingCaption ? 'Saving…' : 'Save caption'}
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

      {images.items.length === 0 && !images.loading && (
        <div className="p-10 text-center text-slate-400">
          <ImageIcon className="w-10 h-10 mx-auto mb-2 opacity-40" />
          No images found.
        </div>
      )}
    </div>
  );
}
