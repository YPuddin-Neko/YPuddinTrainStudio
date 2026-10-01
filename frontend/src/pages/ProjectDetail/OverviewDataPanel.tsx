import DatasetLink from '../../components/datasets/DatasetLink';
import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { ArrowRight, ChevronLeft, ChevronRight, FolderPlus, Images, Search } from 'lucide-react';
import { apiClient, apiUrl } from '../../api/client';
import type { DatasetImage, DatasetImagesPage, DatasetInfo } from '../../api/types';
import type { components } from '../../api/generated';
import Dialog from '../../components/Dialog';
import { LazyImage, LoadingNote } from '../../components/Loading';
import StudioSelect from '../../components/StudioSelect';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import type { OverviewDataset } from './ProjectOverview';
import { SlidingIndicator } from '../../components/motion';
import { useGridPageSize } from '../../api/hooks/useGridPageSize';
import ImageSortSelect, { type ImageSort } from '../../components/datasets/ImageSortSelect';

type DatasetOverview = {
  dataset_id: string;
  folders: { path: string; count: number }[];
  stats: DatasetInfo['stats'];
  caption_stats: components['schemas']['DatasetCaptionStats'];
  images: DatasetImagesPage;
};
type Preview = DatasetImage & { source: string };
const basename = (path: string) => path.split(/[\\/]/).filter(Boolean).pop()?.replace(/^(?:d_[0-9a-f]+-)+/i, '') || path;
const TAG_LIMIT = 30;

/** Mirrors the loaded preview so the page keeps its shape while images and statistics arrive. */
function PreviewSkeleton({ label, gridRef }: { label: string; gridRef?: React.Ref<HTMLDivElement> }) {
  const bars = (count: number) => <div className="overview-skeleton-bars">{Array.from({ length: count }, (_, index) => <span key={index} className="overview-skeleton-bar"><span className="ui-skeleton"/><span className="ui-skeleton"/></span>)}</div>;
  return <div className="overview-data-skeleton" role="status" aria-label={label}>
    <div className="overview-data-grid" aria-hidden="true">
      <div className="overview-panel overview-gallery">
        <span className="ui-skeleton overview-skeleton-title"/><span className="ui-skeleton overview-skeleton-line"/>
        <div ref={gridRef} className="overview-thumbnails">{Array.from({ length: 12 }, (_, index) => <span key={index} className="overview-skeleton-tile"><span className="ui-skeleton"/><span className="ui-skeleton"/><span className="ui-skeleton"/></span>)}</div>
      </div>
      <div className="overview-panel overview-tags"><span className="ui-skeleton overview-skeleton-title"/><span className="ui-skeleton overview-skeleton-line"/>{bars(8)}</div>
    </div>
    <div className="overview-distribution-grid" aria-hidden="true">{[0, 1].map(key => <div key={key} className="overview-panel"><span className="ui-skeleton overview-skeleton-title"/>{bars(4)}</div>)}</div>
  </div>;
}
const ready = (row: OverviewDataset) => row.index_status === 'ready' && !!row.stats && !row.stats.error;

export default function OverviewDataPanel({ datasets, workspaceUrl, projectId, versionId }: {
  datasets: OverviewDataset[]; workspaceUrl: string; projectId: string; versionId?: string;
}) {
  const text = useWorkspaceText();
  const gridPage = useGridPageSize(16);
  const [sort, setSort] = React.useState<ImageSort>('filename');
  const [role, setRole] = React.useState<'train' | 'reg'>('train');
  const [source, setSource] = React.useState('all');
  const [folder, setFolder] = React.useState('');
  const [search, setSearch] = React.useState('');
  const [query, setQuery] = React.useState('');
  const [preview, setPreview] = React.useState<Preview | null>(null);
  React.useEffect(() => { const timer = window.setTimeout(() => setQuery(search.trim()), 250); return () => window.clearTimeout(timer); }, [search]);
  const roleRows = datasets.filter(row => !!row.source.is_reg === (role === 'reg'));
  const selection = roleRows.some(row => row.source.id === source) ? source : 'all';
  const rows = selection === 'all' ? roleRows : roleRows.filter(row => row.source.id === selection);
  const indexedRows = rows.filter(ready);
  const incomplete = indexedRows.length !== rows.length;
  const ids = indexedRows.map(row => row.source.id);
  const scope = ['overview-data', projectId, versionId, role, ids, folder, indexedRows.map(row => row.stats), sort];
  const overview = useQuery({
    queryKey: [...scope, gridPage.pageSize, query],
    // The skeleton's grid is measured first, so the first request already fills whole rows.
    enabled: ids.length > 0 && gridPage.ready,
    queryFn: async ({ signal }) => ({ query, entries: await Promise.all(ids.map(async id => {
      const data = await apiClient.get<DatasetOverview>(`/datasets/${encodeURIComponent(id)}/overview`, {
        params: { project_id: projectId, version_id: versionId, folder: selection === 'all' ? '' : folder, q: query || undefined, page: 1, page_size: gridPage.pageSize, sort }, signal, silent: true,
      });
      if (data.dataset_id !== id || !Array.isArray(data.images?.items) || !Array.isArray(data.caption_stats?.tags) || !Array.isArray(data.folders)) {
        throw new Error(text('数据概览响应格式不完整，请重新读取。', 'The dataset overview response is incomplete. Please retry.'));
      }
      return data;
    })) }),
    staleTime: 30_000,
    // A new search or column count keeps the current images on screen until the new ones arrive.
    placeholderData: (previous, previousQuery) => previousQuery && JSON.stringify(previousQuery.queryKey.slice(0, scope.length)) === JSON.stringify(scope) ? previous : undefined,
  });
  const refreshing = overview.isPlaceholderData;
  const searching = refreshing && overview.data?.query !== query;
  useEventStream(EVENT_TYPES.DATASET_CHANGED, event => { if (ids.includes(event.dataset_id)) void overview.refetch(); });
  const data = overview.data?.entries || [];
  const tags = new Map<string, { tag: string; count: number }>();
  const resolutions = new Map<string, number>();
  const ratios = new Map<string, number>();
  for (const entry of data) {
    for (const tag of entry.caption_stats.tags) {
      const key = tag.tag.toLocaleLowerCase();
      const previous = tags.get(key);
      tags.set(key, { tag: previous?.tag || tag.tag, count: (previous?.count || 0) + tag.count });
    }
    for (const item of entry.stats.resolutions || []) {
      const name = `${item.w} × ${item.h}`;
      resolutions.set(name, (resolutions.get(name) || 0) + item.count);
    }
    for (const item of entry.stats.ar_hist || []) ratios.set(item.ar, (ratios.get(item.ar) || 0) + item.count);
  }
  const topTags = [...tags.values()].sort((a, b) => b.count - a.count || a.tag.localeCompare(b.tag)).slice(0, TAG_LIMIT);
  const total = data.reduce((sum, entry) => sum + entry.stats.images, 0);
  const matching = data.reduce((sum, entry) => sum + entry.images.total, 0);
  // Round-robin sampling keeps a large first source from hiding the other sources.
  const pictures: Preview[] = [];
  for (let index = 0; index < gridPage.pageSize && pictures.length < gridPage.pageSize; index++) for (const entry of data) {
    if (entry.images.items[index] && pictures.length < gridPage.pageSize) pictures.push({ ...entry.images.items[index], source: entry.dataset_id });
  }
  const charts = [
    { title: text('图片分辨率', 'Image resolutions'), items: [...resolutions.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8), truncated: resolutions.size > 8 },
    { title: text('图片长宽比', 'Image aspect ratios'), items: [...ratios.entries()].sort((a, b) => Number(a[0]) - Number(b[0])), truncated: false },
  ];
  const sourceUrl = (id: string) => `/datasets/${encodeURIComponent(id)}?${new URLSearchParams({ project: projectId, ...(versionId ? { version: versionId } : {}) })}`;
  const clearSearch = () => { setSearch(''); setQuery(''); setPreview(null); };
  const changeSource = (value: string) => { setSource(value); setFolder(''); clearSearch(); };
  const countLabel = (value: 'train' | 'reg') => {
    const matchingRows = datasets.filter(row => !!row.source.is_reg === (value === 'reg'));
    return matchingRows.every(ready) ? matchingRows.reduce((sum, row) => sum + (row.stats?.images || 0), 0) : '—';
  };
  const folderOptions = selection === 'all' ? [] : data[0]?.folders || [];
  const captionCount = data.reduce((sum, entry) => sum + entry.caption_stats.captioned, 0);
  const invalidCount = data.reduce((sum, entry) => sum + entry.caption_stats.invalid, 0);
  const previewIndex = preview ? pictures.findIndex(item => item.source === preview.source && item.rel_path === preview.rel_path) : -1;
  const viewAllUrl = selection === 'all' ? `${workspaceUrl}&data_step=${role === 'reg' ? 'reg' : 'datasets'}#version-datasets` : sourceUrl(selection);

  return <section className="overview-data-panel" aria-label={text('版本数据分布', 'Version data distributions')}>
    <div className="overview-role-toolbar">
      <div className="overview-role-switch ui-segmented" role="group" aria-label={text('数据用途', 'Dataset role')}>
        {(['train', 'reg'] as const).map(value => <button type="button" key={value} aria-pressed={role === value} onClick={() => { setRole(value); setSource('all'); setFolder(''); clearSearch(); }}>{value === 'train' ? text('训练集', 'Training set') : text('正则集', 'Regularization')} <strong>{countLabel(value)}</strong></button>)}
        <SlidingIndicator className="ui-segmented-thumb"/>
      </div>
    </div>
    <div className="overview-data-toolbar">
      <StudioSelect aria-label={text('概览数据集', 'Overview dataset')} value={selection} onValueChange={changeSource} options={[{ value: 'all', label: text('全部数据集', 'All datasets') }, ...roleRows.map(row => ({ value: row.source.id, label: `${basename(row.source.path)} · ${ready(row) ? row.stats?.images : '—'} ${text('张', 'images')}` }))]}/>
      <label className="overview-gallery-search"><Search size={15}/><input aria-label={text('筛选概览图片', 'Filter overview images')} placeholder={text('标签或文件名', 'Tag or filename')} value={search} onChange={event => setSearch(event.target.value)}/></label>
    </div>
    {folderOptions.length > 0 && <label className="overview-folder-filter">{text('子目录', 'Subfolder')}<StudioSelect aria-label={text('概览子目录', 'Overview subfolder')} value={folder} onValueChange={value => { setFolder(value); clearSearch(); }} options={[{ value: '', label: text('全部子目录', 'All subfolders') }, ...folderOptions.map(item => ({ value: item.path, label: `${item.path} · ${item.count}` }))]}/></label>}
    {rows.length === 0 ? <div className="overview-panel overview-empty-data"><Images size={30} aria-hidden="true"/><span>{role === 'reg' ? text('当前版本没有正则集。', 'No regularization set in this version.') : text('当前版本还没有训练图片。', 'No training images in this version.')}</span><Link className="ui-btn" to={`${workspaceUrl}&data_step=${role === 'reg' ? 'reg' : 'datasets'}`}><FolderPlus size={15}/>{text('添加数据', 'Add data')}</Link></div> : <>
      {incomplete && <p role="status" className="overview-index-note">{text(`${rows.length - indexedRows.length} 个数据集索引尚未就绪；分布仅显示已就绪的数据。`, `${rows.length - indexedRows.length} datasets are not indexed yet; distributions show only ready datasets.`)}<Link className="ui-link" to={`${workspaceUrl}&data_step=datasets#version-datasets`}>{text('查看数据集状态', 'View dataset status')}</Link></p>}
      {indexedRows.length > 0 && (overview.error ? <div className="overview-panel overview-inline-error" role="alert"><span>{formatApiError(overview.error)}</span><button type="button" className="ui-btn ui-btn-sm" onClick={() => void overview.refetch()}>{text('重新读取数据分布', 'Reload data distributions')}</button></div> : overview.isPending ? <PreviewSkeleton gridRef={gridPage.gridRef} label={text('正在读取图片与分布…', 'Loading images and distributions…')}/> : <>
        <div className="overview-data-grid">
          <section className="overview-panel overview-gallery"><div className="overview-panel-heading"><h3>{role === 'reg' ? text('正则图预览', 'Regularization preview') : text('训练集预览', 'Training set preview')}</h3><DatasetLink className="ui-link" to={viewAllUrl}>{text('查看全部', 'View all')}<ArrowRight size={13}/></DatasetLink></div>
            <div className="overview-preview-controls"><p className="overview-section-detail">{searching ? <LoadingNote label={text('正在筛选图片…', 'Filtering images…')}/> : query ? text(`匹配 ${matching} 张 · 预览 ${pictures.length} 张`, `${matching} matches · ${pictures.length} previewed`) : text(`共 ${total} 张 · 预览 ${pictures.length} 张`, `${total} images · ${pictures.length} previewed`)}</p><ImageSortSelect value={sort} onChange={setSort}/></div>
            {pictures.length ? <div ref={gridPage.gridRef} className={`overview-thumbnails${refreshing ? ' is-refreshing' : ''}`} aria-busy={refreshing || undefined}>{pictures.map(item => <button type="button" key={`${item.source}/${item.rel_path}`} onClick={() => setPreview(item)} aria-label={text(`预览图片：${item.rel_path}`, `Preview image: ${item.rel_path}`)}><span className="overview-thumbnail-image"><LazyImage loading="lazy" src={apiUrl(`/datasets/${encodeURIComponent(item.source)}/images/${encodeURIComponent(item.hash)}/thumb?size=256`)} alt=""/></span><span title={item.rel_path}>{basename(item.rel_path)}</span><small>{item.width} × {item.height}</small></button>)}</div> : <p className="overview-section-detail">{text('没有匹配的图片。', 'No matching images.')}</p>}
            <div className="overview-source-summary">{rows.map(row => <DatasetLink key={row.source.id} to={sourceUrl(row.source.id)}><strong>{basename(row.source.path)}</strong><span>{ready(row) ? row.stats?.images : '—'} {text('张', 'images')} · ×{row.source.repeats ?? 1}{role === 'reg' ? ` · ${text('权重', 'Weight')} ${row.source.prior_weight ?? 1}` : ''}</span><ArrowRight size={13}/></DatasetLink>)}</div>
          </section>
          <section className="overview-panel overview-tags"><div className="overview-panel-heading"><h3>{text('标签分布', 'Tag distribution')}</h3><Link className="ui-link" to={`${workspaceUrl}&data_step=captions${selection !== 'all' ? `&dataset=${encodeURIComponent(selection)}` : ''}`}>{text('编辑标签', 'Edit tags')}<ArrowRight size={13}/></Link></div>
            <p className="overview-tag-coverage">{captionCount} / {total} {text('张已标注', 'images captioned')} · {tags.size} {text('种标签', 'unique tags')}{invalidCount > 0 && ` · ${invalidCount} ${text('份标签读取失败', 'unreadable captions')}`}</p>
            {topTags.length ? <div className="overview-tag-scroll"><ul className="overview-bars">{topTags.map(item => <li key={item.tag}><div><span title={item.tag}>{item.tag}</span><strong>{item.count}</strong></div><meter min={0} max={Math.max(1, ...topTags.map(tag => tag.count))} value={item.count} aria-label={item.tag}/></li>)}</ul></div> : <p className="overview-section-detail">{text('暂无可统计的标签词。自然语言描述可在标签页面查看。', 'No tag tokens to count. View natural-language captions on the caption page.')}</p>}
            {tags.size > TAG_LIMIT && <p className="overview-section-detail">{text(`显示频次最高的 ${TAG_LIMIT} 个标签`, `Showing the ${TAG_LIMIT} most frequent tags`)}</p>}
          </section>
        </div>
        <div className="overview-distribution-grid">{charts.map(chart => <section className="overview-panel" key={chart.title}><div className="overview-panel-heading"><h3>{chart.title}</h3>{chart.truncated && <span className="overview-section-detail">{text('最常见的 8 种尺寸', '8 most common sizes')}</span>}</div><ul className="overview-bars">{chart.items.map(([label, count]) => <li key={label}><div><span>{label}</span><strong>{count}</strong></div><meter min={0} max={Math.max(1, ...chart.items.map(item => item[1]))} value={count} aria-label={`${chart.title} ${label}`}/></li>)}</ul>{!chart.items.length && <p className="overview-section-detail">{text('暂无尺寸统计', 'No dimension statistics')}</p>}</section>)}</div>
      </>)}
    </>}
    {preview && <Dialog title={text('图片预览', 'Image preview')} onClose={() => setPreview(null)} wide><div className="overview-preview-dialog"><div className="overview-preview-frame"><LazyImage key={`${preview.source}/${preview.hash}`} src={apiUrl(`/datasets/${encodeURIComponent(preview.source)}/images/${encodeURIComponent(preview.hash)}/file`)} alt={preview.rel_path}/></div><div className="overview-preview-meta"><strong>{preview.rel_path}</strong><span>{preview.width} × {preview.height}</span></div><p>{preview.caption || text('暂无标签', 'No caption')}</p><footer><button type="button" className="ui-btn" disabled={previewIndex <= 0} onClick={() => setPreview(pictures[previewIndex - 1])}><ChevronLeft size={16}/>{text('上一张', 'Previous')}</button><button type="button" className="ui-btn" disabled={previewIndex < 0 || previewIndex >= pictures.length - 1} onClick={() => setPreview(pictures[previewIndex + 1])}>{text('下一张', 'Next')}<ChevronRight size={16}/></button><DatasetLink className="ui-link" to={sourceUrl(preview.source)}>{text('打开所属数据集', 'Open dataset')}<ArrowRight size={14}/></DatasetLink></footer></div></Dialog>}
  </section>;
}
