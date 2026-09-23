import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { AlertCircle, ArrowUpRight, Folder, Images, Loader2, RefreshCw } from 'lucide-react';
import { apiClient, apiUrl } from '../../api/client';
import type { DatasetInfo, DatasetImagesPage, DatasetSource } from '../../api/types';
import { useWorkspaceText } from '../../utils/workspaceText';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import './project-dataset-cards.css';

export type WorkspaceDataset = { source: DatasetSource; stats?: DatasetInfo['stats']; index_status?: string };
interface Props { datasets: WorkspaceDataset[]; projectId: string; versionId?: string; onRefresh: () => void }

function DatasetCard({ dataset, projectId, versionId }: Omit<Props, 'datasets' | 'onRefresh'> & { dataset: WorkspaceDataset }) {
  const text = useWorkspaceText();
  const { source, stats, index_status: status } = dataset;
  const name = source.path.replace(/\\/g, '/').split('/').filter(Boolean).pop()?.replace(/^(?:d_[0-9a-f]+-)+/i, '') || text('未命名目录', 'Unnamed folder');
  const preview = useQuery({
    queryKey: ['dataset-card-preview', source.id, status, stats?.images],
    queryFn: ({ signal }) => apiClient.get<DatasetImagesPage>(`/datasets/${encodeURIComponent(source.id)}/images`, { params: { page: 1, page_size: 1 }, signal, silent: true }),
    enabled: status !== 'indexing' && status !== 'failed' && !(status === 'ready' && stats?.images === 0),
    staleTime: 60_000, retry: false,
  });
  const cover = preview.data?.items[0];
  const statusText = status === 'indexing' ? text('索引中', 'Indexing') : status === 'failed' ? text('索引失败', 'Index failed') : status === 'stale' ? text('需要更新索引', 'Index needs updating') : status === 'ready' ? text('已就绪', 'Ready') : text('待检查', 'Not checked');
  const placeholder = status === 'indexing' ? text('正在读取图片', 'Reading images') : status === 'failed' ? text('暂时无法读取图片', 'Images could not be read') : preview.isError ? text('预览暂不可用', 'Preview unavailable') : preview.isFetching ? text('读取预览…', 'Loading preview…') : text('目录中还没有图片', 'No images in this folder');
  const query = new URLSearchParams({ project: source.project_id || projectId });
  const ownerVersion = source.version_id || versionId;
  if (ownerVersion) query.set('version', ownerVersion);
  return <li><Link className="project-dataset-card" data-testid={`dataset-card-${source.id}`} aria-label={text(`打开数据集：${name}`, `Open dataset: ${name}`)} to={`/datasets/${encodeURIComponent(source.id)}?${query}`}>
    <div className="project-dataset-preview">
      {cover ? <div className="project-dataset-preview-image" key={`${cover.hash}/${cover.rel_path}`}><Images size={22} aria-hidden="true"/><img src={apiUrl(`/datasets/${encodeURIComponent(source.id)}/images/${encodeURIComponent(cover.hash)}/thumb?size=512`)} alt={cover.rel_path} width={512} height={512} loading="lazy" decoding="async" onError={event => { event.currentTarget.hidden = true; }}/></div> : <div className="project-dataset-preview-empty">{status === 'indexing' || preview.isFetching ? <Loader2 size={24} className="animate-spin" aria-hidden="true"/> : status === 'failed' || preview.isError ? <AlertCircle size={24} aria-hidden="true"/> : <Images size={24} aria-hidden="true"/>}<span>{placeholder}</span></div>}
    </div>
    <div className="project-dataset-card-body"><div className="project-dataset-card-heading"><span className="project-dataset-kind">{source.is_reg ? text('正则图', 'Regularization') : text('训练集', 'Training')}</span><h3 title={name}>{name}</h3></div><div className="project-dataset-card-counts">
      <span><strong>{stats?.images ?? '—'}</strong> {text('张图片', 'images')}</span><span><strong>{stats?.captioned ?? '—'}</strong> {text('份标签', 'captions')}</span><span><strong>{stats?.masks ?? '—'}</strong> {text('张遮罩', 'masks')}</span>
    </div>{stats?.error && <p className="project-dataset-card-error" title={stats.error}>{stats.error}</p>}</div>
    <div className="project-dataset-card-footer"><span className={`project-dataset-status status-${status || 'unknown'}`}>{status === 'indexing' && <Loader2 size={13} className="animate-spin" aria-hidden="true"/>}{status === 'failed' && <AlertCircle size={13} aria-hidden="true"/>}{statusText}</span>{source.repeats > 1 && <small>×{source.repeats} {text('重复', 'repeats')}</small>}<ArrowUpRight size={17} aria-hidden="true"/></div>
  </Link></li>;
}

export default function ProjectDatasetCards({ datasets, projectId, versionId, onRefresh }: Props) {
  const text = useWorkspaceText();
  const client = useQueryClient();
  useEventStream(EVENT_TYPES.DATASET_CHANGED, (event: { dataset_id?: string }) => {
    if (datasets.some(item => item.source.id === event.dataset_id)) void client.invalidateQueries({ queryKey: ['dataset-card-preview', event.dataset_id] });
  });
  const refresh = () => { onRefresh(); for (const dataset of datasets) void client.invalidateQueries({ queryKey: ['dataset-card-preview', dataset.source.id] }); };
  return <section className="project-dataset-library" id="version-datasets" aria-label={text('本版本的数据集', 'Version datasets')}>
    <div className="project-dataset-library-heading"><div><h2>{text('本版本的数据集', 'Version datasets')}</h2><span>{text(`${datasets.length} 个目录`, `${datasets.length} folders`)}</span></div><button type="button" onClick={refresh} aria-label={text('刷新索引状态', 'Refresh index status')}><RefreshCw size={16}/>{text('刷新', 'Refresh')}</button></div>
    {!datasets.length ? <div className="project-dataset-library-empty" data-testid="datasets-empty"><Folder size={28} aria-hidden="true"/><strong>{text('还没有训练图片', 'No training images yet')}</strong></div> : <ul className="project-dataset-card-grid">{datasets.map(dataset => <DatasetCard key={dataset.source.id} dataset={dataset} projectId={projectId} versionId={versionId}/>)}</ul>}
  </section>;
}
