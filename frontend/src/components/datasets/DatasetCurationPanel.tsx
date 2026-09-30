import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { useSearchParams } from 'react-router-dom';
import { ImagePlus, Search, X } from 'lucide-react';
import { apiClient, apiUrl } from '../../api/client';
import type { DatasetInfo } from '../../api/types';
import { useDatasetImages } from '../../api/hooks/useDatasetImages';
import ImageSortSelect from './ImageSortSelect';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { formatApiError } from '../../utils/errors';
import { formatBytes } from '../../utils/format';
import { useWorkspaceText } from '../../utils/workspaceText';
import Dialog from '../Dialog';
import StudioSelect from '../StudioSelect';
import { LazyImage } from '../Loading';
import DatasetImagePane from './DatasetImagePane';
import type { WorkspaceDataset } from './ProjectDatasetCards';

type Props = { datasets: WorkspaceDataset[]; readOnly: boolean; onChanged: () => void; onAddImages: () => void };
const folderName = (path: string) => path.replace(/\\/g, '/').split('/').filter(Boolean).pop()?.replace(/^(?:d_[0-9a-f]+-)+/i, '') || path;

export default function DatasetCurationPanel({ datasets, readOnly, onChanged, onAddImages }: Props) {
  const text = useWorkspaceText();
  const [params, setParams] = useSearchParams();
  const sources = datasets.map(item => item.source).filter(source => !source.is_reg);
  const current = sources.find(source => source.id === params.get('dataset')) || sources[0];
  if (!current) return <div className="dataset-curation-empty" data-testid="curation-empty">
    <strong>{text('当前版本还没有训练图片', 'This version has no training images yet')}</strong>
    <p>{text('先在“数据集”中添加图片，再回来挑选参与训练的图片。', 'Add images under Datasets first, then choose which ones to train on.')}</p>
    {!readOnly && <button type="button" className="ui-btn ui-btn-primary" onClick={onAddImages}><ImagePlus size={15}/>{text('添加训练图片', 'Add training images')}</button>}
  </div>;
  const selector = <label className="dataset-curation-folder">{text('目录', 'Folder')}<StudioSelect aria-label={text('筛选目录', 'Folder to curate')} value={current.id} options={sources.map(source => ({ value: source.id, label: folderName(source.path) }))} onValueChange={value => setParams(previous => { const next = new URLSearchParams(previous); next.set('dataset', value); return next; })}/></label>;
  return <CurationWorkspace key={current.id} datasetId={current.id} selector={selector} readOnly={readOnly} onChanged={onChanged}/>;
}

function CurationWorkspace({ datasetId, selector, readOnly, onChanged }: { datasetId: string; selector: React.ReactNode; readOnly: boolean; onChanged: () => void }) {
  const text = useWorkspaceText();
  const info = useQuery({
    queryKey: ['curation-dataset', datasetId],
    queryFn: ({ signal }) => apiClient.get<DatasetInfo>(`/datasets/${datasetId}`, { params: { include_cache: false }, signal, silent: true }),
  });
  const held = useDatasetImages(datasetId, 60, 'unused');
  const training = useDatasetImages(datasetId, 60, 'training');
  const [query, setQuery] = React.useState('');
  const [thumbnailWidth, setThumbnailWidth] = React.useState(140);
  const [moving, setMoving] = React.useState(false);
  const [error, setError] = React.useState('');
  const [arrivals, setArrivals] = React.useState<{ training: boolean; paths: ReadonlySet<string> } | null>(null);
  const [preview, setPreview] = React.useState<{ hash: string; path: string } | null>(null);
  const arrivalLoading = arrivals?.training ? training.loading : held.loading;
  React.useEffect(() => {
    if (!arrivals || arrivalLoading) return;
    const timer = window.setTimeout(() => setArrivals(null), 1600);
    return () => window.clearTimeout(timer);
  }, [arrivals, arrivalLoading]);
  const refresh = React.useCallback(() => { held.refresh(); training.refresh(); void info.refetch(); }, [held, training, info]);
  useEventStream(EVENT_TYPES.DATASET_CHANGED, (data: any) => { if (data?.dataset_id === datasetId) refresh(); });
  const search = (value: string) => { setQuery(value); held.setQ(value); training.setQ(value); };
  const canEdit = !readOnly && info.isSuccess;
  const move = async (included: boolean) => {
    const source = included ? held : training;
    if (!canEdit || moving || !source.selected.size) return;
    const paths = [...source.selected];
    setMoving(true); setError('');
    try {
      await apiClient.post(`/datasets/${datasetId}/membership`, { paths, included }, { silent: true });
      source.clearSelection();
      setArrivals({ training: included, paths: new Set(paths) });
      refresh(); onChanged();
    } catch (e) { setError(formatApiError(e)); }
    finally { setMoving(false); }
  };
  const stats = info.data?.stats;
  const previewImage = preview && [...held.items, ...training.items].find(image => image.hash === preview.hash && image.rel_path === preview.path);
  return <div className="dataset-curation" data-testid="dataset-curation">
    <div className="dataset-curation-toolbar">
      {selector}
      <label className="dataset-curation-search"><Search size={15}/><input type="search" aria-label={text('搜索文件名或标签', 'Search filenames or captions')} value={query} onChange={event => search(event.target.value)} placeholder={text('搜索文件名或标签…', 'Search filenames or captions…')}/>{query && <button type="button" className="ui-btn ui-btn-quiet ui-btn-sm ui-btn-icon" aria-label={text('清除图片筛选', 'Clear image filter')} onClick={() => search('')}><X size={15}/></button>}</label>
      <ImageSortSelect value={training.sort} onChange={value=>{held.setSort(value);training.setSort(value);}}/>
      <label className="dataset-thumbnail-size">{text('缩略图', 'Thumbnails')}<input type="range" min={120} max={220} step={20} value={thumbnailWidth} onChange={event => setThumbnailWidth(Number(event.target.value))} aria-label={text('缩略图大小', 'Thumbnail size')}/></label>
    </div>
    {(error || info.error) && <div role="alert" className="workspace-message error">{error || formatApiError(info.error)}<button type="button" className="ui-btn ui-btn-sm" onClick={() => { setError(''); refresh(); }}>{text('重试', 'Retry')}</button></div>}
    <div className="dataset-curation-panes">
      <DatasetImagePane datasetId={datasetId} previewOnly images={held} training={false} count={stats?.held_out_images ?? 0} canEdit={canEdit} busy={moving} minWidth={thumbnailWidth} arriving={arrivals && !arrivals.training ? arrivals.paths : undefined} onMove={() => void move(true)} onOpen={(hash, path) => setPreview({ hash, path })}/>
      <DatasetImagePane datasetId={datasetId} previewOnly images={training} training count={stats?.training_images ?? stats?.images ?? 0} canEdit={canEdit} busy={moving} minWidth={thumbnailWidth} arriving={arrivals?.training ? arrivals.paths : undefined} onMove={() => void move(false)} onOpen={(hash, path) => setPreview({ hash, path })}/>
    </div>
    {preview && <Dialog title={text('图片预览', 'Image preview')} wide onClose={() => setPreview(null)}>
      <div className="dataset-preview" data-testid="curation-preview">
        <div>
          <div className="dataset-preview-image" style={previewImage?.width && previewImage.height ? { aspectRatio: `${previewImage.width} / ${previewImage.height}` } : undefined}><LazyImage src={apiUrl(`/datasets/${datasetId}/images/${preview.hash}/file`)} alt={preview.path}/></div>
          {previewImage && <dl>
            <div><dt>{text('文件', 'File')}</dt><dd>{previewImage.rel_path}</dd></div>
            <div><dt>{text('尺寸', 'Size')}</dt><dd>{previewImage.width} × {previewImage.height}</dd></div>
            {typeof previewImage.size === 'number' && <div><dt>{text('大小', 'File size')}</dt><dd>{formatBytes(previewImage.size)}</dd></div>}
            <div><dt>{text('遮罩', 'Mask')}</dt><dd>{previewImage.has_mask ? text('有', 'Yes') : text('无', 'No')}</dd></div>
          </dl>}
        </div>
        <div>
          <h3>{text('标签', 'Caption')}{previewImage?.caption_format && ` · ${previewImage.caption_format.toUpperCase()}`}</h3>
          {previewImage?.caption_error ? <p role="alert">{previewImage.caption_error}</p> : <p>{previewImage?.caption || text('暂无标签', 'No caption')}</p>}
        </div>
      </div>
    </Dialog>}
  </div>;
}
