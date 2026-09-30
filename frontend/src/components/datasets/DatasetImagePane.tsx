import React from 'react';
import { ArrowLeft, ArrowRight, Brush, CheckSquare, Image as ImageIcon, SearchX, Square } from 'lucide-react';
import { apiUrl } from '../../api/client';
import { useDatasetImages } from '../../api/hooks/useDatasetImages';
import { useWorkspaceText } from '../../utils/workspaceText';
import { AnimatedCount } from '../motion';
import './dataset-image-pane.css';
import { LazyImage, LoadingNote } from '../Loading';

type Images = ReturnType<typeof useDatasetImages>;
interface Props {
  datasetId?: string;
  images: Images;
  training: boolean;
  all?: boolean;
  previewOnly?: boolean;
  count: number;
  canEdit: boolean;
  busy: boolean;
  minWidth: number;
  arriving?: ReadonlySet<string>;
  onMove: () => void;
  onOpen: (hash: string, path: string) => void;
}
const GAP = 8;

export default function DatasetImagePane({ datasetId, images, training, all = false, previewOnly = false, count, canEdit, busy, minWidth, arriving, onMove, onOpen }: Props) {
  const text = useWorkspaceText();
  const title = all ? text('全部图片', 'All images') : training ? text('参与训练', 'In training') : text('暂不训练', 'Not in training');
  const grid = React.useRef<HTMLDivElement>(null);
  const anchor = React.useRef<string | null>(null);
  const [viewport, setViewport] = React.useState({ width: 600, height: 600, top: 0 });
  const cols = Math.max(1, Math.floor((viewport.width - GAP) / (minWidth + GAP)));
  const width = Math.max(1, (viewport.width - GAP * (cols + 1)) / cols);
  const rowHeight = Math.round(width) + (previewOnly ? 26 : 52) + GAP;
  const rows = Math.ceil(images.items.length / cols);
  const startRow = Math.max(0, Math.floor(viewport.top / rowHeight) - 2);
  const endRow = Math.min(rows, Math.ceil((viewport.top + viewport.height) / rowHeight) + 2);
  const setColumns = images.setColumns;
  React.useEffect(() => { setColumns?.(cols); }, [cols, setColumns]);
  const firstLoad = images.loading && images.items.length === 0 && !images.error;

  React.useLayoutEffect(() => {
    const element = grid.current;
    if (!element) return;
    const measure = () => setViewport(previous => ({ ...previous, width: element.clientWidth || 600, height: element.clientHeight || 600, top: element.scrollTop }));
    measure();
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(measure);
    observer?.observe(element);
    window.addEventListener('resize', measure);
    return () => { observer?.disconnect(); window.removeEventListener('resize', measure); };
  }, []);
  React.useEffect(() => {
    anchor.current = null;
    if (grid.current) grid.current.scrollTop = 0;
    setViewport(previous => ({ ...previous, top: 0 }));
  }, [images.q]);
  React.useLayoutEffect(() => {
    const element = grid.current;
    if (!element || images.loading) return;
    const top = Math.min(element.scrollTop, Math.max(0, rows * rowHeight - element.clientHeight));
    element.scrollTop = top;
    setViewport(previous => previous.top === top ? previous : { ...previous, top });
  }, [rows, rowHeight, images.loading]);

  const select = (path: string, event: React.MouseEvent) => {
    if (event.shiftKey && anchor.current) {
      const from = images.items.findIndex(item => item.rel_path === anchor.current);
      const to = images.items.findIndex(item => item.rel_path === path);
      if (from >= 0 && to >= 0) images.selectRange(images.items.slice(Math.min(from, to), Math.max(from, to) + 1).map(item => item.rel_path));
      else images.toggleSelect(path);
    } else images.toggleSelect(path);
    anchor.current = path;
  };
  const moveDisabled = !canEdit || busy || images.loading || !images.selected.size;

  return <section className={`dataset-image-pane${all ? '' : training ? ' is-training' : ' is-held-out'}`} aria-label={title}>
    <header className="dataset-pane-heading">
      <h2>{title} <AnimatedCount className="dataset-pane-count" value={images.q ? `${images.total} / ${count}` : count}/></h2>
      <button type="button" className="ui-btn ui-btn-quiet ui-btn-sm" onClick={images.selected.size ? images.clearSelection : images.selectAll} disabled={!canEdit || busy || images.loading || !images.items.length}>
        {images.selected.size ? text(`取消选择 (${images.selected.size})`, `Clear (${images.selected.size})`) : images.hasMore ? text('全选已加载', 'Select loaded') : text('全选', 'Select all')}
      </button>
      {!all && (training
        ? <button type="button" className="ui-btn ui-btn-sm dataset-pane-move" onClick={onMove} disabled={moveDisabled}><ArrowLeft size={14}/>{text('暂时移出训练', 'Remove from training')}</button>
        : <button type="button" className="ui-btn ui-btn-primary ui-btn-sm dataset-pane-move" onClick={onMove} disabled={moveDisabled}>{text('加入训练', 'Add to training')}<ArrowRight size={14}/></button>)}
    </header>
    <div ref={grid} className="dataset-pane-grid" data-testid={all || training ? 'image-grid' : 'held-out-image-grid'} aria-busy={images.loading} onScroll={event => {
      const element = event.currentTarget;
      setViewport(previous => ({ ...previous, top: element.scrollTop }));
      if (element.scrollTop + element.clientHeight >= element.scrollHeight - rowHeight * 2) images.loadMore();
    }}>
      {firstLoad && <div className="dataset-pane-skeleton" role="status" aria-label={text('正在加载图片…', 'Loading images…')} style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`, gap: GAP }}>
        {Array.from({ length: cols * 2 }, (_, index) => <div key={index} className="dataset-pane-skeleton-card" style={{ height: rowHeight - GAP }}><div className="ui-skeleton"/><div className="ui-skeleton"/></div>)}
      </div>}
      {images.items.length > 0 && <div style={{ height: rows * rowHeight + GAP, position: 'relative' }}>
        <div className="dataset-pane-cells" style={{ top: startRow * rowHeight, gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`, gap: GAP }}>
          {images.items.slice(startRow * cols, endRow * cols).map(img => {
            const selected = images.selected.has(img.rel_path);
            return <article key={`${img.hash}:${img.rel_path}`} className={`dataset-image-card${previewOnly ? ' is-compact' : ''}${selected ? ' is-selected' : ''}${arriving?.has(img.rel_path) ? ' is-arriving' : ''}`} style={{ height: rowHeight - GAP }} data-testid={`image-card-${img.hash}`}>
              <button type="button" className="dataset-image-pick" aria-label={`${canEdit && !previewOnly ? text('编辑标签', 'Edit caption') : text('查看图片与标签', 'View image and caption')}: ${img.rel_path}`} onClick={() => onOpen(img.hash, img.rel_path)}>
                <LazyImage src={apiUrl(`/datasets/${datasetId}/images/${img.hash}/thumb?size=256`)} alt={img.rel_path} loading="lazy" decoding="async" draggable={false}/>
              </button>
              {canEdit && <button type="button" className="dataset-image-check" aria-label={`${text('选择图片','Select image')}: ${img.rel_path}`} aria-pressed={selected} disabled={busy} onClick={event => select(img.rel_path,event)}>{selected ? <CheckSquare size={18}/> : <Square size={18}/>}</button>}
              <div className="dataset-image-caption">
                <strong title={img.rel_path}>{img.rel_path}</strong>
                {!previewOnly && <span title={img.caption}>{img.has_mask && <Brush size={12} aria-label={text('已有遮罩', 'Mask saved')}/>} {img.width} × {img.height}{img.caption ? ` · ${img.caption}` : ''}</span>}
              </div>
            </article>;
          })}
        </div>
      </div>}
      {images.error && <div role="alert" className="dataset-pane-empty"><p>{images.error}</p><button type="button" className="ui-btn ui-btn-sm" onClick={images.refresh}>{text('重试', 'Retry')}</button></div>}
      {!images.error && !images.loading && images.items.length === 0 && <div className="dataset-pane-empty" data-testid={images.q ? 'dataset-filter-empty' : 'dataset-empty'}>
        {images.q ? <SearchX size={28}/> : <ImageIcon size={28}/>}
        <strong>{images.q ? text('没有匹配的图片', 'No matching images') : all ? text('这个目录还没有图片', 'This folder has no images yet') : training ? text('还没有参与训练的图片', 'No images in training') : text('没有暂不训练的图片', 'No held-out images')}</strong>
        {!images.q && !all && <p>{training ? text('从左侧选择图片，再加入训练。', 'Select images on the left and add them to training.') : text('移出训练的图片会保留在这里。', 'Images removed from training stay here.')}</p>}
      </div>}
      {images.loading && !firstLoad && <LoadingNote className="dataset-pane-loading" label={text('正在加载图片…', 'Loading images…')}/>}
    </div>
    {images.hasMore && <footer className="dataset-pane-footer"><span>{text(`已加载 ${images.items.length} / ${images.total}`, `${images.items.length} / ${images.total} loaded`)}</span><button type="button" className="ui-btn ui-btn-quiet ui-btn-sm" disabled={images.loading} onClick={images.loadMore}>{text('加载更多', 'Load more')}</button></footer>}
  </section>;
}
