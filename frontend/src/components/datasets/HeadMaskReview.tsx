import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ChevronLeft, ChevronRight, RefreshCw, X } from 'lucide-react';
import MaskIcon from '../icons/MaskIcon';
import { apiClient, apiUrl } from '../../api/client';
import type { HeadProposalImage, HeadProposals, HeadSelection } from '../../api/types';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import Dialog from '../Dialog';
import { LazyImage, LoadingNote } from '../Loading';
import StudioSelect from '../StudioSelect';
import { SlidingIndicator } from '../motion';
import type { PipelineOperation } from './DatasetPipelinePanel';

type Filter = 'heads' | 'none' | 'all';
type Chosen = Record<string, number[]>;
const PAGE_SIZES = [24, 48, 96];
const PAGE_SIZE_KEY = 'studio.headReview.pageSize';
const keyOf = (image: { dataset_id: string; rel_path: string }) => `${image.dataset_id}/${image.rel_path}`;
const percent = (value: number) => `${(value * 100).toFixed(3)}%`;
const fileName = (path: string) => path.split('/').pop() || path;

/** The detected heads of one image drawn over it; each box toggles whether that head is written. */
function HeadFrame({ image, size, chosen, locked, onToggle, onOpen }: {
  image: HeadProposalImage; size: number; chosen: number[]; locked: boolean;
  onToggle: (index: number) => void; onOpen?: () => void;
}) {
  const text = useWorkspaceText();
  const width = image.width || 1, height = image.height || 1;
  const style = { aspectRatio: `${width} / ${height}`, '--head-ratio': width / height } as CSSProperties;
  return <div className="head-frame" style={style}>
    {image.hash ? <LazyImage src={apiUrl(`/datasets/${image.dataset_id}/images/${image.hash}/thumb?size=${size}`)} alt={image.rel_path} loading="lazy" decoding="async"/>
      : <span className="ui-image-fallback">{text('无法预览', 'No preview')}</span>}
    {onOpen && <button type="button" className="head-frame-open" aria-label={text(`放大查看 ${image.rel_path}`, `Enlarge ${image.rel_path}`)} onClick={onOpen}/>}
    {image.regions.map((region, order) => {
      const on = chosen.includes(region.index);
      const label = text(`头部 ${order + 1}（置信度 ${region.score.toFixed(2)}）`, `Head ${order + 1} (confidence ${region.score.toFixed(2)})`);
      return <button key={region.index} type="button" className={`head-box${on ? ' is-on' : ''}`} aria-pressed={on} aria-label={label} title={label} disabled={locked}
        style={{ left: percent(region.x1 / width), top: percent(region.y1 / height), width: percent((region.x2 - region.x1) / width), height: percent((region.y2 - region.y1) / height) }}
        onClick={() => onToggle(region.index)}/>;
    })}
  </div>;
}

/**
 * Detected heads, image by image, before any mask is written: a click on a box drops a false hit,
 * the checkbox leaves a whole image out, and only what stays chosen becomes masks.
 */
export default function HeadMaskReview({ operation, locked, onWrite, onDetectAgain, onDismiss }: {
  operation: PipelineOperation; locked: boolean;
  onWrite: (selections: HeadSelection[]) => Promise<void>; onDetectAgain: () => Promise<void>; onDismiss: () => Promise<void>;
}) {
  const text = useWorkspaceText();
  const proposals = useQuery({
    queryKey: ['head-proposals', operation.id],
    queryFn: ({ signal }) => apiClient.get<HeadProposals>(`/dataset-pipeline/operations/${operation.id}/proposals`, { signal, silent: true }),
    staleTime: Infinity,
  });
  const images = useMemo(() => proposals.data?.images ?? [], [proposals.data]);
  const found = useMemo(() => images.filter(image => image.regions.length > 0), [images]);
  const missed = useMemo(() => images.filter(image => !image.regions.length), [images]);
  // Every detected head starts chosen; the review only takes away.
  const [edits, setEdits] = useState<Chosen | null>(null);
  const chosen: Chosen = useMemo(() => edits ?? Object.fromEntries(found.map(image => [keyOf(image), image.regions.map(region => region.index)])), [edits, found]);
  const [filter, setFilter] = useState<Filter>('heads');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(() => { try { const saved = Number(localStorage.getItem(PAGE_SIZE_KEY)); return PAGE_SIZES.includes(saved) ? saved : PAGE_SIZES[0]; } catch { return PAGE_SIZES[0]; } });
  const [open, setOpen] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const shown = filter === 'heads' ? found : filter === 'none' ? missed : images;
  const pages = Math.max(1, Math.ceil(shown.length / pageSize));
  const current = Math.min(page, pages);
  useEffect(() => { setPage(1); }, [filter, pageSize]);
  const picks = (image: HeadProposalImage) => chosen[keyOf(image)] ?? [];
  const toggle = (image: HeadProposalImage, index: number) => setEdits(() => {
    const next = new Set(picks(image));
    if (next.has(index)) next.delete(index); else next.add(index);
    return { ...chosen, [keyOf(image)]: [...next].sort((a, b) => a - b) };
  });
  const pickImage = (image: HeadProposalImage, on: boolean) => setEdits({ ...chosen, [keyOf(image)]: on ? image.regions.map(region => region.index) : [] });
  const pickAll = (on: boolean) => setEdits(Object.fromEntries(found.map(image => [keyOf(image), on ? image.regions.map(region => region.index) : []])));
  const selections: HeadSelection[] = found.map(image => ({ dataset_id: image.dataset_id, rel_path: image.rel_path, regions: picks(image) })).filter(pick => pick.regions.length > 0);
  const heads = selections.reduce((sum, pick) => sum + pick.regions.length, 0);
  const totalHeads = found.reduce((sum, image) => sum + image.regions.length, 0);
  const unreadable = images.filter(image => image.error).length;
  const run = async (action: () => Promise<void>) => {
    setBusy(true); setError('');
    try { await action(); } catch (e) { setError(formatApiError(e)); } finally { setBusy(false); }
  };
  const changePageSize = (value: string) => { setPageSize(Number(value)); try { localStorage.setItem(PAGE_SIZE_KEY, value); } catch { /* the choice lasts for this page only */ } };
  const openIndex = open ? shown.findIndex(image => keyOf(image) === open) : -1;
  const opened = openIndex >= 0 ? shown[openIndex] : null;
  const disabled = locked || busy;
  // The dialog focuses its close button, so the arrow keys are read for the whole window.
  useEffect(() => {
    if (openIndex < 0) return;
    const step = (event: KeyboardEvent) => {
      const next = event.key === 'ArrowLeft' ? openIndex - 1 : event.key === 'ArrowRight' ? openIndex + 1 : -1;
      if (next < 0 || next >= shown.length) return;
      event.preventDefault();
      setOpen(keyOf(shown[next]));
    };
    window.addEventListener('keydown', step);
    return () => window.removeEventListener('keydown', step);
  }, [openIndex, shown]);

  if (proposals.isPending) return <div className="head-review"><LoadingNote block label={text('正在读取检测结果…', 'Loading detected heads…')}/></div>;
  if (proposals.isError) return <div className="head-review"><p role="alert" className="vision-error">{formatApiError(proposals.error)} <button type="button" className="ui-link" onClick={() => void proposals.refetch()}>{text('重试', 'Retry')}</button></p></div>;

  const summary = (image: HeadProposalImage) => image.error ? text('无法读取', 'Unreadable')
    : image.regions.length ? text(`${picks(image).length} / ${image.regions.length} 个头部`, `${picks(image).length} / ${image.regions.length} heads`) : text('未检测到头部', 'No head found');
  return <>
    <div className="head-review" data-testid="head-review">
      <div className="head-review-head">
        <h4>{text('检测结果', 'Detected heads')}</h4>
        <span className="head-review-count">{text(`${found.length} 张图片共 ${totalHeads} 个头部${unreadable ? `，${unreadable} 张无法读取` : ''}`, `${totalHeads} heads in ${found.length} images${unreadable ? `, ${unreadable} unreadable` : ''}`)}</span>
        <div className="ui-segmented head-review-filter" role="group" aria-label={text('显示', 'Show')}>
          {([['heads', text(`有头部 ${found.length}`, `With heads ${found.length}`)], ['none', text(`未检测到 ${missed.length}`, `None found ${missed.length}`)], ['all', text(`全部 ${images.length}`, `All ${images.length}`)]] as const).map(([value, label]) =>
            <button key={value} type="button" aria-pressed={filter === value} onClick={() => setFilter(value)}>{label}</button>)}
          <SlidingIndicator className="ui-segmented-thumb"/>
        </div>
        <div className="head-review-picks">
          <button type="button" className="ui-btn ui-btn-sm" disabled={disabled || !found.length} onClick={() => pickAll(true)}>{text('全选', 'Select all')}</button>
          <button type="button" className="ui-btn ui-btn-sm" disabled={disabled || !found.length} onClick={() => pickAll(false)}>{text('全不选', 'Select none')}</button>
        </div>
      </div>
      {shown.length ? <div className="head-review-grid">{shown.slice((current - 1) * pageSize, current * pageSize).map(image => {
        const count = picks(image).length;
        return <article key={keyOf(image)} className={`head-review-card${image.regions.length && !count ? ' is-skipped' : ''}`} data-testid={`head-review-${image.rel_path}`}>
          <div className="head-review-frame"><HeadFrame image={image} size={384} chosen={picks(image)} locked={disabled} onToggle={index => toggle(image, index)} onOpen={() => setOpen(keyOf(image))}/></div>
          <footer>
            {image.regions.length ? <label className="head-review-check"><input type="checkbox" checked={count > 0} disabled={disabled} aria-label={text(`写入 ${image.rel_path} 的遮罩`, `Write masks for ${image.rel_path}`)} onChange={event => pickImage(image, event.target.checked)}/><span title={image.rel_path}>{fileName(image.rel_path)}</span></label>
              : <span className="head-review-name" title={image.rel_path}>{fileName(image.rel_path)}</span>}
            <small className={image.error ? 'is-error' : ''} title={image.error || undefined}>{summary(image)}</small>
          </footer>
        </article>;
      })}</div> : <p className="head-review-empty">{text('没有要显示的图片。', 'No images to show.')}</p>}
      {shown.length > PAGE_SIZES[0] && <div className="head-review-pager" role="navigation" aria-label={text('检测结果翻页', 'Detected heads pages')}>
        <label>{text('每页', 'Per page')}<StudioSelect aria-label={text('每页图片数', 'Images per page')} value={String(pageSize)} options={PAGE_SIZES.map(value => ({ value: String(value), label: String(value) }))} onValueChange={changePageSize}/></label>
        <button type="button" className="ui-btn ui-btn-sm" disabled={current <= 1} onClick={() => setPage(current - 1)}>{text('上一页', 'Previous')}</button>
        <span>{current} / {pages}</span>
        <button type="button" className="ui-btn ui-btn-sm" disabled={current >= pages} onClick={() => setPage(current + 1)}>{text('下一页', 'Next')}</button>
      </div>}
    </div>
    <footer className="vision-panel-actions">
      <button type="button" className="ui-btn ui-btn-primary" disabled={disabled || !selections.length} onClick={() => void run(() => onWrite(selections))}><MaskIcon size={15}/>{text(`写入遮罩（${selections.length} 张 · ${heads} 个头部）`, `Write masks (${selections.length} images · ${heads} heads)`)}</button>
      <button type="button" className="ui-btn" disabled={disabled} onClick={() => void run(onDetectAgain)}><RefreshCw size={14}/>{text('重新检测', 'Detect again')}</button>
      <button type="button" className="ui-btn" disabled={disabled} onClick={() => void run(onDismiss)}><X size={14}/>{text('放弃检测结果', 'Discard detection')}</button>
      {error && <p role="alert" className="vision-error">{error}</p>}
    </footer>
    {opened && <Dialog wide title={opened.rel_path} onClose={() => setOpen(null)}>
      <div className="head-review-zoom">
        <div className="head-review-zoom-frame"><HeadFrame image={opened} size={1280} chosen={picks(opened)} locked={disabled} onToggle={index => toggle(opened, index)}/></div>
        <div className="head-review-zoom-bar">
          <button type="button" className="ui-btn ui-btn-sm" disabled={openIndex <= 0} onClick={() => setOpen(keyOf(shown[openIndex - 1]))}><ChevronLeft size={14}/>{text('上一张', 'Previous')}</button>
          <span>{openIndex + 1} / {shown.length}</span>
          <button type="button" className="ui-btn ui-btn-sm" disabled={openIndex >= shown.length - 1} onClick={() => setOpen(keyOf(shown[openIndex + 1]))}>{text('下一张', 'Next')}<ChevronRight size={14}/></button>
          {opened.regions.length > 0 && <label className="head-review-check"><input type="checkbox" checked={picks(opened).length > 0} disabled={disabled} onChange={event => pickImage(opened, event.target.checked)}/>{text('写入这张', 'Write this image')}</label>}
          <small>{summary(opened)}</small>
        </div>
      </div>
    </Dialog>}
  </>;
}
