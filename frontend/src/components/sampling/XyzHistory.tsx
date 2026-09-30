import React from 'react';
import { ChevronLeft, ChevronRight, Grid2X2, Loader2, Trash2, X } from 'lucide-react';
import { apiUrl } from '../../api/client';
import { LazyImage } from '../Loading';
import { prefersReducedMotion } from '../../utils/motion';
import { useWorkspaceText } from '../../utils/workspaceText';
import { isActive, isWaiting, taskProgress, useStatusLabel, type XyzTask } from './xyzTypes';

const imageUrl = (url: string) => url.startsWith('/api/') ? apiUrl(url.slice(4)) : url;

function when(seconds: number) {
  const date = new Date(seconds * 1000);
  const time = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  return date.toDateString() === new Date().toDateString() ? time : `${date.getMonth() + 1}/${date.getDate()} ${time}`;
}

/**
 * The run's comparisons as thumbnails, newest first. Beside the results they form a column; on narrow pages a line
 * that pages sideways.
 */
export default function XyzHistory({ tasks, selected, onSelect, onCancel, onDelete, disabled }: {
  tasks: XyzTask[]; selected: string; onSelect: (id: string) => void; onCancel: (task: XyzTask) => void; onDelete: (task: XyzTask) => void; disabled: boolean;
}) {
  const text = useWorkspaceText();
  const list = React.useRef<HTMLOListElement>(null);
  const [edges, setEdges] = React.useState({ paged: false, before: false, after: false });

  const measure = React.useCallback(() => {
    const element = list.current;
    if (!element) return;
    const paged = element.scrollWidth > element.clientWidth + 1;
    const next = { paged, before: paged && element.scrollLeft > 1, after: paged && element.scrollLeft + element.clientWidth < element.scrollWidth - 1 };
    setEdges(current => current.paged === next.paged && current.before === next.before && current.after === next.after ? current : next);
  }, []);
  React.useLayoutEffect(() => {
    const element = list.current;
    if (!element) return;
    measure();
    element.addEventListener('scroll', measure, { passive: true });
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(measure);
    observer?.observe(element);
    return () => { element.removeEventListener('scroll', measure); observer?.disconnect(); };
  }, [measure, tasks.length]);
  // The chosen comparison stays in view, also after it is picked from elsewhere.
  React.useEffect(() => {
    const item = list.current?.querySelector<HTMLElement>('[aria-current="true"]');
    item?.scrollIntoView?.({ block: 'nearest', inline: 'nearest' });
  }, [selected]);

  const page = (direction: 1 | -1) => {
    const element = list.current;
    if (element) element.scrollBy({ left: direction * element.clientWidth * 0.9, behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
  };
  const arrow = (direction: 1 | -1) => <button type="button" className="ui-btn ui-btn-sm ui-btn-icon xyz-history-page"
    aria-label={direction > 0 ? text('后面的对比记录', 'Later comparisons') : text('前面的对比记录', 'Earlier comparisons')}
    disabled={direction > 0 ? !edges.after : !edges.before} onClick={() => page(direction)}>
    {direction > 0 ? <ChevronRight size={15}/> : <ChevronLeft size={15}/>}
  </button>;

  return <nav className="xyz-history" aria-label={text('对比记录', 'Comparisons')} data-paged={edges.paged || undefined}>
    {edges.paged && arrow(-1)}
    <ol ref={list} className="xyz-history-list">
      {tasks.map(task => <HistoryItem key={task.id} task={task} selected={task.id === selected} disabled={disabled}
        onSelect={() => onSelect(task.id)} onCancel={() => onCancel(task)} onDelete={() => onDelete(task)}/>)}
    </ol>
    {edges.paged && arrow(1)}
  </nav>;
}

function HistoryItem({ task, selected, disabled, onSelect, onCancel, onDelete }: { task: XyzTask; selected: boolean; disabled: boolean; onSelect: () => void; onCancel: () => void; onDelete: () => void }) {
  const text = useWorkspaceText();
  const statusLabel = useStatusLabel();
  const active = isActive(task);
  const waiting = isWaiting(task);
  const cells = task.manifest.cells;
  // A finished comparison shows its first image; one being drawn, the latest.
  const thumb = active ? cells[cells.length - 1] : cells[0];
  const share = taskProgress(task);
  const state = waiting ? 'queued' : active ? 'running' : task.status;
  const time = when(task.created_at);
  const label = `${time} · ${text(`${task.total} 张`, `${task.total} images`)} · ${statusLabel(task.status)}`;
  return <li className="xyz-history-entry" data-state={state}>
    <button type="button" className="xyz-history-item" aria-current={selected ? 'true' : undefined} aria-label={label} title={label} onClick={onSelect}>
      <span className="xyz-history-thumb">
        {thumb ? <LazyImage src={imageUrl(thumb.url)} alt="" draggable={false}/>
          : <span className="xyz-history-placeholder">{waiting ? text('排队中', 'Queued') : active ? <Loader2 size={18} className="animate-spin" aria-hidden="true"/> : <Grid2X2 size={18} aria-hidden="true"/>}</span>}
        {task.total > 1 && <span className="xyz-history-count">{task.total}</span>}
        {(state === 'failed' || state === 'cancelled') && <span className="xyz-history-badge">{statusLabel(task.status)}</span>}
        {state === 'running' && <span className="xyz-history-bar" data-indeterminate={share == null || undefined}><span style={{ width: `${(share ?? 0.3) * 100}%` }}/></span>}
      </span>
      <span className="xyz-history-time">{time}</span>
    </button>
    {task.can_cancel && task.status !== 'cancelling' && <button type="button" className="xyz-history-cancel" disabled={disabled}
      aria-label={text(`取消 ${time} 的对比`, `Cancel the ${time} comparison`)} title={text('取消', 'Cancel')} onClick={onCancel}><X size={13}/></button>}
    {['completed', 'failed', 'cancelled'].includes(task.status) && <button type="button" className="xyz-history-delete" disabled={disabled}
      aria-label={text(`删除 ${time} 的模型测试`, `Delete the ${time} model test`)} title={text('删除模型测试', 'Delete model test')} onClick={onDelete}><Trash2 size={13}/></button>}
  </li>;
}
