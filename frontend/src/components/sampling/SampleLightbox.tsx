import React from 'react';
import { createPortal } from 'react-dom';
import { ChevronLeft, ChevronRight, Download, ImageOff, Minus, Plus, Scan, X } from 'lucide-react';
import { sampleSource } from '../../utils/sampleMedia';
import { useWorkspaceText } from '../../utils/workspaceText';
import './sample-lightbox.css';

const MAX_SCALE = 8;
/** Pointer travel below this many pixels is a tap, not a drag. */
const TAP_SLOP = 4;
const FITTED = { scale: 1, x: 0, y: 0 };
const CENTER = { x: 0, y: 0 };

export type LightboxSample = { url: string; step?: number; prompt: string; prompt_index?: number; seed: number; width: number; height: number; loss?: number | null };
type View = { scale: number; x: number; y: number };
type Point = { x: number; y: number };
type ZoomControls = { zoomIn: () => void; zoomOut: () => void; fit: () => void; actual: () => void };
type ZoomState = { percent: number; zoomed: boolean; ready: boolean; atMost: boolean; atActual: boolean };
type Gesture = { kind: 'pan'; start: Point; from: View; moved: boolean; onImage: boolean } | { kind: 'pinch'; distance: number; middle: Point; from: View };

/**
 * The image fitted to the space between the header and the toolbar. A tap zooms in at that spot and taps again to
 * fit; the wheel or a pinch zoom around the pointer and dragging pans. The image never leaves this area, so it never
 * slides under the controls. A tap beside a fitted image closes the viewer.
 */
const ZoomStage = React.forwardRef<ZoomControls, { src: string; alt: string; onState: (state: ZoomState) => void; onBackdrop: () => void }>(function ZoomStage({ src, alt, onState, onBackdrop }, controls) {
  const text = useWorkspaceText();
  const stage = React.useRef<HTMLDivElement>(null);
  const image = React.useRef<HTMLImageElement>(null);
  const pointers = React.useRef(new Map<number, Point>());
  const gesture = React.useRef<Gesture | null>(null);
  const [view, setView] = React.useState<View>(FITTED);
  const viewRef = React.useRef(view);
  viewRef.current = view;
  const [fit, setFit] = React.useState(1);
  // SampleLightbox keys the stage by image, so these describe the one image it shows.
  const [ready, setReady] = React.useState(false);
  const [failed, setFailed] = React.useState(false);

  const clamp = React.useCallback((next: View): View => {
    const box = stage.current, img = image.current;
    const scale = Math.min(MAX_SCALE, Math.max(1, next.scale));
    if (!box || !img) return { scale, x: 0, y: 0 };
    const maxX = Math.max(0, (img.clientWidth * scale - box.clientWidth) / 2), maxY = Math.max(0, (img.clientHeight * scale - box.clientHeight) / 2);
    return { scale, x: Math.min(maxX, Math.max(-maxX, next.x)), y: Math.min(maxY, Math.max(-maxY, next.y)) };
  }, []);
  // Keep the point under the pointer in place while the scale changes.
  const around = React.useCallback((from: View, scale: number, point: Point) => {
    const target = Math.min(MAX_SCALE, Math.max(1, scale));
    const ratio = target / from.scale;
    return clamp({ scale: target, x: point.x - (point.x - from.x) * ratio, y: point.y - (point.y - from.y) * ratio });
  }, [clamp]);
  const zoom = React.useCallback((scale: (current: number) => number, point: Point = CENTER) => setView(current => around(current, scale(current.scale), point)), [around]);
  // 1:1 shows one image pixel per screen pixel; an image smaller than the area already does when fitted.
  const actual = fit < 1 ? 1 / fit : 1;
  React.useImperativeHandle(controls, () => ({
    zoomIn: () => zoom(current => current * 1.25), zoomOut: () => zoom(current => current / 1.25),
    fit: () => setView(FITTED), actual: () => zoom(() => actual),
  }), [zoom, actual]);
  const pointIn = (clientX: number, clientY: number): Point => {
    const rect = stage.current!.getBoundingClientRect();
    return { x: clientX - rect.left - rect.width / 2, y: clientY - rect.top - rect.height / 2 };
  };
  const pinch = () => {
    const [a, b] = [...pointers.current.values()];
    return { distance: Math.hypot(a.x - b.x, a.y - b.y) || 1, middle: pointIn((a.x + b.x) / 2, (a.y + b.y) / 2) };
  };
  const measure = React.useCallback(() => {
    const img = image.current;
    if (img?.naturalWidth) setFit(img.clientWidth / img.naturalWidth);
    setView(current => clamp(current));
  }, [clamp]);

  React.useLayoutEffect(() => {
    // A cached image can finish loading before React attaches the load listener.
    const img = image.current;
    if (img?.complete && img.naturalWidth) { setReady(true); measure(); }
  }, [measure]);
  React.useEffect(() => {
    const box = stage.current;
    if (!box) return;
    const observer = new ResizeObserver(measure);
    observer.observe(box);
    // The viewer covers the page, so the plain wheel zooms. React's wheel listener is passive and cannot stop the
    // browser from zooming the page on a trackpad pinch, which arrives as a wheel event with ctrlKey.
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      zoom(current => current * Math.exp(-event.deltaY * (event.ctrlKey || event.metaKey ? 0.01 : 0.0015)), pointIn(event.clientX, event.clientY));
    };
    box.addEventListener('wheel', onWheel, { passive: false });
    return () => { observer.disconnect(); box.removeEventListener('wheel', onWheel); };
  }, [measure, zoom]);

  const percent = Math.round(view.scale * fit * 100);
  const atActual = Math.abs(view.scale - actual) < 0.001;
  React.useEffect(() => { onState({ percent, zoomed: view.scale > 1, ready, atMost: view.scale >= MAX_SCALE, atActual }); }, [percent, view.scale, ready, atActual, onState]);

  const release = (id: number) => {
    pointers.current.delete(id);
    const [rest] = [...pointers.current.values()];
    // Lifting one finger of a pinch carries on as a pan and never counts as a tap.
    gesture.current = rest ? { kind: 'pan', start: rest, from: viewRef.current, moved: true, onImage: false } : null;
  };
  return <div ref={stage} className="sample-zoom" data-zoomed={view.scale > 1 || undefined}
    onPointerDown={event => {
      if (event.pointerType === 'mouse' && event.button !== 0) return;
      event.currentTarget.setPointerCapture?.(event.pointerId);
      pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
      if (pointers.current.size === 1) gesture.current = { kind: 'pan', start: { x: event.clientX, y: event.clientY }, from: viewRef.current, moved: false, onImage: event.target === image.current };
      else if (pointers.current.size === 2) gesture.current = { kind: 'pinch', ...pinch(), from: viewRef.current };
    }}
    onPointerMove={event => {
      if (!pointers.current.has(event.pointerId)) return;
      pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
      const current = gesture.current;
      if (current?.kind === 'pinch') { if (pointers.current.size >= 2) setView(around(current.from, current.from.scale * pinch().distance / current.distance, current.middle)); return; }
      if (!current) return;
      const dx = event.clientX - current.start.x, dy = event.clientY - current.start.y;
      if (!current.moved && Math.hypot(dx, dy) < TAP_SLOP) return;
      current.moved = true;
      if (current.from.scale > 1) setView(clamp({ ...current.from, x: current.from.x + dx, y: current.from.y + dy }));
    }}
    onPointerUp={event => {
      if (!pointers.current.has(event.pointerId)) return;
      const current = gesture.current;
      release(event.pointerId);
      if (current?.kind !== 'pan' || current.moved || pointers.current.size) return;
      if (viewRef.current.scale > 1) setView(FITTED);
      else if (current.onImage) zoom(() => Math.max(2, actual), pointIn(event.clientX, event.clientY));
      else onBackdrop();
    }}
    onPointerCancel={event => { if (pointers.current.has(event.pointerId)) release(event.pointerId); }}>
    {failed ? <span className="sample-lightbox-failed"><ImageOff size={28} aria-hidden="true"/>{text('采样图读取失败', 'The preview could not be loaded')}</span> : <>
      {!ready && <span className="ui-skeleton sample-lightbox-skeleton" aria-hidden="true"/>}
      <img ref={image} src={src} alt={alt} draggable={false} className={`sample-zoom-image${ready ? '' : ' is-pending'}`}
        style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})` }}
        onLoad={() => { setReady(true); measure(); }} onError={() => setFailed(true)}/>
    </>}
  </div>;
});

/**
 * One preview over the whole page: what it is at the top, the image, then the paging and zoom toolbar centred
 * below the image and never over it. Esc or the close button returns to the page; the arrow keys page, + and -
 * zoom, 0 fits and 1 shows the image pixels. `details` adds context such as the epoch or the training run.
 */
export default function SampleLightbox({ sample, title, position, total, details = [], onPrevious, onNext, onClose }: {
  sample: LightboxSample; title?: string; position: number; total: number; details?: string[]; onPrevious?: () => void; onNext?: () => void; onClose: () => void;
}) {
  const text = useWorkspaceText();
  const zoom = React.useRef<ZoomControls>(null);
  const dialog = React.useRef<HTMLDivElement>(null);
  const [state, setState] = React.useState<ZoomState>({ percent: 100, zoomed: false, ready: false, atMost: false, atActual: true });
  const onState = React.useCallback((next: ZoomState) => setState(next), []);
  const src = sampleSource(sample.url);
  const loss = sample.step != null && sample.step > 0 && typeof sample.loss === 'number' && Number.isFinite(sample.loss) ? `Loss ${Number(sample.loss.toPrecision(5))}` : sample.step === 0 ? text('初始采样 · 未训练', 'Initial sample · untrained') : '';
  React.useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    dialog.current?.focus();
    return () => opener?.focus?.();
  }, []);
  React.useLayoutEffect(() => {
    // Disabling a paging button can leave keyboard focus outside the viewer.
    const focused = document.activeElement;
    if (focused === document.body || (focused instanceof HTMLButtonElement && focused.disabled && dialog.current?.contains(focused))) dialog.current?.focus();
  }, [src, position, total]);
  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Tab') {
      // Keep focus inside the viewer while it covers the page.
      const nodes = [...dialog.current!.querySelectorAll<HTMLElement>('button:not(:disabled),a[href]')];
      const first = nodes[0], last = nodes[nodes.length - 1];
      if (event.shiftKey && (document.activeElement === first || document.activeElement === dialog.current)) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      return;
    }
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    const actions: Record<string, (() => void) | undefined> = {
      Escape: onClose, ArrowLeft: onPrevious, ArrowRight: onNext,
      '-': () => zoom.current?.zoomOut(), '+': () => zoom.current?.zoomIn(), '=': () => zoom.current?.zoomIn(), '0': () => zoom.current?.fit(), '1': () => zoom.current?.actual(),
    };
    const action = actions[event.key];
    if (action) { event.preventDefault(); event.stopPropagation(); action(); }
  };
  return createPortal(<div ref={dialog} className="sample-lightbox" role="dialog" aria-modal="true" aria-label={title || text('查看采样图', 'Preview')} tabIndex={-1} onKeyDown={onKeyDown}>
    <header className="sample-lightbox-head">
      <div className="sample-lightbox-title">
        <p><strong>{title || (sample.step != null ? text(`第 ${sample.step} 步`, `Step ${sample.step}`) : text('预览图', 'Preview'))}</strong>{details.map(detail => <span key={detail}>{detail}</span>)}
          {sample.prompt_index != null && <span>{text(`提示词 ${sample.prompt_index + 1}`, `Prompt ${sample.prompt_index + 1}`)}</span>}<span>Seed {sample.seed}</span><span>{sample.width} × {sample.height}</span>{loss && <span>{loss}</span>}</p>
        {sample.prompt && <p className="sample-lightbox-prompt" title={sample.prompt}>{sample.prompt}</p>}
      </div>
      <button type="button" className="ui-btn ui-btn-icon sample-lightbox-close" onClick={onClose} aria-label={text('关闭', 'Close')} title={text('关闭（Esc）', 'Close (Esc)')}><X size={18}/></button>
    </header>
    <ZoomStage key={src} ref={zoom} src={src} alt={sample.prompt} onState={onState} onBackdrop={onClose}/>
    <div className="sample-lightbox-toolbar" role="toolbar" aria-label={text('查看工具', 'Viewer tools')}>
      <div className="sample-lightbox-tools">
        <button type="button" onClick={onPrevious} disabled={!onPrevious} aria-label={text('上一张', 'Previous')} title={text('上一张（←）', 'Previous (←)')}><ChevronLeft size={16}/></button>
        <span className="sample-lightbox-position">{position} / {total}</span>
        <button type="button" onClick={onNext} disabled={!onNext} aria-label={text('下一张', 'Next')} title={text('下一张（→）', 'Next (→)')}><ChevronRight size={16}/></button>
        <span className="sample-lightbox-divider" aria-hidden="true"/>
        <button type="button" onClick={() => zoom.current?.zoomOut()} disabled={!state.zoomed} aria-label={text('缩小', 'Zoom out')} title={text('缩小（-）', 'Zoom out (-)')}><Minus size={15}/></button>
        <output className="sample-lightbox-scale" aria-label={text('相对原图的显示比例', 'Scale relative to the image pixels')}>{state.ready ? `${state.percent}%` : '—'}</output>
        <button type="button" onClick={() => zoom.current?.zoomIn()} disabled={!state.ready || state.atMost} aria-label={text('放大', 'Zoom in')} title={text('放大（+）', 'Zoom in (+)')}><Plus size={15}/></button>
        <button type="button" onClick={() => zoom.current?.fit()} disabled={!state.zoomed} aria-label={text('适应窗口', 'Fit')} title={text('适应窗口（0）', 'Fit (0)')}><Scan size={15}/></button>
        <button type="button" className="sample-lightbox-actual" onClick={() => zoom.current?.actual()} disabled={!state.ready || state.atActual} aria-label={text('原图大小', 'Actual size')} title={text('原图大小（1）', 'Actual size (1)')}>1:1</button>
        <span className="sample-lightbox-divider" aria-hidden="true"/>
        <a href={src} download aria-label={text('下载原图', 'Download image')} title={text('下载原图', 'Download image')}><Download size={15}/></a>
      </div>
      <span className="sr-only" aria-live="polite">{state.ready ? text(`显示比例 ${state.percent}%`, `Scale ${state.percent}%`) : ''}</span>
    </div>
  </div>, document.body);
}
