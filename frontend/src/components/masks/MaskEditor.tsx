import React from 'react';
import { Brush, Eraser, Hand, Undo2, Redo2, X, Save, Maximize, ZoomIn, ZoomOut, Loader2 } from 'lucide-react';
import { apiUrl } from '../../api/client';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import { MaskDocument, imagePoint, paintSegment, type MaskOperation, type MaskPoint } from './maskDocument';
import { loadMask, maskEndpoint, saveMask, type MaskInfo } from './maskApi';
import ImageEditor from './ImageEditor';
import Switch from '../Switch';

interface Props { datasetId: string; imageId: string; relPath: string; onClose: () => void; onSaved: () => void; onEnableTraining: () => Promise<void>; allowPaint?: boolean }
const control = 'inline-flex items-center justify-center gap-1.5 min-h-8 rounded-md border border-slate-300 dark:border-slate-600 px-2.5 py-1.5 text-xs hover:bg-slate-100 dark:hover:bg-slate-700 disabled:opacity-40';

export function MaskEditor(props: Props) {
  return props.allowPaint ? <ImageEditor {...props} onReloadList={props.onSaved} /> : <TrainingMaskEditor {...props} />;
}
function TrainingMaskEditor({ datasetId, imageId, relPath, onClose, onSaved, onEnableTraining }: Props) {
  const text = useWorkspaceText();
  const [info, setInfo] = React.useState<MaskInfo | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState('');
  const [message, setMessage] = React.useState('');
  const [saving, setSaving] = React.useState(false);
  const [tool, setTool] = React.useState<'brush' | 'erase' | 'pan'>('erase');
  const [diameter, setDiameter] = React.useState(64);
  const [opacity, setOpacity] = React.useState(0.5);
  const [maskOnly, setMaskOnly] = React.useState(false);
  const [zoom, setZoom] = React.useState(1);
  const [hostWidth, setHostWidth] = React.useState(800);
  const [hostHeight, setHostHeight] = React.useState(484);
  const [revision, redraw] = React.useReducer((value: number) => value + 1, 0);
  const [reload, setReload] = React.useState(0);
  const doc = React.useRef<MaskDocument | null>(null);
  const canvas = React.useRef<HTMLCanvasElement>(null);
  const brushPreview = React.useRef<HTMLDivElement>(null);
  const viewport = React.useRef<HTMLDivElement>(null);
  const dialog = React.useRef<HTMLDivElement>(null);
  const closeButton = React.useRef<HTMLButtonElement>(null);
  const stroke = React.useRef<Extract<MaskOperation, { kind: 'stroke' }> | null>(null);
  const pan = React.useRef<{ x: number; y: number; left: number; top: number } | null>(null);
  const frame = React.useRef<number | null>(null);
  const titleId = React.useId();

  const renderView = React.useCallback(() => {
    const current = doc.current; const element = canvas.current;
    if (!current || !element) return;
    const viewScale = Math.min(1, (hostWidth - 24) / current.width, (hostHeight - 24) / current.height) * zoom;
    const scale = Math.min(1, Math.max(1536 / Math.max(current.width, current.height), viewScale));
    const width = Math.max(1, Math.round(current.width * scale)); const height = Math.max(1, Math.round(current.height * scale));
    if (element.width !== width) element.width = width;
    if (element.height !== height) element.height = height;
    const context = element.getContext('2d'); if (!context) return;
    const output = context.createImageData(width, height);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const value = current.pixels[Math.min(current.height - 1, Math.floor(y / scale)) * current.width + Math.min(current.width - 1, Math.floor(x / scale))];
      const i = (y * width + x) * 4;
      output.data[i] = maskOnly ? value : 239; output.data[i + 1] = maskOnly ? value : 68; output.data[i + 2] = maskOnly ? value : 68;
      output.data[i + 3] = maskOnly ? 255 : Math.round((255 - value) * opacity);
    }
    context.putImageData(output, 0, 0);
  }, [opacity, maskOnly, hostWidth, hostHeight, zoom]);
  const scheduleRender = () => { if (frame.current === null) frame.current = requestAnimationFrame(() => { frame.current = null; renderView(); }); };

  React.useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    closeButton.current?.focus();
    return () => { previous?.focus(); if (frame.current !== null) cancelAnimationFrame(frame.current); };
  }, []);
  React.useEffect(() => {
    const controller = new AbortController(); setLoading(true); setError(''); doc.current = null; setInfo(null);
    loadMask(datasetId, imageId, relPath, controller.signal).then((loaded) => {
      if (controller.signal.aborted) return;
      doc.current = new MaskDocument(loaded.info.width, loaded.info.height, loaded.pixels);
      setInfo(loaded.info); setDiameter(Math.max(8, Math.round(Math.min(loaded.info.width, loaded.info.height) / 12))); setZoom(1); redraw();
    }).catch((error) => { if (!controller.signal.aborted) setError(formatApiError(error)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [datasetId, imageId, relPath, reload]);
  React.useEffect(() => { renderView(); }, [revision, renderView, info, loading]);
  React.useEffect(() => {
    const element = viewport.current; if (!element) return;
    const resize = () => { setHostWidth(element.clientWidth || 800); setHostHeight(element.clientHeight || 484); };
    resize(); const observer = new ResizeObserver(resize); observer.observe(element); return () => observer.disconnect();
  }, [loading]);
  React.useEffect(() => {
    const beforeUnload = (event: BeforeUnloadEvent) => { if (doc.current?.dirty) { event.preventDefault(); event.returnValue = ''; } };
    window.addEventListener('beforeunload', beforeUnload); return () => window.removeEventListener('beforeunload', beforeUnload);
  }, []);

  const close = () => { if (!saving && (!doc.current?.dirty || window.confirm(text('遮罩尚未保存，确定放弃这些修改？', 'Discard the unsaved mask changes?')))) onClose(); };
  const perform = (operation: MaskOperation) => { if (!doc.current || saving || stroke.current) return; doc.current.apply(operation); setMessage(''); redraw(); };
  const undo = () => { if (!saving && !stroke.current) { doc.current?.undo(); setMessage(''); redraw(); } };
  const redo = () => { if (!saving && !stroke.current) { doc.current?.redo(); setMessage(''); redraw(); } };
  const finishStroke = () => {
    if (stroke.current && doc.current) { doc.current.apply(stroke.current, true); stroke.current = null; setMessage(''); redraw(); }
    pan.current = null;
  };
  const point = (event: React.PointerEvent<HTMLCanvasElement>): MaskPoint => imagePoint(event.clientX, event.clientY, event.currentTarget.getBoundingClientRect(), doc.current!.width, doc.current!.height);
  const pointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!doc.current || saving || event.button !== 0) return;
    event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId);
    if (tool === 'pan') { pan.current = { x: event.clientX, y: event.clientY, left: viewport.current?.scrollLeft || 0, top: viewport.current?.scrollTop || 0 }; return; }
    const at = point(event); stroke.current = { kind: 'stroke', points: [at], diameter, value: tool === 'brush' ? 255 : 0 };
    paintSegment(doc.current.pixels, doc.current.width, doc.current.height, at, at, diameter, stroke.current.value); scheduleRender();
  };
  const pointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (brushPreview.current) {
      const bounds = event.currentTarget.getBoundingClientRect();
      brushPreview.current.style.left = `${event.clientX - bounds.left}px`;
      brushPreview.current.style.top = `${event.clientY - bounds.top}px`;
      brushPreview.current.style.visibility = tool === 'pan' ? 'hidden' : 'visible';
    }
    if (pan.current && viewport.current) { viewport.current.scrollLeft = pan.current.left - (event.clientX - pan.current.x); viewport.current.scrollTop = pan.current.top - (event.clientY - pan.current.y); return; }
    if (!stroke.current || !doc.current) return;
    const at = point(event); const previous = stroke.current.points.at(-1)!;
    paintSegment(doc.current.pixels, doc.current.width, doc.current.height, previous, at, stroke.current.diameter, stroke.current.value);
    stroke.current.points.push(at); scheduleRender();
  };
  const save = async (enable: boolean) => {
    if (!doc.current || !info || saving) return;
    finishStroke(); setSaving(true); setError(''); setMessage('');
    try {
      if (doc.current.dirty || !info.has_mask) {
        const result = await saveMask(datasetId, imageId, relPath, info, doc.current.pixels);
        doc.current.markSaved(); setInfo(result); onSaved(); redraw();
      }
      setMessage(text('遮罩已保存，下次训练会读取最新文件。', 'Mask saved. The next training run will read the updated file.'));
      if (enable) await onEnableTraining();
    } catch (error) { setError(formatApiError(error)); } finally { setSaving(false); }
  };
  const keyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Escape') { event.stopPropagation(); close(); }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') { event.preventDefault(); if (event.shiftKey) redo(); else undo(); }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') { event.preventDefault(); void save(false); }
    if (event.key === 'Tab') {
      const controls = dialog.current?.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), canvas[tabindex]');
      if (!controls?.length) return;
      const first = controls[0]; const last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
  };
  const scale = info ? Math.min(1, (hostWidth - 24) / info.width, (hostHeight - 24) / info.height) * zoom : 1;
  const coverage = React.useMemo(() => { void revision; return doc.current?.coverage ?? 0; }, [revision]);

  return <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-2 sm:p-4" onClick={close}>
    <section ref={dialog} role="dialog" aria-modal="true" aria-labelledby={titleId} onKeyDown={keyDown} onClick={(event) => event.stopPropagation()} className="flex max-h-[95vh] w-full max-w-7xl min-w-0 flex-col overflow-y-auto rounded-xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-slate-800">
      <header className="flex items-start justify-between gap-3 border-b border-slate-200 p-3 dark:border-slate-700">
        <div className="min-w-0"><h2 id={titleId} className="text-base font-semibold">{text('编辑训练遮罩', 'Edit training mask')}</h2><p className="break-all text-xs text-slate-500 dark:text-slate-400">{relPath}{info && ` · ${info.width}×${info.height}`}</p></div>
        <button ref={closeButton} type="button" aria-label={text('关闭遮罩编辑器', 'Close mask editor')} className={control} disabled={saving} onClick={close}><X className="h-4 w-4" /></button>
      </header>
      <div className="space-y-2.5 p-3">
        <p className="text-xs text-slate-600 dark:text-slate-300">{text('白色参与训练，黑色忽略。叠加预览中，红色表示被忽略的区域。', 'White participates in training; black is ignored. The red overlay marks ignored areas.')}</p>
        {loading && <p role="status" className="flex items-center gap-2 py-8"><Loader2 className="h-5 w-5 animate-spin" />{text('正在读取原图尺寸与已有遮罩…', 'Loading image dimensions and existing mask…')}</p>}
        {error && <div role="alert" className="rounded-lg bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950 dark:text-red-300"><p className="whitespace-pre-line break-words">{error}</p><button className="mt-2 underline" disabled={saving} onClick={() => { if (!doc.current?.dirty || window.confirm(text('重新读取会放弃未保存的修改，继续？', 'Reload and discard unsaved changes?'))) setReload((value) => value + 1); }}>{text('重新读取遮罩', 'Reload mask')}</button></div>}
        {info && !loading && <>
          <div className="flex flex-wrap items-center gap-2">
            {([{ key: 'brush', Icon: Brush, label: text('笔刷 · 参与', 'Brush · include') }, { key: 'erase', Icon: Eraser, label: text('擦除 · 忽略', 'Erase · ignore') }, { key: 'pan', Icon: Hand, label: text('移动画布', 'Pan canvas') }] as const).map(({ key, Icon, label }) => <button key={key} className={`${control} ${tool === key ? 'bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300' : ''}`} aria-pressed={tool === key} disabled={saving} onClick={() => setTool(key)}><Icon className="h-4 w-4" />{label}</button>)}
            <button className={control} disabled={saving || !doc.current?.canUndo} onClick={undo}><Undo2 className="h-4 w-4" />{text('撤销', 'Undo')}</button>
            <button className={control} disabled={saving || !doc.current?.canRedo} onClick={redo}><Redo2 className="h-4 w-4" />{text('重做', 'Redo')}</button>
            <button className={control} disabled={saving} onClick={() => perform({ kind: 'fill', value: 255 })}>{text('全选 · 全白', 'Select all · white')}</button>
            <button className={control} disabled={saving} onClick={() => perform({ kind: 'fill', value: 0 })}>{text('清空 · 全黑', 'Clear · black')}</button>
            <button className={control} disabled={saving} onClick={() => perform({ kind: 'invert' })}>{text('反转', 'Invert')}</button>
          </div>
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2 rounded-md bg-slate-50 p-2 dark:bg-slate-900">
            <label className="flex min-w-0 items-center gap-2 text-xs">{text('笔刷直径', 'Brush diameter')}<input aria-label={text('笔刷直径', 'Brush diameter')} type="range" min={1} max={Math.max(128, Math.min(2048, Math.max(info.width, info.height)))} value={diameter} onChange={(e) => setDiameter(Number(e.target.value))} className="w-24 sm:w-32" disabled={saving} /><span className="w-14 font-mono">{diameter}px</span></label>
            <label className="flex items-center gap-2 text-xs">{text('叠加透明度', 'Overlay opacity')}<input aria-label={text('叠加透明度', 'Overlay opacity')} type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} className="w-24" /><span>{Math.round(opacity * 100)}%</span></label>
            <Switch className="studio-switch-small" checked={maskOnly} onCheckedChange={setMaskOnly}>{text('仅看黑白遮罩', 'Mask only')}</Switch>
            <div className="flex items-center gap-1"><button className={control} onClick={() => setZoom((value) => Math.max(0.25, value / 1.5))} aria-label={text('缩小', 'Zoom out')}><ZoomOut className="h-4 w-4" /></button><button className={control} onClick={() => { setZoom(1); viewport.current?.scrollTo(0, 0); }}><Maximize className="h-4 w-4" />{text('适应', 'Fit')}</button><button className={control} onClick={() => setZoom((value) => Math.min(8, value * 1.5))} aria-label={text('放大', 'Zoom in')}><ZoomIn className="h-4 w-4" /></button><span className="ml-1 text-xs font-mono">{Math.round(scale * 100)}%</span></div>
          </div>
          <div className="flex flex-wrap justify-between gap-2 text-xs text-slate-500 dark:text-slate-400"><span>{text('来源：', 'Source: ')}{info.source === 'sidecar' ? info.filename : info.source === 'alpha' ? text('原图 Alpha 通道', 'Image alpha channel') : text('无遮罩或 Alpha，默认全图参与', 'No mask or alpha; the full image participates')}{info.resized && text('（已有遮罩尺寸已适配原图）', ' (existing mask fitted to image dimensions)')}</span><span>{text('参与比例', 'Participation')} {Math.round(coverage * 100)}% · {doc.current?.dirty ? text('有未保存修改', 'Unsaved changes') : text('已同步', 'Up to date')}</span></div>
          <div ref={viewport} className="relative h-[52vh] min-h-52 max-h-[580px] overflow-auto rounded-lg bg-slate-950 p-3" data-testid="mask-viewport">
            <div className="relative mx-auto" style={{ width: Math.max(1, info.width * scale), height: Math.max(1, info.height * scale) }}>
              <img src={apiUrl(maskEndpoint(datasetId, imageId, relPath, '/source'))} alt={relPath} draggable={false} onError={() => setError(text('原图预览加载失败，请重新读取或检查训练机连接。', 'Source preview failed to load. Reload or check the trainer connection.'))} className="absolute inset-0 h-full w-full" />
              <canvas ref={canvas} aria-label={text('遮罩绘制画布', 'Mask drawing canvas')} tabIndex={0} onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerLeave={() => { if (brushPreview.current) brushPreview.current.style.visibility = 'hidden'; }} onPointerUp={finishStroke} onPointerCancel={finishStroke} onLostPointerCapture={finishStroke} className={`absolute inset-0 h-full w-full touch-none ${tool === 'pan' ? 'cursor-grab' : 'cursor-crosshair'}`} />
              <div ref={brushPreview} aria-hidden="true" className="pointer-events-none invisible absolute rounded-full border border-white shadow-[0_0_0_1px_black]" style={{ width: diameter * scale, height: diameter * scale, transform: 'translate(-50%, -50%)' }} />
            </div>
          </div>
          {coverage === 0 && <p className="text-sm text-amber-600 dark:text-amber-400">{text('当前为全黑，这张图片不会贡献训练损失。', 'This mask is entirely black; this image will not contribute training loss.')}</p>}
        </>}
        {message && <p role="status" className="text-sm text-green-700 dark:text-green-400">{message}</p>}
      </div>
      <footer className="sticky bottom-0 flex flex-wrap items-center justify-between gap-3 border-t border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-800">
        <span className="text-xs text-slate-500 dark:text-slate-400">{text('保存为同目录 .mask.png；Ctrl/Cmd+Z 撤销，Ctrl/Cmd+S 保存。', 'Saved beside the image as .mask.png. Ctrl/Cmd+Z to undo; Ctrl/Cmd+S to save.')}</span>
        <div className="flex flex-wrap gap-2"><button className={control} disabled={saving} onClick={close}>{text('返回数据集', 'Back to dataset')}</button><button className={control} disabled={saving || loading || !info} onClick={() => void save(false)}><Save className="h-4 w-4" />{saving ? text('保存中…', 'Saving…') : text('保存遮罩', 'Save mask')}</button><button className="min-h-8 rounded-md bg-blue-600 px-3 py-1.5 text-xs font-medium text-white disabled:opacity-40" disabled={saving || loading || !info} onClick={() => void save(true)}>{text('保存并启用遮罩训练', 'Save and enable masked training')}</button></div>
      </footer>
    </section>
  </div>;
}
