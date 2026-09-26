import React from 'react';
import { ChevronLeft, ChevronRight, Download, Image as ImageIcon, ImageOff, Maximize2, Minimize2, Minus, Plus, Scan } from 'lucide-react';
import type { JobSample } from '../../api/types';
import ConfigHelp from '../../components/ConfigHelp';
import CopyButton from '../../components/CopyButton';
import { LazyImage } from '../../components/Loading';
import { epochAt, epochText, inEpochs, parseEpochQuery } from '../../utils/epochFilter';
import { formatTime } from '../../utils/format';
import { sampleSource } from '../../utils/sampleMedia';
import { useWorkspaceText } from '../../utils/workspaceText';
import EpochSearch from './EpochSearch';
import './job-samples.css';

const MAX_SCALE = 8;

type Sample = JobSample & { er_sde_eta?: number | null; er_sde_s_noise?: number | null };
type View = { scale: number; x: number; y: number };
type ZoomControls = { zoomIn: () => void; zoomOut: () => void; fit: () => void };

/** The selected preview, fitted to the stage; zoom with the buttons, Ctrl/⌘ + wheel, pinch or a double click, and drag to pan. */
const ZoomStage = React.forwardRef<ZoomControls, {
  sample: Sample; onPrevious?: () => void; onNext?: () => void; expanded: boolean; onExpand: () => void;
}>(function ZoomStage({ sample, onPrevious, onNext, expanded, onExpand }, controls) {
  const text = useWorkspaceText();
  const src = sampleSource(sample.url);
  const stage = React.useRef<HTMLDivElement>(null);
  const image = React.useRef<HTMLImageElement>(null);
  const drag = React.useRef<{ id: number; x: number; y: number; from: View } | null>(null);
  const [view, setView] = React.useState<View>({ scale: 1, x: 0, y: 0 });
  const [fit, setFit] = React.useState(1);
  const [loaded, setLoaded] = React.useState('');
  const [failed, setFailed] = React.useState('');

  const clamp = React.useCallback((next: View): View => {
    const box = stage.current, img = image.current;
    const scale = Math.min(MAX_SCALE, Math.max(1, next.scale));
    if (!box || !img) return { scale, x: 0, y: 0 };
    const maxX = Math.max(0, (img.clientWidth * scale - box.clientWidth) / 2), maxY = Math.max(0, (img.clientHeight * scale - box.clientHeight) / 2);
    return { scale, x: Math.min(maxX, Math.max(-maxX, next.x)), y: Math.min(maxY, Math.max(-maxY, next.y)) };
  }, []);
  // Keep the point under the pointer in place while the scale changes.
  const zoom = React.useCallback((scale: (current: number) => number, point = { x: 0, y: 0 }) => setView(current => {
    const target = Math.min(MAX_SCALE, Math.max(1, scale(current.scale)));
    const ratio = target / current.scale;
    return clamp({ scale: target, x: point.x - (point.x - current.x) * ratio, y: point.y - (point.y - current.y) * ratio });
  }), [clamp]);
  React.useImperativeHandle(controls, () => ({
    zoomIn: () => zoom(current => current * 1.25), zoomOut: () => zoom(current => current / 1.25), fit: () => setView({ scale: 1, x: 0, y: 0 }),
  }), [zoom]);
  const pointIn = (clientX: number, clientY: number) => {
    const rect = stage.current!.getBoundingClientRect();
    return { x: clientX - rect.left - rect.width / 2, y: clientY - rect.top - rect.height / 2 };
  };

  React.useEffect(() => { setView({ scale: 1, x: 0, y: 0 }); }, [src, expanded]);
  React.useEffect(() => {
    const box = stage.current;
    if (!box) return;
    const measure = () => { const img = image.current; if (img?.naturalWidth) setFit(img.clientWidth / img.naturalWidth); setView(current => clamp(current)); };
    const observer = new ResizeObserver(measure);
    observer.observe(box);
    // Trackpad pinch arrives as a wheel event with ctrlKey; React's wheel listener cannot cancel page zoom.
    const onWheel = (event: WheelEvent) => {
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      zoom(current => current * Math.exp(-event.deltaY * 0.01), pointIn(event.clientX, event.clientY));
    };
    box.addEventListener('wheel', onWheel, { passive: false });
    return () => { observer.disconnect(); box.removeEventListener('wheel', onWheel); };
  }, [clamp, zoom]);

  const actual = fit < 1 ? 1 / fit : 2;
  const percent = Math.round(view.scale * fit * 100);
  const ready = loaded === src;
  return <div className="sample-stage-frame">
    <div ref={stage} className="sample-stage" data-zoomed={view.scale > 1 || undefined}
      onDoubleClick={event => { if (view.scale > 1) setView({ scale: 1, x: 0, y: 0 }); else zoom(() => actual, pointIn(event.clientX, event.clientY)); }}
      onPointerDown={event => {
        // Paging buttons sit on the stage; a press on them is a click, not a pan.
        if (view.scale <= 1 || event.button !== 0 || (event.target as HTMLElement).closest('button')) return;
        event.currentTarget.setPointerCapture(event.pointerId);
        drag.current = { id: event.pointerId, x: event.clientX, y: event.clientY, from: view };
      }}
      onPointerMove={event => {
        const start = drag.current;
        if (start?.id === event.pointerId) setView(clamp({ ...start.from, x: start.from.x + event.clientX - start.x, y: start.from.y + event.clientY - start.y }));
      }}
      onPointerUp={() => { drag.current = null; }} onPointerCancel={() => { drag.current = null; }}>
      {failed === src ? <span className="sample-stage-failed"><ImageOff size={26} aria-hidden="true"/>{text('采样图读取失败', 'The preview could not be loaded')}</span> : <>
        {!ready && <span className="ui-skeleton sample-stage-skeleton" aria-hidden="true"/>}
        <img key={src} ref={image} src={src} alt={sample.prompt} draggable={false} className={`sample-stage-image${ready ? '' : ' is-pending'}`}
          style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})` }}
          onLoad={event => { setLoaded(src); const img = event.currentTarget; if (img.naturalWidth) setFit(img.clientWidth / img.naturalWidth); }}
          onError={() => setFailed(src)}/>
      </>}
      {onPrevious && <button type="button" className="sample-stage-nav is-previous" onClick={onPrevious} onDoubleClick={event => event.stopPropagation()} aria-label={text('上一张', 'Previous')} title={text('上一张（←）', 'Previous (←)')}><ChevronLeft size={22}/></button>}
      {onNext && <button type="button" className="sample-stage-nav is-next" onClick={onNext} onDoubleClick={event => event.stopPropagation()} aria-label={text('下一张', 'Next')} title={text('下一张（→）', 'Next (→)')}><ChevronRight size={22}/></button>}
    </div>
    <div className="sample-stage-tools" role="group" aria-label={text('缩放', 'Zoom')}>
      <button type="button" onClick={() => zoom(current => current / 1.25)} disabled={view.scale <= 1} aria-label={text('缩小', 'Zoom out')} title={text('缩小（-）', 'Zoom out (-)')}><Minus size={15}/></button>
      <output aria-label={text('相对原图的显示比例', 'Scale relative to the image pixels')}>{ready ? `${percent}%` : '—'}</output>
      <button type="button" onClick={() => zoom(current => current * 1.25)} disabled={view.scale >= MAX_SCALE} aria-label={text('放大', 'Zoom in')} title={text('放大（+）', 'Zoom in (+)')}><Plus size={15}/></button>
      <button type="button" onClick={() => setView({ scale: 1, x: 0, y: 0 })} disabled={view.scale <= 1} aria-label={text('适应窗口', 'Fit')} title={text('适应窗口（0）', 'Fit (0)')}><Scan size={15}/></button>
      <button type="button" onClick={onExpand} aria-pressed={expanded} aria-label={expanded ? text('退出全屏', 'Exit full screen') : text('全屏查看', 'Full screen')} title={expanded ? text('退出全屏（Esc）', 'Exit full screen (Esc)') : text('全屏查看（F）', 'Full screen (F)')}>{expanded ? <Minimize2 size={15}/> : <Maximize2 size={15}/>}</button>
    </div>
    <span className="sr-only" aria-live="polite">{ready ? text(`显示比例 ${percent}%`, `Scale ${percent}%`) : ''}</span>
  </div>;
});

function SampleFacts({ sample, stepsPerEpoch, position, total }: { sample: Sample; stepsPerEpoch?: number | null; position: number; total: number }) {
  const text = useWorkspaceText();
  const epoch = epochAt(sample, stepsPerEpoch);
  const lossRecorded = sample.step > 0 && typeof sample.loss === 'number' && Number.isFinite(sample.loss);
  const loss = lossRecorded ? String(Number(sample.loss!.toPrecision(5))) : sample.step === 0 ? text('初始采样 · 未训练', 'Initial sample · untrained') : text('未记录', 'Not recorded');
  const negative = sample.negative?.trim();
  const parameters: Array<[string, React.ReactNode]> = ([
    [text('采样器', 'Sampler'), sample.sampler],
    [text('调度器', 'Scheduler'), sample.scheduler],
    [text('采样步数', 'Steps'), sample.steps],
    ['CFG', sample.cfg],
    ['Shift', sample.shift],
    ['Guidance', sample.guidance],
    // ER-SDE settings are recorded for every preview but only matter to that sampler.
    ...(sample.sampler?.startsWith('er_sde') ? [['ER-SDE eta', sample.er_sde_eta], ['ER-SDE s_noise', sample.er_sde_s_noise]] : []),
  ] as Array<[string, React.ReactNode]>).filter(([, value]) => value != null && value !== '');
  return <aside className="sample-facts" aria-label={text('采样图信息', 'Preview details')}>
    <header>
      <h2>{text(`第 ${sample.step} 步`, `Step ${sample.step}`)}{epoch != null && <span>{text(`第 ${epochText(epoch)} 轮`, `Epoch ${epochText(epoch)}`)}</span>}</h2>
      <span className="sample-facts-position">{position} / {total}</span>
    </header>
    <dl className="sample-facts-list">
      <div><dt>{text('步数', 'Step')}</dt><dd>{sample.step}</dd></div>
      <div><dt>Epoch</dt><dd>{epochText(epoch)}</dd></div>
      <div><dt>{text('种子', 'Seed')}</dt><dd className="sample-facts-seed"><span>{sample.seed}</span><CopyButton value={String(sample.seed)} label={text('复制种子', 'Copy seed')}/></dd></div>
      <div><dt>Loss<ConfigHelp label={text('Loss · 说明', 'Loss · help')}>{text(`第 ${sample.step} 步记录的训练损失，不是这张采样图的质量评分。`, `Training loss recorded at step ${sample.step}, not a quality score for this image.`)}</ConfigHelp></dt><dd data-recorded={lossRecorded}>{loss}</dd></div>
      <div><dt>{text('尺寸', 'Size')}</dt><dd>{sample.width} × {sample.height}</dd></div>
      <div><dt>{text('生成时间', 'Created')}</dt><dd>{formatTime(sample.created_at)}</dd></div>
    </dl>
    <section className="sample-facts-text">
      <h3>{text(`提示词 · 第 ${sample.prompt_index + 1} 条`, `Prompt · #${sample.prompt_index + 1}`)}<CopyButton value={sample.prompt} label={text('复制提示词', 'Copy prompt')}/></h3>
      <p>{sample.prompt || '—'}</p>
    </section>
    {negative && <section className="sample-facts-text">
      <h3>{text('负面提示词', 'Negative prompt')}<CopyButton value={negative} label={text('复制负面提示词', 'Copy negative prompt')}/></h3>
      <p>{negative}</p>
    </section>}
    <section className="sample-facts-parameters">
      <h3>{text('采样参数', 'Sampling parameters')}</h3>
      {parameters.length ? <dl className="sample-facts-list">{parameters.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
        : <p>{text('此采样图没有记录采样参数（较早的任务）。', 'No sampling parameters were recorded for this preview (earlier jobs).')}</p>}
    </section>
    <a className="ui-btn ui-btn-sm sample-facts-download" href={sampleSource(sample.url)} download><Download size={14}/>{text('下载原图', 'Download image')}</a>
  </aside>;
}

function SkeletonViewer() {
  return <div className="sample-viewer" aria-busy="true">
    <div className="sample-main"><div className="sample-stage-frame"><div className="sample-stage"><span className="ui-skeleton sample-stage-skeleton" aria-hidden="true"/></div></div>
      <div className="sample-strip">{Array.from({ length: 6 }, (_, index) => <span key={index} className="ui-skeleton sample-thumb-skeleton" aria-hidden="true"/>)}</div></div>
    <div className="sample-facts">{Array.from({ length: 7 }, (_, index) => <span key={index} className="ui-skeleton sample-facts-skeleton" aria-hidden="true"/>)}</div>
  </div>;
}

/**
 * Two-column preview browser: the image on the left with paging and zoom, what produced it on the right.
 * Without a selection it follows the newest preview.
 */
export default function SampleViewer({ samples, stepsPerEpoch, loaded, selected, onSelect }: {
  samples: JobSample[]; stepsPerEpoch?: number | null; loaded: boolean; selected: string | null; onSelect: (url: string | null) => void;
}) {
  const text = useWorkspaceText();
  const [query, setQuery] = React.useState('');
  const [expanded, setExpanded] = React.useState(false);
  const strip = React.useRef<HTMLDivElement>(null);
  const zoom = React.useRef<ZoomControls>(null);
  const ranges = React.useMemo(() => parseEpochQuery(query), [query]);
  const shown = React.useMemo(() => samples.filter(sample => inEpochs(epochAt(sample, stepsPerEpoch), ranges)) as Sample[], [samples, stepsPerEpoch, ranges]);
  const found = selected ? shown.findIndex(sample => sample.url === selected) : -1;
  const index = found >= 0 ? found : shown.length - 1;
  const current = shown[index];
  const go = React.useCallback((next: number) => {
    const target = shown[next];
    // Reaching the newest preview follows new ones again.
    if (target) onSelect(next === shown.length - 1 ? null : target.url);
  }, [shown, onSelect]);
  const groups = React.useMemo(() => {
    const byStep: Array<{ step: number; items: Array<{ sample: Sample; index: number }> }> = [];
    shown.forEach((sample, position) => {
      const last = byStep[byStep.length - 1];
      if (last?.step === sample.step) last.items.push({ sample, index: position }); else byStep.push({ step: sample.step, items: [{ sample, index: position }] });
    });
    return byStep;
  }, [shown]);

  React.useLayoutEffect(() => {
    const box = strip.current, thumb = box?.querySelector<HTMLElement>('[aria-current="true"]');
    if (!box || !thumb) return;
    const start = thumb.offsetLeft, end = start + thumb.offsetWidth;
    if (start < box.scrollLeft) box.scrollLeft = start - 12;
    else if (end > box.scrollLeft + box.clientWidth) box.scrollLeft = end - box.clientWidth + 12;
  }, [index, shown.length, expanded]);
  React.useEffect(() => {
    if (!expanded) return;
    const close = (event: KeyboardEvent) => { if (event.key === 'Escape') setExpanded(false); };
    window.addEventListener('keydown', close);
    return () => window.removeEventListener('keydown', close);
  }, [expanded]);

  if (!loaded) return <SkeletonViewer/>;
  if (!samples.length) return <div className="sample-empty"><ImageIcon size={30} aria-hidden="true"/><p>{text('暂无采样图片', 'No previews yet')}</p><span>{text('训练中生成的预览图会自动出现在这里。', 'Previews generated during training appear here automatically.')}</span></div>;

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.target instanceof HTMLInputElement || event.altKey || event.ctrlKey || event.metaKey) return;
    const actions: Record<string, () => void> = {
      ArrowLeft: () => go(index - 1), ArrowRight: () => go(index + 1),
      '-': () => zoom.current?.zoomOut(), '+': () => zoom.current?.zoomIn(), '=': () => zoom.current?.zoomIn(), '0': () => zoom.current?.fit(),
      f: () => setExpanded(value => !value),
    };
    const action = actions[event.key];
    if (action) { event.preventDefault(); action(); }
  };

  return <section className="sample-browser" aria-label={text('采样图', 'Previews')}>
    <div className="sample-toolbar">
      <EpochSearch value={query} onChange={setQuery} label={text('按轮次搜索采样图', 'Search previews by epoch')}/>
      <span className="sample-toolbar-count">{ranges && ranges !== 'invalid' ? text(`找到 ${shown.length} 张，共 ${samples.length} 张`, `${shown.length} of ${samples.length}`) : text(`共 ${samples.length} 张`, `${samples.length} previews`)}</span>
    </div>
    {!current ? <div className="sample-empty"><ImageIcon size={26} aria-hidden="true"/><p>{text('没有符合轮次的采样图', 'No previews in these epochs')}</p><button type="button" className="ui-link" onClick={() => setQuery('')}>{text('清除搜索', 'Clear search')}</button></div>
      : <div className={`sample-viewer${expanded ? ' is-expanded' : ''}`} tabIndex={-1} onKeyDown={onKeyDown} role={expanded ? 'dialog' : undefined} aria-modal={expanded || undefined} aria-label={expanded ? text('全屏查看采样图', 'Full-screen preview') : undefined}>
        <div className="sample-main">
          <ZoomStage ref={zoom} sample={current} expanded={expanded} onExpand={() => setExpanded(value => !value)}
            onPrevious={index > 0 ? () => go(index - 1) : undefined} onNext={index < shown.length - 1 ? () => go(index + 1) : undefined}/>
          <div ref={strip} className="sample-strip" aria-label={text('采样图列表', 'Preview list')}>
            {groups.map(group => <div key={group.step} className="sample-strip-group">
              <span className="sample-strip-label">{text(`步 ${group.step}`, `Step ${group.step}`)}</span>
              <div className="sample-strip-items">{group.items.map(({ sample, index: position }) => <button type="button" key={sample.url} className="sample-thumb" aria-current={position === index}
                aria-label={text(`第 ${sample.step} 步 · 第 ${sample.prompt_index + 1} 条提示词`, `Step ${sample.step} · prompt ${sample.prompt_index + 1}`)} onClick={() => go(position)}>
                <LazyImage src={sampleSource(sample.url)} alt="" loading="lazy" draggable={false}/>
              </button>)}</div>
            </div>)}
          </div>
        </div>
        <SampleFacts sample={current} stepsPerEpoch={stepsPerEpoch} position={index + 1} total={shown.length}/>
      </div>}
  </section>;
}
