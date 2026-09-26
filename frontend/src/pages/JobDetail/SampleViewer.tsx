import React from 'react';
import { Download, Image as ImageIcon, ImageOff } from 'lucide-react';
import type { JobSample } from '../../api/types';
import ConfigHelp from '../../components/ConfigHelp';
import CopyButton from '../../components/CopyButton';
import { LazyImage } from '../../components/Loading';
import SampleLightbox from '../../components/sampling/SampleLightbox';
import { epochAt, epochText, inEpochs, parseEpochQuery } from '../../utils/epochFilter';
import { formatTime } from '../../utils/format';
import { sampleSource } from '../../utils/sampleMedia';
import { useWorkspaceText } from '../../utils/workspaceText';
import EpochSearch from './EpochSearch';
import './job-samples.css';

type Sample = JobSample & { er_sde_eta?: number | null; er_sde_s_noise?: number | null };

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
    <div className="sample-main"><div className="sample-stage"><span className="ui-skeleton sample-stage-skeleton" aria-hidden="true"/></div>
      <div className="sample-strip">{Array.from({ length: 6 }, (_, index) => <span key={index} className="ui-skeleton sample-thumb-skeleton" aria-hidden="true"/>)}</div></div>
    <div className="sample-facts">{Array.from({ length: 7 }, (_, index) => <span key={index} className="ui-skeleton sample-facts-skeleton" aria-hidden="true"/>)}</div>
  </div>;
}

/**
 * Two-column preview browser: the chosen preview on the left, what produced it on the right. Clicking the preview
 * opens it over the whole page with zoom and paging. Without a selection it follows the newest preview.
 */
export default function SampleViewer({ samples, stepsPerEpoch, loaded, selected, onSelect }: {
  samples: JobSample[]; stepsPerEpoch?: number | null; loaded: boolean; selected: string | null; onSelect: (url: string | null) => void;
}) {
  const text = useWorkspaceText();
  const [query, setQuery] = React.useState('');
  const [open, setOpen] = React.useState(false);
  const [failed, setFailed] = React.useState('');
  const strip = React.useRef<HTMLDivElement>(null);
  const ranges = React.useMemo(() => parseEpochQuery(query), [query]);
  const shown = React.useMemo(() => samples.filter(sample => inEpochs(epochAt(sample, stepsPerEpoch), ranges)) as Sample[], [samples, stepsPerEpoch, ranges]);
  const found = selected ? shown.findIndex(sample => sample.url === selected) : -1;
  const index = found >= 0 ? found : shown.length - 1;
  const current = shown[index];
  const go = React.useCallback((next: number) => {
    const target = shown[next];
    // Reaching the newest preview follows new ones again, except in the viewer, which keeps what it shows.
    if (target) onSelect(!open && next === shown.length - 1 ? null : target.url);
  }, [shown, onSelect, open]);
  const openViewer = () => { if (current) onSelect(current.url); setOpen(true); };
  const closeViewer = () => { setOpen(false); if (index === shown.length - 1) onSelect(null); };
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
  }, [index, shown.length]);

  if (!loaded) return <SkeletonViewer/>;
  if (!samples.length) return <div className="sample-empty"><ImageIcon size={30} aria-hidden="true"/><p>{text('暂无采样图片', 'No previews yet')}</p><span>{text('训练中生成的预览图会自动出现在这里。', 'Previews generated during training appear here automatically.')}</span></div>;

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (open || event.target instanceof HTMLInputElement || event.altKey || event.ctrlKey || event.metaKey) return;
    const actions: Record<string, () => void> = { ArrowLeft: () => go(index - 1), ArrowRight: () => go(index + 1), f: openViewer };
    const action = actions[event.key];
    if (action) { event.preventDefault(); action(); }
  };
  const src = current ? sampleSource(current.url) : '';
  const epoch = current ? epochAt(current, stepsPerEpoch) : null;

  return <section className="sample-browser" aria-label={text('采样图', 'Previews')}>
    <div className="sample-toolbar">
      <EpochSearch value={query} onChange={setQuery} label={text('按轮次搜索采样图', 'Search previews by epoch')}/>
      <span className="sample-toolbar-count">{ranges && ranges !== 'invalid' ? text(`找到 ${shown.length} 张，共 ${samples.length} 张`, `${shown.length} of ${samples.length}`) : text(`共 ${samples.length} 张`, `${samples.length} previews`)}</span>
    </div>
    {!current ? <div className="sample-empty"><ImageIcon size={26} aria-hidden="true"/><p>{text('没有符合轮次的采样图', 'No previews in these epochs')}</p><button type="button" className="ui-link" onClick={() => setQuery('')}>{text('清除搜索', 'Clear search')}</button></div>
      : <div className="sample-viewer" tabIndex={-1} onKeyDown={onKeyDown}>
        <div className="sample-main">
          <button type="button" className="sample-stage" onClick={event => { event.currentTarget.focus(); openViewer(); }} aria-label={text(`放大查看第 ${current.step} 步的采样图`, `Enlarge the step ${current.step} preview`)} title={text('点击放大查看', 'Click to enlarge')}>
            {failed === src ? <span className="sample-stage-failed"><ImageOff size={26} aria-hidden="true"/>{text('采样图读取失败', 'The preview could not be loaded')}</span>
              : <LazyImage key={src} src={src} alt={current.prompt} draggable={false} className="sample-stage-image" onError={() => setFailed(src)}/>}
          </button>
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
    {open && current && <SampleLightbox sample={current} position={index + 1} total={shown.length} onClose={closeViewer}
      details={epoch != null ? [text(`第 ${epochText(epoch)} 轮`, `Epoch ${epochText(epoch)}`)] : []}
      onPrevious={index > 0 ? () => go(index - 1) : undefined} onNext={index < shown.length - 1 ? () => go(index + 1) : undefined}/>}
  </section>;
}
