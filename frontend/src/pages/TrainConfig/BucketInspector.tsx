import { useState } from 'react';
import { BarChart3, Grid2X2, Database, Loader2 } from 'lucide-react';
import type { Plan } from '../../api/types';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatBytesMB, formatParams } from '../../utils/format';
import SourceBalance from './SourceBalance';
import {configOptionLabel} from '../../utils/configPresentation';
import ConfigHelp from '../../components/ConfigHelp';
import { SlidingIndicator } from '../../components/motion';

type DatasetSizing = { resolution_mode?: string; native_max_pixels?: number; native_max_side?: number };
type PlanBucket = NonNullable<Plan['buckets']>[number];
type SourceRoute = NonNullable<PlanBucket['sources']>[number];
const bucketKey = (bucket: PlanBucket) => `${bucket.base ?? 0}:${bucket.w}x${bucket.h}`;
/** Buckets from different base resolutions share one grid only when there is a single base. */
function groupByBase(buckets: PlanBucket[]) {
  const groups = new Map<number, PlanBucket[]>();
  for (const bucket of buckets) groups.set(bucket.base ?? 0, [...(groups.get(bucket.base ?? 0) || []), bucket]);
  return [...groups].sort(([a], [b]) => a - b).map(([base, items]) => ({ base, buckets: items, items: items.reduce((sum, bucket) => sum + bucket.items, 0) }));
}
/** Memory fixes the planner can name, each linked to the setting it changes. */
function memoryFixes(suggestions: string[], native: boolean, text: (zh: string, en: string) => string) {
  const known: [RegExp, string, string][] = [
    [/activation_checkpointing/, 'memory.activation_checkpointing', text('开启重算中间结果', 'Turn on activation checkpointing')],
    [/blocks_to_swap/, 'memory.blocks_to_swap', text('把部分模型块换出到内存', 'Swap model blocks to system memory')],
    [/adamw8bit/, 'optimizer.type', text('改用 AdamW 8-bit 优化器', 'Use the AdamW 8-bit optimizer')],
    [/text_encoding/, 'dataset.text_encoding', text('训练前缓存文本特征', 'Cache text features before training')],
    [/Adafactor/, 'optimizer.type', text('改用 Adafactor 并不设置 beta1', 'Use Adafactor without beta1')],
  ];
  const fixes = known.filter(([pattern]) => suggestions.some(item => pattern.test(item))).map(([, path, label]) => ({ path, label }));
  fixes.push({ path: 'dataset.batch_size', label: text('减小批量大小', 'Reduce the batch size') });
  fixes.push(native ? { path: 'dataset.native_max_pixels', label: text('降低图像面积上限', 'Lower the image area limit') } : { path: 'dataset.resolutions', label: text('降低训练分辨率', 'Lower the training resolution') });
  return fixes.filter((fix, index) => fixes.findIndex(item => item.path === fix.path) === index);
}

export default function BucketInspector({ plan, loading, onData, hasSources = false, indexed, onIssues, onField, error, onRetry, dataset }: { plan: Plan | null; loading: boolean; onData: () => void; hasSources?: boolean; indexed?: { images: number; captioned: number }; onIssues?: () => void; onField?: (path: string) => void; error?:string; onRetry?:()=>void; dataset?: DatasetSizing }) {
  const text = useWorkspaceText();
  const [view, setView] = useState<'shape' | 'table'>('shape');
  const [selected, setSelected] = useState<string | null>(null);
  const buckets = plan?.buckets || [];
  const native = plan?.native;
  const nativeMode = dataset ? dataset.resolution_mode === 'native' : !!native;
  const awaitingPlan = hasSources && !buckets.length && !plan?.ok;
  const maxCount = Math.max(1, ...buckets.map(bucket => bucket.items));
  const chosen = buckets.find(bucket => bucketKey(bucket) === selected);
  const groups = nativeMode ? [{ base: 0, buckets, items: 0 }] : groupByBase(buckets);
  const grouped = groups.length > 1;
  const baseLabel = (base: number) => text(`分辨率 ${base}`, `Resolution ${base}`);
  const fitMode = plan?.image_fit?.mode === 'pad' ? 'pad' : 'crop';
  const sourceLine = (bucket: PlanBucket) => {
    const routes = bucket.sources || [];
    if ((bucket.source_variants || routes.length) > 1) return text(`${bucket.source_variants || routes.length} 种原图尺寸`, `${bucket.source_variants || routes.length} source sizes`);
    const route = routes[0];
    if (!route) return '';
    return route.width === bucket.w && route.height === bucket.h ? text('原图尺寸', 'Source size') : text(`原图 ${route.width} × ${route.height}`, `from ${route.width} × ${route.height}`);
  };
  // Source → resize → crop or padding → training size, in the order the loader applies them.
  const routeSteps = (bucket: PlanBucket, route: SourceRoute) => {
    const steps = [text(`原图 ${route.width} × ${route.height}`, `Source ${route.width} × ${route.height}`)];
    if (route.resized_width !== route.width || route.resized_height !== route.height) {
      const smaller = route.resized_width * route.resized_height < route.width * route.height;
      steps.push(smaller ? text(`缩小到 ${route.resized_width} × ${route.resized_height}`, `downscaled to ${route.resized_width} × ${route.resized_height}`) : text(`放大到 ${route.resized_width} × ${route.resized_height}`, `upscaled to ${route.resized_width} × ${route.resized_height}`));
    }
    const dw = Math.abs(route.resized_width - bucket.w); const dh = Math.abs(route.resized_height - bucket.h);
    if (dw || dh) {
      const action = fitMode === 'pad' ? text('补边', 'padded') : text('裁去', 'cropped');
      steps.push(dw && dh ? text(`${action} ${dw} × ${dh} px`, `${action} ${dw} × ${dh} px`)
        : dw ? text(`宽度${action} ${dw} px`, `${action} ${dw} px wide`) : text(`高度${action} ${dh} px`, `${action} ${dh} px tall`));
    }
    return steps.length === 1 ? [text(`与原图一致，未缩放或${fitMode === 'pad' ? '补边' : '裁切'}`, `Same as the source, no resize or ${fitMode === 'pad' ? 'padding' : 'crop'}`)] : steps;
  };
  const sizedBuckets = [...buckets].filter(bucket => bucket.sources?.length).sort((a, b) => b.items - a.items);
  // Lines break only between steps, so a size never splits across lines.
  const routeRows = (bucket: PlanBucket, limit: number) => {
    const routes = (bucket.sources || []).slice(0, limit);
    const more = Math.max(bucket.source_variants || 0, bucket.sources?.length || 0) - routes.length;
    return <>
      {routes.map(route => <div key={`${route.width}x${route.height}/${route.resized_width}x${route.resized_height}`} className="size-route"><p>{routeSteps(bucket, route).map(step => step.replace(/ /g, '\u00a0')).join(' → ')}</p><small>{route.images} {text('张', 'images')}</small></div>)}
      {more > 0 && <small className="size-route-more">{text(`另有 ${more} 种原图尺寸`, `${more} more source sizes`)}</small>}
    </>;
  };
  const sizeSummary = (() => {
    if (!native) return fitMode === 'pad'
      ? text('每张图片先等比缩放到能完整放入分桶，再补边填满。', 'Each image is scaled to fit inside its bucket, then padded to fill it.')
      : text('每张图片先等比缩放到刚好覆盖分桶，再裁去超出的部分。', 'Each image is scaled to just cover its bucket, then the overflow is cropped.');
    const kept = native.images - native.downscaled;
    const scaled = native.downscaled === 0
      ? text(`${native.images} 张都在面积上限内，保持原尺寸`, `All ${native.images} images fit the area limit and keep their source size`)
      : kept <= 0
        ? text(`${native.images} 张都超过面积上限，已等比缩小`, `All ${native.images} images exceed the area limit and are scaled down`)
        : text(`${kept} 张在面积上限内保持原尺寸，${native.downscaled} 张超过上限等比缩小`, `${kept} images keep their source size within the area limit; ${native.downscaled} exceed it and are scaled down`);
    const aligned = fitMode === 'pad'
      ? text(`宽高不是 ${native.alignment} 的倍数时，再在边缘补少量空白对齐`, `sizes that are not multiples of ${native.alignment} are then padded slightly at the edges`)
      : text(`宽高不是 ${native.alignment} 的倍数时，再裁去少量边缘对齐`, `sizes that are not multiples of ${native.alignment} are then trimmed slightly at the edges`);
    return text(`${scaled}；${aligned}。`, `${scaled}; ${aligned}.`);
  })();
  const peak = plan?.memory?.peak_mb_estimate ?? null;
  const capacity = plan?.memory?.gpu_total_mb ?? null;
  const overCapacity = peak != null && !!capacity && peak > capacity * 0.95;
  const tightMemory = !overCapacity && peak != null && !!capacity && peak > capacity * 0.9;
  const memoryBlocked = overCapacity && !!plan?.errors?.some(item => item.loc === 'memory');
  const tile = (bucket: PlanBucket) => {
    const key = bucketKey(bucket);
    const longest = Math.max(bucket.w, bucket.h);
    return <button type="button" key={key} className={`bucket-tile ${key === selected ? 'is-selected' : ''}`} aria-pressed={key === selected} aria-label={`${grouped ? `${baseLabel(bucket.base)} · ` : ''}${bucket.w} × ${bucket.h}, ${bucket.items} ${text('样本', 'samples')}`} onClick={() => setSelected(key === selected ? null : key)}>
      <span className="bucket-shape-space"><span className="bucket-shape" style={{width: `${bucket.w / longest * 56}px`, height: `${bucket.h / longest * 56}px`}}><span>{bucket.items}</span></span></span>
      <span className="bucket-size">{bucket.w} × {bucket.h}</span>
      {nativeMode && sourceLine(bucket) && <span className="bucket-source" title={sourceLine(bucket)}>{sourceLine(bucket)}</span>}
      <span className="bucket-count-track"><span style={{width: `${bucket.items / maxCount * 100}%`}} /></span>
    </button>;
  };
  return <section className="bucket-inspector" aria-label={native ? text('原生尺寸与训练估算', 'Native sizes and training estimates') : text('数据分桶与训练估算', 'Buckets and training estimates')}>
    <div className="inspector-heading"><h3><BarChart3 size={15} />{text('数据分布', 'Dataset distribution')}</h3>{loading && <Loader2 size={14} className="animate-spin" aria-label={text('正在更新', 'Updating')} />}</div>
    <div className={`inspector-content ${loading ? 'opacity-60' : ''}`} aria-busy={loading}>
      {loading && <p role="status" className="inspector-calculation-status">{plan ? text('正在更新数据分布，以下为上次结果…','Updating dataset distribution; the previous results are shown below…') : text('正在计算数据分布…','Calculating dataset distribution…')}</p>}
      {error && <div role="alert" className="inspector-calculation-error"><strong>{text('数据分布计算失败','Dataset distribution could not be calculated')}</strong><p>{error}</p>{plan && <p>{text('以下为上次计算结果。','The previous calculation is shown below.')}</p>}{onRetry && <button type="button" className="ui-link" onClick={onRetry} disabled={loading}>{text('重新计算','Recalculate')}</button>}</div>}
      <dl className="dataset-metrics">
        <div><dt>{awaitingPlan && indexed ? text('已索引图片', 'Indexed images') : text('原始图片', 'Images')}</dt><dd>{awaitingPlan ? indexed?.images ?? '—' : plan?.images ?? '—'}</dd></div>
        <div><dt>{text('重复后样本', 'Repeated samples')}</dt><dd>{awaitingPlan ? '—' : plan?.items ?? '—'}</dd></div>
        <div><dt>{text('已配标签', 'Captioned')}</dt><dd>{awaitingPlan ? indexed?.captioned ?? '—' : plan?.captioned ?? '—'}</dd></div>
        <div><dt>{native ? text('独立尺寸', 'Distinct sizes') : text('分桶数量', 'Buckets')}</dt><dd>{awaitingPlan ? '—' : plan ? buckets.length : '—'}</dd></div>
      </dl>
      <SourceBalance sources={plan?.source_balance} loading={loading} hasSources={hasSources}/>
      {!!native?.synchronization_groups && <p className="inspector-note">{text(`多卡每轮额外 ${native.synchronization_groups} 次补齐计算，不计入损失，也不增加训练样本。`, `Multi-GPU training adds ${native.synchronization_groups} padding runs per epoch; they carry no loss and add no samples.`)}</p>}
      <div className="bucket-heading"><h4>{nativeMode ? text('实际训练尺寸', 'Training sizes') : text('分桶布局', 'Bucket layout')}</h4><div className="ui-segmented ui-segmented-sm" role="group" aria-label={text('分桶显示方式', 'Bucket view')}><button type="button" aria-label={text('分桶图形视图', 'Bucket shape view')} aria-pressed={view === 'shape'} onClick={() => setView('shape')}><Grid2X2 size={13} /></button><button type="button" aria-label={text('分桶明细表', 'Bucket table')} aria-pressed={view === 'table'} onClick={() => setView('table')}><BarChart3 size={13} /></button><SlidingIndicator className="ui-segmented-thumb"/></div></div>
      {buckets.length === 0 ? <div className="bucket-empty"><Database size={23} /><p>{loading ? text('正在计算实际分桶…', 'Computing buckets…') : awaitingPlan ? text('请完成待配置项后计算。', 'Complete the pending settings to calculate.') : text('尚无训练图片。', 'No training images yet.')}</p><button type="button" className="ui-link" onClick={awaitingPlan && onIssues ? onIssues : onData}>{awaitingPlan && onIssues ? text('检查待配置项', 'Review pending settings') : text('配置训练数据', 'Configure dataset')}</button></div> : <>
        {view === 'shape' ? <div className="bucket-groups" data-testid="plan-buckets">{groups.map(group => grouped
          ? <section key={group.base} className="bucket-group" aria-label={baseLabel(group.base)}>
            <h5 className="bucket-group-heading"><span>{baseLabel(group.base)}</span><small>{text(`${group.buckets.length} 个分桶 · ${group.items} 样本`, `${group.buckets.length} buckets · ${group.items} samples`)}</small></h5>
            <div className="bucket-grid">{group.buckets.map(tile)}</div>
          </section>
          : <div key={group.base} className="bucket-grid">{group.buckets.map(tile)}</div>)}</div>
          : <div className="bucket-table-wrap" data-testid="plan-buckets"><table className="bucket-table"><thead><tr>{grouped && <th>{text('分辨率', 'Resolution')}</th>}<th>{text('尺寸', 'Size')}</th><th>{text('样本', 'Items')}</th><th>{native ? <span className="bucket-column-help">{text('计算次数', 'Model runs')}<ConfigHelp label={text('计算次数说明', 'Model runs help')}>{text('每轮对这个尺寸运行模型的次数。同尺寸图片在不超过图像面积上限时合并为一次计算；单张已接近上限时逐张计算，因此常与样本数相同。', 'How many times the model runs on this size per epoch. Same-size images are combined while they fit the image area limit; images near the limit run one at a time, so this often equals the sample count.')}</ConfigHelp></span> : text('批次', 'Batches')}</th></tr></thead><tbody>{buckets.map(bucket => <tr key={bucketKey(bucket)}>{grouped && <td>{bucket.base}</td>}<td>{bucket.w} × {bucket.h}</td><td>{bucket.items}</td><td>{bucket.batches ?? '—'}</td></tr>)}</tbody></table></div>}
        {chosen && <div className="bucket-selection"><strong>{grouped ? `${baseLabel(chosen.base)} · ` : ''}{chosen.w} × {chosen.h}</strong><span>{chosen.items} {text('样本', 'samples')} · {chosen.batches ?? '—'} {native ? text('次计算 / 轮', 'model runs / epoch') : text('批次 / 轮', 'batches / epoch')}</span><span>{text('长宽比', 'Aspect ratio')} {(chosen.w / chosen.h).toFixed(2)}</span>{!!chosen.sources?.length && <div className="bucket-selection-routes">{routeRows(chosen, 4)}</div>}</div>}
      </>}
      {sizedBuckets.length > 0 && <section className="size-routes" aria-label={text('训练尺寸来源', 'How training sizes are reached')}>
        <h4>{text('尺寸变化', 'Size changes')}</h4>
        <p>{sizeSummary}</p>
        <ul>{sizedBuckets.slice(0, 6).map(bucket => <li key={bucketKey(bucket)} aria-label={`${grouped ? `${baseLabel(bucket.base)} · ` : ''}${bucket.w} × ${bucket.h}`}>
          <strong>{grouped ? `${baseLabel(bucket.base)} · ` : ''}{bucket.w} × {bucket.h}</strong>
          {routeRows(bucket, 2)}
        </li>)}</ul>
        {sizedBuckets.length > 6 && <p>{text(`另有 ${sizedBuckets.length - 6} 种训练尺寸，点选上方尺寸可查看。`, `${sizedBuckets.length - 6} more training sizes; select one above to see how it is reached.`)}</p>}
      </section>}
      {plan?.image_fit && <div className="image-fit-summary">
        <strong>{plan.image_fit.mode==='pad'?text('完整画面保留','Whole image preserved'):text('裁切保留位置','Crop anchor') + ' · ' + configOptionLabel('dataset.crop_anchor',plan.image_fit.crop_anchor || 'center',text('zh','en') === 'en')}</strong>
        <dl><div><dt>{text('裁切图片','Cropped images')}</dt><dd>{plan.image_fit.cropped_images}</dd></div><div><dt>{text('填充占比','Padding share')}</dt><dd>{(plan.image_fit.padding_fraction*100).toFixed(2)}%</dd></div></dl>
        {!!plan.image_fit.items.length && <details><summary>{text('查看尺寸适配明细','Inspect image sizing')}</summary><div className="image-fit-details">{plan.image_fit.items.map((item,index)=><div key={`${item.path}/${item.width}/${item.height}/${index}`}><strong title={item.path}>{item.path.replace(/\\/g,'/').split('/').pop()}</strong><span>{item.source_width} × {item.source_height} → {item.resized_width} × {item.resized_height}</span>{plan.image_fit?.mode==='crop' && item.cropped_pixels>0 && <small>{text(`裁去：上 ${item.top} · 下 ${item.resized_height-item.bottom} · 左 ${item.left} · 右 ${item.resized_width-item.right} px`,`Trim: top ${item.top} · bottom ${item.resized_height-item.bottom} · left ${item.left} · right ${item.resized_width-item.right} px`)}</small>}<small>{text('训练画布','Training canvas')} {item.width} × {item.height} · {item.cropped_pixels ? text(`裁切 ${item.cropped_pixels} 像素`,`Crops ${item.cropped_pixels} pixels`) : text('无裁切','No cropping')}</small></div>)}{plan.image_fit.truncated && <p>{text('显示前 100 种图片与尺寸组合。','Showing the first 100 image-size pairs.')}</p>}</div></details>}
      </div>}
      <div className="estimate-block"><h4>{text('执行估算', 'Execution estimate')}</h4><dl>
        {plan?.distributed && plan.distributed.world_size > 1 && <>
          <div><dt>{text('训练显卡', 'Training GPUs')}</dt><dd>{plan.distributed.world_size}</dd></div>
          <div><dt>{text('每卡批量', 'Batch per GPU')}</dt><dd>{plan.distributed.per_device_batch_size}</dd></div>
          <div><dt>{text('有效批量上限', 'Maximum effective batch')}</dt><dd>{plan.distributed.effective_batch_size}</dd></div>
          {plan.distributed.dropped_samples > 0 && <div><dt>{text('首轮末尾略过', 'First epoch tail skipped')}</dt><dd>{plan.distributed.dropped_samples} {text('张', 'images')}</dd></div>}
        </>}
        {native && <><div><dt>{text('缩小的图片', 'Downscaled images')}</dt><dd>{native.downscaled}</dd></div><div><dt>{text('批次 / 轮', 'Batches / epoch')}</dt><dd>{native.logical_batches}</dd></div><div><dt>{text('计算次数 / 轮', 'Model runs / epoch')}</dt><dd>{native.forward_groups ?? '—'}</dd></div></>}
        <div><dt>{text('每轮步数', 'Steps / epoch')}</dt><dd>{plan?.steps_per_epoch ?? '—'}</dd></div>
        <div><dt>{text('总训练步数', 'Total steps')}</dt><dd>{plan?.total_steps ?? '—'}</dd></div>
        <div><dt>{text('可训练参数', 'Trainable parameters')}</dt><dd>{formatParams(plan?.params?.trainable)}</dd></div>
        <div className={overCapacity ? 'estimate-over' : tightMemory ? 'estimate-tight' : undefined}><dt>{plan?.distributed && plan.distributed.world_size > 1 ? text('每卡显存峰值估算', 'Estimated peak per GPU') : text('显存峰值估算', 'Estimated peak memory')}</dt><dd>{formatBytesMB(peak)}{capacity ? <small> / {formatBytesMB(capacity)}</small> : null}</dd></div>
      </dl>
      {overCapacity && capacity && peak != null && <div className="estimate-alert" role="alert">
        <strong>{memoryBlocked ? text('超过显卡容量，无法开始训练', 'Exceeds GPU memory; training cannot start') : text('超过显卡容量', 'Exceeds GPU memory')}</strong>
        <p>{memoryBlocked
          ? text(`预计峰值比本机最大显卡的 ${formatBytesMB(capacity)} 多 ${formatBytesMB(peak - capacity * 0.95)}（保留 5% 余量）。减少以下任一项的显存占用后即可开始：`, `The estimate needs ${formatBytesMB(peak - capacity * 0.95)} more than this machine's largest GPU (${formatBytesMB(capacity)}, keeping 5% headroom). Reduce memory with any of these to start:`)
          : text('启动前显存检查已在任务队列的调度设置中关闭，训练可能因显存不足而失败。', 'The pre-launch memory check is off in the queue settings, so training may fail for lack of memory.')}</p>
        <ul>{memoryFixes(plan?.memory?.suggestions || [], nativeMode, text).map(fix => <li key={fix.path}>{onField ? <button type="button" className="ui-link" onClick={() => onField(fix.path)}>{fix.label}</button> : fix.label}</li>)}</ul>
      </div>}
      {tightMemory && <p className="estimate-note">{text('显存余量不足 10%，训练中可能因显存不足失败。', 'Less than 10% memory headroom; training may run out of memory.')}</p>}
      </div>
    </div>
  </section>;
}
