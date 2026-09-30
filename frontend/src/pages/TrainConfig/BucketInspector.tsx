import { useId, useState } from 'react';
import { ArrowDown, BarChart3, Grid2X2, Database, Loader2 } from 'lucide-react';
import type { Plan } from '../../api/types';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatBytesMB, formatParams } from '../../utils/format';
import SourceBalance from './SourceBalance';
import {configOptionLabel, OPAQUE_CONFIG_ISSUE, presentConfigIssues} from '../../utils/configPresentation';
import ConfigHelp from '../../components/ConfigHelp';
import { SlidingIndicator } from '../../components/motion';
import BucketGeometry from './BucketGeometry';

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
/** Memory fixes the planner can name, each linked to the setting it changes and, where known, the peak it leaves. */
function memoryFixes(memory: Plan['memory'], native: boolean, text: (zh: string, en: string) => string) {
  const suggestions = memory?.suggestions || [];
  const estimate = (mode: string) => {
    const peak = memory?.checkpointing_peak_mb_estimates?.[mode];
    return peak == null ? '' : text(`（预计 ${formatBytesMB(peak)}）`, ` (about ${formatBytesMB(peak)})`);
  };
  const known: [RegExp, string, string][] = [
    [/activation_checkpointing = 'block'/, 'memory.activation_checkpointing', text('开启梯度检查点', 'Turn on gradient checkpointing') + estimate('block')],
    [/activation_checkpointing = 'unsloth'/, 'memory.activation_checkpointing', text('梯度检查点改为“开启并卸载到内存”', 'Offload gradient checkpoints to system memory') + estimate('unsloth')],
    [/blocks_to_swap/, 'memory.blocks_to_swap', text('把部分模型块换出到内存', 'Swap model blocks to system memory')],
    [/adamw8bit/, 'optimizer.type', text('改用 AdamW 8-bit 优化器', 'Use the AdamW 8-bit optimizer')],
    [/text_encoding/, 'dataset.text_encoding', text('训练前缓存文本特征', 'Cache text features before training')],
    [/Adafactor/, 'optimizer.type', text('改用 Adafactor 并不设置 beta1', 'Use Adafactor without beta1')],
  ];
  const fixes = known.filter(([pattern]) => suggestions.some(item => pattern.test(item))).map(([, path, label]) => ({ path, label }));
  fixes.push({ path: 'dataset.batch_size', label: text('减小批大小', 'Reduce the batch size') });
  fixes.push(native ? { path: 'dataset.native_max_pixels', label: text('降低图像面积上限', 'Lower the image area limit') } : { path: 'dataset.resolutions', label: text('降低训练分辨率', 'Lower the training resolution') });
  return fixes.filter((fix, index) => fixes.findIndex(item => item.path === fix.path) === index);
}

export default function BucketInspector({ plan, loading, onData, hasSources = false, indexed, onIssues, onField, error, onRetry, dataset }: { plan: Plan | null; loading: boolean; onData: () => void; hasSources?: boolean; indexed?: { images: number; captioned: number }; onIssues?: () => void; onField?: (path: string) => void; error?:string; onRetry?:()=>void; dataset?: DatasetSizing }) {
  const text = useWorkspaceText();
  const routeId = useId();
  const [view, setView] = useState<'shape' | 'table'>('shape');
  const [selected, setSelected] = useState<string | null>(null);
  const buckets = plan?.buckets || [];
  const native = plan?.native;
  const nativeMode = dataset ? dataset.resolution_mode === 'native' : !!native;
  const awaitingPlan = hasSources && !buckets.length && !plan?.ok;
  const maxCount = Math.max(1, ...buckets.map(bucket => bucket.items));
  const groups = nativeMode ? [{ base: 0, buckets, items: 0 }] : groupByBase(buckets);
  const grouped = groups.length > 1;
  // Native tiles carry the source size and an arrow, so they get wider columns.
  const gridClass = `bucket-grid${nativeMode && buckets.some(bucket => bucket.sources?.length) ? ' has-routes' : ''}`;
  const baseLabel = (base: number) => text(`分辨率 ${base}`, `Resolution ${base}`);
  const fitMode = plan?.image_fit?.mode === 'pad' ? 'pad' : 'crop';
  const cropAnchor = plan?.image_fit?.crop_anchor || 'center';
  const fitName = fitMode === 'pad' ? text('补边', 'pad') : text('裁切', 'crop');
  const resized = (route: SourceRoute) => route.resized_width !== route.width || route.resized_height !== route.height;
  const trimmed = (bucket: PlanBucket, route: SourceRoute) => route.resized_width !== bucket.w || route.resized_height !== bucket.h;
  // Source → resize → crop or padding → training size, in the order the loader applies them.
  const routeSteps = (bucket: PlanBucket, route: SourceRoute) => {
    const steps = [text(`原图 ${route.width} × ${route.height}`, `Source ${route.width} × ${route.height}`)];
    if (resized(route)) {
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
  // A native tile reads source size → what was done → training size.
  const tileRoute = (bucket: PlanBucket) => {
    const routes = bucket.sources || [];
    if (!nativeMode || !routes.length) return null;
    const variants = Math.max(bucket.source_variants || 0, routes.length);
    const scaled = routes.some(resized), fitted = routes.some(route => trimmed(bucket, route));
    return {
      source: variants > 1 ? text(`${variants} 种原图尺寸`, `${variants} source sizes`) : `${routes[0].width} × ${routes[0].height}`,
      step: scaled && fitted ? text(`缩放后${fitName}`, `resize + ${fitName}`) : scaled ? text('缩放', 'resize') : fitted ? fitName : text('不变', 'unchanged'),
    };
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
  // Loss masks: sidecar files first, then transparent pixels; both only count with masked loss on.
  const masks = awaitingPlan ? null : plan?.masks;
  const masked = masks ? masks.files + masks.alpha : 0;
  const maskSources = masks ? [masks.files && text(`${masks.files} 张遮罩文件`, `${masks.files} mask files`), masks.alpha && text(`${masks.alpha} 张含透明像素的图片`, `${masks.alpha} images with transparent pixels`)].filter(Boolean).join(text('、', ', ')) : '';
  const maskLine = !masks || (!masked && !masks.enabled) ? ''
    : !masks.enabled ? text(`找到 ${maskSources}；遮罩加权训练未开启，训练时不会使用。`, `Found ${maskSources}; masked loss is off, so they are not used.`)
      : !masked ? text('遮罩加权训练已开启，但没有找到遮罩文件或含透明像素的图片，所有图片按整图计算。', 'Masked loss is on, but no mask files or images with transparent pixels were found; every image counts in full.')
        : text(`按遮罩计算损失：${maskSources}；其余 ${Math.max(0, (plan?.images ?? masked) - masked)} 张按整图计算。`, `Loss follows ${maskSources}; the other ${Math.max(0, (plan?.images ?? masked) - masked)} images count in full.`);
  const distributed = plan?.distributed;
  const gpus = distributed?.world_size ?? 1;
  const perBatch = distributed?.per_device_batch_size ?? null;
  const effectiveBatch = distributed?.effective_batch_size ?? null;
  const accumulation = effectiveBatch && perBatch ? Math.max(1, Math.round(effectiveBatch / (perBatch * gpus))) : null;
  const batchFormula = perBatch == null || accumulation == null ? ''
    : gpus > 1 ? text(`${perBatch}\u00a0张 × ${gpus}\u00a0卡 × 累积\u00a0${accumulation}`, `${perBatch} × ${gpus}\u00a0GPUs × ${accumulation}\u00a0accumulation`)
      : accumulation > 1 ? text(`${perBatch}\u00a0张 × 累积\u00a0${accumulation}`, `${perBatch} × ${accumulation}\u00a0accumulation`) : text(`每批 ${perBatch}\u00a0张`, `${perBatch} per batch`);
  const batchesPerEpoch = native ? native.logical_batches : buckets.reduce((sum, bucket) => sum + (bucket.batches ?? 0), 0);
  const stepSource = !plan?.steps_per_epoch ? '' : gpus > 1 && distributed ? text(`每卡 ${distributed.batches_per_rank} 批`, `${distributed.batches_per_rank} batches per GPU`) : batchesPerEpoch ? text(`${batchesPerEpoch} 批`, `${batchesPerEpoch} batches`) : '';
  const totalSource = !plan?.total_steps || !plan.steps_per_epoch ? '' : plan.epochs && plan.total_steps === plan.epochs * plan.steps_per_epoch ? text(`${plan.epochs} 轮`, `${plan.epochs} epochs`) : text('按最大步数', 'max steps');
  const peak = plan?.memory?.peak_mb_estimate ?? null;
  const capacity = plan?.memory?.gpu_total_mb ?? null;
  const memoryIssue = peak == null && plan?.memory?.unavailable_issue
    ? presentConfigIssues([plan.memory.unavailable_issue], text('zh', 'en') === 'en')[0] : null;
  const memoryIssueDetail = memoryIssue?.message === OPAQUE_CONFIG_ISSUE ? memoryIssue.detail : memoryIssue?.message;
  const overCapacity = peak != null && !!capacity && peak > capacity * 0.95;
  const tightMemory = !overCapacity && peak != null && !!capacity && peak > capacity * 0.9;
  const memoryBlocked = overCapacity && !!plan?.errors?.some(item => item.loc === 'memory');
  const tile = (bucket: PlanBucket) => {
    const key = bucketKey(bucket);
    const route = tileRoute(bucket);
    const source = bucket.sources?.[0];
    const variants = Math.max(bucket.source_variants || 0, bucket.sources?.length || 0);
    const geometryTitle = source ? routeSteps(bucket, source).join(' → ') + (variants > 1 ? text(`；图示为 ${source.images} 张图片的尺寸`, `; shows the size of ${source.images} images`) : '') : undefined;
    return <button type="button" key={key} className={`bucket-tile${key === selected ? ' is-selected' : ''}${route ? ' has-route' : ''}`} aria-pressed={key === selected} aria-label={`${grouped ? `${baseLabel(bucket.base)} · ` : ''}${bucket.w} × ${bucket.h}, ${bucket.items} ${text('样本', 'samples')}`} aria-describedby={route ? `${routeId}-${key}` : undefined} onClick={() => setSelected(key === selected ? null : key)}>
      <span className="bucket-shape-space" title={geometryTitle}><BucketGeometry size={bucket} route={source} fit={fitMode} anchor={cropAnchor} count={bucket.items}/></span>
      {route && <span className="bucket-route" aria-hidden="true">
        <span className="bucket-route-source" title={route.source}>{route.source}</span>
        <span className="bucket-route-step">{route.step}</span>
        <ArrowDown size={11} className="bucket-route-arrow"/>
      </span>}
      <span className="bucket-size">{bucket.w} × {bucket.h}</span>
      <span className="bucket-count-track"><span style={{width: `${bucket.items / maxCount * 100}%`}} /></span>
      {route && <span id={`${routeId}-${key}`} className="sr-only">{text(`原图 ${route.source}，${route.step}`, `Source ${route.source}, ${route.step}`)}</span>}
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
      {maskLine && <p className="dataset-masks"><strong>{text('遮罩', 'Masks')}</strong><span>{maskLine}</span>{masks && !masks.enabled && onField && <button type="button" className="ui-link" onClick={() => onField('dataset.masked_loss')}>{text('开启遮罩加权训练', 'Turn on masked loss')}</button>}</p>}
      <SourceBalance sources={plan?.source_balance} loading={loading} hasSources={hasSources}/>
      {!!native?.synchronization_groups && <p className="inspector-note">{text(`多卡每轮额外 ${native.synchronization_groups} 次补齐计算，不计入损失，也不增加训练样本。`, `Multi-GPU training adds ${native.synchronization_groups} padding runs per epoch; they carry no loss and add no samples.`)}</p>}
      <div className="bucket-heading"><h4>{nativeMode ? text('实际训练尺寸', 'Training sizes') : text('分桶布局', 'Bucket layout')}</h4><div className="ui-segmented ui-segmented-sm" role="group" aria-label={text('分桶显示方式', 'Bucket view')}><button type="button" aria-label={text('分桶图形视图', 'Bucket shape view')} aria-pressed={view === 'shape'} onClick={() => setView('shape')}><Grid2X2 size={13} /></button><button type="button" aria-label={text('分桶明细表', 'Bucket table')} aria-pressed={view === 'table'} onClick={() => setView('table')}><BarChart3 size={13} /></button><SlidingIndicator className="ui-segmented-thumb"/></div></div>
      {buckets.length === 0 ? <div className="bucket-empty"><Database size={23} /><p>{loading ? text('正在计算实际分桶…', 'Computing buckets…') : awaitingPlan ? text('请完成待配置项后计算。', 'Complete the pending settings to calculate.') : text('尚无训练图片。', 'No training images yet.')}</p><button type="button" className="ui-link" onClick={awaitingPlan && onIssues ? onIssues : onData}>{awaitingPlan && onIssues ? text('检查待配置项', 'Review pending settings') : text('配置训练数据', 'Configure dataset')}</button></div> : <>
        {view === 'shape' ? <div className="bucket-groups" data-testid="plan-buckets">{groups.map(group => grouped
          ? <section key={group.base} className="bucket-group" aria-label={baseLabel(group.base)}>
            <h5 className="bucket-group-heading"><span>{baseLabel(group.base)}</span><small>{text(`${group.buckets.length} 个分桶 · ${group.items} 样本`, `${group.buckets.length} buckets · ${group.items} samples`)}</small></h5>
            <div className={gridClass}>{group.buckets.map(tile)}</div>
          </section>
          : <div key={group.base} className={gridClass}>{group.buckets.map(tile)}</div>)}</div>
          : <div className="bucket-table-wrap" data-testid="plan-buckets"><table className="bucket-table"><thead><tr>{grouped && <th>{text('分辨率', 'Resolution')}</th>}<th>{text('尺寸', 'Size')}</th><th>{text('样本', 'Items')}</th><th>{native ? <span className="bucket-column-help">{text('计算次数', 'Model runs')}<ConfigHelp label={text('计算次数说明', 'Model runs help')}>{text('每轮对这个尺寸运行模型的次数。同尺寸图片在不超过图像面积上限时合并为一次计算；单张已接近上限时逐张计算，因此常与样本数相同。', 'How many times the model runs on this size per epoch. Same-size images are combined while they fit the image area limit; images near the limit run one at a time, so this often equals the sample count.')}</ConfigHelp></span> : text('批次', 'Batches')}</th></tr></thead><tbody>{buckets.map(bucket => <tr key={bucketKey(bucket)}>{grouped && <td>{bucket.base}</td>}<td>{bucket.w} × {bucket.h}</td><td>{bucket.items}</td><td>{bucket.batches ?? '—'}</td></tr>)}</tbody></table></div>}
      </>}
      {plan?.image_fit && <div className="image-fit-summary">
        <div className="image-fit-head">
          <strong>{plan.image_fit.mode==='pad'?text('完整画面保留','Whole image preserved'):text('裁切保留位置','Crop anchor') + ' · ' + configOptionLabel('dataset.crop_anchor',plan.image_fit.crop_anchor || 'center',text('zh','en') === 'en')}</strong>
          <span>{plan.image_fit.mode==='pad' ? text(`填充占比 ${(plan.image_fit.padding_fraction*100).toFixed(2)}%`, `Padding ${(plan.image_fit.padding_fraction*100).toFixed(2)}%`) : text(`裁切 ${plan.image_fit.cropped_images} 张`, `${plan.image_fit.cropped_images} cropped`)}</span>
        </div>
        {buckets.length > 0 && <p>{sizeSummary}</p>}
        {!!plan.image_fit.items.length && <details><summary>{text('查看尺寸适配明细','Inspect image sizing')}</summary><div className="image-fit-details">{plan.image_fit.items.map((item,index)=><div key={`${item.path}/${item.width}/${item.height}/${index}`}><strong title={item.path}>{item.path.replace(/\\/g,'/').split('/').pop()}</strong><span>{item.source_width} × {item.source_height} → {item.resized_width} × {item.resized_height}</span>{plan.image_fit?.mode==='crop' && item.cropped_pixels>0 && <small>{text(`裁去：上 ${item.top} · 下 ${item.resized_height-item.bottom} · 左 ${item.left} · 右 ${item.resized_width-item.right} px`,`Trim: top ${item.top} · bottom ${item.resized_height-item.bottom} · left ${item.left} · right ${item.resized_width-item.right} px`)}</small>}<small>{text('训练画布','Training canvas')} {item.width} × {item.height} · {item.cropped_pixels ? text(`裁切 ${item.cropped_pixels} 像素`,`Crops ${item.cropped_pixels} pixels`) : text('无裁切','No cropping')}</small></div>)}{plan.image_fit.truncated && <p>{text('显示前 100 种图片与尺寸组合。','Showing the first 100 image-size pairs.')}</p>}</div></details>}
      </div>}
      <div className="estimate-block"><h4>{text('执行估算', 'Execution estimate')}</h4>
        <dl className="estimate-stats">
          <div><dt>{text('等效批次', 'Effective batch')}<ConfigHelp label={text('等效批次说明', 'Effective batch help')}>{text('每次更新参数所用的图片数 = 批大小 × 梯度累积 × 显卡数。原生分辨率下，同一批里尺寸不同的图片会分开计算，整批算完后才合并梯度更新参数，所以等效批次不会因此变小。', 'Images behind each parameter update = batch size × gradient accumulation × GPUs. In native resolution, differently sized images in a batch are computed separately, and their gradients are combined before the update, so the effective batch stays the same.')}</ConfigHelp></dt><dd>{effectiveBatch ? text(`${effectiveBatch} 张`, `${effectiveBatch}`) : '—'}</dd>{batchFormula && <small>{batchFormula}</small>}</div>
          <div><dt>{text('每轮步数', 'Steps / epoch')}</dt><dd>{plan?.steps_per_epoch ?? '—'}</dd>{stepSource && <small>{stepSource}</small>}</div>
          <div><dt>{text('总训练步数', 'Total steps')}</dt><dd>{plan?.total_steps ?? '—'}</dd>{totalSource && <small>{totalSource}</small>}</div>
        </dl>
        <dl className="estimate-list">
          {native && <div><dt>{text('每轮计算次数', 'Model runs / epoch')}<ConfigHelp label={text('每轮计算次数说明', 'Model runs help')}>{text('原生分辨率下，同一批里尺寸不同的图片要分开计算，所以计算次数可能多于批次数。这只影响速度；整批算完后才更新参数，等效批次不变。', 'In native resolution, differently sized images in a batch are computed separately, so there can be more runs than batches. This only affects speed; parameters update after the whole batch, so the effective batch is unchanged.')}</ConfigHelp></dt><dd>{native.forward_groups == null ? '—' : text(`${native.forward_groups} 次`, `${native.forward_groups}`)}</dd></div>}
          {gpus > 1 && <div><dt>{text('训练显卡', 'Training GPUs')}</dt><dd>{text(`${gpus} 张`, `${gpus}`)}</dd></div>}
          {!!distributed?.dropped_samples && gpus > 1 && <div><dt>{text('首轮末尾略过', 'First epoch tail skipped')}</dt><dd>{distributed.dropped_samples} {text('张', 'images')}</dd></div>}
          <div><dt>{text('可训练参数', 'Trainable parameters')}</dt><dd>{formatParams(plan?.params?.trainable)}</dd></div>
        </dl>
        <div className={`estimate-memory${overCapacity ? ' is-over' : tightMemory ? ' is-tight' : ''}`}>
          <div><span>{gpus > 1 ? text('每卡显存峰值估算', 'Estimated peak per GPU') : text('显存峰值估算', 'Estimated peak memory')}</span><strong>{memoryIssue
            ? <button type="button" className="ui-link" title={memoryIssueDetail} onClick={() => onField && memoryIssue.path ? onField(memoryIssue.path) : onIssues?.()}>{text(`检查${memoryIssue.label}`, `Check ${memoryIssue.label}`)}</button>
            : <>{formatBytesMB(peak)}{capacity ? <small> / {formatBytesMB(capacity)}</small> : null}</>}</strong></div>
          {capacity && peak != null ? <span className="estimate-meter" aria-hidden="true"><span style={{width: `${Math.min(100, peak / capacity * 100)}%`}}/></span> : null}
        </div>
        {overCapacity && capacity && peak != null && <div className="estimate-alert" role="alert">
          <strong>{memoryBlocked ? text('超过显卡容量，无法开始训练', 'Exceeds GPU memory; training cannot start') : text('超过显卡容量', 'Exceeds GPU memory')}</strong>
          <p>{memoryBlocked
            ? text(`预计峰值比单卡可用预算多 ${formatBytesMB(peak - capacity * 0.95)}（容量 ${formatBytesMB(capacity)}，保留 5% 余量）。可以这样减少显存占用：`, `The estimate exceeds the per-GPU budget by ${formatBytesMB(peak - capacity * 0.95)} (${formatBytesMB(capacity)} capacity, keeping 5% headroom). Reduce memory with:`)
            : text('启动前显存检查已在任务队列的调度设置中关闭，训练可能因显存不足而失败。', 'The pre-launch memory check is off in the queue settings, so training may fail for lack of memory.')}</p>
          <ul>{memoryFixes(plan?.memory, nativeMode, text).map(fix => <li key={fix.path}>{onField ? <button type="button" className="ui-link" onClick={() => onField(fix.path)}>{fix.label}</button> : fix.label}</li>)}</ul>
        </div>}
        {tightMemory && <p className="estimate-note">{text('显存余量不足 10%，训练中可能因显存不足失败。', 'Less than 10% memory headroom; training may run out of memory.')}</p>}
      </div>
    </div>
  </section>;
}
