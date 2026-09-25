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
const bucketKey = (bucket: PlanBucket) => `${bucket.base ?? 0}:${bucket.w}x${bucket.h}`;
/** Buckets from different base resolutions share one grid only when there is a single base. */
function groupByBase(buckets: PlanBucket[]) {
  const groups = new Map<number, PlanBucket[]>();
  for (const bucket of buckets) groups.set(bucket.base ?? 0, [...(groups.get(bucket.base ?? 0) || []), bucket]);
  return [...groups].sort(([a], [b]) => a - b).map(([base, items]) => ({ base, buckets: items, items: items.reduce((sum, bucket) => sum + bucket.items, 0) }));
}
export default function BucketInspector({ plan, loading, onData, hasSources = false, indexed, onIssues, error, onRetry, dataset }: { plan: Plan | null; loading: boolean; onData: () => void; hasSources?: boolean; indexed?: { images: number; captioned: number }; onIssues?: () => void; error?:string; onRetry?:()=>void; dataset?: DatasetSizing }) {
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
  const tile = (bucket: PlanBucket) => {
    const key = bucketKey(bucket);
    const longest = Math.max(bucket.w, bucket.h);
    return <button type="button" key={key} className={`bucket-tile ${key === selected ? 'is-selected' : ''}`} aria-pressed={key === selected} aria-label={`${grouped ? `${baseLabel(bucket.base)} · ` : ''}${bucket.w} × ${bucket.h}, ${bucket.items} ${text('样本', 'samples')}`} onClick={() => setSelected(key === selected ? null : key)}>
      <span className="bucket-shape-space"><span className="bucket-shape" style={{width: `${bucket.w / longest * 56}px`, height: `${bucket.h / longest * 56}px`}}><span>{bucket.items}</span></span></span>
      <span className="bucket-size">{bucket.w} × {bucket.h}</span>
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
        {chosen && <div className="bucket-selection"><strong>{grouped ? `${baseLabel(chosen.base)} · ` : ''}{chosen.w} × {chosen.h}</strong><span>{chosen.items} {text('样本', 'samples')} · {chosen.batches ?? '—'} {native ? text('次计算 / 轮', 'model runs / epoch') : text('批次 / 轮', 'batches / epoch')}</span><span>{text('长宽比', 'Aspect ratio')} {(chosen.w / chosen.h).toFixed(2)}</span></div>}
      </>}
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
        <div><dt>{plan?.distributed && plan.distributed.world_size > 1 ? text('每卡显存峰值估算', 'Estimated peak per GPU') : text('显存峰值估算', 'Estimated peak memory')}</dt><dd>{formatBytesMB(plan?.memory?.peak_mb_estimate)}</dd></div>
      </dl></div>
    </div>
  </section>;
}
