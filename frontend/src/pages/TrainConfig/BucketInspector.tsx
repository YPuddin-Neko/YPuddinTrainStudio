import { useState } from 'react';
import { BarChart3, Grid2X2, Database, Loader2 } from 'lucide-react';
import type { Plan } from '../../api/types';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatBytesMB, formatParams } from '../../utils/format';

export default function BucketInspector({ plan, loading, onData, hasSources = false, indexed, onIssues }: { plan: Plan | null; loading: boolean; onData: () => void; hasSources?: boolean; indexed?: { images: number; captioned: number }; onIssues?: () => void }) {
  const text = useWorkspaceText();
  const [view, setView] = useState<'shape' | 'table'>('shape');
  const [selected, setSelected] = useState<string | null>(null);
  const buckets = plan?.buckets || [];
  const native = plan?.native;
  const awaitingPlan = hasSources && !buckets.length && !plan?.ok;
  const maxCount = Math.max(1, ...buckets.map(bucket => bucket.items));
  const chosen = buckets.find(bucket => `${bucket.w}x${bucket.h}` === selected);
  return <section className="bucket-inspector" aria-label={native ? text('原生尺寸与训练估算', 'Native sizes and training estimates') : text('数据分桶与训练估算', 'Buckets and training estimates')}>
    <div className="inspector-heading"><h3><BarChart3 size={15} />{text('数据分布', 'Dataset distribution')}</h3>{loading && <Loader2 size={14} className="animate-spin" aria-label={text('正在更新', 'Updating')} />}</div>
    <div className={`inspector-content ${loading ? 'opacity-60' : ''}`} aria-busy={loading}>
      <dl className="dataset-metrics">
        <div><dt>{awaitingPlan && indexed ? text('已索引图片', 'Indexed images') : text('原始图片', 'Images')}</dt><dd>{awaitingPlan ? indexed?.images ?? '—' : plan?.images ?? '—'}</dd></div>
        <div><dt>{text('重复后样本', 'Repeated samples')}</dt><dd>{awaitingPlan ? '—' : plan?.items ?? '—'}</dd></div>
        <div><dt>{text('已配标签', 'Captioned')}</dt><dd>{awaitingPlan ? indexed?.captioned ?? '—' : plan?.captioned ?? '—'}</dd></div>
        <div><dt>{native ? text('独立尺寸', 'Distinct sizes') : text('分桶数量', 'Buckets')}</dt><dd>{awaitingPlan ? '—' : plan ? buckets.length : '—'}</dd></div>
      </dl>
      <div className="bucket-heading"><h4>{native ? text('原生尺寸分布', 'Native size distribution') : text('分桶布局', 'Bucket layout')}</h4><div className="segmented-small"><button type="button" aria-label={text('分桶图形视图', 'Bucket shape view')} aria-pressed={view === 'shape'} onClick={() => setView('shape')}><Grid2X2 size={13} /></button><button type="button" aria-label={text('分桶明细表', 'Bucket table')} aria-pressed={view === 'table'} onClick={() => setView('table')}><BarChart3 size={13} /></button></div></div>
      {buckets.length === 0 ? <div className="bucket-empty"><Database size={23} /><p>{loading ? text('正在计算实际分桶…', 'Computing buckets…') : awaitingPlan ? text('已配置数据来源。完成待配置项后计算分桶与训练估算。', 'Dataset sources are configured. Complete the pending settings to calculate buckets and training estimates.') : text('添加训练图片后，显示实际分桶尺寸与样本分布。', 'Add training images to inspect their bucket dimensions and distribution.')}</p><button type="button" className="studio-link" onClick={awaitingPlan && onIssues ? onIssues : onData}>{awaitingPlan && onIssues ? text('检查待配置项', 'Review pending settings') : text('配置训练数据', 'Configure dataset')}</button></div> : <>
        {view === 'shape' ? <div className="bucket-grid" data-testid="plan-buckets">{buckets.map(bucket => {
          const key = `${bucket.w}x${bucket.h}`;
          const longest = Math.max(bucket.w, bucket.h);
          return <button type="button" key={key} className={`bucket-tile ${key === selected ? 'is-selected' : ''}`} aria-pressed={key === selected} aria-label={`${bucket.w} × ${bucket.h}, ${bucket.items} ${text('样本', 'samples')}`} onClick={() => setSelected(key === selected ? null : key)}>
            <span className="bucket-shape-space"><span className="bucket-shape" style={{width: `${bucket.w / longest * 56}px`, height: `${bucket.h / longest * 56}px`}}><span>{bucket.items}</span></span></span>
            <span className="bucket-size">{bucket.w} × {bucket.h}</span>
            <span className="bucket-count-track"><span style={{width: `${bucket.items / maxCount * 100}%`}} /></span>
          </button>;
        })}</div> : <div className="bucket-table-wrap" data-testid="plan-buckets"><table className="bucket-table"><thead><tr><th>{text('尺寸', 'Size')}</th><th>{text('样本', 'Items')}</th><th>{native ? text('前向次数', 'Forwards') : text('批次', 'Batches')}</th></tr></thead><tbody>{buckets.map(bucket => <tr key={`${bucket.w}x${bucket.h}`}><td>{bucket.w} × {bucket.h}</td><td>{bucket.items}</td><td>{bucket.batches}</td></tr>)}</tbody></table></div>}
        {chosen && <div className="bucket-selection"><strong>{chosen.w} × {chosen.h}</strong><span>{chosen.items} {text('样本', 'samples')} · {chosen.batches} {native ? text('前向 / 轮', 'forwards / epoch') : text('批次 / 轮', 'batches / epoch')}</span><span>{text('长宽比', 'Aspect ratio')} {(chosen.w / chosen.h).toFixed(2)}</span></div>}
        <p className="inspector-note">{native ? text('保留每图独立尺寸，不放大小图；超出预算才按策略处理。同尺寸图片分组前向，梯度按图片数等权累积。', 'Retains individual sizes without upscaling; only over-budget images follow the overflow policy. Same-size images run together, with equal per-image gradient weights.') : text('矩形按实际长宽比展示，横条表示样本数量。调整分辨率和分桶参数后自动更新。', 'Shapes represent aspect ratios; bars represent item counts. Updates with resolution and bucket settings.')}</p>
      </>}
      <div className="estimate-block"><h4>{text('执行估算', 'Execution estimate')}</h4><dl>
        {native && <><div><dt>{text('缩小的图片', 'Downscaled images')}</dt><dd>{native.downscaled}</dd></div><div><dt>{text('逻辑批次 / 轮', 'Logical batches / epoch')}</dt><dd>{native.logical_batches}</dd></div><div><dt>{text('分组前向 / 轮', 'Grouped forwards / epoch')}</dt><dd>{native.forward_groups}</dd></div></>}
        <div><dt>{text('每轮步数', 'Steps / epoch')}</dt><dd>{plan?.steps_per_epoch ?? '—'}</dd></div>
        <div><dt>{text('总训练步数', 'Total steps')}</dt><dd>{plan?.total_steps ?? '—'}</dd></div>
        <div><dt>{text('可训练参数', 'Trainable parameters')}</dt><dd>{formatParams(plan?.params?.trainable)}</dd></div>
        <div><dt>{text('显存峰值估算', 'Estimated peak memory')}</dt><dd>{formatBytesMB(plan?.memory?.peak_mb_estimate)}</dd></div>
      </dl><p className="inspector-note">{text('显存是规划估算；实际占用以训练时的设备监控为准。', 'Memory is a planning estimate; observe actual device use during training.')}</p></div>
    </div>
  </section>;
}
