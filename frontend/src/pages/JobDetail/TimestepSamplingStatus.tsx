import type { TimestepSamplingStatus as SamplingStatus } from '../../api/types';
import ConfigHelp from '../../components/ConfigHelp';
import { useWorkspaceText } from '../../utils/workspaceText';

const number = (value: number | null | undefined) => typeof value === 'number' && Number.isFinite(value) ? String(Number(value.toFixed(4))) : '—';
const percent = (value: number | null | undefined) => typeof value === 'number' && Number.isFinite(value) ? `${Number((value * 100).toFixed(2))}%` : '—';

export default function TimestepSamplingStatus({ status, configured, jobStatus }: {
  status?: SamplingStatus | null;
  configured: boolean;
  jobStatus: string;
}) {
  const text = useWorkspaceText();
  if (!status && !configured) return null;
  const label = text('时间步采样', 'Timestep sampling');
  if (!status) return <div className="job-timestep-status" role="group" aria-label={label}>
    <strong>{label}</strong><span>{text('尚无运行记录', 'No runtime record yet')}</span>
  </div>;

  const adaptive = status.mode === 'infonoise' || status.mode === 'combined';
  const resolution = status.mode === 'resolution_sqrt' || status.mode === 'combined';
  const name = status.mode === 'combined' ? text('InfoNoise + 按分辨率修正 Shift', 'InfoNoise + resolution-corrected Shift')
    : status.mode === 'resolution_sqrt' ? text('按分辨率修正 Shift', 'Resolution-corrected Shift') : 'InfoNoise';
  const stages = {
    baseline: text('基础采样', 'Baseline sampling'),
    warmup: text(`热身 ${number(status.warmup_completed_steps)} / ${number(status.warmup_steps)} 步`, `Warmup ${number(status.warmup_completed_steps)} / ${number(status.warmup_steps)} steps`),
    collecting: text('收集统计', 'Collecting statistics'),
    active: text('自适应采样', 'Adaptive sampling'),
    fallback: text('回退到基础采样', 'Baseline fallback'),
  };
  const reasons: Record<string, string> = {
    warmup: text('正在收集热身统计', 'Collecting warmup statistics'),
    insufficient_samples: text('有效主集样本不足', 'Not enough valid main-dataset samples'),
    insufficient_coverage: text('时间区间覆盖不足', 'Not enough timestep-bin coverage'),
    active: text('使用已学习的时间步分布', 'Using the learned timestep distribution'),
    zero_information: text('尚无有效信息量', 'No usable information yet'),
    no_main_observations: text('尚无有效主集样本', 'No valid main-dataset samples yet'),
    nonfinite_statistics: text('统计出现非有限值', 'Non-finite statistics'),
  };
  const reason = status.reason ? reasons[status.reason] || text(`原因：${status.reason}`, `Reason: ${status.reason}`) : '';
  const ended = ['paused', 'completed', 'failed', 'cancelled'].includes(jobStatus);
  const stage = stages[status.stage] || text('状态未知', 'Unknown state');
  const recentBatch = status.token_scope === 'last_primary_microbatch';
  const baselineNames = {
    uniform: text('均匀分布', 'Uniform'), logit_normal: 'Logit-normal',
    shift: text('固定 Shift', 'Fixed Shift'), resolution_sqrt: text('按分辨率修正 Shift', 'Resolution-corrected Shift'),
  };
  const help = [
    text(`第 ${number(status.step)} 步的采样记录。基础分布：${baselineNames[status.baseline] || status.baseline}。`, `Sampling record at step ${number(status.step)}. Baseline: ${baselineNames[status.baseline] || status.baseline}.`),
    reason,
    ...(adaptive ? [
      text(`学习分布 ${percent(status.adaptive_fraction)}，基础分布 ${percent(status.baseline_fraction)}，均匀探索 ${percent(status.exploration_fraction)}。比例随实际统计更新。`, `Learned ${percent(status.adaptive_fraction)}, baseline ${percent(status.baseline_fraction)}, uniform exploration ${percent(status.exploration_fraction)}. These shares follow the recorded statistics.`),
      text(`统计截至第 ${number(status.statistics_step)} 步：有效主集样本 ${number(status.main_samples)}，启用所需样本 ${number(status.min_samples)}；覆盖率 ${percent(status.coverage)}，启用所需覆盖率 ${percent(status.min_coverage)}。`, `Statistics through step ${number(status.statistics_step)}: ${number(status.main_samples)} valid main-dataset samples, ${number(status.min_samples)} required; coverage ${percent(status.coverage)}, ${percent(status.min_coverage)} required.`),
      text('覆盖率按出现过有效主集样本的时间区间计算。正则样本、空遮罩和无效样本不计入主集样本统计。', 'Coverage counts timestep bins with valid main-dataset samples. Regularization samples, empty masks, and invalid samples do not count toward main-dataset statistics.'),
    ] : []),
    ...(resolution ? [recentBatch
      ? text('tokens 和 Shift 对应最近一个实际微批次；分布式训练取主进程最近微批次。tokens 按画布潜空间计算，包含位置填充，不是遮罩的有效面积。恢复后将在首个批次取得尺寸。', 'Tokens and Shift describe the most recent actual microbatch; distributed training uses the primary process’s most recent microbatch. Tokens count the latent canvas, including positional padding, rather than the valid mask area. After a restore, the first batch supplies the size.')
      : text('此记录的批次范围未知。', 'The batch scope of this record is unknown.')] : []),
  ].filter(Boolean).join('\n\n');

  return <div className="job-timestep-status" role="group" aria-label={label} data-stage={status.stage}>
    <span className="job-timestep-heading"><strong>{name}</strong><ConfigHelp label={text('时间步采样 · 说明', 'About timestep sampling')}>{help}</ConfigHelp></span>
    <span role="status">{ended ? text(`最后记录：${stage}`, `Last recorded: ${stage}`) : stage}</span>
    {reason && !['warmup', 'active'].includes(status.reason || '') && <span className="job-timestep-reason">{reason}</span>}
    {adaptive && <>
      <span>{text('覆盖', 'Coverage')} {number(status.observed_bins)} / {number(status.bins)} · {percent(status.coverage)}</span>
      <span>{text('学习分布', 'Learned')} {percent(status.adaptive_fraction)} · {text('基础分布', 'Baseline')} {percent(status.baseline_fraction)} · {text('均匀探索', 'Exploration')} {percent(status.exploration_fraction)}</span>
      <span>{text(`统计至第 ${number(status.statistics_step)} 步`, `Statistics through step ${number(status.statistics_step)}`)}</span>
    </>}
    {resolution && <span>{recentBatch ? text('最近批次', 'Latest batch') : text('批次范围未知', 'Unknown batch scope')} {number(status.recent_num_tokens)} tokens · Shift {number(status.effective_baseline_shift)}</span>}
  </div>;
}
