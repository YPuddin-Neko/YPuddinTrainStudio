import React from 'react';
import { useTranslation } from 'react-i18next';
import { Activity, Thermometer } from 'lucide-react';
import { EChart } from '../../components/EChart';
import { SlidingIndicator } from '../../components/motion';
import type { JobMetrics } from '../../api/types';
import { shapeValidationSeries, smoothLoss } from '../../utils/metrics';
import { useWorkspaceText } from '../../utils/workspaceText';
import { metricChartBase, metricLabels } from './metricPresentation';
import './job-metrics.css';

const LR_COLORS = ['#a78bfa', '#f59e0b', '#22d3ee', '#fb7185', '#84cc16', '#e879f9'];
const GROUP_NAMES: Record<string, string> = { dora: 'DoRA' };

/**
 * The largest ratio between parameter groups at one step. Groups 100× apart (a DoRA or w2 group beside w1)
 * need a log axis to be seen at all; a schedule decaying over time does not.
 */
function groupSpread(groups: Array<Array<number | null>>): number {
  let spread = 1;
  const steps = Math.max(0, ...groups.map(values => values.length));
  for (let index = 0; index < steps; index += 1) {
    let low = Infinity, high = 0;
    for (const values of groups) {
      const value = values[index];
      if (typeof value === 'number' && value > 0 && Number.isFinite(value)) { low = Math.min(low, value); high = Math.max(high, value); }
    }
    if (high > 0) spread = Math.max(spread, high / low);
  }
  return spread;
}

type Chart = { key: string; title: string; note?: string; option?: Record<string, unknown>; empty?: React.ReactNode };

function line(name: string, data: Array<[number, number | null]>, color: string, extra: Record<string, unknown> = {}) {
  return { name, type: 'line', showSymbol: false, sampling: 'lttb', data, lineStyle: { width: 1.5, color }, itemStyle: { color }, ...extra };
}

/** Training curves and GPU readings, two charts per row. */
export default function JobMetricsPanel({ metrics, stepsPerEpoch, vramMetric }: {
  metrics: JobMetrics | null; stepsPerEpoch?: number | null; vramMetric?: string | null;
}) {
  const { t, i18n } = useTranslation();
  const text = useWorkspaceText();
  const chinese = (i18n.resolvedLanguage || i18n.language || '').startsWith('zh');
  const labels = React.useMemo(() => metricLabels(chinese), [chinese]);
  const [xAxisMode, setXAxisMode] = React.useState<'step' | 'epoch'>('step');
  const [emaAlpha, setEmaAlpha] = React.useState(0.9);
  const useEpoch = xAxisMode === 'epoch' && !!stepsPerEpoch;
  const xAxisName = useEpoch ? t('job.epoch') : t('job.step');
  const xs = React.useMemo(() => useEpoch && stepsPerEpoch ? metrics?.steps.map(step => step / stepsPerEpoch) || [] : metrics?.steps || [], [metrics, useEpoch, stepsPerEpoch]);
  const chartVramMetric = metrics?.vram_metric ?? vramMetric;

  const charts = React.useMemo<Chart[]>(() => {
    if (!metrics?.steps.length) return [];
    const points = (values: Array<number | null | undefined> | undefined, scale = 1) => xs.map((x, index): [number, number | null] => {
      const value = values?.[index];
      return [x, typeof value === 'number' ? value / scale : null];
    });
    const chart = (yName: string, series: unknown[]) => ({ ...metricChartBase(xAxisName, yName), series });
    const list: Chart[] = [
      {
        key: 'loss', title: labels.loss,
        note: text('每步 Loss 是每个优化步的训练损失；平滑曲线按上方 EMA 系数计算，只影响显示，不改变训练。', 'Loss per step is the training loss of each optimizer step; the smoothed curve uses the EMA coefficient above and only changes the chart.'),
        option: chart(labels.loss, [
          line(labels.raw, points(metrics.loss), '#93c5fd', { lineStyle: { width: 1.2, color: '#93c5fd' } }),
          line(labels.ema, smoothLoss(metrics.loss || [], emaAlpha).map((loss, index): [number, number | null] => [xs[index], loss]), '#2563eb', { lineStyle: { width: 2, color: '#2563eb' } }),
        ]),
      },
      (() => {
        const groups = Object.entries(metrics.lr || {});
        const logScale = groupSpread(groups.map(([, values]) => values)) >= 100;
        const option = chart(labels.lr, groups.map(([group, values], index) => line(`${labels.lr} · ${GROUP_NAMES[group] || group}`,
          // A log axis has no zero: warmup's first steps are left out of the line.
          points(logScale ? values.map(value => typeof value === 'number' && value > 0 ? value : null) : values), LR_COLORS[index % LR_COLORS.length])));
        const base = metricChartBase(xAxisName, labels.lr);
        return {
          key: 'lr', title: labels.lr,
          note: logScale
            ? text('每条线代表一个参数组。各组学习率相差 100 倍以上，纵轴用对数刻度，DoRA 这类很小的值也能看清。', 'Each line is one parameter group. The groups differ by 100× or more, so the axis is logarithmic and small rates such as DoRA stay visible.')
            : text('每条线代表一个参数组；LoKr 的 w1 / w2 可设置不同学习率。', 'Each line is one parameter group; LoKr w1 / w2 can use different learning rates.'),
          option: logScale ? { ...option, yAxis: { type: 'log', logBase: 10, name: base.yAxis.name, axisLabel: base.yAxis.axisLabel } } : option,
        };
      })(),
      {
        key: 'gradient', title: labels.gradient,
        note: text('每步梯度的大小，用于观察更新是否稳定。', 'Gradient magnitude per step, to inspect update stability.'),
        option: chart(labels.gradient, [line(labels.gradient, points(metrics.grad_norm), '#d97706')]),
      },
    ];
    if (metrics.validation?.length) {
      const { series } = shapeValidationSeries(metrics.validation);
      list.push({
        key: 'validation', title: t('job.validationTitle'),
        option: chart(labels.validation, series.map((item, index) => {
          const color = item.name === 'mean' ? '#f43f5e' : LR_COLORS[index % LR_COLORS.length];
          return {
            name: item.name === 'mean' ? labels.mean : `${labels.timestep} ${item.name}`, type: 'line', showSymbol: true,
            data: item.data.map(([step, loss]) => [useEpoch && stepsPerEpoch ? step / stepsPerEpoch : step, loss]),
            lineStyle: { width: item.name === 'mean' ? 2.5 : 1.2, color }, itemStyle: { color },
          };
        })),
      });
    }
    const memoryName = chartVramMetric === 'current_allocated' ? `${t('job.currentTrainingAllocated')} (GB)` : chartVramMetric === 'peak_allocated' ? `${t('job.vramPeak')} (GB)` : labels.memory;
    list.push(
      { key: 'speed', title: labels.speed, note: text('每秒完成的优化步数。', 'Optimizer steps completed per second.'), option: chart('it/s', [line(labels.speed, points(metrics.it_s), '#10b981')]) },
      {
        key: 'memory', title: memoryName,
        note: chartVramMetric === 'current_allocated' ? text('训练进程当前占用的显存。', 'Memory currently allocated by the training process.') : text('训练进程到这一步为止的显存峰值。', 'Peak memory allocated by the training process so far.'),
        option: chart('GB', [line(memoryName, points(metrics.vram_mb, 1024), '#ec4899')]),
      },
    );
    const gpu = [
      { key: 'power', title: labels.power, values: metrics.gpu_power_w, unit: 'W', color: '#8b5cf6', note: text('显卡驱动报告的整卡功耗。', 'Board power reported by the GPU driver.') },
      { key: 'temperature', title: labels.temperature, values: metrics.gpu_temp_c, unit: '°C', color: '#ef4444', note: text('显卡核心温度。', 'GPU core temperature.') },
      { key: 'utilization', title: labels.utilization, values: metrics.gpu_util_pct, unit: '%', color: '#0ea5e9', note: text('显卡计算单元的忙碌比例；持续偏低通常说明在等待数据或内存交换。', 'Share of time the GPU was busy; staying low usually means it waits for data or memory transfers.') },
    ].filter(item => item.values?.length);
    if (gpu.length) {
      for (const item of gpu) list.push({ key: item.key, title: item.title, note: item.note, option: chart(item.unit, [line(item.title, points(item.values), item.color)]) });
    } else {
      list.push({
        key: 'gpu', title: text('GPU 功率与温度', 'GPU power and temperature'),
        empty: text('此任务没有记录显卡驱动读数。NVIDIA 显卡训练时会记录功率、温度和利用率；更早的任务没有这些数据。', 'This job has no GPU driver readings. Training on an NVIDIA GPU records power, temperature and utilization; earlier jobs have none.'),
      });
    }
    return list;
  }, [metrics, xs, xAxisName, labels, emaAlpha, chartVramMetric, useEpoch, stepsPerEpoch, t, text]);

  return <div className="job-metrics">
    <div className="job-metrics-toolbar">
      <div className="job-metrics-axis ui-segmented ui-segmented-sm" role="group" aria-label={text('横轴单位', 'Horizontal axis')}>
        <button type="button" aria-pressed={xAxisMode === 'step'} onClick={() => setXAxisMode('step')}>{t('job.step')}</button>
        <button type="button" aria-pressed={xAxisMode === 'epoch'} onClick={() => setXAxisMode('epoch')} disabled={!stepsPerEpoch} title={!stepsPerEpoch ? t('job.epochUnavailable') : undefined}>{t('job.epoch')}</button><SlidingIndicator className="ui-segmented-thumb"/>
      </div>
      <label className="job-metrics-smoothing">{text('平滑 EMA 系数', 'Smoothing EMA coefficient')}<input aria-label={text('平滑 EMA 系数', 'Smoothing EMA coefficient')} type="range" min="0" max="0.99" step="0.01" value={emaAlpha} onChange={event => setEmaAlpha(Number(event.target.value))}/><output>{emaAlpha.toFixed(2)}</output></label>
    </div>
    {!charts.length ? <div className="job-metrics-empty"><Activity size={30} aria-hidden="true"/><p>{t('job.noMetrics', '暂无训练指标')}</p><span>{text('等待训练步数记录。', 'Waiting for recorded training steps.')}</span></div>
      : <div className="job-metrics-grid">{charts.map((chart, index) => <section key={chart.key} className={`job-metrics-chart${index === charts.length - 1 && charts.length % 2 ? ' is-wide' : ''}`} aria-label={chart.title}>
        <h2>{chart.title}</h2>{chart.note && <p>{chart.note}</p>}
        {chart.option ? <EChart option={chart.option} style={{ height: 280 }}/> : <div className="job-metrics-chart-empty"><Thermometer size={22} aria-hidden="true"/><span>{chart.empty}</span></div>}
      </section>)}</div>}
  </div>;
}
