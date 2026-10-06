import React from 'react';
import { useTranslation } from 'react-i18next';
import { Link, useLocation } from 'react-router-dom';
import { Activity, Settings2, Thermometer } from 'lucide-react';
import { apiClient } from '../../api/client';
import { EChart } from '../../components/EChart';
import { SlidingIndicator } from '../../components/motion';
import StudioSelect, { type StudioSelectOption } from '../../components/StudioSelect';
import ConfigHelp from '../../components/ConfigHelp';
import type { JobMetrics, Settings } from '../../api/types';
import { chartTitle, DEFAULT_METRIC_CHARTS, MAX_PLOT_SERIES, METRICS, type MetricChartSetting, type MetricKey } from '../../utils/metricCharts';
import { shapeValidationSeries, smoothLoss } from '../../utils/metrics';
import { gpuDeviceValues, gpuMetricDeviceLabel, gpuSourceLabel, hasSeveralGpus, resolveGpuMetricSource, GPU_SENSOR_METRICS } from '../../utils/gpuMetricSeries';
import { useWorkspaceText } from '../../utils/workspaceText';
import { axisTickLabels, formatMetricValue, formatRateValue, layoutValueAxes, learningRateGroupName, metricChartBase, metricLabels, metricRange } from './metricPresentation';
import './job-metrics.css';

const EXTRA_COLORS = ['#f59e0b', '#22d3ee', '#fb7185', '#84cc16', '#e879f9', '#a78bfa'];
const MIN_PLOT_WIDTH = 240;
const EMPTY_GPU_SOURCES: Record<string, string> = {};
/** Jobs that record nothing more; any other job may still record a missing reading. */
const FINISHED = ['completed', 'failed', 'cancelled'];
const hasReadings = (values: Array<number | null | undefined> | undefined) => values?.some(value => typeof value === 'number' && Number.isFinite(value)) ?? false;

type Line = { unit: string; name: string; color: string; data: Array<[number, number | null]>; rangeValues?: Array<number | null>; width?: number; symbols?: boolean };
type SeriesRange = { name: string; color: string; unit: string; min: number; max: number; rate?: boolean };
type Chart = { key: string; title: string; note?: string; option?: Record<string, unknown>; ranges?: SeriesRange[]; empty?: React.ReactNode; gpu?: { layoutId: string; value: string; options: StudioSelectOption[]; help: string } };

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

/** The legend's icon for a line series, drawn as ECharts draws it: a 25×14 line through a hollow circle. */
function LegendIcon({ color, name }: { color: string; name: string }) {
  return <span className="job-metrics-range-icon" title={name}>
    <svg width="25" height="14" viewBox="0 0 25 14" aria-hidden="true" focusable="false">
      <path d="M0 7H25" stroke={color} strokeWidth="2"/>
      <circle cx="12.5" cy="7" r="5.6" fill="#fff" stroke={color} strokeWidth="2"/>
    </svg>
  </span>;
}

/**
 * The job page's saved chart layout: none until settings answer, so the built-in layout is never drawn first;
 * the built-in one when none is saved or settings cannot be read.
 */
function useChartLayout(): MetricChartSetting[] | null {
  const [layout, setLayout] = React.useState<MetricChartSetting[] | null>(null);
  React.useEffect(() => {
    const controller = new AbortController();
    let changedSinceRequest = false;
    const apply = (settings: Settings | undefined) => { const saved = settings?.ui?.metric_charts; setLayout(saved?.length ? saved : DEFAULT_METRIC_CHARTS); };
    void apiClient.get<Settings>('/settings', { signal: controller.signal, silent: true })
      .then(settings => { if (!controller.signal.aborted && !changedSinceRequest) apply(settings); })
      .catch(() => { if (!controller.signal.aborted) setLayout(current => current ?? DEFAULT_METRIC_CHARTS); });
    // A layout saved while this page stays open (the settings dialog) applies at once.
    const changed = (event: Event) => { changedSinceRequest = true; apply((event as CustomEvent<Settings>).detail); };
    window.addEventListener('studio.settings.changed', changed);
    return () => { controller.abort(); window.removeEventListener('studio.settings.changed', changed); };
  }, []);
  return layout;
}

/** Training curves and GPU readings in the configured charts, two per row. */
export default function JobMetricsPanel({ metrics, stepsPerEpoch, vramMetric, device, status }: {
  metrics: JobMetrics | null; stepsPerEpoch?: number | null; vramMetric?: string | null; device?: string | null; status?: string | null;
}) {
  const { t, i18n } = useTranslation();
  const text = useWorkspaceText();
  const location = useLocation();
  const chinese = (i18n.resolvedLanguage || i18n.language || '').startsWith('zh');
  const labels = React.useMemo(() => metricLabels(chinese), [chinese]);
  const layout = useChartLayout();
  const [gpuSelection, setGpuSelection] = React.useState<{ layout: MetricChartSetting[] | null; values: Record<string, string> }>({ layout: null, values: {} });
  const gpuSources = gpuSelection.layout === layout ? gpuSelection.values : EMPTY_GPU_SOURCES;
  const [xAxisMode, setXAxisMode] = React.useState<'step' | 'epoch'>('step');
  const [emaAlpha, setEmaAlpha] = React.useState(0.9);
  const useEpoch = xAxisMode === 'epoch' && !!stepsPerEpoch;
  const xAxisName = useEpoch ? text('训练轮数', 'Epoch') : text('训练步数', 'Training step');
  const xs = React.useMemo(() => useEpoch && stepsPerEpoch ? metrics?.steps.map(step => step / stepsPerEpoch) || [] : metrics?.steps || [], [metrics, useEpoch, stepsPerEpoch]);
  const chartVramMetric = metrics?.vram_metric ?? vramMetric;
  const apple = device === 'mps' || !!metrics?.gpu_devices?.some(item => item.id === 'mps');
  const cpu = device === 'cpu';
  const live = !!status && !FINISHED.includes(status);

  const charts = React.useMemo<Chart[]>(() => {
    if (!metrics?.steps.length || !layout) return [];
    const points = (values: Array<number | null | undefined> | undefined, scale = 1) => xs.map((x, index): [number, number | null] => {
      const value = values?.[index];
      return [x, typeof value === 'number' && Number.isFinite(value) ? value / scale : null];
    });
    const memoryName = chartVramMetric === 'current_allocated' ? `${t('job.currentTrainingAllocated')} (GB)` : chartVramMetric === 'peak_allocated' ? `${t('job.vramPeak')} (GB)` : labels.memory;
    const plain: Partial<Record<MetricKey, { name: string; values: Array<number | null | undefined> | undefined; scale?: number }>> = {
      loss: { name: labels.raw, values: metrics.loss },
      loss_ema: { name: labels.ema, values: smoothLoss(metrics.loss || [], emaAlpha) },
      grad_norm: { name: labels.gradient, values: metrics.grad_norm },
      it_s: { name: labels.speed, values: metrics.it_s },
      vram: { name: memoryName, values: metrics.vram_mb, scale: 1024 },
      // Apple chips report power from the system's energy model, as the top bar's 估算功率 does.
      gpu_power: { name: apple ? text('GPU 功率（估算，W）', 'GPU power (estimated, W)') : labels.power, values: metrics.gpu_power_w },
      gpu_temp: { name: labels.temperature, values: metrics.gpu_temp_c },
      gpu_util: { name: labels.utilization, values: metrics.gpu_util_pct },
      gpu_memory: { name: text('设备显存已用 (GB)', 'Device memory used (GB)'), values: [], scale: 1024 },
    };
    const axisNames: Record<string, string> = { Loss: labels.loss, LR: labels.lr, Norm: labels.gradient };
    const list: Chart[] = [];
    for (const chart of layout) {
      const source = resolveGpuMetricSource(metrics, gpuSources[chart.id] ?? chart.gpu ?? 'primary');
      const hasGpuSensors = chart.series.some(item => GPU_SENSOR_METRICS.some(key => key === item.metric));
      // On one GPU the primary GPU, the average and that GPU are the same readings, so only several GPUs offer a choice.
      const gpuOptions: StudioSelectOption[] = [
        { value: 'primary', label: gpuSourceLabel('primary', !chinese) },
        { value: 'average', label: gpuSourceLabel('average', !chinese) },
        ...(metrics.gpu_devices || []).map(item => ({ value: item.id, label: gpuMetricDeviceLabel(item, !chinese) })),
      ];
      if (!gpuOptions.some(option => option.value === source)) gpuOptions.push({ value: source, label: `${gpuSourceLabel(source, !chinese)} · ${text('未记录', 'Not recorded')}` });
      const gpuHelp = [
        text('主训练 GPU：主训练进程所在的显卡。', 'Primary training GPU: the GPU of the main training process.'),
        text('训练 GPU 平均：每一步有读数的训练 GPU 的平均值，缺失的读数不计入。', 'Training GPU average: the mean of the training GPUs with a reading at that step; missing readings are left out.'),
        text('训练 GPU 的编号以本次任务为准。', 'Training GPU numbers count within this job.'),
        chart.series.some(item => item.metric === 'vram') && text('“显存”是主训练进程的分配值，不随显卡来源切换；所选显卡的占用见“设备显存”指标。', '“VRAM” is the main training process allocation and does not follow the GPU source; add “Device memory” for the selected GPU.'),
      ].filter(Boolean).join('\n');
      const gpu = hasGpuSensors && hasSeveralGpus(metrics) ? { layoutId: chart.id, value: source, options: gpuOptions, help: gpuHelp } : undefined;
      const chartPlain = { ...plain,
        gpu_power: { ...plain.gpu_power!, values: gpuDeviceValues(metrics, source, 'power_w') },
        gpu_temp: { ...plain.gpu_temp!, values: gpuDeviceValues(metrics, source, 'temp_c') },
        gpu_util: { ...plain.gpu_util!, values: gpuDeviceValues(metrics, source, 'util_pct') },
        gpu_memory: { ...plain.gpu_memory!, values: gpuDeviceValues(metrics, source, 'mem_used_mb') },
      };
      const allLines: Line[] = [];
      let logRates = false;
      for (const item of chart.series) {
        if (item.metric === 'lr') {
          const groups = Object.entries(metrics.lr || {});
          logRates = groupSpread(groups.map(([, values]) => values)) >= 100;
          groups.forEach(([group, values], index) => allLines.push({
            unit: 'LR', name: `${labels.lr} · ${learningRateGroupName(group)}`, color: index ? EXTRA_COLORS[(index - 1) % EXTRA_COLORS.length] : item.color,
            // A log axis has no zero: warmup's first steps are left out of the line.
            data: points(logRates ? values.map(value => typeof value === 'number' && value > 0 ? value : null) : values),
            rangeValues: values,
          }));
        } else if (item.metric === 'validation') {
          if (!metrics.validation?.length) continue;
          shapeValidationSeries(metrics.validation).series.forEach((line, index) => allLines.push({
            unit: 'Loss', name: line.name === 'mean' ? labels.mean : `${labels.timestep} ${line.name}`,
            color: line.name === 'mean' ? item.color : EXTRA_COLORS[index % EXTRA_COLORS.length], width: line.name === 'mean' ? 2.5 : 1.2, symbols: true,
            data: line.data.map(([step, loss]): [number, number | null] => [useEpoch && stepsPerEpoch ? step / stepsPerEpoch : step, loss]),
          }));
        } else {
          const values = chartPlain[item.metric];
          if (!values || !hasReadings(values.values)) continue;
          allLines.push({ unit: METRICS[item.metric].unit, name: values.name, color: item.color, data: points(values.values, values.scale), width: item.metric === 'loss' ? 1.2 : item.metric === 'loss_ema' ? 2 : undefined });
        }
      }
      const title = chartTitle(chart, !chinese);
      const keys = chart.series.map(item => item.metric);
      if (!allLines.length) {
        const gpuKeys = keys.filter(key => METRICS[key].gpu);
        if (gpuKeys.length) list.push({ key: chart.id, title, gpu, empty: cpu ? text('CPU 训练不记录 GPU 指标。', 'CPU training records no GPU metrics.')
          : apple && gpuKeys.every(key => key === 'gpu_memory') ? text('Apple 芯片使用统一内存，不单独记录设备显存。', 'Apple chips use unified memory, so device memory is not recorded separately.')
            : live ? text('此任务尚未记录所选的 GPU 指标。', 'No values have been recorded yet for the selected GPU metrics.')
              : text('此任务没有记录所选的 GPU 指标。', 'This job has no recorded values for the selected GPU metrics.') });
        continue;
      }
      const parts = Math.ceil(allLines.length / MAX_PLOT_SERIES);
      for (let part = 0; part < parts; part += 1) {
        const lines = allLines.slice(part * MAX_PLOT_SERIES, (part + 1) * MAX_PLOT_SERIES);
        const units = [...new Set(lines.map(line => line.unit))];
        const rates = new Set(lines.filter(line => line.unit === 'LR').map(line => line.name));
        const base = metricChartBase(xAxisName, axisNames[units[0]] || units[0], (name, value) => (name && rates.has(name) ? formatRateValue : formatMetricValue)(value));
        // Units alternate left and right; further axes move outward past the labels of the axis inside them.
        const placed = layoutValueAxes(units.map(unit => {
          let low = Infinity, high = -Infinity;
          for (const line of lines) {
            if (line.unit !== unit) continue;
            for (const [, value] of line.data) if (typeof value === 'number' && Number.isFinite(value)) { low = Math.min(low, value); high = Math.max(high, value); }
          }
          return { name: axisNames[unit] || unit, labels: axisTickLabels(low, high, unit === 'LR' && logRates, unit === 'LR' ? formatRateValue : formatMetricValue) };
        }));
        // Stable ids let each update change series and axes in place; the chart removes the ones that are gone.
        const axes = units.map((unit, index) => ({
          // `show` is set every time so that an axis hidden at a narrow width comes back when the chart widens.
          ...base.yAxis, id: unit, show: true, name: axisNames[unit] || unit, position: index % 2 ? 'right' as const : 'left' as const, offset: placed.offsets[index],
          splitLine: { show: index === 0 }, ...(unit === 'LR' ? { axisLabel: { formatter: formatRateValue } } : {}),
          ...(unit === 'LR' && logRates ? { type: 'log' as const, logBase: 10, scale: undefined } : {}),
        }));
        const option = {
          ...base,
          grid: { ...base.grid, left: base.grid.left + placed.left, right: base.grid.right + placed.right },
          yAxis: axes.length === 1 ? axes[0] : axes,
          series: lines.map(line => ({
            // Replaced axes retain their original indices when an earlier axis disappears.
            id: line.name, name: line.name, type: 'line', showSymbol: !!line.symbols, sampling: 'lttb', data: line.data, yAxisId: line.unit,
            lineStyle: { width: line.width ?? 1.5, color: line.color }, itemStyle: { color: line.color },
          })),
          // A chart too narrow for every axis keeps one per side; the other lines keep their own scales and
          // show their values in the tooltip.
          ...(axes.length > 2 ? { media: [{
            query: { maxWidth: Math.round(base.grid.left + base.grid.right + placed.width + MIN_PLOT_WIDTH) },
            option: { grid: { left: base.grid.left, right: base.grid.right }, yAxis: axes.map((_, index) => index < 2 ? {} : { show: false }) },
          }] } : {}),
        };
        const only = keys.length === 1 ? keys[0] : null;
        const sensorKeys = keys.filter(key => GPU_SENSOR_METRICS.some(sensor => sensor === key));
        // CPU runs have no GPU readings and Apple chips no separate device memory; neither is reported as missing.
        const missingSensors = sensorKeys.filter(key => !cpu && !(apple && key === 'gpu_memory') && !hasReadings(chartPlain[key]?.values));
        const readings = sensorKeys.some(key => hasReadings(chartPlain[key]?.values));
        const missingNames = (english: boolean) => missingSensors.map(key => METRICS[key].label[english ? 1 : 0]).join(english ? ', ' : '、');
        const missingNote = !missingSensors.length ? '' : live
          ? text(`此任务尚未记录：${missingNames(false)}。`, `Not recorded yet for this job: ${missingNames(true)}.`)
          : text(`此任务没有记录：${missingNames(false)}。`, `Not recorded for this job: ${missingNames(true)}.`);
        const appleNote = [
          apple && readings && text('Apple 芯片的功率是系统能耗估算值，温度是 GPU 各温区的平均值，利用率是整块 GPU 的占用。', 'On Apple chips, power is the system’s energy estimate, temperature the mean of the GPU’s thermal zones and utilization the whole GPU’s load.'),
          apple && keys.includes('gpu_memory') && text('Apple 芯片使用统一内存，不单独记录设备显存。', 'Apple chips use unified memory, so device memory is not recorded separately.'),
        ].filter(Boolean).join(' ');
        const chartNote = keys.every(key => key === 'loss' || key === 'loss_ema')
          ? text('每步 Loss 是每个优化步的训练损失；平滑曲线按上方 EMA 系数计算，只影响显示，不改变训练。', 'Loss per step is the training loss of each optimizer step; the smoothed curve uses the EMA coefficient above and only changes the chart.')
          : only === 'lr' ? (logRates
            ? text('每条线代表一个参数组。各组学习率相差 100 倍以上，纵轴用对数刻度，DoRA 这类很小的值也能看清。', 'Each line is one parameter group. The groups differ by 100× or more, so the axis is logarithmic and small rates such as DoRA stay visible.')
            : text('每条线代表一个参数组；LoKr 的 w1 / w2 可设置不同学习率。', 'Each line is one parameter group; LoKr w1 / w2 can use different learning rates.'))
            : only === 'grad_norm' ? text('每步梯度的大小，用于观察更新是否稳定。', 'Gradient magnitude per step, to inspect update stability.')
              : only === 'it_s' ? text('每秒完成的优化步数。', 'Optimizer steps completed per second.')
                : only === 'vram' ? (chartVramMetric === 'current_allocated' ? text('训练进程当前占用的显存。', 'Memory currently allocated by the training process.') : text('训练进程到这一步为止的显存峰值。', 'Peak memory allocated by the training process so far.'))
                  : units.length > 1 ? text('单位不同的指标各用一条纵轴，颜色与图例一致。', 'Metrics with different units each use their own axis, in the legend’s colors.') : undefined;
        const note = [chartNote, missingNote, appleNote].filter(Boolean).join(' ') || undefined;
        const ranges = lines.flatMap(line => {
          const range = metricRange(line.rangeValues ?? line.data.map(([, value]) => value));
          return range ? [{ ...range, name: line.name, color: line.color, unit: ['Loss', 'LR', 'Norm'].includes(line.unit) ? '' : line.unit, rate: line.unit === 'LR' }] : [];
        });
        list.push({
          key: part === 0 ? chart.id : `${chart.id}/part-${part + 1}`,
          title: parts === 1 ? title : text(`${title}（${part + 1}/${parts}）`, `${title} (${part + 1}/${parts})`),
          note, option, ranges, gpu,
        });
      }
    }
    return list;
  }, [metrics, xs, xAxisName, labels, emaAlpha, chartVramMetric, apple, cpu, live, useEpoch, stepsPerEpoch, layout, chinese, t, text, gpuSources]);

  return <div className="job-metrics">
    <div className="job-metrics-toolbar">
      <div className="job-metrics-axis ui-segmented ui-segmented-sm" role="group" aria-label={text('横轴单位', 'Horizontal axis')}>
        <button type="button" aria-pressed={xAxisMode === 'step'} onClick={() => setXAxisMode('step')}>{text('训练步数', 'Training steps')}</button>
        <button type="button" aria-pressed={xAxisMode === 'epoch'} onClick={() => setXAxisMode('epoch')} disabled={!stepsPerEpoch} title={!stepsPerEpoch ? t('job.epochUnavailable') : undefined}>{text('训练轮数', 'Epochs')}</button><SlidingIndicator className="ui-segmented-thumb"/>
      </div>
      <div className="job-metrics-tools">
        <label className="job-metrics-smoothing">{text('平滑 EMA 系数', 'Smoothing EMA coefficient')}<input aria-label={text('平滑 EMA 系数', 'Smoothing EMA coefficient')} type="range" min="0" max="0.99" step="0.01" value={emaAlpha} onChange={event => setEmaAlpha(Number(event.target.value))}/><output>{emaAlpha.toFixed(2)}</output></label>
        <Link className="ui-btn ui-btn-sm ui-btn-quiet" to="/settings/charts" state={{ backgroundLocation: location }}><Settings2 size={14}/>{text('自定义图表', 'Customize charts')}</Link>
      </div>
    </div>
    {!layout && !!metrics?.steps.length ? <div className="job-metrics-grid" aria-busy="true" aria-label={text('读取图表布局', 'Loading the chart layout')}>{Array.from({ length: 4 }, (_, index) => <div key={index} className="job-metrics-chart" aria-hidden="true"><span className="ui-skeleton job-metrics-skeleton-title"/><span className="ui-skeleton job-metrics-skeleton-plot"/></div>)}</div>
      : !charts.length ? <div className="job-metrics-empty"><Activity size={30} aria-hidden="true"/><p>{t('job.noMetrics', '暂无训练指标')}</p><span>{text('等待训练步数记录。', 'Waiting for recorded training steps.')}</span></div>
      : <div className="job-metrics-grid">{charts.map((chart, index) => <section key={chart.key} className={`job-metrics-chart${index === charts.length - 1 && charts.length % 2 ? ' is-wide' : ''}`} aria-label={chart.title}>
        <div className="job-metrics-chart-heading"><h2>{chart.title}</h2>{chart.gpu && <div className="job-metrics-gpu-source"><StudioSelect className="job-metrics-gpu-select" value={chart.gpu.value} options={chart.gpu.options} aria-label={text(`${chart.title}：GPU 数据来源`, `${chart.title}: GPU data source`)} onValueChange={value => setGpuSelection(current => ({ layout, values: { ...(current.layout === layout ? current.values : {}), [chart.gpu!.layoutId]: value } }))}/><ConfigHelp label={text(`${chart.title}：GPU 数据来源说明`, `${chart.title}: GPU data source help`)}>{chart.gpu.help}</ConfigHelp></div>}</div>{chart.note && <p>{chart.note}</p>}
        {chart.option ? <EChart option={chart.option} style={{ height: 280 }}/> : <div className="job-metrics-chart-empty"><Thermometer size={22} aria-hidden="true"/><span>{chart.empty}</span></div>}
        {!!chart.ranges?.length && <ul className="job-metrics-ranges" aria-label={text('已记录指标的最大值和最小值', 'Maximum and minimum of recorded metrics')}>
          {chart.ranges.map(range => <li key={range.name} aria-label={range.name}>
            <LegendIcon color={range.color} name={range.name}/>
            <dl>{([['max', text('最大值', 'Max')], ['min', text('最小值', 'Min')]] as const).map(([key, label]) => <div key={key}><dt>{label}</dt><dd>{(range.rate ? formatRateValue : formatMetricValue)(range[key])}{range.unit && ` ${range.unit}`}</dd></div>)}</dl>
          </li>)}
        </ul>}
      </section>)}</div>}
  </div>;
}
