import React from 'react';
import { ArrowDown, ArrowUp, GripVertical, Plus, RotateCcw, Trash2 } from 'lucide-react';
import StudioSelect from '../../components/StudioSelect';
import { apiClient } from '../../api/client';
import type { SystemStats } from '../../api/types';
import { chartTitle, cloneCharts, DEFAULT_METRIC_CHARTS, isDefaultLayout, MAX_CHARTS, MAX_SERIES, METRIC_KEYS, METRICS, metricLabel, newChartId, type MetricChartSetting, type MetricKey } from '../../utils/metricCharts';
import { useWorkspaceText } from '../../utils/workspaceText';
import { useMetricChartDrag } from './useMetricChartDrag';
import './metric-chart-settings.css';

/**
 * Which charts the job page shows, which metrics share a chart and each metric's color.
 * Metrics with different units in one chart each get their own vertical axis.
 */
export default function MetricChartEditor({ id, charts, onChange, onSortingChange }: { id: string; charts: MetricChartSetting[]; onChange: (charts: MetricChartSetting[]) => void; onSortingChange?: (sorting: boolean) => void }) {
  const text = useWorkspaceText();
  const sorting = useMetricChartDrag(charts, onChange);
  const sortingActive = sorting.drag !== null;
  React.useEffect(() => { onSortingChange?.(sortingActive); }, [sortingActive, onSortingChange]);
  const english = text('zh', 'en') === 'en';
  const [gpus, setGpus] = React.useState<SystemStats['gpus']>([]);
  React.useEffect(() => {
    const controller = new AbortController();
    void apiClient.get<SystemStats>('/system/stats', { signal: controller.signal, silent: true })
      .then(stats => { if (!controller.signal.aborted) setGpus(stats.gpus ?? []); })
      .catch(() => { /* Saved device IDs remain selectable when telemetry is unavailable. */ });
    return () => controller.abort();
  }, []);
  const gpuOptions = [
    { value: 'primary', label: text('主训练 GPU', 'Primary training GPU') },
    { value: 'average', label: text('训练 GPU 平均', 'Average of training GPUs') },
    ...gpus.map((gpu, index) => ({ value: gpu.kind === 'mps' ? 'mps' : `cuda:${index}`, label: gpu.kind === 'mps' ? 'Apple GPU' : text(`训练 GPU ${index}`, `Training GPU ${index}`) })),
  ];

  const updateChart = (index: number, patch: Partial<MetricChartSetting>) => { if (sortingActive) return; onChange(charts.map((chart, other) => other === index ? { ...chart, ...patch } : chart)); };
  const move = (index: number, offset: number) => {
    if (sortingActive) return;
    sorting.measure();
    const next = [...charts]; const [chart] = next.splice(index, 1); next.splice(index + offset, 0, chart); onChange(next);
  };
  const unused = (chart: MetricChartSetting) => METRIC_KEYS.filter(key => !chart.series.some(item => item.metric === key));

  return <section id={id} data-settings-section tabIndex={-1} className="settings-section metric-chart-settings" aria-labelledby={`${id}-heading`}>
    <div className="settings-section-heading"><div>
      <h2 id={`${id}-heading`}>{text('指标图表', 'Metric charts')}</h2>
    </div></div>
    <ol ref={sorting.list} onLostPointerCapture={sorting.lostCapture} className="metric-chart-list" data-dragging={sorting.drag ? 'true' : undefined}>
      {sorting.charts.map((chart, index) => <li key={chart.id} className="metric-chart-card" data-chart-id={chart.id} data-placeholder={sorting.drag?.id === chart.id ? 'true' : undefined} aria-label={chartTitle(chart, english)} style={sorting.drag?.id === chart.id ? { height: sorting.drag.rect.height } : undefined}>
        <div className="metric-chart-body" data-floating={sorting.drag?.id === chart.id ? 'true' : undefined} data-settling={sorting.drag?.id === chart.id && sorting.drag.settling ? 'true' : undefined} style={sorting.drag?.id === chart.id ? { left: sorting.drag.rect.left, top: sorting.drag.rect.top, width: sorting.drag.rect.width, height: sorting.drag.rect.height } : undefined}>
        <div className="metric-chart-card-head">
          <span className="metric-chart-order">{index + 1}</span>
          <label className="metric-chart-title"><span className="sr-only">{text('图表标题', 'Chart title')}</span>
            <input disabled={sortingActive} value={chart.title} maxLength={40} placeholder={text(`自动：${chartTitle({ ...chart, title: '' }, false)}`, `Automatic: ${chartTitle({ ...chart, title: '' }, true)}`)} aria-label={text(`第 ${index + 1} 张图的标题`, `Title of chart ${index + 1}`)}
              onChange={event => updateChart(index, { title: event.target.value })}/></label>
          <div className="metric-chart-card-actions">
            <button type="button" className="ui-btn ui-btn-icon ui-btn-sm ui-btn-quiet" disabled={sortingActive || index === 0} onClick={() => move(index, -1)} aria-label={text('上移', 'Move up')} title={text('上移', 'Move up')}><ArrowUp size={14}/></button>
            <button type="button" className="ui-btn ui-btn-icon ui-btn-sm ui-btn-quiet" disabled={sortingActive || index === charts.length - 1} onClick={() => move(index, 1)} aria-label={text('下移', 'Move down')} title={text('下移', 'Move down')}><ArrowDown size={14}/></button>
            <button type="button" className="ui-btn ui-btn-icon ui-btn-sm ui-btn-quiet ui-btn-danger" disabled={sortingActive || charts.length === 1} onClick={() => onChange(charts.filter((_, other) => other !== index))} aria-label={text('删除这张图', 'Remove this chart')} title={text('删除这张图', 'Remove this chart')}><Trash2 size={14}/></button>
          </div>
        </div>
        <ul className="metric-series-list">
          {chart.series.map((item, position) => <li key={item.metric} className="metric-series-row">
            <input disabled={sortingActive} type="color" value={item.color} aria-label={text(`${metricLabel(item.metric, false)} · 颜色`, `${metricLabel(item.metric, true)} · color`)}
              onChange={event => updateChart(index, { series: chart.series.map((other, at) => at === position ? { ...other, color: event.target.value } : other) })}/>
            <StudioSelect disabled={sortingActive} aria-label={text(`第 ${index + 1} 张图的第 ${position + 1} 个指标`, `Metric ${position + 1} of chart ${index + 1}`)} value={item.metric}
              options={[item.metric, ...unused(chart)].map(key => ({ value: key, label: `${metricLabel(key, english)} · ${METRICS[key].unit}` }))}
              onValueChange={value => updateChart(index, { series: chart.series.map((other, at) => at === position ? { metric: value as MetricKey, color: METRICS[value as MetricKey].color } : other) })}/>
            <button type="button" className="ui-btn ui-btn-icon ui-btn-sm ui-btn-quiet" disabled={sortingActive || chart.series.length === 1} onClick={() => updateChart(index, { series: chart.series.filter((_, at) => at !== position) })}
              aria-label={text(`移除${metricLabel(item.metric, false)}`, `Remove ${metricLabel(item.metric, true)}`)} title={text('移除', 'Remove')}><Trash2 size={14}/></button>
          </li>)}
        </ul>
        {chart.series.some(item => ['gpu_memory', 'gpu_power', 'gpu_temp', 'gpu_util'].includes(item.metric)) && <label className="metric-chart-gpu-source" title={text('GPU 编号以本次训练任务为准。', 'GPU indices refer to the current training job.')}><span>{text('显卡', 'GPU')}</span>
          <StudioSelect disabled={sortingActive} aria-label={text(`第 ${index + 1} 张图的显卡`, `GPU for chart ${index + 1}`)} value={chart.gpu ?? 'primary'}
            options={gpuOptions.some(option => option.value === (chart.gpu ?? 'primary')) ? gpuOptions : [...gpuOptions, { value: chart.gpu!, label: chart.gpu === 'mps' ? 'Apple GPU' : chart.gpu!.replace('cuda:', text('训练 GPU ', 'Training GPU ')) }]}
            onValueChange={value => updateChart(index, { gpu: value })}/>
        </label>}
        <div className="metric-chart-card-footer"><button type="button" className="ui-btn ui-btn-sm metric-series-add" disabled={sortingActive || chart.series.length >= MAX_SERIES || !unused(chart).length}
          onClick={() => { const metric = unused(chart)[0]; updateChart(index, { series: [...chart.series, { metric, color: METRICS[metric].color }] }); }}><Plus size={14}/>{text('添加指标', 'Add metric')}</button>
          <button type="button" className="ui-btn ui-btn-icon ui-btn-sm ui-btn-quiet metric-chart-drag-handle" disabled={charts.length < 2} aria-label={text(`拖动排序：${chartTitle(chart, false)}`, `Drag to reorder: ${chartTitle(chart, true)}`)} title={text('拖动排序', 'Drag to reorder')} onPointerDown={event => sorting.begin(event, chart.id)}><GripVertical size={16}/></button>
        </div></div>
      </li>)}
    </ol>
    <div className="metric-chart-footer">
      <button type="button" className="ui-btn" disabled={sortingActive || charts.length >= MAX_CHARTS} onClick={() => onChange([...charts, { id: newChartId(charts), title: '', series: [{ metric: 'loss', color: METRICS.loss.color }] }])}><Plus size={14}/>{text('添加图表', 'Add chart')}</button>
      <button type="button" className="ui-btn ui-btn-quiet" disabled={sortingActive || isDefaultLayout(charts)} onClick={() => onChange(cloneCharts(DEFAULT_METRIC_CHARTS))}
        title={text('换回内置的图表布局，保存后生效', 'Switch back to the built-in layout; takes effect when saved')}><RotateCcw size={14}/>{text('恢复默认', 'Restore defaults')}</button>
    </div>
  </section>;
}
