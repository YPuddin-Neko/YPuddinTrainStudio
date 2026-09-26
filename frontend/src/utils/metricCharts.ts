import type { components } from '../api/generated';

export type MetricChartSetting = components['schemas']['MetricChartSetting'];
export type MetricSeriesSetting = components['schemas']['MetricSeriesSetting'];
export type MetricKey = MetricSeriesSetting['metric'];

export interface MetricInfo {
  label: [string, string];
  /** Metrics with the same unit share a vertical axis in one chart. */
  unit: string;
  color: string;
  /** Learning rate and validation draw one line per parameter group or timestep. */
  expands?: boolean;
  gpu?: boolean;
}

export const METRICS: Record<MetricKey, MetricInfo> = {
  loss: { label: ['每步 Loss', 'Loss per step'], unit: 'Loss', color: '#93c5fd' },
  loss_ema: { label: ['平滑 Loss (EMA)', 'Smoothed loss (EMA)'], unit: 'Loss', color: '#2563eb' },
  lr: { label: ['学习率', 'Learning rate'], unit: 'LR', color: '#a78bfa', expands: true },
  grad_norm: { label: ['梯度范数', 'Gradient norm'], unit: 'Norm', color: '#d97706' },
  it_s: { label: ['训练速度', 'Training speed'], unit: 'it/s', color: '#10b981' },
  vram: { label: ['显存', 'VRAM'], unit: 'GB', color: '#ec4899', gpu: true },
  gpu_power: { label: ['GPU 功率', 'GPU power'], unit: 'W', color: '#8b5cf6', gpu: true },
  gpu_temp: { label: ['GPU 温度', 'GPU temperature'], unit: '°C', color: '#ef4444', gpu: true },
  gpu_util: { label: ['GPU 利用率', 'GPU utilization'], unit: '%', color: '#0ea5e9', gpu: true },
  validation: { label: ['验证损失', 'Validation loss'], unit: 'Loss', color: '#f43f5e', expands: true },
};

export const METRIC_KEYS = Object.keys(METRICS) as MetricKey[];
export const MAX_CHARTS = 16;
export const MAX_SERIES = 6;

const series = (...keys: MetricKey[]): MetricSeriesSetting[] => keys.map(metric => ({ metric, color: METRICS[metric].color }));

/** The built-in layout: training curves, then the GPU's memory, power, temperature and load in one chart. */
export const DEFAULT_METRIC_CHARTS: MetricChartSetting[] = [
  { id: 'loss', title: '', series: series('loss', 'loss_ema') },
  { id: 'lr', title: '', series: series('lr') },
  { id: 'gradient', title: '', series: series('grad_norm') },
  { id: 'speed', title: '', series: series('it_s') },
  { id: 'validation', title: '', series: series('validation') },
  { id: 'gpu', title: '', series: series('vram', 'gpu_power', 'gpu_temp', 'gpu_util') },
];

export function metricLabel(metric: MetricKey, english: boolean): string {
  return METRICS[metric].label[english ? 1 : 0];
}

/** An untitled chart is named after what it draws. */
export function chartTitle(chart: MetricChartSetting, english: boolean): string {
  if (chart.title.trim()) return chart.title.trim();
  const keys = chart.series.map(item => item.metric);
  if (keys.length > 1 && keys.every(key => METRICS[key].gpu)) return english ? 'GPU status' : 'GPU 状态';
  if (keys.length && keys.every(key => key === 'loss' || key === 'loss_ema')) return 'Loss';
  return keys.map(key => metricLabel(key, english)).join(' · ');
}

export function cloneCharts(charts: MetricChartSetting[]): MetricChartSetting[] {
  return charts.map(chart => ({ ...chart, series: chart.series.map(item => ({ ...item })) }));
}

/** Compares layouts by content; colors from a color input come back lowercase. */
export function layoutKey(charts: MetricChartSetting[]): string {
  return JSON.stringify(charts.map(chart => [chart.id, chart.title, chart.series.map(item => [item.metric, item.color.toLowerCase()])]));
}

const DEFAULT_KEY = layoutKey(DEFAULT_METRIC_CHARTS);
export const isDefaultLayout = (charts: MetricChartSetting[]) => layoutKey(charts) === DEFAULT_KEY;

export function newChartId(charts: MetricChartSetting[]): string {
  let index = charts.length + 1;
  while (charts.some(chart => chart.id === `chart-${index}`)) index += 1;
  return `chart-${index}`;
}
