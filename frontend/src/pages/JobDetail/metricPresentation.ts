/** Five significant digits; values below 0.001 always use exponent form so rates of one chart read alike. */
export function formatMetricValue(value: unknown): string {
  const scalar = Array.isArray(value) ? value.at(-1) : value;
  if (typeof scalar !== 'number' || !Number.isFinite(scalar)) return '—';
  const rounded = Number(scalar.toPrecision(5));
  return rounded !== 0 && Math.abs(rounded) < 1e-3 ? rounded.toExponential() : String(rounded);
}

/** Learning rates always read in exponent form (0.013965 → 1.3965e-2), so every group of a chart reads alike. */
export function formatRateValue(value: unknown): string {
  const scalar = Array.isArray(value) ? value.at(-1) : value;
  if (typeof scalar !== 'number' || !Number.isFinite(scalar)) return '—';
  return scalar === 0 ? '0' : Number(scalar.toPrecision(5)).toExponential().replace('e+', 'e');
}

/** Use recorded values before chart downsampling; missing readings are not zero. */
export function metricRange(values: Iterable<number | null | undefined>): { min: number; max: number } | null {
  let min = Infinity, max = -Infinity;
  for (const value of values) {
    if (typeof value !== 'number' || !Number.isFinite(value)) continue;
    min = Math.min(min, value); max = Math.max(max, value);
  }
  return min === Infinity ? null : { min, max };
}

const GROUP_NAMES: Record<string, string> = { dora: 'DoRA', lambda: 'λ' };

/** An optimizer parameter group's name, the same in the summary card and the chart legend. */
export function learningRateGroupName(group: string): string {
  return GROUP_NAMES[group] || group;
}

export function metricLabels(chinese: boolean) {
  return {
    loss: 'Loss',
    raw: chinese ? '每步 Loss' : 'Loss per step',
    ema: chinese ? '平滑曲线 (EMA)' : 'Smoothed (EMA)',
    lr: chinese ? '学习率' : 'Learning rate',
    gradient: chinese ? '梯度范数' : 'Gradient norm',
    speed: chinese ? '训练速度 (it/s)' : 'Training speed (it/s)',
    validation: chinese ? '验证损失' : 'Validation loss',
    mean: chinese ? '验证均值' : 'Validation mean',
    memory: chinese ? '显存 (GB)' : 'VRAM (GB)',
    power: chinese ? 'GPU 功率 (W)' : 'GPU power (W)',
    temperature: chinese ? 'GPU 温度 (°C)' : 'GPU temperature (°C)',
    utilization: chinese ? 'GPU 利用率 (%)' : 'GPU utilization (%)',
    timestep: chinese ? '噪声时间步' : 'Noise timestep',
  };
}

export function metricChartBase(xAxisName: string, yAxisName: string, formatSeries: (seriesName: string | undefined, value: unknown) => string = (_, value) => formatMetricValue(value)) {
  return {
    tooltip: {
      trigger: 'axis' as const, renderMode: 'richText' as const, confine: true,
      textStyle: { fontSize: 12 },
      formatter: (input: unknown) => {
        const points = (Array.isArray(input) ? input : [input]).filter((point): point is { axisValue?: unknown; seriesName?: string; value?: unknown } => !!point && typeof point === 'object');
        if (!points.length) return '';
        return [`${xAxisName}: ${formatMetricValue(points[0].axisValue)}`, ...points.map(point => `${point.seriesName || yAxisName}: ${formatSeries(point.seriesName, point.value)}`)].join('\n');
      },
    },
    legend: { type: 'scroll' as const, top: 4, left: 12, right: 12, textStyle: { fontSize: 11 } },
    // Reserve separate top legend, axes and bottom zoom regions at every width.
    grid: { left: 12, right: 26, top: 62, bottom: 52, containLabel: true },
    xAxis: { type: 'value' as const, splitLine: { show: false }, axisLabel: { formatter: formatMetricValue } },
    yAxis: { type: 'value' as const, name: yAxisName, scale: true, axisLabel: { formatter: formatMetricValue } },
    dataZoom: [{ type: 'slider' as const, bottom: 8, height: 18, showDetail: false, brushSelect: false }],
  };
}

/** Approximate width in px of ECharts' 12px sans-serif axis text (labels and axis names). */
export function labelTextWidth(text: string): number {
  let width = 0;
  for (const char of text) {
    width += /[\u2E80-\uFFFF]/.test(char) ? 12 : /[.,:;|!il]/.test(char) ? 3.5 : /[-–]/.test(char) ? 4.5 : /[mwMW%]/.test(char) ? 10 : 7;
  }
  return width;
}

/** The labels a value axis shows for data between min and max: ECharts' nice ticks in the chart's number format. */
export function axisTickLabels(min: number, max: number, log = false, format: (value: number) => string = formatMetricValue): string[] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min > max) return [];
  if (log) {
    const labels: string[] = [];
    for (let exponent = Math.floor(Math.log10(min)); exponent <= Math.ceil(Math.log10(max)) && labels.length < 40; exponent += 1) labels.push(format(10 ** exponent));
    return labels;
  }
  if (min === max) { const pad = Math.abs(min) / 2 || 1; min -= pad; max += pad; }
  const raw = (max - min) / 5;
  const exponent = Math.floor(Math.log10(raw));
  const fraction = raw / 10 ** exponent;
  const interval = (fraction < 1.5 ? 1 : fraction < 2.5 ? 2 : fraction < 4 ? 3 : fraction < 7 ? 5 : 10) * 10 ** exponent;
  // The extent is rounded out to whole ticks.
  const first = Math.floor(min / interval), last = Math.ceil(max / interval);
  const labels: string[] = [];
  for (let step = first; step <= last && labels.length < 40; step += 1) labels.push(format(Number((step * interval).toPrecision(10))));
  return labels;
}

const LABEL_MARGIN = 8; // ECharts' default gap between an axis line and its labels
const AXIS_GAP = 12;
const ZOOM_SLACK = 7; // zooming in can add a digit to the labels

/**
 * Offsets for value axes that alternate left and right. Each further axis on a side moves out past the labels
 * of the one inside it, and far enough that the axis names above them do not touch.
 */
export function layoutValueAxes(axes: Array<{ name: string; labels: string[] }>): { offsets: number[]; left: number; right: number; width: number } {
  const edge = { left: 0, right: 0 };
  const inner: Partial<Record<'left' | 'right', { offset: number; labels: number; name: number }>> = {};
  const offsets = axes.map((axis, index) => {
    const side = index % 2 ? 'right' : 'left';
    const name = labelTextWidth(axis.name);
    const previous = inner[side];
    const offset = previous ? previous.offset + Math.max(LABEL_MARGIN + previous.labels + AXIS_GAP, (previous.name + name) / 2 + 8) : 0;
    inner[side] = { offset, labels: Math.max(0, ...axis.labels.map(labelTextWidth)) + ZOOM_SLACK, name };
    edge[side] = offset;
    return offset;
  });
  // Room all axes take beside the plot, the outermost labels included.
  const width = (['left', 'right'] as const).reduce((sum, side) => sum + (inner[side] ? inner[side]!.offset + LABEL_MARGIN + inner[side]!.labels : 0), 0);
  return { offsets, left: edge.left, right: edge.right, width };
}
