/** Five significant digits keep small learning rates readable without long tails. */
export function formatMetricValue(value: unknown): string {
  const scalar = Array.isArray(value) ? value.at(-1) : value;
  return typeof scalar === 'number' && Number.isFinite(scalar) ? String(Number(scalar.toPrecision(5))) : '—';
}

export function metricLabels(chinese: boolean) {
  return {
    loss: chinese ? '训练损失' : 'Training loss',
    raw: chinese ? '原始损失' : 'Raw loss',
    ema: chinese ? '显示 EMA' : 'Display EMA',
    lr: chinese ? '学习率' : 'Learning rate',
    gradient: chinese ? '梯度范数' : 'Gradient norm',
    speed: chinese ? '训练速度 (it/s)' : 'Training speed (it/s)',
    validation: chinese ? '验证损失' : 'Validation loss',
    mean: chinese ? '验证均值' : 'Validation mean',
    memory: chinese ? '显存 (GB)' : 'VRAM (GB)',
    timestep: chinese ? '噪声时间步' : 'Noise timestep',
  };
}

export function metricChartBase(xAxisName: string, yAxisName: string) {
  return {
    tooltip: {
      trigger: 'axis' as const, renderMode: 'richText' as const, confine: true,
      textStyle: { fontSize: 12 },
      formatter: (input: unknown) => {
        const points = (Array.isArray(input) ? input : [input]).filter((point): point is { axisValue?: unknown; seriesName?: string; value?: unknown } => !!point && typeof point === 'object');
        if (!points.length) return '';
        return [`${xAxisName}: ${formatMetricValue(points[0].axisValue)}`, ...points.map(point => `${point.seriesName || yAxisName}: ${formatMetricValue(point.value)}`)].join('\n');
      },
    },
    legend: { type: 'scroll' as const, top: 4, left: 12, right: 12, textStyle: { fontSize: 11 } },
    // Reserve separate top legend, axes and bottom zoom regions at every width.
    grid: { left: 12, right: 26, top: 62, bottom: 78, containLabel: true },
    xAxis: { type: 'value' as const, name: xAxisName, nameLocation: 'middle' as const, nameGap: 30, splitLine: { show: false }, axisLabel: { formatter: formatMetricValue } },
    yAxis: { type: 'value' as const, name: yAxisName, scale: true, axisLabel: { formatter: formatMetricValue } },
    dataZoom: [{ type: 'slider' as const, bottom: 8, height: 18, showDetail: false, brushSelect: false }],
  };
}
