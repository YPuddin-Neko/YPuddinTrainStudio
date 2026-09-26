/** Five significant digits; values below 0.001 always use exponent form so rates of one chart read alike. */
export function formatMetricValue(value: unknown): string {
  const scalar = Array.isArray(value) ? value.at(-1) : value;
  if (typeof scalar !== 'number' || !Number.isFinite(scalar)) return '—';
  const rounded = Number(scalar.toPrecision(5));
  return rounded !== 0 && Math.abs(rounded) < 1e-3 ? rounded.toExponential() : String(rounded);
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
