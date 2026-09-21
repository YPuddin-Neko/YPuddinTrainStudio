import { describe, expect, it } from 'vitest';
import { formatMetricValue, metricChartBase, metricLabels } from '../../../frontend/src/pages/JobDetail/metricPresentation';

describe('monitor metric presentation', () => {
  it('keeps small learning rates, zero and missing values distinct with finite precision', () => {
    expect(formatMetricValue(0.123456789012)).toBe('0.12346');
    expect(formatMetricValue(0.000000123456789)).toBe('1.2346e-7');
    expect(formatMetricValue(0)).toBe('0');
    for (const missing of [null, undefined, NaN, Infinity]) expect(formatMetricValue(missing)).toBe('—');
  });
  it('formats axis tooltips consistently without exposing long decimal tails or inventing missing metrics', () => {
    const option = metricChartBase('轮', '训练损失');
    expect(option.tooltip.formatter([
      {axisValue:1/3,seriesName:'原始损失',value:[1/3,0.123456789]},
      {axisValue:1/3,seriesName:'显示 EMA',value:[1/3,null]},
    ])).toBe('轮: 0.33333\n原始损失: 0.12346\n显示 EMA: —');
    expect(option.legend.type).toBe('scroll');
    expect(option.grid.top).toBeGreaterThan(option.legend.top + 30);
    expect(option.dataZoom.map(zoom => zoom.type)).toEqual(['slider']);
    const slider = option.dataZoom[0];
    expect(option.grid.bottom).toBeGreaterThan(slider.bottom! + slider.height! + 30);
    expect(metricLabels(false).gradient).toBe('Gradient norm'); expect(metricLabels(true).gradient).toBe('梯度范数');
  });
});
