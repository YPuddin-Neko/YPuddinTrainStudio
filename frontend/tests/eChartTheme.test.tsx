import {act, render, waitFor} from '@testing-library/react';
import {afterEach, describe, expect, it, vi} from 'vitest';
import * as echarts from 'echarts/core';
import {SVGRenderer} from 'echarts/renderers';
import {EChart} from '../src/components/EChart';
import {readChartTheme} from '../src/components/chartTheme';
import {metricChartBase} from '../src/pages/JobDetail/metricPresentation';

echarts.use(SVGRenderer);
vi.mock('echarts/core', async () => {
  const actual = await vi.importActual<typeof import('echarts/core')>('echarts/core');
  return {...actual, init: vi.fn((_element: unknown, theme: object) => actual.init(null, theme, {renderer: 'svg', ssr: true, width: 640, height: 360}))};
});
afterEach(() => {
  document.documentElement.classList.remove('dark', 'unrelated');
  document.documentElement.style.removeProperty('--studio-text');
  vi.restoreAllMocks();
  vi.clearAllMocks();
});

describe('shared chart theme', () => {
  it('resolves CSS colors for Canvas without introducing a series palette', () => {
    document.documentElement.style.setProperty('--studio-text', '#123456');
    const theme = readChartTheme();
    expect(theme.textStyle.color).toBe('#123456');
    expect(theme.valueAxis.nameTextStyle.color).toBe('#123456');
    expect(theme).not.toHaveProperty('color');
    expect(theme).not.toHaveProperty('series');
    expect(theme.backgroundColor).toBe('transparent');
  });

  it('updates real ECharts axes, legend, tooltip and slider in both themes while preserving data and interaction', async () => {
    // Real ECharts option/theme merging and actions, rendered in CPU SVG SSR so
    // jsdom's absent Canvas implementation cannot turn this into a chart mock.
    const init = vi.mocked(echarts.init);
    const base = metricChartBase('步骤', '损失');
    const data = [[0, 0.5], [1, 0.3], [2, 0.2]];
    const option = {
      ...base, animation: false,
      yAxis: [base.yAxis, {...base.yAxis, name: 'GB', splitLine: {show: false}}],
      series: [{name: '损失', type: 'line', data, lineStyle: {color: '#f43f5e'}, itemStyle: {color: '#f43f5e'}}],
    };
    document.documentElement.classList.add('dark');
    const {rerender, unmount} = render(<EChart option={option}/>);
    const chart = init.mock.results[0].value as echarts.ECharts;
    const read = () => chart.getOption() as any;
    const assertTheme = (dark: boolean) => {
      const current = read();
      expect(current.textStyle.color).toBe(dark ? '#cbd5e1' : '#334155');
      for (const axis of [...current.xAxis, ...current.yAxis]) {
        expect(axis.axisLabel.color).toBe(dark ? '#94a3b8' : '#64748b');
        expect(axis.nameTextStyle.color).toBe(dark ? '#cbd5e1' : '#334155');
      }
      expect(current.yAxis[0].splitLine.lineStyle.color).toBe(dark ? '#475569' : '#cbd5e1');
      expect(current.yAxis[1].splitLine.show).toBe(false);
      expect(current.legend[0].textStyle.color).toBe(dark ? '#cbd5e1' : '#334155');
      expect(current.legend[0].pageTextStyle.color).toBe(dark ? '#94a3b8' : '#64748b');
      expect(current.tooltip[0].backgroundColor).toBe(dark ? '#0f172a' : '#ffffff');
      expect(current.tooltip[0].textStyle.color).toBe(dark ? '#cbd5e1' : '#334155');
      expect(current.dataZoom[0].textStyle.color).toBe(dark ? '#94a3b8' : '#64748b');
      expect(current.dataZoom[0].handleStyle.borderColor).toBe(dark ? '#94a3b8' : '#64748b');
      expect(current.series[0].lineStyle.color).toBe('#f43f5e');
      expect(current.series[0].data).toEqual(data);
      expect(current.tooltip[0].formatter([{axisValue: 1, seriesName: '损失', value: [1, 0.3]}])).toBe('步骤: 1\n损失: 0.3');
    };
    assertTheme(true);
    chart.dispatchAction({type: 'dataZoom', start: 20, end: 70});
    chart.dispatchAction({type: 'legendUnSelect', name: '损失'});
    await act(async () => { document.documentElement.classList.remove('dark'); });
    await waitFor(() => assertTheme(false));
    expect(read().dataZoom[0]).toMatchObject({start: 20, end: 70});
    expect(read().legend[0].selected).toEqual({'损失': false});

    await act(async () => { document.documentElement.classList.add('dark'); });
    await waitFor(() => assertTheme(true));
    expect(read().dataZoom[0]).toMatchObject({start: 20, end: 70});
    expect(read().legend[0].selected).toEqual({'损失': false});
    rerender(<EChart option={{...option, grid: {...base.grid, right: 30}}}/>);
    assertTheme(true);
    expect(init).toHaveBeenCalledTimes(1);
    const setTheme = vi.spyOn(chart, 'setTheme');
    await act(async () => { document.documentElement.classList.add('unrelated'); });
    expect(setTheme).not.toHaveBeenCalled();
    unmount();
    expect(chart.isDisposed()).toBe(true);
    await act(async () => { document.documentElement.classList.remove('dark'); });
    expect(setTheme).not.toHaveBeenCalled();
  });
});
