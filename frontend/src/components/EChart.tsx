import React from 'react';
import * as echarts from 'echarts/core';
import { LineChart } from 'echarts/charts';
import {
  DataZoomSliderComponent,
  GridComponent,
  LegendComponent,
  LegendScrollComponent,
  TooltipComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import type { EChartsCoreOption } from 'echarts/core';
import { readChartTheme, updateChartTheme } from './chartTheme';

// 按需注册，避免打包未使用的图表模块。
echarts.use([
  LineChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  LegendScrollComponent,
  DataZoomSliderComponent,
  CanvasRenderer,
]);

/** Index of the point in [x, y] data closest to x; -1 without data. */
function nearestIndex(data: unknown, x: number): number {
  if (!Array.isArray(data) || !data.length || !Number.isFinite(x)) return -1;
  const at = (index: number) => { const point = data[index]; return Array.isArray(point) ? Number(point[0]) : index; };
  let low = 0, high = data.length - 1;
  while (low < high) { const middle = (low + high) >> 1; if (at(middle) < x) low = middle + 1; else high = middle; }
  return low > 0 && Math.abs(at(low - 1) - x) <= Math.abs(at(low) - x) ? low - 1 : low;
}

interface EChartProps {
  option: EChartsCoreOption;
  style?: React.CSSProperties;
}

export function EChart({ option, style }: EChartProps) {
  const ref = React.useRef<HTMLDivElement>(null);
  const chartRef = React.useRef<echarts.ECharts | null>(null);
  // Where the pointer rests over the chart, so a live update can show the hovered points again.
  const pointer = React.useRef<{ x: number; y: number } | null>(null);
  const restoreTip = React.useRef(false);
  const latest = React.useRef<EChartsCoreOption | null>(null);

  React.useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const root = document.documentElement;
    const theme = readChartTheme(root);
    let dark = theme.darkMode;
    const chart = echarts.init(el, theme);
    chartRef.current = chart;
    const zr = chart.getZr();
    const move = (event: { offsetX: number; offsetY: number }) => { pointer.current = { x: event.offsetX, y: event.offsetY }; };
    const leave = () => { pointer.current = null; };
    zr.on('mousemove', move);
    zr.on('globalout', leave);
    // Redrawn series drop the markers of the hovered step, while the axis pointer still counts that step as
    // highlighted and so never draws them again. Refresh the tip and highlight each line's point under the pointer.
    const rendered = () => {
      if (!restoreTip.current) return;
      restoreTip.current = false;
      if (!pointer.current) return;
      const { x, y } = pointer.current;
      chart.dispatchAction({ type: 'showTip', x, y });
      const series = (latest.current as { series?: Array<{ data?: unknown }> } | null)?.series;
      if (!Array.isArray(series)) return;
      const [value] = chart.convertFromPixel({ gridIndex: 0 }, [x, y]) as unknown as number[];
      const batch = series.map((item, seriesIndex) => ({ seriesIndex, dataIndex: nearestIndex(item.data, value) })).filter(item => item.dataIndex >= 0);
      if (batch.length) chart.dispatchAction({ type: 'highlight', batch, notBlur: true });
    };
    chart.on('rendered', rendered);
    const themeObserver = new MutationObserver(() => {
      const nextDark = root.classList.contains('dark');
      if (nextDark === dark) return;
      dark = nextDark;
      updateChartTheme(chart, readChartTheme(root));
    });
    themeObserver.observe(root, {attributes: true, attributeFilter: ['class']});
    // 没有 ResizeObserver 时使用窗口 resize 事件。
    const onResize = () => chart.resize();
    let observer: ResizeObserver | null = null;
    if (typeof ResizeObserver !== 'undefined') {
      observer = new ResizeObserver(onResize);
      observer.observe(el);
    } else {
      window.addEventListener('resize', onResize);
    }
    return () => {
      themeObserver.disconnect();
      observer?.disconnect();
      window.removeEventListener('resize', onResize);
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  React.useEffect(() => {
    // Series and value axes are replaced, not merged by name: a chart that loses series (a new layout) must not keep
    // the old ones. Other components merge, so zoom and legend state survive live updates.
    restoreTip.current = !!pointer.current;
    latest.current = option;
    chartRef.current?.setOption(option, { replaceMerge: ['series', 'yAxis'], lazyUpdate: true });
  }, [option]);

  return <div ref={ref} style={style} />;
}
