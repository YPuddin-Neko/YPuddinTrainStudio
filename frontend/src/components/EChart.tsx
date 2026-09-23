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

interface EChartProps {
  option: EChartsCoreOption;
  style?: React.CSSProperties;
}

export function EChart({ option, style }: EChartProps) {
  const ref = React.useRef<HTMLDivElement>(null);
  const chartRef = React.useRef<echarts.ECharts | null>(null);

  React.useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const root = document.documentElement;
    const theme = readChartTheme(root);
    let dark = theme.darkMode;
    const chart = echarts.init(el, theme);
    chartRef.current = chart;
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
    chartRef.current?.setOption(option, { notMerge: false, lazyUpdate: true });
  }, [option]);

  return <div ref={ref} style={style} />;
}
