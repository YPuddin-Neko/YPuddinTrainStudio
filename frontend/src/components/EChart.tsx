import React from 'react';
import * as echarts from 'echarts/core';
import { LineChart } from 'echarts/charts';
import {
  DataZoomInsideComponent,
  DataZoomSliderComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import type { EChartsCoreOption } from 'echarts/core';

// 按需注册：只打包用得到的模块（完整 echarts 包约 1 MB，这里约 1/3）
echarts.use([
  LineChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomInsideComponent,
  DataZoomSliderComponent,
  CanvasRenderer,
]);

interface EChartProps {
  option: EChartsCoreOption;
  style?: React.CSSProperties;
}

/** 轻量 ECharts 封装：init / setOption / resize / dispose（替代 echarts-for-react，支持 echarts 6 按需引入） */
export function EChart({ option, style }: EChartProps) {
  const ref = React.useRef<HTMLDivElement>(null);
  const chartRef = React.useRef<echarts.ECharts | null>(null);

  React.useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const chart = echarts.init(el);
    chartRef.current = chart;
    // jsdom（测试环境）没有 ResizeObserver，退化为 window resize 监听
    const onResize = () => chart.resize();
    let observer: ResizeObserver | null = null;
    if (typeof ResizeObserver !== 'undefined') {
      observer = new ResizeObserver(onResize);
      observer.observe(el);
    } else {
      window.addEventListener('resize', onResize);
    }
    return () => {
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
