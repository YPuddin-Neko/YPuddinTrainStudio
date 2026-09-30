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
import { prefersReducedMotion } from '../utils/motion';

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

type MediaRule = { query?: { maxWidth?: number }; option: Record<string, unknown> };
const isRecord = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);

/** A rule's settings over the option: arrays entry by entry (one per axis), objects key by key. */
function mergeRule(base: Record<string, unknown>, rule: Record<string, unknown>): Record<string, unknown> {
  const merged = { ...base };
  for (const [key, value] of Object.entries(rule)) {
    const current = merged[key];
    merged[key] = Array.isArray(current) && Array.isArray(value) ? current.map((item, index) => ({ ...item, ...value[index] }))
      : isRecord(current) && isRecord(value) ? { ...current, ...value } : value;
  }
  return merged;
}

/**
 * ECharts applies `media` rules with the `replaceMerge` of `setOption`, which empties every listed type a rule leaves
 * out: a narrow chart lost all its series. Apply `maxWidth` rules against the chart's width here instead.
 */
function resolveMedia(option: EChartsCoreOption, width: number): { option: EChartsCoreOption; matched: string } {
  const { media, ...base } = option as EChartsCoreOption & { media?: MediaRule[] };
  if (!Array.isArray(media)) return { option, matched: '' };
  let resolved: Record<string, unknown> = base;
  let matched = '';
  media.forEach((rule, index) => {
    const maxWidth = rule.query?.maxWidth;
    if (maxWidth != null && width <= maxWidth) { resolved = mergeRule(resolved, rule.option); matched += `${index} `; }
  });
  return { option: resolved as EChartsCoreOption, matched };
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
  const matchedMedia = React.useRef<string | null>(null);

  // Series and value axes are replaced, not merged by name: a chart that loses series (a new layout) must not keep
  // the old ones. Other components merge, so zoom and legend state survive live updates.
  const apply = React.useCallback((always: boolean) => {
    const chart = chartRef.current, option = latest.current;
    if (!chart || !option || !ref.current) return;
    const resolved = resolveMedia(option, ref.current.clientWidth);
    if (!always && resolved.matched === matchedMedia.current) return;
    matchedMedia.current = resolved.matched;
    restoreTip.current = !!pointer.current;
    chart.setOption({ ...resolved.option, animation: prefersReducedMotion() ? false : resolved.option.animation ?? true }, { replaceMerge: ['series', 'yAxis'], lazyUpdate: true });
  }, []);

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
    // Redrawing removes temporary markers but leaves the axis pointer's highlight cache intact.
    // Reset that cache, then let ECharts find the displayed points after sampling, zoom and legend filtering.
    const rendered = () => {
      if (!restoreTip.current) return;
      restoreTip.current = false;
      if (!pointer.current) return;
      const { x, y } = pointer.current;
      chart.dispatchAction({ type: 'updateAxisPointer', currTrigger: 'leave' });
      chart.dispatchAction({ type: 'showTip', x, y });
    };
    chart.on('rendered', rendered);
    const motionPreference = window.matchMedia?.('(prefers-reduced-motion: reduce)');
    const updateMotion = () => apply(true);
    motionPreference?.addEventListener?.('change', updateMotion);
    const themeObserver = new MutationObserver(() => {
      const nextDark = root.classList.contains('dark');
      if (nextDark === dark) return;
      dark = nextDark;
      updateChartTheme(chart, readChartTheme(root));
    });
    themeObserver.observe(root, {attributes: true, attributeFilter: ['class']});
    // 没有 ResizeObserver 时使用窗口 resize 事件。
    const onResize = () => { chart.resize(); apply(false); };
    let observer: ResizeObserver | null = null;
    if (typeof ResizeObserver !== 'undefined') {
      observer = new ResizeObserver(onResize);
      observer.observe(el);
    } else {
      window.addEventListener('resize', onResize);
    }
    return () => {
      themeObserver.disconnect();
      motionPreference?.removeEventListener?.('change', updateMotion);
      observer?.disconnect();
      window.removeEventListener('resize', onResize);
      chart.dispose();
      chartRef.current = null;
    };
  }, [apply]);

  React.useEffect(() => {
    latest.current = option;
    apply(true);
  }, [option, apply]);

  return <div ref={ref} style={style} />;
}
