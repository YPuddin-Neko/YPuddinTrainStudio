import type { ECharts } from 'echarts/core';

/** Canvas and rich-text tooltips need resolved colors, not CSS variable strings. */
export function readChartTheme(root = document.documentElement) {
  const dark = root.classList.contains('dark');
  const styles = getComputedStyle(root);
  const token = (name: string, light: string, night: string) => styles.getPropertyValue(name).trim() || (dark ? night : light);
  const text = token('--studio-text', '#334155', '#cbd5e1');
  const dim = token('--studio-dim', '#64748b', '#94a3b8');
  const panel = token('--studio-panel', '#ffffff', '#0f172a');
  const muted = token('--studio-muted', '#f8fafc', '#111c30');
  const accent = token('--studio-accent', '#2563eb', '#60a5fa');
  const gridLine = dark ? '#475569' : '#cbd5e1';
  const axis = {
    axisLabel: {color: dim},
    nameTextStyle: {color: text},
    axisLine: {lineStyle: {color: dim}},
    axisTick: {lineStyle: {color: dim}},
    splitLine: {lineStyle: {color: gridLine}},
    minorSplitLine: {lineStyle: {color: gridLine, opacity: 0.5}},
    splitArea: {areaStyle: {color: [panel, muted]}},
  };
  return {
    // Do not set `color` or series styles: keep each metric's existing palette.
    darkMode: dark,
    backgroundColor: 'transparent',
    textStyle: {color: text},
    title: {textStyle: {color: text}, subtextStyle: {color: dim}},
    categoryAxis: axis, valueAxis: axis, timeAxis: axis, logAxis: axis,
    axisPointer: {
      lineStyle: {color: dim}, crossStyle: {color: dim},
      label: {color: text, backgroundColor: panel, borderColor: gridLine},
    },
    legend: {
      textStyle: {color: text}, inactiveColor: dim,
      pageTextStyle: {color: dim}, pageIconColor: accent, pageIconInactiveColor: dim,
    },
    tooltip: {backgroundColor: panel, borderColor: gridLine, textStyle: {color: text}},
    dataZoom: {
      backgroundColor: muted, borderColor: gridLine, textStyle: {color: dim},
      fillerColor: dark ? 'rgba(96,165,250,0.20)' : 'rgba(37,99,235,0.14)',
      handleStyle: {color: panel, borderColor: dim},
      moveHandleStyle: {color: dim},
      dataBackground: {lineStyle: {color: dim}, areaStyle: {color: dim, opacity: 0.2}},
      selectedDataBackground: {lineStyle: {color: accent}, areaStyle: {color: accent, opacity: 0.2}},
      emphasis: {handleStyle: {color: panel, borderColor: accent}, moveHandleStyle: {color: accent}},
    },
  };
}

/** ECharts 6 setTheme recreates its model, so retain user zoom/legend choices. */
export function updateChartTheme(chart: ECharts, theme: ReturnType<typeof readChartTheme>) {
  const current = chart.getOption();
  const zoom = (current.dataZoom as Array<Record<string, unknown>> | undefined)?.map(item => ({
    ...(item.id ? {id: item.id} : {}), start: item.start, end: item.end,
  }));
  const legend = (current.legend as Array<Record<string, unknown>> | undefined)?.map(item => ({
    ...(item.id ? {id: item.id} : {}), selected: item.selected, scrollDataIndex: item.scrollDataIndex,
  }));
  chart.setTheme(theme, {silent: true});
  if (zoom?.length || legend?.length) chart.setOption({
    ...(zoom?.length ? {dataZoom: zoom} : {}),
    ...(legend?.length ? {legend} : {}),
  }, {silent: true});
}
