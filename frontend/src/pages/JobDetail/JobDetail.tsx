import { useParams } from 'react-router-dom';
import ReactECharts from 'echarts-for-react';
import React from 'react';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';

export default function JobDetail() {
  const { id } = useParams<{ id: string }>();
  const [lossData, setLossData] = React.useState<[number, number][]>(() => {
    // 初始模拟一些数据点
    const initial: [number, number][] = [];
    for (let i = 0; i < 500; i++) {
      initial.push([i, Math.exp(-i / 200) + Math.random() * 0.02 + 0.05]);
    }
    return initial;
  });

  // 监听增量 step 事件更新实时曲线
  useEventStream(EVENT_TYPES.JOB_STEP, (event: any) => {
    if (event.job_id === id && event.step !== undefined && event.loss !== undefined) {
      setLossData((prev) => [...prev, [event.step, event.loss]]);
    }
  });

  const chartOption = {
    title: {
      text: 'Training Loss (LTTB Optimized)',
      textStyle: { color: '#888', fontSize: 14 },
    },
    tooltip: {
      trigger: 'axis',
    },
    xAxis: {
      type: 'value',
      name: 'Step',
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      name: 'Loss',
      scale: true,
      splitLine: { lineStyle: { color: '#33333320' } },
    },
    dataZoom: [
      { type: 'inside', start: 0, end: 100 },
      { type: 'slider', start: 0, end: 100 },
    ],
    series: [
      {
        name: 'Loss',
        type: 'line',
        showSymbol: false,
        sampling: 'lttb', // 采用 LTTB 下采样支持十万点流畅渲染
        data: lossData,
        lineStyle: { width: 1.5, color: '#3b82f6' },
      },
    ],
    grid: {
      left: '3%',
      right: '4%',
      bottom: '15%',
      containLabel: true,
    },
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold">Job Monitor</h2>
          <p className="text-slate-500">Viewing Job: {id}</p>
        </div>
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6">
        <ReactECharts option={chartOption} style={{ height: 400 }} notMerge={false} lazyUpdate={true} />
      </div>
    </div>
  );
}
