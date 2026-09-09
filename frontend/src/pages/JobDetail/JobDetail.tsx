import React from 'react';
import { useParams } from 'react-router-dom';
import ReactECharts from 'echarts-for-react';
import { apiClient } from '../../api/client';
import { Job, JobMetrics, JobSample, JobCheckpoint, JobLogLine } from '../../api/types';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import {
  Activity,
  Layers,
  Image as ImageIcon,
  Download,
  Terminal,
  Code,
  CheckCircle2,
} from 'lucide-react';

export default function JobDetail() {
  const { id } = useParams<{ id: string }>();

  const [job, setJob] = React.useState<Job | null>(null);
  const [metrics, setMetrics] = React.useState<JobMetrics | null>(null);
  const [samples, setSamples] = React.useState<JobSample[]>([]);
  const [checkpoints, setCheckpoints] = React.useState<JobCheckpoint[]>([]);
  const [logs, setLogs] = React.useState<JobLogLine[]>([]);
  const [configSnapshot, setConfigSnapshot] = React.useState<any>(null);

  const [activeTab, setActiveTab] = React.useState<'metrics' | 'samples' | 'checkpoints' | 'logs' | 'config'>('metrics');
  const [xAxisMode, setXAxisMode] = React.useState<'step' | 'epoch'>('step');
  const [emaAlpha, setEmaAlpha] = React.useState<number>(0.9);
  const [logFilter, setLogFilter] = React.useState<string>('all');
  const [autoScrollLog, setAutoScrollLog] = React.useState<boolean>(true);

  const logContainerRef = React.useRef<HTMLDivElement>(null);

  // 1. 初始化数据加载
  React.useEffect(() => {
    if (!id) return;
    apiClient.get<Job>(`/jobs/${id}`).then(setJob).catch(console.error);
    apiClient.get<JobMetrics>(`/jobs/${id}/metrics`).then(setMetrics).catch(console.error);
    apiClient.get<JobSample[]>(`/jobs/${id}/samples`).then(setSamples).catch(console.error);
    apiClient.get<JobCheckpoint[]>(`/jobs/${id}/checkpoints`).then(setCheckpoints).catch(console.error);
    apiClient.get<{ lines: JobLogLine[] }>(`/jobs/${id}/log`).then((r) => setLogs(r.lines || [])).catch(console.error);
    apiClient.get<any>(`/jobs/${id}/config`).then(setConfigSnapshot).catch(console.error);
  }, [id]);

  // 2. SSE 增量监听
  useEventStream(EVENT_TYPES.JOB_STATE, (data: any) => {
    if (data.job_id === id) {
      setJob((prev) => (prev ? { ...prev, status: data.status, progress: data.progress || prev.progress } : null));
    }
  });

  useEventStream(EVENT_TYPES.JOB_STEP, (data: any) => {
    if (data.job_id === id) {
      setMetrics((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          steps: [...prev.steps, data.step],
          loss: [...prev.loss, data.loss],
          loss_ema: [...prev.loss_ema, data.loss_ema],
          grad_norm: [...prev.grad_norm, data.grad_norm || 0],
          vram_mb: [...prev.vram_mb, data.vram_mb || 0],
          it_s: [...prev.it_s, data.it_s || 0],
        };
      });
    }
  });

  useEventStream(EVENT_TYPES.JOB_SAMPLE, (sample: JobSample) => {
    setSamples((prev) => [...prev, sample]);
  });

  useEventStream(EVENT_TYPES.JOB_LOG, (data: any) => {
    if (data.job_id === id && data.lines) {
      setLogs((prev) => [...prev, ...data.lines]);
    }
  });

  // 日志自动触底
  React.useEffect(() => {
    if (autoScrollLog && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs, autoScrollLog]);

  // ECharts 图表配置 (原始+EMA+多指标)
  const chartOption = React.useMemo(() => {
    if (!metrics || metrics.steps.length === 0) return {};

    const steps = metrics.steps;
    return {
      tooltip: { trigger: 'axis' },
      legend: { data: ['Loss', 'Loss (EMA)', 'Grad Norm', 'VRAM (GB)', 'Speed (it/s)'], textStyle: { color: '#888' } },
      grid: { left: '3%', right: '4%', bottom: '15%', containLabel: true },
      xAxis: {
        type: 'category',
        data: steps.map((s) => (xAxisMode === 'step' ? `Step ${s}` : `Epoch ${(s / 500).toFixed(1)}`)),
        splitLine: { show: false },
      },
      yAxis: [
        { type: 'value', name: 'Loss', scale: true, splitLine: { lineStyle: { color: '#33333320' } } },
        { type: 'value', name: 'VRAM / Speed', scale: true, splitLine: { show: false } },
      ],
      dataZoom: [
        { type: 'inside', start: 0, end: 100 },
        { type: 'slider', start: 0, end: 100 },
      ],
      series: [
        {
          name: 'Loss',
          type: 'line',
          showSymbol: false,
          sampling: 'lttb',
          data: metrics.loss,
          lineStyle: { width: 1.2, color: '#93c5fd' },
        },
        {
          name: 'Loss (EMA)',
          type: 'line',
          showSymbol: false,
          sampling: 'lttb',
          data: metrics.loss_ema,
          lineStyle: { width: 2, color: '#2563eb' },
        },
        {
          name: 'Grad Norm',
          type: 'line',
          showSymbol: false,
          sampling: 'lttb',
          data: metrics.grad_norm,
          lineStyle: { width: 1, color: '#eab308' },
        },
        {
          name: 'VRAM (GB)',
          type: 'line',
          yAxisIndex: 1,
          showSymbol: false,
          sampling: 'lttb',
          data: metrics.vram_mb.map((mb) => (mb / 1024).toFixed(1)),
          lineStyle: { width: 1, color: '#ec4899' },
        },
      ],
    };
  }, [metrics, xAxisMode]);

  const filteredLogs = logs.filter((l) => logFilter === 'all' || l.level === logFilter);

  // 阶段时间线定义
  const phases = ['preparing', 'caching', 'training', 'finalizing'];
  const currentPhaseIndex = phases.indexOf(job?.progress?.phase || 'preparing');

  return (
    <div className="space-y-6" data-testid="job-detail-page">
      {/* 1. 头部指标与阶段时间线 */}
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700 space-y-6">
        <div className="flex flex-wrap justify-between items-center gap-4">
          <div>
            <div className="flex items-center space-x-3">
              <h2 className="text-2xl font-bold">{job?.name || `Job #${id}`}</h2>
              <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${
                job?.status === 'running' ? 'bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300' : 'bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-300'
              }`}>
                {job?.status || 'unknown'}
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1 font-mono">{id}</p>
          </div>
          <div className="flex items-center space-x-6 text-sm">
            <div>
              <span className="text-xs text-slate-400 block">Progress</span>
              <span className="font-semibold">{job?.progress?.step || 0} / {job?.progress?.total_steps || 0}</span>
            </div>
            <div>
              <span className="text-xs text-slate-400 block">Speed</span>
              <span className="font-semibold">{job?.progress?.it_s || 0} it/s</span>
            </div>
            <div>
              <span className="text-xs text-slate-400 block">VRAM Peak</span>
              <span className="font-semibold">{Math.round((job?.progress?.vram_peak_mb || 0) / 1024)} GB</span>
            </div>
            <div>
              <span className="text-xs text-slate-400 block">ETA</span>
              <span className="font-semibold">{job?.progress?.eta_s ? `${Math.round(job.progress.eta_s / 60)}m` : '--'}</span>
            </div>
          </div>
        </div>

        {/* 阶段时间线 */}
        <div className="flex items-center space-x-2 pt-2 border-t dark:border-slate-700">
          {phases.map((ph, idx) => {
            const isDone = idx < currentPhaseIndex;
            const isCurrent = idx === currentPhaseIndex;
            return (
              <React.Fragment key={ph}>
                <div className={`flex items-center space-x-1.5 text-xs font-medium px-3 py-1.5 rounded-md ${
                  isCurrent ? 'bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300' : isDone ? 'text-green-600 dark:text-green-400' : 'text-slate-400'
                }`}>
                  {isDone ? <CheckCircle2 className="w-3.5 h-3.5" /> : <span>{idx + 1}.</span>}
                  <span className="capitalize">{ph}</span>
                </div>
                {idx < phases.length - 1 && <div className="flex-1 h-0.5 bg-slate-200 dark:bg-slate-700 mx-1" />}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* 2. Tabs 切换导航 */}
      <div className="border-b border-slate-200 dark:border-slate-700">
        <nav className="flex space-x-6 text-sm font-medium">
          <button
            onClick={() => setActiveTab('metrics')}
            className={`flex items-center space-x-2 py-3 border-b-2 ${activeTab === 'metrics' ? 'border-blue-500 text-blue-600 dark:text-blue-400' : 'border-transparent text-slate-500 hover:text-slate-700'}`}
          >
            <Activity className="w-4 h-4" />
            <span>Metrics & Charts</span>
          </button>
          <button
            onClick={() => setActiveTab('samples')}
            className={`flex items-center space-x-2 py-3 border-b-2 ${activeTab === 'samples' ? 'border-blue-500 text-blue-600 dark:text-blue-400' : 'border-transparent text-slate-500 hover:text-slate-700'}`}
          >
            <ImageIcon className="w-4 h-4" />
            <span>Samples ({samples.length})</span>
          </button>
          <button
            onClick={() => setActiveTab('checkpoints')}
            className={`flex items-center space-x-2 py-3 border-b-2 ${activeTab === 'checkpoints' ? 'border-blue-500 text-blue-600 dark:text-blue-400' : 'border-transparent text-slate-500 hover:text-slate-700'}`}
          >
            <Layers className="w-4 h-4" />
            <span>Checkpoints ({checkpoints.length})</span>
          </button>
          <button
            onClick={() => setActiveTab('logs')}
            className={`flex items-center space-x-2 py-3 border-b-2 ${activeTab === 'logs' ? 'border-blue-500 text-blue-600 dark:text-blue-400' : 'border-transparent text-slate-500 hover:text-slate-700'}`}
          >
            <Terminal className="w-4 h-4" />
            <span>Logs</span>
          </button>
          <button
            onClick={() => setActiveTab('config')}
            className={`flex items-center space-x-2 py-3 border-b-2 ${activeTab === 'config' ? 'border-blue-500 text-blue-600 dark:text-blue-400' : 'border-transparent text-slate-500 hover:text-slate-700'}`}
          >
            <Code className="w-4 h-4" />
            <span>Config Snapshot</span>
          </button>
        </nav>
      </div>

      {/* 3. 详细内容区域 */}
      {activeTab === 'metrics' && (
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6 space-y-4">
          <div className="flex justify-between items-center">
            <div className="flex space-x-2 text-xs">
              <button
                onClick={() => setXAxisMode('step')}
                className={`px-2.5 py-1 rounded ${xAxisMode === 'step' ? 'bg-blue-600 text-white' : 'bg-slate-100 dark:bg-slate-700'}`}
              >
                Step
              </button>
              <button
                onClick={() => setXAxisMode('epoch')}
                className={`px-2.5 py-1 rounded ${xAxisMode === 'epoch' ? 'bg-blue-600 text-white' : 'bg-slate-100 dark:bg-slate-700'}`}
              >
                Epoch
              </button>
            </div>
            <div className="flex items-center space-x-2 text-xs text-slate-500">
              <span>EMA Alpha:</span>
              <input
                type="range"
                min="0.1"
                max="0.99"
                step="0.05"
                value={emaAlpha}
                onChange={(e) => setEmaAlpha(Number(e.target.value))}
              />
              <span>{emaAlpha}</span>
            </div>
          </div>
          <ReactECharts option={chartOption} style={{ height: 420 }} notMerge={false} lazyUpdate={true} />
        </div>
      )}

      {activeTab === 'samples' && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4" data-testid="samples-gallery">
          {samples.map((s, idx) => (
            <div key={idx} className="bg-white dark:bg-slate-800 rounded-xl overflow-hidden border border-slate-200 dark:border-slate-700">
              <img src={s.url} alt={s.prompt} className="w-full h-56 object-cover bg-slate-100 dark:bg-slate-900" />
              <div className="p-3 space-y-1 text-xs">
                <div className="flex justify-between font-semibold">
                  <span>Step {s.step}</span>
                  <span className="text-slate-400">Seed {s.seed}</span>
                </div>
                <p className="text-slate-500 line-clamp-2">{s.prompt}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {activeTab === 'checkpoints' && (
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-50 dark:bg-slate-800/50 text-xs text-slate-400 border-b dark:border-slate-700">
              <tr>
                <th className="p-4">Step</th>
                <th className="p-4">Kind</th>
                <th className="p-4">Path</th>
                <th className="p-4">Size</th>
                <th className="p-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
              {checkpoints.map((cp, idx) => (
                <tr key={idx} className="hover:bg-slate-50 dark:hover:bg-slate-750">
                  <td className="p-4 font-semibold">{cp.step}</td>
                  <td className="p-4 capitalize">{cp.kind}</td>
                  <td className="p-4 font-mono text-xs text-slate-500">{cp.path}</td>
                  <td className="p-4">{Math.round(cp.size / (1024 * 1024))} MB</td>
                  <td className="p-4 text-right space-x-2">
                    <button className="px-2.5 py-1 text-xs bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 rounded inline-flex items-center space-x-1">
                      <Download className="w-3.5 h-3.5" />
                      <span>Download</span>
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {activeTab === 'logs' && (
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4 space-y-3">
          <div className="flex justify-between items-center text-xs">
            <div className="flex space-x-2">
              {['all', 'info', 'warn', 'error'].map((lvl) => (
                <button
                  key={lvl}
                  onClick={() => setLogFilter(lvl)}
                  className={`px-2.5 py-1 rounded capitalize ${logFilter === lvl ? 'bg-blue-600 text-white' : 'bg-slate-100 dark:bg-slate-700'}`}
                >
                  {lvl}
                </button>
              ))}
            </div>
            <label className="flex items-center space-x-1 cursor-pointer">
              <input
                type="checkbox"
                checked={autoScrollLog}
                onChange={(e) => setAutoScrollLog(e.target.checked)}
                className="rounded"
              />
              <span>Follow Bottom</span>
            </label>
          </div>
          <div
            ref={logContainerRef}
            className="h-80 overflow-y-auto bg-slate-900 text-slate-200 font-mono text-xs p-3 rounded-lg space-y-1"
          >
            {filteredLogs.map((l, idx) => (
              <div key={idx} className="flex space-x-2">
                <span className="text-slate-500">[{l.ts}]</span>
                <span className={`uppercase font-bold ${l.level === 'warn' ? 'text-yellow-400' : l.level === 'error' ? 'text-red-400' : 'text-blue-400'}`}>
                  [{l.level}]
                </span>
                <span className="flex-1 break-all">{l.msg}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === 'config' && (
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6">
          <pre className="text-xs font-mono bg-slate-50 dark:bg-slate-900 p-4 rounded-lg overflow-x-auto">
            {JSON.stringify(configSnapshot, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
