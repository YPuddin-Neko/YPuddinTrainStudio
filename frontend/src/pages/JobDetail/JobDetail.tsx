import { mergeJobEvent } from '../../utils/jobs';
import React from 'react';
import { Link, useParams, useNavigate, useSearchParams, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { EChart } from '../../components/EChart';
import { apiClient, apiUrl } from '../../api/client';
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
  Loader2,
} from 'lucide-react';
import { shapeValidationSeries, mergeValidationPoint, appendCapped, smoothLoss, appendMetricStep } from '../../utils/metrics';
import { formatBytes, formatBytesMB, formatEta, formatTime } from '../../utils/format';
import { formatApiError } from '../../utils/errors';
import { projectUrl, type ProjectVersion } from '../../utils/projectVersions';
import { useWorkspaceText } from '../../utils/workspaceText';
import StudioSelect from '../../components/StudioSelect';
import { JobActions } from '../Queue/jobPresentation';
import '../Queue/queue.css';
import './job-detail.css';

type VersionedJob = Job & { version_id?: string | null };

const LR_COLORS = ['#a78bfa', '#f59e0b', '#22d3ee', '#fb7185', '#84cc16', '#e879f9'];

const PHASE_KEYS = ['preparing', 'caching', 'training', 'finalizing'];
const LOG_LEVELS = ['all', 'info', 'warn', 'error', 'debug'];

function mergeSamples(previous: JobSample[], incoming: JobSample[]): JobSample[] {
  // Step and epoch triggers may share step/seed while saving different image files.
  const samples = new Map(previous.map(sample => [sample.url, sample]));
  for (const sample of incoming) samples.set(sample.url, { ...samples.get(sample.url), ...sample });
  return [...samples.values()].sort((a, b) => a.step - b.step || a.created_at - b.created_at || a.prompt_index - b.prompt_index);
}

/** 大数字指标卡：label 小写 xs + 值 2xl font-mono */
function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="p-3 bg-slate-50 dark:bg-slate-900 rounded-lg min-w-0">
      <div className="text-xs lowercase text-slate-400 truncate">{label}</div>
      <div className="text-2xl font-bold font-mono mt-1 truncate">{value}</div>
    </div>
  );
}

/** 空态：lucide 图标 + 标题 + 一行提示 */
function EmptyState({ icon: Icon, title, hint }: { icon: React.ComponentType<{ className?: string }>; title: string; hint: string }) {
  return (
    <div className="col-span-full flex flex-col items-center justify-center py-14 text-center">
      <Icon className="w-8 h-8 text-slate-300 dark:text-slate-600" />
      <p className="mt-3 text-sm font-medium text-slate-500 dark:text-slate-400">{title}</p>
      <p className="mt-1 text-xs text-slate-400 dark:text-slate-500">{hint}</p>
    </div>
  );
}

/** 日志级别颜色：error 红 / warn 黄 / info 蓝 / debug 弱化 */
function logLevelColor(level: string): string {
  if (level === 'error') return 'text-red-400';
  if (level === 'warn') return 'text-yellow-400';
  if (level === 'debug') return 'text-slate-500';
  return 'text-blue-400';
}

export default function JobDetail() {
  const { id } = useParams<{ id: string }>();
  const { t } = useTranslation();
  const text = useWorkspaceText();
  const navigate = useNavigate();
  const location = useLocation();
  const queueReturnTo = typeof location.state?.queueReturnTo === 'string' && /^\/queue(?:\?|$)/.test(location.state.queueReturnTo) ? location.state.queueReturnTo : '/queue';
  const [params, setParams] = useSearchParams();
  const [dataError, setDataError] = React.useState('');
  const [actionError, setActionError] = React.useState('');
  const [resuming, setResuming] = React.useState(false);

  const [job, setJob] = React.useState<VersionedJob | null>(null);
  const [resolvedVersion, setResolvedVersion] = React.useState<{ projectId: string; versionId: string; name: string } | null>(null);
  const [metrics, setMetrics] = React.useState<JobMetrics | null>(null);
  const [samples, setSamples] = React.useState<JobSample[]>([]);
  const [checkpoints, setCheckpoints] = React.useState<JobCheckpoint[]>([]);
  const [logs, setLogs] = React.useState<JobLogLine[]>([]);
  const [configSnapshot, setConfigSnapshot] = React.useState<any>(null);
  const [sampleProgress, setSampleProgress] = React.useState<{ step: number; promptIndex: number; prompts: number; done: number; total: number } | null>(null);

  const activeTab = ['metrics', 'samples', 'checkpoints', 'logs', 'config'].includes(params.get('tab') || '') ? params.get('tab')! : 'metrics';
  const setActiveTab = (tab: string) => { const next = new URLSearchParams(params); next.set('tab', tab); setParams(next, { state: location.state }); };
  const [xAxisMode, setXAxisMode] = React.useState<'step' | 'epoch'>('step');
  const [emaAlpha, setEmaAlpha] = React.useState<number>(0.9);
  const [logFilter, setLogFilter] = React.useState<string>('all');
  const [autoScrollLog, setAutoScrollLog] = React.useState<boolean>(true);
  const [logMode, setLogMode] = React.useState<'live' | 'history'>('live');
  const [logOffsets, setLogOffsets] = React.useState([0]);
  const [nextLogOffset, setNextLogOffset] = React.useState(0);
  const [hasMoreLogs, setHasMoreLogs] = React.useState(false);
  const [logLoading, setLogLoading] = React.useState(false);
  const [logError, setLogError] = React.useState('');
  const [logQuery, setLogQuery] = React.useState('');
  const [samplePage, setSamplePage] = React.useState(1);
  const [sampleStep, setSampleStep] = React.useState('');
  const logRequest = React.useRef<AbortController | null>(null);
  const logOffset = logOffsets[logOffsets.length - 1];
  const fetchLogs = React.useCallback(async () => {
    if (!id) return;
    logRequest.current?.abort(); const controller = new AbortController(); logRequest.current = controller;
    setLogLoading(true); setLogError('');
    try {
      const response = await apiClient.get<{ lines: JobLogLine[]; next_offset: number; has_more?: boolean }>(`/jobs/${id}/log`, { params: { offset: logOffset, limit: 500, tail: logMode === 'live' }, signal: controller.signal, silent: true });
      if (!controller.signal.aborted) { setLogs(response.lines || []); setNextLogOffset(response.next_offset); setHasMoreLogs(response.has_more ?? response.lines.length >= 500); }
    } catch (error) { if (!controller.signal.aborted) setLogError(formatApiError(error)); }
    finally { if (!controller.signal.aborted) setLogLoading(false); }
  }, [id, logMode, logOffset]);
  React.useEffect(() => { void fetchLogs(); return () => logRequest.current?.abort(); }, [fetchLogs]);


  const logContainerRef = React.useRef<HTMLDivElement>(null);
  const samplesRequestRef = React.useRef<AbortController | null>(null);
  const refreshSamples = React.useCallback(async () => {
    if (!id) return;
    samplesRequestRef.current?.abort();
    const controller = new AbortController(); samplesRequestRef.current = controller;
    try {
      const history = await apiClient.get<JobSample[]>(`/jobs/${id}/samples`, { signal: controller.signal });
      if (!controller.signal.aborted) setSamples(previous => mergeSamples(previous, history));
    } catch (error) { if (!controller.signal.aborted) console.error(error); }
  }, [id]);

  // Reset route-specific state and ignore responses from a previous task.
  React.useEffect(() => {
    if (!id) return;
    const controller = new AbortController();
    const options = { signal: controller.signal };
    const ignoreAbort = (error: Error) => { if (!controller.signal.aborted) setDataError(formatApiError(error)); };
    setDataError(''); setSamplePage(1); setSampleStep(''); setLogOffsets([0]); setLogMode('live'); setJob(null); setMetrics(null); setSamples([]); setCheckpoints([]); setLogs([]); setConfigSnapshot(null); setSampleProgress(null);
    apiClient.get<VersionedJob>(`/jobs/${id}`, options).then(setJob).catch(ignoreAbort);
    apiClient.get<JobMetrics>(`/jobs/${id}/metrics`, options).then(setMetrics).catch(ignoreAbort);
    void refreshSamples();
    apiClient.get<JobCheckpoint[]>(`/jobs/${id}/checkpoints`, options).then(setCheckpoints).catch(ignoreAbort);
    apiClient.get<any>(`/jobs/${id}/config`, options).then(setConfigSnapshot).catch(ignoreAbort);
    return () => { controller.abort(); samplesRequestRef.current?.abort(); };
  }, [id, refreshSamples]);

  // Version names are optional context; missing legacy metadata must not block monitoring.
  React.useEffect(() => {
    const projectId = job?.project_id, versionId = job?.version_id;
    if (!projectId || !versionId) return;
    const controller = new AbortController();
    void apiClient.get<ProjectVersion[]>(`/projects/${encodeURIComponent(projectId)}/versions`, { params: { include_archived: true }, signal: controller.signal, silent: true })
      .then(versions => {
        if (controller.signal.aborted) return;
        const match = versions.find(version => version.id === versionId && version.project_id === projectId);
        setResolvedVersion({ projectId, versionId, name: match?.name.trim() || '' });
      }).catch(() => { /* The job and its results remain available without version metadata. */ });
    return () => controller.abort();
  }, [job?.project_id, job?.version_id]);
  const versionName = resolvedVersion?.projectId === job?.project_id && resolvedVersion?.versionId === job?.version_id ? resolvedVersion?.name : '';

  // 2. SSE 增量监听
  useEventStream(EVENT_TYPES.JOB_STATE, (data: any) => {
    if (data.job_id === id) {
      setJob((prev) => prev ? mergeJobEvent(prev, data) : null);
      if (['completed', 'failed', 'cancelled', 'paused'].includes(data.status)) void refreshSamples();
    }
  });

  useEventStream(EVENT_TYPES.JOB_STEP, (data: any) => {
    if (data.job_id === id) {
      setJob((prev) => prev ? mergeJobEvent(prev, data) : null);
      setMetrics((prev) => prev ? appendMetricStep(prev, data) : prev);
    }
  });

  useEventStream(EVENT_TYPES.JOB_PHASE, (data: any) => {
    if (data.job_id === id) setJob((prev) => prev ? mergeJobEvent(prev, data) : null);
  });
  useEventStream(EVENT_TYPES.JOB_CHECKPOINT, (data: any) => {
    if (data.job_id === id) apiClient.get<JobCheckpoint[]>(`/jobs/${id}/checkpoints`).then(setCheckpoints).catch(console.error);
  });

  useEventStream(EVENT_TYPES.JOB_SAMPLE, (sample: JobSample & { job_id?: string; ts?: number }) => {
    if (sample.job_id && sample.job_id !== id) return;
    setSamples(previous => mergeSamples(previous, [{ ...sample, created_at: sample.created_at ?? sample.ts ?? Date.now() / 1000 }]));
  });

  // 采样进度：让预览生成阶段有明确进度，不再像"卡死"
  useEventStream(EVENT_TYPES.JOB_SAMPLE_PROGRESS, (data: any) => {
    if (data.job_id !== id) return;
    if (data.done >= data.total && data.prompt_index + 1 >= (data.prompts || 1)) {
      // 最后一 prompt 完成 → 清除进度显示
      setSampleProgress(null);
    } else {
      setSampleProgress({
        step: data.step,
        promptIndex: data.prompt_index,
        prompts: data.prompts,
        done: data.done,
        total: data.total,
      });
    }
  });

  useEventStream(EVENT_TYPES.JOB_LOG, (data: any) => {
    if (data.job_id === id && data.lines && logMode === 'live') {
      // History is read from disk; only the current live window receives events.
      setLogs((prev) => appendCapped(prev, data.lines, 500));
    }
  });

  // validation 增量：按 step 去重合并
  useEventStream(EVENT_TYPES.JOB_VALIDATION, (data: any) => {
    if (data.job_id !== id) return;
    setMetrics((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        validation: mergeValidationPoint(prev.validation || [], {
          step: data.step,
          per_t: data.per_t || {},
          mean: data.mean,
        }),
      };
    });
  });

  // 日志自动触底
  React.useEffect(() => {
    if (autoScrollLog && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs, autoScrollLog, activeTab]);

  const stepLabel = t('job.step');
  const epochLabel = t('job.epoch');
  const stepsPerEpoch = job?.progress?.steps_per_epoch;
  const useEpoch = xAxisMode === 'epoch' && !!stepsPerEpoch;
  const xAxisName = useEpoch ? epochLabel : stepLabel;

  // 图 1：Loss（原始 + EMA + Grad Norm + 各参数组 lr）
  const lossChartOption = React.useMemo(() => {
    if (!metrics || metrics.steps.length === 0) return {};
    const steps = useEpoch && stepsPerEpoch ? metrics.steps.map((step) => step / stepsPerEpoch) : metrics.steps;
    const smoothed = smoothLoss(metrics.loss, emaAlpha);
    const lrSeries = Object.entries(metrics.lr || {}).map(([group, values], i) => ({
      name: `lr:${group}`,
      type: 'line' as const,
      yAxisIndex: 1,
      showSymbol: false,
      sampling: 'lttb' as const,
      data: values.map((v, idx) => [steps[idx], v] as [number, number]),
      lineStyle: { width: 1, type: 'dashed' as const, color: LR_COLORS[i % LR_COLORS.length] },
    }));
    return {
      tooltip: { trigger: 'axis' },
      legend: { textStyle: { color: '#888' } },
      grid: { left: '3%', right: '4%', bottom: '15%', containLabel: true },
      xAxis: { type: 'value', name: xAxisName, splitLine: { show: false } },
      yAxis: [
        { type: 'value', name: 'Loss', scale: true, splitLine: { lineStyle: { color: '#33333320' } } },
        { type: 'value', name: 'LR', scale: true, splitLine: { show: false } },
      ],
      dataZoom: [{ type: 'inside' }, { type: 'slider' }],
      series: [
        {
          name: 'Loss', type: 'line', showSymbol: false, sampling: 'lttb',
          data: steps.map((s, i) => [s, metrics.loss[i]] as [number, number | null]),
          lineStyle: { width: 1.2, color: '#93c5fd' },
        },
        {
          name: 'Loss (EMA)', type: 'line', showSymbol: false, sampling: 'lttb',
          data: steps.map((s, i) => [s, smoothed[i]] as [number, number | null]),
          lineStyle: { width: 2, color: '#2563eb' },
        },
        {
          name: 'Grad Norm', type: 'line', showSymbol: false, sampling: 'lttb',
          data: steps.map((s, i) => [s, metrics.grad_norm[i]] as [number, number | null]),
          lineStyle: { width: 1, color: '#eab308' },
        },
        ...lrSeries,
      ],
    };
  }, [metrics, xAxisName, useEpoch, stepsPerEpoch, emaAlpha]);

  // 图 2：Validation（每个固定时间步一条线 + 均值）
  const validationChartOption = React.useMemo(() => {
    if (!metrics || !metrics.validation || metrics.validation.length === 0) return null;
    const { series } = shapeValidationSeries(metrics.validation);
    return {
      tooltip: { trigger: 'axis' },
      legend: { textStyle: { color: '#888' } },
      grid: { left: '3%', right: '4%', bottom: '12%', containLabel: true },
      xAxis: { type: 'value', name: stepLabel, splitLine: { show: false } },
      yAxis: { type: 'value', name: 'val loss', scale: true, splitLine: { lineStyle: { color: '#33333320' } } },
      series: series.map((s, i) => ({
        name: s.name,
        type: 'line' as const,
        showSymbol: true,
        data: s.data,
        lineStyle: {
          width: s.name === 'mean' ? 2.5 : 1.2,
          color: s.name === 'mean' ? '#f43f5e' : LR_COLORS[i % LR_COLORS.length],
        },
      })),
    };
  }, [metrics, stepLabel]);

  const vramMetric = job?.progress?.vram_metric ?? metrics?.vram_metric;
  const currentAllocated = vramMetric === 'current_allocated';
  const chartVramMetric = metrics?.vram_metric ?? vramMetric;
  const memorySeriesLabel = chartVramMetric === 'current_allocated'
    ? `${t('job.currentTrainingAllocated')} (GB)`
    : chartVramMetric === 'peak_allocated' ? `${t('job.vramPeak')} (GB)` : 'VRAM (GB)';

  // Memory values retain the backend metric: current MPS allocation or CUDA peak.
  const perfChartOption = React.useMemo(() => {
    if (!metrics || metrics.steps.length === 0) return {};
    const steps = useEpoch && stepsPerEpoch ? metrics.steps.map((step) => step / stepsPerEpoch) : metrics.steps;
    return {
      tooltip: { trigger: 'axis' },
      legend: { textStyle: { color: '#888' } },
      grid: { left: '3%', right: '4%', bottom: '15%', containLabel: true },
      xAxis: { type: 'value', name: xAxisName, splitLine: { show: false } },
      yAxis: [
        { type: 'value', name: 'it/s', scale: true, splitLine: { lineStyle: { color: '#33333320' } } },
        { type: 'value', name: memorySeriesLabel, scale: true, splitLine: { show: false } },
      ],
      dataZoom: [{ type: 'inside' }, { type: 'slider' }],
      series: [
        {
          name: 'Speed (it/s)', type: 'line', showSymbol: false, sampling: 'lttb',
          data: steps.map((s, i) => [s, metrics.it_s[i]] as [number, number | null]),
          lineStyle: { width: 1.5, color: '#10b981' },
        },
        {
          name: memorySeriesLabel, type: 'line', yAxisIndex: 1, showSymbol: false, sampling: 'lttb',
          data: steps.map((s, i) => [s, metrics.vram_mb[i] == null ? null : metrics.vram_mb[i]! / 1024] as [number, number | null]),
          lineStyle: { width: 1.2, color: '#ec4899' },
        },
      ],
    };
  }, [metrics, xAxisName, useEpoch, stepsPerEpoch, memorySeriesLabel]);

  const filteredLogs = logs.filter(line => (logFilter === 'all' || (line.level === 'warning' ? 'warn' : line.level) === logFilter) && (!logQuery || line.msg.toLocaleLowerCase().includes(logQuery.toLocaleLowerCase())));
  const filteredSamples = [...samples].reverse().filter(sample => !sampleStep || String(sample.step) === sampleStep);
  const samplePages = Math.max(1, Math.ceil(filteredSamples.length / 24));
  const visibleSamples = filteredSamples.slice((samplePage - 1) * 24, samplePage * 24);


  // Unknown phases only show activity while the worker is actually running.
  const phaseLabels: Record<string, string> = {
    preparing: t('job.phasePreparing', '准备'),
    caching: t('job.phaseCaching', '缓存'),
    training: t('job.phaseTraining', '训练'),
    finalizing: t('job.phaseFinalizing', '收尾'),
  };
  const phase = job?.progress?.phase || '';
  const rawPhase = ['starting', 'loading', 'indexing', 'injecting', 'prepared'].includes(phase) ? 'preparing' : phase.startsWith('caching_') ? 'caching' : phase;
  const currentPhaseIndex = job?.status === 'completed' ? PHASE_KEYS.length : PHASE_KEYS.findIndex((k) => k === rawPhase);

  // 任务状态徽章（文案 + 颜色）
  const statusLabels: Record<string, string> = {
    queued: t('job.statusQueued', '排队中'),
    scheduled: t('job.statusScheduled', '已排期'),
    running: t('job.statusRunning', '运行中'),
    pausing: t('job.statusPausing', '暂停中'),
    cancelling: t('job.statusCancelling', '取消中'),
    paused: t('job.statusPaused', '已暂停'),
    completed: t('job.statusCompleted', '已完成'),
    failed: t('job.statusFailed', '失败'),
    cancelled: t('job.statusCancelled', '已取消'),
  };
  const statusText = job?.status ? statusLabels[job.status] || job.status : t('job.statusUnknown', '未知');
  const statusBadgeClass =
    job?.status === 'running'
      ? 'bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300'
      : job?.status === 'failed'
        ? 'bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300'
        : job?.status === 'paused' || job?.status === 'pausing'
          ? 'bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300'
          : 'bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-300';

  const checkpointKindLabel = (kind: string) =>
    kind === 'weights' ? t('job.kindWeights', '仅权重') : kind === 'full' ? t('job.kindFull', '完整') : kind;

  const logLevelLabels: Record<string, string> = {
    all: t('job.levelAll'),
    info: t('job.levelInfo', 'INFO'),
    warn: t('job.levelWarn', 'WARN'),
    error: t('job.levelError', 'ERROR'),
    debug: t('job.levelDebug', 'DEBUG'),
  };

  const tabs = [
    { key: 'metrics', icon: Activity, label: t('job.tabMetrics') },
    { key: 'samples', icon: ImageIcon, label: `${t('job.tabSamples')} (${samples.length})` },
    { key: 'checkpoints', icon: Layers, label: `${t('job.tabCheckpoints')} (${checkpoints.length})` },
    { key: 'logs', icon: Terminal, label: t('job.tabLogs') },
    { key: 'config', icon: Code, label: t('job.tabConfig') },
  ] as const;

  const resumeCheckpoint = async (checkpoint: JobCheckpoint) => {
    if (!job || !configSnapshot || resuming) return;
    setResuming(true); setActionError('');
    try {
      const next = await apiClient.post<Job>('/jobs', {
        type: 'train', project_id: job.project_id, name: `${job.name} · ${t('job.continueTraining')} ${checkpoint.step}`,
        ...(job.version_id ? { version_id: job.version_id } : {}),
        config: { ...configSnapshot, checkpoint: { ...configSnapshot.checkpoint, resume: checkpoint.path } },
      }, { silent: true });
      navigate(`/jobs/${next.id}`);
    } catch (err: unknown) { setActionError(formatApiError(err)); }
    finally { setResuming(false); }
  };

  return (
    <div className="job-monitor task-workspace" data-testid="job-detail-page">
      <div className="job-monitor-bar"><div className="job-monitor-links"><Link to={queueReturnTo}>← {text('全局训练队列', 'Training queue')}</Link>
      {job?.project_id && <Link to={projectUrl(job.project_id, job.version_id, 'results')} className="inline-flex flex-wrap gap-1 text-sm text-blue-600 hover:underline">← {job.project_id} · {text('训练结果', 'Training results')}{job.version_id && <span className="break-words text-xs" title={job.version_id}> · {versionName ? `${text('版本', 'Version')} ${versionName}` : text('所属版本', 'Version')}</span>}</Link>}
      </div><div className="job-monitor-identity"><div><h1>{job?.name || text('读取任务…', 'Loading job…')}</h1><small>{id} · {job?.type === 'cache' ? text('缓存任务', 'Cache job') : text('训练任务', 'Training job')}</small></div><span className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${statusBadgeClass}`}>{statusText}</span>{job && <JobActions key={job.id} job={job} onUpdated={updated => { if (updated.id === job.id) setJob(updated); else navigate(`/jobs/${updated.id}`); }}/>}</div></div>
      {dataError && <div className="task-error" role="alert">{dataError}</div>}
      {(actionError || job?.error) && <div role="alert" className="whitespace-pre-line break-words rounded bg-red-50 text-red-700 p-3 dark:bg-red-950 dark:text-red-300">{actionError || job?.error}</div>}
      {/* 1. 头部指标与阶段时间线 */}
      <div className="job-monitor-summary bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700 space-y-3">
        {/* 大数字指标 StatCard */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard
            label={t('job.progress')}
            value={`${job?.progress?.step ?? 0} / ${job?.progress?.total_steps ?? 0}`}
          />
          <StatCard
            label={t('job.speed')}
            value={job?.progress?.it_s != null ? `${Number(job.progress.it_s).toFixed(2)} it/s` : '--'}
          />
          <StatCard label={t(currentAllocated ? 'job.currentAllocated' : 'job.vramPeak')} value={formatBytesMB(job?.progress?.vram_peak_mb)} />
          <StatCard label={t('job.eta')} value={formatEta(job?.progress?.eta_s)} />
        </div>

        {/* 阶段时间线 */}
        <div className="flex items-center space-x-2 pt-2 border-t dark:border-slate-700">
          {currentPhaseIndex === -1 && job?.status === 'running' && (
            <>
              <span
                className="flex items-center space-x-1.5 text-xs font-medium px-3 py-1.5 rounded-md bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300"
                title={rawPhase || undefined}
              >
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>{t(`phase.${phase}`, t('job.phaseInProgress', '进行中'))}</span>
              </span>
              <div className="flex-1 h-0.5 bg-slate-200 dark:bg-slate-700 mx-1" />
            </>
          )}
          {PHASE_KEYS.map((ph, idx) => {
            const isDone = currentPhaseIndex > idx;
            const isCurrent = idx === currentPhaseIndex;
            return (
              <React.Fragment key={ph}>
                <div className={`flex items-center space-x-1.5 text-xs font-medium px-3 py-1.5 rounded-md ${
                  isCurrent ? 'bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300' : isDone ? 'text-green-600 dark:text-green-400' : 'text-slate-400'
                }`}>
                  {isDone ? <CheckCircle2 className="w-3.5 h-3.5" /> : <span>{idx + 1}.</span>}
                  <span>{phaseLabels[ph]}</span>
                </div>
                {idx < PHASE_KEYS.length - 1 && <div className="flex-1 h-0.5 bg-slate-200 dark:bg-slate-700 mx-1" />}
              </React.Fragment>
            );
          })}
        </div>

        {/* 采样预览进度（job.sample_progress SSE） */}
        {sampleProgress && (
          <div className="flex items-center space-x-3 pt-3 mt-1 border-t dark:border-slate-700" data-testid="sample-progress">
            <span className="text-xs font-medium text-indigo-500 whitespace-nowrap">
              {t('job.samplingProgress', {
                k: sampleProgress.promptIndex + 1,
                n: sampleProgress.prompts,
                done: sampleProgress.done,
                total: sampleProgress.total,
              })}
            </span>
            <div className="flex-1 bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
              <div
                className="bg-indigo-500 h-1.5 rounded-full transition-all"
                style={{ width: `${sampleProgress.total > 0 ? (sampleProgress.done / sampleProgress.total) * 100 : 0}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* 2. Tabs 切换导航 */}
      <div className="border-b border-slate-200 dark:border-slate-700">
        <nav className="job-monitor-tabs task-tabs" role="tablist" aria-label={text('任务详情分区', 'Job details tabs')}>
          {tabs.map((tab, index) => (
            <button
              key={tab.key}
              role="tab" id={`job-tab-${tab.key}`} aria-controls={`job-panel-${tab.key}`} tabIndex={activeTab === tab.key ? 0 : -1} aria-selected={activeTab === tab.key}
              onKeyDown={event => { const next = event.key === 'ArrowRight' ? (index + 1) % tabs.length : event.key === 'ArrowLeft' ? (index + tabs.length - 1) % tabs.length : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : -1; if (next >= 0) { event.preventDefault(); setActiveTab(tabs[next].key); document.getElementById(`job-tab-${tabs[next].key}`)?.focus(); } }}
              onClick={() => setActiveTab(tab.key)}
              className={`flex items-center space-x-2 py-3 border-b-2 ${activeTab === tab.key ? 'border-blue-500 text-blue-600 dark:text-blue-400' : 'border-transparent text-slate-500 hover:text-slate-700'}`}
            >
              <tab.icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </nav>
      </div>

      {/* 3. 详细内容区域 */}
      <div role="tabpanel" id={`job-panel-${activeTab}`} aria-labelledby={`job-tab-${activeTab}`}>
      {activeTab === 'metrics' && (
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6 space-y-4">
          <div className="flex justify-between items-center">
            <div className="flex space-x-2 text-xs">
              <button
                onClick={() => setXAxisMode('step')}
                className={`px-2.5 py-1 rounded ${xAxisMode === 'step' ? 'bg-blue-600 text-white' : 'bg-slate-100 dark:bg-slate-700'}`}
              >
                {stepLabel}
              </button>
              <button
                onClick={() => setXAxisMode('epoch')} disabled={!stepsPerEpoch} title={!stepsPerEpoch ? t('job.epochUnavailable') : undefined}
                className={`px-2.5 py-1 rounded ${xAxisMode === 'epoch' ? 'bg-blue-600 text-white' : 'bg-slate-100 dark:bg-slate-700'}`}
              >
                {epochLabel}
              </button>
            </div>
            <div className="flex items-center space-x-2 text-xs text-slate-500">
              <span>{t('job.emaAlpha')}:</span>
              <input
                aria-label={t('job.emaAlpha')}
                type="range"
                min="0.1"
                max="0.99"
                step="0.05"
                value={emaAlpha}
                onChange={(e) => setEmaAlpha(Number(e.target.value))}
              />
              <span className="font-mono">{emaAlpha}</span>
            </div>
          </div>

          {!metrics || metrics.steps.length === 0 ? (
            <EmptyState
              icon={Activity}
              title={t('job.noMetrics', '暂无训练指标')}
              hint={t('job.noMetricsHint', '任务产出 step 数据后，Loss / 吞吐图表会显示在这里。')}
            />
          ) : (
            <>
              <EChart option={lossChartOption} style={{ height: 310 }} />

              {validationChartOption && (
                <div className="pt-4 border-t dark:border-slate-700">
                  <div className="text-xs text-slate-400 mb-2">{t('job.validationTitle')}</div>
                  <EChart option={validationChartOption} style={{ height: 260 }} />
                </div>
              )}

              <div className="pt-4 border-t dark:border-slate-700">
                <div className="text-xs text-slate-400 mb-2">{t(chartVramMetric === 'current_allocated' ? 'job.currentMemoryPerfTitle' : 'job.perfTitle')}</div>
                <EChart option={perfChartOption} style={{ height: 280 }} />
              </div>
            </>
          )}
        </div>
      )}

      {activeTab === 'samples' && (
        <section><div className="job-sample-controls"><StudioSelect aria-label={text('采样步数', 'Sample step')} value={sampleStep} options={[{ value: '', label: text('全部步数', 'All steps') }, ...[...new Set(samples.map(sample => sample.step))].sort((a,b) => b-a).map(step => ({ value: String(step), label: `${text('步数', 'Step')} ${step}` }))]} onValueChange={value => { setSampleStep(value); setSamplePage(1); }}/><span>{text(`共 ${filteredSamples.length} 张 · 每页 24 张`, `${filteredSamples.length} samples · 24 per page`)}</span>{samplePages > 1 && <div className="task-actions"><button className="task-button" disabled={samplePage <= 1} onClick={() => setSamplePage(page => page - 1)}>{text('上一页', 'Previous')}</button><span>{samplePage} / {samplePages}</span><button className="task-button" disabled={samplePage >= samplePages} onClick={() => setSamplePage(page => page + 1)}>{text('下一页', 'Next')}</button></div>}</div><div className="job-sample-gallery" data-testid="samples-gallery">
          {samples.length === 0 && (
            <EmptyState
              icon={ImageIcon}
              title={t('job.noSamples', '暂无采样图片')}
              hint={t('job.noSamplesHint', '训练过程中的采样预览会自动出现在这里。')}
            />
          )}
          {visibleSamples.map((s, idx) => (
            <div key={idx} className="bg-white dark:bg-slate-800 rounded-xl overflow-hidden border border-slate-200 dark:border-slate-700">
              <a href={s.url.startsWith('/api/') ? apiUrl(s.url.slice(4)) : s.url} target="_blank" rel="noreferrer" aria-label={`${text('打开完整采样图', 'Open full sample')}: ${s.prompt}`}><img loading="lazy" src={s.url.startsWith('/api/') ? apiUrl(s.url.slice(4)) : s.url} alt={s.prompt} className="w-full h-56 object-contain bg-slate-100 dark:bg-slate-900" /></a>
              <div className="p-3 space-y-1 text-xs">
                <div className="flex justify-between font-semibold">
                  <span className="font-mono">{t('job.step')} {s.step}</span>
                  <span className="text-slate-400 font-mono">{t('job.seed')} {s.seed}</span>
                </div>
                <p className="text-slate-500 line-clamp-2" title={s.prompt}>{s.prompt}</p>
              </div>
            </div>
          ))}
        </div></section>
      )}

      {activeTab === 'checkpoints' && (
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-x-auto">
          {checkpoints.length === 0 ? (
            <EmptyState
              icon={Layers}
              title={t('job.noCheckpoints', '暂无检查点')}
              hint={t('job.noCheckpointsHint', '训练到保存步数后会自动生成检查点。')}
            />
          ) : (
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-50 dark:bg-slate-800/50 text-xs text-slate-400 border-b dark:border-slate-700">
                <tr>
                  <th className="p-4">{t('job.stepCol')}</th>
                  <th className="p-4">{t('job.kind')}</th>
                  <th className="p-4">{t('job.path')}</th>
                  <th className="p-4">{t('job.size')}</th>
                  <th className="p-4">{t('queue.created')}</th>
                  <th className="p-4 text-right">{t('common.actions')}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
                {checkpoints.map((cp, idx) => (
                  <tr key={idx} className="hover:bg-slate-50 dark:hover:bg-slate-750">
                    <td className="p-4 font-semibold font-mono">{cp.step}</td>
                    <td className="p-4">{checkpointKindLabel(cp.kind)}{cp.ema ? ' (EMA)' : ''}</td>
                    <td className="p-4 font-mono text-xs text-slate-500">{cp.path}</td>
                    <td className="p-4 font-mono">{formatBytes(cp.size)}</td>
                    <td className="p-4 text-xs text-slate-500 whitespace-nowrap">{formatTime(cp.created_at)}</td>
                    <td className="p-4 text-right space-x-2">
                      {cp.kind === 'full' ? (
                        <button disabled={resuming || !configSnapshot} onClick={() => resumeCheckpoint(cp)} className="px-2.5 py-1 text-xs bg-blue-600 text-white rounded disabled:opacity-50">
                          {t('job.continueTraining')}
                        </button>
                      ) : cp.artifact_id ? (
                        <a href={apiUrl(`/artifacts/${cp.artifact_id}/download`)} className="px-2.5 py-1 text-xs bg-slate-100 dark:bg-slate-700 rounded inline-flex items-center space-x-1">
                          <Download className="w-3.5 h-3.5" /><span>{t('job.download')}</span>
                        </a>
                      ) : <span className="text-xs text-slate-400">{t('job.downloadUnavailable')}</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {activeTab === 'logs' && (
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4 space-y-3">
          <div className="job-log-toolbar"><StudioSelect aria-label={text('日志模式', 'Log mode')} value={logMode} options={[{ value: 'live', label: text('实时末尾 · 500 行', 'Live tail · 500 lines') }, { value: 'history', label: text('完整历史 · 分页读取', 'Full history · paginated') }]} onValueChange={value => { setLogMode(value as 'live' | 'history'); setLogOffsets([0]); setLogs([]); }}/><StudioSelect aria-label={text('日志级别', 'Log level')} value={logFilter} options={LOG_LEVELS.map(value => ({ value, label: logLevelLabels[value] }))} onValueChange={setLogFilter}/><input aria-label={text('搜索当前页日志', 'Search this log page')} value={logQuery} onChange={event => setLogQuery(event.target.value)} placeholder={text('搜索当前页日志', 'Search this log page')}/><button className="task-button" disabled={logLoading} onClick={() => void fetchLogs()}>{text('刷新日志', 'Refresh logs')}</button><label><input type="checkbox" checked={autoScrollLog} onChange={event => setAutoScrollLog(event.target.checked)}/>{t('job.followBottom')}</label></div>
          {logError && <p role="alert" className="task-error">{logError}</p>}
          {logMode === 'history' && <div className="task-pagination"><span>{text('按原始顺序读取，每页最多 500 行；筛选作用于当前页。', 'Original order, up to 500 lines per page; filters apply to this page.')}</span><div><button className="task-button" disabled={logLoading || logOffsets.length === 1} onClick={() => { setLogs([]); setLogOffsets(offsets => offsets.slice(0, -1)); }}>{text('上一页日志', 'Previous log page')}</button><span>{logOffsets.length}</span><button className="task-button" disabled={logLoading || !hasMoreLogs} onClick={() => { setLogs([]); setLogOffsets(offsets => [...offsets, nextLogOffset]); }}>{text('下一页日志', 'Next log page')}</button></div></div>}
          <div
            ref={logContainerRef}
            className="h-80 overflow-y-auto bg-slate-900 text-slate-200 font-mono text-xs p-3 rounded-lg space-y-1"
          >
            {filteredLogs.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center">
                <Terminal className="w-6 h-6 text-slate-600" />
                <p className="mt-2 text-slate-400">{t('job.noLogs', '暂无日志')}</p>
                <p className="mt-1 text-slate-500">{t('job.noLogsHint', '任务运行日志会实时输出到这里。')}</p>
              </div>
            ) : (
              filteredLogs.map((l, idx) => (
                <div key={idx} className={`flex space-x-2 ${l.level === 'debug' ? 'opacity-60' : ''}`}>
                  <span className="text-slate-500 shrink-0">
                    [{l.ts == null ? '--' : typeof l.ts === 'number' ? new Date(l.ts * 1000).toLocaleTimeString() : l.ts}]
                  </span>
                  <span className={`uppercase font-bold ${logLevelColor(l.level)}`}>
                    [{l.level}]
                  </span>
                  <span className="flex-1 break-all">{l.msg}</span>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {activeTab === 'config' && (
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6">
          {configSnapshot == null ? (
            <EmptyState
              icon={Code}
              title={t('job.noConfig', '暂无配置快照')}
              hint={t('job.noConfigHint', '任务启动时记录的训练配置会显示在这里。')}
            />
          ) : (
            <pre className="text-xs font-mono bg-slate-50 dark:bg-slate-900 p-4 rounded-lg overflow-x-auto">
              {JSON.stringify(configSnapshot, null, 2)}
            </pre>
          )}
        </div>
      )}
      </div>
    </div>
  );
}
