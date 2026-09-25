import { jobTypeLabel, mergeJobEvent } from '../../utils/jobs';
import React from 'react';
import { Link, useParams, useNavigate, useSearchParams, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { EChart } from '../../components/EChart';
import { apiClient, apiUrl } from '../../api/client';
import { Job, JobMetrics, JobSample, JobCheckpoint } from '../../api/types';
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
  ChevronDown,
  ArrowLeft,
} from 'lucide-react';
import { shapeValidationSeries, mergeValidationPoint, smoothLoss, appendMetricStep } from '../../utils/metrics';
import { formatBytes, formatEta, formatTime } from '../../utils/format';
import { formatApiError } from '../../utils/errors';
import { projectUrl, type ProjectVersion } from '../../utils/projectVersions';
import { useWorkspaceText } from '../../utils/workspaceText';
import StudioSelect from '../../components/StudioSelect';
import { JobActions, JobStatus } from '../Queue/jobPresentation';
import SampleLoss from '../../components/SampleLoss';
import ConfigHelp from '../../components/ConfigHelp';
import { metricChartBase, metricLabels } from './metricPresentation';
import '../Queue/queue.css';
import './job-detail.css';
import './job-metrics.css';
import JobLogView from './JobLogView';
import { SlidingIndicator } from '../../components/motion';
import { useEnterAnimation } from '../../utils/motion';

type VersionedJob = Job & { version_id?: string | null; latest: Job['latest'] & {loss_mean?:number|null; loss_count?:number|null; loss_mean_scope?:string|null} };

const LR_COLORS = ['#a78bfa', '#f59e0b', '#22d3ee', '#fb7185', '#84cc16', '#e879f9'];

const PHASE_KEYS = ['preparing', 'caching', 'training', 'finalizing'];

function mergeSamples(previous: JobSample[], incoming: JobSample[]): JobSample[] {
  // Step and epoch triggers may share step/seed while saving different image files.
  const samples = new Map(previous.map(sample => [sample.url, sample]));
  for (const sample of incoming) samples.set(sample.url, { ...samples.get(sample.url), ...sample });
  return [...samples.values()].sort((a, b) => a.step - b.step || a.created_at - b.created_at || a.prompt_index - b.prompt_index);
}

function StatCard({ label, value, hint }: { label: string; value: React.ReactNode; hint?:string }) {
  return <div className="job-stat"><div className="job-stat-label">{label}{hint && <ConfigHelp label={`${label} · 说明`}>{hint}</ConfigHelp>}</div><div className="job-stat-value">{value}</div></div>;
}

/** 空态：lucide 图标 + 标题 + 一行提示 */
function EmptyState({ icon: Icon, title, hint }: { icon: React.ComponentType<{ className?: string }>; title: string; hint?: string }) {
  return (
    <div className="col-span-full flex flex-col items-center justify-center py-14 text-center">
      <Icon className="w-8 h-8 text-slate-300 dark:text-slate-600" />
      <p className="mt-3 text-sm font-medium text-slate-500 dark:text-slate-400">{title}</p>
      {hint && <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">{hint}</p>}
    </div>
  );
}

export default function JobDetail() {
  const { id } = useParams<{ id: string }>();
  const { t, i18n } = useTranslation();
  const text = useWorkspaceText();
  const navigate = useNavigate();
  const location = useLocation();
  const [params, setParams] = useSearchParams();
  const [dataError, setDataError] = React.useState('');
  const [actionError, setActionError] = React.useState('');
  const [resuming, setResuming] = React.useState(false);

  const [job, setJob] = React.useState<VersionedJob | null>(null);
  const [clock, setClock] = React.useState(() => Date.now() / 1000);
  React.useEffect(() => {
    if (!job?.started_at || !['running','pausing','cancelling'].includes(job.status)) return;
    const timer = window.setInterval(() => setClock(Date.now() / 1000), 1000);
    return () => window.clearInterval(timer);
  }, [job?.started_at, job?.status]);
  const [resolvedVersion, setResolvedVersion] = React.useState<{ projectId: string; versionId: string; name: string } | null>(null);
  const [metrics, setMetrics] = React.useState<JobMetrics | null>(null);
  const [samples, setSamples] = React.useState<JobSample[]>([]);
  const [checkpoints, setCheckpoints] = React.useState<JobCheckpoint[]>([]);
  const [configSnapshot, setConfigSnapshot] = React.useState<any>(null);
  const [sampleProgress, setSampleProgress] = React.useState<{ step: number; promptIndex: number; prompts: number; done: number; total: number } | null>(null);

  const requestedTab = params.get('tab') || '';
  const allowedTabs = job?.type === 'xyz' ? ['logs', 'config'] : ['metrics', 'samples', 'checkpoints', 'logs', 'config'];
  const activeTab = allowedTabs.includes(requestedTab) ? requestedTab : job?.type === 'xyz' ? 'logs' : 'metrics';
  const tabPanel = useEnterAnimation<HTMLDivElement>(activeTab, { skipFirst: true });
  // Tabs replace the entry so Back leaves the job instead of stepping through tabs.
  const setActiveTab = (tab: string) => { const next = new URLSearchParams(params); next.set('tab', tab); setParams(next, { replace: true, state: location.state }); };
  const [xAxisMode, setXAxisMode] = React.useState<'step' | 'epoch'>('step');
  const [emaAlpha, setEmaAlpha] = React.useState<number>(0.9);
  const [showDiagnostics, setShowDiagnostics] = React.useState(false);
  const chinese = (i18n.resolvedLanguage || i18n.language).startsWith('zh');
  const labels = React.useMemo(() => metricLabels(chinese), [chinese]);
  const [samplePage, setSamplePage] = React.useState(1);
  const [sampleStep, setSampleStep] = React.useState('');
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
    setDataError(''); setSamplePage(1); setSampleStep(''); setJob(null); setMetrics(null); setSamples([]); setCheckpoints([]); setConfigSnapshot(null); setSampleProgress(null);
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
  const resultsUrl = job?.project_id ? projectUrl(job.project_id, job.version_id, 'results') : '/queue';
  const versionNumber = job?.version_number ? `v${job.version_number}` : '';
  const versionTitle = (job?.version_name || versionName || '').trim();
  const versionLabel = versionNumber && versionTitle && !new RegExp(`^${versionNumber}(?:$|[\\s·:：-])`, 'i').test(versionTitle) ? `${versionNumber} · ${versionTitle}` : versionTitle || versionNumber;
  // Several pages open a job; return to the one the user came from, or its version results.
  const goBack = () => {
    // The router numbers its history entries; 0 is the page this app session opened on.
    const index = (window.history.state as { idx?: number } | null)?.idx;
    if (typeof index === 'number' ? index > 0 : location.key !== 'default') navigate(-1); else navigate(resultsUrl);
  };

  // 2. SSE 增量监听
  useEventStream(EVENT_TYPES.JOB_STATE, (data: any) => {
    if (data.job_id === id) {
      setJob((prev) => prev ? mergeJobEvent(prev, data) : null);
      if (['completed', 'failed', 'cancelled', 'paused'].includes(data.status)) {
        void refreshSamples();
        void apiClient.get<VersionedJob>(`/jobs/${id}`, {silent:true}).then(updated => setJob(previous => previous?.id === updated.id ? updated : previous)).catch(() => {});
      }
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

  // 采样进度
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

  const stepLabel = t('job.step');
  const epochLabel = t('job.epoch');
  const stepsPerEpoch = job?.progress?.steps_per_epoch;
  const useEpoch = xAxisMode === 'epoch' && !!stepsPerEpoch;
  const xAxisName = useEpoch ? epochLabel : stepLabel;

  const metricSteps = React.useMemo(() => useEpoch && stepsPerEpoch ? metrics?.steps.map(step => step / stepsPerEpoch) || [] : metrics?.steps || [], [metrics, useEpoch, stepsPerEpoch]);
  const lossChartOption = React.useMemo(() => ({
    ...metricChartBase(xAxisName, labels.loss),
    series: [
      { name: labels.raw, type: 'line', showSymbol: false, sampling: 'lttb', data: metricSteps.map((step, index) => [step, metrics?.loss[index] ?? null]), lineStyle: { width: 1.2, color: '#93c5fd' }, itemStyle: {color:'#93c5fd'} },
      { name: labels.ema, type: 'line', showSymbol: false, sampling: 'lttb', data: smoothLoss(metrics?.loss || [], emaAlpha).map((loss, index) => [metricSteps[index], loss]), lineStyle: { width: 2, color: '#2563eb' }, itemStyle: {color:'#2563eb'} },
    ],
  }), [metrics, metricSteps, xAxisName, labels, emaAlpha]);
  const learningRateChartOption = React.useMemo(() => ({
    ...metricChartBase(xAxisName, labels.lr),
    series: Object.entries(metrics?.lr || {}).map(([group, values], index) => ({
      name: `${labels.lr} · ${group}`, type: 'line', showSymbol: false, sampling: 'lttb',
      data: metricSteps.map((step, point) => [step, values[point] ?? null]),
      lineStyle: { width: 1.5, color: LR_COLORS[index % LR_COLORS.length] }, itemStyle: {color:LR_COLORS[index % LR_COLORS.length]},
    })),
  }), [metrics, metricSteps, xAxisName, labels]);
  const gradientChartOption = React.useMemo(() => ({
    ...metricChartBase(xAxisName, labels.gradient),
    series: [{name:labels.gradient,type:'line',showSymbol:false,sampling:'lttb',data:metricSteps.map((step,index) => [step,metrics?.grad_norm[index] ?? null]),lineStyle:{width:1.5,color:'#d97706'},itemStyle:{color:'#d97706'}}],
  }), [metrics, metricSteps, xAxisName, labels]);

  const validationChartOption = React.useMemo(() => {
    if (!metrics?.validation?.length) return null;
    const { series } = shapeValidationSeries(metrics.validation);
    return {
      ...metricChartBase(xAxisName, labels.validation),
      series: series.map((series, index) => ({
        name: series.name === 'mean' ? labels.mean : `${labels.timestep} ${series.name}`,
        type: 'line', showSymbol: true,
        data: series.data.map(([step, loss]) => [useEpoch && stepsPerEpoch ? step / stepsPerEpoch : step, loss]),
        lineStyle: { width: series.name === 'mean' ? 2.5 : 1.2, color: series.name === 'mean' ? '#f43f5e' : LR_COLORS[index % LR_COLORS.length] },
        itemStyle: {color:series.name === 'mean' ? '#f43f5e' : LR_COLORS[index % LR_COLORS.length]},
      })),
    };
  }, [metrics, xAxisName, useEpoch, stepsPerEpoch, labels]);

  const vramMetric = job?.progress?.vram_metric ?? metrics?.vram_metric;
  const chartVramMetric = metrics?.vram_metric ?? vramMetric;
  const memorySeriesLabel = chartVramMetric === 'current_allocated'
    ? `${t('job.currentTrainingAllocated')} (GB)`
    : chartVramMetric === 'peak_allocated' ? `${t('job.vramPeak')} (GB)` : labels.memory;

  // Keep the backend's MPS current-allocation / CUDA peak semantics.
  const perfChartOption = React.useMemo(() => {
    const base = metricChartBase(xAxisName, 'it/s');
    return {
      ...base,
      yAxis: [base.yAxis, {...base.yAxis, name:'GB', splitLine:{show:false}}],
      series: [
        { name:labels.speed,type:'line',showSymbol:false,sampling:'lttb',data:metricSteps.map((step,index)=>[step,metrics?.it_s[index] ?? null]),lineStyle:{width:1.5,color:'#10b981'},itemStyle:{color:'#10b981'} },
        { name:memorySeriesLabel,type:'line',yAxisIndex:1,showSymbol:false,sampling:'lttb',data:metricSteps.map((step,index)=>[step,metrics?.vram_mb[index] == null ? null : metrics.vram_mb[index]! / 1024]),lineStyle:{width:1.2,color:'#ec4899'},itemStyle:{color:'#ec4899'} },
      ],
    };
  }, [metrics, metricSteps, xAxisName, memorySeriesLabel, labels]);

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
  const rawPhase = ['starting', 'checking_communication', 'loading', 'indexing', 'injecting', 'prepared'].includes(phase) ? 'preparing' : phase.startsWith('caching_') ? 'caching' : phase;
  const currentPhaseIndex = job?.status === 'completed' ? PHASE_KEYS.length : PHASE_KEYS.findIndex((k) => k === rawPhase);

  const checkpointKindLabel = (kind: string) =>
    kind === 'model' ? text('全量模型组件', 'Full-model components') : kind === 'weights' ? t('job.kindWeights', '仅权重') : kind === 'full' ? t('job.kindFull', '完整') : kind;

  const tabs = [
    { key: 'metrics', icon: Activity, label: t('job.tabMetrics') },
    { key: 'samples', icon: ImageIcon, label: `${t('job.tabSamples')} (${samples.length})` },
    { key: 'checkpoints', icon: Layers, label: `${t('job.tabCheckpoints')} (${checkpoints.length})` },
    { key: 'logs', icon: Terminal, label: t('job.tabLogs') },
    { key: 'config', icon: Code, label: t('job.tabConfig') },
  ].filter(tab => allowedTabs.includes(tab.key));

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

  const lossNumber = (value: number | null | undefined) => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(4) : '—';
  const recordedLosses = metrics?.loss.filter((value): value is number => typeof value === 'number' && Number.isFinite(value)) || [];
  const meanLoss = job?.latest?.loss_mean ?? (recordedLosses.length ? recordedLosses.reduce((sum,value) => sum + value, 0) / recordedLosses.length : null);
  const meanScope = job?.latest?.loss_mean != null ? (job.latest.loss_mean_scope === 'since_resume' ? text('从此次恢复训练起，所有已完成训练步的损失平均值。','Mean loss over completed steps since this training was resumed.') : text('所有已完成训练步的损失平均值。','Mean loss over all completed optimizer steps.')) : text('旧任务没有完整累计值，显示已有日志中训练步的平均值。','This legacy run has no complete accumulator; this is the mean of recorded steps.');
  const learningRates = Object.entries(job?.latest?.lr || {}).filter((entry): entry is [string, number] => typeof entry[1] === 'number' && Number.isFinite(entry[1]));
  const epochProgress = stepsPerEpoch && job?.progress?.step != null ? (job.progress.step / stepsPerEpoch).toFixed(2).replace(/\.00$/, '') : job?.progress?.epoch != null ? String(job.progress.epoch + 1) : '—';
  const totalEpochs = stepsPerEpoch && job?.progress?.total_steps != null ? Math.ceil(job.progress.total_steps / stepsPerEpoch) : configSnapshot?.loop?.epochs;
  const elapsed = job?.started_at != null ? Math.max(0, (job.finished_at ?? (['running','pausing','cancelling'].includes(job.status) ? clock : job.started_at)) - job.started_at) : null;
  const configurationName = job?.version_name || versionName || job?.name;

  return (
    <div className="job-monitor task-workspace" data-view={activeTab} data-testid="job-detail-page">
      <header className="job-monitor-bar">
        <div className="job-monitor-nav">
          <button type="button" className="ui-btn ui-btn-sm" onClick={goBack}><ArrowLeft size={14}/>{text('返回', 'Back')}</button>
          <nav className="job-monitor-breadcrumb" aria-label={text('当前位置', 'Current location')}>
            <Link to="/queue">{text('任务队列', 'Job queue')}</Link>
            {job?.project_id && <><span aria-hidden="true">/</span><Link to={resultsUrl} title={text('打开版本训练结果', 'Open version results')}>{job.project_name || job.project_id}{versionLabel && <span className="job-monitor-version"> · {versionLabel}</span>}</Link></>}
          </nav>
        </div>
        <div className="job-monitor-identity">
          <div className="job-monitor-title"><h1>{job?.name || text('读取任务…', 'Loading job…')}</h1>{job && <JobStatus status={job.status}/>}</div>
          {job && <JobActions key={job.id} job={job} onUpdated={updated => { if (updated.id === job.id) setJob(updated); else navigate(`/jobs/${updated.id}`, { replace: true, state: location.state }); }}/>}
        </div>
      </header>
      <dl className="job-run-metadata" aria-label={text('运行信息','Run information')}>
        <div><dt>{text('任务类型','Job type')}</dt><dd>{job ? jobTypeLabel(job.type, text) : '—'}</dd></div>
        <div><dt>{text('开始时间','Started')}</dt><dd>{formatTime(job?.started_at)}</dd></div>
        <div><dt>{job?.type === 'train' ? text('训练时长','Training elapsed') : text('运行时长','Elapsed')}</dt><dd>{formatEta(elapsed)}</dd></div>
        <div><dt>{job?.type === 'train' ? text('训练配置','Training configuration') : text('任务配置','Task configuration')}</dt><dd>{configurationName ? `${configurationName} · ${text('参数快照','snapshot')}` : '—'}</dd></div>
        <div><dt>{text('运行 ID','Run ID')}</dt><dd><code>{job?.id || id}</code></dd></div>
      </dl>
      {dataError && <div className="task-error" role="alert">{dataError}</div>}
      {actionError && <div role="alert" className="task-error">{actionError}</div>}
      {job?.error && <div role="alert" className="job-failure"><div><strong>{job.type === 'train' ? text('训练失败', 'Training failed') : text('任务失败', 'Job failed')}</strong><p>{job.error}</p></div>{activeTab !== 'logs' && <button type="button" className="ui-btn ui-btn-sm" onClick={() => setActiveTab('logs')}><Terminal size={14}/>{text('查看日志', 'Open log')}</button>}</div>}
      {/* 1. 头部指标与阶段时间线 */}
      {job?.type !== 'xyz' && <div className="job-monitor-summary bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700 space-y-3">
        <div className="job-stat-grid" aria-label={text('训练核心指标','Training metrics')}>
          <StatCard label={text('步数','Steps')} value={`${job?.progress?.step ?? '—'} / ${job?.progress?.total_steps ?? '—'}`}/>
          <StatCard label={text('轮次','Epochs')} value={`${epochProgress} / ${totalEpochs ?? '—'}`}/>
          <StatCard label="Loss" value={lossNumber(job?.latest?.loss)} />
          <StatCard label={text('平均 Loss','Mean loss')} value={lossNumber(meanLoss)} hint={meanScope}/>
          <StatCard label={text('学习率','Learning rate')} value={learningRates.length ? <div className="job-learning-rates">{learningRates.map(([name, rate]) => <span key={name}>{learningRates.length > 1 && <small>{name}</small>}{rate.toExponential(2)}</span>)}</div> : '—'}/>
          <StatCard label={t('job.speed')} value={job?.progress?.it_s != null ? `${Number(job.progress.it_s).toFixed(2)} it/s` : '—'}/>
          <StatCard label={t('job.eta')} value={job?.status === 'completed' ? '0s' : formatEta(job?.progress?.eta_s)}/>
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
                  isCurrent ? 'bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300' : isDone ? 'text-green-700 dark:text-green-400' : 'text-slate-500 dark:text-slate-400'
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
      </div>}

      {/* 2. Tabs 切换导航 */}
      <nav className="job-monitor-tabs ui-tabs" role="tablist" aria-label={text('任务详情分区', 'Job details tabs')}>
          {tabs.map((tab, index) => (
            <button
              key={tab.key}
              role="tab" id={`job-tab-${tab.key}`} aria-controls={`job-panel-${tab.key}`} tabIndex={activeTab === tab.key ? 0 : -1} aria-selected={activeTab === tab.key}
              onKeyDown={event => { const next = event.key === 'ArrowRight' ? (index + 1) % tabs.length : event.key === 'ArrowLeft' ? (index + tabs.length - 1) % tabs.length : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : -1; if (next >= 0) { event.preventDefault(); setActiveTab(tabs[next].key); document.getElementById(`job-tab-${tabs[next].key}`)?.focus(); } }}
              onClick={() => setActiveTab(tab.key)}
            >
              <tab.icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          ))}
          <SlidingIndicator className="ui-tabs-indicator"/>
        </nav>

      {/* 3. 详细内容区域 */}
      <div ref={tabPanel} role="tabpanel" id={`job-panel-${activeTab}`} aria-labelledby={`job-tab-${activeTab}`}>
      {activeTab === 'metrics' && (
        <div className="job-metrics">
          <div className="job-metrics-toolbar">
            <div className="job-metrics-axis ui-segmented ui-segmented-sm" role="group" aria-label={text('横轴单位','Horizontal axis')}>
              <button type="button" aria-pressed={xAxisMode === 'step'} onClick={() => setXAxisMode('step')}>{stepLabel}</button>
              <button type="button" aria-pressed={xAxisMode === 'epoch'} onClick={() => setXAxisMode('epoch')} disabled={!stepsPerEpoch} title={!stepsPerEpoch ? t('job.epochUnavailable') : undefined}>{epochLabel}</button><SlidingIndicator className="ui-segmented-thumb"/>
            </div>
            <label className="job-metrics-smoothing">{text('显示 EMA 系数','Display EMA coefficient')}<input aria-label={text('显示 EMA 系数','Display EMA coefficient')} type="range" min="0" max="0.99" step="0.01" value={emaAlpha} onChange={event => setEmaAlpha(Number(event.target.value))}/><output>{emaAlpha.toFixed(2)}</output></label>
          </div>
          {!metrics?.steps.length ? <EmptyState icon={Activity} title={t('job.noMetrics', '暂无训练指标')} hint={text('等待训练步数记录。','Waiting for recorded training steps.')}/> : <>
            <section className="job-metrics-chart" aria-label={labels.loss}>
              <h2>{labels.loss}</h2><p>{text('EMA 系数越大，曲线越平滑；不改变训练参数或权重 EMA。','A higher EMA coefficient smooths the chart more; training parameters and weight EMA are unchanged.')}</p>

              <EChart option={lossChartOption} style={{ height: 320 }}/>
            </section>
            <section className="job-metrics-diagnostics">
              <button type="button" className="job-metrics-diagnostics-toggle" aria-expanded={showDiagnostics} aria-controls="job-metrics-diagnostic-charts" onClick={() => setShowDiagnostics(value => !value)}>{text('学习率、梯度与性能诊断','Learning rate, gradients and performance')}<ChevronDown size={15}/></button>
              {showDiagnostics && <div id="job-metrics-diagnostic-charts">
                <section className="job-metrics-chart" aria-label={labels.lr}><h2>{labels.lr}</h2><p>{text('每条线代表一个参数组；LoKr 的 w1 / w2 可设置不同学习率。','Each line represents a parameter group; LoKr w1 / w2 can use different learning rates.')}</p><EChart option={learningRateChartOption} style={{height:300}}/></section>
                <section className="job-metrics-chart" aria-label={labels.gradient}><h2>{labels.gradient}</h2><p>{text('每步梯度的大小，用于观察更新是否稳定。','Gradient magnitude per step, to inspect update stability.')}</p><EChart option={gradientChartOption} style={{height:280}}/></section>
                {validationChartOption && <section className="job-metrics-chart" aria-label={labels.validation}><h2>{t('job.validationTitle')}</h2><EChart option={validationChartOption} style={{height:280}}/></section>}
                <section className="job-metrics-chart" aria-label={text('吞吐与内存诊断','Throughput and memory diagnostics')}><h2>{t(chartVramMetric === 'current_allocated' ? 'job.currentMemoryPerfTitle' : 'job.perfTitle')}</h2><EChart option={perfChartOption} style={{height:300}}/></section>
              </div>}
            </section>
          </>}
        </div>
      )}

      {activeTab === 'samples' && (
        <section><div className="job-sample-controls"><StudioSelect aria-label={text('采样步数', 'Sample step')} value={sampleStep} options={[{ value: '', label: text('全部步数', 'All steps') }, ...[...new Set(samples.map(sample => sample.step))].sort((a,b) => b-a).map(step => ({ value: String(step), label: `${text('步数', 'Step')} ${step}` }))]} onValueChange={value => { setSampleStep(value); setSamplePage(1); }}/><span>{text(`共 ${filteredSamples.length} 张 · 每页 24 张`, `${filteredSamples.length} samples · 24 per page`)}</span>{samplePages > 1 && <div className="task-actions"><button type="button" className="ui-btn" disabled={samplePage <= 1} onClick={() => setSamplePage(page => page - 1)}>{text('上一页', 'Previous')}</button><span>{samplePage} / {samplePages}</span><button type="button" className="ui-btn" disabled={samplePage >= samplePages} onClick={() => setSamplePage(page => page + 1)}>{text('下一页', 'Next')}</button></div>}</div><div className="job-sample-gallery" data-testid="samples-gallery">
          {samples.length === 0 && (
            <EmptyState
              icon={ImageIcon}
              title={t('job.noSamples', '暂无采样图片')}

            />
          )}
          {visibleSamples.map((s, idx) => (
            <div key={idx} className="bg-white dark:bg-slate-800 rounded-xl overflow-hidden border border-slate-200 dark:border-slate-700">
              <a href={s.url.startsWith('/api/') ? apiUrl(s.url.slice(4)) : s.url} target="_blank" rel="noreferrer" aria-label={`${text('打开完整采样图', 'Open full sample')}: ${s.prompt}`}><img loading="lazy" src={s.url.startsWith('/api/') ? apiUrl(s.url.slice(4)) : s.url} alt={s.prompt} className="w-full h-56 object-contain bg-slate-100 dark:bg-slate-900" /></a>
              <div className="p-3 space-y-1 text-xs">
                <div className="flex justify-between font-semibold">
                  <span className="font-mono">{t('job.step')} {s.step}</span>
                  <span className="text-slate-500 dark:text-slate-400 font-mono">{t('job.seed')} {s.seed}</span>
                </div>
                <SampleLoss sample={s}/><p className="text-slate-500 dark:text-slate-400 line-clamp-2" title={s.prompt}>{s.prompt}</p>
              </div>
            </div>
          ))}
        </div></section>
      )}

      {activeTab === 'checkpoints' && <section className="job-checkpoints" aria-label={text('训练权重与恢复状态','Weights and training states')}>
        {checkpoints.length === 0 ? <EmptyState icon={Layers} title={t('job.noCheckpoints')}/> : checkpoints.map((cp,index) => <article key={`${cp.path}-${index}`} className="job-checkpoint">
          <div className="job-checkpoint-file"><strong>{cp.path.replace(/\\/g,'/').split('/').pop()}</strong><div><span>{checkpointKindLabel(cp.kind)}{cp.ema ? ' · EMA' : ''}</span><span>{text('步','Step')} {cp.step}</span>{stepsPerEpoch && <span>{text('轮','Epoch')} {(cp.step/stepsPerEpoch).toFixed(2).replace(/\.00$/,'')}</span>}<span>{formatBytes(cp.size)}</span></div></div>
          <div className="job-checkpoint-actions">{cp.kind === 'full' ? <button type="button" disabled={resuming || !configSnapshot} onClick={() => resumeCheckpoint(cp)} className="ui-btn ui-btn-sm">{t('job.continueTraining')}</button> : cp.artifact_id ? <a href={apiUrl(`/artifacts/${cp.artifact_id}/download`)} className="ui-btn ui-btn-sm"><Download size={14}/>{t('job.download')}</a> : <span>{t('job.downloadUnavailable')}</span>}</div>
          <details className="job-checkpoint-details"><summary>{text('文件详情','File details')}</summary><dl><div><dt>{text('保存时间','Saved')}</dt><dd>{formatTime(cp.created_at)}</dd></div><div><dt>{text('本机位置','Local location')}</dt><dd><code>{cp.path}</code></dd></div></dl></details>
        </article>)}
      </section>}

      {activeTab === 'logs' && id && <JobLogView jobId={id} active live={!!job && ['running', 'pausing', 'cancelling'].includes(job.status)} recordedLevel={job?.type === 'xyz' ? null : configSnapshot?.logging?.level}/>}

      {activeTab === 'config' && (
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6">
          {configSnapshot == null ? (
            <EmptyState
              icon={Code}
              title={t('job.noConfig', '暂无配置快照')}

            />
          ) : (
            <pre className="job-config-snapshot text-xs font-mono bg-slate-50 dark:bg-slate-900 p-4 rounded-lg overflow-x-auto">
              {JSON.stringify(configSnapshot, null, 2)}
            </pre>
          )}
        </div>
      )}
      </div>
    </div>
  );
}
