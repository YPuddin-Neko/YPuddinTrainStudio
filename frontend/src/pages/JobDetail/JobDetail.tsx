import { jobTypeLabel, mergeJobEvent } from '../../utils/jobs';
import React from 'react';
import { Link, useParams, useNavigate, useSearchParams, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { apiClient } from '../../api/client';
import { Job, JobMetrics, JobSample, JobCheckpoint, ValidationPoint } from '../../api/types';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { Activity, Archive, ArchiveRestore, Layers, History, Image as ImageIcon, Terminal, Code, ArrowLeft, ArrowDown, ArrowUp, Minus, AudioLines } from 'lucide-react';
import { mergeValidationPoint, appendMetricStep } from '../../utils/metrics';
import { formatEta, formatTime } from '../../utils/format';
import { formatApiError } from '../../utils/errors';
import { projectUrl, type ProjectVersion } from '../../utils/projectVersions';
import { useWorkspaceText } from '../../utils/workspaceText';
import { JobActions, JobStatus } from '../Queue/jobPresentation';
import ConfigHelp from '../../components/ConfigHelp';
import '../Queue/queue.css';
import './job-detail.css';
import JobLogView from './JobLogView';
import JobMetricsPanel from './JobMetricsPanel';
import OverflowStrip from '../../components/OverflowStrip';
import { learningRateGroupName } from './metricPresentation';
import SampleViewer from './SampleViewer';
import ArtifactGrid from './ArtifactGrid';
import ResumePointList from './ResumePointList';
import Dialog from '../../components/Dialog';
import JobStepper from './JobStepper';
import { SlidingIndicator } from '../../components/motion';
import { useEnterAnimation } from '../../utils/motion';
import TopbarBreadcrumb from '../../components/TopbarBreadcrumb';
import { isTtsJob } from '../../api/tts';
import TtsJobOutputs from '../Tts/TtsJobOutputs';
import { ttsAction, ttsJobError } from '../Tts/ttsJobActions';
import { TtsJobLineage } from '../Tts/TtsTaskActions';

type VersionedJob = Job & { version_id?: string | null; latest: Job['latest'] & {loss_mean?:number|null; loss_count?:number|null; loss_mean_scope?:string|null} };
type PendingMetrics = { id: string; pending: boolean; steps: Record<string, any>[]; validation: ValidationPoint[] };

function mergeSamples(previous: JobSample[], incoming: JobSample[]): JobSample[] {
  // Step and epoch triggers may share step/seed while saving different image files.
  const samples = new Map(previous.map(sample => [sample.url, sample]));
  for (const sample of incoming) samples.set(sample.url, { ...samples.get(sample.url), ...sample });
  return [...samples.values()].sort((a, b) => a.step - b.step || a.created_at - b.created_at || a.prompt_index - b.prompt_index);
}

function StatCard({ label, value, hint, detail }: { label: string; value: React.ReactNode; hint?: string; detail?: React.ReactNode }) {
  const text = useWorkspaceText();
  return <div className={`job-stat${hint ? ' has-help' : ''}`}>
    <div className="job-stat-label">{label}</div>
    {hint && <span className="job-stat-help"><ConfigHelp label={text(`${label} · 说明`, `About ${label}`)}>{hint}</ConfigHelp></span>}
    <div className="job-stat-value">{value}</div>{detail}
  </div>;
}

/** What each learning rate in the summary card belongs to, and how an adaptive optimizer sets it. */
function learningRateHelp(groups: string[], algo: unknown, optimizer: unknown, text: (zh: string, en: string) => string): string | undefined {
  const has = (...names: string[]) => names.some(name => groups.includes(name));
  const lines: string[] = [];
  if (has('down', 'up')) lines.push(algo === 'tlora'
    ? text('down、up：T-LoRA 的两个低秩矩阵。', 'down, up: the two low-rank matrices of T-LoRA.')
    : text('down、up：LoRA 的两个低秩矩阵。', 'down, up: the two low-rank matrices of LoRA.'));
  if (has('lambda')) lines.push(text('λ：T-LoRA 正交初始化时每个秩的强度。', 'λ: the strength of each rank in orthogonal T-LoRA.'));
  if (has('rotation')) lines.push(text('rotation：OrthoLoRA 在主方向之间的旋转。', "rotation: OrthoLoRA's rotation among the main directions."));
  if (has('scale')) lines.push(text('scale：OrthoLoRA 对各主方向长度的缩放。', "scale: OrthoLoRA's rescaling of each main direction."));
  if (has('w1', 'w2')) lines.push(algo === 'loha'
    ? text('w1、w2：LoHa 的两组低秩矩阵。', 'w1, w2: the two low-rank pairs of LoHa.')
    : text('w1、w2：LoKr 把权重拆成的两个矩阵，w1 较小，w2 较大。', 'w1, w2: the two matrices LoKr splits a weight into; w1 is the smaller one.'));
  if (has('scalar')) lines.push(text('scalar：LoKr 的整体缩放系数。', 'scalar: the overall LoKr scale.'));
  if (has('dora')) lines.push(text('DoRA：单独训练的幅度参数，按计算方向决定每个输入或输出通道的强度。', 'DoRA: magnitudes trained on their own, the strength of each input or output channel along the DoRA axis.'));
  if (has('weight')) lines.push(text('weight：Full 方式直接训练的权重。', 'weight: the weights Full trains directly.'));
  if (has('backbone')) lines.push(text('backbone：主模型。', 'backbone: the main model.'));
  if (has('text_encoder')) lines.push(text('text_encoder：文本编码器。', 'text_encoder: the text encoder.'));
  if (has('text_encoder_2')) lines.push(text('text_encoder_2：第二个文本编码器。', 'text_encoder_2: the second text encoder.'));
  const key = typeof optimizer === 'string' ? optimizer.toLowerCase() : '';
  const adaptive = key.includes('prodigy') ? text('Prodigy 会自动调整学习率，这里是调整后的实际值。', 'Prodigy adjusts these rates itself; these are the rates it uses.')
    : key.includes('automagic') ? text('Automagic 会逐个参数调整学习率，这里是平均值。', 'Automagic adjusts the rate of each parameter; these are the means.') : '';
  if (!lines.length && !adaptive) return undefined;
  return [lines.length ? text('每个参数组当前的学习率：', 'The current learning rate of each parameter group:') : '', ...lines, adaptive].filter(Boolean).join('\n');
}

const tinyChange = new Intl.NumberFormat('en-US', { maximumSignificantDigits: 2, maximumFractionDigits: 20, useGrouping: false });

/** How a live value moved since the previous reading: down in green, up in red. */
function StepChange({ delta, text }: { delta: number | null; text: (zh: string, en: string) => string }) {
  if (delta == null || !Number.isFinite(delta)) return null;
  const direction = delta < 0 ? 'down' : delta > 0 ? 'up' : 'flat';
  const size = Math.abs(delta);
  // Plain decimals throughout; a change below 0.0001 keeps two significant digits (0.0000025).
  const amount = size !== 0 && size < 1e-4 ? tinyChange.format(size) : size.toFixed(4);
  const Icon = direction === 'down' ? ArrowDown : direction === 'up' ? ArrowUp : Minus;
  const said = direction === 'down' ? text(`比上一次下降 ${amount}`, `Down ${amount} from the previous reading`)
    : direction === 'up' ? text(`比上一次上升 ${amount}`, `Up ${amount} from the previous reading`) : text('与上一次持平', 'Same as the previous reading');
  return <div className="job-stat-change" data-direction={direction} title={said}>
    <Icon size={12} aria-hidden="true"/><span aria-hidden="true">{direction === 'flat' ? '0' : amount}</span><span className="sr-only">{said}</span>
  </div>;
}

/** The change of a live value since the step before; `fallback` gives the earlier value before a live step arrives. */
function useStepChange(step: number | null | undefined, value: number | null | undefined, fallback: number | null): number | null {
  const last = React.useRef<{ step: number; value: number } | null>(null);
  const [previous, setPrevious] = React.useState<{ step: number; value: number } | null>(null);
  React.useEffect(() => {
    if (step == null || typeof value !== 'number' || !Number.isFinite(value)) return;
    const seen = last.current;
    if (seen && seen.step !== step) setPrevious(seen);
    last.current = { step, value };
  }, [step, value]);
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  const base = previous && previous.step !== step ? previous.value : fallback;
  return base == null || !Number.isFinite(base) ? null : value - base;
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
  const { t } = useTranslation();
  const text = useWorkspaceText();
  const navigate = useNavigate();
  const location = useLocation();
  const [params, setParams] = useSearchParams();
  const [dataError, setDataError] = React.useState('');
  const [actionError, setActionError] = React.useState('');
  const [resuming, setResuming] = React.useState(false);

  const [job, setJob] = React.useState<VersionedJob | null>(null);
  const jobKind = React.useRef<{ id: string; type: string } | null>(null);
  const ttsProjectionRequest = React.useRef<AbortController | null>(null);
  React.useEffect(() => () => ttsProjectionRequest.current?.abort(), [id]);
  const refreshTtsProjection = () => {
    if (!id) return;
    ttsProjectionRequest.current?.abort();
    const controller = new AbortController(); ttsProjectionRequest.current = controller;
    setJob(previous => previous?.id === id ? { ...previous, allowed_actions: undefined, action_reasons: undefined } : previous);
    void apiClient.get<VersionedJob>(`/jobs/${encodeURIComponent(id)}`, { silent: true, signal: controller.signal }).then(updated => {
      if (!controller.signal.aborted && updated.id === id) setJob(previous => previous?.id === id ? updated : previous);
    }).catch(() => {});
  };
  const [restoring, setRestoring] = React.useState(false);
  const restore = async () => {
    if (!job || restoring) return;
    const permission = ttsAction(job, 'unarchive', text);
    if (!permission.allowed) { setActionError(permission.reason); return; }
    setRestoring(true); setActionError('');
    try { const updated = await apiClient.patch<Job>(`/jobs/${encodeURIComponent(job.id)}`, { archived: false }, { silent: true }); if (isTtsJob(job)) ttsProjectionRequest.current?.abort(); setJob(current => current?.id === job.id ? isTtsJob(job) ? updated : { ...current, archived_at: null } : current); }
    catch (failure) { setActionError(isTtsJob(job) ? ttsJobError(failure) : formatApiError(failure)); }
    finally { setRestoring(false); }
  };
  const [clock, setClock] = React.useState(() => Date.now() / 1000);
  React.useEffect(() => {
    if (!job?.started_at || !['running','pausing','cancelling'].includes(job.status)) return;
    const timer = window.setInterval(() => setClock(Date.now() / 1000), 1000);
    return () => window.clearInterval(timer);
  }, [job?.started_at, job?.status]);
  const [resolvedVersion, setResolvedVersion] = React.useState<{ projectId: string; versionId: string; name: string } | null>(null);
  const [metrics, setMetrics] = React.useState<JobMetrics | null>(null);
  const metricsRequestRef = React.useRef<PendingMetrics | null>(null);
  const [samples, setSamples] = React.useState<JobSample[]>([]);
  const [samplesLoaded, setSamplesLoaded] = React.useState(false);
  const [selectedSample, setSelectedSample] = React.useState<string | null>(null);
  const [checkpoints, setCheckpoints] = React.useState<JobCheckpoint[]>([]);
  const [checkpointsLoaded, setCheckpointsLoaded] = React.useState(false);
  const [deleting, setDeleting] = React.useState<JobCheckpoint[] | null>(null);
  const [deleteBusy, setDeleteBusy] = React.useState(false);
  const [deleteError, setDeleteError] = React.useState('');
  const [deleteProgress, setDeleteProgress] = React.useState({ done: 0, total: 0 });
  const deleteJobId = React.useRef(id); deleteJobId.current = id;
  React.useEffect(() => { setDeleting(null); setDeleteBusy(false); setDeleteError(''); }, [id]);
  const [configSnapshot, setConfigSnapshot] = React.useState<any>(null);
  const [sampleProgress, setSampleProgress] = React.useState<{ step: number; promptIndex: number; prompts: number; done: number; total: number } | null>(null);

  const requestedTab = params.get('tab') || '';
  const allowedTabs = job?.type === 'tts_sample' ? ['audio', 'logs', 'config'] : job?.type === 'tts_train' ? ['metrics', 'audio', 'logs', 'config'] : job?.type === 'xyz' ? ['logs', 'config'] : ['metrics', 'samples', 'checkpoints', 'states', 'logs', 'config'];
  const activeTab = allowedTabs.includes(requestedTab) ? requestedTab : job?.type === 'tts_sample' ? 'audio' : job?.type === 'xyz' ? 'logs' : 'metrics';
  const tabPanel = useEnterAnimation<HTMLDivElement>(activeTab, { skipFirst: true });
  // Tabs replace the entry so Back leaves the job instead of stepping through tabs.
  const setActiveTab = (tab: string) => { const next = new URLSearchParams(params); next.set('tab', tab); setParams(next, { replace: true, state: location.state }); };
  const samplesRequestRef = React.useRef<AbortController | null>(null);
  const refreshSamples = React.useCallback(async () => {
    if (!id || jobKind.current?.id === id && isTtsJob(jobKind.current)) return;
    samplesRequestRef.current?.abort();
    const controller = new AbortController(); samplesRequestRef.current = controller;
    try {
      const history = await apiClient.get<JobSample[]>(`/jobs/${id}/samples`, { signal: controller.signal });
      if (!controller.signal.aborted) setSamples(previous => mergeSamples(previous, history));
    } catch (error) { if (!controller.signal.aborted) console.error(error); }
    finally { if (!controller.signal.aborted) setSamplesLoaded(true); }
  }, [id]);

  // Reset route-specific state and ignore responses from a previous task.
  React.useEffect(() => {
    if (!id) return;
    const controller = new AbortController();
    const options = { signal: controller.signal };
    const ignoreAbort = (error: Error) => { if (!controller.signal.aborted) setDataError(formatApiError(error)); };
    const pending: PendingMetrics = { id, pending: true, steps: [], validation: [] };
    metricsRequestRef.current = pending;
    const finishMetrics = (snapshot: JobMetrics) => {
      if (controller.signal.aborted || metricsRequestRef.current !== pending) return;
      // Events arriving during the history request must survive its older snapshot.
      let next = pending.steps.sort((a, b) => a.step - b.step).reduce(appendMetricStep, snapshot);
      for (const point of pending.validation) next = { ...next, validation: mergeValidationPoint(next.validation || [], point) };
      pending.pending = false; pending.steps = []; pending.validation = [];
      setMetrics(next);
    };
    setDataError(''); setJob(null); setMetrics(null); setSamples([]); setSamplesLoaded(false); setSelectedSample(null); setCheckpoints([]); setCheckpointsLoaded(false); setConfigSnapshot(null); setSampleProgress(null);
    jobKind.current = null;
    apiClient.get<VersionedJob>(`/jobs/${id}`, options).then(loaded => {
      if (controller.signal.aborted) return;
      jobKind.current = { id: loaded.id, type: loaded.type }; setJob(loaded);
      if (!isTtsJob(loaded)) {
        void refreshSamples();
        void apiClient.get<JobCheckpoint[]>(`/jobs/${id}/checkpoints`, options).then(setCheckpoints).catch(ignoreAbort).finally(() => { if (!controller.signal.aborted) setCheckpointsLoaded(true); });
      }
    }).catch(ignoreAbort);
    apiClient.get<JobMetrics>(`/jobs/${id}/metrics`, options).then(finishMetrics).catch(error => {
      ignoreAbort(error);
      finishMetrics({ steps: [], loss: [], loss_ema: [], lr: {}, grad_norm: [], vram_mb: [], it_s: [], validation: [] });
    });
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
      setJob((prev) => prev ? isTtsJob(prev) ? { ...mergeJobEvent(prev, data), allowed_actions: undefined, action_reasons: undefined } : mergeJobEvent(prev, data) : null);
      // A resumed run also brings its pause history.
      if (isTtsJob(jobKind.current)) refreshTtsProjection();
      else if (['completed', 'failed', 'cancelled', 'paused', 'running'].includes(data.status)) {
        void refreshSamples();
        void apiClient.get<VersionedJob>(`/jobs/${id}`, {silent:true}).then(updated => setJob(previous => previous?.id === updated.id ? updated : previous)).catch(() => {});
      }
    }
  });

  useEventStream(EVENT_TYPES.QUEUE_CHANGED, () => {
    if (!id || !isTtsJob(jobKind.current)) return;
    refreshTtsProjection();
  });

  useEventStream(EVENT_TYPES.JOB_STEP, (data: any) => {
    if (data.job_id === id) {
      setJob((prev) => prev ? mergeJobEvent(prev, data) : null);
      const pending = metricsRequestRef.current;
      if (!pending || pending.id !== id) return;
      if (pending.pending) pending.steps.push(data);
      else setMetrics((prev) => prev ? appendMetricStep(prev, data) : prev);
    }
  });

  useEventStream(EVENT_TYPES.JOB_PHASE, (data: any) => {
    if (data.job_id === id) setJob((prev) => prev ? mergeJobEvent(prev, data) : null);
  });
  useEventStream(EVENT_TYPES.JOB_CHECKPOINT, (data: any) => {
    if (data.job_id === id && !isTtsJob(jobKind.current)) apiClient.get<JobCheckpoint[]>(`/jobs/${id}/checkpoints`).then(setCheckpoints).catch(console.error);
  });

  useEventStream(EVENT_TYPES.JOB_SAMPLE, (sample: JobSample & { job_id?: string; ts?: number }) => {
    if (isTtsJob(jobKind.current)) return;
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
    const pending = metricsRequestRef.current;
    if (!pending || pending.id !== id) return;
    const point = { step: data.step, per_t: data.per_t || {}, mean: data.mean };
    if (pending.pending) { pending.validation.push(point); return; }
    setMetrics((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        validation: mergeValidationPoint(prev.validation || [], point),
      };
    });
  });

  const stepsPerEpoch = job?.progress?.steps_per_epoch;

  // Exported weights are outputs; full training states are resume points with their own tab.
  const outputs = checkpoints.filter(item => item.kind !== 'full');
  const resumePoints = checkpoints.filter(item => item.kind === 'full');
  const requestCheckpointDelete = (items: JobCheckpoint[]) => {
    setDeleteError('');
    setDeleting([...new Map(items.map(item => [item.path, item])).values()]);
  };
  const removeCheckpoint = async () => {
    if (!deleting?.length || !id || deleteBusy) return;
    const jobId = id;
    setDeleteBusy(true); setDeleteError(''); setDeleteProgress({ done: 0, total: deleting.length });
    const failed: JobCheckpoint[] = [];
    const errors: string[] = [];
    for (const [index, item] of deleting.entries()) {
      try {
        await apiClient.delete(`/jobs/${encodeURIComponent(jobId)}/checkpoints`, { params: { path: item.path }, silent: true });
        if (deleteJobId.current === jobId) setCheckpoints(previous => previous.filter(point => point.path !== item.path));
      } catch (failure) {
        failed.push(item);
        errors.push(`${item.path.replace(/\\/g, '/').split('/').pop()}: ${formatApiError(failure)}`);
      }
      if (deleteJobId.current === jobId) setDeleteProgress({ done: index + 1, total: deleting.length });
    }
    if (deleteJobId.current === jobId) {
      setDeleting(failed.length ? failed : null); setDeleteError(errors.join('\n')); setDeleteBusy(false);
    }
  };

  const tabs = [
    { key: 'metrics', icon: Activity, label: t('job.tabMetrics') },
    { key: 'audio', icon: AudioLines, label: job?.type === 'tts_train' ? text('权重与试听', 'Weights and preview') : text('试听音频', 'Preview audio') },
    { key: 'samples', icon: ImageIcon, label: `${t('job.tabSamples')} (${samples.length})` },
    { key: 'checkpoints', icon: Layers, label: `${t('job.tabCheckpoints')} (${outputs.length})` },
    { key: 'states', icon: History, label: `${text('恢复点', 'Resume points')} (${resumePoints.length})` },
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
  // Before a live step arrives, the last loss compares with the reading before it, and the running mean with the
  // mean before the last step.
  const currentLoss = job?.latest?.loss;
  const lastRecorded = recordedLosses.at(-1);
  const previousLoss = typeof currentLoss === 'number' && recordedLosses.length ? (lastRecorded === currentLoss ? recordedLosses.at(-2) ?? null : lastRecorded ?? null) : null;
  const lossCount = Number(job?.latest?.loss_count);
  const previousMean = typeof job?.latest?.loss_mean === 'number' && typeof currentLoss === 'number' && lossCount > 1 ? (job.latest.loss_mean * lossCount - currentLoss) / (lossCount - 1) : null;
  const lossChange = useStepChange(job?.progress?.step, currentLoss, previousLoss);
  const meanChange = useStepChange(job?.progress?.step, meanLoss, previousMean);
  const meanScope = job?.latest?.loss_mean != null ? (job.latest.loss_mean_scope === 'since_resume' ? text('从此次恢复训练起，所有已完成训练步的损失平均值。','Mean loss over completed steps since this training was resumed.') : text('所有已完成训练步的损失平均值。','Mean loss over all completed optimizer steps.')) : text('旧任务没有完整累计值，显示已有日志中训练步的平均值。','This legacy run has no complete accumulator; this is the mean of recorded steps.');
  const learningRates = Object.entries(job?.latest?.lr || {}).filter((entry): entry is [string, number] => typeof entry[1] === 'number' && Number.isFinite(entry[1]));
  // Two rows; groups are listed by name, so a DoRA rate sits alone above the w1 / w2 (or down / up) pair.
  const rateSplit = Math.floor(learningRates.length / 2);
  const learningRateRows = [learningRates.slice(0, rateSplit), learningRates.slice(rateSplit)];
  const epochProgress = isTtsJob(job)
    ? typeof job?.progress?.epoch === 'number' && Number.isFinite(job.progress.epoch) ? String(Number(job.progress.epoch.toFixed(2))) : '—'
    : stepsPerEpoch && job?.progress?.step != null ? (job.progress.step / stepsPerEpoch).toFixed(2).replace(/\.00$/, '') : job?.progress?.epoch != null ? String(job.progress.epoch + 1) : '—';
  const totalEpochs = stepsPerEpoch && job?.progress?.total_steps != null ? Math.ceil(job.progress.total_steps / stepsPerEpoch) : configSnapshot?.loop?.epochs;
  const elapsed = job?.started_at != null ? Math.max(0, (job.finished_at ?? (['running','pausing','cancelling'].includes(job.status) ? clock : job.started_at)) - job.started_at) : null;
  const configurationName = job?.version_name || versionName || job?.name;

  return (
    <div className="job-monitor task-workspace" data-view={activeTab} data-testid="job-detail-page">
      <header className="job-monitor-bar">
        <TopbarBreadcrumb>
          <nav className="job-monitor-breadcrumb" aria-label={text('当前位置', 'Current location')}>
            <Link to="/queue">{text('任务队列', 'Job queue')}</Link>
            {isTtsJob(job) && <><span aria-hidden="true">/</span><Link to="/projects">{text('语音训练', 'Speech training')}</Link></>}
            {job?.project_id && <><span aria-hidden="true">/</span><Link to={resultsUrl} title={text('打开版本训练结果', 'Open version results')}>{job.project_name || job.project_id}{versionLabel && <span className="job-monitor-version"> · {versionLabel}</span>}</Link></>}
          </nav>
        </TopbarBreadcrumb>
        <div className="job-monitor-identity">
          <div className="job-monitor-title"><button type="button" className="ui-btn ui-btn-sm job-monitor-back" onClick={goBack}><ArrowLeft size={14}/>{text('返回', 'Back')}</button><h1>{job?.name || text('读取任务…', 'Loading job…')}</h1>{job && <JobStatus status={job.status}/>}</div>
          {job && <JobActions key={job.id} job={job} onUpdated={updated => { if (isTtsJob(job)) ttsProjectionRequest.current?.abort(); if (updated.id === job.id) setJob(updated); else navigate(`/jobs/${updated.id}`, { replace: true, state: location.state }); }}/>}
        </div>
      </header>
      <dl className="job-run-metadata" aria-label={text('运行信息','Run information')}>
        <div><dt>{text('任务类型','Job type')}</dt><dd>{job ? jobTypeLabel(job, text) : '—'}</dd></div>
        <div><dt>{text('开始时间','Started')}</dt><dd>{formatTime(job?.started_at)}</dd></div>
        <div><dt>{job?.type === 'train' ? text('训练时长','Training elapsed') : text('运行时长','Elapsed')}</dt><dd>{formatEta(elapsed)}</dd></div>
        <div><dt>{job?.type === 'train' ? text('训练配置','Training configuration') : text('任务配置','Task configuration')}</dt><dd>{configurationName ? `${configurationName} · ${text('参数快照','snapshot')}` : '—'}</dd></div>
        <div><dt>{text('运行 ID','Run ID')}</dt><dd><code>{job?.id || id}</code></dd></div>
        {job && isTtsJob(job) && <TtsJobLineage key={job.id} job={job}/>}
      </dl>
      {dataError && <div className="task-error" role="alert">{dataError}</div>}
      {actionError && <div role="alert" className="task-error">{actionError}</div>}
      {job?.archived_at != null && <div className="job-archived" role="status"><Archive size={15} aria-hidden="true"/><span>{text('此任务已归档，不再显示在队列和项目结果中，权重和记录仍保留。', 'This job is archived: it is hidden from the queue and project results; its weights and records are kept.')}</span>
        <button type="button" className="ui-btn ui-btn-sm" disabled={restoring || !ttsAction(job, 'unarchive', text).allowed} title={ttsAction(job, 'unarchive', text).reason || undefined} onClick={() => void restore()}><ArchiveRestore size={14}/>{text('恢复到训练历史', 'Restore to History')}</button>{!ttsAction(job, 'unarchive', text).allowed && <span>{ttsAction(job, 'unarchive', text).reason}</span>}</div>}
      {job?.error && <div role="alert" className="job-failure"><div><strong>{job.type === 'train' ? text('训练失败', 'Training failed') : text('任务失败', 'Job failed')}</strong><p>{job.error}</p></div>{activeTab !== 'logs' && <button type="button" className="ui-btn ui-btn-sm" onClick={() => setActiveTab('logs')}><Terminal size={14}/>{text('查看日志', 'Open log')}</button>}</div>}
      {/* 1. 头部指标与阶段时间线 */}
      {job?.type !== 'xyz' && job?.type !== 'tts_sample' && <div className="job-monitor-summary bg-white dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700 space-y-3">
        <OverflowStrip snap role="group" label={text('训练核心指标','Training metrics')} containerClassName="job-stat-strip" className="job-stat-grid"
          pageLabels={{ previous: text('上一组指标', 'Previous metrics'), next: text('下一组指标', 'Next metrics') }}>
          <StatCard label={text('步数','Steps')} value={`${job?.progress?.step ?? '—'} / ${job?.progress?.total_steps ?? '—'}`}/>
          <StatCard label={text('轮次','Epochs')} value={isTtsJob(job) && totalEpochs == null ? epochProgress : `${epochProgress} / ${totalEpochs ?? '—'}`}/>
          <StatCard label="Loss" value={lossNumber(job?.latest?.loss)} detail={<StepChange delta={lossChange} text={text}/>}/>
          <StatCard label={text('平均 Loss','Mean loss')} value={lossNumber(meanLoss)} hint={meanScope} detail={<StepChange delta={meanChange} text={text}/>}/>
          <StatCard label={text('学习率','Learning rate')} hint={learningRates.length ? learningRateHelp(learningRates.map(([name]) => name), configSnapshot?.adapter?.algo, configSnapshot?.optimizer?.type, text) : undefined}
            value={learningRates.length > 1 ? <span className="job-learning-rates">{learningRateRows.map((row, index) => <span key={index}>{row.map(([name, rate]) => <span key={name}><small>{learningRateGroupName(name)}</small>{rate.toExponential(2)}</span>)}</span>)}</span> : learningRates.length ? learningRates[0][1].toExponential(2) : '—'}/>
          <StatCard label={t('job.speed')} value={job?.progress?.it_s != null ? `${Number(job.progress.it_s).toFixed(2)} it/s` : '—'}/>
          <StatCard label={t('job.eta')} value={job?.status === 'completed' ? '0s' : formatEta(job?.progress?.eta_s)}/>
        </OverflowStrip>

        {job && !isTtsJob(job) && <JobStepper status={job.status} phase={job.progress?.phase || ''} progress={job.progress}/>}

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
      <OverflowStrip className="job-monitor-tabs ui-tabs" label={text('任务详情分区', 'Job details tabs')} activeKey={activeTab}>
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
        </OverflowStrip>

      {/* 3. 详细内容区域 */}
      <div ref={tabPanel} role="tabpanel" id={`job-panel-${activeTab}`} aria-labelledby={`job-tab-${activeTab}`}>
      {activeTab === 'metrics' && <JobMetricsPanel metrics={metrics} stepsPerEpoch={stepsPerEpoch} vramMetric={job?.progress?.vram_metric} device={job?.progress?.device} status={job?.status}/>}

      {activeTab === 'audio' && job && <TtsJobOutputs key={job.id} jobId={job.id} training={job.type === 'tts_train'} live={['queued', 'scheduled', 'running', 'cancelling'].includes(job.status)}/>}

      {activeTab === 'samples' && <SampleViewer samples={samples} stepsPerEpoch={stepsPerEpoch} loaded={samplesLoaded} selected={selectedSample} onSelect={setSelectedSample}/>}

      {activeTab === 'checkpoints' && <ArtifactGrid key={id} checkpoints={outputs} stepsPerEpoch={stepsPerEpoch} loaded={checkpointsLoaded}
        onOpenSample={url => { setSelectedSample(url); setActiveTab('samples'); }} deleting={deleteBusy} onDelete={item => requestCheckpointDelete([item])} onDeleteMany={requestCheckpointDelete}/>}

      {activeTab === 'states' && <ResumePointList key={id} points={resumePoints} stepsPerEpoch={stepsPerEpoch} loaded={checkpointsLoaded} resumeFrom={job?.resume_from ?? null} jobStatus={job?.status}
        resuming={resuming} canResume={!!configSnapshot} onResume={point => void resumeCheckpoint(point)} deleting={deleteBusy} onDelete={item => requestCheckpointDelete([item])} onDeleteMany={requestCheckpointDelete}/>}

      {!!deleting?.length && <Dialog title={deleting[0].kind === 'full' ? text('删除恢复点', deleting.length > 1 ? 'Delete resume points' : 'Delete resume point') : text('删除产物', deleting.length > 1 ? 'Delete outputs' : 'Delete output')} onClose={() => setDeleting(null)} closeDisabled={deleteBusy}>
        <div className="checkpoint-delete">
          <p>{deleting.length > 1
            ? deleting[0].kind === 'full'
              ? text(`将从磁盘删除所选的 ${deleting.length} 个恢复点，删除后无法再从这些位置继续训练。`, `Delete ${deleting.length} selected resume points from disk? Training can no longer continue from these points.`)
              : text(`将从磁盘删除所选的 ${deleting.length} 个产物，无法恢复。`, `Delete ${deleting.length} selected outputs from disk? This cannot be undone.`)
            : deleting[0].kind === 'full'
              ? text(`将从磁盘删除恢复点“${deleting[0].path.replace(/\\/g, '/').split('/').pop()}”（第 ${deleting[0].step} 步）。删除后无法再从这一步继续训练。`, `The resume point at step ${deleting[0].step} will be removed from disk; training can no longer continue from it.`)
              : text(`将从磁盘删除“${deleting[0].path.replace(/\\/g, '/').split('/').pop()}”（第 ${deleting[0].step} 步），无法恢复。`, `The file saved at step ${deleting[0].step} will be removed from disk and cannot be restored.`)}</p>
          {deleting.length > 1 && <ul className="checkpoint-delete-list">{deleting.map(item => <li key={item.path}>{item.path.replace(/\\/g, '/').split('/').pop()}</li>)}</ul>}
          {deleteError && <p role="alert" className="studio-error">{deleteError}</p>}
          <div className="checkpoint-delete-actions"><button type="button" className="ui-btn" disabled={deleteBusy} onClick={() => setDeleting(null)}>{text('取消', 'Cancel')}</button>
            <button type="button" className="ui-btn ui-btn-primary ui-btn-danger" disabled={deleteBusy} onClick={() => void removeCheckpoint()}>{deleteBusy ? text(`正在删除 ${deleteProgress.done}/${deleteProgress.total}`, `Deleting ${deleteProgress.done}/${deleteProgress.total}`) : text('确认删除', 'Delete')}</button></div>
        </div>
      </Dialog>}

      {activeTab === 'logs' && id && <JobLogView jobId={id} active live={!!job && ['running', 'pausing', 'cancelling'].includes(job.status)} exitCode={job?.exit_code} recordedLevel={job?.type === 'xyz' ? null : configSnapshot?.logging?.level}/>}

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
