import React from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { Loader2, Play } from 'lucide-react';
import Dialog from '../../components/Dialog';
import { ApiError, type Job } from '../../api/types';
import { ttsApi, type TtsConfigResponse, type TtsEngine, type TtsIssue, type TtsSourceChanged, type TtsSourcesResponse, type TtsTrainingBody, type TtsValidationReport } from '../../api/tts';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { formatApiError } from '../../utils/errors';
import { gpuDeviceLabel } from '../../utils/gpuDevices';
import { useWorkspaceText } from '../../utils/workspaceText';
import TtsGpuPicker from './TtsGpuPicker';
import GptSovitsValidationResult from './GptSovitsValidationResult';
import { gptSovitsFieldCopy } from './gptSovitsVersionFields';
import { fieldCopy, fields, type TtsField } from './ttsVersionFields';
import './tts-training.css';

type Props = {
  projectId: string; versionId: string; readOnly: boolean; draftDirty: boolean; engine?: TtsEngine;
  ensureSaved: () => Promise<TtsConfigResponse | null>; onStarted: (job: Job) => void;
};
type Attempt = { scope: { project_id: string; version_id: string }; key: string; body: TtsTrainingBody; blocked: boolean; message?: string };
const environmentChecks = [['python', 'Python 解释器', 'Python executable'], ['upstream', '训练器版本', 'Trainer version'], ['model', '模型资产', 'Model assets'], ['dependencies', '依赖环境', 'Dependencies'], ['cuda', 'CUDA 显卡', 'CUDA GPU'], ['bf16', 'BF16 支持', 'BF16 support'], ['tokenizer', '分词器', 'Tokenizer'], ['lora_targets', 'LoRA 目标层', 'LoRA targets']] as const;
const storageKey = (pid: string, vid: string) => `tts-training-request:v1:${pid}:${vid}`;
const sameDevices = (left: string[] | null, right: string[]) => left !== null && left.length === right.length && left.every((device, index) => device === right[index]);
function readAttempt(pid: string, vid: string): Attempt | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(storageKey(pid, vid)) || 'null') as Attempt | null;
    if (!value || value.scope.project_id !== pid || value.scope.version_id !== vid
      || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(value.key)
      || !Number.isInteger(value.body.revision) || value.body.revision < 1 || !Number.isInteger(value.body.data_revision) || value.body.data_revision < 1
      || typeof value.body.name !== 'string' || !Array.isArray(value.body.gpu_devices)
      || value.body.gpu_devices.length > 1 || value.body.gpu_devices.some(device => !/^cuda:\d+$/.test(device))) return null;
    return value;
  } catch { return null; }
}

export default function TtsTrainingActions(props: Props) {
  return <TrainingActions key={`${props.projectId}:${props.versionId}:${props.engine || 'voxcpm1.5'}`} {...props}/>;
}

function TrainingActions({ projectId: pid, versionId: vid, readOnly, draftDirty, ensureSaved, onStarted, engine = 'voxcpm1.5' }: Props) {
  const text = useWorkspaceText();
  const queryClient = useQueryClient();
  const [open, setOpen] = React.useState(false), [resetting, setResetting] = React.useState(false);
  const [attempt, setAttempt] = React.useState(() => readAttempt(pid, vid));
  const [report, setReport] = React.useState<TtsValidationReport | null>(null);
  const [reportGpu, setReportGpu] = React.useState<string[] | null>(null);
  const [stale, setStale] = React.useState(false);
  const [phase, setPhase] = React.useState<'idle' | 'checking' | 'starting'>('idle');
  const [error, setError] = React.useState('');
  const [issues, setIssues] = React.useState<TtsIssue[]>([]);
  const [name, setName] = React.useState(attempt?.body.name || text('语音训练', 'Speech training'));
  const [gpu, setGpu] = React.useState<string[]>(attempt?.body.gpu_devices || []);
  const [gpuValid, setGpuValid] = React.useState(false);
  const controller = React.useRef<AbortController | null>(null), generation = React.useRef(0), locked = React.useRef(false);
  const latest = React.useRef({ readOnly, draftDirty, gpu }); latest.current = { readOnly, draftDirty, gpu };
  const alive = React.useRef(true);
  React.useEffect(() => { const cycle = generation; alive.current = true; return () => { alive.current = false; cycle.current++; controller.current?.abort(); }; }, []);
  React.useEffect(() => { if (draftDirty && report) setStale(true); }, [draftDirty, report]);
  useEventStream<TtsSourceChanged>(EVENT_TYPES.TTS_SOURCE_CHANGED, event => {
    if (event.scope.project_id === pid && event.scope.version_id === vid && report && event.data_revision !== report.data_revision) setStale(true);
  });
  const owns = (scope: { project_id: string; version_id: string }) => scope.project_id === pid && scope.version_id === vid;
  const publishLatest = (config: TtsConfigResponse, sources: TtsSourcesResponse) => {
    if (owns(config.scope)) queryClient.setQueryData(['tts-version-config', pid, vid], config);
    if (owns(sources.scope)) queryClient.setQueryData(['tts-sources', pid, vid], sources);
  };
  const changedMessage = () => text('参数、数据或运行显卡已变化，请重新检查后启动。', 'Parameters, data, or GPU selection changed. Check again before starting.');
  const showFailure = (failure: unknown) => {
    const message = formatApiError(failure);
    setError(message);
    setIssues(failure instanceof ApiError && Array.isArray(failure.details?.issues) ? (failure.details.issues as TtsIssue[]).filter(issue => issue.message !== message || issue.loc.length > 0) : []);
  };
  const close = () => {
    if (phase === 'starting') return;
    generation.current++; controller.current?.abort(); locked.current = false;
    setOpen(false); setPhase('idle');
  };
  const changeGpu = (devices: string[]) => {
    if (attempt || phase === 'starting' || sameDevices(latest.current.gpu, devices)) return;
    latest.current.gpu = [...devices]; setGpu([...devices]);
    setStale(true); setError(''); setIssues([]);
    if (phase === 'checking') {
      generation.current++; controller.current?.abort(); locked.current = false; setPhase('idle');
    }
  };
  const check = async (freshRequest = false) => {
    if (locked.current || latest.current.readOnly || attempt && !freshRequest) return;
    locked.current = true; const token = ++generation.current;
    controller.current?.abort(); const abort = new AbortController(); controller.current = abort;
    const checkedDevices = [...latest.current.gpu];
    setOpen(true); setPhase('checking'); setReport(null); setReportGpu(null); setStale(false); setError(''); setIssues([]);
    const active = () => alive.current && token === generation.current && !abort.signal.aborted && sameDevices(checkedDevices, latest.current.gpu);
    try {
      const saved = await ensureSaved();
      if (!active()) return;
      if (!saved) { setOpen(false); return; }
      if (!owns(saved.scope)) throw new Error(text('保存结果不属于当前版本，请重新打开版本。', 'The saved parameters belong to another version. Reopen this version.'));
      const [config, sources] = await Promise.all([ttsApi.versionConfig(pid, vid, abort.signal), ttsApi.sources(pid, vid, abort.signal)]);
      if (!active()) return;
      publishLatest(config, sources);
      if (config.config.engine !== engine || !owns(config.scope) || !owns(sources.scope) || config.revision !== saved.revision || config.data_revision !== sources.data_revision) throw new Error(changedMessage());
      const result = await ttsApi.validateVersion(pid, vid, { revision: config.revision, data_revision: sources.data_revision, gpu_devices: checkedDevices }, abort.signal);
      if (!active()) return;
      if (('engine' in result.dataset ? result.dataset.engine : 'voxcpm1.5') !== engine || ('engine' in result.environment ? result.environment.engine : 'voxcpm1.5') !== engine || !owns(result.scope) || result.revision !== config.revision || result.data_revision !== sources.data_revision) throw new Error(changedMessage());
      const devices = result.environment.devices;
      if (devices ? !sameDevices(devices.requested_devices, checkedDevices) : result.valid) throw new Error(changedMessage());
      setReport(result); setReportGpu(checkedDevices);
    } catch (failure) { if (active()) showFailure(failure); }
    finally { if (active()) { locked.current = false; setPhase('idle'); } }
  };
  const store = (value: Attempt) => { sessionStorage.setItem(storageKey(pid, vid), JSON.stringify(value)); setAttempt(value); };
  const start = async () => {
    if (locked.current || attempt?.blocked) return;
    if (!attempt && (!report?.valid || stale || !sameDevices(reportGpu, latest.current.gpu) || latest.current.readOnly || latest.current.draftDirty || !name.trim() || !gpuValid)) return;
    locked.current = true; const token = ++generation.current;
    setPhase('starting'); setError(''); setIssues([]);
    let request = attempt;
    const active = () => alive.current && token === generation.current;
    try {
      if (!request && report) {
        const [config, sources] = await Promise.all([ttsApi.versionConfig(pid, vid), ttsApi.sources(pid, vid)]);
        if (!active()) return;
        publishLatest(config, sources);
        if (config.config.engine !== engine || latest.current.readOnly || latest.current.draftDirty || !sameDevices(reportGpu, latest.current.gpu) || !owns(config.scope) || !owns(sources.scope)
          || config.revision !== report.revision || config.data_revision !== report.data_revision || sources.data_revision !== report.data_revision) {
          setStale(true); throw new Error(changedMessage());
        }
        request = { scope: { project_id: pid, version_id: vid }, key: crypto.randomUUID(), blocked: false,
          body: { revision: report.revision, data_revision: report.data_revision, name: name.trim(), gpu_devices: [...latest.current.gpu] } };
        try { store(request); }
        catch { request = null; throw new Error(text('浏览器无法保存启动请求，请允许网站存储后重试。', 'This browser cannot retain the start request. Enable site storage and try again.')); }
      }
      if (!request) return;
      // Replays must keep the original body, even after the version changes.
      const job = await ttsApi.startVersion(pid, vid, request.body, request.key);
      if (!active()) return;
      try { sessionStorage.removeItem(storageKey(pid, vid)); } catch { /* A retained request can safely replay the same job. */ }
      setAttempt(null); setOpen(false); setReport(null); onStarted(job);
    } catch (failure) {
      if (!active()) return;
      showFailure(failure);
      if (request && failure instanceof ApiError && failure.status >= 400 && failure.status < 500 && failure.code !== 'tts.request_pending') {
        const blocked = { ...request, blocked: true, message: formatApiError(failure) };
        setAttempt(blocked);
        try { sessionStorage.setItem(storageKey(pid, vid), JSON.stringify(blocked)); } catch { /* The original key remains safe to replay after reload. */ }
      }
    } finally { if (active()) { locked.current = false; setPhase('idle'); } }
  };
  const reset = async () => {
    try { sessionStorage.removeItem(storageKey(pid, vid)); }
    catch { setError(text('无法移除已保存的启动请求，请检查浏览器的网站存储权限。', 'Cannot remove the saved request. Check site storage permissions.')); setResetting(false); return; }
    setAttempt(null); setReport(null); setStale(false); setError(''); setIssues([]); setResetting(false);
    void check(true);
  };
  return <>
    <button type="button" className="ui-btn ui-btn-primary" disabled={phase !== 'idle' || readOnly && !attempt} onClick={() => { if (attempt) { setOpen(true); setError(attempt.message || ''); } else void check(); }}>
      {phase !== 'idle' ? <Loader2 size={14} className="animate-spin"/> : <Play size={14}/>}{attempt ? text('继续确认启动', 'Confirm start request') : text('检查并启动', 'Check and start')}
    </button>
    {open && <Dialog title={text('启动语音训练', 'Start speech training')} onClose={close} closeDisabled={phase === 'starting'} wide>
      <div className="tts-training-content" aria-busy={phase !== 'idle'}>
        {phase === 'checking' && <p role="status" className="tts-training-progress"><Loader2 size={16} className="animate-spin"/>{text('正在检查环境与数据，模型文件检查可能需要一些时间…', 'Checking environment and data. Reading model files can take some time…')}</p>}
        {error && <p role="alert" className="workspace-message error">{error}</p>}
        {issues.length > 0 && <IssueList issues={issues} engine={engine}/>}
        {attempt && <p role="status" className="tts-training-note">{attempt.blocked ? text('此启动请求不能继续。重新检查前，请先确认原任务状态。', 'This request cannot continue. Check the original task before starting another check.') : text('启动结果尚待确认。继续确认会使用原请求，不会重复创建任务。', 'The start result needs confirmation. Confirming reuses the original request without creating a duplicate job.')}</p>}
        {report && <ValidationResult report={report} stale={stale}/>}
        {stale && !attempt && <p role="status" className="workspace-message error">{changedMessage()}</p>}
        {phase !== 'checking' && <div className="tts-training-inputs"><label className="tts-training-name">{text('任务名称', 'Job name')}<input maxLength={200} value={name} disabled={!!attempt || phase === 'starting'} onChange={event => setName(event.target.value)} /></label><div><span className="tts-training-label">{text('运行显卡', 'Run on GPU')}</span><TtsGpuPicker value={gpu} onChange={changeGpu} onValidityChange={setGpuValid} disabled={!!attempt || phase === 'starting'} training/></div></div>}
        <div className="tts-training-actions"><button type="button" className="ui-btn" disabled={phase === 'starting'} onClick={close}>{text('关闭', 'Close')}</button>
          {attempt ? <><button type="button" className="ui-btn" disabled={phase !== 'idle'} onClick={() => setResetting(true)}>{text('重新开始检查', 'Start a new check')}</button><button type="button" className="ui-btn ui-btn-primary" disabled={phase !== 'idle' || attempt.blocked} onClick={() => void start()}>{phase === 'starting' ? text('正在确认…', 'Confirming…') : text('继续确认原请求', 'Confirm original request')}</button></>
            : <><button type="button" className="ui-btn" disabled={phase !== 'idle' || readOnly} onClick={() => void check()}>{text('重新检查', 'Check again')}</button><button type="button" className="ui-btn ui-btn-primary" disabled={phase !== 'idle' || readOnly || draftDirty || stale || !sameDevices(reportGpu, gpu) || !report?.valid || !gpuValid || !name.trim()} onClick={() => void start()}>{phase === 'starting' ? text('正在加入队列…', 'Adding to queue…') : text('加入训练队列', 'Add to training queue')}</button></>}
        </div>
      </div>
    </Dialog>}
    {resetting && <Dialog nested title={text('重新开始检查', 'Start a new check')} onClose={() => setResetting(false)}><p className="workspace-confirm-message">{text('原请求可能已经创建任务，请先查看队列。继续将放弃本地的原请求记录；之后启动会创建新的训练任务。', 'The original request may already have created a job. Check the queue first. Continuing discards its local request record; a later start creates a new training job.')}</p><div className="tts-training-actions"><button type="button" className="ui-btn" onClick={() => setResetting(false)}>{text('保留原请求', 'Keep original request')}</button><button type="button" className="ui-btn ui-btn-primary" onClick={() => void reset()}>{text('已确认，重新检查', 'Confirmed, check again')}</button></div></Dialog>}
  </>;
}

function gpuIssueMessage(issue: TtsIssue, english: boolean) {
  if (!english || issue.loc[0] !== 'gpu_devices' || issue.loc.length > 2) return issue.message;
  const device = typeof issue.loc[1] === 'string' && /^cuda:\d+$/.test(issue.loc[1]) ? issue.loc[1] : null;
  if (issue.code === 'tts.device.bf16' && device) {
    if (issue.message === `显卡 ${device} 不支持 VoxCPM 1.5 训练所需的 BF16。`) return `${gpuDeviceLabel(device)} does not support BF16, which is required for VoxCPM 1.5 training.`;
    const prefix = `无法检查 ${device} 的 BF16 支持：`;
    if (issue.message.startsWith(prefix)) return `Unable to check BF16 support on ${gpuDeviceLabel(device)}: ${issue.message.slice(prefix.length)}`;
  }
  if (issue.code === 'tts.device.unavailable') {
    const known: Record<string, string> = {
      '当前 TTS 环境没有可用的 NVIDIA CUDA 显卡。': 'No NVIDIA CUDA GPU is available in the current TTS environment.',
      'TTS 环境需要 NVIDIA CUDA 版 PyTorch。': 'The TTS environment requires the NVIDIA CUDA build of PyTorch.',
      'TTS 环境无法使用 NVIDIA CUDA，请检查驱动与所选 Python 环境。': 'NVIDIA CUDA is unavailable in the TTS environment. Check the driver and selected Python environment.',
      'TTS 任务须选择 NVIDIA CUDA 显卡。': 'Select an NVIDIA CUDA GPU for this TTS job.',
    };
    if (Object.prototype.hasOwnProperty.call(known, issue.message)) return known[issue.message];
    if (device && issue.message === `显卡 ${device} 不在当前 CUDA 可见设备中。`) return `${gpuDeviceLabel(device)} is not visible in the current CUDA environment.`;
  }
  return issue.message;
}

function IssueList({ issues, engine = 'voxcpm1.5' }: { issues: TtsIssue[]; engine?: TtsEngine }) {
  const text = useWorkspaceText(), english = text('zh', 'en') === 'en';
  const location = (loc: TtsIssue['loc']) => {
    if (loc[0] === 'gpu_devices') return [text('运行显卡', 'Run on GPU'), ...loc.slice(1).map(value => typeof value === 'string' && /^cuda:\d+$/.test(value) ? gpuDeviceLabel(value) : value)].join(' / ');
    if (engine === 'gpt-sovits-v5' && loc[0] === 'config') return gptSovitsFieldCopy(loc.slice(1).join('.'), english).label;
    if (loc[0] === 'environment' && loc[1] === 'precision') return [text('训练精度', 'Training precision'), ...loc.slice(2)].join(' / ');
    const field = loc[1];
    const environment = loc[0] === 'environment' ? environmentChecks.find(([key]) => key === field) : undefined;
    if (environment) return [text(environment[1], environment[2]), ...loc.slice(2)].join(' / ');
    return (loc[0] === 'config' && typeof field === 'string' && fields.includes(field as TtsField)
      ? [fieldCopy(field as TtsField, english).label, ...loc.slice(2)] : loc).join(' / ');
  };
  return <ul className="tts-training-issues">{issues.map((issue, index) => <li key={`${issue.code}:${index}`} data-severity={issue.severity}>{issue.loc.length > 0 && <small className="tts-training-issue-location">{location(issue.loc)}</small>}<span>{gpuIssueMessage(issue, english)}</span></li>)}</ul>;
}
const issueIdentity = (issue: TtsIssue) => JSON.stringify([issue.code, issue.loc, issue.message, issue.severity]);
function ValidationResult({ report, stale }: { report: TtsValidationReport; stale: boolean }) {
  const text = useWorkspaceText();
  const value = (number: number | null | undefined) => number == null ? text('未知', 'Unknown') : number.toLocaleString();
  const state = (key: string) => ({ available: text('可用', 'Available'), unavailable: text('不可用', 'Unavailable'), unchecked: text('未检查', 'Not checked'), disabled: text('已关闭', 'Disabled'), not_applicable: text('不适用', 'Not applicable'), ready: text('就绪', 'Ready'), blocked: text('受阻', 'Blocked'), error: text('检查失败', 'Check failed'), missing: text('未登记', 'Not registered'), checking: text('正在检查', 'Checking'), valid: text('通过', 'Passed'), invalid: text('存在无效数据', 'Contains invalid data'), stale: text('内容已变化', 'Content changed') }[key] || key);
  if ('engine' in report.dataset && report.dataset.engine === 'gpt-sovits-v5' && 'engine' in report.environment && report.environment.engine === 'gpt-sovits-v5') return <GptSovitsValidationResult report={report} dataset={report.dataset} environment={report.environment} stale={stale} renderIssues={issues => <IssueList issues={issues} engine="gpt-sovits-v5"/>}/>;
  if ('engine' in report.dataset || 'engine' in report.environment) return <p role="alert" className="workspace-message error">{text('检查结果的训练引擎不一致，请重新检查。', 'The check results use different training engines. Run the check again.')}</p>;
  const train = report.dataset.train, validation = report.dataset.validation;
  const summarized = new Set([...report.errors, ...report.warnings].map(issueIdentity));
  const remaining = (items: TtsIssue[] = [], represented: TtsIssue[] = []) => {
    const seen = new Set([...summarized, ...represented.map(issueIdentity)]);
    return items.filter(issue => { const key = issueIdentity(issue); if (seen.has(key)) return false; seen.add(key); return true; });
  };
  const detail = (content: React.ReactNode, items: TtsIssue[] = []) => <>{content}{!!items.length && <IssueList issues={items}/>}</>;
  const rows: [string, React.ReactNode, React.ReactNode][] = [
    [text('数据状态', 'Data status'), detail(state(train.state), remaining([...(train.issues || []), ...(train.validation_execution.issues || [])], [...(train.token_filter.issues || []), ...(train.batching.issues || [])])), detail(state(validation.state), remaining(validation.issues, [...(validation.token_filter.issues || []), ...(validation.batching.issues || []), ...(validation.validation_execution.issues || [])]))],
    [text('清单检查', 'Manifest check'), state(train.source_state), state(validation.source_state)],
    [text('清单音频数', 'Manifest clips'), value(train.source_summary?.clips_count), value(validation.source_summary?.clips_count)],
    [text('有效音频数', 'Valid clips'), value(train.source_summary?.valid_clips_count), value(validation.source_summary?.valid_clips_count)],
    [text('无效音频数', 'Invalid clips'), value(train.source_summary?.invalid_count), value(validation.source_summary?.invalid_count)],
    [text('长度过滤', 'Length filtering'), detail(state(train.token_filter.state), remaining(train.token_filter.issues)), detail(state(validation.token_filter.state), remaining(validation.token_filter.issues))],
    [text('单条 token 上限', 'Tokens per sample limit'), value(train.token_filter.max_sample_tokens), value(validation.token_filter.max_sample_tokens)],
    [text('过滤前 / 保留 / 排除', 'Before / kept / filtered'), `${value(train.token_filter.before_count)} / ${value(train.token_filter.kept_count)} / ${value(train.token_filter.filtered_count)}`, `${value(validation.token_filter.before_count)} / ${value(validation.token_filter.kept_count)} / ${value(validation.token_filter.filtered_count)}`],
    [text('分批状态', 'Batching status'), detail(state(train.batching.state), remaining(train.batching.issues)), detail(state(validation.batching.state), remaining(validation.batching.issues))],
    [text('每批音频数', 'Clips per batch'), value(train.batching.batch_size), value(validation.batching.batch_size)],
    [text('每轮完整批次', 'Full batches per pass'), value(train.batching.full_batches_per_pass), value(validation.batching.full_batches_per_pass)],
    [text('尾批音频 / 丢弃音频', 'Tail clips / dropped clips'), `${value(train.batching.tail_samples_per_pass)} / ${value(train.batching.dropped_tail_samples_per_pass)}`, `${value(validation.batching.tail_samples_per_pass)} / ${value(validation.batching.dropped_tail_samples_per_pass)}`],
    [text('每轮输出批次 / 音频', 'Yielded batches / clips per pass'), `${value(train.batching.yielded_batches_per_pass)} / ${value(train.batching.yielded_samples_per_pass)}`, `${value(validation.batching.yielded_batches_per_pass)} / ${value(validation.batching.yielded_samples_per_pass)}`],
  ];
  return <div className="tts-training-report">
    {!stale && <p role="status" className="tts-training-result" data-valid={report.valid}>{report.valid ? text('检查通过，可以加入训练队列。', 'Checks passed. Ready to join the training queue.') : text('检查未通过，请处理以下问题。', 'Checks failed. Resolve the issues below.')}</p>}
    {!!report.errors.length && <section aria-label={text('检查错误', 'Check errors')}><IssueList issues={report.errors}/></section>}
    {!!report.warnings.length && <section aria-label={text('检查提醒', 'Check warnings')}><h3>{text('提醒', 'Warnings')}</h3><IssueList issues={report.warnings}/></section>}
    <h3>{text('环境检查', 'Environment checks')}</h3><ul className="tts-training-checks">{environmentChecks.map(([key, zh, en]) => { const item = report.environment.checks.find(check => check.key === key), localIssues = remaining(item?.issues); return <li key={key}><div><span>{text(zh, en)}</span><strong data-state={item?.state || 'unchecked'}>{state(item?.state || 'unchecked')}</strong></div>{!!localIssues.length && <IssueList issues={localIssues}/>}</li>; })}</ul>
    <section aria-label={text('显卡检查', 'GPU checks')}><h3>{text('显卡检查', 'GPU checks')}</h3>
      {report.environment.devices ? <>
        <dl className="tts-training-facts mb-2"><div><dt>{text('选择方式', 'Selection')}</dt><dd>{report.environment.devices.requested_devices.length ? report.environment.devices.requested_devices.map(gpuDeviceLabel).join(', ') : text('自动选择', 'Automatic')}</dd></div><div><dt>{text('符合训练要求', 'Eligible for training')}</dt><dd>{report.environment.devices.eligible_devices.length ? report.environment.devices.eligible_devices.map(gpuDeviceLabel).join(', ') : text('无', 'None')}</dd></div></dl>
        <ul className="tts-training-checks">{report.environment.devices.checked_devices.map(device => <li key={device.device}>
          <div><span className="min-w-0 break-words">{gpuDeviceLabel(device.device)}{device.name ? ` · ${device.name}` : ''}</span><strong className="shrink-0" data-state={device.state}>{state(device.state)}</strong></div>
          <div><span>BF16</span><strong>{device.bf16 === null ? text('未确认', 'Unknown') : device.bf16 ? text('支持', 'Supported') : text('不支持', 'Unsupported')}</strong></div>
          {!!remaining(device.issues).length && <IssueList issues={remaining(device.issues)}/>}</li>)}</ul>
      </> : <p className="tts-training-note">{text('尚未完成显卡检查。', 'GPU checks have not completed.')}</p>}
    </section>
    <h3>{text('数据与分批', 'Data and batching')}</h3><div className="tts-training-table" role="region" aria-label={text('数据与分批结果', 'Data and batching results')} tabIndex={0}><table><thead><tr><th>{text('检查项', 'Check')}</th><th>{text('训练数据', 'Training data')}</th><th>{text('验证数据', 'Validation data')}</th></tr></thead><tbody>{rows.map(([label, training, validating]) => <tr key={label}><th scope="row">{label}</th><td>{training}</td><td>{validating}</td></tr>)}</tbody></table></div>
    <h3>{text('每次验证的执行范围', 'Each validation run')}</h3><dl className="tts-training-facts"><div><dt>{text('执行状态', 'Execution status')}</dt><dd>{state(validation.validation_execution.state)}</dd></div><div><dt>{text('批次上限', 'Batch limit')}</dt><dd>{value(validation.validation_execution.max_batches_per_validation)}</dd></div><div><dt>{text('每次计算批次 / 音频', 'Planned batches / clips per validation')}</dt><dd>{value(validation.validation_execution.evaluated_batches_per_validation)} / {value(validation.validation_execution.evaluated_samples_per_validation)}</dd></div><div><dt>{text('未参与的音频', 'Clips not evaluated')}</dt><dd>{value(validation.validation_execution.not_evaluated_samples_per_validation)}</dd></div></dl>{!!remaining(validation.validation_execution.issues).length && <IssueList issues={remaining(validation.validation_execution.issues)}/>}
  </div>;
}
