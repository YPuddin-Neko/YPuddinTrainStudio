import React from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useSearchParams } from 'react-router-dom';
import { Activity, AudioLines, Box, ChevronLeft, ChevronRight, Download, ExternalLink, Loader2, PackageOpen, Play, RefreshCw } from 'lucide-react';
import { ttsApi, ttsResourceUrl, type TtsAudio, type TtsCheckpoint, type TtsSampleJob } from '../../api/tts';
import Switch from '../../components/Switch';
import OverflowStrip from '../../components/OverflowStrip';
import { SlidingIndicator } from '../../components/motion';
import { useEventStream, useEventStreamStatus } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { formatApiError } from '../../utils/errors';
import { formatBytes, formatTime } from '../../utils/format';
import { useWorkspaceText } from '../../utils/workspaceText';
import { JobStatus } from '../Queue/jobPresentation';
import TtsSampleDialog from './TtsSampleDialog';
import { pendingSampleAttempts } from './ttsResultRequests';
import { checkpointProgress, gptSovitsLanguages, gptSovitsNumbers, gptSovitsStage } from './gptSovitsResults';
import '../../styles/project-results.css';
import './tts-results.css';

type Scope = { jobId: string } | { projectId: string; versionId: string };
type JobEvent = { job_id?: string; project_id?: string; version_id?: string; source_job_id?: string };
type ResultTab = 'checkpoints' | 'samples' | 'jobs';
type CheckpointKind = 'paired' | 'lora' | 'mixed';
export type TrainingRecords = { content: React.ReactNode; selector: React.ReactNode; projectId: string; total?: number; loading: boolean; refresh: () => void };
const activeSample = (item: TtsSampleJob) => ['queued', 'scheduled', 'running', 'cancelling'].includes(item.job.status);

function AudioPlayer({ audio, autoPlay, playRequest }: { audio: TtsAudio; autoPlay: boolean; playRequest: number }) {
  const text = useWorkspaceText();
  const element = React.useRef<HTMLAudioElement>(null);
  const [error, setError] = React.useState<'load' | 'play' | null>(null);
  const url = ttsResourceUrl(audio.url)!;
  React.useEffect(() => {
    const current = element.current;
    if (current && current.getAttribute('src') !== url) current.setAttribute('src', url);
    return () => { current?.pause(); current?.removeAttribute('src'); current?.load(); };
  }, [url]);
  React.useEffect(() => {
    let cancelled = false;
    if (autoPlay) {
      setError(null);
      void element.current?.play()?.catch(failure => { if (!cancelled && !(failure instanceof DOMException && failure.name === 'AbortError')) setError(current => current === 'load' ? current : 'play'); });
    }
    return () => { cancelled = true; };
  }, [autoPlay, url, playRequest]);
  return <div className="tts-result-player"><strong>{audio.filename}</strong><audio ref={element} controls preload="none" src={url} onPlay={() => setError(null)} onError={() => setError('load')} aria-label={text(`试听 ${audio.filename}`, `Listen to ${audio.filename}`)}/>{error && <p role="alert" className="tts-result-error">{error === 'load' ? text('音频无法读取，请刷新结果后重试。', 'Audio could not be read. Refresh the results and try again.') : text('播放未能开始，请点击播放器重试。', 'Playback could not start. Try the audio player again.')}</p>}</div>;
}

export function SampleRecords({ items, single = false }: { items: TtsSampleJob[]; single?: boolean }) {
  const text = useWorkspaceText();
  const [selection, setSelection] = React.useState<string | null>(null);
  const [playRequest, setPlayRequest] = React.useState(0);
  const available = items.filter(item => item.audio?.available && ttsResourceUrl(item.audio.url));
  const selected = available.find(item => item.audio?.id === selection) || available[0];
  const value = (number: number | null | undefined) => number == null ? text('未知', 'Unknown') : number.toLocaleString();
  return <div className="tts-sample-records">
    {selected?.audio && <AudioPlayer key={`${selected.audio.id}:${selected.audio.url}`} audio={selected.audio} autoPlay={selection === selected.audio.id} playRequest={playRequest}/>}
    <div className="tts-sample-list">{items.map(item => {
      const gsv = !!(item.audio?.gpt_sovits || item.request.gpt_sovits);
      const options = item.audio ? item.audio.gpt_sovits : item.request.gpt_sovits;
      const audio = item.audio, url = audio?.available ? ttsResourceUrl(audio.url) : undefined, content = audio?.text ?? item.request.text;
      return <article className="tts-sample-record" key={item.job.id} aria-label={item.job.name}>
        <header><JobStatus status={item.job.status}/>{single ? <strong>{item.job.name}</strong> : <Link className="ui-link" to={`/jobs/${encodeURIComponent(item.job.id)}?tab=audio`}>{item.job.name}</Link>}<time>{formatTime(item.job.created_at)}</time></header>
        {content !== null && <p className="tts-result-transcript tts-sample-excerpt">{content}</p>}{item.job.error && <p className="tts-result-error">{item.job.error}</p>}
        <div className="tts-sample-summary">
          <dl className="tts-result-facts"><div><dt>{text('音频时长', 'Audio duration')}</dt><dd>{audio?.duration_seconds == null ? text('未知', 'Unknown') : `${audio.duration_seconds.toFixed(2)} s`}</dd></div><div><dt>{text('采样率', 'Sample rate')}</dt><dd>{audio?.sample_rate == null ? text('未知', 'Unknown') : `${audio.sample_rate} Hz`}</dd></div></dl>
          <div className="tts-result-record-actions">{url && audio ? <><button type="button" className="ui-btn ui-btn-sm" aria-pressed={selected?.audio?.id === audio.id} onClick={() => { setSelection(audio.id); setPlayRequest(value => value + 1); }}><Play size={13}/>{text('试听音频', 'Play audio')}</button><a className="ui-link" href={url} download={audio.filename}><Download size={13}/>{text('下载音频', 'Download audio')}{audio.size !== null && ` · ${formatBytes(audio.size)}`}</a></> : <span className="tts-result-muted">{audio ? text('音频文件不可用', 'Audio file unavailable') : activeSample(item) ? text('音频尚未生成', 'Audio not generated yet') : text('未生成音频', 'No audio generated')}</span>}</div>
        </div>
        <details className="tts-result-identity tts-sample-details" open={single || undefined}><summary>{text('生成参数与来源', 'Generation parameters and source')}</summary>
          {content !== null && <div className="tts-sample-full-text"><strong>{text('试听文本', 'Preview text')}</strong><p className="tts-result-transcript">{content}</p></div>}
          <dl>
            <div><dt>{text('请求种子', 'Requested seed')}</dt><dd>{value(audio?.requested_seed ?? item.request.seed)}</dd></div><div><dt>{text('实际种子', 'Actual seed')}</dt><dd>{value(audio?.seed)}</dd></div>
            <div><dt>CFG</dt><dd>{value(audio ? audio.cfg_value : gsv ? item.request.gpt_sovits?.cfg_scale : item.request.cfg_value)}</dd></div>{!gsv && <div><dt>{text('推理步数', 'Inference steps')}</dt><dd>{value(audio ? audio.inference_timesteps : item.request.inference_timesteps)}</dd></div>}
            <div><dt>{text('检查点 ID', 'Checkpoint ID')}</dt><dd tabIndex={0}>{item.checkpoint_id ?? text('未知', 'Unknown')}</dd></div><div><dt>{text('检查点版本', 'Checkpoint revision')}</dt><dd tabIndex={0}>{item.checkpoint_revision ?? text('未知', 'Unknown')}</dd></div>
            <div><dt>{text('来源训练', 'Source training')}</dt><dd>{item.source.record_exists ? <Link className="ui-link" to={`/jobs/${encodeURIComponent(item.source.job_id)}?tab=audio`}>{text('来源训练', 'Source training')} · {item.source.name || item.source.job_id}</Link> : <span>{text('来源训练记录已删除', 'Source training record deleted')}{item.source.name ? ` · ${item.source.name}` : ''}</span>}</dd></div>
            <div><dt>{text('来源权重', 'Source weights')}</dt><dd>{item.source.checkpoint_available ? text('可用', 'Available') : text('不可用', 'Unavailable')}</dd></div><div><dt>{text('参考音频', 'Reference audio')}</dt><dd tabIndex={0}>{item.request.reference_audio === null ? text('未知', 'Unknown') : item.request.reference_audio || text('未使用', 'Not used')}</dd></div><div><dt>{text('参考转写', 'Reference transcript')}</dt><dd className="tts-result-transcript">{item.request.reference_text === null ? text('未知', 'Unknown') : item.request.reference_text || '—'}</dd></div><div><dt>{text('音频生成时间', 'Audio created at')}</dt><dd>{audio?.created_at == null ? text('未知', 'Unknown') : formatTime(audio.created_at)}</dd></div>
            {gsv && <><div><dt>{text('参数记录', 'Parameter record')}</dt><dd>{audio ? text('实际生成参数', 'Actual generation parameters') : text('请求参数', 'Requested parameters')}</dd></div>{(['text_language', 'reference_language'] as const).map(key => <div key={key}><dt>{key === 'text_language' ? text('试听文本语言', 'Preview text language') : text('参考音频语言', 'Reference audio language')}</dt><dd>{gptSovitsLanguages.find(([language]) => language === options?.[key])?.[text('zh', 'en') === 'zh' ? 1 : 2] || text('未知', 'Unknown')}</dd></div>)}{gptSovitsNumbers.map(field => <div key={field.key}><dt>{text(field.zh, field.en)}</dt><dd>{value(options?.[field.key])}</dd></div>)}<div><dt>{text('采样步数', 'Sampling steps')}</dt><dd>{value(options?.sample_steps)}</dd></div></>}
          </dl>
        </details>
      </article>;
    })}</div>
  </div>;
}

function CheckpointTable({ items, onPreview, showSource, kind, label }: { items: TtsCheckpoint[]; onPreview: (item: TtsCheckpoint) => void; showSource: boolean; kind: CheckpointKind; label: string }) {
  const text = useWorkspaceText();
  return <div className="results-table-wrap tts-checkpoint-table" tabIndex={0} role="region" aria-label={label}><table className="results-job-table"><thead><tr><th scope="col">{text('检查点', 'Checkpoint')}</th><th scope="col">{kind !== 'lora' ? text('训练进度', 'Training progress') : text('更新步数', 'Update step')}</th><th scope="col">{text('保存时间', 'Saved at')}</th><th scope="col">{text('文件', 'Files')}</th><th scope="col">{text('试听', 'Preview')}</th></tr></thead><tbody>{items.map(item => <tr key={item.id}>
    <td><strong>{item.name}</strong>{item.gpt_sovits && <p className="tts-result-muted">{item.gpt_sovits.variant} · {gptSovitsStage(item.gpt_sovits.stage, text)}</p>}{showSource && <Link className="ui-link tts-result-source-link" to={`/jobs/${encodeURIComponent(item.source_job_id)}?tab=audio`}>{item.source_job_id}</Link>}<details className="tts-result-identity"><summary>{text('身份与路径', 'Identity and path')}</summary><dl><div><dt>ID</dt><dd tabIndex={0}>{item.id}</dd></div><div><dt>{text('版本', 'Revision')}</dt><dd tabIndex={0}>{item.revision}</dd></div><div><dt>{text('相对路径', 'Relative path')}</dt><dd tabIndex={0}>{item.relative_path}</dd></div>{!item.gpt_sovits && <div><dt>{text('上游步数', 'Upstream step')}</dt><dd>{item.upstream_step ?? text('未知', 'Unknown')}</dd></div>}<div><dt>{text('首次发现', 'First discovered')}</dt><dd>{formatTime(item.discovered_at)}</dd></div></dl></details></td>
    <td className="tts-checkpoint-progress">{item.gpt_sovits || kind === 'mixed' ? checkpointProgress(item, text) : item.step ?? text('未知', 'Unknown')}</td><td>{item.created_at === null ? text('未知', 'Unknown') : formatTime(item.created_at)}</td>
    <td><ul className="tts-checkpoint-files">{item.files.map(file => { const url = ttsResourceUrl(file.download_url); return <li key={file.id}>{url ? <a className="ui-link" href={url} download={file.name}><Download size={12}/>{file.name}</a> : <span>{file.name}</span>}<small>{item.gpt_sovits && `${file.role === 'config' ? text('配对配置', 'Pair configuration') : file.role === 'gpt_weights' ? text('GPT 权重', 'GPT weights') : file.role === 'sovits_weights' ? text('SoVITS 权重', 'SoVITS weights') : file.role} · `}{formatBytes(file.size)}</small></li>; })}</ul></td>
    <td><button type="button" className="ui-btn ui-btn-sm" disabled={!item.can_preview} onClick={() => onPreview(item)}><Play size={13}/>{text('生成试听', 'Generate preview')}</button>{!item.can_preview && <p className="tts-result-muted">{item.unavailable_reason?.message || text('暂不能试听', 'Preview unavailable')}</p>}</td>
  </tr>)}</tbody></table></div>;
}

export default function TtsResultsView({ scope, live = false, engine, trainingRecords }: { scope: Scope; live?: boolean; engine?: string; trainingRecords?: TrainingRecords }) {
  const text = useWorkspaceText();
  const client = useQueryClient();
  const [params, setParams] = useSearchParams();
  const [localTab, setLocalTab] = React.useState<ResultTab>('checkpoints');
  const requestedTab = params.get('result_tab');
  const tab: ResultTab = trainingRecords ? requestedTab === 'jobs' ? 'jobs' : requestedTab === 'samples' ? 'samples' : 'checkpoints' : localTab;
  const setTab = (next: ResultTab) => {
    if (trainingRecords) setParams(previous => { const value = new URLSearchParams(previous); value.set('result_tab', next === 'checkpoints' ? 'artifacts' : next); return value; });
    else setLocalTab(next);
  };
  const [checkpointCursors, setCheckpointCursors] = React.useState<(string | undefined)[]>([undefined]);
  const [sampleCursors, setSampleCursors] = React.useState<(string | undefined)[]>([undefined]);
  const [includeArchived, setIncludeArchived] = React.useState(false);
  const [selection, setSelection] = React.useState<TtsCheckpoint | null>(null);
  const checkpointCursor = checkpointCursors[checkpointCursors.length - 1], sampleCursor = sampleCursors[sampleCursors.length - 1];
  const scopeKey = 'jobId' in scope ? ['job', scope.jobId] : ['version', scope.projectId, scope.versionId];
  const checkpoints = useQuery({ queryKey: ['tts-result-checkpoints', ...scopeKey, checkpointCursor], queryFn: async ({ signal }) => 'jobId' in scope ? { items: await ttsApi.checkpoints(scope.jobId, signal), next_cursor: null } : ttsApi.versionCheckpoints(scope.projectId, scope.versionId, { cursor: checkpointCursor, limit: 20 }, signal), staleTime: Infinity, refetchOnMount: 'always', refetchInterval: live ? 5000 : false });
  const samples = useQuery({ queryKey: ['tts-result-samples', ...scopeKey, sampleCursor, includeArchived], queryFn: ({ signal }) => 'jobId' in scope ? ttsApi.sampleJobs(scope.jobId, { cursor: sampleCursor, limit: 20, include_archived: includeArchived }, signal) : ttsApi.versionSampleJobs(scope.projectId, scope.versionId, { cursor: sampleCursor, limit: 20, include_archived: includeArchived }, signal), staleTime: Infinity, refetchOnMount: 'always', refetchInterval: query => query.state.data?.items.some(activeSample) ? 3000 : false });
  const related = (event: JobEvent) => 'jobId' in scope ? event.job_id === scope.jobId || event.source_job_id === scope.jobId || !!samples.data?.items.some(item => item.job.id === event.job_id) : event.project_id ? event.project_id === scope.projectId && event.version_id === scope.versionId : !!samples.data?.items.some(item => item.job.id === event.job_id) || !!checkpoints.data?.items.some(item => item.source_job_id === event.job_id);
  const refreshVisible = () => { void checkpoints.refetch({ cancelRefetch: false }); void samples.refetch({ cancelRefetch: false }); };
  useEventStream<JobEvent>(EVENT_TYPES.JOB_STATE, event => { if (related(event)) refreshVisible(); });
  useEventStream<JobEvent>(EVENT_TYPES.JOB_CHECKPOINT, event => { if (!('jobId' in scope) && !event.project_id || related(event)) void checkpoints.refetch({ cancelRefetch: false }); });
  useEventStream<JobEvent>(EVENT_TYPES.QUEUE_CHANGED, event => { if (!event.project_id && !event.source_job_id || related(event)) refreshVisible(); });
  const connection = useEventStreamStatus(), previousConnection = React.useRef(connection);
  React.useEffect(() => { if (connection === 'connected' && previousConnection.current !== 'connected') refreshVisible(); previousConnection.current = connection; });
  const previousLive = React.useRef(live);
  React.useEffect(() => { if (previousLive.current && !live) refreshVisible(); previousLive.current = live; });
  const firstPage = (target: 'checkpoints' | 'samples') => {
    const key = target === 'checkpoints' ? ['tts-result-checkpoints', ...scopeKey, undefined] : ['tts-result-samples', ...scopeKey, undefined, includeArchived];
    void client.invalidateQueries({ queryKey: key, exact: true });
    if (target === 'checkpoints') setCheckpointCursors([undefined]); else setSampleCursors([undefined]);
  };
  const refresh = () => { if (tab === 'jobs') trainingRecords?.refresh(); else firstPage(tab); };
  const checkpointItems = checkpoints.data?.items || [];
  const checkpointKind: CheckpointKind = checkpointItems.length
    ? checkpointItems.every(item => item.gpt_sovits) ? 'paired' : checkpointItems.some(item => item.gpt_sovits) ? 'mixed' : 'lora'
    : engine === 'gpt-sovits-v5' ? 'paired' : 'lora';
  const active = tab === 'checkpoints' ? checkpoints : samples, cursors = tab === 'checkpoints' ? checkpointCursors : sampleCursors, setCursors = tab === 'checkpoints' ? setCheckpointCursors : setSampleCursors;
  const attempts = pendingSampleAttempts(scope);
  const selected = selection && (checkpoints.data?.items.find(item => item.id === selection.id) || { ...selection, can_preview: false, unavailable_reason: null });
  const id = React.useId();
  const tabs = [
    { key: 'checkpoints' as const, label: text('模型产物', 'Model outputs'), icon: Box },
    { key: 'samples' as const, label: text('试听音频', 'Audio previews'), icon: AudioLines },
    ...(trainingRecords ? [{ key: 'jobs' as const, label: text('训练记录', 'Training records'), icon: Activity }] : []),
  ];
  const loading = tab === 'jobs' ? trainingRecords?.loading : active.isFetching;
  const checkpointLabel = checkpointKind === 'mixed' ? text('模型检查点', 'Model checkpoints') : checkpointKind === 'paired' ? text('配对检查点', 'Paired checkpoints') : text('LoRA 检查点', 'LoRA checkpoints');
  return <section className="version-results tts-results" aria-label={text('语音产物与试听', 'Speech outputs and preview')}>
    <div className="results-toolbar">
      <OverflowStrip className="results-tabs ui-tabs" containerClassName="results-tab-navigation" label={text('语音结果类型', 'Speech result type')} activeKey={tab}>
        {tabs.map((item, index) => <button type="button" key={item.key} role="tab" id={`${id}-${item.key}`} aria-selected={tab === item.key} aria-controls={`${id}-panel-${item.key}`} tabIndex={tab === item.key ? 0 : -1} onClick={() => setTab(item.key)} onKeyDown={event => {
          const next = event.key === 'ArrowRight' ? (index + 1) % tabs.length : event.key === 'ArrowLeft' ? (index + tabs.length - 1) % tabs.length : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : -1;
          if (next >= 0) { event.preventDefault(); setTab(tabs[next].key); document.getElementById(`${id}-${tabs[next].key}`)?.focus(); }
        }}><item.icon size={14} aria-hidden="true"/>{item.label}</button>)}
        <SlidingIndicator className="ui-tabs-indicator"/>
      </OverflowStrip>
      {trainingRecords && <div className="results-overview">{trainingRecords.total !== undefined && <span>{text(`共 ${trainingRecords.total} 次训练`, `${trainingRecords.total} training runs`)}</span>}<Link className="ui-link" to={`/queue?project_id=${encodeURIComponent(trainingRecords.projectId)}&type=tts_train`}>{text('全局队列', 'Queue')}<ExternalLink size={12}/></Link></div>}
      <button type="button" className="ui-btn results-refresh" disabled={loading} onClick={refresh}><RefreshCw size={13} className={loading ? 'animate-spin' : undefined}/>{text('刷新', 'Refresh')}</button>
    </div>
    {attempts.map(attempt => <div className="tts-result-pending" key={attempt.sourceJobId}><span>{text('有尚待确认的试听请求', 'A preview request needs confirmation')} · {attempt.checkpoint.name}</span><button type="button" className="ui-link" onClick={() => setSelection(attempt.checkpoint)}>{text('继续确认试听请求', 'Confirm preview request')}</button></div>)}
    <div role="tabpanel" id={`${id}-panel-${tab}`} aria-labelledby={`${id}-${tab}`}>
      {tab === 'jobs' ? trainingRecords?.content : <>
        {trainingRecords?.selector}
        <div className="results-sample-toolbar tts-results-controls"><span className="tts-results-kind">{tab === 'checkpoints' ? checkpointLabel : text('试听任务', 'Preview jobs')}{active.data && <small>{text(`本页 ${active.data.items.length} 项`, `${active.data.items.length} on this page`)}</small>}</span>{tab === 'samples' && <Switch className="studio-switch-small" checked={includeArchived} onCheckedChange={value => { setIncludeArchived(value); setSampleCursors([undefined]); }}>{text('显示已归档任务', 'Show archived jobs')}</Switch>}</div>
        {active.error && <div role="alert" className="results-error"><span>{formatApiError(active.error)}</span><button type="button" className="ui-btn ui-btn-sm" disabled={active.isFetching} onClick={refresh}>{text('重试', 'Retry')}</button></div>}
        {active.isPending && <p role="status" className="results-empty"><Loader2 size={16} className="animate-spin"/>{text('正在读取结果…', 'Loading results…')}</p>}
        {!active.error && active.data && !active.data.items.length && <div className="results-empty">{tab === 'checkpoints' ? <PackageOpen size={22}/> : <AudioLines size={22}/>}<p>{tab === 'checkpoints' ? checkpointKind === 'paired' ? text('尚无配对检查点', 'No paired checkpoints yet') : text('尚无 LoRA 检查点', 'No LoRA checkpoints yet') : text('暂无试听任务。', 'No preview jobs yet.')}</p>{tab === 'samples' && <button type="button" className="ui-link" onClick={() => setTab('checkpoints')}>{text('选择模型产物生成试听', 'Choose model outputs to generate a preview')}</button>}</div>}
        {tab === 'checkpoints' && !!checkpoints.data?.items.length && <CheckpointTable items={checkpoints.data.items} onPreview={setSelection} showSource={!('jobId' in scope)} kind={checkpointKind} label={checkpointLabel}/>}
        {tab === 'samples' && !!samples.data?.items.length && <SampleRecords key={`${scopeKey.join(':')}:${sampleCursor || 'first'}:${includeArchived}`} items={samples.data.items}/>}
        {(cursors.length > 1 || active.data?.next_cursor) && <nav className="results-pagination" aria-label={text('结果分页', 'Result pages')}><span>{text(`第 ${cursors.length} 页`, `Page ${cursors.length}`)}</span><div><button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('上一页', 'Previous')} disabled={cursors.length === 1 || active.isFetching} onClick={() => setCursors(previous => previous.slice(0, -1))}><ChevronLeft size={14}/></button><button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('下一页', 'Next')} disabled={!active.data?.next_cursor || active.isFetching} onClick={() => { if (active.data?.next_cursor) setCursors(previous => [...previous, active.data!.next_cursor!]); }}><ChevronRight size={14}/></button></div></nav>}
      </>}
    </div>
    {selected && <TtsSampleDialog checkpoint={selected} onClose={() => setSelection(null)} onCreated={() => { setSelection(null); setTab('samples'); firstPage('samples'); }}/>}
  </section>;

}

export function SingleSampleResult({ jobId, live = false }: { jobId: string; live?: boolean }) {
  const text = useWorkspaceText();
  const query = useQuery({ queryKey: ['tts-result-sample', jobId], queryFn: ({ signal }) => ttsApi.sampleJob(jobId, signal), refetchInterval: value => value.state.data && activeSample(value.state.data) ? 3000 : false });
  useEventStream<JobEvent>(EVENT_TYPES.JOB_STATE, event => { if (event.job_id === jobId) void query.refetch(); });
  const connection = useEventStreamStatus(), previousConnection = React.useRef(connection);
  React.useEffect(() => { if (connection === 'connected' && previousConnection.current !== 'connected') void query.refetch(); previousConnection.current = connection; });
  const previousLive = React.useRef(live);
  React.useEffect(() => { if (previousLive.current && !live) void query.refetch(); previousLive.current = live; });
  return <section className="version-results tts-results" aria-label={text('生成的音频', 'Generated audio')}><header className="tts-results-toolbar"><h2>{text('生成的音频', 'Generated audio')}</h2><button type="button" className="ui-btn ui-btn-sm ui-btn-quiet" disabled={query.isFetching} onClick={() => void query.refetch()}><RefreshCw size={14}/>{text('刷新', 'Refresh')}</button></header>{query.isPending && <p role="status" className="tts-result-muted">{text('正在读取试听任务…', 'Loading preview job…')}</p>}{query.error && <p role="alert" className="tts-result-error">{formatApiError(query.error)}</p>}{query.data && <SampleRecords items={[query.data]} single/>}</section>;
}
