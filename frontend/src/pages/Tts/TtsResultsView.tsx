import React from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { ChevronLeft, ChevronRight, Download, Play, RefreshCw } from 'lucide-react';
import { ttsApi, ttsResourceUrl, type TtsAudio, type TtsCheckpoint, type TtsSampleJob } from '../../api/tts';
import Switch from '../../components/Switch';
import { useEventStream, useEventStreamStatus } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { formatApiError } from '../../utils/errors';
import { formatBytes, formatTime } from '../../utils/format';
import { useWorkspaceText } from '../../utils/workspaceText';
import { JobStatus } from '../Queue/jobPresentation';
import TtsSampleDialog from './TtsSampleDialog';
import { pendingSampleAttempts } from './ttsResultRequests';
import './tts-results.css';

type Scope = { jobId: string } | { projectId: string; versionId: string };
type JobEvent = { job_id?: string; project_id?: string; version_id?: string; source_job_id?: string };
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
      void element.current?.play()?.catch(() => { if (!cancelled) setError('play'); });
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
    {items.map(item => {
      const audio = item.audio, url = audio?.available ? ttsResourceUrl(audio.url) : undefined, content = audio?.text ?? item.request.text;
      return <article className="tts-sample-record" key={item.job.id} aria-label={item.job.name}>
        <header><JobStatus status={item.job.status}/>{single ? <strong>{item.job.name}</strong> : <Link className="ui-link" to={`/jobs/${encodeURIComponent(item.job.id)}?tab=audio`}>{item.job.name}</Link>}<time>{formatTime(item.job.created_at)}</time></header>
        {content !== null && <p className="tts-result-transcript">{content}</p>}{item.job.error && <p className="tts-result-error">{item.job.error}</p>}
        <dl className="tts-result-facts"><div><dt>{text('请求种子', 'Requested seed')}</dt><dd>{value(audio?.requested_seed ?? item.request.seed)}</dd></div><div><dt>{text('实际种子', 'Actual seed')}</dt><dd>{value(audio?.seed)}</dd></div><div><dt>CFG</dt><dd>{value(audio ? audio.cfg_value : item.request.cfg_value)}</dd></div><div><dt>{text('推理步数', 'Inference steps')}</dt><dd>{value(audio ? audio.inference_timesteps : item.request.inference_timesteps)}</dd></div><div><dt>{text('音频时长', 'Audio duration')}</dt><dd>{audio?.duration_seconds == null ? text('未知', 'Unknown') : `${audio.duration_seconds.toFixed(2)} s`}</dd></div><div><dt>{text('采样率', 'Sample rate')}</dt><dd>{audio?.sample_rate == null ? text('未知', 'Unknown') : `${audio.sample_rate} Hz`}</dd></div></dl>
        <div className="tts-result-record-actions">{url && audio ? <><button type="button" className="ui-btn ui-btn-sm" aria-pressed={selected?.audio?.id === audio.id} onClick={() => { setSelection(audio.id); setPlayRequest(value => value + 1); }}><Play size={13}/>{text('试听音频', 'Play audio')}</button><a className="ui-link" href={url} download={audio.filename}><Download size={13}/>{text('下载音频', 'Download audio')}{audio.size !== null && ` · ${formatBytes(audio.size)}`}</a></> : <span className="tts-result-muted">{audio ? text('音频文件不可用', 'Audio file unavailable') : activeSample(item) ? text('音频尚未生成', 'Audio not generated yet') : text('未生成音频', 'No audio generated')}</span>}
          {item.source.record_exists ? <Link className="ui-link" to={`/jobs/${encodeURIComponent(item.source.job_id)}?tab=audio`}>{text('来源训练', 'Source training')} · {item.source.name || item.source.job_id}</Link> : <span className="tts-result-muted">{text('来源训练记录已删除', 'Source training record deleted')}{item.source.name ? ` · ${item.source.name}` : ''}</span>}</div>
        <details className="tts-result-identity"><summary>{text('生成参数与来源', 'Generation parameters and source')}</summary><dl><div><dt>{text('检查点 ID', 'Checkpoint ID')}</dt><dd tabIndex={0}>{item.checkpoint_id ?? text('未知', 'Unknown')}</dd></div><div><dt>{text('检查点版本', 'Checkpoint revision')}</dt><dd tabIndex={0}>{item.checkpoint_revision ?? text('未知', 'Unknown')}</dd></div><div><dt>{text('来源权重', 'Source weights')}</dt><dd>{item.source.checkpoint_available ? text('可用', 'Available') : text('不可用', 'Unavailable')}</dd></div><div><dt>{text('参考音频', 'Reference audio')}</dt><dd tabIndex={0}>{item.request.reference_audio === null ? text('未知', 'Unknown') : item.request.reference_audio || text('未使用', 'Not used')}</dd></div><div><dt>{text('参考转写', 'Reference transcript')}</dt><dd className="tts-result-transcript">{item.request.reference_text === null ? text('未知', 'Unknown') : item.request.reference_text || '—'}</dd></div><div><dt>{text('音频生成时间', 'Audio created at')}</dt><dd>{audio?.created_at == null ? text('未知', 'Unknown') : formatTime(audio.created_at)}</dd></div></dl></details>
      </article>;
    })}
  </div>;
}

function CheckpointTable({ items, onPreview, showSource }: { items: TtsCheckpoint[]; onPreview: (item: TtsCheckpoint) => void; showSource: boolean }) {
  const text = useWorkspaceText();
  return <div className="tts-checkpoint-table" tabIndex={0} role="region" aria-label={text('LoRA 检查点', 'LoRA checkpoints')}><table><thead><tr><th scope="col">{text('检查点', 'Checkpoint')}</th><th scope="col">{text('更新步数', 'Update step')}</th><th scope="col">{text('保存时间', 'Saved at')}</th><th scope="col">{text('文件', 'Files')}</th><th scope="col">{text('试听', 'Preview')}</th></tr></thead><tbody>{items.map(item => <tr key={item.id}>
    <td><strong>{item.name}</strong>{showSource && <Link className="ui-link tts-result-source-link" to={`/jobs/${encodeURIComponent(item.source_job_id)}?tab=audio`}>{item.source_job_id}</Link>}<details className="tts-result-identity"><summary>{text('身份与路径', 'Identity and path')}</summary><dl><div><dt>ID</dt><dd tabIndex={0}>{item.id}</dd></div><div><dt>{text('版本', 'Revision')}</dt><dd tabIndex={0}>{item.revision}</dd></div><div><dt>{text('相对路径', 'Relative path')}</dt><dd tabIndex={0}>{item.relative_path}</dd></div><div><dt>{text('上游步数', 'Upstream step')}</dt><dd>{item.upstream_step ?? text('未知', 'Unknown')}</dd></div><div><dt>{text('首次发现', 'First discovered')}</dt><dd>{formatTime(item.discovered_at)}</dd></div></dl></details></td>
    <td>{item.step ?? text('未知', 'Unknown')}</td><td>{item.created_at === null ? text('未知', 'Unknown') : formatTime(item.created_at)}</td>
    <td><ul className="tts-checkpoint-files">{item.files.map(file => { const url = ttsResourceUrl(file.download_url); return <li key={file.id}>{url ? <a className="ui-link" href={url} download={file.name}><Download size={12}/>{file.name}</a> : <span>{file.name}</span>}<small>{formatBytes(file.size)}</small></li>; })}</ul></td>
    <td><button type="button" className="ui-btn ui-btn-sm" disabled={!item.can_preview} onClick={() => onPreview(item)}><Play size={13}/>{text('生成试听', 'Generate preview')}</button>{!item.can_preview && <p className="tts-result-muted">{item.unavailable_reason?.message || text('暂不能试听', 'Preview unavailable')}</p>}</td>
  </tr>)}</tbody></table></div>;
}

export default function TtsResultsView({ scope, live = false }: { scope: Scope; live?: boolean }) {
  const text = useWorkspaceText();
  const client = useQueryClient();
  const [tab, setTab] = React.useState<'checkpoints' | 'samples'>('checkpoints');
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
  const refresh = () => firstPage(tab);
  const active = tab === 'checkpoints' ? checkpoints : samples, cursors = tab === 'checkpoints' ? checkpointCursors : sampleCursors, setCursors = tab === 'checkpoints' ? setCheckpointCursors : setSampleCursors;
  const attempts = pendingSampleAttempts(scope);
  const selected = selection && (checkpoints.data?.items.find(item => item.id === selection.id) || { ...selection, can_preview: false, unavailable_reason: null });
  const id = React.useId();
  return <section className="tts-results" aria-label={text('语音产物与试听', 'Speech outputs and preview')}>
    <header className="tts-results-toolbar"><div className="ui-segmented" role="tablist" aria-label={text('语音结果类型', 'Speech result type')} onKeyDown={event => { if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return; event.preventDefault(); const next = tab === 'checkpoints' ? 'samples' : 'checkpoints'; setTab(next); document.getElementById(`${id}-${next}`)?.focus(); }}><button type="button" role="tab" id={`${id}-checkpoints`} aria-selected={tab === 'checkpoints'} aria-controls={`${id}-panel`} tabIndex={tab === 'checkpoints' ? 0 : -1} onClick={() => setTab('checkpoints')}>{text('LoRA 检查点', 'LoRA checkpoints')}</button><button type="button" role="tab" id={`${id}-samples`} aria-selected={tab === 'samples'} aria-controls={`${id}-panel`} tabIndex={tab === 'samples' ? 0 : -1} onClick={() => setTab('samples')}>{text('试听任务', 'Preview jobs')}</button></div><button type="button" className="ui-btn ui-btn-sm ui-btn-quiet" disabled={active.isFetching} onClick={refresh}><RefreshCw size={14}/>{text('刷新', 'Refresh')}</button></header>
    {attempts.map(attempt => <div className="tts-result-pending" key={attempt.sourceJobId}><span>{text('有尚待确认的试听请求', 'A preview request needs confirmation')} · {attempt.checkpoint.name}</span><button type="button" className="ui-link" onClick={() => setSelection(attempt.checkpoint)}>{text('继续确认试听请求', 'Confirm preview request')}</button></div>)}
    <div role="tabpanel" id={`${id}-panel`} aria-labelledby={`${id}-${tab}`}>
      {tab === 'samples' && <Switch className="studio-switch-small tts-results-archived" checked={includeArchived} onCheckedChange={value => { setIncludeArchived(value); setSampleCursors([undefined]); }}>{text('显示已归档任务', 'Show archived jobs')}</Switch>}
      {active.isPending && <p role="status" className="tts-result-muted">{text('正在读取结果…', 'Loading results…')}</p>}{active.error && <p role="alert" className="tts-result-error">{formatApiError(active.error)}</p>}
      {active.data && !active.data.items.length && <p className="tts-result-muted">{tab === 'checkpoints' ? text('尚无 LoRA 检查点。训练保存后会在这里显示。', 'No LoRA checkpoints yet. Saved training weights will appear here.') : text('暂无试听任务。', 'No preview jobs yet.')}</p>}
      {tab === 'checkpoints' && !!checkpoints.data?.items.length && <CheckpointTable items={checkpoints.data.items} onPreview={setSelection} showSource={!('jobId' in scope)}/>}
      {tab === 'samples' && samples.data && <SampleRecords key={`${scopeKey.join(':')}:${sampleCursor || 'first'}:${includeArchived}`} items={samples.data.items}/>}
      {(cursors.length > 1 || active.data?.next_cursor) && <nav className="tts-results-pagination" aria-label={text('结果分页', 'Result pages')}><button type="button" className="ui-btn ui-btn-sm" disabled={cursors.length === 1 || active.isFetching} onClick={() => setCursors(previous => previous.slice(0, -1))}><ChevronLeft size={14}/>{text('上一页', 'Previous')}</button><span>{text(`第 ${cursors.length} 页`, `Page ${cursors.length}`)}</span><button type="button" className="ui-btn ui-btn-sm" disabled={!active.data?.next_cursor || active.isFetching} onClick={() => { if (active.data?.next_cursor) setCursors(previous => [...previous, active.data!.next_cursor!]); }}>{text('下一页', 'Next')}<ChevronRight size={14}/></button></nav>}
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
  return <section className="tts-results" aria-label={text('生成的音频', 'Generated audio')}><header className="tts-results-toolbar"><h2>{text('生成的音频', 'Generated audio')}</h2><button type="button" className="ui-btn ui-btn-sm ui-btn-quiet" disabled={query.isFetching} onClick={() => void query.refetch()}><RefreshCw size={14}/>{text('刷新', 'Refresh')}</button></header>{query.isPending && <p role="status" className="tts-result-muted">{text('正在读取试听任务…', 'Loading preview job…')}</p>}{query.error && <p role="alert" className="tts-result-error">{formatApiError(query.error)}</p>}{query.data && <SampleRecords items={[query.data]} single/>}</section>;
}
