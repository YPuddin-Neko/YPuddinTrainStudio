import React from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { AudioLines, ChevronLeft, ChevronRight, FileAudio, FolderOpen, ListChecks, Loader2, Play, RefreshCw, X } from 'lucide-react';
import { ttsApi, ttsAudioUrl, type TtsEngine, type TtsIssue, type TtsSource, type TtsSourceChanged, type TtsSourceRow, type TtsSourcesResponse, type TtsSplit } from '../../api/tts';
import Dialog from '../../components/Dialog';
import ConfigHelp from '../../components/ConfigHelp';
import OverflowStrip from '../../components/OverflowStrip';
import { SlidingIndicator } from '../../components/motion';
import { PathInput } from '../../components/PathBrowser';
import StudioSelect from '../../components/StudioSelect';
import { EVENT_TYPES } from '../../events/eventTypes';
import { useEventStream, useEventStreamStatus } from '../../events/useEventStream';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import { useEnterAnimation } from '../../utils/motion';
import '../../components/datasets/dataset-pipeline.css';
import '../../styles/project-results.css';
import './tts-sources.css';

interface Props { projectId: string; versionId: string; readOnly: boolean; engine?: TtsEngine }
type SourceDialog = { split: TtsSplit; action: 'put' | 'remove'; path: string; revision: number; conflict: boolean };
const splits: TtsSplit[] = ['train', 'validation'];
const complete = (source: TtsSource) => (source.state === 'valid' || source.state === 'invalid') && source.snapshot_id !== null;
const statusCode = (error: unknown) => (error as { status?: number } | null)?.status;
function listParent(value: string): string | undefined {
  const path = value.trim();
  if (!/\.list$/i.test(path)) return undefined;
  const separator = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
  if (separator < 0) return '.';
  if (separator === 0) return path[0];
  return /^[a-z]:$/i.test(path.slice(0, separator)) ? path.slice(0, separator + 1) : path.slice(0, separator);
}

function IssueList({ issues }: { issues: TtsIssue[] }) {
  const text = useWorkspaceText();
  const location = (issue: TtsIssue) => {
    const rowAt = issue.loc.indexOf('rows');
    if (rowAt >= 0 && typeof issue.loc[rowAt + 1] === 'number') return [text(`第 ${issue.loc[rowAt + 1]} 行`, `Line ${issue.loc[rowAt + 1]}`), ...issue.loc.slice(rowAt + 2)].join(' · ');
    return (issue.loc[0] === 'sources' ? issue.loc.slice(2) : issue.loc).join(' · ');
  };
  return <ul className="tts-source-issues">{issues.map((issue, index) => <li key={`${issue.code}:${index}`} data-severity={issue.severity}>
    <span>{issue.severity === 'warning' ? text('警告', 'Warning') : text('错误', 'Error')}</span>
    <div>{issue.message}{location(issue) && <small>{location(issue)}</small>}</div>
  </li>)}</ul>;
}

function AudioPreview({ url, label, playRequest, onClose }: { url: string; label: string; playRequest: number; onClose: () => void }) {
  const text = useWorkspaceText();
  const audio = React.useRef<HTMLAudioElement>(null);
  const [failure, setFailure] = React.useState<'load' | 'play' | null>(null);
  React.useEffect(() => {
    const element = audio.current;
    if (element && element.getAttribute('src') !== url) element.setAttribute('src', url);
    return () => { element?.pause(); element?.removeAttribute('src'); element?.load(); };
  }, [url]);
  React.useEffect(() => {
    let cancelled = false;
    setFailure(null);
    void audio.current?.play()?.catch(error => { if (!cancelled && !(error instanceof DOMException && error.name === 'AbortError')) setFailure(current => current === 'load' ? current : 'play'); });
    return () => { cancelled = true; };
  }, [url, playRequest]);
  return <div className="tts-source-player">
    <div><strong>{label}</strong><button type="button" className="ui-btn ui-btn-quiet ui-btn-icon" onClick={onClose} aria-label={text('关闭试听', 'Close audio preview')}><X size={14}/></button></div>
    <audio ref={audio} controls preload="none" src={url} onPlay={() => setFailure(null)} onError={() => setFailure('load')} aria-label={label}/>
    {failure && <p role="alert" className="tts-source-error">{failure === 'load' ? text('音频无法读取。请重新检查来源后再试听。', 'Audio could not be read. Check the source again before previewing.') : text('播放未能开始，请再次点击试听或播放器重试。', 'Playback could not start. Try the preview button or audio player again.')}</p>}
  </div>;
}

function SourceRows({ source, refresh, gptSovits }: { source: TtsSource; refresh: () => void; gptSovits: boolean }) {
  const text = useWorkspaceText();
  const [page, setPage] = React.useState(1);
  const [pageSize, setPageSize] = React.useState(50);
  const [preview, setPreview] = React.useState<{ url: string; label: string; playRequest: number } | null>(null);
  const { project_id: pid, version_id: vid } = source.scope;
  const rows = useQuery({
    queryKey: ['tts-source-rows', pid, vid, source.id, source.snapshot_id, page, pageSize],
    queryFn: ({ signal }) => ttsApi.rows(pid, vid, source.id, { snapshot_id: source.snapshot_id!, page, page_size: pageSize }, signal),
    retry: false,
    refetchOnWindowFocus: false,
  });
  const play = (row: TtsSourceRow, reference: boolean) => {
    const url = ttsAudioUrl(reference ? row.reference_audio_url : row.audio_url);
    if (url) setPreview(previous => ({ url, label: text(`第 ${row.line} 行 · ${reference ? '参考音频' : '录音'}`, `Line ${row.line} · ${reference ? 'Reference audio' : 'Audio'}`), playRequest: (previous?.playRequest || 0) + 1 }));
  };
  const move = (next: number) => { setPreview(null); setPage(next); };
  const total = rows.data?.total;
  const pages = total === undefined ? null : Math.max(1, Math.ceil(total / pageSize));
  return <div className="tts-source-rows" aria-label={text('音频明细', 'Audio rows')}>
    <div className="tts-source-pagination">
      <span>{total === undefined ? text('正在读取明细…', 'Loading rows…') : text(`共 ${total.toLocaleString()} 条`, `${total.toLocaleString()} rows`)}</span>
      <StudioSelect aria-label={text('每页条数', 'Rows per page')} value={String(pageSize)} options={[25, 50, 100, 200].map(value => ({ value: String(value), label: text(`${value} 条 / 页`, `${value} / page`) }))} onValueChange={value => { setPreview(null); setPage(1); setPageSize(Number(value)); }}/>
      <button type="button" className="ui-btn ui-btn-sm ui-btn-quiet ui-btn-icon" aria-label={text('上一页', 'Previous page')} disabled={page === 1 || rows.isFetching} onClick={() => move(page - 1)}><ChevronLeft size={15}/></button>
      <span>{pages === null ? `${page} / —` : `${page} / ${pages}`}</span>
      <button type="button" className="ui-btn ui-btn-sm ui-btn-quiet ui-btn-icon" aria-label={text('下一页', 'Next page')} disabled={pages === null || page >= pages || rows.isFetching} onClick={() => move(page + 1)}><ChevronRight size={15}/></button>
    </div>
    {rows.isPending && <p role="status" className="tts-source-muted">{text('正在读取音频明细…', 'Loading audio rows…')}</p>}
    {rows.error && <div role="alert" className="tts-source-error">{formatApiError(rows.error)} <button type="button" className="ui-link" onClick={() => { setPreview(null); refresh(); void rows.refetch(); }}>{text('重新读取', 'Reload')}</button></div>}
    {preview && !rows.error && <AudioPreview key={preview.url} {...preview} onClose={() => setPreview(null)}/>}
    {rows.data && !rows.error && <div className="tts-source-table-scroll" tabIndex={0} role="region" aria-label={text('音频明细表格', 'Audio rows table')}><table className={gptSovits ? 'tts-source-gpt-sovits' : undefined}><thead><tr>
      <th scope="col">{text('行', 'Line')}</th><th scope="col">{text('音频', 'Audio')}</th>{gptSovits && <><th scope="col">{text('语言', 'Language')}</th><th scope="col">{text('说话人', 'Speaker')}</th></>}<th scope="col">{text('转写', 'Transcript')}</th><th scope="col">{text('检查结果', 'Check results')}</th>
    </tr></thead><tbody>{rows.data.items.map(row => <tr key={row.id}>
      <td>{row.line}</td>
      <td><div className="tts-source-audio-name">{row.audio_name || '—'}</div><small>{row.duration_seconds === null ? '—' : `${row.duration_seconds.toFixed(2)} s`} · {row.sample_rate === null ? '—' : `${row.sample_rate} Hz`} · {row.channels === null ? '—' : text(`${row.channels} 声道`, `${row.channels} ch`)}</small>
        <div className="tts-source-row-actions"><button type="button" className="ui-btn ui-btn-sm ui-btn-quiet" disabled={!ttsAudioUrl(row.audio_url)} onClick={() => play(row, false)} aria-label={text(`试听第 ${row.line} 行音频`, `Preview line ${row.line} audio`)}><Play size={12}/>{text('试听', 'Preview')}</button>
          {!gptSovits && row.reference_audio_name && <button type="button" className="ui-btn ui-btn-sm ui-btn-quiet" disabled={!ttsAudioUrl(row.reference_audio_url)} title={row.reference_audio_name} onClick={() => play(row, true)} aria-label={text(`试听第 ${row.line} 行参考音频`, `Preview line ${row.line} reference audio`)}><Play size={12}/>{text('参考音频', 'Reference')}</button>}</div></td>
      {gptSovits && <><td>{row.language || '—'}</td><td>{row.speaker || '—'}</td></>}<td className="tts-source-transcript">{row.text === null ? '—' : row.text}</td>
      <td>{row.issues.length ? <IssueList issues={row.issues}/> : <span className="tts-source-muted">{text('通过', 'Passed')}</span>}</td>
    </tr>)}</tbody></table>{!rows.data.items.length && <p className="tts-source-muted">{text('没有音频条目。', 'No audio rows.')}</p>}</div>}
  </div>;
}

function SourcesForVersion({ projectId: pid, versionId: vid, readOnly, engine = 'voxcpm1.5' }: Props) {
  const gptSovits = engine === 'gpt-sovits-v5';
  const text = useWorkspaceText();
  const client = useQueryClient();
  const connection = useEventStreamStatus();
  const previousConnection = React.useRef(connection);
  const queryKey = React.useMemo(() => ['tts-sources', pid, vid], [pid, vid]);
  const mounted = React.useRef(true);
  const locked = React.useRef(false);
  const [pending, setPending] = React.useState(false);
  const [dialog, setDialog] = React.useState<SourceDialog | null>(null);
  const [error, setError] = React.useState('');
  const [dialogError, setDialogError] = React.useState('');
  const [browsing, setBrowsing] = React.useState<TtsSplit | null>(null);
  const [stage, setStage] = React.useState<'sources' | 'rows' | 'issues'>('sources');
  const stageId = React.useId();
  const stageBody = useEnterAnimation<HTMLDivElement>(stage, { skipFirst: true });
  React.useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  const sources = useQuery({ queryKey, queryFn: ({ signal }) => ttsApi.sources(pid, vid, signal),
    refetchInterval: query => query.state.data?.items.some(source => source.state === 'checking') ? 2000 : false,
    refetchOnWindowFocus: true,
  });
  const signature = sources.data ? JSON.stringify(sources.data) : null;
  const lastSignature = React.useRef<string | null>(null);
  React.useEffect(() => {
    if (signature === null || signature === lastSignature.current) return;
    const changed = lastSignature.current !== null;
    lastSignature.current = signature;
    if (changed) {
      void client.invalidateQueries({ queryKey: ['tts-version-config', pid, vid] });
      void client.invalidateQueries({ queryKey: ['project-versions', pid] });
      void client.invalidateQueries({ queryKey: ['project', pid] });
    }
  }, [signature, client, pid, vid]);
  useEventStream<TtsSourceChanged>(EVENT_TYPES.TTS_SOURCE_CHANGED, event => {
    if (event.scope.project_id === pid && event.scope.version_id === vid) void client.invalidateQueries({ queryKey, exact: true });
  });
  React.useEffect(() => {
    if (connection === 'connected' && previousConnection.current !== 'connected') void client.invalidateQueries({ queryKey, exact: true });
    previousConnection.current = connection;
  }, [connection, client, queryKey]);
  const refresh = () => { void sources.refetch(); };
  const sourceFor = (split: TtsSplit) => sources.data?.items.find(source => source.split === split);
  const busy = sources.data?.items.some(source => source.state === 'checking') || false;
  const blocked = readOnly || pending || busy;
  const splitName = (split: TtsSplit) => split === 'train' ? text('训练数据', 'Training data') : text('验证数据', 'Validation data');
  const stateName = (state: TtsSource['state']) => ({ unchecked: text('未检查', 'Unchecked'), checking: text('正在检查', 'Checking'), valid: text('检查通过', 'Passed'), invalid: text('存在无效条目', 'Invalid rows'), stale: text('内容已变化', 'Content changed'), error: text('检查失败', 'Check failed') })[state];
  const open = (split: TtsSplit, action: SourceDialog['action']) => {
    if (blocked || !sources.data || gptSovits && split === 'validation' && action !== 'remove') return;
    setDialog({ split, action, path: sourceFor(split)?.path || '', revision: sources.data.data_revision, conflict: false });
    setDialogError('');
  };
  const updateCache = (response: TtsSourcesResponse) => client.setQueryData<TtsSourcesResponse>(queryKey, previous => !previous || response.data_revision >= previous.data_revision ? response : previous);
  const refreshDialogRevision = async () => {
    const fresh = await sources.refetch();
    if (mounted.current && fresh.data && !fresh.error) setDialog(value => value && { ...value, revision: fresh.data.data_revision });
  };
  const mutate = async (action: 'check' | 'put' | 'remove', split: TtsSplit) => {
    if (locked.current || blocked || !sources.data || gptSovits && split === 'validation' && action !== 'remove') return;
    const currentDialog = dialog;
    const source = sourceFor(split);
    const revision = action === 'check' ? sources.data.data_revision : currentDialog?.revision;
    if (revision === undefined || (action === 'check' && !source)) return;
    locked.current = true; setPending(true); setError(''); setDialogError('');
    try {
      await client.cancelQueries({ queryKey, exact: true });
      if (action === 'check') {
        const checked = await ttsApi.checkSource(pid, vid, source!.id, { expected_data_revision: revision });
        client.setQueryData<TtsSourcesResponse>(queryKey, previous => {
          if (!previous || previous.data_revision > checked.data_revision) return previous;
          const latest = previous.items.find(item => item.id === checked.id);
          if (latest?.check_id === checked.check_id && latest.state !== 'unchecked' && latest.state !== 'checking') return previous;
          return { ...previous, data_revision: checked.data_revision, items: previous.items.map(item => item.id === checked.id ? checked : item) };
        });
      } else {
        const result = action === 'put' ? await ttsApi.putSource(pid, vid, split, { expected_data_revision: revision, path: currentDialog!.path.trim() }) : await ttsApi.removeSource(pid, vid, split, revision);
        updateCache(result);
        if (mounted.current) { setDialog(null); if (browsing === split) setBrowsing(null); }
      }
      await client.invalidateQueries({ queryKey, exact: true });
    } catch (caught) {
      if (!mounted.current) return;
      const message = formatApiError(caught);
      if (action !== 'check' && statusCode(caught) === 409) {
        setDialogError(`${message}\n${text('输入已保留。请核对最新来源后再次确认。', 'Your input is retained. Review the latest source, then confirm again.')}`);
        setDialog(value => value && { ...value, conflict: true });
        await refreshDialogRevision();
      } else {
        if (action === 'check') setError(message); else setDialogError(message);
        if (statusCode(caught) === 409) await sources.refetch();
      }
    } finally { locked.current = false; if (mounted.current) setPending(false); }
  };
  const visibleSplits = splits.filter(split => !gptSovits || split === 'train' || sourceFor(split));
  const chosenSplit = browsing && visibleSplits.includes(browsing) ? browsing : 'train';
  const selectedSource = sourceFor(chosenSplit);
  const browse = (split: TtsSplit) => { setBrowsing(split); setStage('rows'); };
  const formatHelp = gptSovits
    ? text('JSONL：每行包含 audio、text、language，可选 speaker。\n.list：使用 audio|speaker|language|text 四列。\n语言支持 zh、en、ja、ko、yue。音频须为 32000、44100 或 48000 Hz 单声道 PCM WAV。\nGPT-SoVITS 不使用 ref_audio、dataset_id 或独立验证清单；数据检查保留原始音频。', 'JSONL: each line contains audio, text, language and optional speaker.\n.list: use four columns, audio|speaker|language|text.\nSupported languages: zh, en, ja, ko, yue. Audio must be 32000, 44100 or 48000 Hz mono PCM WAV.\nGPT-SoVITS does not use ref_audio, dataset_id or a separate validation manifest. Data checks preserve the original audio.')
    : text('JSONL：每行包含 audio 和 text；参考音频使用 ref_audio。\n音频须为 44100 Hz 单声道 PCM WAV。\n验证清单可选，留空时不运行验证。登记与检查不会修改原始音频。', 'JSONL: each line contains audio and text; use ref_audio for reference audio.\nAudio must be 44100 Hz mono PCM WAV.\nA validation manifest is optional. Without one, validation will not run. Registration and checks preserve the original audio.');
  const stageOptions = [
    { id: 'sources', label: text('数据清单', 'Data sources'), icon: FolderOpen },
    { id: 'rows', label: text('音频明细', 'Audio rows'), icon: AudioLines },
    { id: 'issues', label: text('检查问题', 'Check issues'), icon: ListChecks },
  ] as const;
  return <section className="tts-sources-panel dataset-pipeline" aria-label={text('音频与转写', 'Audio and transcripts')}>
    <div className="pipeline-navigation">
      <OverflowStrip className="dataset-stages" label={text('语音数据处理', 'Speech data workflow')} activeKey={stage} role="navigation">
        {stageOptions.map(({ id, label, icon: Icon }) => <button key={id} type="button" aria-current={stage === id ? 'step' : undefined} aria-controls={stageId} onClick={() => setStage(id)}><Icon size={15} aria-hidden="true"/><span>{label}</span></button>)}
        <SlidingIndicator className="ui-segmented-thumb dataset-stage-indicator"/>
      </OverflowStrip>
    </div>
    {readOnly && <p className="tts-source-muted">{text('当前版本只读，可查看检查结果和试听音频。', 'This version is read-only. Check results and audio previews remain available.')}</p>}
    {sources.isPending && <p role="status" className="workspace-loading"><Loader2 size={14} className="animate-spin"/>{text('正在读取数据来源…', 'Loading data sources…')}</p>}
    {(sources.error || error) && <p role="alert" className="workspace-message error">{error || formatApiError(sources.error)}</p>}
    <div className="pipeline-stage-body" id={stageId} ref={stageBody}>
      <header className="tts-sources-heading">
        <div><h2>{stageOptions.find(item => item.id === stage)?.label}</h2>{stage === 'sources' && <ConfigHelp label={text('清单格式', 'Manifest format')}>{formatHelp}</ConfigHelp>}</div>
        <div className="tts-source-actions">
          {stage !== 'sources' && visibleSplits.length > 1 && <StudioSelect aria-label={text('数据来源', 'Data source')} value={chosenSplit} onValueChange={value => setBrowsing(value as TtsSplit)} options={visibleSplits.map(split => ({ value: split, label: splitName(split) }))}/>}
          {stage !== 'sources' && selectedSource && !(gptSovits && chosenSplit === 'validation') && <button type="button" className="ui-btn ui-btn-sm" disabled={blocked} onClick={() => void mutate('check', chosenSplit)}><ListChecks size={14}/>{text('检查数据', 'Check data')}</button>}
          <button type="button" className="ui-btn ui-btn-sm ui-btn-icon" onClick={refresh} disabled={sources.isFetching} title={text('刷新', 'Refresh')} aria-label={text('刷新', 'Refresh')}><RefreshCw size={14}/></button>
        </div>
      </header>
      {stage === 'sources' && sources.data && <div className="tts-source-library">{visibleSplits.map(split => {
        const source = sourceFor(split), summary = source?.summary;
        const unsupported = gptSovits && split === 'validation';
        return <section className="tts-source-section" key={split} aria-label={splitName(split)}>
          <header><FileAudio size={20} className="tts-source-icon" aria-hidden="true"/><div className="tts-source-title"><h3>{splitName(split)}</h3>{source ? <span title={source.path}>{source.path.split(/[\\/]/).pop()}</span> : <span>{split === 'train' ? text('尚未登记训练清单。', 'No training manifest registered.') : text('尚未登记验证清单。留空时不运行验证。', 'No validation manifest registered. Validation will not run without one.')}</span>}</div>
            {source && <span role="status" className="tts-source-state" data-state={source.state}>{source.state === 'checking' && <Loader2 size={13} className="animate-spin"/>}{stateName(source.state)}</span>}
            <div className="tts-source-actions">{!unsupported && <button type="button" className={`ui-btn ui-btn-sm${source ? '' : ' ui-btn-primary'}`} disabled={blocked} onClick={() => open(split, 'put')}><FolderOpen size={14}/>{source ? text('更换清单', 'Replace manifest') : text('登记清单', 'Register manifest')}</button>}
              {source && <>{!unsupported && <button type="button" className="ui-btn ui-btn-sm" disabled={blocked} onClick={() => void mutate('check', split)}><ListChecks size={14}/>{text('检查数据', 'Check data')}</button>}<button type="button" className="ui-btn ui-btn-sm ui-btn-quiet" disabled={blocked} onClick={() => open(split, 'remove')}>{text('移除', 'Remove')}</button></>}</div>
          </header>
          {unsupported && <p className="tts-source-muted">{text('此验证清单不参与 GPT-SoVITS 训练，可查看已有结果或移除登记。', 'This validation manifest is not used by GPT-SoVITS training. You can inspect existing results or remove its registration.')}</p>}
          {source && <>
            <dl className="tts-source-summary"><div><dt>{text('音频条目', 'Audio rows')}</dt><dd>{summary?.clips_count.toLocaleString() ?? '—'}</dd></div><div><dt>{text('有效条目', 'Valid rows')}</dt><dd>{summary?.valid_clips_count.toLocaleString() ?? '—'}</dd></div><div><dt>{text('无效条目', 'Invalid rows')}</dt><dd>{summary?.invalid_count.toLocaleString() ?? '—'}</dd></div><div><dt>{text('总时长', 'Total duration')}</dt><dd>{summary ? `${summary.duration_seconds.toFixed(1)} s` : '—'}</dd></div></dl>
            <footer className="tts-source-footer"><details className="tts-source-location"><summary>{text('清单路径', 'Manifest path')}</summary><div className="tts-source-path" tabIndex={0}>{source.path}</div></details>
              {!!source.issues.length && <button type="button" className="ui-link" onClick={() => { setBrowsing(split); setStage('issues'); }}>{text('查看检查问题', 'View check issues')}{source.issues_total !== null && ` (${source.issues_total})`}</button>}
              {complete(source) && <button type="button" className="ui-link tts-source-browse" onClick={() => browse(split)}>{text('浏览音频明细', 'Browse audio rows')}<ChevronRight size={13}/></button>}
            </footer>
          </>}
        </section>;
      })}</div>}
      {stage !== 'sources' && sources.data && <>
        {selectedSource && <div className="tts-source-context"><span title={selectedSource.path}>{selectedSource.path.split(/[\\/]/).pop()}</span><span role="status" className="tts-source-state" data-state={selectedSource.state}>{stateName(selectedSource.state)}</span></div>}
        {stage === 'rows' && selectedSource && complete(selectedSource) ? <SourceRows key={`${selectedSource.id}:${selectedSource.snapshot_id}`} source={selectedSource} refresh={refresh} gptSovits={gptSovits}/> : stage === 'issues' && selectedSource?.issues.length ? <div className="tts-source-issue-panel"><IssueList issues={selectedSource.issues}/>{selectedSource.issues_truncated && <p className="tts-source-muted">{text('这里只列出部分问题；音频明细保留每行的完整检查结果。', 'This summary shows only some issues. Browse the audio rows for complete results.')}</p>}{complete(selectedSource) && <button type="button" className="ui-link" onClick={() => browse(chosenSplit)}>{text('浏览音频明细', 'Browse audio rows')}<ChevronRight size={13}/></button>}</div> : <div className="results-empty"><AudioLines size={24} aria-hidden="true"/><strong>{!selectedSource ? text('尚未登记音频清单', 'No audio manifest registered') : selectedSource.state === 'checking' ? text('正在检查数据', 'Checking data') : stage === 'issues' && selectedSource.state === 'valid' ? text('未发现检查问题', 'No check issues found') : text('尚无检查结果', 'No check results yet')}</strong><span>{!selectedSource ? text('先登记清单，再检查音频和转写。', 'Register a manifest, then check audio and transcripts.') : selectedSource.state === 'checking' ? text('检查完成后，结果会自动更新。', 'Results update automatically when checking finishes.') : selectedSource.state !== 'valid' ? text('检查数据后，可查看逐行结果和试听音频。', 'Check data to view row results and preview audio.') : text('可在音频明细中查看转写和试听录音。', 'View transcripts and preview recordings in Audio rows.')}</span>{!selectedSource && <button type="button" className="ui-btn ui-btn-sm" onClick={() => setStage('sources')}>{text('登记音频清单', 'Register audio manifest')}</button>}</div>}
      </>}
    </div>
    {dialog && <Dialog title={`${dialog.action === 'put' ? text('登记清单', 'Register manifest') : text('移除清单', 'Remove manifest')} · ${splitName(dialog.split)}`} onClose={() => setDialog(null)} closeDisabled={pending}>
      {dialog.action === 'put' ? <><fieldset className="tts-source-path-field" disabled={pending || readOnly}><label>{gptSovits ? text('JSONL / .list 清单路径', 'JSONL / .list manifest path') : text('JSONL 清单路径', 'JSONL manifest path')}</label><PathInput ariaLabel={gptSovits ? text('JSONL / .list 清单路径', 'JSONL / .list manifest path') : text('JSONL 清单路径', 'JSONL manifest path')} value={dialog.path} browsePath={gptSovits ? listParent(dialog.path) : undefined} onChange={path => setDialog(value => value && { ...value, path })}/></fieldset><div className="tts-source-format-help"><p className="tts-source-muted">{text('填写训练服务可访问的清单路径。', 'Enter a manifest path accessible to the training service.')}</p><ConfigHelp label={text('清单格式', 'Manifest format')}>{formatHelp}</ConfigHelp></div></> : <p>{gptSovits ? text('仅移除本版本的登记与索引，原始清单和音频文件都会保留。', 'Remove the registration and index from this version. Original manifests and audio files will be kept.') : text('仅移除本版本的登记与索引，原始 JSONL、音频和参考音频文件都会保留。', 'Remove the registration and index from this version. Original JSONL, audio, and reference audio files will be kept.')}</p>}
      {dialog.conflict && <p className="tts-source-muted">{text('当前登记', 'Current registration')}：<span className="tts-source-conflict-path">{sourceFor(dialog.split)?.path || text('未登记', 'Not registered')}</span></p>}
      {dialogError && <p role="alert" className="tts-source-error">{dialogError}</p>}
      {dialog.conflict && sources.error && <div role="alert" className="tts-source-error">{formatApiError(sources.error)} <button type="button" className="ui-link" disabled={sources.isFetching} onClick={() => void refreshDialogRevision()}>{text('读取最新登记', 'Reload current registration')}</button></div>}
      <footer className="tts-source-dialog-actions"><button type="button" className="ui-btn" disabled={pending} onClick={() => setDialog(null)}>{text('取消', 'Cancel')}</button><button type="button" className="ui-btn ui-btn-primary" disabled={blocked || sources.isFetching || !dialog.path.trim() || (dialog.conflict && !!sources.error)} onClick={() => void mutate(dialog.action, dialog.split)}>{pending ? text('正在保存…', 'Saving…') : dialog.conflict ? text('确认重试', 'Confirm and retry') : dialog.action === 'put' ? text('登记', 'Register') : text('移除', 'Remove')}</button></footer>
    </Dialog>}
  </section>;
}

export default function TtsSourcesPanel(props: Props) {
  return <SourcesForVersion key={`${props.projectId}:${props.versionId}:${props.engine || 'voxcpm1.5'}`} {...props}/>;
}
