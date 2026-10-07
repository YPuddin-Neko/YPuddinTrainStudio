import React from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { ChevronLeft, ChevronRight, Loader2, Play, RefreshCw, X } from 'lucide-react';
import { ttsApi, ttsAudioUrl, type TtsIssue, type TtsSource, type TtsSourceChanged, type TtsSourceRow, type TtsSourcesResponse, type TtsSplit } from '../../api/tts';
import Dialog from '../../components/Dialog';
import { PathInput } from '../../components/PathBrowser';
import StudioSelect from '../../components/StudioSelect';
import { EVENT_TYPES } from '../../events/eventTypes';
import { useEventStream, useEventStreamStatus } from '../../events/useEventStream';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import './tts-sources.css';

interface Props { projectId: string; versionId: string; readOnly: boolean }
type SourceDialog = { split: TtsSplit; action: 'put' | 'remove'; path: string; revision: number; conflict: boolean };
const splits: TtsSplit[] = ['train', 'validation'];
const complete = (source: TtsSource) => (source.state === 'valid' || source.state === 'invalid') && source.snapshot_id !== null;
const statusCode = (error: unknown) => (error as { status?: number } | null)?.status;

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

function AudioPreview({ url, label, onClose }: { url: string; label: string; onClose: () => void }) {
  const text = useWorkspaceText();
  const audio = React.useRef<HTMLAudioElement>(null);
  const [failed, setFailed] = React.useState(false);
  React.useEffect(() => {
    const element = audio.current;
    if (element && element.getAttribute('src') !== url) element.setAttribute('src', url);
    return () => { element?.pause(); element?.removeAttribute('src'); element?.load(); };
  }, [url]);
  return <div className="tts-source-player">
    <div><strong>{label}</strong><button type="button" className="ui-btn ui-btn-quiet ui-btn-icon" onClick={onClose} aria-label={text('关闭试听', 'Close audio preview')}><X size={14}/></button></div>
    <audio ref={audio} controls autoPlay preload="none" src={url} onError={() => setFailed(true)} aria-label={label}/>
    {failed && <p role="alert" className="tts-source-error">{text('音频无法读取。请重新检查来源后再试听。', 'Audio could not be read. Check the source again before previewing.')}</p>}
  </div>;
}

function SourceRows({ source, refresh }: { source: TtsSource; refresh: () => void }) {
  const text = useWorkspaceText();
  const [page, setPage] = React.useState(1);
  const [pageSize, setPageSize] = React.useState(50);
  const [preview, setPreview] = React.useState<{ url: string; label: string } | null>(null);
  const { project_id: pid, version_id: vid } = source.scope;
  const rows = useQuery({
    queryKey: ['tts-source-rows', pid, vid, source.id, source.snapshot_id, page, pageSize],
    queryFn: ({ signal }) => ttsApi.rows(pid, vid, source.id, { snapshot_id: source.snapshot_id!, page, page_size: pageSize }, signal),
    retry: false,
    refetchOnWindowFocus: false,
  });
  const play = (row: TtsSourceRow, reference: boolean) => {
    const url = ttsAudioUrl(reference ? row.reference_audio_url : row.audio_url);
    if (url) setPreview({ url, label: text(`第 ${row.line} 行 · ${reference ? '参考音频' : '录音'}`, `Line ${row.line} · ${reference ? 'Reference audio' : 'Audio'}`) });
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
    {rows.data && !rows.error && <div className="tts-source-table-scroll" tabIndex={0} role="region" aria-label={text('音频明细表格', 'Audio rows table')}><table><thead><tr>
      <th scope="col">{text('行', 'Line')}</th><th scope="col">{text('音频', 'Audio')}</th><th scope="col">{text('转写', 'Transcript')}</th><th scope="col">{text('检查结果', 'Check results')}</th>
    </tr></thead><tbody>{rows.data.items.map(row => <tr key={row.id}>
      <td>{row.line}</td>
      <td><div className="tts-source-audio-name">{row.audio_name || '—'}</div><small>{row.duration_seconds === null ? '—' : `${row.duration_seconds.toFixed(2)} s`} · {row.sample_rate === null ? '—' : `${row.sample_rate} Hz`} · {row.channels === null ? '—' : text(`${row.channels} 声道`, `${row.channels} ch`)}</small>
        <div className="tts-source-row-actions"><button type="button" className="ui-btn ui-btn-sm ui-btn-quiet" disabled={!ttsAudioUrl(row.audio_url)} onClick={() => play(row, false)} aria-label={text(`试听第 ${row.line} 行音频`, `Preview line ${row.line} audio`)}><Play size={12}/>{text('试听', 'Preview')}</button>
          {row.reference_audio_name && <button type="button" className="ui-btn ui-btn-sm ui-btn-quiet" disabled={!ttsAudioUrl(row.reference_audio_url)} title={row.reference_audio_name} onClick={() => play(row, true)} aria-label={text(`试听第 ${row.line} 行参考音频`, `Preview line ${row.line} reference audio`)}><Play size={12}/>{text('参考音频', 'Reference')}</button>}</div></td>
      <td className="tts-source-transcript">{row.text === null ? '—' : row.text}</td>
      <td>{row.issues.length ? <IssueList issues={row.issues}/> : <span className="tts-source-muted">{text('通过', 'Passed')}</span>}</td>
    </tr>)}</tbody></table>{!rows.data.items.length && <p className="tts-source-muted">{text('没有音频条目。', 'No audio rows.')}</p>}</div>}
  </div>;
}

function SourcesForVersion({ projectId: pid, versionId: vid, readOnly }: Props) {
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
    if (blocked || !sources.data) return;
    setDialog({ split, action, path: sourceFor(split)?.path || '', revision: sources.data.data_revision, conflict: false });
    setDialogError('');
  };
  const updateCache = (response: TtsSourcesResponse) => client.setQueryData<TtsSourcesResponse>(queryKey, previous => !previous || response.data_revision >= previous.data_revision ? response : previous);
  const refreshDialogRevision = async () => {
    const fresh = await sources.refetch();
    if (mounted.current && fresh.data && !fresh.error) setDialog(value => value && { ...value, revision: fresh.data.data_revision });
  };
  const mutate = async (action: 'check' | 'put' | 'remove', split: TtsSplit) => {
    if (locked.current || blocked || !sources.data) return;
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
  return <section className="tts-sources-panel" aria-label={text('音频与转写', 'Audio and transcripts')}>
    <header className="tts-sources-heading"><h2>{text('音频与转写', 'Audio and transcripts')}</h2><button type="button" className="ui-btn ui-btn-sm ui-btn-quiet" onClick={refresh} disabled={sources.isFetching}><RefreshCw size={14}/>{text('刷新', 'Refresh')}</button></header>
    <p className="tts-source-muted">{text('登记 JSONL 清单，每行包含 audio 和 text；参考音频使用 ref_audio。音频须为 44100 Hz 单声道 PCM WAV。', 'Register a JSONL manifest with audio and text on each line; use ref_audio for reference audio. Audio must be 44100 Hz mono PCM WAV.')}</p>
    {readOnly && <p className="tts-source-muted">{text('当前版本只读，可查看检查结果和试听音频。', 'This version is read-only. Check results and audio previews remain available.')}</p>}
    {sources.isPending && <p role="status" className="tts-source-muted">{text('正在读取数据来源…', 'Loading data sources…')}</p>}
    {(sources.error || error) && <p role="alert" className="tts-source-error">{error || formatApiError(sources.error)}</p>}
    {sources.data && splits.map(split => {
      const source = sourceFor(split);
      const summary = source?.summary;
      return <section className="tts-source-section" key={split} aria-label={splitName(split)}>
        <header><h3>{splitName(split)}</h3>{source && <span role="status" className="tts-source-state" data-state={source.state}>{source.state === 'checking' && <Loader2 size={13} className="animate-spin"/>}{stateName(source.state)}</span>}
          <div className="tts-source-actions"><button type="button" className="ui-btn ui-btn-sm" disabled={blocked} onClick={() => open(split, 'put')}>{source ? text('更换清单', 'Replace manifest') : text('登记清单', 'Register manifest')}</button>
            {source && <><button type="button" className="ui-btn ui-btn-sm" disabled={blocked} onClick={() => void mutate('check', split)}>{text('检查数据', 'Check data')}</button><button type="button" className="ui-btn ui-btn-sm ui-btn-quiet" disabled={blocked} onClick={() => open(split, 'remove')}>{text('移除', 'Remove')}</button></>}</div></header>
        {!source ? <p className="tts-source-muted">{split === 'train' ? text('尚未登记训练清单。', 'No training manifest registered.') : text('尚未登记验证清单。留空时不运行验证。', 'No validation manifest registered. Validation will not run without one.')}</p> : <>
          <div className="tts-source-path" title={source.path}>{source.path}</div>
          <dl className="tts-source-summary"><div><dt>{text('音频条目', 'Audio rows')}</dt><dd>{summary?.clips_count.toLocaleString() ?? '—'}</dd></div><div><dt>{text('有效条目', 'Valid rows')}</dt><dd>{summary?.valid_clips_count.toLocaleString() ?? '—'}</dd></div><div><dt>{text('无效条目', 'Invalid rows')}</dt><dd>{summary?.invalid_count.toLocaleString() ?? '—'}</dd></div><div><dt>{text('总时长', 'Total duration')}</dt><dd>{summary ? `${summary.duration_seconds.toFixed(1)} s` : '—'}</dd></div></dl>
          {!!source.issues.length && <details className="tts-source-issue-summary"><summary>{text('查看检查问题', 'View check issues')}{source.issues_total !== null && ` (${source.issues_total})`}</summary><IssueList issues={source.issues}/>{source.issues_truncated && <p className="tts-source-muted">{text('这里只列出部分问题；音频明细保留每行的完整检查结果。', 'This summary shows only some issues. Browse the audio rows for complete results.')}</p>}</details>}
          {complete(source) && <button type="button" className="ui-link tts-source-browse" aria-expanded={browsing === split} onClick={() => setBrowsing(browsing === split ? null : split)}>{browsing === split ? text('收起音频明细', 'Hide audio rows') : text('浏览音频明细', 'Browse audio rows')}</button>}
          {browsing === split && complete(source) && <SourceRows key={`${source.id}:${source.snapshot_id}`} source={source} refresh={refresh}/>}
        </>}
      </section>;
    })}
    {dialog && <Dialog title={`${dialog.action === 'put' ? text('登记清单', 'Register manifest') : text('移除清单', 'Remove manifest')} · ${splitName(dialog.split)}`} onClose={() => setDialog(null)} closeDisabled={pending}>
      {dialog.action === 'put' ? <><fieldset className="tts-source-path-field" disabled={pending || readOnly}><label>{text('JSONL 清单路径', 'JSONL manifest path')}</label><PathInput ariaLabel={text('JSONL 清单路径', 'JSONL manifest path')} value={dialog.path} onChange={path => setDialog(value => value && { ...value, path })}/></fieldset><p className="tts-source-muted">{text('填写训练服务可访问的文件路径。登记后检查数据，可查看逐行问题和试听音频。', 'Enter a file path accessible to the training service. Check the registered data to inspect row issues and preview audio.')}</p></> : <p>{text('仅移除本版本的登记与索引，原始 JSONL、音频和参考音频文件都会保留。', 'Remove the registration and index from this version. Original JSONL, audio, and reference audio files will be kept.')}</p>}
      {dialog.conflict && <p className="tts-source-muted">{text('当前登记', 'Current registration')}：<span className="tts-source-conflict-path">{sourceFor(dialog.split)?.path || text('未登记', 'Not registered')}</span></p>}
      {dialogError && <p role="alert" className="tts-source-error">{dialogError}</p>}
      {dialog.conflict && sources.error && <div role="alert" className="tts-source-error">{formatApiError(sources.error)} <button type="button" className="ui-link" disabled={sources.isFetching} onClick={() => void refreshDialogRevision()}>{text('读取最新登记', 'Reload current registration')}</button></div>}
      <footer className="tts-source-dialog-actions"><button type="button" className="ui-btn" disabled={pending} onClick={() => setDialog(null)}>{text('取消', 'Cancel')}</button><button type="button" className="ui-btn ui-btn-primary" disabled={blocked || sources.isFetching || !dialog.path.trim() || (dialog.conflict && !!sources.error)} onClick={() => void mutate(dialog.action, dialog.split)}>{pending ? text('正在保存…', 'Saving…') : dialog.conflict ? text('确认重试', 'Confirm and retry') : dialog.action === 'put' ? text('登记', 'Register') : text('移除', 'Remove')}</button></footer>
    </Dialog>}
  </section>;
}

export default function TtsSourcesPanel(props: Props) {
  return <SourcesForVersion key={`${props.projectId}:${props.versionId}`} {...props}/>;
}
