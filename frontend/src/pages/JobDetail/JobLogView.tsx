import React from 'react';
import { useTranslation } from 'react-i18next';
import { ArrowDown, Check, Copy, Download, Loader2, Search, Terminal } from 'lucide-react';
import { apiClient, apiUrl } from '../../api/client';
import type { JobLogLine, JobLogResponse } from '../../api/types';
import StudioSelect from '../../components/StudioSelect';
import Switch from '../../components/Switch';
import { copyText } from '../../utils/clipboard';
import { formatApiError } from '../../utils/errors';
import { groupLogLines, logEntryText, logLevelTag, logSource, logTime, translateLogEntries, visibleLogEntries, type LogEntry, type LogFilter } from '../../utils/jobLogs';
import { logTone } from '../../utils/logTranslations';
import { useWorkspaceText } from '../../utils/workspaceText';
import './job-log.css';

const PAGE = 1000;
const MAX_LINES = 5000;
const POLL_MS = 1000;
const DEBUG_KEY = 'ypuddin.jobLog.debug';
const ORIGINAL_KEY = 'ypuddin.jobLog.original';

function storedFlag(key: string): boolean {
  try { return window.localStorage.getItem(key) === '1'; } catch { return false; }
}

function storeFlag(key: string, value: boolean): void {
  try { window.localStorage.setItem(key, value ? '1' : '0'); } catch { /* Preference only. */ }
}

function Highlight({ text, query }: { text: string; query: string }) {
  const needle = query.trim();
  if (!needle) return <>{text}</>;
  const lower = text.toLocaleLowerCase(), target = needle.toLocaleLowerCase();
  const parts: React.ReactNode[] = [];
  let start = 0, index = lower.indexOf(target);
  while (index >= 0) {
    if (index > start) parts.push(text.slice(start, index));
    parts.push(<mark key={index}>{text.slice(index, index + needle.length)}</mark>);
    start = index + needle.length; index = lower.indexOf(target, start);
  }
  parts.push(text.slice(start));
  return <>{parts}</>;
}

function LogRow({ entry, query }: { entry: LogEntry; query: string }) {
  const text = useWorkspaceText();
  const time = logTime(entry.ts);
  const source = entry.source || 'Unknown source';
  const detail = entry.translatedDetail ?? entry.detail;
  const original = entry.translated || entry.translatedDetail ? [entry.msg, ...entry.detail].join('\n') : undefined;
  return <div className="job-log-entry" data-level={entry.level} data-kind={entry.kind} data-tone={entry.level === 'info' ? logTone(entry.msg) ?? undefined : undefined}>
    <time className="job-log-time" dateTime={entry.ts == null ? undefined : new Date(entry.ts * 1000).toISOString()} title={entry.ts == null ? text('原始日志未记录时间', 'The original log has no timestamp') : new Date(entry.ts * 1000).toLocaleString()}>{`[${time || '--:--:--'}]`}</time>
    <span className="job-log-level">{logLevelTag(entry)}</span>
    <span className="job-log-source" title={source}>{logSource(source)}</span>
    <div className="job-log-message" title={original}><Highlight text={entry.translated ?? entry.msg} query={query}/>{detail.length > 0 && <pre><Highlight text={detail.join('\n')} query={query}/></pre>}</div>
  </div>;
}

/**
 * Live job log: reads new bytes while the worker runs and follows the end until
 * the reader scrolls up. Earlier output loads on demand above the first line.
 */
export default function JobLogView({ jobId, live, active, recordedLevel }: {
  jobId: string; live: boolean; active: boolean; recordedLevel?: string | null;
}) {
  const text = useWorkspaceText();
  const { i18n } = useTranslation();
  const chinese = (i18n.resolvedLanguage || i18n.language || '').startsWith('zh');
  const [original, setOriginal] = React.useState(() => storedFlag(ORIGINAL_KEY));
  const [lines, setLines] = React.useState<JobLogLine[]>([]);
  const [loaded, setLoaded] = React.useState(false);
  const [error, setError] = React.useState('');
  const [hasEarlier, setHasEarlier] = React.useState(false);
  const [loadingEarlier, setLoadingEarlier] = React.useState(false);
  const [filter, setFilter] = React.useState<LogFilter>('all');
  const [debug, setDebug] = React.useState(() => storedFlag(DEBUG_KEY));
  const [query, setQuery] = React.useState('');
  const [following, setFollowing] = React.useState(true);
  const [copied, setCopied] = React.useState<'done' | 'failed' | ''>('');
  const cursor = React.useRef({ start: 0, end: 0, busy: false, generation: 0 });
  const linesRef = React.useRef<JobLogLine[]>([]);
  const followingRef = React.useRef(true);
  const seenId = React.useRef(-1);
  const lastVisibleId = React.useRef(-1);
  const prependHeight = React.useRef<number | null>(null);
  const body = React.useRef<HTMLDivElement>(null);
  const endpoint = `/jobs/${encodeURIComponent(jobId)}/log`;

  const commit = (next: JobLogLine[]) => { linesRef.current = next; setLines(next); };
  const setFollow = (value: boolean) => {
    followingRef.current = value; setFollowing(value);
    seenId.current = lastVisibleId.current;
  };

  const load = React.useCallback(async () => {
    const generation = cursor.current.generation + 1;
    cursor.current = { start: 0, end: 0, busy: true, generation };
    commit([]); setLoaded(false); setError(''); setHasEarlier(false); followingRef.current = true; setFollowing(true);
    try {
      const page = await apiClient.get<JobLogResponse>(endpoint, { params: { tail: true, limit: PAGE }, silent: true });
      if (cursor.current.generation !== generation) return;
      cursor.current = { start: page.start_offset, end: page.next_offset, busy: false, generation };
      commit(page.lines); setHasEarlier(page.has_earlier); setLoaded(true);
    } catch (failure) {
      if (cursor.current.generation !== generation) return;
      cursor.current.busy = false; setError(formatApiError(failure)); setLoaded(true);
    }
  }, [endpoint]);
  React.useEffect(() => { void load(); return () => { cursor.current.generation += 1; }; }, [load]);

  const pull = React.useCallback(async () => {
    const state = cursor.current;
    if (state.busy) return;
    state.busy = true;
    const generation = state.generation;
    try {
      // A burst of output can exceed one read; catch up before the next tick.
      for (let round = 0; round < 5; round += 1) {
        const page = await apiClient.get<JobLogResponse>(endpoint, { params: { offset: state.end, limit: PAGE }, silent: true });
        if (cursor.current.generation !== generation) return;
        const fresh = page.lines.filter(line => line.offset >= state.end);
        state.end = Math.max(state.end, page.next_offset);
        if (fresh.length) {
          let next = linesRef.current.concat(fresh);
          if (next.length > MAX_LINES) {
            next = next.slice(next.length - MAX_LINES);
            state.start = next[0].offset; setHasEarlier(true);
          }
          commit(next);
        }
        setError('');
        if (!page.has_more) break;
      }
    } catch (failure) {
      if (cursor.current.generation === generation) setError(formatApiError(failure));
    } finally {
      if (cursor.current.generation === generation) state.busy = false;
    }
  }, [endpoint]);

  React.useEffect(() => {
    if (!active || !live || !loaded) return;
    void pull();
    const timer = window.setInterval(() => void pull(), POLL_MS);
    return () => window.clearInterval(timer);
  }, [active, live, loaded, pull]);
  // The worker writes its last lines as it exits; read them once after it stops.
  const wasLive = React.useRef(live);
  React.useEffect(() => {
    if (wasLive.current && !live && loaded) void pull();
    wasLive.current = live;
  }, [live, loaded, pull]);

  const loadEarlier = async () => {
    const state = cursor.current, generation = state.generation;
    if (loadingEarlier || !hasEarlier) return;
    setLoadingEarlier(true);
    try {
      const page = await apiClient.get<JobLogResponse>(endpoint, { params: { before: state.start, limit: PAGE }, silent: true });
      if (cursor.current.generation !== generation) return;
      const older = page.lines.filter(line => line.offset < state.start);
      state.start = Math.min(state.start, page.start_offset);
      prependHeight.current = body.current?.scrollHeight ?? null;
      commit(older.concat(linesRef.current));
      setHasEarlier(page.has_earlier);
    } catch (failure) {
      if (cursor.current.generation === generation) setError(formatApiError(failure));
    } finally {
      if (cursor.current.generation === generation) setLoadingEarlier(false);
    }
  };

  const grouped = React.useMemo(() => groupLogLines(lines), [lines]);
  // Fixed trainer lines read in Chinese; the original shows on hover and in downloads.
  const entries = React.useMemo(() => chinese && !original ? translateLogEntries(grouped) : grouped, [grouped, chinese, original]);
  const visible = React.useMemo(() => visibleLogEntries(entries, { filter, debug, query }), [entries, filter, debug, query]);
  lastVisibleId.current = visible.length ? visible[visible.length - 1].id : -1;
  const unseen = following ? 0 : visible.filter(entry => entry.id > seenId.current).length;

  // Scroll only when content or filters change: an empty poll renders nothing.
  React.useLayoutEffect(() => {
    const element = body.current;
    if (!element) return;
    if (prependHeight.current != null) {
      element.scrollTop += element.scrollHeight - prependHeight.current;
      prependHeight.current = null;
    } else if (followingRef.current) {
      element.scrollTop = element.scrollHeight;
      seenId.current = lastVisibleId.current;
    }
  }, [lines, filter, debug, query, active, original]);

  const onScroll = () => {
    const element = body.current;
    if (!element || prependHeight.current != null) return;
    const atEnd = element.scrollHeight - element.scrollTop - element.clientHeight < 24;
    if (atEnd !== followingRef.current) setFollow(atEnd);
  };
  const jumpToEnd = () => {
    const element = body.current;
    if (element) element.scrollTop = element.scrollHeight;
    setFollow(true);
  };
  const toggleDebug = (checked: boolean) => { setDebug(checked); storeFlag(DEBUG_KEY, checked); };
  const toggleOriginal = (checked: boolean) => { setOriginal(checked); storeFlag(ORIGINAL_KEY, checked); };
  const copyVisible = async () => {
    try { await copyText(visible.map(logEntryText).join('\n')); setCopied('done'); }
    catch { setCopied('failed'); }
    window.setTimeout(() => setCopied(''), 1600);
  };

  const levelNames: Record<string, string> = { info: text('信息', 'Info'), warning: text('警告', 'Warning') };
  const debugMissing = debug && filter === 'all' && !!recordedLevel && recordedLevel !== 'debug';
  const filtered = filter !== 'all' || !!query.trim();
  // A burst of debug records can fill the loaded window while debug lines are hidden.
  const onlyDebug = loaded && !filtered && !debug && lines.length > 0 && entries.every(entry => entry.level === 'debug');
  const empty = !loaded ? text('读取日志…', 'Loading log…')
    : lines.length === 0 ? (live ? text('等待任务输出…', 'Waiting for output…') : text('暂无日志', 'No log output'))
      : onlyDebug ? text('最近的日志都是调试信息', 'The latest lines are all debug records')
        : text('没有符合筛选条件的日志', 'No log lines match the filters');

  return <section className="job-log" aria-label={text('任务日志', 'Job log')}>
    <div className="job-log-toolbar">
      <StudioSelect className="job-log-filter" aria-label={text('显示级别', 'Levels shown')} value={filter} onValueChange={value => setFilter(value as LogFilter)} options={[
        { value: 'all', label: text('全部级别', 'All levels') },
        { value: 'info', label: text('仅信息', 'Info only') },
        { value: 'warn', label: text('警告及错误', 'Warnings and errors') },
        { value: 'error', label: text('仅错误', 'Errors only') },
      ]}/>
      <label className="job-log-search"><Search size={14} aria-hidden="true"/><input type="search" aria-label={text('搜索日志', 'Search log')} placeholder={text('搜索日志内容或来源', 'Search messages or sources')} value={query} onChange={event => setQuery(event.target.value)}/></label>
      <Switch className="job-log-debug" checked={debug} disabled={filter !== 'all'} onCheckedChange={toggleDebug}>{text('调试日志', 'Debug')}</Switch>
      {chinese && <Switch className="job-log-debug" checked={original} onCheckedChange={toggleOriginal}>显示原文</Switch>}
      <div className="job-log-actions">
        <button type="button" className="ui-btn ui-btn-icon ui-btn-quiet" disabled={!visible.length} onClick={() => void copyVisible()} aria-label={text('复制显示的日志', 'Copy shown lines')} title={copied === 'done' ? text('已复制', 'Copied') : copied === 'failed' ? text('复制失败', 'Copy failed') : text('复制显示的日志', 'Copy shown lines')}>{copied === 'done' ? <Check size={15}/> : <Copy size={15}/>}</button>
        <a className="ui-btn ui-btn-icon ui-btn-quiet" href={apiUrl(`/jobs/${encodeURIComponent(jobId)}/log/raw`)} download aria-label={text('下载完整日志', 'Download full log')} title={text('下载完整日志', 'Download full log')} aria-disabled={!lines.length || undefined} onClick={event => { if (!lines.length) event.preventDefault(); }}><Download size={15}/></a>
      </div>
    </div>
    {error && <div className="task-error" role="alert">{error}<button type="button" className="ui-btn ui-btn-sm" onClick={() => void (loaded && lines.length ? pull() : load())}>{text('重试', 'Retry')}</button></div>}
    {debugMissing && <p className="job-log-note" role="status">{text(`此任务按“${levelNames[recordedLevel!] || recordedLevel}”级别记录，没有调试日志。`, `This job was recorded at the ${levelNames[recordedLevel!] || recordedLevel} level and has no debug lines.`)}</p>}
    <div className="job-log-frame">
      <div ref={body} className="job-log-body" onScroll={onScroll} tabIndex={0} aria-label={text('日志内容', 'Log lines')} aria-busy={!loaded}>
        {hasEarlier && <div className="job-log-earlier"><button type="button" className="ui-btn ui-btn-sm ui-btn-quiet" disabled={loadingEarlier} onClick={() => void loadEarlier()}>{loadingEarlier ? <><Loader2 size={13} className="animate-spin" aria-hidden="true"/>{text('读取中…', 'Loading…')}</> : text('加载更早的日志', 'Load earlier lines')}</button></div>}
        {visible.length ? visible.map(entry => <LogRow key={entry.id} entry={entry} query={query}/>)
          : <div className="job-log-empty"><Terminal size={22} aria-hidden="true"/><span>{empty}</span>{filtered && lines.length > 0 && <button type="button" className="ui-link" onClick={() => { setFilter('all'); setQuery(''); }}>{text('清除筛选', 'Clear filters')}</button>}{onlyDebug && <button type="button" className="ui-link" onClick={() => toggleDebug(true)}>{text('显示调试日志', 'Show debug lines')}</button>}</div>}
      </div>
      {!following && visible.length > 0 && <button type="button" className="ui-btn ui-btn-sm job-log-latest" onClick={jumpToEnd}><ArrowDown size={14}/>{unseen ? text(`${unseen} 条新日志`, `${unseen} new entries`) : text('回到最新', 'Latest')}</button>}
    </div>
  </section>;
}
