import type { JobLogLine } from '../api/types';
import { translateLogMessage } from './logTranslations';

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';
export type LogFilter = 'all' | 'info' | 'warn' | 'error';

/** One record with the lines that belong to it (traceback frames, multi-line text). */
export interface LogEntry {
  /** Byte offset of the record's first line: unique and ordered across reads. */
  id: number;
  kind: 'record' | 'traceback' | 'text';
  standalone?: boolean;
  ts: number | null;
  level: LogLevel;
  source: string | null;
  msg: string;
  /** Chinese reading of a fixed trainer message; the original stays in ``msg``. */
  translated?: string;
  detail: string[];
  translatedDetail?: string[];
}

const RANK: Record<LogLevel, number> = { debug: 10, info: 20, warn: 30, error: 40 };
const PROCESS_RANK = /^\[rank(\d+)\]:/;
const PROCESS_PREFIX = /^\[rank\d+\]:\s?/;
const WARNING_HEAD = /^(?:\[rank\d+\]:\s*)?(?:[^\n]+:\d+:\s*)?\w*Warning:/;
const FATAL_PYTHON = /^\s*Fatal Python error:/i;
const THREAD_DUMP = /^\s*(?:(?:Current thread|Thread)\s+0x[\da-f]+|Stack \(most recent call first\):|Extension modules:)/i;
const EXCEPTION_END = /^(?:[A-Za-z_][\p{L}\p{N}_]*\.)*(?:(?:[A-Za-z_][\p{L}\p{N}_]*)?(?:Error|Exception|Warning|Failure)|KeyboardInterrupt|SystemExit|GeneratorExit|StopIteration|StopAsyncIteration)(?::(?:\s|$)|$)/u;
const TRACEBACK_FRAME = /^\s+File ".+", line \d+(?:, in .*)?$/;
const PYTHON_IDENTIFIER = /^[\p{ID_Start}_][\p{ID_Continue}_]*$/u;
const INDEPENDENT_OUTPUT = /^(?:(?:Epoch|Total):\s*\d+(?:\s*\/\s*\d+)?\s*|Loading(?:\s.*|:.*)?|(?:(?:phoneme_data_len|wav_data_len|skipped_phone|skipped_dur):\s*\d+\s*)(?:,\s*(?:phoneme_data_len|wav_data_len|skipped_phone|skipped_dur):\s*\d+\s*)*|[^\s:]+\.(?:wav|flac|mp3|ogg|m4a|aac))$/i;
// Python prints one of these between an exception and the one raised while handling it.
const CHAINED = /^(?:During handling of the above exception, another exception occurred:|The above exception was the direct cause of the following exception:)$/;

function exceptionTerminal(body: string, hasFrame: boolean): boolean {
  // Captured progress can interrupt a traceback; its own record keeps its level.
  if (INDEPENDENT_OUTPUT.test(body)) return false;
  if (!hasFrame) return EXCEPTION_END.test(body);
  const colon = body.indexOf(':');
  if (colon >= 0 && body.length > colon + 1 && !/\s/.test(body[colon + 1])) return false;
  const parts = (colon < 0 ? body : body.slice(0, colon)).split('.');
  return parts.every((part, index) => (part === '<locals>' && index < parts.length - 1)
    || PYTHON_IDENTIFIER.test(part.normalize('NFKC')));
}

export function logLevel(value: string | null | undefined): LogLevel {
  const level = (value || '').toLowerCase();
  if (level === 'warning') return 'warn';
  if (level === 'critical' || level === 'fatal') return 'error';
  return level === 'debug' || level === 'warn' || level === 'error' ? level : 'info';
}

/**
 * Preserve each captured line unless its syntax identifies a continuation.
 * Logged tracebacks retain their header's level; independent output retains its
 * own time and source. Group over all loaded lines to cover page boundaries.
 */
export function groupLogLines(lines: JobLogLine[]): LogEntry[] {
  const entries: LogEntry[] = [];
  let current: LogEntry | null = null;
  let groupRank: string | undefined;
  // Frames end at an unindented exception; unrelated output starts a new entry.
  let traceback: 'none' | 'frames' | 'done' = 'none';
  let hasFrame = false;
  // The last line said the next traceback is chained to the one before.
  let chained = false;
  for (const line of lines) {
    const kind = line.kind ?? 'text';
    const level = logLevel(line.level);
    const body = line.msg.replace(PROCESS_PREFIX, '');
    const rank = line.msg.match(PROCESS_RANK)?.[1];
    const differentRank = rank !== undefined && groupRank !== undefined && rank !== groupRank;
    const differentSource = !!line.source && line.source !== 'process.output' && line.source !== current?.source;
    const warningHead = kind === 'text' && traceback !== 'frames' && WARNING_HEAD.test(line.msg);
    const header = current?.msg.replace(PROCESS_PREFIX, '') || '';
    const indented = /^\s/.test(body) && !!body.trim();
    const legacyDetail = line.ts == null && !line.source && current?.kind === 'record';
    const textDetail = level === 'info' && (legacyDetail
      || indented && (current?.kind === 'record' || WARNING_HEAD.test(header) || FATAL_PYTHON.test(header) || header.trimEnd().endsWith(':'))
      || !body.trim() && (current?.kind === 'record' || WARNING_HEAD.test(header) || FATAL_PYTHON.test(header) || header.trimEnd().endsWith(':'))
      || FATAL_PYTHON.test(header) && THREAD_DUMP.test(body));
    const joins = current !== null && !current.standalone && !line.standalone && !differentRank && !differentSource && (kind === 'traceback'
      ? (traceback === 'none' || chained) && (current.level === 'error' || current.kind === 'record' && current.level === 'warn')
      : kind === 'text' && !warningHead && (traceback === 'frames' && (!body.trim() || indented || exceptionTerminal(body, hasFrame))
        || traceback === 'done' && (!body.trim() || CHAINED.test(body.trim()))
        || traceback === 'none' && textDetail));
    if (joins && current) {
      current.detail.push(line.msg);
    } else {
      current = { id: line.offset, kind, ts: line.ts ?? null, level, source: line.source ?? null, msg: line.msg, detail: [],
        ...(line.standalone ? { standalone: true } : {}) };
      entries.push(current);
      traceback = 'none';
      hasFrame = false;
      groupRank = rank;
    }
    if (joins && groupRank === undefined && rank !== undefined) groupRank = rank;
    if (kind === 'traceback') { traceback = 'frames'; hasFrame = false; }
    else if (traceback === 'frames' && TRACEBACK_FRAME.test(body)) hasFrame = true;
    else if (traceback === 'frames' && body.trim() && !/^\s/.test(body)) traceback = 'done';
    if (body.trim()) chained = CHAINED.test(body.trim());
  }
  return entries;
}

export function translateLogEntries(entries: LogEntry[]): LogEntry[] {
  return entries.map(entry => {
    const translated = translateLogMessage(entry.msg);
    const detail = entry.detail.map(message => translateLogMessage(message));
    const hasDetail = detail.some(message => message !== null);
    return translated || hasDetail ? {
      ...entry,
      ...(translated ? { translated } : {}),
      ...(hasDetail ? { translatedDetail: detail.map((message, index) => message ?? entry.detail[index]) } : {}),
    } : entry;
  });
}

export function visibleLogEntries(entries: LogEntry[], { filter, debug, query }: { filter: LogFilter; debug: boolean; query: string }): LogEntry[] {
  const minimum = filter === 'error' ? RANK.error : filter === 'warn' ? RANK.warn : debug ? RANK.debug : RANK.info;
  const needle = query.trim().toLocaleLowerCase();
  return entries.filter(entry => (filter === 'info' ? entry.level === 'info' : RANK[entry.level] >= minimum) && (!needle
    || entry.msg.toLocaleLowerCase().includes(needle)
    || (entry.translated || '').toLocaleLowerCase().includes(needle)
    || (entry.source || 'process.output').toLocaleLowerCase().includes(needle)
    || entry.translatedDetail?.some(line => line.toLocaleLowerCase().includes(needle))
    || entry.detail.some(line => line.toLocaleLowerCase().includes(needle))));
}

/** Drop the package prefix: every trainer logger starts with it. */
export function logSource(source: string | null): string {
  return (source || 'process.output').replace(/^ypuddin\./, '');
}

export function logTime(ts: number | null): string {
  if (ts == null) return '';
  const date = new Date(ts * 1000);
  return [date.getHours(), date.getMinutes(), date.getSeconds()].map(part => String(part).padStart(2, '0')).join(':');
}

/** Every entry uses the level assigned by the log API. */
export function logLevelTag(entry: LogEntry): string {
  return `[${entry.level.toUpperCase()}]`;
}

export function logEntryText(entry: LogEntry): string {
  const time = logTime(entry.ts);
  const head = [time && `[${time}]`, logLevelTag(entry), `${entry.source || 'process.output'}:`, entry.translated ?? entry.msg].filter(Boolean).join(' ');
  return [head, ...(entry.translatedDetail ?? entry.detail)].join('\n');
}
