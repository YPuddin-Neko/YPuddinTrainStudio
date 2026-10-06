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
// Python prints one of these between an exception and the one raised while handling it.
const CHAINED = /^(?:During handling of the above exception, another exception occurred:|The above exception was the direct cause of the following exception:)$/;

export function logLevel(value: string | null | undefined): LogLevel {
  const level = (value || '').toLowerCase();
  if (level === 'warning') return 'warn';
  if (level === 'critical' || level === 'fatal') return 'error';
  return level === 'debug' || level === 'warn' || level === 'error' ? level : 'info';
}

/**
 * Group parsed lines in file order. A header starts a record and its level never
 * changes: the lines joined to it are details. A warning or error record keeps the
 * traceback logged with it (``exc_info``), so a warning with a traceback stays a
 * warning. Any other traceback starts an error entry of its own: one after an
 * info or debug record, after plain output, or after a warning whose own traceback
 * has ended. Grouping runs over every loaded line because a record can span two reads.
 */
export function groupLogLines(lines: JobLogLine[]): LogEntry[] {
  const entries: LogEntry[] = [];
  let current: LogEntry | null = null;
  // Inside a traceback's frames every line belongs to it, up to the unindented exception line.
  let traceback: 'none' | 'frames' | 'done' = 'none';
  // The last line said the next traceback is chained to the one before.
  let chained = false;
  for (const line of lines) {
    const kind = line.kind ?? 'text';
    const level = logLevel(line.level);
    const body = line.msg.replace(PROCESS_PREFIX, '');
    const rank = line.msg.match(PROCESS_RANK)?.[1];
    const previousRank = current?.msg.match(PROCESS_RANK)?.[1];
    const differentRank = rank !== undefined && previousRank !== undefined && rank !== previousRank;
    const warningHead = kind === 'text' && traceback !== 'frames' && WARNING_HEAD.test(line.msg);
    const joins = current !== null && !current.standalone && !line.standalone && !differentRank && (kind === 'traceback'
      ? current.kind === 'traceback' || current.level === 'error'
        || (current.kind === 'record' && current.level === 'warn' && (traceback === 'none' || chained))
      : kind === 'text' && !warningHead
        && (traceback === 'frames' || level === 'info' || RANK[level] <= RANK[current.level]));
    if (joins && current) {
      current.detail.push(line.msg);
    } else {
      current = { id: line.offset, kind, ts: line.ts ?? null, level, source: line.source ?? null, msg: line.msg, detail: [],
        ...(line.standalone ? { standalone: true } : {}) };
      entries.push(current);
      traceback = 'none';
    }
    if (kind === 'traceback') traceback = 'frames';
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
