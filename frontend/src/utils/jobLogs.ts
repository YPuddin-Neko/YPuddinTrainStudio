import type { JobLogLine } from '../api/types';
import { translateLogMessage } from './logTranslations';

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';
export type LogFilter = 'all' | 'info' | 'warn' | 'error';

/** One record with the lines that belong to it (traceback frames, multi-line text). */
export interface LogEntry {
  /** Byte offset of the record's first line: unique and ordered across reads. */
  id: number;
  kind: 'record' | 'traceback' | 'text';
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
const WARNING_HEAD = /^(?:\[rank\d+\]:\s*)?(?:[^\n]+:\d+:\s*)?\w*Warning:/;

export function logLevel(value: string | null | undefined): LogLevel {
  const level = (value || '').toLowerCase();
  if (level === 'warning') return 'warn';
  if (level === 'critical' || level === 'fatal') return 'error';
  return level === 'debug' || level === 'warn' || level === 'error' ? level : 'info';
}

/**
 * Group parsed lines in file order. A header starts a record; a traceback starts
 * one unless it follows a warning or error record (``log.exception`` output).
 * Grouping runs over every loaded line because a record can span two reads.
 */
export function groupLogLines(lines: JobLogLine[]): LogEntry[] {
  const entries: LogEntry[] = [];
  let current: LogEntry | null = null;
  let traceback = false;
  for (const line of lines) {
    const kind = line.kind ?? 'text';
    const level = logLevel(line.level);
    const rank = line.msg.match(PROCESS_RANK)?.[1];
    const previousRank = current?.msg.match(PROCESS_RANK)?.[1];
    const differentRank = rank !== undefined && previousRank !== undefined && rank !== previousRank;
    const warningHead = kind === 'text' && !traceback && WARNING_HEAD.test(line.msg);
    const joins = current !== null && !differentRank && !warningHead
      && ((kind === 'text' && (level === 'info' || RANK[level] <= RANK[current.level]))
      || (kind === 'traceback' && RANK[current.level] >= RANK.warn));
    if (joins && current) {
      current.detail.push(line.msg);
      // Continuation text reads as info and keeps the record's level; an attached traceback raises a warning to an error.
      if (level !== 'info' && RANK[level] > RANK[current.level]) current.level = level;
      if (kind === 'traceback') traceback = true;
      continue;
    }
    current = { id: line.offset, kind, ts: line.ts ?? null, level, source: line.source ?? null, msg: line.msg, detail: [] };
    traceback = kind === 'traceback';
    entries.push(current);
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
    || (entry.source || '').toLocaleLowerCase().includes(needle)
    || entry.translatedDetail?.some(line => line.toLocaleLowerCase().includes(needle))
    || entry.detail.some(line => line.toLocaleLowerCase().includes(needle))));
}

/** Drop the package prefix: every trainer logger starts with it. */
export function logSource(source: string | null): string {
  return source ? source.replace(/^ypuddin\./, '') : '';
}

export function logTime(ts: number | null): string {
  if (ts == null) return '';
  const date = new Date(ts * 1000);
  return [date.getHours(), date.getMinutes(), date.getSeconds()].map(part => String(part).padStart(2, '0')).join(':');
}

/** ``[INFO]`` for records; plain output without a declared level has none. */
export function logLevelTag(entry: LogEntry): string {
  return entry.kind === 'text' && entry.level === 'info' ? '' : `[${entry.level.toUpperCase()}]`;
}

export function logEntryText(entry: LogEntry): string {
  const time = logTime(entry.ts);
  const head = [time && `[${time}]`, logLevelTag(entry), entry.source ? `${entry.source}:` : '', entry.translated ?? entry.msg].filter(Boolean).join(' ');
  return [head, ...(entry.translatedDetail ?? entry.detail)].join('\n');
}
