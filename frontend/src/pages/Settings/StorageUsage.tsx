import React from 'react';
import { RefreshCw } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { components } from '../../api/generated';
import { LoadingNote } from '../../components/Loading';
import { formatApiError } from '../../utils/errors';
import { formatBytes } from '../../utils/format';
import { useWorkspaceText } from '../../utils/workspaceText';
import './storage-usage.css';

type Usage = components['schemas']['StorageUsage'];
type Part = components['schemas']['StoragePart'];
type Group = Part['group'];
type Text = ReturnType<typeof useWorkspaceText>;

const GROUPS: Group[] = ['trainer', 'projects', 'models', 'other'];
const ROWS_SHOWN = 8;
// Parts smaller than this are counted in their group but not listed.
const LISTED_BYTES = 1e6;

function groupLabel(group: Group | 'foreign' | 'free', text: Text) {
  return ({
    trainer: text('训练器', 'Trainer'), projects: text('项目', 'Projects'), models: text('模型权重', 'Model weights'),
    other: text('其他数据', 'Other data'), foreign: text('其他文件', 'Other files'), free: text('可用', 'Free'),
  } as const)[group];
}

function partLabel(part: Part, text: Text) {
  if (part.group === 'projects') return `${part.project_name ?? ''} · ${part.name}`;
  if (part.group === 'models') return part.kind === 'external' ? text(`${part.name}（外部登记）`, `${part.name} (registered elsewhere)`) : part.name;
  return ({
    program: text('程序文件', 'Program files'), environment: text('运行环境', 'Runtime environments'),
    cache: text('共享缓存', 'Shared cache'), thumbnails: text('缩略图', 'Thumbnails'), runs: text('独立任务产物', 'Standalone job outputs'),
    database: text('数据库', 'Database'), rest: text('其余数据', 'Remaining data'),
  } as Record<string, string>)[part.key] ?? part.key;
}

const KINDS = ['products', 'resume', 'data', 'cache', 'other'] as const;
function kindLabel(kind: string, text: Text) {
  return ({
    products: text('训练产物', 'Outputs'), resume: text('恢复点', 'Resume points'), data: text('训练数据', 'Training data'),
    cache: text('编码缓存', 'Encoding cache'), other: text('其他', 'Other'),
  } as Record<string, string>)[kind] ?? kind;
}

type Row = { key: string; group: Group; label: string; bytes: number; detail?: string };

/** One row per listed part; a project version gathers its kinds into one row with the kinds beneath. */
function rows(parts: Part[], text: Text): Row[] {
  const versions = new Map<string, { part: Part; kinds: Record<string, number> }>();
  const listed: Row[] = [];
  for (const part of parts) {
    if (part.group === 'projects') {
      // Files of the project itself (such as its cover) belong to no version.
      const id = part.version_id ?? `${part.project_id}:project`;
      const entry = versions.get(id) ?? { part, kinds: {} };
      entry.kinds[part.kind ?? 'other'] = (entry.kinds[part.kind ?? 'other'] ?? 0) + part.bytes;
      versions.set(id, entry);
    } else {
      listed.push({ key: `${part.group}:${part.key}`, group: part.group, label: partLabel(part, text), bytes: part.bytes });
    }
  }
  for (const [id, { part, kinds }] of versions) {
    const bytes = Object.values(kinds).reduce((sum, value) => sum + value, 0);
    const label = part.version_id ? partLabel(part, text) : text(`${part.project_name}（项目文件）`, `${part.project_name} (project files)`);
    const detail = part.version_id ? KINDS.filter(kind => kinds[kind] >= LISTED_BYTES).map(kind => `${kindLabel(kind, text)} ${formatBytes(kinds[kind])}`).join(' · ') : undefined;
    listed.push({ key: `projects:${id}`, group: 'projects', label, bytes, detail });
  }
  return listed.filter(row => row.bytes >= LISTED_BYTES).sort((a, b) => b.bytes - a.bytes);
}

const percent = (value: number, total: number) => {
  const share = total ? value / total * 100 : 0;
  return share >= 10 ? `${Math.round(share)}%` : share >= 0.1 ? `${share.toFixed(1)}%` : '<0.1%';
};

/** Where one disk's space goes: the studio's parts, other files, and what is left. */
function DiskBar({ usage, volume, text }: { usage: Usage; volume: Usage['volumes'][number]; text: Text }) {
  const [tip, setTip] = React.useState<{ value: string; label: string; x: number } | null>(null);
  const bar = React.useRef<HTMLDivElement>(null);
  const studio = GROUPS.map(group => ({ id: group, bytes: usage.parts.filter(part => part.group === group).reduce((sum, part) => sum + (part.by_volume[volume.id] ?? 0), 0) }));
  const own = studio.reduce((sum, item) => sum + item.bytes, 0);
  const segments: { id: Group | 'foreign' | 'free'; bytes: number }[] = [...studio, { id: 'foreign', bytes: Math.max(0, volume.used - own) }, { id: 'free', bytes: volume.free }];
  const share = (segment: typeof segments[number]) => text(`占 ${percent(segment.bytes, volume.total)}`, `${percent(segment.bytes, volume.total)} of the disk`);
  const describe = (segment: typeof segments[number]) => `${groupLabel(segment.id, text)} ${formatBytes(segment.bytes)}，${share(segment)}`;
  const show = (segment: typeof segments[number], target: HTMLElement) => {
    const box = bar.current?.getBoundingClientRect();
    const rect = target.getBoundingClientRect();
    if (box) setTip({ value: formatBytes(segment.bytes), label: `${groupLabel(segment.id, text)} · ${share(segment)}`, x: Math.min(Math.max(rect.left + rect.width / 2 - box.left, 70), box.width - 70) });
  };
  return <figure className="storage-disk">
    <figcaption><strong title={volume.path}>{volume.path}</strong><span>{text(`本软件占用 ${formatBytes(own)} · 已用 ${formatBytes(volume.used)} / 共 ${formatBytes(volume.total)}`, `Studio ${formatBytes(own)} · ${formatBytes(volume.used)} used of ${formatBytes(volume.total)}`)}</span></figcaption>
    <div className="storage-bar-frame">
      <div ref={bar} className="storage-bar" role="img" aria-label={segments.map(describe).join(text('；', '; '))} onPointerLeave={() => setTip(null)}>
        {segments.map(segment => segment.bytes / volume.total >= 0.001 && <span key={segment.id} className="storage-segment" data-segment={segment.id} style={{ flexGrow: segment.bytes }} tabIndex={0} aria-label={describe(segment)}
          onPointerEnter={event => show(segment, event.currentTarget)} onFocus={event => show(segment, event.currentTarget)} onBlur={() => setTip(null)}/>)}
      </div>
      {tip && <span className="storage-tooltip" role="tooltip" style={{ left: tip.x }}><strong>{tip.value}</strong>{tip.label}</span>}
    </div>
    <ul className="storage-legend">
      {segments.map(segment => <li key={segment.id}><span className="storage-swatch" data-segment={segment.id} aria-hidden="true"/><span>{groupLabel(segment.id, text)}</span><strong>{formatBytes(segment.bytes)}</strong></li>)}
    </ul>
  </figure>;
}

function Breakdown({ usage, text }: { usage: Usage; text: Text }) {
  const [expanded, setExpanded] = React.useState<Set<Group>>(new Set());
  const listed = rows(usage.parts, text);
  const largest = Math.max(1, ...listed.map(row => row.bytes));
  return <div className="storage-breakdown">
    {GROUPS.map(group => {
      const members = listed.filter(row => row.group === group);
      const total = usage.parts.filter(part => part.group === group).reduce((sum, part) => sum + part.bytes, 0);
      if (!total) return null;
      const open = expanded.has(group);
      const shown = open ? members : members.slice(0, ROWS_SHOWN);
      return <section key={group} className="storage-group" aria-label={groupLabel(group, text)}>
        <h3><span className="storage-swatch" data-segment={group} aria-hidden="true"/>{groupLabel(group, text)}<span>{formatBytes(total)}</span></h3>
        <ul>{shown.map(row => <li key={row.key} className="storage-row">
          <span className="storage-row-label"><span title={row.label}>{row.label}</span>{row.detail && <small>{row.detail}</small>}</span>
          <span className="storage-row-bar" aria-hidden="true"><span data-segment={group} style={{ width: `${Math.max(0.5, row.bytes / largest * 100)}%` }}/></span>
          <strong>{formatBytes(row.bytes)}</strong>
        </li>)}</ul>
        {members.length > ROWS_SHOWN && <button type="button" className="ui-link storage-more" aria-expanded={open} onClick={() => setExpanded(previous => { const next = new Set(previous); if (open) next.delete(group); else next.add(group); return next; })}>
          {open ? text('收起', 'Show fewer') : text(`显示全部 ${members.length} 项`, `Show all ${members.length}`)}</button>}
      </section>;
    })}
  </div>;
}

/** Disk use of the trainer, each project version, model weights and other data, below the storage paths. */
export default function StorageUsage() {
  const text = useWorkspaceText();
  const [usage, setUsage] = React.useState<Usage | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState('');
  const controller = React.useRef<AbortController | null>(null);
  const load = React.useCallback(async (refresh: boolean) => {
    controller.current?.abort();
    const request = new AbortController(); controller.current = request;
    setLoading(true); setError('');
    try {
      const result = await apiClient.get<Usage>('/storage/usage', { params: refresh ? { refresh: true } : undefined, signal: request.signal, silent: true });
      if (!request.signal.aborted) setUsage(result);
    } catch (failure) { if (!request.signal.aborted) setError(formatApiError(failure)); }
    finally { if (!request.signal.aborted) setLoading(false); }
  }, []);
  React.useEffect(() => { void load(false); return () => controller.current?.abort(); }, [load]);
  const scanned = usage ? new Date(usage.scanned_at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
  return <section id="preferences-storage-usage" data-settings-section tabIndex={-1} className="settings-section storage-usage">
    <div className="settings-section-heading"><h2>{text('空间占用', 'Disk usage')}</h2>
      <div className="storage-usage-actions">{usage && <span className="settings-note">{text(`统计于 ${scanned}`, `Counted at ${scanned}`)}</span>}
        <button type="button" className="ui-btn ui-btn-sm ui-btn-icon" disabled={loading} onClick={() => void load(true)} aria-label={text('重新统计', 'Count again')} title={text('重新统计', 'Count again')}><RefreshCw size={14} className={loading ? 'animate-spin' : ''}/></button></div>
    </div>
    {error && <div role="alert" className="settings-alert">{error}<button type="button" className="ui-btn" onClick={() => void load(true)}>{text('重试', 'Retry')}</button></div>}
    {usage ? <div className="storage-usage-body" data-refreshing={loading || undefined} aria-busy={loading}>
      {usage.volumes.map(volume => <DiskBar key={volume.id} usage={usage} volume={volume} text={text}/>)}
      <Breakdown usage={usage} text={text}/>
    </div> : loading && <LoadingNote block label={text('正在统计磁盘占用…', 'Counting disk usage…')}/>}
  </section>;
}
