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
type Group = 'trainer' | 'projects' | 'models' | 'other';
type Text = ReturnType<typeof useWorkspaceText>;

const GROUPS: Group[] = ['trainer', 'projects', 'models', 'other'];
const ROWS_SHOWN = 8;

function groupLabel(group: Group, text: Text) {
  return ({ trainer: text('训练器', 'Trainer'), projects: text('项目', 'Projects'), models: text('模型权重', 'Model weights'), other: text('其他数据', 'Other data') } as const)[group];
}

const percent = (value: number, total: number) => {
  const share = total ? value / total * 100 : 0;
  return share >= 10 ? `${Math.round(share)}%` : share >= 0.1 ? `${share.toFixed(1)}%` : '<0.1%';
};

/** The studio's total, split into the trainer, projects, model weights and other data. */
function UsageBar({ usage, text }: { usage: Usage; text: Text }) {
  const [tip, setTip] = React.useState<{ value: string; label: string; x: number } | null>(null);
  const bar = React.useRef<HTMLDivElement>(null);
  const segments = GROUPS.map(group => ({ id: group, bytes: usage.groups[group] }));
  const total = segments.reduce((sum, segment) => sum + segment.bytes, 0);
  const share = (segment: typeof segments[number]) => text(`占 ${percent(segment.bytes, total)}`, `${percent(segment.bytes, total)} of the total`);
  const describe = (segment: typeof segments[number]) => `${groupLabel(segment.id, text)} ${formatBytes(segment.bytes)}，${share(segment)}`;
  const show = (segment: typeof segments[number], target: HTMLElement) => {
    const box = bar.current?.getBoundingClientRect();
    const rect = target.getBoundingClientRect();
    if (box) setTip({ value: formatBytes(segment.bytes), label: `${groupLabel(segment.id, text)} · ${share(segment)}`, x: Math.min(Math.max(rect.left + rect.width / 2 - box.left, 70), box.width - 70) });
  };
  return <figure className="storage-chart">
    <figcaption><span>{text('共占用', 'Total')}</span><strong>{formatBytes(total)}</strong></figcaption>
    <div className="storage-bar-frame">
      <div ref={bar} className="storage-bar" role="img" aria-label={segments.map(describe).join(text('；', '; '))} onPointerLeave={() => setTip(null)}>
        {segments.map(segment => total > 0 && segment.bytes / total >= 0.001 && <span key={segment.id} className="storage-segment" data-segment={segment.id} style={{ flexGrow: segment.bytes }} tabIndex={0} aria-label={describe(segment)}
          onPointerEnter={event => show(segment, event.currentTarget)} onFocus={event => show(segment, event.currentTarget)} onBlur={() => setTip(null)}/>)}
      </div>
      {tip && <span className="storage-tooltip" role="tooltip" style={{ left: tip.x }}><strong>{tip.value}</strong>{tip.label}</span>}
    </div>
    <ul className="storage-legend">
      {segments.map(segment => <li key={segment.id}><span className="storage-swatch" data-segment={segment.id} aria-hidden="true"/><span>{groupLabel(segment.id, text)}</span><strong>{formatBytes(segment.bytes)}</strong></li>)}
    </ul>
  </figure>;
}

/** Each project version's total, largest first; versions under 1 MB are left out. */
function VersionList({ usage, text }: { usage: Usage; text: Text }) {
  const [open, setOpen] = React.useState(false);
  const rows = usage.versions.filter(row => row.bytes >= 1e6).sort((a, b) => b.bytes - a.bytes);
  if (!rows.length) return null;
  return <section className="storage-versions" aria-label={text('项目版本', 'Project versions')}>
    <h3>{text('项目版本', 'Project versions')}</h3>
    <ul>{(open ? rows : rows.slice(0, ROWS_SHOWN)).map(row => {
      const label = `${row.project_name} · ${row.name}`;
      return <li key={row.version_id}><span title={label}>{label}</span><strong>{formatBytes(row.bytes)}</strong></li>;
    })}</ul>
    {rows.length > ROWS_SHOWN && <button type="button" className="ui-link storage-more" aria-expanded={open} onClick={() => setOpen(value => !value)}>
      {open ? text('收起', 'Show fewer') : text(`显示全部 ${rows.length} 个版本`, `Show all ${rows.length} versions`)}</button>}
  </section>;
}

/** The disk space the studio's own files take, below the storage paths. */
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
      <UsageBar usage={usage} text={text}/>
      <VersionList usage={usage} text={text}/>
    </div> : loading && <LoadingNote block label={text('正在统计磁盘占用…', 'Counting disk usage…')}/>}
  </section>;
}
