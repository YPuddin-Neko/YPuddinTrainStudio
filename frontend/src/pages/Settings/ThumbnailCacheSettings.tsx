import React from 'react';
import { Loader2, RefreshCw, Trash2 } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { components } from '../../api/generated';
import { formatApiError } from '../../utils/errors';
import { formatCacheSize, THUMBNAIL_LIMIT_GB, thumbnailLimitValid } from '../../utils/thumbnailCache';
import { useWorkspaceText } from '../../utils/workspaceText';

type CacheStatus = components['schemas']['ThumbnailCacheStatus'];
/** An empty field has no value, so it is invalid rather than 0. */
const parseLimit = (raw: string) => raw.trim() ? Number(raw) : NaN;
const sameLimit = (a: number, b: number) => a === b || (Number.isNaN(a) && Number.isNaN(b));

export default function ThumbnailCacheSettings({ limit, onChange, refreshKey, disabled }: {
  limit: number; onChange: (value: number) => void; refreshKey: number; disabled: boolean;
}) {
  const text = useWorkspaceText();
  const [status, setStatus] = React.useState<CacheStatus | null>(null);
  const [action, setAction] = React.useState<'refresh' | 'clear' | null>(null);
  const busy = action !== null;
  const [error, setError] = React.useState('');
  const [cleared, setCleared] = React.useState<number | null>(null);
  const controller = React.useRef<AbortController | null>(null);
  const refresh = React.useCallback(async (clear = false) => {
    controller.current?.abort();
    const request = new AbortController(); controller.current = request;
    setAction(clear ? 'clear' : 'refresh'); setError(''); setCleared(null);
    try {
      const result = clear
        ? await apiClient.delete<CacheStatus>('/cache/thumbnails', { signal: request.signal, silent: true })
        : await apiClient.get<CacheStatus>('/cache/thumbnails', { signal: request.signal, silent: true });
      if (!request.signal.aborted) { setStatus(result); if (clear) setCleared(result.cleared_bytes); }
    } catch (failure) { if (!request.signal.aborted) setError(formatApiError(failure)); }
    finally { if (!request.signal.aborted) setAction(null); }
  }, []);
  React.useEffect(() => { void refresh(); return () => controller.current?.abort(); }, [refresh, refreshKey]);
  // Keep what is being typed ("0.", an empty field) while it still means the current limit.
  const [draft, setDraft] = React.useState(() => String(limit));
  React.useEffect(() => { setDraft(current => sameLimit(parseLimit(current), limit) ? current : String(limit)); }, [limit]);
  const invalid = !thumbnailLimitValid(limit);
  return <section id="preferences-thumbnails" data-settings-section tabIndex={-1} className="settings-section">
    <div className="settings-section-heading"><h2>{text('缩略图缓存', 'Thumbnail cache')}</h2></div>
    <div className="settings-field"><span className="settings-field-label">{text('当前占用', 'Storage used')}</span><div className="settings-field-control">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-sm tabular-nums" aria-live="polite">{status ? <><strong>{formatCacheSize(status.used_bytes)}</strong><span className="settings-note"> / {formatCacheSize(status.max_bytes)} · {text(`${status.file_count} 张`, `${status.file_count} images`)}</span></> : busy ? text('正在统计…', 'Calculating…') : '—'}</span>
        <button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('刷新缓存占用', 'Refresh cache usage')} disabled={disabled || busy} onClick={() => void refresh()}><RefreshCw size={14} className={action === 'refresh' ? 'animate-spin' : ''}/></button>
        <button type="button" className="ui-btn ui-btn-sm" disabled={disabled || busy || !status?.file_count} onClick={() => void refresh(true)}>{action === 'clear' ? <Loader2 size={14} className="animate-spin"/> : <Trash2 size={14}/>} {action === 'clear' ? text('清理中…', 'Clearing…') : text('清理缓存', 'Clear cache')}</button>
      </div>
      <p className="settings-note">{text('只清理界面缩略图，浏览时会重新生成。', 'Clears display thumbnails only. They are recreated when browsing.')}</p>
      {cleared !== null && <p className="settings-note" role="status">{text(`已清理 ${formatCacheSize(cleared)}`, `Cleared ${formatCacheSize(cleared)}`)}</p>}
      {!!status?.failed_files && <p role="alert" className="settings-note">{text(`${status.failed_files} 个缓存文件无法删除，请检查目录权限。`, `${status.failed_files} cache files could not be deleted. Check directory permissions.`)}</p>}
      {error && <p role="alert" className="settings-alert">{error}</p>}
    </div></div>
    <div className="settings-field"><label htmlFor="preferences-thumbnail-limit">{text('最大占用', 'Maximum size')}</label><div className="settings-field-control">
      <div className="flex items-center gap-2"><input id="preferences-thumbnail-limit" type="number" inputMode="decimal" className="settings-input" style={{maxWidth:160}} min={THUMBNAIL_LIMIT_GB.min} max={THUMBNAIL_LIMIT_GB.max} step={0.1} value={draft} disabled={disabled} aria-invalid={invalid} aria-describedby="thumbnail-limit-help" onChange={event => { setDraft(event.target.value); onChange(parseLimit(event.target.value)); }}/><span className="settings-note">GB</span></div>
      <p id="thumbnail-limit-help" className="settings-note">{invalid ? text(`请输入 ${THUMBNAIL_LIMIT_GB.min}–${THUMBNAIL_LIMIT_GB.max} GB 之间的数值。`, `Enter a value from ${THUMBNAIL_LIMIT_GB.min} to ${THUMBNAIL_LIMIT_GB.max} GB.`) : text('超出上限时自动清理较久未使用的缩略图。', 'Older, unused thumbnails are removed when the limit is reached.')}</p>
    </div></div>
  </section>;
}
