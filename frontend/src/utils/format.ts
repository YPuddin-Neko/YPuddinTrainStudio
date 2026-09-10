/** 字节/容量/速率/时长统一格式化（数字等宽展示） */

export function formatBytesMB(mb: number | null | undefined): string {
  if (mb == null) return '--';
  const gb = mb / 1024;
  if (gb >= 1024) return `${(gb / 1024).toFixed(1)} TB`;
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  return `${Math.round(mb)} MB`;
}

export function formatBytesGB(gb: number | null | undefined): string {
  if (gb == null) return '--';
  if (gb >= 1024) return `${(gb / 1024).toFixed(1)} TB`;
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  return `${Math.round(gb * 1024)} MB`;
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null) return '--';
  if (bytes >= 1e12) return `${(bytes / 1e12).toFixed(1)} TB`;
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} KB`;
  return `${bytes} B`;
}

export function formatParams(n: number | null | undefined): string {
  if (n == null) return '--';
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} K`;
  return String(n);
}

/** 秒 → "1h 23m" / "12m 30s" / "45s" */
export function formatEta(seconds: number | null | undefined): string {
  if (seconds == null) return '--';
  const s = Math.max(0, Math.round(seconds));
  if (s >= 3600) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  if (s >= 60) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${s}s`;
}

/** Unix 秒/ISO 字符串 → 本地化时间 */
export function formatTime(t: number | string | null | undefined): string {
  if (t == null) return '--';
  const d = typeof t === 'number' ? new Date(t * 1000) : new Date(t);
  return d.toLocaleString();
}

export function formatPercent(p: number | null | undefined): string {
  if (p == null) return '--';
  return `${Math.round(p)}%`;
}
