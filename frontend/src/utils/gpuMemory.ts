type MemoryReading = {
  mem_used_mb?: number | null;
  mem_total_mb?: number | null;
  mem_free_mb?: number | null;
  mem_reserved_mb?: number | null;
};
type Text = (zh: string, en: string) => string;
const known = (value: number | null | undefined): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0;

export function formatGpuMemory(mib: number | null | undefined): string {
  if (!known(mib)) return '—';
  if (mib < 1024) return `${Math.floor(mib)} MiB`;
  if (mib < 1024 * 1024) return `${(mib / 1024).toFixed(1)} GiB`;
  return `${(mib / (1024 * 1024)).toFixed(1)} TiB`;
}

export function gpuMemoryUsage(memory: MemoryReading | undefined): string {
  return `${formatGpuMemory(memory?.mem_used_mb)} / ${formatGpuMemory(memory?.mem_total_mb)}`;
}

export function gpuMemorySummary(memory: MemoryReading | undefined, text: Text): string {
  const parts = [];
  if (known(memory?.mem_used_mb) || known(memory?.mem_total_mb)) parts.push(`${text('已用', 'Used')} ${gpuMemoryUsage(memory)}`);
  if (known(memory?.mem_free_mb)) parts.push(`${text('可用', 'Free')} ${formatGpuMemory(memory.mem_free_mb)}`);
  return parts.join(' · ');
}

export function gpuMemoryAvailability(memory: MemoryReading | undefined, text: Text): string {
  const parts = [];
  if (known(memory?.mem_free_mb)) parts.push(`${text('可用', 'Free')} ${formatGpuMemory(memory.mem_free_mb)}`);
  if (known(memory?.mem_reserved_mb) && memory.mem_reserved_mb > 0) {
    parts.push(`${text('驱动预留', 'Driver reserved')} ${formatGpuMemory(memory.mem_reserved_mb)}`);
  }
  return parts.join(' · ');
}

export function gpuMemoryDetails(memory: MemoryReading | undefined, text: Text): string | undefined {
  const usage = known(memory?.mem_used_mb) || known(memory?.mem_total_mb) ? `${text('已用', 'Used')} ${gpuMemoryUsage(memory)}` : '';
  return [usage, gpuMemoryAvailability(memory, text)].filter(Boolean).join('\n') || undefined;
}
