/** The thumbnail cache limit in GB; the service counts 1 GB as 1024³ bytes. */
export const THUMBNAIL_LIMIT_GB = { min: 0.1, max: 1024, fallback: 1 } as const;

const GB = 1024 ** 3;

export function thumbnailLimitValid(value: number): boolean {
  return Number.isFinite(value) && value >= THUMBNAIL_LIMIT_GB.min && value <= THUMBNAIL_LIMIT_GB.max;
}

/** Sizes in GB; a nearly empty cache keeps a third decimal instead of reading as 0. */
export function formatCacheSize(bytes: number): string {
  const size = bytes / GB;
  if (!(size > 0)) return '0 GB';
  if (size < 0.001) return '< 0.001 GB';
  return `${Number(size.toFixed(size < 0.01 ? 3 : 2))} GB`;
}
