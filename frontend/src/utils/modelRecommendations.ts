import type { ImageRecommendation } from '../api/hooks/useImageModelResources';

export function modelRecommendationName(entry: Pick<ImageRecommendation, 'name' | 'dtype'>): string {
  const precision = entry.dtype.toUpperCase();
  return new RegExp(`\\b${precision}\\b`, 'i').test(entry.name)
    ? entry.name
    : `${entry.name} · ${precision}`;
}
