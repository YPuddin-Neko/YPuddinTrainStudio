import { apiUrl } from '../api/client';

/** Preview and file URLs arrive with the /api prefix; the client adds the backend origin. */
export function sampleSource(url: string): string {
  return url.startsWith('/api/') ? apiUrl(url.slice(4)) : url;
}
