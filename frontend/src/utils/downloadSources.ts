import type { Settings } from '../api/types';

export const DOWNLOAD_SOURCE_DEFAULTS: NonNullable<Settings['downloads']> = { pypi: 'auto', pytorch: 'auto', fallback: true };
