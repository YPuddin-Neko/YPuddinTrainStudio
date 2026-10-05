type Config = Record<string, any>;

export const INTERVAL_FIELDS: Record<string, { fallback: number; enabled: boolean }> = {
  'checkpoint.save_every_steps': { fallback: 100, enabled: false },
  'checkpoint.save_every_epochs': { fallback: 1, enabled: true },
  'checkpoint.save_state_every_steps': { fallback: 100, enabled: true },
  'checkpoint.save_state_every_epochs': { fallback: 1, enabled: false },
  'sampling.every_steps': { fallback: 100, enabled: false },
  'sampling.every_epochs': { fallback: 1, enabled: true },
  'validation.every_steps': { fallback: 100, enabled: false },
  'validation.every_epochs': { fallback: 1, enabled: true },
};

/** Migrate a partial legacy config before merging defaults or another draft. */
export function normalizeLegacyIntervals(source: Config): Config {
  let result = source;
  for (const [path, { fallback }] of Object.entries(INTERVAL_FIELDS)) {
    const [group, key] = path.split('.');
    const section = result[group];
    if (!section || typeof section !== 'object' || Array.isArray(section) || !Object.prototype.hasOwnProperty.call(section, key)) continue;
    const enabledKey = `${key}_enabled`;
    const hasEnabled = Object.prototype.hasOwnProperty.call(section, enabledKey);
    const enabled = hasEnabled ? section[enabledKey] : section[key] !== null;
    const interval = section[key] === null && enabled === false ? fallback : section[key];
    if (hasEnabled && interval === section[key]) continue;
    if (result === source) result = { ...source };
    result[group] = { ...section, [key]: interval, [enabledKey]: enabled };
  }
  return result;
}

export function intervalFieldPath(path: string): string {
  const interval = path.replace(/_enabled$/, '');
  return INTERVAL_FIELDS[interval] ? interval : path;
}
