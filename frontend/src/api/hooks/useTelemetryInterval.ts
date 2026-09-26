import React from 'react';
import { apiClient } from '../client';
import type { Settings } from '../types';

const DEFAULT_SECONDS = 2.5;
const clamp = (value: number) => Math.min(60, Math.max(1, value));

/** Seconds between hardware readings, from the interface settings; follows a saved change at once. */
export function useTelemetryInterval() {
  const [seconds, setSeconds] = React.useState(DEFAULT_SECONDS);
  React.useEffect(() => {
    let active = true;
    apiClient.get<Settings>('/settings', { silent: true })
      .then(settings => { if (active && typeof settings?.ui?.telemetry_interval === 'number') setSeconds(clamp(settings.ui.telemetry_interval)); })
      .catch(() => {});
    const changed = (event: Event) => {
      const value = (event as CustomEvent<Settings>).detail?.ui?.telemetry_interval;
      if (typeof value === 'number') setSeconds(clamp(value));
    };
    window.addEventListener('studio.settings.changed', changed);
    return () => { active = false; window.removeEventListener('studio.settings.changed', changed); };
  }, []);
  return seconds;
}
