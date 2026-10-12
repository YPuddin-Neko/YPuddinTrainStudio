import type { Location } from 'react-router-dom';

const isSettingsPath = (pathname: string) => pathname === '/settings' || pathname.startsWith('/settings/');

function settingsBackground(location: Location): Location | null {
  if (!isSettingsPath(location.pathname)) return null;
  const background = (location.state as { backgroundLocation?: Location } | null)?.backgroundLocation;
  return background && typeof background.pathname === 'string' && background.pathname.startsWith('/')
    && !isSettingsPath(background.pathname) && typeof background.search === 'string' && typeof background.hash === 'string'
    ? background : null;
}

export function keepsSettingsWorkspace(currentLocation: Location, nextLocation: Location): boolean {
  const currentBackground = settingsBackground(currentLocation);
  const nextBackground = settingsBackground(nextLocation);
  if (!currentBackground && !nextBackground) return false;
  const current = currentBackground ?? currentLocation;
  const next = nextBackground ?? nextLocation;
  return current.pathname === next.pathname && current.search === next.search && current.hash === next.hash;
}
