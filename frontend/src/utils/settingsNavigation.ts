/** Keep model and project filters when moving between settings sections. */
export function settingsDestination(search: string, destination: string) {
  const target = new URL(destination, 'http://settings.local');
  const params = new URLSearchParams(search);
  if (target.pathname === '/settings/preferences') params.delete('tab');
  else if (target.pathname === '/settings/environment') params.delete('section');
  target.searchParams.forEach((value, key) => params.set(key, value));
  const query = params.toString();
  return `${target.pathname}${query ? `?${query}` : ''}${target.hash}`;
}
