import { Navigate, useLocation } from 'react-router-dom';

/** Keep bookmarked model/project filters while moving their pages into Settings. */
export default function SettingsRedirect({ tab }: { tab?: 'runtime' | 'models' | 'artifacts' | 'credentials' }) {
  const location = useLocation();
  const params = new URLSearchParams(location.search);
  const legacyHash = location.hash.slice(1);
  const selected = tab || (legacyHash === 'models' ? 'models' : params.get('tab')) || 'runtime';
  params.set('tab', selected);
  const hash = ['models', 'environment'].includes(legacyHash) ? '' : location.hash;
  return <Navigate replace state={location.state} to={`/settings/environment?${params.toString()}${hash}`} />;
}
