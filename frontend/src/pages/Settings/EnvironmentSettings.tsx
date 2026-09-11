import React from 'react';
import { useSearchParams } from 'react-router-dom';
import { EnvironmentManagerPanel } from '../../components/EnvironmentManagerPanel';
import { useWorkspaceText } from '../../utils/workspaceText';
const Models = React.lazy(() => import('../Models/Models'));

export default function EnvironmentSettings() {
  const text = useWorkspaceText();
  const [params] = useSearchParams();
  return <div data-testid="environment-settings"><React.Suspense fallback={<p role="status" className="settings-note">{text('正在加载…', 'Loading…')}</p>}>
    {params.get('tab') === 'models' ? <Models embedded /> : <EnvironmentManagerPanel />}
  </React.Suspense></div>;
}
