import React from 'react';
import { useLocation, useSearchParams } from 'react-router-dom';
import { EnvironmentManagerPanel } from '../../components/EnvironmentManagerPanel';
import { useWorkspaceText } from '../../utils/workspaceText';
import { LoadingNote } from '../../components/Loading';
const Models = React.lazy(() => import('../Models/Models'));
const AccessKeys = React.lazy(() => import('./AccessKeys'));
const TaggingSettings = React.lazy(() => import('./TaggingSettings'));

export default function EnvironmentSettings() {
  const text = useWorkspaceText();
  const [params, setParams] = useSearchParams();
  const location = useLocation();
  const environmentView = params.get('environment');
  return <div data-testid="environment-settings"><React.Suspense fallback={<LoadingNote block label={text('正在加载…', 'Loading…')}/>}>
    {params.get('tab') === 'credentials' ? <AccessKeys /> : params.get('tab') === 'models' ? <Models embedded /> : params.get('tab') === 'tagging' ? <TaggingSettings /> : <EnvironmentManagerPanel focusPackage={params.get('package') || undefined} initialView={environmentView === 'image' || environmentView === 'speech' ? environmentView : undefined} onViewChange={view => {
      const next = new URLSearchParams(params);
      next.set('environment', view);
      next.delete('package');
      setParams(next, { replace: true, state: location.state });
    }} />}
  </React.Suspense></div>;
}
