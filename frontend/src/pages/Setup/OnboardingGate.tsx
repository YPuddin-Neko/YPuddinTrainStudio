import React from 'react';
import { Navigate } from 'react-router-dom';
import { Loader2 } from 'lucide-react';
import { apiClient, READ_TIMEOUT_MS } from '../../api/client';
import type { Settings } from '../../api/types';
import { useWorkspaceText } from '../../utils/workspaceText';

export default function OnboardingGate({ children }: { children: React.ReactNode }) {
  const text = useWorkspaceText();
  const [completed, setCompleted] = React.useState<boolean | null>(null);
  const [failed, setFailed] = React.useState(false);
  const [attempt, setAttempt] = React.useState(0);
  React.useEffect(() => {
    const controller = new AbortController();
    setFailed(false);
    void apiClient.get<Settings>('/settings', { signal: controller.signal, silent: true, timeout: READ_TIMEOUT_MS })
      .then(settings => { if (!controller.signal.aborted) setCompleted(settings.ui.onboarding_completed !== false); })
      .catch(() => { if (!controller.signal.aborted) setFailed(true); });
    return () => controller.abort();
  }, [attempt]);
  if (completed === false) return <Navigate to="/setup" replace/>;
  if (completed === true) return children;
  return <div className="setup-loading" role="status">{failed
    ? <><span>{text('无法读取设置', 'Could not load settings')}</span><button className="ui-btn" onClick={() => setAttempt(value => value + 1)}>{text('重试', 'Retry')}</button></>
    : <Loader2 size={22} className="animate-spin" aria-label={text('正在读取设置', 'Loading settings')}/>}</div>;
}
