import React from 'react';
import { Navigate } from 'react-router-dom';
import { Loader2 } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { Settings } from '../../api/types';

export default function OnboardingGate({ children }: { children: React.ReactNode }) {
  const [completed, setCompleted] = React.useState<boolean | null>(null);
  const [failed, setFailed] = React.useState(false);
  const [attempt, setAttempt] = React.useState(0);
  React.useEffect(() => {
    const controller = new AbortController();
    setFailed(false);
    void apiClient.get<Settings>('/settings', { signal: controller.signal, silent: true })
      .then(settings => { if (!controller.signal.aborted) setCompleted(settings.ui.onboarding_completed !== false); })
      .catch(() => { if (!controller.signal.aborted) setFailed(true); });
    return () => controller.abort();
  }, [attempt]);
  if (completed === false) return <Navigate to="/setup" replace/>;
  if (completed === true) return children;
  return <div className="setup-loading" role="status">{failed
    ? <><span>无法读取设置</span><button className="ui-btn" onClick={() => setAttempt(value => value + 1)}>重试</button></>
    : <Loader2 size={22} className="animate-spin" aria-label="正在读取设置"/>}</div>;
}
