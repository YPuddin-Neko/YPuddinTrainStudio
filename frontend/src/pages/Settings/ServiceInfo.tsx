import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { apiClient } from '../../api/client';
import type { SystemInfo } from '../../api/types';
import { formatApiError } from '../../utils/errors';

export function ServiceInfo() {
  const { t } = useTranslation();
  const [info, setInfo] = useState<SystemInfo | null>(null);
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    apiClient.get<SystemInfo>('/system/info', { signal: controller.signal, silent: true })
      .then(value => { if (!controller.signal.aborted) setInfo(value); })
      .catch(reason => { if (!controller.signal.aborted) setError(formatApiError(reason)); });
    return () => controller.abort();
  }, [attempt]);

  return <div className="settings-field" data-testid="settings-service-version">
    <span className="settings-field-label">{t('hardware.serverVersion')}</span>
    <div className="settings-field-control py-1.5">
      {error ? <div role="alert" className="text-red-600 dark:text-red-400 break-words">{error}<button type="button" className="ml-3 underline" onClick={() => { setError(''); setAttempt(value => value + 1); }}>{t('common.retry')}</button></div>
        : info ? <span className="font-mono break-all">{info.ypuddin || t('hardware.unavailable')}</span>
          : <span role="status" className="settings-note">{t('common.loading')}</span>}
    </div>
  </div>;
}
