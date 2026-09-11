import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

export function ApiErrorNotice() {
  const [message, setMessage] = useState('');
  const { t } = useTranslation();
  useEffect(() => {
    const handler = (event: Event) => setMessage((event as CustomEvent<string>).detail);
    window.addEventListener('api.error', handler);
    return () => window.removeEventListener('api.error', handler);
  }, []);
  if (!message) return null;
  return <div role="alert" className="flex items-start justify-between gap-3 bg-red-50 text-red-700 dark:bg-red-950 dark:text-red-300 border-b border-red-200 px-6 py-3 text-sm">
    <span className="whitespace-pre-line break-words">{message}</span><button onClick={() => setMessage('')} aria-label={t('common.close')}>✕</button>
  </div>;
}
