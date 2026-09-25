import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { X } from 'lucide-react';

export function ApiErrorNotice() {
  const [message, setMessage] = useState('');
  const { t } = useTranslation();
  useEffect(() => {
    const handler = (event: Event) => setMessage((event as CustomEvent<string>).detail);
    window.addEventListener('api.error', handler);
    return () => window.removeEventListener('api.error', handler);
  }, []);
  if (!message) return null;
  return <div role="alert" className="api-error-notice">
    <span className="whitespace-pre-line break-words">{message}</span><button type="button" className="ui-btn ui-btn-quiet ui-btn-sm ui-btn-icon" onClick={() => setMessage('')} aria-label={t('common.close')}><X size={15}/></button>
  </div>;
}
