import i18n from '../i18n';

/** Keep server validation details, while giving connection failures an actionable message. */
export function formatApiError(error: unknown): string {
  if (!error || typeof error !== 'object') return String(error);
  const payload = error as { message?: unknown; details?: { errors?: unknown } };
  const message = typeof payload.message === 'string' ? payload.message : 'API request failed';
  if (/^(Failed to fetch|Load failed|NetworkError when attempting to fetch resource\.?)$/i.test(message)) {
    return i18n.language?.startsWith('en') ? 'Cannot connect to the training service. Check that it is running, then retry.' : '暂时无法连接训练服务，请确认服务已启动后重试。';
  }
  const errors = payload.details?.errors;
  if (!Array.isArray(errors)) return message;
  const details = errors.flatMap((item: unknown) => {
    if (!item || typeof item !== 'object') return [];
    const { loc, msg } = item as { loc?: unknown; msg?: unknown };
    if (typeof msg !== 'string') return [];
    const path = Array.isArray(loc) ? loc.join('.') : typeof loc === 'string' ? loc : '';
    return [path ? `${path}: ${msg}` : msg];
  });
  return [...new Set([message, ...details])].filter(Boolean).join('\n');
}
