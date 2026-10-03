import i18n from '../i18n';
import { describeValidation, type ValidationIssue } from './validationMessages';

/** Keep server validation details, while giving connection failures an actionable message. */
export function formatApiError(error: unknown): string {
  if (!error || typeof error !== 'object') return String(error);
  if ((error as { name?: unknown }).name === 'TimeoutError') {
    return i18n.language?.startsWith('en') ? 'The training service did not answer in time. Retry.' : '训练服务长时间没有响应，请重试。';
  }
  const payload = error as { message?: unknown; details?: { errors?: unknown } };
  const message = typeof payload.message === 'string' ? payload.message : 'API request failed';
  if (/^(Failed to fetch|Load failed|NetworkError when attempting to fetch resource\.?)$/i.test(message)) {
    return i18n.language?.startsWith('en') ? 'Cannot connect to the training service. Check that it is running, then retry.' : '暂时无法连接训练服务，请确认服务已启动后重试。';
  }
  const errors = payload.details?.errors;
  if (!Array.isArray(errors)) return message;
  const english = i18n.language?.startsWith('en');
  const details = errors.flatMap((item: unknown) => {
    if (!item || typeof item !== 'object') return [];
    const { loc, msg } = item as ValidationIssue;
    if (typeof msg !== 'string') return [];
    // Where FastAPI found the value (body, query) is not part of the field's name.
    const parts = Array.isArray(loc) ? loc.filter((part, index) => index > 0 || !['body', 'query', 'path'].includes(String(part))) : [];
    const path = parts.length ? parts.join('.') : typeof loc === 'string' ? loc : '';
    const text = describeValidation(item as ValidationIssue, { english }) ?? msg;
    return [path ? `${path}: ${text}` : text];
  });
  return [...new Set([message, ...details])].filter(Boolean).join('\n');
}
