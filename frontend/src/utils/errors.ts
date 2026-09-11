/** Keep the server's field-level validation details in every error surface. */
export function formatApiError(error: unknown): string {
  if (!error || typeof error !== 'object') return String(error);
  const payload = error as { message?: unknown; details?: { errors?: unknown } };
  const message = typeof payload.message === 'string' ? payload.message : 'API request failed';
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
