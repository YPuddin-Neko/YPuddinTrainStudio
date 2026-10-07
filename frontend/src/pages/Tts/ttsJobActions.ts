import { ApiError, type Job } from '../../api/types';
import { isTtsJob, type TtsIssue } from '../../api/tts';
import { formatApiError } from '../../utils/errors';

type Text = (zh: string, en: string) => string;
export type TtsJobAction = NonNullable<Job['allowed_actions']>[number];
export function ttsAction(job: Job, action: TtsJobAction, text: Text) {
  if (!isTtsJob(job)) return { allowed: true, reason: '' };
  const denied = job.action_reasons?.[action];
  const reason = denied?.message;
  const allowed = Array.isArray(job.allowed_actions) && !!job.action_reasons && job.allowed_actions.includes(action) && !denied;
  return { allowed, reason: allowed ? '' : reason || text('任务操作状态尚未确认，请刷新后重试。', 'Job actions are not confirmed. Refresh and try again.') };
}
export function ttsJobError(failure: unknown): string {
  const first = formatApiError(failure);
  const issues = failure instanceof ApiError && Array.isArray(failure.details?.issues) ? failure.details.issues as TtsIssue[] : [];
  return [...new Set([first, ...issues.map(issue => issue.message)])].join('\n');
}
