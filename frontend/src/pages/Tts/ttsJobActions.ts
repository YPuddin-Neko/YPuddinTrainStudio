import { ApiError, type Job } from '../../api/types';
import { isTtsJob, type TtsIssue } from '../../api/tts';
import { formatApiError } from '../../utils/errors';

type Text = (zh: string, en: string) => string;
export type TtsJobAction = NonNullable<Job['allowed_actions']>[number];
export function ttsAction(job: Job, action: TtsJobAction, text: Text) {
  if (!isTtsJob(job)) return { allowed: true, reason: '' };
  const denied = job.action_reasons?.[action];
  const reason = denied ? actionReason(denied, text) : '';
  const allowed = Array.isArray(job.allowed_actions) && !!job.action_reasons && job.allowed_actions.includes(action) && !denied;
  return { allowed, reason: allowed ? '' : reason || text('任务操作状态尚未确认，请刷新后重试。', 'Job actions are not confirmed. Refresh and try again.') };
}
export function ttsJobError(failure: unknown): string {
  const first = formatApiError(failure);
  const issues = failure instanceof ApiError && Array.isArray(failure.details?.issues) ? failure.details.issues as TtsIssue[] : [];
  return [...new Set([first, ...issues.map(issue => issue.message)])].join('\n');
}

const actionMessages: Record<string, Record<string, string>> = {
  'job.bad_state': {
    '语音任务不支持暂停、恢复或手动保存。': 'Speech jobs do not support pause, resume or manual saving.',
    '当前任务状态不能取消。': 'This job cannot be cancelled in its current state.',
    '任务尚未归档。': 'This job has not been archived.',
    '任务进程尚未退出，请等待退出后再操作。': 'Wait for the job process to exit before continuing.',
    '只能归档已结束的任务。': 'Only finished jobs can be archived.',
    '只能删除已结束的任务。': 'Only finished jobs can be deleted.',
    '当前任务状态不能更换显卡。': 'The GPU cannot be changed in this job state.',
    '只能调整排队任务的优先级。': 'Only queued jobs can change priority.',
    '只有尚未强制开始的排队任务可以强制开始。': 'Only queued jobs that have not been forced can be started immediately.',
    '只能重试已结束的任务。': 'Only finished jobs can be retried.',
    '只能重试已结束或中断的任务。': 'Only finished or interrupted jobs can be retried.',
  },
  'job.deleting': { '这个任务正在删除。': 'This job is being deleted.', '来源训练任务正在删除。': 'The source training job is being deleted.' },
  'job.archived': { '请先恢复已归档的任务。': 'Restore the archived job first.', '请先恢复来源训练任务，再重试试听。': 'Restore the source training job before retrying the preview.' },
  'job.archive_required': { '请先归档任务，再永久删除。': 'Archive the job before deleting it permanently.' },
  'job.files_in_use': { '来源训练仍被活动试听任务使用，请等试听结束后再删除。': 'Active preview jobs still use this training job. Wait for them to finish before deleting it.' },
  'project.deleting': { '任务所属项目正在删除。': 'The project that owns this job is being deleted.' },
  'project.not_found': { '任务所属项目不存在。': 'The project that owns this job no longer exists.' },
  'project.type_mismatch': { '语音任务的项目类型不匹配。': 'The project type does not match this speech job.', '任务所属版本与项目不匹配。': 'The job version does not belong to its project.' },
  'project.archived': { '请先恢复项目，再创建或调整语音任务。': 'Restore the project before creating or changing speech jobs.' },
  'version.not_found': { '任务所属版本不存在。': 'The version that owns this job no longer exists.' },
  'version.busy': { '任务所属版本已归档或尚未就绪。': 'The version that owns this job is archived or is not ready.' },
  'tts.source_unavailable': {
    '任务的语音配置快照不可用。': 'The speech configuration snapshot is unavailable.',
    '试听任务缺少来源训练身份。': 'The preview job has no source training job identity.',
    '试听任务缺少冻结的检查点信息。': 'The preview job has no saved checkpoint information.',
    '试听记录缺少来源检查点。': 'The preview record has no source checkpoint.',
    '检查点配置或权重已不可读取。': 'The checkpoint configuration or weights can no longer be read.',
    '冻结数据清单缺少有效的绝对路径。': 'The saved dataset manifest has no valid absolute path.',
    '冻结数据清单不存在或不是有效文件。': 'The saved dataset manifest is missing or is not a valid file.',
    '参考音频缺少有效的绝对路径。': 'The reference audio has no valid absolute path.',
    '参考音频不存在或不是有效文件。': 'The reference audio is missing or is not a valid file.',
  },
  'tts.job_type_mismatch': { '试听来源不是语音训练任务。': 'The preview source is not a speech training job.' },
  'tts.path_denied': { '检查点不在来源任务的目录中。': 'The checkpoint is outside the source job directory.' },
  'tts.checkpoint_stale': {
    '检查点内容已变化，请重新选择。': 'The checkpoint contents changed. Select it again.',
    '检查点文件身份已变化，请重新检查。': 'The checkpoint file identity changed. Check it again.',
  },
};
function actionReason(issue: TtsIssue, text: Text) {
  const translated = actionMessages[issue.code]?.[issue.message];
  return translated ? text(issue.message, translated) : issue.message;
}
