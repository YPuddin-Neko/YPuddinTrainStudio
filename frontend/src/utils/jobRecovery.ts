import type { Job } from '../api/types';
import { isTtsJob } from '../api/tts';
import { ttsAction } from '../pages/Tts/ttsJobActions';

type Text = (zh: string, en: string) => string;

export function jobRecoveryMessage(job: Job, text: Text): string {
  if (isTtsJob(job)) {
    const retry = ttsAction(job, 'retry', text);
    if (!retry.allowed) return retry.reason;
    return job.type === 'tts_train'
      ? text('可从头重新训练，已有产物保留。', 'Start training again from the beginning. Existing outputs are kept.')
      : text('可重新生成试听，已有产物保留。', 'Generate the preview again. Existing outputs are kept.');
  }
  const recovery = job.recovery;
  if (!recovery) return text('恢复信息暂不可用，请刷新任务。', 'Recovery information is unavailable. Refresh the job.');
  if (recovery.worker_alive) return text('原任务进程尚未退出，请先取消或等待退出。', 'The original worker is still running. Cancel it or wait for it to exit.');
  if (recovery.can_resume) return recovery.step != null
    ? text(`可从第 ${recovery.step} 步的恢复点继续训练。`, `Continue training from the resume point at step ${recovery.step}.`)
    : text('已有完整恢复点，可以继续训练。', 'A complete resume point is available to continue training.');
  if (job.type === 'train') return text('没有可用的完整恢复点，可重新训练。', 'No complete resume point is available. Start a new training run.');
  return job.type === 'cache' ? text('可重新准备缓存。', 'Start cache preparation again.')
    : text('可重新生成。', 'Start generation again.');
}
