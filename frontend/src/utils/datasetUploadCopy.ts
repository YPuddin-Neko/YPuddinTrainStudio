import { ApiError } from '../api/types';
import { formatDatasetImportError } from './datasetImportErrors';
import type { UploadEntry } from './datasetUploads';

type Text = (zh: string, en: string) => string;

/** Why an upload stopped, worded for the import panel and the task center. */
export function uploadErrorMessage(entry: UploadEntry, text: Text): string {
  const error = entry.error;
  if (!error) return '';
  switch (error.problem) {
    case 'send':
      return text('训练服务仍可连接，但文件未能发送。请确认文件没有被移动或删除，重新选择文件夹后重试。', 'The training service is reachable, but the files could not be sent. Check that they have not moved or been deleted, then select the folder again.');
    case 'connection':
      return text('上传连接已中断，所选文件已保留。请检查训练服务与网络连接后重试。', 'The upload connection was interrupted. Your selection is retained. Check the training service and network, then retry.');
    case 'expired':
      return text('上传已失效：超过 1 小时没有继续，或训练服务已重启。重新选择文件后会从头上传。', 'The upload is no longer available: it was idle for over an hour, or the training service restarted. Choose the files again to upload them from the start.');
    case 'mismatch': {
      const names = error.files || [];
      if (!names.length) return text('所选文件与已上传的部分内容不同。请选择原来的文件夹或文件。', 'The chosen files differ from what was already uploaded. Choose the original folder or files.');
      const shown = names.slice(0, 3).join(text('、', ', '));
      const rest = names.length > 3 ? text(` 等 ${names.length} 个文件`, ` and ${names.length - 3} more`) : '';
      return text(`所选文件与未完成的上传不一致：${shown}${rest}。请选择原来的文件夹或文件。`, `The chosen files differ from the unfinished upload: ${shown}${rest}. Choose the original folder or files.`);
    }
    case 'unconfirmed':
      return entry.canRetry
        ? formatDatasetImportError(new ApiError(0, { code: 'upload.result_unconfirmed', message: '' }))
        : text('无法确认导入结果，请先检查数据集列表，再决定是否重新上传。', 'The import result could not be confirmed. Check the dataset list before uploading again.');
    default:
      return formatDatasetImportError(error.error instanceof ApiError ? error.error : new ApiError(0, { code: error.code, message: error.message }));
  }
}

/** Whether choosing the files again continues this upload instead of starting a new one. */
export const canChooseFilesAgain = (entry: UploadEntry) => !entry.remote && !!entry.sessionId && entry.bytesDone < entry.bytesTotal
  && (entry.needsFiles || entry.phase === 'interrupted' || ['send', 'mismatch'].includes(entry.error?.problem || ''));

/** Upload sizes in binary units, as the import progress shows them. */
export function formatUploadBytes(value: number): string {
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GiB`;
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(1)} MiB`;
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KiB`;
  return `${Math.max(0, Math.round(value))} B`;
}
