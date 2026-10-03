import { ApiError } from '../api/types';
import i18n from '../i18n';
import { formatApiError } from './errors';

/** Explain dataset conflicts here while retaining the server's exact file/path evidence. */
export function formatDatasetImportError(error: unknown): string {
  const details = formatApiError(error);
  if (!(error instanceof ApiError)) return details;
  const english = i18n.language?.startsWith('en');
  if (error.code === 'upload.result_unconfirmed') return english
    ? 'The import result could not be confirmed. Your selection is retained. Check the dataset list before retrying.'
    : '未能确认导入结果，所选文件已保留。请检查数据集列表后重试。';
  if (error.code === 'upload.update_required') return english
    ? 'This training service does not support chunked uploads. Update it, then retry.'
    : '当前训练服务不支持分片上传，请更新训练器后重试。';
  // The service words these two in English; an upload waiting on them is kept for another attempt.
  if (error.code === 'version.indexing') return english
    ? 'A dataset of this version is being indexed. Retry the import when indexing finishes.'
    : '当前版本的数据集正在建立索引，完成后可以重试导入。';
  if (error.code === 'version.jobs_busy') return english
    ? 'A queued or running job uses this version\'s data. Retry the import after it finishes.'
    : '当前版本的数据正在被排队或运行中的任务使用，任务结束后可以重试导入。';
  if (error.status === 413 && error.code === 'http_413') return english
    ? "The upload exceeds the connection's request size limit. Increase the proxy upload limit, or upload to the server and use Import from server computer."
    : '上传请求超过连接入口的大小限制。请提高代理上传限制，或先把文件上传到服务器，再从服务端电脑导入。';
  const summary = error.code === 'upload.conflict'
    ? english
      ? 'The import conflicts with files already in this version. Check the path below, rename the conflicting file or folder, then retry.'
      : '导入内容与当前版本的已有文件冲突。请按下方路径检查，重命名冲突的文件或文件夹后重试。'
    : error.code === 'dataset.overlap'
      ? english
        ? 'The import overlaps an existing training, regularization or validation source. Check the details below, adjust the sources or choose the registered child folder, then retry.'
        : '导入目录与已有训练、正则或验证数据源重叠。请按下方详情调整数据源，或选择已登记的子文件夹后重试。'
      : '';
  return summary ? `${summary}\n${details}` : details;
}
