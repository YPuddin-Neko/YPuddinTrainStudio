import { ApiError } from '../api/types';
import i18n from '../i18n';
import { formatApiError } from './errors';

/** Explain dataset conflicts here while retaining the server's exact file/path evidence. */
export function formatDatasetImportError(error: unknown): string {
  const details = formatApiError(error);
  if (!(error instanceof ApiError)) return details;
  const english = i18n.language?.startsWith('en');
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
