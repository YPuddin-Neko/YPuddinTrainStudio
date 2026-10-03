import type { ReactNode } from 'react';
import type { DatasetImportOperation, DatasetImportPhase } from '../../utils/useDatasetImportProgress';
import { useWorkspaceText } from '../../utils/workspaceText';
import DatasetOperationProgress from './DatasetOperationProgress';

const phaseLabels: Record<DatasetImportPhase, [string, string]> = {
  receiving: ['上传文件', 'Uploading files'],
  extracting: ['解压文件', 'Extracting files'],
  validating: ['校验文件与标签', 'Validating files and captions'],
  copying: ['同步到当前版本', 'Copying into this version'],
  registering: ['登记图片目录', 'Registering image folders'],
  completed: ['正在返回导入结果', 'Waiting for the import result'],
  failed: ['正在返回导入结果', 'Waiting for the import result'],
};

const waitingLabels: Record<string, [string, string]> = {
  'version.indexing': ['等待数据集索引完成后导入', 'Waiting for dataset indexing to finish'],
  'service.restarting': ['等待训练服务重启后导入', 'Waiting for the training service to restart'],
};

function bytes(value: number): string {
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GiB`;
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(1)} MiB`;
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KiB`;
  return `${Math.max(0, Math.round(value))} B`;
}

export default function DatasetImportProgress({ operation, actions }: { operation: DatasetImportOperation; actions?: ReactNode }) {
  const text = useWorkspaceText();
  const { snapshot, state, unavailable, upload } = operation;
  if (state === 'active' && (operation.preparing || operation.waiting)) {
    const phaseText = operation.preparing ? text('正在准备上传', 'Preparing the upload')
      : text(...(waitingLabels[operation.waiting!] || ['等待当前版本的其他操作完成后导入', 'Waiting for another operation on this version to finish']));
    return <DatasetOperationProgress label={text('导入进度', 'Import progress')} phaseText={phaseText} state={state} done={null} total={null} elapsed={operation.elapsed} actions={actions}/>;
  }
  if (state === 'active' && upload && !upload.complete) {
    const remaining = upload.bytesPerSecond != null && upload.bytesPerSecond > 0
      ? (upload.bytesTotal - upload.bytesDone) / upload.bytesPerSecond : null;
    return <DatasetOperationProgress label={text('导入进度', 'Import progress')}
      phaseText={upload.bytesDone >= upload.bytesTotal ? text('等待训练服务确认上传', 'Waiting for upload confirmation') : text('上传文件', 'Uploading files')}
      state={state} done={upload.bytesDone} total={upload.bytesTotal}
      detail={`${bytes(upload.bytesDone)} / ${bytes(upload.bytesTotal)} · ${text(`${upload.filesDone} / ${upload.filesTotal} 个文件`, `${upload.filesDone} / ${upload.filesTotal} files`)}`}
      speed={upload.bytesPerSecond != null ? `${(upload.bytesPerSecond / 1024 ** 2).toFixed(2)} MiB/s` : null}
      elapsed={operation.elapsed} remaining={remaining} actions={actions}/>;
  }
  const waitingForProcessing = upload?.complete && (!snapshot || snapshot.phase === 'receiving' || unavailable);
  const terminalSnapshot = snapshot?.phase === 'completed' || snapshot?.phase === 'failed';
  const phaseText = state === 'completed' ? text('导入与登记已完成', 'Import and registration completed')
    : state === 'failed' && operation.unconfirmed ? text('暂未确认导入结果，来源文件已保留', 'Import result not confirmed; source files are retained')
    : state === 'failed' ? text('导入未完成，来源文件已保留', 'Import did not complete; source files are retained')
      : waitingForProcessing ? text('文件已上传，正在等待导入结果', 'Files uploaded; waiting for the import result')
      : unavailable ? text('暂时无法读取进度，正在等待导入结果', 'Progress is unavailable; waiting for the import result')
        : snapshot ? text(...phaseLabels[snapshot.phase])
          : operation.mode === 'upload' ? text('等待训练服务接收文件', 'Waiting for the training service to receive files') : text('读取来源目录', 'Reading the source folder');
  const measured = snapshot && !terminalSnapshot && !unavailable && !waitingForProcessing;
  const hasBytes = measured && (snapshot.phase === 'receiving' || snapshot.bytes_done > 0 || (snapshot.bytes_total ?? 0) > 0);
  const done = measured ? hasBytes ? snapshot.bytes_done : snapshot.files_done : null;
  const total = measured ? hasBytes ? snapshot.bytes_total : snapshot.files_total : null;
  const counts: string[] = [];
  if (measured) {
    if (hasBytes) counts.push(snapshot.bytes_total == null ? bytes(snapshot.bytes_done) : `${bytes(snapshot.bytes_done)} / ${bytes(snapshot.bytes_total)}`);
    if (snapshot.files_total != null || snapshot.files_done > 0) counts.push(snapshot.files_total == null
      ? text(`已处理 ${snapshot.files_done} 个文件`, `${snapshot.files_done} files processed`)
      : text(`${snapshot.files_done} / ${snapshot.files_total} 个文件`, `${snapshot.files_done} / ${snapshot.files_total} files`));
  }
  const rate = measured && snapshot.bytes_per_second != null && Number.isFinite(snapshot.bytes_per_second) && snapshot.bytes_per_second >= 0 ? snapshot.bytes_per_second : null;
  return <DatasetOperationProgress label={text('导入进度', 'Import progress')} phaseText={phaseText}
    state={state} done={done} total={total}
    detail={counts.length ? counts.join(' · ') : undefined}
    speed={state === 'active' ? rate != null ? `${(rate / 1024 ** 2).toFixed(2)} MiB/s` : null : undefined}
    elapsed={operation.elapsed} remaining={measured ? snapshot.eta_seconds : null} actions={actions}/>
}
