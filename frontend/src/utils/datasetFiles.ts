import i18n from '../i18n';

export type DatasetUploadFile = { file: File; relativePath: string };

// Keep these bounds aligned with server/dataset_uploads.py.
export const MAX_DATASET_FILES = 5000;
export const MAX_DATASET_UPLOAD_BYTES = 2 * 1024 ** 3;
const MAX_IMAGE_BYTES = 256 * 1024 ** 2;
const MAX_CAPTION_BYTES = 1024 ** 2;
const IMAGE_SUFFIXES = new Set(['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff', '.avif', '.jxl']);

const message = (zh: string, en: string) => i18n.language?.startsWith('en') ? en : zh;
const selectionHint = () => message('请使用“选择文件夹”，或将文件夹压缩为一个 ZIP 后上传。', 'Use “Choose a folder”, or upload the folder as one ZIP archive.');
const ignoredPath = (path: string) => path.split('/').some(part => part === '__MACOSX' || part === '.DS_Store' || part.startsWith('._'));

function relativePath(path: string): string {
  const normalized = path.replace(/\\/g, '/').normalize('NFC');
  const parts = normalized.split('/');
  if (normalized.length > 240 || parts.length > 16 || parts.some(part => !part || part === '.' || part === '..')) {
    throw new Error(message(`文件路径过长或无效：${path}。请减少文件夹层级后重试。`, `The file path is too long or invalid: ${path}. Reduce the folder nesting and retry.`));
  }
  return normalized;
}

function collector() {
  const files: DatasetUploadFile[] = [];
  const paths = new Map<string, File>();
  let totalBytes = 0;
  return {
    add(file: File, path: string) {
      if (ignoredPath(path)) return;
      const normalized = relativePath(path);
      const key = normalized.toLowerCase();
      const previous = paths.get(key);
      if (previous === file) return;
      if (previous) {
        throw new Error(message(`文件路径重复：${normalized}。请重命名同名文件或文件夹，再重新选择。`, `Duplicate file path: ${normalized}. Rename the conflicting file or folder, then select again.`));
      }
      if (files.length >= MAX_DATASET_FILES) {
        throw new Error(message('一次最多上传 5,000 个文件，请分批导入。', 'Upload at most 5,000 files at a time. Split the import into smaller batches.'));
      }
      const suffix = normalized.slice(normalized.lastIndexOf('.')).toLowerCase();
      const limit = suffix === '.zip' ? MAX_DATASET_UPLOAD_BYTES : IMAGE_SUFFIXES.has(suffix) || suffix === '.mask' ? MAX_IMAGE_BYTES : MAX_CAPTION_BYTES;
      if (file.size > limit) {
        const limitLabel = limit === MAX_DATASET_UPLOAD_BYTES ? '2 GiB' : limit === MAX_IMAGE_BYTES ? '256 MiB' : '1 MiB';
        throw new Error(message(`文件过大：${normalized}，单个文件上限为 ${limitLabel}。`, `File too large: ${normalized}. Its size limit is ${limitLabel}.`));
      }
      totalBytes += file.size;
      if (totalBytes > MAX_DATASET_UPLOAD_BYTES) {
        throw new Error(message('所选文件总大小超过 2 GiB，请分批导入。', 'The selected files exceed 2 GiB. Split the import into smaller batches.'));
      }
      paths.set(key, file);
      files.push({ file, relativePath: normalized });
    },
    finish() {
      if (files.length > 1 && files.some(({ relativePath: path }) => path.toLowerCase().endsWith('.zip'))) {
        throw new Error(message('请单独上传一个 ZIP，或选择文件夹及图片、标签文件；不能混合上传。', 'Upload one ZIP on its own, or select folders, images and captions without ZIP archives.'));
      }
      return files;
    },
  };
}

/** Folder-picker files already carry their path; retain captions and masks without guessing formats. */
export function filesFromSelection(files: File[]): DatasetUploadFile[] {
  const selected = collector();
  for (const file of files) selected.add(file, file.webkitRelativePath || file.name);
  return selected.finish();
}

type CapturedDropItem = { entry: FileSystemEntry | null; file: File | null };

function unreadable(path: string): Error {
  return new Error(message(`无法读取“${path}”。请确认文件仍在原位置且允许浏览器读取。${selectionHint()}`, `Cannot read “${path}”. Check that it still exists and that the browser can read it. ${selectionHint()}`));
}

/** Capture drag-store handles synchronously: browsers revoke DataTransfer access after the drop event. */
export async function filesFromDrop(transfer: DataTransfer): Promise<DatasetUploadFile[]> {
  const captured: CapturedDropItem[] = [];
  let fallbackFiles: File[];
  try {
    fallbackFiles = Array.from(transfer.files || []);
    for (const item of Array.from(transfer.items || [])) {
      if (item.kind !== 'file') continue;
      const compatibleItem = item as DataTransferItem & { getAsEntry?: () => FileSystemEntry | null };
      const entry = typeof item.webkitGetAsEntry === 'function' ? item.webkitGetAsEntry() : compatibleItem.getAsEntry?.() || null;
      captured.push({ entry, file: item.getAsFile() });
    }
  } catch {
    throw new Error(message(`浏览器未允许读取拖入的内容。${selectionHint()}`, `The browser did not allow access to the dropped items. ${selectionHint()}`));
  }
  const selected = collector();
  const visit = async (entry: FileSystemEntry, parent = ''): Promise<void> => {
    const path = parent ? `${parent}/${entry.name}` : entry.name;
    if (ignoredPath(path)) return;
    relativePath(path);
    if (entry.isFile) {
      const file = await new Promise<File>((resolve, reject) => {
        try { (entry as FileSystemFileEntry).file(resolve, () => reject(unreadable(path))); }
        catch { reject(unreadable(path)); }
      });
      selected.add(file, path);
    } else if (entry.isDirectory) {
      let reader: FileSystemDirectoryReader;
      try { reader = (entry as FileSystemDirectoryEntry).createReader(); }
      catch { throw unreadable(path); }
      // Chromium returns at most 100 entries per read; one read silently drops the remaining files.
      while (true) {
        const children = await new Promise<FileSystemEntry[]>((resolve, reject) => {
          try { reader.readEntries(resolve, () => reject(unreadable(path))); }
          catch { reject(unreadable(path)); }
        });
        if (children.length === 0) break;
        for (const child of children) await visit(child, path);
      }
    } else {
      throw unreadable(path);
    }
  };
  const addFallback = (file: File) => {
    // Some browsers expose a directory as an empty File; sending it causes a misleading fetch error.
    if (!file.type && file.size === 0 && !file.name.includes('.') && !file.webkitRelativePath) {
      throw new Error(message(`浏览器无法展开拖入的文件夹“${file.name}”。${selectionHint()}`, `The browser cannot expand the dropped folder “${file.name}”. ${selectionHint()}`));
    }
    selected.add(file, file.webkitRelativePath || file.name);
  };
  if (captured.length) {
    for (const { entry, file } of captured) {
      if (entry) await visit(entry);
      else if (file) addFallback(file);
      else throw new Error(message(`浏览器无法读取拖入的文件或文件夹。${selectionHint()}`, `The browser cannot read the dropped file or folder. ${selectionHint()}`));
    }
  } else {
    for (const file of fallbackFiles) addFallback(file);
  }
  const result = selected.finish();
  if (result.length === 0) {
    throw new Error(message(`没有读到可上传的文件；文件夹可能为空，或仅含系统隐藏文件。${selectionHint()}`, `No uploadable files were found. The folder may be empty or contain only system metadata. ${selectionHint()}`));
  }
  return result;
}
