import { beforeEach, describe, expect, it, vi } from 'vitest';
import i18n from '../../../frontend/src/i18n';
import { filesFromDrop, filesFromSelection, MAX_DATASET_FILES, MAX_DATASET_UPLOAD_BYTES } from '../../../frontend/src/utils/datasetFiles';

beforeEach(async () => { await i18n.changeLanguage('zh-CN'); });

const imageFile = (name: string, size = 1) => {
  const file = new File(['x'], name, { type: 'image/png' });
  Object.defineProperty(file, 'size', { value: size });
  return file;
};
const selectionFile = (path: string) => {
  const file = new File(['caption'], path.split('/').pop()!);
  Object.defineProperty(file, 'webkitRelativePath', { value: path });
  return file;
};
const fileEntry = (file: File, fail = false): FileSystemFileEntry => ({
  name: file.name, isFile: true, isDirectory: false,
  file: vi.fn((success: FileCallback, error: ErrorCallback) => queueMicrotask(() => fail ? error(new DOMException('denied')) : success(file))),
}) as unknown as FileSystemFileEntry;
const directory = (name: string, batches: FileSystemEntry[][], fail = false): FileSystemDirectoryEntry => ({
  name, isFile: false, isDirectory: true,
  createReader: vi.fn(() => {
    let batch = 0;
    return { readEntries: vi.fn((success: FileSystemEntriesCallback, error: ErrorCallback) => queueMicrotask(() => fail ? error(new DOMException('denied')) : success(batches[batch++] || []))) };
  }),
}) as unknown as FileSystemDirectoryEntry;
const transfer = (entries: FileSystemEntry[], files: File[] = []): DataTransfer => ({
  files,
  items: entries.map(entry => ({ kind: 'file', webkitGetAsEntry: () => entry, getAsFile: () => null })),
}) as unknown as DataTransfer;

describe('dataset file selections', () => {
  it('retains folder paths, images, both caption formats, masks and custom caption suffixes', () => {
    const paths = ['角色/正面.png', '角色/正面.txt', '角色/正面.JSON', '角色/正面.mask.png', '角色/背面.mask', '角色/背面.caption'];
    expect(filesFromSelection(paths.map(selectionFile)).map(item => item.relativePath)).toEqual(paths);
  });

  it('supports plain file selections and a ZIP on its own', () => {
    const image = imageFile('face.png');
    const zip = new File(['zip'], 'images.zip');
    expect(filesFromSelection([image])).toEqual([{ file: image, relativePath: 'face.png' }]);
    expect(filesFromSelection([zip])).toEqual([{ file: zip, relativePath: 'images.zip' }]);
    expect(filesFromSelection([])).toEqual([]);
  });

  it('filters only the system metadata ignored by the backend', () => {
    const paths = ['data/.DS_Store', '__MACOSX/data._a.jpg', 'data/._face.jpg', 'data/._hidden/sub.png', 'data/notes.txt', 'data/.caption.json', 'data/Thumbs.db'];
    expect(filesFromSelection(paths.map(selectionFile)).map(item => item.relativePath)).toEqual(paths.slice(4));
  });

  it('deduplicates repeated references, but rejects conflicting case-insensitive paths', () => {
    const file = imageFile('face.png');
    expect(filesFromSelection([file, file])).toHaveLength(1);
    expect(() => filesFromSelection([file, imageFile('FACE.png')])).toThrow('文件路径重复');
    expect(() => filesFromSelection([selectionFile('cafe\u0301/face.png'), selectionFile('café/face.png')])).toThrow('文件路径重复');
  });

  it('rejects mixed ZIPs without silently dropping the other files', () => {
    expect(() => filesFromSelection([imageFile('face.png'), new File(['x'], 'data.zip')])).toThrow('请单独上传一个 ZIP');
  });

  it('bounds file count at the backend limit', () => {
    const files = Array.from({ length: MAX_DATASET_FILES }, (_, index) => imageFile(`${index}.png`));
    expect(filesFromSelection(files)).toHaveLength(MAX_DATASET_FILES);
    expect(() => filesFromSelection([...files, imageFile('extra.png')])).toThrow('5,000');
  });

  it('checks the total bytes and image, caption, ZIP part limits without reading payloads', () => {
    const imageLimit = 256 * 1024 ** 2;
    const files = Array.from({ length: 8 }, (_, index) => imageFile(`${index}.png`, imageLimit));
    expect(filesFromSelection(files)).toHaveLength(8);
    expect(() => filesFromSelection([...files, imageFile('extra.png')])).toThrow('总大小超过 2 GiB');
    expect(() => filesFromSelection([imageFile('large.png', imageLimit + 1)])).toThrow('256 MiB');
    expect(() => filesFromSelection([imageFile('caption.txt', 1024 ** 2 + 1)])).toThrow('1 MiB');
    expect(() => filesFromSelection([imageFile('data.zip', MAX_DATASET_UPLOAD_BYTES + 1)])).toThrow('2 GiB');
  });

  it('rejects unsafe traversal and excessive nesting before upload', () => {
    expect(() => filesFromSelection([selectionFile('../../../frontend/face.png')])).toThrow('文件路径过长或无效');
    expect(() => filesFromSelection([selectionFile(`${'nested/'.repeat(16)}face.png`)])).toThrow('文件路径过长或无效');
  });
});

describe('dropped dataset folders', () => {
  it('recurses through nested folders and preserves separate roots with matching filenames', async () => {
    const first = directory('人物', [[fileEntry(imageFile('face.png')), directory('标签', [[fileEntry(new File(['caption'], 'face.json'))]])]]);
    const second = directory('画风', [[fileEntry(imageFile('face.png'))]]);
    const result = await filesFromDrop(transfer([first, second]));
    expect(result.map(item => item.relativePath)).toEqual(['人物/face.png', '人物/标签/face.json', '画风/face.png']);
  });

  it('keeps reading directory batches after the first 100 entries', async () => {
    const hundred = Array.from({ length: 100 }, (_, index) => fileEntry(imageFile(`${index}.png`)));
    const folder = directory('dataset', [hundred, [fileEntry(imageFile('100.png'))]]);
    const result = await filesFromDrop(transfer([folder]));
    expect(result).toHaveLength(101);
    expect(result[100].relativePath).toBe('dataset/100.png');
    const reader = vi.mocked(folder.createReader).mock.results[0].value;
    expect(reader.readEntries).toHaveBeenCalledTimes(3);
  });

  it('captures every entry and fallback file before awaiting a directory read', async () => {
    let readable = true;
    const folder = directory('root', [[fileEntry(imageFile('one.png'))]]);
    const fallback = imageFile('two.png');
    const access = (value: unknown) => { if (!readable) throw new Error('drag data expired'); return value; };
    const dropped = { files: [], items: [
      { kind: 'file', webkitGetAsEntry: () => access(folder), getAsFile: () => access(null) },
      { kind: 'file', webkitGetAsEntry: () => access(null), getAsFile: () => access(fallback) },
    ] } as unknown as DataTransfer;
    const pending = filesFromDrop(dropped);
    readable = false;
    expect((await pending).map(item => item.relativePath)).toEqual(['root/one.png', 'two.png']);
  });

  it('supports ordinary dropped files when the browser has no directory API', async () => {
    const file = imageFile('one.png');
    const dropped = { files: [file], items: [{ kind: 'file', getAsFile: () => file }] } as unknown as DataTransfer;
    expect(await filesFromDrop(dropped)).toEqual([{ file, relativePath: 'one.png' }]);
    expect(await filesFromDrop({ files: [file] } as unknown as DataTransfer)).toHaveLength(1);
  });

  it('never uploads a directory placeholder File returned by the drag store', async () => {
    const pseudoFile = new File([], 'dataset');
    const folder = directory('dataset', [[fileEntry(imageFile('one.png'))]]);
    const dropped = { files: [pseudoFile], items: [{ kind: 'file', webkitGetAsEntry: () => folder, getAsFile: () => pseudoFile }] } as unknown as DataTransfer;
    expect((await filesFromDrop(dropped)).map(item => item.relativePath)).toEqual(['dataset/one.png']);
    await expect(filesFromDrop({ files: [pseudoFile] } as unknown as DataTransfer)).rejects.toThrow('选择文件夹');
  });

  it('reports empty folders and unreadable dropped items with an alternative', async () => {
    await expect(filesFromDrop(transfer([directory('empty', [])]))).rejects.toThrow('文件夹可能为空');
    await expect(filesFromDrop(transfer([]))).rejects.toThrow('选择文件夹');
    const inaccessible = { files: [], items: [{ kind: 'file', getAsFile: () => null }] } as unknown as DataTransfer;
    await expect(filesFromDrop(inaccessible)).rejects.toThrow('浏览器无法读取');
  });

  it.each(['directory', 'file'])('reports the failing %s path without partial upload', async kind => {
    const broken = kind === 'directory' ? directory('broken', [], true) : fileEntry(imageFile('broken.png'), true);
    const folder = directory('dataset', [[fileEntry(imageFile('readable.png')), broken]]);
    await expect(filesFromDrop(transfer([folder]))).rejects.toThrow(`无法读取“dataset/broken${kind === 'file' ? '.png' : ''}”`);
  });

  it('skips metadata directories before trying to read them, while preserving valid captions', async () => {
    const metadata = directory('__MACOSX', [], true);
    const folder = directory('dataset', [[metadata, fileEntry(new File(['caption'], 'face.caption')), fileEntry(imageFile('face.png'))]]);
    expect((await filesFromDrop(transfer([folder]))).map(item => item.relativePath)).toEqual(['dataset/face.caption', 'dataset/face.png']);
    expect(metadata.createReader).not.toHaveBeenCalled();
  });

  it('fails clearly when separately dropped folders have conflicting paths', async () => {
    const folders = [directory('dataset', [[fileEntry(imageFile('face.png'))]]), directory('DATASET', [[fileEntry(imageFile('FACE.png'))]])];
    await expect(filesFromDrop(transfer(folders))).rejects.toThrow('文件路径重复');
  });
});
