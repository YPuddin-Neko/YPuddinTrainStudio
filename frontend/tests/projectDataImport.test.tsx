import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { MemoryRouter } from 'react-router-dom';
import { File as NodeFile } from 'node:buffer';
import ProjectDataImport from '../src/pages/ProjectDetail/ProjectDataImport';
import i18n from '../src/i18n';

const server = setupServer(http.get('/api/projects/:pid/datasets/import-progress/:progressId', () => HttpResponse.json({}, { status: 404 })));
beforeAll(async () => {
  const form = await new Request('http://localhost', { method: 'POST', headers: { 'content-type': 'application/x-www-form-urlencoded' }, body: '' }).formData();
  vi.stubGlobal('FormData', form.constructor);
  vi.stubGlobal('File', NodeFile);
  server.listen({ onUnhandledRequest: 'error' });
});
beforeEach(async () => { await i18n.changeLanguage('zh-CN'); });
afterEach(() => { server.resetHandlers(); vi.restoreAllMocks(); });
afterAll(() => { server.close(); vi.unstubAllGlobals(); });

const source = (id = 'd_character', path = '/qa/traindata/character') => ({
  source: { id, path, project_id: 'p_import', version_id: 'v_import', repeats: 1, caption_ext: 'auto', is_reg: false, prior_weight: 1, class_prompt: null, created_at: 1 },
  stats: { images: 1, captioned: 1, masks: 0 }, index_status: 'ready', cache: {},
});
const image = (name = 'portrait.png') => new File(['image bytes'], name, { type: 'image/png' });
const entry = (file: File): FileSystemFileEntry => ({
  name: file.name, isFile: true, isDirectory: false,
  file: (success: FileCallback) => queueMicrotask(() => success(file)),
}) as FileSystemFileEntry;
const directory = (name: string, children: FileSystemEntry[]): FileSystemDirectoryEntry => ({
  name, isFile: false, isDirectory: true,
  createReader: () => {
    let sent = false;
    return { readEntries: (success: FileSystemEntriesCallback) => {
      const batch = sent ? [] : children; sent = true;
      queueMicrotask(() => success(batch));
    } };
  },
}) as FileSystemDirectoryEntry;
const drop = (entries: FileSystemEntry[]) => {
  fireEvent.drop(screen.getByTestId('dataset-dropzone'), { dataTransfer: {
    files: entries.map(item => new File([], item.name)),
    items: entries.map(item => ({ kind: 'file', webkitGetAsEntry: () => item, getAsFile: () => new File([], item.name) })),
  } });
};
const show = (onImported = vi.fn()) => ({
  ...render(<MemoryRouter><ProjectDataImport projectId="p_import" versionId="v_import" onImported={onImported}/></MemoryRouter>),
  onImported,
});
const selectImage = () => fireEvent.change(screen.getByLabelText('选择训练文件'), { target: { files: [image()] } });
const submit = () => fireEvent.click(screen.getByRole('button', { name: '导入当前版本' }));

const progressSnapshot = (id: string) => ({ id, phase: 'receiving', bytes_done: 1024 ** 2, bytes_total: 2 * 1024 ** 2,
  files_done: 0, files_total: null, elapsed_seconds: 2, phase_elapsed_seconds: 2, bytes_per_second: 512 * 1024, eta_seconds: 2, error: null });

describe('project import with browser folder collection and actual multipart requests', () => {
  it.each(['upload', 'path'])('imports via %s on LAN HTTP where crypto.randomUUID is unavailable', async mode => {
    const originalCrypto = globalThis.crypto;
    vi.stubGlobal('crypto', { getRandomValues: originalCrypto.getRandomValues.bind(originalCrypto) });
    let id = '';
    server.use(http.post(`/api/projects/p_import/datasets${mode === 'upload' ? '/upload' : ''}`, ({ request }) => {
      id = new URL(request.url).searchParams.get('progress_id') || '';
      return HttpResponse.json(source());
    }));
    try {
      const { onImported } = show();
      if (mode === 'upload') selectImage();
      else {
        fireEvent.click(screen.getByRole('button', { name: '从训练电脑导入' }));
        fireEvent.change(screen.getByRole('textbox', { name: '文件或目录路径' }), { target: { value: '/qa/训练图' } });
      }
      submit(); await screen.findByRole('link', { name: '查看图片与标签' });
      expect(id).toMatch(/^[\da-f-]{36}$/);
      expect(onImported).toHaveBeenCalledOnce();
      expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    } finally { vi.stubGlobal('crypto', originalCrypto); }
  });

  it.each(['upload', 'path'])('still submits %s without progress_id when cryptographic APIs are absent', async mode => {
    const originalCrypto = globalThis.crypto;
    vi.stubGlobal('crypto', undefined);
    let finishUpload = () => {};
    const pending = new Promise<void>(resolve => { finishUpload = resolve; });
    let requestUrl: URL | undefined;
    const polled = vi.fn();
    server.use(
      http.post(`/api/projects/p_import/datasets${mode === 'upload' ? '/upload' : ''}`, async ({ request }) => {
        requestUrl = new URL(request.url);
        await pending;
        return HttpResponse.json(source());
      }),
      http.get('/api/projects/p_import/datasets/import-progress/:progressId', () => { polled(); return HttpResponse.json({}); }),
    );
    try {
      const { onImported } = show();
      if (mode === 'upload') selectImage();
      else {
        fireEvent.click(screen.getByRole('button', { name: '从训练电脑导入' }));
        fireEvent.change(screen.getByRole('textbox', { name: '文件或目录路径' }), { target: { value: '/qa/训练图' } });
      }
      submit();
      await waitFor(() => expect(requestUrl).toBeDefined());
      expect(requestUrl!.searchParams.has('progress_id')).toBe(false);
      expect(requestUrl!.searchParams.get('version_id')).toBe('v_import');
      expect(screen.getByText('暂时无法读取进度，正在等待导入结果')).toBeInTheDocument();
      expect(screen.getByRole('progressbar')).not.toHaveAttribute('aria-valuenow');
      expect(onImported).not.toHaveBeenCalled();
      await act(async () => { finishUpload(); });
      await screen.findByRole('link', { name: '查看图片与标签' });
      expect(onImported).toHaveBeenCalledOnce();
      expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: '继续添加' })).toBeEnabled();
      expect(polled).not.toHaveBeenCalled();
      expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    } finally { finishUpload(); vi.stubGlobal('crypto', originalCrypto); }
  });

  it('displays actual byte progress and speed under the same ID as the upload, then retains confirmed completion', async () => {
    let finishUpload: () => void = () => {};
    const upload = new Promise<void>(resolve => { finishUpload = resolve; });
    let requestId = '';
    let polledId = '';
    server.use(
      http.post('/api/projects/p_import/datasets/upload', async ({ request }) => {
        requestId = new URL(request.url).searchParams.get('progress_id') || '';
        await upload; return HttpResponse.json({ ...source(), index_status: 'indexing' });
      }),
      http.get('/api/projects/p_import/datasets/import-progress/:progressId', ({ params }) => {
        polledId = String(params.progressId); return HttpResponse.json(progressSnapshot(polledId));
      }),
    );
    const { onImported } = show(); selectImage(); submit();
    expect(await screen.findByRole('progressbar', { name: '导入进度 · 上传文件' })).toHaveAttribute('aria-valuenow', '50');
    expect(screen.getByText('0.50 MiB/s')).toBeInTheDocument();
    expect(screen.getByText('当前阶段：1.0 MiB / 2.0 MiB')).toBeInTheDocument();
    await waitFor(() => expect(requestId).toBe(polledId));
    expect(requestId).toMatch(/^[\da-f-]{36}$/);
    expect(onImported).not.toHaveBeenCalled();
    await act(async () => { finishUpload(); });
    await screen.findByRole('link', { name: '查看图片与标签' });
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: '继续添加' })).toBeEnabled();
    expect(screen.getByText('已导入当前版本，共 1 组图片。')).toBeInTheDocument();
    expect(screen.queryByText('检查已完成')).not.toBeInTheDocument();
    expect(onImported).toHaveBeenCalledOnce();
  });

  it('keeps a fast local-path import successful even if its progress endpoint is unavailable', async () => {
    let requestId = '';
    server.use(http.post('/api/projects/p_import/datasets', ({ request }) => {
      requestId = new URL(request.url).searchParams.get('progress_id') || '';
      return HttpResponse.json(source());
    }));
    show(); fireEvent.click(screen.getByRole('button', { name: '从训练电脑导入' }));
    fireEvent.change(screen.getByRole('textbox', { name: '文件或目录路径' }), { target: { value: '/qa/训练图' } }); submit();
    await screen.findByRole('link', { name: '查看图片与标签' });
    expect(requestId).toMatch(/^[\da-f-]{36}$/);
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: '继续添加' })).toBeEnabled();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('does not publish late upload success or retain polling after the import panel unmounts', async () => {
    let finishUpload: () => void = () => {};
    const upload = new Promise<void>(resolve => { finishUpload = resolve; });
    let started = false;
    server.use(http.post('/api/projects/p_import/datasets/upload', async () => {
      started = true; await upload; return HttpResponse.json(source());
    }));
    const old = show(); selectImage(); submit();
    await waitFor(() => expect(started).toBe(true));
    old.unmount();
    const next = show(); selectImage();
    await act(async () => { finishUpload(); await upload; });
    expect(old.onImported).not.toHaveBeenCalled();
    expect(next.onImported).not.toHaveBeenCalled();
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    expect(screen.getByText('portrait.png')).toBeInTheDocument();
  });

  it('uploads nested drop paths, captions and masks without directory placeholder files', async () => {
    let fields: FormData | undefined;
    let version = '';
    server.use(http.post('/api/projects/p_import/datasets/upload', async ({ request }) => {
      version = new URL(request.url).searchParams.get('version_id') || '';
      fields = await request.formData();
      return HttpResponse.json(source());
    }));
    const { onImported } = show();
    expect(screen.queryByRole('textbox', { name: '数据集名称（可选）' })).not.toBeInTheDocument();
    const folder = directory('角色', [directory('正面', [
      entry(image()), entry(new File(['character'], 'portrait.json')), entry(image('portrait.mask.png')),
    ])]);
    drop([folder]);
    expect(await screen.findByText('角色/正面/portrait.json')).toBeInTheDocument();
    expect(screen.getByText(/已选 3 个文件/)).toBeInTheDocument();
    submit();
    expect(await screen.findByRole('link', { name: '查看图片与标签' })).toHaveAttribute('href', '/datasets/d_character');
    expect(version).toBe('v_import');
    expect(fields?.getAll('files').map(file => (file as File).name)).toEqual(['角色/正面/portrait.png', '角色/正面/portrait.json', '角色/正面/portrait.mask.png']);
    expect(fields?.get('name')).toBe('角色');
    expect(fields?.get('caption_ext')).toBe('auto');
    expect(onImported).toHaveBeenCalledOnce();
    expect(screen.queryByText('角色/正面/portrait.png')).not.toBeInTheDocument();
  });

  it('uses folder-picker relative paths and derives a ZIP name without an extra naming field', async () => {
    const names: string[] = [];
    const filenames: string[][] = [];
    server.use(http.post('/api/projects/p_import/datasets/upload', async ({ request }) => {
      const form = await request.formData(); names.push(String(form.get('name')));
      filenames.push(form.getAll('files').map(file => (file as File).name));
      return HttpResponse.json(source());
    }));
    show();
    const file = image(); Object.defineProperty(file, 'webkitRelativePath', { value: '选择的目录/嵌套/portrait.png' });
    fireEvent.change(screen.getByLabelText('选择训练文件夹'), { target: { files: [file] } });
    submit();
    await screen.findByRole('link', { name: '查看图片与标签' });
    fireEvent.click(screen.getByRole('button', { name: '继续添加' }));
    fireEvent.change(screen.getByLabelText('选择训练文件'), { target: { files: [new File(['archive'], '多概念数据.zip')] } });
    submit();
    await waitFor(() => expect(names).toHaveLength(2));
    expect(names).toEqual(['选择的目录', '多概念数据']);
    expect(filenames).toEqual([['选择的目录/嵌套/portrait.png'], ['多概念数据.zip']]);
    await screen.findByRole('link', { name: '查看图片与标签' });
  });

  it('shows a link for every dataset returned by a multi-folder import', async () => {
    const first = source(); const second = source('d_style', 'D:\\qa\\traindata\\画风');
    server.use(http.post('/api/projects/p_import/datasets/upload', () => HttpResponse.json({ ...first, datasets: [first, second] })));
    const { onImported } = show();
    selectImage(); submit();
    expect(await screen.findByText('已导入当前版本，共 2 组图片。')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'character' })).toHaveAttribute('href', '/datasets/d_character');
    expect(screen.getByRole('link', { name: '画风' })).toHaveAttribute('href', '/datasets/d_style');
    expect(onImported).toHaveBeenCalledOnce();
    expect(screen.queryByRole('button', { name: '导入当前版本' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '继续添加' }));
    expect(screen.getByRole('button', { name: '导入当前版本' })).toBeDisabled();
  });

  it('distinguishes failed file transmission from an unreachable service and retains the selection', async () => {
    const health = vi.fn();
    server.use(
      http.post('/api/projects/p_import/datasets/upload', () => HttpResponse.error()),
      http.get('/api/health', () => { health(); return HttpResponse.json({ ok: true }); }),
    );
    const { onImported } = show();
    selectImage(); submit();
    expect(await screen.findByRole('alert')).toHaveTextContent('训练服务仍可连接，但文件未能发送');
    expect(screen.queryByText('暂时无法连接训练服务，请确认服务已启动后重试。')).not.toBeInTheDocument();
    expect(screen.getByText('portrait.png')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '导入当前版本' })).toBeEnabled();
    expect(health).toHaveBeenCalledOnce();
    expect(onImported).not.toHaveBeenCalled();
  });

  it('retains files and permits retry after a genuine connection failure', async () => {
    server.use(
      http.post('/api/projects/p_import/datasets/upload', () => HttpResponse.error()),
      http.get('/api/health', () => HttpResponse.error()),
    );
    const { onImported } = show();
    selectImage(); submit();
    expect(await screen.findByRole('alert')).toHaveTextContent('上传连接已中断，所选文件已保留');
    expect(screen.getByText('portrait.png')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '导入当前版本' })).toBeEnabled();
    server.use(http.post('/api/projects/p_import/datasets/upload', () => HttpResponse.json(source())));
    submit();
    await screen.findByRole('link', { name: '查看图片与标签' });
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(onImported).toHaveBeenCalledOnce();
  });

  it('preserves an HTTP conflict reason and does not misdiagnose it using health checks', async () => {
    const health = vi.fn();
    server.use(
      http.post('/api/projects/p_import/datasets/upload', () => HttpResponse.json({ error: { code: 'upload.duplicate', message: '目标目录含同名文件：portrait.png' } }, { status: 409 })),
      http.get('/api/health', () => { health(); return HttpResponse.json({ ok: true }); }),
    );
    show(); selectImage(); submit();
    expect(await screen.findByRole('alert')).toHaveTextContent('目标目录含同名文件：portrait.png');
    expect(screen.getByText('portrait.png')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '导入当前版本' })).toBeEnabled();
    expect(health).not.toHaveBeenCalled();
  });

  it.each([
    ['upload.conflict', 'existing file has different content: 角色/正面/portrait.png; rename it before importing', '导入内容与当前版本的已有文件冲突'],
    ['dataset.overlap', 'a child of /qa/traindata/角色 is already a dataset; import that child folder instead', '导入目录与已有训练、正则或验证数据源重叠'],
  ])('explains %s in Chinese without losing the actual server path', async (code, message, summary) => {
    server.use(http.post('/api/projects/p_import/datasets/upload', () => HttpResponse.json({ error: { code, message } }, { status: 409 })));
    const { onImported } = show(); selectImage(); submit();
    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent(summary);
    expect(alert).toHaveTextContent(message);
    expect(screen.getByText('portrait.png')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '导入当前版本' })).toBeEnabled();
    expect(onImported).not.toHaveBeenCalled();
  });

  it('explains overlap for training-computer imports and preserves their entered path', async () => {
    const message = 'configured source /qa/训练图 overlaps an indexed child dataset';
    server.use(http.post('/api/projects/p_import/datasets', () => HttpResponse.json({ error: { code: 'dataset.overlap', message } }, { status: 409 })));
    show();
    fireEvent.click(screen.getByRole('button', { name: '从训练电脑导入' }));
    const path = screen.getByRole('textbox', { name: '文件或目录路径' });
    fireEvent.change(path, { target: { value: '/qa/训练图' } }); submit();
    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('导入目录与已有训练、正则或验证数据源重叠');
    expect(alert).toHaveTextContent(message);
    expect(path).toHaveValue('/qa/训练图');
  });

  it('uses an English conflict explanation when the interface language is English', async () => {
    await i18n.changeLanguage('en');
    const message = 'upload file and directory names conflict: 角色/portrait.png';
    server.use(http.post('/api/projects/p_import/datasets/upload', () => HttpResponse.json({ error: { code: 'upload.conflict', message } }, { status: 409 })));
    show();
    fireEvent.change(screen.getByLabelText('Choose training files'), { target: { files: [image()] } });
    fireEvent.click(screen.getByRole('button', { name: 'Import into this version' }));
    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('The import conflicts with files already in this version');
    expect(alert).toHaveTextContent(message);
    expect(alert).not.toHaveTextContent('导入内容');
  });

  it('locks selection and submission while reading a folder, then unlocks after the final read', async () => {
    let finishRead: FileSystemEntriesCallback | undefined;
    let first = true;
    const folder = { name: 'delayed', isDirectory: true, isFile: false, createReader: () => ({ readEntries: (success: FileSystemEntriesCallback) => {
      if (first) { first = false; finishRead = success; } else success([]);
    } }) } as FileSystemDirectoryEntry;
    show(); selectImage(); drop([folder]);
    expect(screen.getByText('正在读取文件夹…')).toBeInTheDocument();
    expect(screen.getByTestId('project-data-import')).toHaveAttribute('aria-busy', 'true');
    for (const name of ['选择文件夹', '选择文件 / ZIP', '从训练电脑导入', '清空选择', '导入当前版本']) {
      expect(screen.getByRole('button', { name })).toBeDisabled();
    }
    expect(screen.getByLabelText('选择训练文件')).toBeDisabled();
    expect(screen.getByLabelText('选择训练文件夹')).toBeDisabled();
    fireEvent.drop(screen.getByTestId('dataset-dropzone'), { dataTransfer: { files: [image('ignored.png')] } });
    await act(async () => { finishRead?.([entry(image('ready.png'))]); });
    expect(await screen.findByText('delayed/ready.png')).toBeInTheDocument();
    expect(screen.queryByText('ignored.png')).not.toBeInTheDocument();
    expect(screen.queryByText('portrait.png')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: '导入当前版本' })).toBeEnabled();
  });

  it('does not publish a late directory result after unmounting', async () => {
    let finishRead: FileSystemEntriesCallback | undefined;
    let first = true;
    const folder = { name: 'old-folder', isDirectory: true, isFile: false, createReader: () => ({ readEntries: (success: FileSystemEntriesCallback) => {
      if (first) { first = false; finishRead = success; } else success([]);
    } }) } as FileSystemDirectoryEntry;
    const old = show(); drop([folder]); old.unmount();
    const next = show(); selectImage();
    await act(async () => { finishRead?.([entry(image('old.png'))]); });
    expect(screen.getByText('portrait.png')).toBeInTheDocument();
    expect(screen.queryByText('old-folder/old.png')).not.toBeInTheDocument();
    expect(screen.getByTestId('project-data-import')).toHaveAttribute('aria-busy', 'false');
    expect(old.onImported).not.toHaveBeenCalled(); expect(next.onImported).not.toHaveBeenCalled();
  });

  it('keeps the previous selection on directory access errors without making a network request', async () => {
    const network = vi.fn();
    server.use(http.all('/api/*', () => { network(); return HttpResponse.json({}); }));
    show(); selectImage();
    const folder = { name: '不可读目录', isDirectory: true, isFile: false, createReader: () => ({ readEntries: (_success: FileSystemEntriesCallback, fail: ErrorCallback) => fail(new DOMException('denied')) }) } as FileSystemDirectoryEntry;
    drop([folder]);
    expect(await screen.findByRole('alert')).toHaveTextContent('无法读取“不可读目录”');
    expect(screen.getByRole('alert')).toHaveTextContent('选择文件夹');
    expect(screen.getByText('portrait.png')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '导入当前版本' })).toBeEnabled();
    expect(network).not.toHaveBeenCalled();
  });
});
