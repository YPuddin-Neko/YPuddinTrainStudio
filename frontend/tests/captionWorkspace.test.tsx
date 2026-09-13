import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createMemoryRouter, Link, MemoryRouter, RouterProvider } from 'react-router-dom';
import CaptionWorkspace from '../src/components/datasets/CaptionWorkspace';
import { apiClient } from '../src/api/client';
import type { DatasetImage } from '../src/api/types';
import i18n from '../src/i18n';

const source = (id: string) => ({ source: { id, path: `/qa/${id}` }, index_status: 'ready' });
const picture = (path: string, changes: Partial<DatasetImage> = {}): DatasetImage => ({ hash: 'samehash', rel_path: path, width: 768, height: 1024,
  caption: 'blue eyes, portrait. Original natural sentence.', caption_tags: 'blue eyes, portrait', caption_description: 'Original natural sentence.',
  caption_format: 'json', caption_status: 'captioned', has_mask: false, ...changes });
let pictures: DatasetImage[];
let requests: { url: string; params: any }[];
const stats = { images: 90, captioned: 80, missing: 8, invalid: 2, formats: { json: 50, txt: 32 }, unique_tags: 2, tags: [{ tag: 'blue eyes', count: 70 }, { tag: 'portrait', count: 30 }] };
function show(props: Partial<React.ComponentProps<typeof CaptionWorkspace>> = {}) {
  const onChanged = vi.fn();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return { ...render(<QueryClientProvider client={client}><MemoryRouter><CaptionWorkspace projectId="p_caption" versionId="v_caption" onChanged={onChanged} {...props}/></MemoryRouter></QueryClientProvider>), onChanged };
}
const addPending = (value = 'new tag') => fireEvent.change(screen.getByRole('textbox', { name: '添加标签' }), { target: { value } });
beforeEach(async () => {
  await i18n.changeLanguage('zh-CN'); requests = [];
  pictures = [picture('folderA/one.png'), picture('folderB/one.png', { caption: 'TXT natural text', caption_tags: 'TXT natural text', caption_description: null, caption_format: 'txt' })];
  vi.spyOn(apiClient, 'get').mockImplementation(async (url, options) => {
    const params = options?.params; requests.push({ url, params });
    if (url === '/projects/p_caption/datasets') return [source('d_first'), source('d_second')] as any;
    if (url.endsWith('/caption-stats')) return stats as any;
    return { items: pictures, total: 90, page: params?.page || 1, page_size: 30 } as any;
  });
  vi.spyOn(apiClient, 'put').mockImplementation(async (_url, payload, options) => {
    const body = payload as { caption: string; description?: string };
    pictures = pictures.map(item => item.rel_path === options?.params?.rel_path ? { ...item, caption_tags: body.caption, caption: body.caption, caption_description: body.description ?? item.caption_description } : item);
    return { caption: body.caption } as any;
  });
});
afterEach(() => vi.restoreAllMocks());

describe('independent caption workspace', () => {
  it('shows whole-dataset statistics rather than counts from the current image page and sends exact tag/status filters', async () => {
    show(); await screen.findByRole('img', { name: '大图：folderA/one.png' });
    expect(screen.getByText('整个目录 90 张图片 · 2 个不同标签')).toBeInTheDocument();
    expect(within(screen.getByRole('group', { name: '标签状态筛选' })).getByRole('button', { name: '全部图片90' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '筛选标签：blue eyes，70 张图片' }));
    await waitFor(() => expect(requests.some(request => request.url.endsWith('/images') && request.params?.tag === 'blue eyes' && !request.params.q)).toBe(true));
    fireEvent.click(screen.getByRole('button', { name: '缺少标签8' }));
    await waitFor(() => expect(requests.at(-1)?.params).toMatchObject({ page: 1, tag: 'blue eyes', caption_status: 'missing' }));
    expect(screen.getByText('整个目录 90 张图片 · 2 个不同标签')).toBeInTheDocument();
    expect(apiClient.put).not.toHaveBeenCalled();
  });

  it.each([['d_second', 'd_second'], ['not-in-this-version', 'd_first']])('opens initial source %s safely as %s', async (initialDatasetId, expected) => {
    show({ initialDatasetId }); await screen.findByRole('img', { name: '大图：folderA/one.png' });
    expect(requests.find(request => request.url.endsWith('/images'))?.url).toBe(`/datasets/${expected}/images`);
    expect(requests.some(request => request.url.includes('not-in-this-version'))).toBe(false);
    expect(requests[0].params).toEqual({ version_id: 'v_caption' });
  });

  it('saves JSON tags using the exact relative path without combining the natural-language field or rewriting metadata', async () => {
    const { onChanged } = show(); await screen.findByRole('img', { name: '大图：folderA/one.png' });
    expect(screen.getByRole('textbox', { name: '自然语言描述' })).toHaveValue('Original natural sentence.');
    addPending();
    expect(apiClient.put).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: '保存标签' }));
    await screen.findByText('标签已保存');
    expect(apiClient.put).toHaveBeenCalledWith('/datasets/d_first/images/samehash/caption', { caption: 'blue eyes, portrait, new tag' }, { params: { rel_path: 'folderA/one.png' }, silent: true });
    await waitFor(() => expect(onChanged).toHaveBeenCalledOnce());
    expect(requests.filter(request => request.url.endsWith('/caption-stats'))).toHaveLength(2);
    expect(screen.getByRole('button', { name: '保存标签' })).toBeDisabled();
  });

  it('saves an explicitly edited JSON description separately, including clearing it', async () => {
    show(); await screen.findByRole('textbox', { name: '自然语言描述' });
    fireEvent.change(screen.getByRole('textbox', { name: '自然语言描述' }), { target: { value: '' } });
    fireEvent.click(screen.getByRole('button', { name: '保存标签' }));
    await screen.findByText('标签已保存');
    expect(apiClient.put).toHaveBeenCalledWith(expect.any(String), { caption: 'blue eyes, portrait', description: '' }, expect.objectContaining({ params: { rel_path: 'folderA/one.png' } }));
  });

  it('edits TXT as full text and targets the second path even when two images have the same hash', async () => {
    show(); await screen.findByRole('img', { name: '大图：folderA/one.png' });
    fireEvent.click(screen.getByRole('button', { name: '选择图片：folderB/one.png' }));
    await screen.findByRole('img', { name: '大图：folderB/one.png' });
    fireEvent.click(screen.getByRole('button', { name: '文本 / 自然语言' }));
    const caption = 'A full sentence, with its punctuation.\n第二行描述。';
    fireEvent.change(screen.getByRole('textbox', { name: '标签文本' }), { target: { value: caption } });
    fireEvent.click(screen.getByRole('button', { name: '保存标签' }));
    await screen.findByText('标签已保存');
    expect(apiClient.put).toHaveBeenCalledWith('/datasets/d_first/images/samehash/caption', { caption }, { params: { rel_path: 'folderB/one.png' }, silent: true });
    expect(screen.queryByRole('textbox', { name: '自然语言描述' })).not.toBeInTheDocument();
  });

  it('blocks malformed JSON editing and saving while retaining the parsing error', async () => {
    pictures = [picture('broken.png', { caption: '', caption_tags: '', caption_status: 'invalid', caption_error: 'expected JSON object' })];
    show(); await screen.findByRole('img', { name: '大图：broken.png' });
    expect(screen.getByRole('alert')).toHaveTextContent('expected JSON object');
    expect(screen.getByRole('textbox', { name: '添加标签' })).toBeDisabled();
    expect(screen.getByRole('textbox', { name: '自然语言描述' })).toBeDisabled();
    expect(screen.getByRole('button', { name: '保存标签' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: '保存标签' }));
    expect(apiClient.put).not.toHaveBeenCalled();
  });

  it('requires an explicit choice before switching images and preserves pending tag input on cancel', async () => {
    show(); await screen.findByRole('img', { name: '大图：folderA/one.png' }); addPending();
    fireEvent.click(screen.getByRole('button', { name: '下一张' }));
    let dialog = await screen.findByRole('dialog', { name: '标签修改尚未保存' });
    expect(apiClient.put).not.toHaveBeenCalled();
    fireEvent.click(within(dialog).getByRole('button', { name: '继续编辑' }));
    expect(screen.getByRole('textbox', { name: '添加标签' })).toHaveValue('new tag');
    expect(screen.getByRole('img', { name: '大图：folderA/one.png' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '下一张' })); dialog = await screen.findByRole('dialog');
    fireEvent.click(within(dialog).getByRole('button', { name: '放弃并继续' }));
    await screen.findByRole('img', { name: '大图：folderB/one.png' });
    expect(apiClient.put).not.toHaveBeenCalled();
  });

  it('saves before a requested page switch, and keeps the editor when saving fails', async () => {
    vi.mocked(apiClient.put).mockRejectedValueOnce(new Error('read-only folder'));
    show(); await screen.findByRole('img', { name: '大图：folderA/one.png' }); addPending();
    fireEvent.click(screen.getByRole('button', { name: '下一页' }));
    const dialog = await screen.findByRole('dialog');
    fireEvent.click(within(dialog).getByRole('button', { name: '保存并继续' }));
    expect(await within(dialog).findByRole('alert')).toHaveTextContent('read-only folder');
    expect(screen.getByRole('textbox', { name: '添加标签' })).toHaveValue('new tag');
    expect(requests.some(request => request.params?.page === 2)).toBe(false);
    fireEvent.click(within(dialog).getByRole('button', { name: '保存并继续' }));
    await waitFor(() => expect(requests.some(request => request.params?.page === 2)).toBe(true));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('protects refresh and source changes, and includes draft text in browser unload protection', async () => {
    show(); await screen.findByRole('img', { name: '大图：folderA/one.png' }); addPending();
    const unload = new Event('beforeunload', { cancelable: true }); window.dispatchEvent(unload);
    expect(unload.defaultPrevented).toBe(true);
    fireEvent.click(screen.getByRole('button', { name: '刷新标签与统计' }));
    fireEvent.click(within(await screen.findByRole('dialog')).getByRole('button', { name: '继续编辑' }));
    fireEvent.click(screen.getByRole('combobox', { name: '图片目录' }));
    fireEvent.click(screen.getByRole('option', { name: 'd_second' }));
    const dialog = await screen.findByRole('dialog');
    expect(requests.some(request => request.url === '/datasets/d_second/images')).toBe(false);
    fireEvent.click(within(dialog).getByRole('button', { name: '放弃并继续' }));
    await waitFor(() => expect(requests.some(request => request.url === '/datasets/d_second/images')).toBe(true));
  });

  it('protects router navigation until the user saves or discards the current draft', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const router = createMemoryRouter([
      { path: '/', element: <><CaptionWorkspace projectId="p_caption" versionId="v_caption"/><Link to="/other">其他页面</Link></> },
      { path: '/other', element: <p>已离开标签工作区</p> },
    ]);
    render(<QueryClientProvider client={client}><RouterProvider router={router}/></QueryClientProvider>);
    await screen.findByRole('img', { name: '大图：folderA/one.png' }); addPending();
    fireEvent.click(screen.getByRole('link', { name: '其他页面' }));
    const dialog = await screen.findByRole('dialog');
    expect(router.state.location.pathname).toBe('/');
    fireEvent.click(within(dialog).getByRole('button', { name: '保存并继续' }));
    await screen.findByText('已离开标签工作区');
    expect(apiClient.put).toHaveBeenCalledOnce();
  });

  it('supports read-only inspection and filtering without exposing save or tag mutation actions', async () => {
    show({ readOnly: true }); await screen.findByRole('img', { name: '大图：folderA/one.png' });
    expect(screen.queryByRole('button', { name: '保存标签' })).not.toBeInTheDocument();
    expect(screen.queryByRole('textbox', { name: '添加标签' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '删除标签：blue eyes' })).not.toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: '自然语言描述' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: '下一张' }));
    await screen.findByRole('img', { name: '大图：folderB/one.png' });
    expect(apiClient.put).not.toHaveBeenCalled();
  });

  it('does not notify a departed workspace when an outstanding save completes', async () => {
    let resolve: () => void = () => {};
    vi.mocked(apiClient.put).mockImplementation(() => new Promise<void>(done => { resolve = done; }) as any);
    const view = show(); await screen.findByRole('img', { name: '大图：folderA/one.png' }); addPending();
    fireEvent.click(screen.getByRole('button', { name: '保存标签' }));
    await waitFor(() => expect(apiClient.put).toHaveBeenCalledOnce());
    view.unmount(); await act(async () => { resolve(); });
    expect(view.onChanged).not.toHaveBeenCalled();
  });
});
