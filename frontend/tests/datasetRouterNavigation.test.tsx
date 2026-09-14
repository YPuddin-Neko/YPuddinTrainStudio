import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { createMemoryRouter, RouterProvider } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import AppRoutes from '../src/router';
import { handlers } from '../src/mocks/handlers';
import * as maskApi from '../src/components/masks/maskApi';
import i18n from '../src/i18n';

const { updateCaption } = vi.hoisted(() => ({ updateCaption: vi.fn() }));
vi.mock('../src/events/useEventStream', () => ({ useEventStream: () => {}, useEventStreamStatus: () => 'connected' }));
vi.mock('../src/components/masks/maskApi', async original => ({ ...(await original<typeof maskApi>()), loadMask: vi.fn(), saveMask: vi.fn() }));
vi.mock('../src/api/hooks/useDatasetImages', () => ({ useDatasetImages: () => ({
  items: [{ hash: 'image1', rel_path: 'photo.png', width: 8, height: 8, has_mask: false, caption: 'cat' }], total: 1, q: '', selected: new Set(), loading: false, error: null,
  refresh: vi.fn(), loadMore: vi.fn(), setQ: vi.fn(), selectAll: vi.fn(), clearSelection: vi.fn(), toggleSelect: vi.fn(), updateCaption,
}) }));
const server = setupServer(...handlers);
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterAll(() => server.close());
beforeEach(async () => {
  await i18n.changeLanguage('zh-CN'); updateCaption.mockClear();
  vi.mocked(maskApi.loadMask).mockResolvedValue({ info: { source: 'alpha', width: 8, height: 8, coverage: 128 / 255, has_mask: false, filename: 'photo.mask.png', revision: 'alpha:1:100', resized: false }, pixels: new Uint8Array(64).fill(128) });
  vi.mocked(maskApi.saveMask).mockImplementation(async (_did, _hash, _path, info) => ({ ...info, source: 'sidecar', has_mask: true, revision: 'sidecar:2:90' }));
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(() => ({ createImageData: (w: number, h: number) => ({ data: new Uint8ClampedArray(w * h * 4) }), putImageData: vi.fn() }) as any);
});
afterEach(() => { server.resetHandlers(); vi.restoreAllMocks(); vi.clearAllMocks(); });

function show() {
  const version = { id: 'v2', name: 'Version two', project_id: 'p_dataset', status: 'ready', archived: false, busy: false, number: 2, paths: { root: 'D:/project/v2' }, stats: { images: 1, datasets: 1, jobs: 0, artifacts: 0 } };
  server.use(
    http.get('/api/datasets/d_known', () => HttpResponse.json({ source: { id: 'd_known', project_id: 'p_dataset', version_id: 'v2', path: 'D:/project/v2/traindata/photos', repeats: 1, caption_ext: '.txt' }, index_status: 'ready', stats: { images: 1, captioned: 1, masks: 0 }, cache: {} })),
    http.get('/api/projects/p_dataset', () => HttpResponse.json({ id: 'p_dataset', name: 'Dataset owner', active_version_id: 'v2', layout_version: 2 })),
    http.get('/api/projects/p_dataset/versions', () => HttpResponse.json([version])),
    http.get('/api/projects/p_dataset/versions/v2', () => HttpResponse.json(version)),
    http.get('/api/projects/p_dataset/config', () => HttpResponse.json({ dataset: { masked_loss: false } })),
    http.put('/api/projects/p_dataset/config', async ({request}) => HttpResponse.json(await request.json())),
  );
  const router = createMemoryRouter([{ path: '*', element: <AppRoutes/> }], { initialEntries: ['/queue', '/datasets/d_known', '/queue?status=failed'], initialIndex: 1 });
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><RouterProvider router={router}/></QueryClientProvider>);
  return router;
}
async function editCaption() {
  fireEvent.click(await screen.findByRole('button', { name: '编辑标签: photo.png' }));
  fireEvent.change(screen.getByTestId('tag-add-input'), { target: { value: 'blue eyes' } });
  fireEvent.keyDown(screen.getByTestId('tag-add-input'), { key: 'Enter' });
}

describe('dataset drafts survive actual data-router history navigation', () => {
  it.each([-1, 1])('blocks history direction %s on a caption save failure and waits for a successful retry', async direction => {
    let fail = true; let release = () => {}; const writes: unknown[] = [];
    server.use(http.put('/api/datasets/d_known/images/image1/caption', async ({request}) => {
      writes.push(await request.json());
      if (fail) return HttpResponse.json({error:{code:'caption.conflict',message:'Caption changed on disk'}}, {status:409});
      await new Promise<void>(resolve => { release = resolve; }); return HttpResponse.json({ok:true});
    }));
    const router = show(); await editCaption();
    const editor = screen.getByRole('dialog', {name:'编辑图片标签'});
    await act(async () => { await router.navigate(direction); });
    expect(await within(editor).findByRole('alert')).toHaveTextContent('Caption changed on disk');
    expect(router.state.location.pathname).toBe('/datasets/d_known');
    expect(screen.getByTestId('tag-chip-1')).toHaveTextContent('blue eyes'); expect(updateCaption).not.toHaveBeenCalled();
    fail = false; await act(async () => { await router.navigate(direction); });
    await waitFor(() => expect(writes).toHaveLength(2));
    expect(router.state.location.pathname).toBe('/datasets/d_known'); expect(screen.getByRole('dialog', {name:'编辑图片标签'})).toBe(editor);
    await act(async () => release());
    await waitFor(() => expect(router.state.location.pathname).toBe('/queue'));
    expect(router.state.location.search).toBe(direction === 1 ? '?status=failed' : '');
    expect(writes).toEqual([{caption:'cat, blue eyes'}, {caption:'cat, blue eyes'}]);
    expect(updateCaption).toHaveBeenCalledExactlyOnceWith('image1','cat, blue eyes','photo.png');
  });

  it('keeps the actual mask pixels and undo history through Back and Forward, then allows its saved training action', async () => {
    const router = show(); fireEvent.click(await screen.findByRole('button', {name:'编辑遮罩'}));
    fireEvent.click(await screen.findByRole('button', {name:'清空 · 全黑'}));
    const canvas = screen.getByLabelText('遮罩绘制画布');
    for (const direction of [-1, 1]) {
      await act(async () => { await router.navigate(direction); });
      await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('请先在遮罩编辑器中保存或关闭'));
      expect(router.state.location.pathname).toBe('/datasets/d_known'); expect(screen.getByLabelText('遮罩绘制画布')).toBe(canvas);
      expect(screen.getByText('当前为全黑，这张图片不会贡献训练损失。')).toBeInTheDocument();
      expect(screen.getByRole('button', {name:'撤销'})).toBeEnabled();
    }
    fireEvent.click(screen.getByRole('button', {name:'撤销'})); expect(screen.queryByText('当前为全黑，这张图片不会贡献训练损失。')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', {name:'重做'}));
    fireEvent.click(screen.getByRole('button', {name:'保存并启用遮罩训练'}));
    await waitFor(() => expect(router.state.location.pathname).toBe('/projects/p_dataset/v/v2/train'));
    expect(maskApi.saveMask).toHaveBeenCalledOnce();
    expect(vi.mocked(maskApi.saveMask).mock.calls[0][4].every(value => value === 0)).toBe(true);
  });

  it('preserves the real settings background route after its blocker saves a caption', async () => {
    const writes: unknown[] = [];
    server.use(http.put('/api/datasets/d_known/images/image1/caption', async ({request}) => {writes.push(await request.json()); return HttpResponse.json({ok:true});}));
    const router = show(); await editCaption(); const page = screen.getByTestId('dataset-page');
    fireEvent.click(within(screen.getByRole('complementary')).getByRole('link', {name:'系统设置'}));
    const drawer = await screen.findByRole('dialog', {name:'系统设置'});
    expect(router.state.location.state.backgroundLocation.pathname).toBe('/datasets/d_known');
    expect(screen.getByTestId('dataset-page')).toBe(page); expect(writes).toEqual([{caption:'cat, blue eyes'}]);
    fireEvent.click(within(drawer).getByRole('button', {name:'关闭设置，返回工作区'}));
    await waitFor(() => expect(router.state.location.pathname).toBe('/datasets/d_known'));
    expect(screen.getByTestId('dataset-page')).toBe(page);
  });
});
