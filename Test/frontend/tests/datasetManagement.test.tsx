import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterAll, afterEach, beforeAll, expect, it, vi } from 'vitest';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import Dataset from '../../../frontend/src/pages/Dataset/Dataset';
import { handlers } from '../mocks/handlers';
import i18n from '../../../frontend/src/i18n';

vi.mock('../../../frontend/src/events/useEventStream', () => ({ useEventStream: () => {} }));
const server = setupServer(...handlers);
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());
function Location() { return <output data-testid="route">{useLocation().pathname}</output>; }

it('holds out and restores a specific duplicate file, and saves folder settings without navigation', async () => {
  await i18n.changeLanguage('zh-CN');
  const excluded = new Set<string>();
  const writes: unknown[] = [];
  const version = { id: 'v1', project_id: 'p_manage', name: 'v1', status: 'ready', archived: false, busy: false, paths: { root: '/data' }, stats: {} };
  const source = { id: 'd_manage', project_id: 'p_manage', version_id: 'v1', path: '/data/images', repeats: 1, caption_ext: 'auto', can_rename: true };
  let masked = false;
  const info = () => ({ source, masked_loss: masked, stats: { images: 2, captioned: 2, masks: 2, training_images: 2 - excluded.size, held_out_images: excluded.size }, index_status: 'ready', cache: {} });
  const items = ['a.png', 'b.png'].map(rel_path => ({ hash: 'same-content', rel_path, width: 64, height: 64, has_mask: true, caption: 'edited caption', caption_format: 'txt' }));
  server.use(
    http.get('/api/datasets/d_manage', () => HttpResponse.json(info())),
    http.get('/api/datasets/d_manage/images', ({ request }) => {
      const membership = new URL(request.url).searchParams.get('membership');
      const visible = items.filter(i => membership === 'all' || excluded.has(i.rel_path) === (membership === 'unused'));
      return HttpResponse.json({ items: visible.map(i => ({ ...i, training_enabled: !excluded.has(i.rel_path) })), total: visible.length, page: 1, page_size: 60 });
    }),
    http.get('/api/projects/p_manage', () => HttpResponse.json({ id: 'p_manage', name: 'Manage', active_version_id: 'v1' })),
    http.get('/api/projects/p_manage/versions', () => HttpResponse.json([version])),
    http.get('/api/projects/p_manage/versions/v1', () => HttpResponse.json(version)),
    http.post('/api/datasets/d_manage/membership', async ({ request }) => {
      const body = await request.json() as { paths: string[]; included: boolean }; writes.push(body);
      body.paths.forEach(path => body.included ? excluded.delete(path) : excluded.add(path));
      return HttpResponse.json({ changed: body.paths.length });
    }),
    http.patch('/api/datasets/d_manage', async ({ request }) => {
      const body = await request.json() as { name?: string; repeats?: number; masked_loss?: boolean }; writes.push(body);
      if (body.name) source.path = '/data/' + body.name;
      if (body.repeats) source.repeats = body.repeats;
      if (body.masked_loss !== undefined) masked = body.masked_loss;
      return HttpResponse.json(info());
    }),
  );
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><MemoryRouter initialEntries={['/datasets/d_manage']}><Routes><Route path="/datasets/:id" element={<Dataset/>}/></Routes><Location/></MemoryRouter></QueryClientProvider>);
  await waitFor(() => expect(screen.getByRole('button', { name: '选择图片: a.png' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: '选择图片: a.png' }));
  expect(screen.getByRole('button', { name: '选择图片: b.png' })).toHaveAttribute('aria-pressed', 'false');
  fireEvent.click(screen.getByRole('button', { name: '暂时移出训练' }));
  await waitFor(() => expect(screen.queryByRole('button', { name: '选择图片: a.png' })).not.toBeInTheDocument());
  expect(writes).toEqual([{ paths: ['a.png'], included: false }]);
  fireEvent.click(screen.getByRole('tab', { name: /暂不参与/ }));
  fireEvent.click(await screen.findByRole('button', { name: '选择图片: a.png' }));
  expect(screen.getByText('64 × 64 · edited caption')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: '编辑遮罩 · 已有文件' })).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: '加入训练' }));
  await waitFor(() => expect(excluded.size).toBe(0));
  fireEvent.click(screen.getByRole('tab', { name: /参与训练/ }));
  await screen.findByRole('button', { name: '选择图片: a.png' });
  fireEvent.change(screen.getByRole('textbox', { name: '文件夹名称' }), { target: { value: 'portraits' } });
  fireEvent.change(screen.getByRole('spinbutton', { name: '每轮重复次数' }), { target: { value: '4' } });
  fireEvent.click(screen.getByRole('button', { name: '保存目录设置' }));
  await screen.findByRole('heading', { name: 'portraits' });
  expect(source.repeats).toBe(4);
  fireEvent.click(screen.getByRole('switch', { name: '使用遮罩训练' }));
  await waitFor(() => expect(screen.getByRole('switch', { name: '使用遮罩训练' })).toBeChecked());
  expect(screen.getByTestId('route')).toHaveTextContent('/datasets/d_manage');
  expect(writes).toEqual([{ paths: ['a.png'], included: false }, { paths: ['a.png'], included: true }, { name: 'portraits', repeats: 4 }, { masked_loss: true }]);
  expect(screen.queryByRole('button', { name: '分布与分桶' })).not.toBeInTheDocument();
});
