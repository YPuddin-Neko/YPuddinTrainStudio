import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { MemoryRouter, Route, Routes, useLocation, useParams } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { apiClient } from '../src/api/client';
import { handlers } from '../src/mocks/handlers';
import TrainConfig from '../src/pages/TrainConfig/TrainConfig';
import trainSchema from '../src/schema/train-schema.json';
import { schemaDefaults } from '../src/utils/config';
import { normalizeDatasetPath } from '../src/utils/workspaceConfig';
import i18n from '../src/i18n';

const server = setupServer(...handlers);
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
beforeEach(async () => { await i18n.changeLanguage('zh-CN'); });
afterEach(() => { server.resetHandlers(); vi.restoreAllMocks(); });
afterAll(() => server.close());

function Destination() {
  const { pathname, search } = useLocation();
  const { id } = useParams();
  const [epochs, setEpochs] = React.useState<number | null>(null);
  React.useEffect(() => {
    if (pathname.startsWith('/projects/')) void apiClient.get<any>(`/projects/${id}/config`).then(config => setEpochs(config.loop.epochs));
  }, [id, pathname]);
  return <div>Destination {pathname}{search}<span>Saved epochs: {epochs}</span></div>;
}

function show() {
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <MemoryRouter initialEntries={['/projects/p_nav/train']}><Routes>
      <Route path="/projects/:id/train" element={<TrainConfig />} />
      <Route path="/projects/:id" element={<Destination />} />
      <Route path="/datasets/:id" element={<Destination />} />
    </Routes></MemoryRouter>
  </QueryClientProvider>);
}

function fixtures(paths = ['d:\\TRAINING\\B\\']) {
  const state = { config: schemaDefaults(trainSchema) };
  state.config.loop.epochs = 2;
  state.config.dataset.sources = paths.map(path => ({ path, repeats: 1 }));
  server.use(
    http.get('/api/projects/p_nav', () => HttpResponse.json({ id: 'p_nav', name: 'Navigation test' })),
    http.get('/api/projects/p_nav/config', () => HttpResponse.json(state.config)),
    http.put('/api/projects/p_nav/config', async ({ request }) => { state.config = await request.json(); return HttpResponse.json(state.config); }),
    http.get('/api/projects/p_nav/datasets', () => HttpResponse.json(['A', 'B'].map(name => ({ source: { id: `d_${name}`, project_id: 'p_nav', path: `D:/training/${name}` }, index_status: 'ready' })))),
  );
  return state;
}

describe('training dataset destinations and saved navigation', () => {
  it('opens only current source B even when registered A is first, with Windows spelling normalized', async () => {
    fixtures(); show();
    await screen.findByRole('spinbutton', { name: 'loop.epochs' });
    fireEvent.click(screen.getByRole('tab', { name: '数据与分桶' }));
    const link = await screen.findByRole('link', { name: '标签与遮罩编辑' });
    expect(link).toHaveAttribute('href', '/datasets/d_B');
    fireEvent.click(link);
    await screen.findByText('Destination /datasets/d_B');
  });

  it.each([
    { paths: ['D:/outside'], label: '导入数据后编辑标签与遮罩' },
    { paths: ['D:/training/A', 'D:/training/B'], label: '选择数据集编辑标签与遮罩' },
  ])('uses project source selection when a direct editor would be ambiguous: $label', async ({ paths, label }) => {
    fixtures(paths); show();
    await screen.findByRole('spinbutton', { name: 'loop.epochs' });
    fireEvent.click(screen.getByRole('tab', { name: '数据与分桶' }));
    expect(await screen.findByRole('link', { name: label })).toHaveAttribute('href', '/projects/p_nav?step=data');
    expect(screen.queryByRole('link', { name: '标签与遮罩编辑' })).not.toBeInTheDocument();
  });

  it('normalizes Windows and UNC casing without merging distinct POSIX paths', () => {
    expect(normalizeDatasetPath(' D:\\Photos\\B\\ ')).toBe(normalizeDatasetPath('d:/photos/b'));
    expect(normalizeDatasetPath('\\\\SERVER\\Share\\Photos\\')).toBe(normalizeDatasetPath('//server/share/photos'));
    expect(normalizeDatasetPath('/photos/B')).not.toBe(normalizeDatasetPath('/photos/b'));
  });

  it('waits for an older autosave and saves edits made during navigation before the destination reads config', async () => {
    const state = fixtures();
    const writes: { config: any; finish: () => void }[] = [];
    server.use(http.put('/api/projects/p_nav/config', async ({ request }) => {
      const config = await request.json();
      await new Promise<void>(resolve => writes.push({ config, finish: resolve }));
      state.config = config;
      return HttpResponse.json(config);
    }));
    show();
    const epochs = await screen.findByRole('spinbutton', { name: 'loop.epochs' });
    fireEvent.change(epochs, { target: { value: '7' } });
    await waitFor(() => expect(writes).toHaveLength(1), { timeout: 2000 });
    fireEvent.change(epochs, { target: { value: '9' } });
    fireEvent.click(screen.getByRole('link', { name: /模型准备/ }));
    expect(await screen.findByText('正在保存草稿…')).toBeInTheDocument();
    expect(screen.queryByText(/Destination/)).not.toBeInTheDocument();
    expect(writes).toHaveLength(1);
    writes[0].finish();
    await waitFor(() => expect(writes).toHaveLength(2));
    expect(writes[1].config.loop.epochs).toBe(9);
    fireEvent.change(epochs, { target: { value: '11' } });
    writes[1].finish();
    await waitFor(() => expect(writes).toHaveLength(3));
    expect(writes[2].config.loop.epochs).toBe(11);
    expect(screen.queryByText(/Destination/)).not.toBeInTheDocument();
    writes[2].finish();
    await screen.findByText('Destination /projects/p_nav?step=models');
    await screen.findByText('Saved epochs: 11');
    expect(state.config.loop.epochs).toBe(11);
  });

  it('stays on the editable draft when saving fails and allows retrying the same navigation', async () => {
    const state = fixtures();
    let fail = true;
    server.use(http.put('/api/projects/p_nav/config', async ({ request }) => {
      if (fail) return HttpResponse.json({ error: { code: 'storage.error', message: 'Disk is full' } }, { status: 500 });
      state.config = await request.json(); return HttpResponse.json(state.config);
    }));
    show();
    const epochs = await screen.findByRole('spinbutton', { name: 'loop.epochs' });
    fireEvent.change(epochs, { target: { value: '13' } });
    fireEvent.click(screen.getByRole('link', { name: /模型准备/ }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Disk is full');
    expect(screen.getByRole('alert')).toHaveTextContent('已留在当前页面');
    expect(epochs).toHaveValue(13);
    expect(screen.queryByText(/Destination/)).not.toBeInTheDocument();
    fail = false;
    fireEvent.click(screen.getByRole('link', { name: /模型准备/ }));
    await screen.findByText('Saved epochs: 13');
  });

  it('does not intercept modified or new-tab link clicks', async () => {
    fixtures(); show();
    const epochs = await screen.findByRole('spinbutton', { name: 'loop.epochs' });
    fireEvent.change(epochs, { target: { value: '17' } });
    const link = screen.getByRole('link', { name: /模型准备/ });
    const prevented: boolean[] = [];
    // Observe after React's capture handler, then suppress jsdom's real navigation.
    link.addEventListener('click', event => { prevented.push(event.defaultPrevented); event.preventDefault(); });
    for (const modifier of ['ctrlKey', 'metaKey', 'shiftKey', 'altKey']) fireEvent.click(link, { [modifier]: true });
    link.setAttribute('target', '_blank');
    fireEvent.click(link);
    expect(prevented).toEqual([false, false, false, false, false]);
    expect(screen.queryByText('正在保存草稿…')).not.toBeInTheDocument();
  });
});
