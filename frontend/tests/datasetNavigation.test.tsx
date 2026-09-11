import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter, Route, Routes, useLocation, useNavigate, type Location as RouterLocation } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import Layout from '../src/components/Layout';
import SettingsDrawer from '../src/components/SettingsDrawer';
import Dataset from '../src/pages/Dataset/Dataset';
import { handlers } from '../src/mocks/handlers';
import i18n from '../src/i18n';

const { updateCaption } = vi.hoisted(() => ({ updateCaption: vi.fn() }));
vi.mock('../src/events/useEventStream', () => ({ useEventStream: () => {}, useEventStreamStatus: () => 'connected' }));
vi.mock('../src/api/hooks/useDatasetImages', () => ({ useDatasetImages: () => ({
  items: [{ hash: 'image1', rel_path: 'photo.png', width: 64, height: 64, has_mask: false, caption: 'cat' }],
  total: 1, q: '', selected: new Set(), loading: false, error: null,
  refresh: vi.fn(), loadMore: vi.fn(), setQ: vi.fn(), selectAll: vi.fn(), clearSelection: vi.fn(), toggleSelect: vi.fn(), updateCaption,
}) }));
vi.mock('../src/components/masks/MaskEditor', () => ({ MaskEditor: ({ onClose }: { onClose: () => void }) => <div role="dialog" aria-label="Mask draft"><button onClick={onClose}>Close mask draft</button></div> }));
const server = setupServer(...handlers);
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterAll(() => server.close());
beforeEach(async () => { await i18n.changeLanguage('zh-CN'); updateCaption.mockClear(); });
afterEach(() => { server.resetHandlers(); vi.restoreAllMocks(); });

function fixture() {
  const versions = ['v1', 'v2'].map((id, index) => ({ id, name: id, project_id: 'p_dataset', status: 'ready', archived: false, busy: false, number: index + 1, paths: { root: `D:/project/${id}` }, stats: { images: 1, datasets: 1, jobs: 0, artifacts: 0 } }));
  server.use(
    http.get('/api/datasets/d_known', () => HttpResponse.json({ source: { id: 'd_known', project_id: 'p_dataset', version_id: 'v2', path: 'D:/project/v2/traindata/photos', repeats: 1, caption_ext: '.txt' }, index_status: 'ready', stats: { images: 1, captioned: 1, masks: 0 }, cache: {} })),
    http.get('/api/projects/p_dataset', () => HttpResponse.json({ id: 'p_dataset', name: 'Dataset owner', active_version_id: 'v2', layout_version: 2 })),
    http.get('/api/projects/p_dataset/versions', () => HttpResponse.json(versions)),
    http.get('/api/projects/p_dataset/versions/:vid', ({ params }) => HttpResponse.json(versions.find(version => version.id === params.vid))),
  );
}
function Location() {
  const location = useLocation(); const navigate = useNavigate();
  return <><output data-testid="route">{location.pathname}{location.search}</output><button onClick={() => navigate('/datasets/d_missing')}>Open missing dataset</button></>;
}
function show() {
  fixture();
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><MemoryRouter initialEntries={['/datasets/d_known']}><WorkspaceRoutes/><Location/></MemoryRouter></QueryClientProvider>);
}
function WorkspaceRoutes() {
  const location = useLocation(); const navigate = useNavigate();
  const background = (location.state as {backgroundLocation?:RouterLocation}|null)?.backgroundLocation;
  return <><Routes location={background || location}><Route element={<Layout/>}>
    <Route path="/datasets/:id" element={<Dataset/>}/><Route path="/projects/:id/v/:versionId" element={<p>Version destination</p>}/>
  </Route></Routes>{background && <SettingsDrawer onClose={() => navigate(`${background.pathname}${background.search}`, {replace:true})}><p>Preferences fixture</p></SettingsDrawer>}</>;
}
async function editCaption() {
  fireEvent.click(await screen.findByRole('button', { name: '编辑标签: photo.png' }));
  const input = screen.getByTestId('tag-add-input');
  fireEvent.change(input, { target: { value: 'blue eyes' } }); fireEvent.keyDown(input, { key: 'Enter' });
  expect(screen.getByTestId('tag-chip-1')).toHaveTextContent('blue eyes');
}

describe('dataset project sidebar and navigation protection', () => {
  it('uses the dataset owner/version and waits for its caption PUT before switching versions', async () => {
    let release = () => {}; const writes: unknown[] = [];
    server.use(http.put('/api/datasets/d_known/images/image1/caption', async ({ request }) => {
      writes.push(await request.json()); await new Promise<void>(resolve => { release = resolve; }); return HttpResponse.json({ ok: true });
    }));
    show();
    const slot = screen.getByTestId('project-sidebar-slot');
    expect(await within(slot).findByRole('link', { name: /Dataset owner/ })).toBeInTheDocument();
    expect(within(slot).getByRole('combobox', { name: '项目版本' })).toHaveTextContent('v2');
    const body = within(screen.getByTestId('app-page-frame'));
    expect(body.queryByRole('navigation', { name: '项目训练步骤' })).not.toBeInTheDocument();
    expect(body.getAllByRole('heading')).toHaveLength(1); expect(body.getByRole('heading', { name: 'photos' })).toBeInTheDocument();
    expect(within(screen.getByTestId('dataset-overview')).getByText('已有标签')).toBeInTheDocument();
    expect(within(screen.getByTestId('dataset-overview')).getByRole('button', { name: '分布与分桶' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '计算分桶' })).not.toBeInTheDocument();
    await editCaption();
    fireEvent.click(screen.getByRole('combobox', { name: '项目版本' })); fireEvent.click(screen.getByRole('option', { name: 'v1' }));
    await waitFor(() => expect(writes).toEqual([{ caption: 'cat, blue eyes' }]));
    expect(screen.getByTestId('route')).toHaveTextContent('/datasets/d_known');
    expect(screen.getByRole('dialog', { name: '编辑图片标签' })).toBeInTheDocument();
    await act(async () => release());
    await screen.findByText('Version destination');
    expect(screen.getByTestId('route')).toHaveTextContent('/projects/p_dataset/v/v1');
    expect(updateCaption).toHaveBeenCalledWith('image1', 'cat, blue eyes');
  });

  it('keeps the caption draft and dataset route on a failed save, then retries before following a stage link', async () => {
    let fail = true; const writes: unknown[] = [];
    server.use(http.put('/api/datasets/d_known/images/image1/caption', async ({ request }) => {
      writes.push(await request.json()); return fail ? HttpResponse.json({ error: { code: 'caption.conflict', message: 'Caption changed on disk' } }, { status: 409 }) : HttpResponse.json({ ok: true });
    }));
    show(); await screen.findByRole('combobox', { name: '项目版本' }); await editCaption();
    const stages = screen.getByRole('navigation', { name: '项目训练步骤' });
    fireEvent.click(within(stages).getByRole('link', { name: /^2\s*模型准备$/ }));
    expect(await within(screen.getByRole('dialog', { name: '编辑图片标签' })).findByRole('alert')).toHaveTextContent('Caption changed on disk');
    expect(screen.getByTestId('route')).toHaveTextContent('/datasets/d_known');
    expect(screen.getByTestId('tag-chip-1')).toHaveTextContent('blue eyes'); expect(updateCaption).not.toHaveBeenCalled();
    fail = false; fireEvent.click(within(stages).getByRole('link', { name: /^2\s*模型准备$/ }));
    await screen.findByText('Version destination'); expect(screen.getByTestId('route')).toHaveTextContent('/projects/p_dataset/v/v2?step=models');
    expect(writes).toEqual([{ caption: 'cat, blue eyes' }, { caption: 'cat, blue eyes' }]);
  });

  it('requires closing or saving the mask editor before using sidebar navigation', async () => {
    show(); await screen.findByRole('combobox', { name: '项目版本' });
    fireEvent.click(await screen.findByRole('button', { name: '编辑遮罩' }));
    fireEvent.click(within(screen.getByRole('navigation', { name: '项目训练步骤' })).getByRole('link', { name: /^2\s*模型准备$/ }));
    expect(await screen.findByRole('alert')).toHaveTextContent('请先在遮罩编辑器中保存或关闭');
    expect(screen.getByTestId('route')).toHaveTextContent('/datasets/d_known'); expect(screen.getByRole('dialog', { name: 'Mask draft' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Close mask draft' }));
    fireEvent.click(within(screen.getByRole('navigation', { name: '项目训练步骤' })).getByRole('link', { name: /^2\s*模型准备$/ }));
    await screen.findByText('Version destination');
  });

  it('saves a dirty caption before opening settings as a drawer with the dataset still mounted behind it', async () => {
    let release = () => {}; const writes: unknown[] = [];
    server.use(http.put('/api/datasets/d_known/images/image1/caption', async ({ request }) => {
      writes.push(await request.json()); await new Promise<void>(resolve => { release = resolve; }); return HttpResponse.json({ ok: true });
    }));
    show(); await screen.findByRole('combobox', { name: '项目版本' }); await editCaption();
    const page = screen.getByTestId('dataset-page');
    const controls = screen.getByRole('region', { name: '当前项目工作区' });
    fireEvent.click(within(screen.getByRole('navigation', { name: '主导航' })).getByRole('link', { name: '系统设置' }));
    await waitFor(() => expect(writes).toEqual([{ caption: 'cat, blue eyes' }]));
    expect(screen.getByTestId('route')).toHaveTextContent('/datasets/d_known');
    expect(screen.queryByRole('dialog', { name: '系统设置' })).not.toBeInTheDocument();
    await act(async () => release());
    const drawer = await screen.findByRole('dialog', { name: '系统设置' });
    expect(screen.getByTestId('route')).toHaveTextContent('/settings');
    expect(screen.getByTestId('dataset-page')).toBe(page); expect(screen.getByTestId('project-sidebar-slot')).toContainElement(controls);
    expect(screen.queryByRole('dialog', { name: '编辑图片标签' })).not.toBeInTheDocument();
    fireEvent.click(within(drawer).getByRole('button', { name: '关闭设置，返回工作区' }));
    expect(screen.getByTestId('route')).toHaveTextContent('/datasets/d_known'); expect(screen.getByTestId('dataset-page')).toBe(page);
    expect(updateCaption).toHaveBeenCalledExactlyOnceWith('image1', 'cat, blue eyes');
  });

  it('clears the old sidebar as soon as a different dataset opens and keeps it empty if metadata fails', async () => {
    let release = () => {}; let requested = false;
    server.use(http.get('/api/datasets/d_missing', async () => { requested = true; await new Promise<void>(resolve => { release = resolve; }); return HttpResponse.json({ error: { code: 'dataset.not_found', message: 'Dataset missing' } }, { status: 404 }); }));
    show(); await screen.findByRole('combobox', { name: '项目版本' });
    const slot = screen.getByTestId('project-sidebar-slot'); fireEvent.click(screen.getByRole('button', { name: 'Open missing dataset' }));
    await waitFor(() => expect(requested).toBe(true)); expect(slot).toBeEmptyDOMElement();
    await act(async () => release());
    expect(await screen.findByRole('alert')).toHaveTextContent('Dataset missing');
    expect(slot).toBeEmptyDOMElement(); expect(screen.queryByRole('combobox', { name: '项目版本' })).not.toBeInTheDocument();
  });
});
