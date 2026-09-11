import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { handlers } from '../src/mocks/handlers';
import ProjectWorkspaceHeader from '../src/components/projects/ProjectWorkspaceHeader';
import TrainConfig from '../src/pages/TrainConfig/TrainConfig';
import Dataset from '../src/pages/Dataset/Dataset';
import { apiClient } from '../src/api/client';
import { schemaDefaults } from '../src/utils/config';
import type { ProjectVersion, VersionedProject } from '../src/utils/projectVersions';
import trainSchema from '../src/schema/train-schema.json';
import i18n from '../src/i18n';

vi.mock('../src/events/useEventStream', () => ({ useEventStream: () => {} }));
vi.mock('../src/api/hooks/useDatasetImages', () => ({ useDatasetImages: () => ({ items: [], total: 0, q: '', selected: new Set(), loading: false, error: null, refresh: vi.fn(), loadMore: vi.fn(), setQ: vi.fn(), selectAll: vi.fn(), clearSelection: vi.fn(), toggleSelect: vi.fn() }) }));
const server = setupServer(...handlers);
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
beforeEach(async () => { await i18n.changeLanguage('zh-CN'); });
afterEach(() => { server.resetHandlers(); vi.restoreAllMocks(); });
afterAll(() => server.close());

const project = { id: 'p_versions', name: 'Version scope test', active_version_id: 'v1', version_count: 2, dataset_ids: [], note: '', archived: false, created_at: 1, updated_at: 1, stats: { jobs: 0, artifacts: 0 } } as VersionedProject;
const version = (id: string): ProjectVersion => ({ id, project_id: project.id, name: id, note: '', status: 'ready', archived: false, created_at: 1, updated_at: 1, dataset_ids: [], stats: { datasets: 0, images: 0, jobs: 0, artifacts: 0 }, paths: { root: `D:/versions/${id}`, config: `D:/versions/${id}/config.toml`, datasets: `D:/versions/${id}/datasets`, runs: `D:/versions/${id}/runs`, cache: `D:/versions/${id}/cache` } });
const versions = [version('v1'), version('v2')];
function Location() { const location = useLocation(); return <output data-testid="location">{location.pathname}{location.search}</output>; }
function wrap(children: React.ReactNode, path = '/projects/p_versions/v/v1/train') {
  return <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><MemoryRouter initialEntries={[path]}>{children}<Location/></MemoryRouter></QueryClientProvider>;
}
function trainingFixture() {
  const configs = { v1: schemaDefaults(trainSchema), v2: schemaDefaults(trainSchema) };
  configs.v1.loop.epochs = 2; configs.v2.loop.epochs = 30;
  const reads: string[] = [], writes: { version: string; config: any }[] = [], datasetScopes: string[] = [];
  server.use(
    http.get('/api/projects/p_versions', () => HttpResponse.json(project)),
    http.get('/api/projects/p_versions/versions', () => HttpResponse.json(versions)),
    http.get('/api/projects/p_versions/versions/:vid', ({ params }) => HttpResponse.json(versions.find(item => item.id === params.vid))),
    http.patch('/api/projects/p_versions', async ({ request }) => HttpResponse.json({ ...project, ...await request.json() as object })),
    http.get('/api/projects/p_versions/config', ({ request }) => {
      const scope = new URL(request.url).searchParams.get('version_id') || ''; reads.push(scope);
      return scope in configs ? HttpResponse.json(configs[scope as keyof typeof configs]) : HttpResponse.json({ error: { message: 'Missing explicit version' } }, { status: 400 });
    }),
    http.put('/api/projects/p_versions/config', async ({ request }) => {
      const scope = new URL(request.url).searchParams.get('version_id') || ''; const config = await request.json(); writes.push({ version: scope, config });
      if (!(scope in configs)) return HttpResponse.json({ error: { message: 'Missing explicit version' } }, { status: 400 });
      configs[scope as keyof typeof configs] = config; return HttpResponse.json(config);
    }),
    http.get('/api/projects/p_versions/datasets', ({ request }) => { datasetScopes.push(new URL(request.url).searchParams.get('version_id') || ''); return HttpResponse.json([]); }),
    http.post('/api/plan', () => HttpResponse.json({ ok: true, errors: [], warnings: [], total_steps: 10, steps_per_epoch: 5 })),
  );
  return { configs, reads, writes, datasetScopes };
}

describe('explicit project version actions', () => {
  it('flushes a pending draft before opening create, then copies the saved source using the real version endpoint', async () => {
    const calls: string[] = []; let finishSave: () => void = () => {}; let created: any;
    server.use(
      http.put('/api/projects/p_versions/config', async ({ request }) => {
        expect(new URL(request.url).searchParams.get('version_id')).toBe('v1');
        expect(await request.json()).toEqual({ optimizer: { lr: 0.0007 } }); calls.push('save-start');
        await new Promise<void>(resolve => { finishSave = resolve; }); calls.push('save-finished'); return HttpResponse.json({});
      }),
      http.post('/api/projects/p_versions/versions', async ({ request }) => { calls.push('create'); created = await request.json(); return HttpResponse.json(version('v3')); }),
    );
    const flush = vi.fn(async () => { await apiClient.put('/projects/p_versions/config?version_id=v1', { optimizer: { lr: 0.0007 } }); });
    const refresh = vi.fn().mockResolvedValue(undefined);
    render(wrap(<ProjectWorkspaceHeader project={project} versionId="v1" versions={versions} current={versions[0]} active="train" refresh={refresh} beforeAction={flush}/>));
    fireEvent.click(screen.getByRole('button', { name: '新版本' }));
    await waitFor(() => expect(calls).toEqual(['save-start']));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: '新版本' })).toBeDisabled();
    expect(screen.getByTestId('location')).toHaveTextContent('/projects/p_versions/v/v1/train');
    await act(async () => finishSave());
    const dialog = await screen.findByRole('dialog', { name: '新建实验版本' });
    fireEvent.change(within(dialog).getByRole('textbox', { name: '版本名称' }), { target: { value: 'Clothing experiment' } });
    fireEvent.click(within(dialog).getByRole('button', { name: '创建版本' }));
    await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/projects/p_versions/v/v3'));
    expect(calls).toEqual(['save-start', 'save-finished', 'create']);
    expect(created).toEqual({ name: 'Clothing experiment', note: '', source_version_id: 'v1', data_mode: 'copy', copy_config: true });
    expect(flush).toHaveBeenCalledOnce(); expect(refresh).toHaveBeenCalledOnce();
  });

  it('keeps the creation dialog closed and exposes the reason if draft flushing fails', async () => {
    const flush = vi.fn().mockRejectedValueOnce(new Error('Version draft disk is full')).mockResolvedValue(undefined);
    render(wrap(<ProjectWorkspaceHeader project={project} versionId="v1" versions={versions} current={versions[0]} active="train" refresh={vi.fn().mockResolvedValue(undefined)} beforeAction={flush}/>));
    fireEvent.click(screen.getByRole('button', { name: '新版本' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Version draft disk is full');
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '新版本' }));
    expect(await screen.findByRole('dialog', { name: '新建实验版本' })).toBeInTheDocument();
  });

  it.each([{ active: 'train' as const, suffix: '/train' }, { active: 'results' as const, suffix: '?step=results' }])('switches versions using an explicit path while preserving $active', ({ active, suffix }) => {
    render(wrap(<ProjectWorkspaceHeader project={project} versionId="v1" versions={versions} current={versions[0]} active={active} refresh={vi.fn().mockResolvedValue(undefined)}/>));
    const link = within(screen.getByRole('navigation', { name: '项目版本' })).getByRole('link', { name: 'v2' });
    expect(link).toHaveAttribute('href', `/projects/p_versions/v/v2${suffix}`);
    fireEvent.click(link);
    expect(screen.getByTestId('location')).toHaveTextContent(`/projects/p_versions/v/v2${suffix}`);
  });

  it('waits for the current version PUT before switching and never writes that draft into the destination version', async () => {
    const state = trainingFixture(); let finishSave: () => void = () => {}; const pending: any[] = [];
    server.use(http.put('/api/projects/p_versions/config', async ({ request }) => {
      const scope = new URL(request.url).searchParams.get('version_id'); const config = await request.json() as any; pending.push({ scope, config });
      await new Promise<void>(resolve => { finishSave = resolve; }); state.configs.v1 = config; return HttpResponse.json(config);
    }));
    render(wrap(<Routes><Route path="/projects/:id/v/:versionId/train" element={<TrainConfig/>}/></Routes>));
    await screen.findByRole('link', { name: 'v2' });
    await waitFor(() => expect(screen.getByRole('spinbutton', { name: 'loop.epochs' })).toHaveValue(2));
    fireEvent.change(screen.getByRole('spinbutton', { name: 'loop.epochs' }), { target: { value: '7' } });
    fireEvent.click(within(screen.getByRole('navigation', { name: '项目版本' })).getByRole('link', { name: 'v2' }));
    await waitFor(() => expect(pending).toHaveLength(1));
    expect(pending[0]).toMatchObject({ scope: 'v1', config: { loop: { epochs: 7 } } });
    expect(screen.getByTestId('location')).toHaveTextContent('/projects/p_versions/v/v1/train');
    expect(state.reads).not.toContain('v2');
    await act(async () => finishSave());
    await waitFor(() => expect(screen.getByRole('spinbutton', { name: 'loop.epochs' })).toHaveValue(30));
    expect(screen.getByTestId('location')).toHaveTextContent('/projects/p_versions/v/v2/train');
    expect(pending).toHaveLength(1); expect(state.configs.v1.loop.epochs).toBe(7); expect(state.configs.v2.loop.epochs).toBe(30);
    expect(state.reads).toContain('v1'); expect(state.reads).toContain('v2'); expect(state.reads).not.toContain('');
    expect(state.datasetScopes).toEqual(['v1', 'v2']);
  });

  it('waits for the explicit version to be ready before loading an editable draft', async () => {
    const state = trainingFixture(); let releaseVersions: () => void = () => {}; let requested = false;
    server.use(http.get('/api/projects/p_versions/versions', async () => {
      requested = true; await new Promise<void>(resolve => { releaseVersions = resolve; }); return HttpResponse.json(versions);
    }));
    render(wrap(<Routes><Route path="/projects/:id/v/:versionId/train" element={<TrainConfig/>}/></Routes>));
    await waitFor(() => expect(requested).toBe(true));
    expect(screen.queryByRole('spinbutton', { name: 'loop.epochs' })).not.toBeInTheDocument();
    expect(state.reads).toEqual([]);
    await act(async () => releaseVersions());
    const epochs = await screen.findByRole('spinbutton', { name: 'loop.epochs' });
    expect(epochs).toHaveValue(2);
    fireEvent.change(epochs, { target: { value: '9' } });
    await waitFor(() => expect(state.writes).toHaveLength(1), { timeout: 2000 });
    expect(state.writes[0]).toMatchObject({ version: 'v1', config: { loop: { epochs: 9 } } });
    expect(state.reads).toEqual(['v1']);
  });

  it('submits training to the route version with that version configuration', async () => {
    trainingFixture(); let submitted: any;
    server.use(http.post('/api/jobs', async ({ request }) => { submitted = await request.json(); return HttpResponse.json({ id: 'j_v2' }); }));
    render(wrap(<Routes><Route path="/projects/:id/v/:versionId/train" element={<TrainConfig/>}/><Route path="/jobs/:id" element={<p>Version job created</p>}/></Routes>, '/projects/p_versions/v/v2/train'));
    await waitFor(() => expect(screen.getByRole('spinbutton', { name: 'loop.epochs' })).toHaveValue(30));
    const start = screen.getByRole('button', { name: '开始训练' });
    await waitFor(() => expect(start).toBeEnabled()); fireEvent.click(start);
    await screen.findByText('Version job created');
    expect(submitted).toMatchObject({ project_id: 'p_versions', version_id: 'v2', type: 'train', config: { loop: { epochs: 30 } } });
  });

  it('uses a dataset own version for navigation, cache creation and enabling masks even when another version is active', async () => {
    const state = trainingFixture(); const submitted: any[] = [];
    const configBefore = structuredClone(state.configs.v2);
    server.use(
      http.get('/api/datasets/d_v2', () => HttpResponse.json({ source: { id: 'd_v2', project_id: project.id, version_id: 'v2', path: 'D:/versions/v2/datasets/portraits', repeats: 3, caption_ext: '.txt' }, index_status: 'ready', stats: { images: 1, captioned: 1, masks: 1 } })),
      http.post('/api/jobs', async ({ request }) => { submitted.push(await request.json()); return HttpResponse.json({ id: 'j_cache' }); }),
    );
    vi.spyOn(window, 'alert').mockImplementation(() => {});
    render(wrap(<Routes><Route path="/datasets/:id" element={<Dataset/>}/><Route path="/projects/:id/v/:versionId/train" element={<p>Version training destination</p>}/></Routes>, '/datasets/d_v2'));
    const workflow = await screen.findByRole('navigation', { name: '项目训练步骤' });
    expect(within(workflow).getByRole('link', { name: /^3\s*训练参数$/ })).toHaveAttribute('href', '/projects/p_versions/v/v2/train');
    expect(within(workflow).getByRole('link', { name: /^4\s*任务与结果$/ })).toHaveAttribute('href', '/projects/p_versions/v/v2?step=results');
    fireEvent.click(screen.getByRole('button', { name: i18n.t('dataset.precache') }));
    await waitFor(() => expect(submitted).toHaveLength(1));
    expect(submitted[0]).toMatchObject({ type: 'cache', project_id: project.id, version_id: 'v2' });
    fireEvent.click(screen.getByRole('button', { name: '启用遮罩并前往训练' }));
    await screen.findByText('Version training destination');
    expect(state.writes).toEqual([{ version: 'v2', config: { ...configBefore, dataset: { ...configBefore.dataset, masked_loss: true } } }]);
    expect(state.configs.v1.dataset.masked_loss).toBe(false);
    expect(screen.getByTestId('location')).toHaveTextContent('/projects/p_versions/v/v2/train');
  });
});
