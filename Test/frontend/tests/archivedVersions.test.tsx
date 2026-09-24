import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query';
import { handlers } from '../mocks/handlers';
import { apiClient } from '../../../frontend/src/api/client';
import ProjectDetail from '../../../frontend/src/pages/ProjectDetail/ProjectDetail';
import TrainConfig from '../../../frontend/src/pages/TrainConfig/TrainConfig';
import Dataset from '../../../frontend/src/pages/Dataset/Dataset';
import { schemaDefaults } from '../../../frontend/src/utils/config';
import trainSchema from '../../../frontend/src/schema/train-schema.json';
import i18n from '../../../frontend/src/i18n';

vi.mock('../../../frontend/src/events/useEventStream', () => ({ useEventStream: () => {} }));
vi.mock('../../../frontend/src/api/hooks/useDatasetImages', () => ({ useDatasetImages: () => ({ items: [{ hash: 'img1', rel_path: 'portrait.png', width: 64, height: 64, has_mask: true, caption: 'original caption' }], total: 1, q: '', selected: new Set(['img1']), loading: false, error: null, refresh: vi.fn(), loadMore: vi.fn(), setQ: vi.fn(), selectAll: vi.fn(), clearSelection: vi.fn(), toggleSelect: vi.fn() }) }));
const server = setupServer(...handlers);
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
beforeEach(async () => { await i18n.changeLanguage('zh-CN'); });
afterEach(() => { server.resetHandlers(); vi.restoreAllMocks(); });
afterAll(() => server.close());

function fixture(archived = false) {
  const version = (id: string) => ({ id, project_id: 'p_archive', name: id, note: '', status: 'ready', busy: false, archived: id === 'v1' && archived, dataset_ids: ['d_archive'], created_at: 1, updated_at: 1, stats: { datasets: 1, images: 1, jobs: 0, artifacts: 0 }, paths: { root: `D:/${id}`, config: `D:/${id}/config.toml`, datasets: `D:/${id}/datasets`, runs: `D:/${id}/runs`, cache: `D:/${id}/cache` } });
  const state = {
    project: { id: 'p_archive', name: 'Archive test', active_version_id: archived ? 'v2' : 'v1', version_count: 2, archived: false, dataset_ids: [], note: '', created_at: 1, updated_at: 1, stats: { jobs: 0, artifacts: 0 } },
    versions: [version('v1'), version('v2')], config: schemaDefaults(trainSchema), writes: [] as { method: string; path: string; body?: any }[], configReads: [] as string[],
  };
  state.config.loop.epochs = 2;
  const dataset = { source: { id: 'd_archive', project_id: 'p_archive', version_id: 'v1', path: 'D:/v1/datasets/portraits', repeats: 1, caption_ext: '.txt' }, stats: { images: 1, captioned: 1, masks: 1 }, index_status: 'ready', cache: {} };
  server.use(
    http.get('/api/projects/p_archive', () => HttpResponse.json(state.project)),
    http.get('/api/projects/p_archive/versions', () => HttpResponse.json(state.versions)),
    http.get('/api/projects/p_archive/versions/:vid', ({ params }) => HttpResponse.json(state.versions.find(item => item.id === params.vid))),
    http.get('/api/projects/p_archive/versions/:vid/pipeline', ({ params }) => HttpResponse.json({
      signature: 'archive-fixture', inspection: null, plan: null, operations: [], busy: false,
      archived: state.versions.find(item => item.id === params.vid)?.archived ?? false,
      stale: false, ready_to_train: false, prepared_job_id: null,
    })),
    http.patch('/api/projects/p_archive/versions/:vid', async ({ request, params }) => {
      const body = await request.json() as any; const target = state.versions.find(item => item.id === params.vid)!;
      state.writes.push({ method: 'PATCH', path: `/versions/${params.vid}`, body });
      Object.assign(target, body);
      if (body.archived && state.project.active_version_id === target.id) state.project.active_version_id = 'v2';
      return HttpResponse.json(target);
    }),
    http.patch('/api/projects/p_archive', async ({ request }) => {
      const body = await request.json() as any; state.writes.push({ method: 'PATCH', path: '/project', body });
      if (body.active_version_id && state.versions.find(item => item.id === body.active_version_id)?.archived) return HttpResponse.json({ error: { code: 'version.busy', message: 'Archived version cannot be activated' } }, { status: 409 });
      Object.assign(state.project, body); return HttpResponse.json(state.project);
    }),
    http.get('/api/projects/p_archive/config', ({ request }) => { state.configReads.push(new URL(request.url).searchParams.get('version_id') || ''); return HttpResponse.json(state.config); }),
    http.put('/api/projects/p_archive/config', async ({ request }) => { const body = await request.json(); state.writes.push({ method: 'PUT', path: '/config', body }); state.config = body; return HttpResponse.json(body); }),
    http.get('/api/projects/p_archive/datasets', () => HttpResponse.json([dataset])),
    http.get('/api/datasets/d_archive', () => HttpResponse.json(dataset)),
    http.patch('/api/datasets/d_archive', async ({ request }) => { const body = await request.json(); state.writes.push({ method: 'PATCH', path: '/datasets/d_archive', body }); return HttpResponse.json(dataset); }),
    http.get('/api/jobs', () => HttpResponse.json({ items: [], total: 0, page: 1, page_size: 50 })),
    http.get('/api/artifacts', () => HttpResponse.json([{ id: 'a_archive', kind: 'weights', name: 'archive.safetensors', project_id: 'p_archive', version_id: 'v1', job_id: 'j_old', path: 'D:/v1/output.safetensors', size: 100, created_at: 1, metadata: { test: 'kept' } }])),
    http.post('/api/jobs', async ({ request }) => { const body = await request.json(); state.writes.push({ method: 'POST', path: '/jobs', body }); return HttpResponse.json({ id: 'j_cache' }); }),
  );
  return state;
}
function Location() { const location = useLocation(); return <output data-testid="location">{location.pathname}{location.search}</output>; }
function ProjectCacheProbe() {
  const { data } = useQuery<{ active_version_id: string }>({ queryKey: ['project', 'p_archive'], queryFn: () => apiClient.get('/projects/p_archive', { silent: true }), enabled: false });
  return <span data-testid="remembered-version">{data?.active_version_id}</span>;
}
function show(path = '/projects/p_archive/v/v1?step=data') {
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><MemoryRouter initialEntries={[path]}><Routes>
    <Route path="/projects/:id/v/:versionId" element={<ProjectDetail/>}/><Route path="/projects/:id/v/:versionId/train" element={<TrainConfig/>}/><Route path="/datasets/:id" element={<Dataset/>}/>
  </Routes><Location/><ProjectCacheProbe/></MemoryRouter></QueryClientProvider>);
}

describe('archived versions remain readable without write actions', () => {
  it('archives the displayed active version in place, hides edits, keeps results and restores editing', async () => {
    const state = fixture(); let releaseArchivedVersions: () => void = () => {}; let awaitingArchivedVersions = false;
    server.use(http.get('/api/projects/p_archive/versions', async () => {
      if (state.versions[0].archived) {
        awaitingArchivedVersions = true;
        await new Promise<void>(resolve => { releaseArchivedVersions = resolve; });
      }
      return HttpResponse.json(state.versions);
    }));
    show();
    await screen.findByTestId('project-data-import');
    fireEvent.click(screen.getByRole('button', { name: '版本设置' }));
    const dialog = await screen.findByRole('dialog', { name: '版本设置' });
    fireEvent.click(within(dialog).getByRole('button', { name: '归档版本' }));
    await waitFor(() => expect(awaitingArchivedVersions).toBe(true));
    // The fallback active version has rendered while the versions refetch is still
    // pending. The PATCH response may already have updated the archived row locally.
    await waitFor(() => expect(screen.getByTestId('remembered-version')).toHaveTextContent('v2'));
    expect(within(dialog).getByRole('button', { name: /^(归档版本|恢复版本)$/ })).toBeDisabled();
    expect(state.writes).toEqual([{ method: 'PATCH', path: '/versions/v1', body: { archived: true } }]);
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    await act(async () => releaseArchivedVersions());
    await screen.findByText('此版本已归档 · 只读');
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.queryByTestId('project-data-import')).not.toBeInTheDocument();
    expect(screen.getByTestId('dataset-card-d_archive')).toBeInTheDocument();
    expect(screen.getByTestId('location')).toHaveTextContent('/projects/p_archive/v/v1');
    expect(state.project.active_version_id).toBe('v2');
    const configReadsBeforeTrain = [...state.configReads];
    fireEvent.click(screen.getByRole('link', { name: /^2\s*训练参数$/ }));
    await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/projects/p_archive/v/v1/train'));
    expect(screen.queryByRole('spinbutton', { name: 'loop.epochs' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '开始训练' })).not.toBeInTheDocument();
    expect(state.configReads).toEqual(configReadsBeforeTrain);
    fireEvent.click(screen.getByRole('link', { name: /^3\s*训练结果$/ }));
    expect(await screen.findByTestId('version-results')).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: '配置并启动训练' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('tab', { name: '模型权重' }));
    const output = await screen.findByTestId('artifact-row-a_archive');
    expect(within(output).getByRole('link', { name: '下载: archive.safetensors' })).toHaveAttribute('href', expect.stringContaining('/artifacts/a_archive/download'));
    expect(within(output).getByRole('combobox', { name: '转换格式: archive.safetensors' })).toBeDisabled();
    expect(within(output).getByRole('button', { name: '移除产物记录: archive.safetensors' })).toBeDisabled();
    fireEvent.click(within(output).getByRole('button', { name: '查看元数据: archive.safetensors' }));
    const metadata = screen.getByRole('dialog'); expect(metadata).toHaveTextContent('kept');
    fireEvent.click(within(metadata).getByRole('button', { name: '关闭' }));
    expect(state.writes).toEqual([{ method: 'PATCH', path: '/versions/v1', body: { archived: true } }]);
    fireEvent.click(screen.getByRole('button', { name: '恢复版本' }));
    await waitFor(() => expect(screen.queryByText('此版本已归档 · 只读')).not.toBeInTheDocument());
    fireEvent.click(screen.getByRole('link', { name: /^1\s*训练数据$/ }));
    expect(await screen.findByTestId('project-data-import')).toBeInTheDocument();
    expect(state.writes.some(write => write.path === '/versions/v1' && write.body.archived === false)).toBe(true);
    expect(state.writes.filter(write => write.method !== 'PATCH')).toEqual([]);
  });

  it('keeps archived train pages free of editor/start controls and writes, with compare/results and restore available', async () => {
    const state = fixture(true); show('/projects/p_archive/v/v1/train');
    await screen.findByText('此版本已归档 · 只读');
    expect(screen.queryByRole('spinbutton', { name: 'loop.epochs' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '开始训练' })).not.toBeInTheDocument();
    expect(state.configReads).toEqual([]); expect(state.writes).toEqual([]);
    expect(screen.getByRole('link', { name: '查看此版本的训练结果' })).toHaveAttribute('href', '/projects/p_archive/v/v1?step=results');
    fireEvent.click(screen.getByRole('button', { name: '比较' }));
    const dialog = await screen.findByRole('dialog', { name: '比较版本' });
    fireEvent.click(within(dialog).getByRole('button', { name: '查看差异' }));
    await screen.findByText('两个版本的参数相同。');
    expect(state.configReads).toEqual(['v2', 'v1']); expect(state.writes).toEqual([]);
    fireEvent.click(within(dialog).getByRole('button', { name: '关闭' }));
    fireEvent.click(screen.getByRole('button', { name: '恢复版本' }));
    expect(await screen.findByRole('spinbutton', { name: 'loop.epochs' })).toHaveValue(2);
    expect(screen.getByRole('button', { name: '开始训练' })).toBeInTheDocument();
    expect(state.writes.filter(write => write.method !== 'PATCH')).toEqual([]);
  });

  it('excludes archived and busy sources when creating but includes archived versions in comparisons', async () => {
    const state = fixture(); state.versions[1].archived = true;
    state.versions.push({ ...state.versions[0], id: 'v3', name: 'v3', busy: true });
    show(); await screen.findByTestId('project-data-import');
    fireEvent.click(screen.getByRole('button', { name: '新版本' }));
    const dialog = await screen.findByRole('dialog', { name: '新建实验版本' });
    const source = within(dialog).getByRole('combobox', { name: '创建来源' });
    fireEvent.click(source);
    expect(screen.getAllByRole('option').map(option=>option.textContent)).toEqual(['默认配置 · 空白版本','v1']);
    fireEvent.keyDown(source,{key:'Escape'});
    fireEvent.click(within(dialog).getByRole('button', { name: '取消' }));
    fireEvent.click(screen.getByRole('button', { name: '比较' }));
    const compare = await screen.findByRole('dialog', { name: '比较版本' });
    fireEvent.click(within(compare).getByRole('combobox',{name:'对比版本'}));
    expect(screen.getByRole('option',{name:'v2'})).toBeInTheDocument();
    expect(state.writes).toEqual([]);
  });

  it('locks a directly opened dataset while resolving its archived version and permits viewing original captions', async () => {
    const state = fixture(true); let release: () => void = () => {}; let requested = false;
    server.use(http.get('/api/projects/p_archive/versions/v1', async () => { requested = true; await new Promise<void>(resolve => { release = resolve; }); return HttpResponse.json(state.versions[0]); }));
    show('/datasets/d_archive'); await screen.findByText('正在确认版本状态，暂以只读方式查看。');
    expect(screen.getByRole('switch', { name:'使用遮罩训练' })).toBeDisabled();
    expect(screen.queryByRole('button', { name: /编辑遮罩/ })).not.toBeInTheDocument();
    await waitFor(() => expect(requested).toBe(true)); await act(async () => release());
    await screen.findByText('此版本已归档，图片、标签和遮罩只读。');
    for (const name of [i18n.t('dataset.rescan'), i18n.t('dataset.remove'), '保存目录设置']) {
      const button = screen.getByRole('button', { name }); expect(button).toBeDisabled(); fireEvent.click(button);
    }
    expect(screen.queryByTestId('batch-add-input')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '查看图片与标签: portrait.png' }));
    const preview = screen.getByTestId('caption-editor');
    expect(within(preview).getByRole('img')).toHaveAttribute('src', expect.stringContaining('/datasets/d_archive/images/img1/file'));
    expect(within(preview).getByText('original caption')).toBeInTheDocument();
    expect(within(preview).queryByRole('textbox')).not.toBeInTheDocument();
    expect(within(preview).getByTestId('caption-save-btn')).toBeDisabled();
    fireEvent.click(within(preview).getByTestId('caption-save-btn'));
    expect(state.writes).toEqual([]);
    fireEvent.click(within(preview).getByRole('button', { name: i18n.t('common.cancel') }));
    state.versions[0].archived = false;
    server.use(http.get('/api/projects/p_archive/versions/v1', () => HttpResponse.json(state.versions[0])));
    vi.spyOn(window, 'alert').mockImplementation(() => {});
    fireEvent.click(screen.getByRole('button', { name: i18n.t('common.refresh') }));
    expect(await screen.findByRole('button', { name: '编辑遮罩 · 已有文件' })).toBeEnabled();
    fireEvent.click(screen.getByRole('switch', { name: '使用遮罩训练' }));
    await waitFor(() => expect(state.writes).toHaveLength(1));
    expect(state.writes[0]).toMatchObject({ method: 'PATCH', path: '/datasets/d_archive', body: { masked_loss: true } });
  });
});
