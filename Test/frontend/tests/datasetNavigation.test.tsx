import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter, Route, Routes, useLocation, useNavigate, type Location as RouterLocation } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import Layout from '../../../frontend/src/components/Layout';
import SettingsDrawer from '../../../frontend/src/components/SettingsDrawer';
import Dataset from '../../../frontend/src/pages/Dataset/Dataset';
import { handlers } from '../mocks/handlers';
import i18n from '../../../frontend/src/i18n';

const { updateCaption, refreshImages, imageFixtures } = vi.hoisted(() => ({ updateCaption: vi.fn(), refreshImages:vi.fn(), imageFixtures:{items:[{ hash:'image1', rel_path:'photo.png', width:64, height:64, has_mask:false, caption:'cat' }] as any[]} }));
vi.mock('../../../frontend/src/events/useEventStream', () => ({ useEventStream: () => {}, useEventStreamStatus: () => 'connected' }));
vi.mock('../../../frontend/src/api/hooks/useDatasetImages', () => ({ useDatasetImages: () => ({
  items: imageFixtures.items,
  total: 1, q: '', selected: new Set(), loading: false, error: null,
  refresh: refreshImages, loadMore: vi.fn(), setQ: vi.fn(), selectAll: vi.fn(), clearSelection: vi.fn(), toggleSelect: vi.fn(), updateCaption,
}) }));
vi.mock('../../../frontend/src/components/masks/MaskEditor', () => ({ MaskEditor: ({ onClose }: { onClose: () => void }) => <div role="dialog" aria-label="Mask draft"><button onClick={onClose}>Close mask draft</button></div> }));
const server = setupServer(...handlers);
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterAll(() => server.close());
beforeEach(async () => { await i18n.changeLanguage('zh-CN'); updateCaption.mockClear();refreshImages.mockClear();imageFixtures.items=[{hash:'image1',rel_path:'photo.png',width:64,height:64,has_mask:false,caption:'cat'}]; });
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
  return <><output data-testid="route">{location.pathname}{location.search}{location.hash}</output><button onClick={() => navigate('/datasets/d_missing')}>Open missing dataset</button></>;
}
function show(path = '/datasets/d_known') {
  fixture();
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><MemoryRouter initialEntries={[path]}><WorkspaceRoutes/><Location/></MemoryRouter></QueryClientProvider>);
}
function WorkspaceRoutes() {
  const location = useLocation(); const navigate = useNavigate();
  const background = (location.state as {backgroundLocation?:RouterLocation}|null)?.backgroundLocation;
  return <><Routes location={background || location}><Route element={<Layout/>}>
    <Route path="/projects/:id/v/:versionId/train" element={<p>Version destination</p>}/><Route path="/datasets/:id" element={<Dataset/>}/><Route path="/projects/:id/v/:versionId" element={<p>Version destination</p>}/>
  </Route></Routes>{background && <SettingsDrawer onClose={() => navigate(`${background.pathname}${background.search}`, {replace:true})}><p>Preferences fixture</p></SettingsDrawer>}</>;
}
async function editCaption() {
  fireEvent.click(await screen.findByRole('button', { name: '编辑标签: photo.png' }));
  const input = screen.getByTestId('tag-add-input');
  fireEvent.change(input, { target: { value: 'blue eyes' } }); fireEvent.keyDown(input, { key: 'Enter' });
  expect(screen.getByTestId('tag-chip-1')).toHaveTextContent('blue eyes');
}

describe('dataset project sidebar and navigation protection', () => {
  it('calculates only this folder with the saved version settings and offers its sizing controls',async()=>{
    const requested:any[]=[];
    server.use(
      http.get('/api/projects/p_dataset/config',({request})=>{
        expect(new URL(request.url).searchParams.get('version_id')).toBe('v2');
        return HttpResponse.json({dataset:{resolution_mode:'native',native_max_pixels:16777216}});
      }),
      http.post('/api/plan',async({request})=>{requested.push(await request.json());return HttpResponse.json({ok:true,errors:[],buckets:[{base: 0,w:2992,h:2448,items:1,batches:1}],native:{max_pixels:16777216}});}),
    );
    show();await screen.findByRole('combobox',{name:'项目版本'});
    fireEvent.click(screen.getByRole('button',{name:'分布与分桶'}));
    expect(await screen.findByText('2992×2448')).toBeVisible();
    expect(requested).toHaveLength(1);
    expect(requested[0]).toMatchObject({dataset_ids:['d_known'],project_id:'p_dataset',version_id:'v2',config:{dataset:{native_max_pixels:16777216}}});
    expect(screen.getByText('原生 · 像素上限 16,777,216')).toBeVisible();
    expect(screen.getByRole('link',{name:/调整分辨率与分桶/})).toHaveAttribute('href','/projects/p_dataset/v/v2/train?tab=data&group=dataset');
    fireEvent.focus(window);
    await waitFor(()=>expect(requested).toHaveLength(2));
    expect(await screen.findByText('2992×2448')).toBeVisible();
  });
  it('edits JSON by original field path and targets the selected relative path even with duplicate hashes',async()=>{
    const structure={format:'full',editable:true,legacy_override:false,revision:'json-file-sha',document:{ai_output:{appearance:['blue coat'],nl:'Natural prose.'},meta:{score:2}},fields:[{path:['ai_output','appearance'],role:'appearance',value:['blue coat'],present:true},{path:['ai_output','nl'],role:'nl',value:'Natural prose.',present:true}]};
    imageFixtures.items=['first/a.png','second/a.png'].map(rel_path=>({hash:'same',rel_path,width:64,height:64,has_mask:false,caption:'blue coat. Natural prose.',caption_format:'json',caption_structure:structure}));
    let requestUrl='';let payload:unknown;
    server.use(http.put('/api/datasets/d_known/images/same/caption',async({request})=>{requestUrl=request.url;payload=await request.json();return HttpResponse.json({ok:true});}));
    show();await screen.findByRole('combobox',{name:'项目版本'});
    fireEvent.click(await screen.findByRole('button',{name:'编辑标签: second/a.png'}));
    expect(screen.getByRole('textbox',{name:'自然语言描述'})).toHaveValue('Natural prose.');
    expect(screen.queryByTestId('tag-add-input')).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole('textbox',{name:'外观与服装 · 标签 1'}),{target:{value:'red coat'}});
    fireEvent.click(screen.getByTestId('caption-save-btn'));
    await waitFor(()=>expect(refreshImages).toHaveBeenCalledOnce());
    expect(new URL(requestUrl).searchParams.get('rel_path')).toBe('second/a.png');
    expect(payload).toEqual({caption_fields:[{path:['ai_output','appearance'],value:['red coat']}],caption_revision:'json-file-sha'});
    expect(updateCaption).not.toHaveBeenCalled();expect(structure.document.meta.score).toBe(2);
  });

  it('returns to the same version dataset library after removing a dataset', async () => {
    let deleted = false;
    server.use(http.delete('/api/datasets/d_known', () => { deleted = true; return HttpResponse.json({ ok: true }); }));
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    show('/datasets/d_known?project=p_wrong&version=v_wrong');
    await screen.findByRole('combobox', { name: '项目版本' });
    const remove = screen.getByRole('button', { name: String(i18n.t('dataset.remove')) });
    await waitFor(() => expect(remove).toBeEnabled());
    fireEvent.click(remove);
    await screen.findByText('Version destination');
    expect(deleted).toBe(true);
    expect(screen.getByTestId('route')).toHaveTextContent('/projects/p_dataset/v/v2?step=data&data_step=import#version-datasets');
  });

  it('rebuilds the explicit dataset return route from metadata after a direct visit, overriding stale query context', async () => {
    show('/datasets/d_known?project=p_wrong&version=v_wrong');
    await screen.findByRole('combobox',{name:'项目版本'});
    const back=screen.getByRole('link',{name:'返回本版本数据集'});
    expect(back).toHaveAttribute('href','/projects/p_dataset/v/v2?step=data&data_step=import#version-datasets');
    expect(back.closest('nav')).toHaveClass('workspace-breadcrumb');
    expect(back.closest('header')).toContainElement(screen.getByRole('heading', {name:'photos'}));
    fireEvent.click(back);
    await screen.findByText('Version destination');
    expect(screen.getByTestId('route')).toHaveTextContent('/projects/p_dataset/v/v2?step=data&data_step=import');
  });

  it('saves a dirty caption before following the visible return link and stays on failure', async () => {
    let fail=true; let release=()=>{};
    server.use(http.put('/api/datasets/d_known/images/image1/caption',async()=>{if(fail)return HttpResponse.json({error:{message:'Save failed'}},{status:409}); await new Promise<void>(resolve=>{release=resolve;});return HttpResponse.json({ok:true});}));
    show(); await screen.findByRole('combobox',{name:'项目版本'}); await editCaption();
    fireEvent.click(screen.getByRole('link',{name:'返回本版本数据集'}));
    expect(await screen.findByRole('alert')).toHaveTextContent('Save failed');
    expect(screen.getByTestId('route')).toHaveTextContent('/datasets/d_known');
    fail=false; fireEvent.click(screen.getByRole('link',{name:'返回本版本数据集'}));
    await waitFor(()=>expect(screen.getByTestId('caption-save-btn')).toBeDisabled());
    expect(screen.getByTestId('route')).toHaveTextContent('/datasets/d_known');
    await act(async()=>release()); await screen.findByText('Version destination');
    expect(updateCaption).toHaveBeenCalledExactlyOnceWith('image1','cat, blue eyes','photo.png');
  });

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
    expect(updateCaption).toHaveBeenCalledWith('image1', 'cat, blue eyes', 'photo.png');
  });

  it('keeps the caption draft and dataset route on a failed save, then retries before following a stage link', async () => {
    let fail = true; const writes: unknown[] = [];
    server.use(http.put('/api/datasets/d_known/images/image1/caption', async ({ request }) => {
      writes.push(await request.json()); return fail ? HttpResponse.json({ error: { code: 'caption.conflict', message: 'Caption changed on disk' } }, { status: 409 }) : HttpResponse.json({ ok: true });
    }));
    show(); await screen.findByRole('combobox', { name: '项目版本' }); await editCaption();
    const stages = screen.getByRole('navigation', { name: '项目训练步骤' });
    fireEvent.click(within(stages).getByRole('link', { name: /^2\s*训练参数$/ }));
    expect(await within(screen.getByRole('dialog', { name: '编辑图片标签' })).findByRole('alert')).toHaveTextContent('Caption changed on disk');
    expect(screen.getByTestId('route')).toHaveTextContent('/datasets/d_known');
    expect(screen.getByTestId('tag-chip-1')).toHaveTextContent('blue eyes'); expect(updateCaption).not.toHaveBeenCalled();
    fail = false; fireEvent.click(within(stages).getByRole('link', { name: /^2\s*训练参数$/ }));
    await screen.findByText('Version destination'); expect(screen.getByTestId('route')).toHaveTextContent('/projects/p_dataset/v/v2/train');
    expect(writes).toEqual([{ caption: 'cat, blue eyes' }, { caption: 'cat, blue eyes' }]);
  });

  it('requires closing or saving the mask editor before using sidebar navigation', async () => {
    show(); await screen.findByRole('combobox', { name: '项目版本' });
    fireEvent.click(await screen.findByRole('button', { name: '编辑遮罩' }));
    fireEvent.click(within(screen.getByRole('navigation', { name: '项目训练步骤' })).getByRole('link', { name: /^2\s*训练参数$/ }));
    expect(await screen.findByRole('alert')).toHaveTextContent('请先在遮罩编辑器中保存或关闭');
    expect(screen.getByTestId('route')).toHaveTextContent('/datasets/d_known'); expect(screen.getByRole('dialog', { name: 'Mask draft' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Close mask draft' }));
    fireEvent.click(within(screen.getByRole('navigation', { name: '项目训练步骤' })).getByRole('link', { name: /^2\s*训练参数$/ }));
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
    fireEvent.click(within(screen.getByRole('complementary')).getByRole('link', { name: '系统设置' }));
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
    expect(updateCaption).toHaveBeenCalledExactlyOnceWith('image1', 'cat, blue eyes', 'photo.png');
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
