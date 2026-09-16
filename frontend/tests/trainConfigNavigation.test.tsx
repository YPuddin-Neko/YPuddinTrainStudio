import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { MemoryRouter, Route, Routes, useLocation, useNavigate, useParams } from 'react-router-dom';
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
beforeEach(async () => { sessionStorage.clear(); await i18n.changeLanguage('zh-CN'); });
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
    <MemoryRouter initialEntries={['/projects/p_nav/train?tab=train']}><Routes>
      <Route path="/projects/:id/train" element={<TrainConfig />} />
      <Route path="/projects/:id" element={<Destination />} />
      <Route path="/datasets/:id" element={<Destination />} />
    </Routes></MemoryRouter>
  </QueryClientProvider>);
}

function HistoryButtons() {
  const navigate = useNavigate();
  return <><button onClick={() => navigate(-1)}>Browser back</button><button onClick={() => navigate(1)}>Browser forward</button></>;
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
  it('shows no save hint while loading or before the first edit of a clean configuration', async () => {
    const state = fixtures();
    let complete: (() => void) | undefined;
    server.use(http.get('/api/projects/p_nav/config', async () => {
      await new Promise<void>(resolve => { complete = resolve; });
      return HttpResponse.json(state.config);
    }));
    show();
    await waitFor(() => expect(complete).toBeDefined());
    expect(screen.queryByLabelText('配置保存状态')).not.toBeInTheDocument();
    complete!();
    await screen.findByRole('spinbutton', { name: 'loop.epochs' });
    expect(screen.queryByLabelText('配置保存状态')).not.toBeInTheDocument();
    expect(screen.queryByText(/修改自动保存|更改会自动保存|草稿已自动保存/)).not.toBeInTheDocument();
  });

  it('opens only current source B even when registered A is first, with Windows spelling normalized', async () => {
    fixtures(); show();
    await screen.findByRole('spinbutton', { name: 'loop.epochs' });
    fireEvent.click(screen.getByRole('button', { name: /数据与分桶$/ }));
    const link = await screen.findByRole('link', { name: '标签编辑' });
    expect(link).toHaveAttribute('href', '/projects/p_nav?step=data&data_step=captions&dataset=d_B');
    fireEvent.click(link);
    await screen.findByText('Destination /projects/p_nav?step=data&data_step=captions&dataset=d_B');
  });

  it.each([
    { paths: ['D:/outside'], label: '导入数据后编辑标签与遮罩' },
    { paths: ['D:/training/A', 'D:/training/B'], label: '选择数据集编辑标签与遮罩' },
  ])('uses project source selection when a direct editor would be ambiguous: $label', async ({ paths }) => {
    fixtures(paths); show();
    await screen.findByRole('spinbutton', { name: 'loop.epochs' });
    fireEvent.click(screen.getByRole('button', { name: /数据与分桶$/ }));
    expect(await screen.findByRole('link', { name: '标签编辑' })).toHaveAttribute('href', '/projects/p_nav?step=data&data_step=captions');
    expect(screen.getByRole('link', {name:'涂抹与遮罩'})).toHaveAttribute('href','/projects/p_nav?step=data&data_step=paint');
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
    fireEvent.click(screen.getByRole('link', { name: /^3\s*训练结果$/ }));
    expect(screen.getByLabelText('配置保存状态')).toHaveTextContent('保存中…');
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
    await screen.findByText('Destination /projects/p_nav?step=results');
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
    fireEvent.click(screen.getByRole('link', { name: /^3\s*训练结果$/ }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Disk is full');
    expect(screen.getByRole('alert')).toHaveTextContent('已留在当前页面');
    expect(epochs).toHaveValue(13);
    expect(screen.queryByText(/Destination/)).not.toBeInTheDocument();
    fail = false;
    fireEvent.click(screen.getByRole('link', { name: /^3\s*训练结果$/ }));
    await screen.findByText('Saved epochs: 13');
  });

  it('saves a return to the original value after an older autosave finishes, without discarding the newest session draft', async () => {
    const state = fixtures();
    const writes: {config:any;finish:()=>void}[] = [];
    server.use(http.put('/api/projects/p_nav/config',async({request})=>{
      const config=await request.json();await new Promise<void>(resolve=>writes.push({config,finish:resolve}));state.config=config;return HttpResponse.json(config);
    }));
    show();
    const epochs=await screen.findByRole('spinbutton',{name:'loop.epochs'});
    fireEvent.change(epochs,{target:{value:'7'}});
    expect(screen.getByLabelText('配置保存状态')).toHaveTextContent('等待保存…');
    await waitFor(()=>expect(writes).toHaveLength(1),{timeout:2000});
    expect(screen.getByLabelText('配置保存状态')).toHaveTextContent('保存中…');
    fireEvent.change(epochs,{target:{value:'2'}});
    expect(JSON.parse(sessionStorage.getItem('training-draft:p_nav:legacy')!).draft.loop.epochs).toBe(2);
    writes[0].finish();
    await waitFor(()=>expect(writes).toHaveLength(2));
    expect(writes[1].config.loop.epochs).toBe(2);
    expect(screen.getByLabelText('配置保存状态')).toHaveTextContent('保存中…');
    expect(screen.queryByTestId('draft-saved')).not.toBeInTheDocument();
    expect(JSON.parse(sessionStorage.getItem('training-draft:p_nav:legacy')!).draft.loop.epochs).toBe(2);
    writes[1].finish();
    expect(await screen.findByTestId('draft-saved')).toHaveTextContent(/^已保存 \d{2}:\d{2}:\d{2}$/);
    expect(state.config.loop.epochs).toBe(2);
    expect(epochs).toHaveValue(2);
    expect(sessionStorage.getItem('training-draft:p_nav:legacy')).toBeNull();
    expect(writes).toHaveLength(2);
  });

  it('keeps failed autosave edits and reports saved only after an explicit retry succeeds', async () => {
    const state = fixtures();
    let fail = true;
    let complete: (() => void) | undefined;
    server.use(http.put('/api/projects/p_nav/config', async ({ request }) => {
      const body = await request.json();
      if (fail) return HttpResponse.json({ error: { code: 'storage.error', message: 'Disk is full' } }, { status: 500 });
      await new Promise<void>(resolve => { complete = resolve; });
      state.config = body;
      return HttpResponse.json(body);
    }));
    show();
    const epochs = await screen.findByRole('spinbutton', { name: 'loop.epochs' });
    fireEvent.change(epochs, { target: { value: '19' } });
    expect(screen.getByLabelText('配置保存状态')).toHaveTextContent('等待保存…');
    await waitFor(() => expect(screen.getByLabelText('配置保存状态')).toHaveTextContent('保存失败，请重试'), { timeout: 2000 });
    expect(screen.getByRole('alert')).toHaveTextContent('Disk is full');
    expect(screen.queryByTestId('draft-saved')).not.toBeInTheDocument();
    expect(epochs).toHaveValue(19);
    expect(state.config.loop.epochs).toBe(2);
    expect(JSON.parse(sessionStorage.getItem('training-draft:p_nav:legacy')!).draft.loop.epochs).toBe(19);

    fail = false;
    fireEvent.click(screen.getByRole('button', { name: '保存草稿' }));
    await waitFor(() => expect(complete).toBeDefined());
    expect(screen.getByLabelText('配置保存状态')).toHaveTextContent('保存中…');
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.queryByTestId('draft-saved')).not.toBeInTheDocument();
    complete!();
    expect(await screen.findByTestId('draft-saved')).toHaveTextContent(/^已保存 \d{2}:\d{2}:\d{2}$/);
    expect(state.config.loop.epochs).toBe(19);
    expect(sessionStorage.getItem('training-draft:p_nav:legacy')).toBeNull();
    expect(screen.getByRole('button', { name: '保存草稿' })).toBeDisabled();
  });

  it('keeps draft edits across parameter tabs and explicitly saves before reporting saved', async () => {
    const state=fixtures();let complete: (()=>void)|undefined;
    server.use(http.put('/api/projects/p_nav/config',async({request})=>{
      const body=await request.json();await new Promise<void>(resolve=>{complete=resolve;});state.config=body;return HttpResponse.json(body);
    }));
    show();
    const epochs=await screen.findByRole('spinbutton',{name:'loop.epochs'});
    fireEvent.change(epochs,{target:{value:'23'}});
    expect(screen.getByLabelText('配置保存状态')).toHaveTextContent('等待保存…');
    fireEvent.click(screen.getByRole('button', { name: /数据与分桶$/ }));
    fireEvent.click(screen.getByRole('button', { name: /设备与时长$/ }));
    expect(screen.getByRole('spinbutton',{name:'loop.epochs'})).toHaveValue(23);
    fireEvent.click(screen.getByRole('button',{name:'保存草稿'}));
    await waitFor(()=>expect(complete).toBeDefined());
    expect(screen.queryByTestId('draft-saved')).not.toBeInTheDocument();
    expect(screen.getByRole('button',{name:'保存中…'})).toBeDisabled();
    complete!();
    await screen.findByTestId('draft-saved');
    expect(state.config.loop.epochs).toBe(23);
    expect(screen.getByRole('button',{name:'保存草稿'})).toBeDisabled();
    expect(screen.queryByText(/Destination/)).not.toBeInTheDocument();
  });

  it('guards same-tab application links outside the training panel and warns before closing an unsaved draft', async () => {
    const state=fixtures();let complete:(()=>void)|undefined;
    server.use(http.put('/api/projects/p_nav/config',async({request})=>{
      const body=await request.json();await new Promise<void>(resolve=>{complete=resolve;});state.config=body;return HttpResponse.json(body);
    }));
    show();
    fireEvent.change(await screen.findByRole('spinbutton',{name:'loop.epochs'}),{target:{value:'31'}});
    const closing=new Event('beforeunload',{cancelable:true});window.dispatchEvent(closing);expect(closing.defaultPrevented).toBe(true);
    const outside=document.createElement('a');outside.href='/projects/p_nav?step=results';outside.textContent='Application navigation';document.body.append(outside);
    try {
      fireEvent.click(outside);
      await waitFor(()=>expect(complete).toBeDefined());
      expect(screen.queryByText(/Destination/)).not.toBeInTheDocument();
      complete!();
      await screen.findByText('Saved epochs: 31');
    } finally {outside.remove();}
  });

  it('does not intercept modified or new-tab link clicks', async () => {
    fixtures(); show();
    const epochs = await screen.findByRole('spinbutton', { name: 'loop.epochs' });
    fireEvent.change(epochs, { target: { value: '17' } });
    const link = screen.getByRole('link', { name: /^3\s*训练结果$/ });
    const prevented: boolean[] = [];
    // Observe after React's capture handler, then suppress jsdom's real navigation.
    link.addEventListener('click', event => { prevented.push(event.defaultPrevented); event.preventDefault(); });
    for (const modifier of ['ctrlKey', 'metaKey', 'shiftKey', 'altKey']) fireEvent.click(link, { [modifier]: true });
    link.setAttribute('target', '_blank');
    fireEvent.click(link);
    expect(prevented).toEqual([false, false, false, false, false]);
    expect(screen.getByLabelText('配置保存状态')).not.toHaveTextContent('保存中…');
  });

  it('recovers a version draft after browser Back and a failed unmount save, preserving newer server data and clearing only after saving', async () => {
    const state = fixtures();
    let failed = true;
    let failedWrites = 0;
    const saved: any[] = [];
    server.use(
      http.get('/api/projects/p_nav', () => HttpResponse.json({id:'p_nav',name:'Navigation test',active_version_id:'v2'})),
      http.get('/api/projects/p_nav/versions', () => HttpResponse.json([{id:'v2',project_id:'p_nav',name:'Second version',status:'ready',archived:false,paths:{root:'/v2'},stats:{images:2,jobs:0,artifacts:0}}])),
      http.put('/api/projects/p_nav/config', async ({request}) => {
        expect(new URL(request.url).searchParams.get('version_id')).toBe('v2');
        if (failed) { failedWrites++; return HttpResponse.json({error:{code:'storage.error',message:'Disk is full'}},{status:500}); }
        state.config = await request.json(); saved.push(state.config); return HttpResponse.json(state.config);
      }),
    );
    const key = 'training-draft:p_nav:v2';
    const otherDraft = JSON.stringify({version:1,base:state.config,draft:{...state.config,loop:{...state.config.loop,epochs:91}}});
    sessionStorage.setItem('training-draft:p_nav:v1',otherDraft);
    render(<QueryClientProvider client={new QueryClient({defaultOptions:{queries:{retry:false}}})}><MemoryRouter initialEntries={['/history','/projects/p_nav/v/v2/train?tab=train']} initialIndex={1}><HistoryButtons/><Routes>
      <Route path="/history" element={<p>Earlier page</p>}/><Route path="/projects/:id/v/:versionId/train" element={<TrainConfig/>}/>
    </Routes></MemoryRouter></QueryClientProvider>);
    const epochs = await screen.findByRole('spinbutton',{name:'loop.epochs'});
    expect(epochs).toHaveValue(2);
    fireEvent.change(epochs,{target:{value:'27'}});
    expect(JSON.parse(sessionStorage.getItem(key)!).draft.loop.epochs).toBe(27);
    fireEvent.click(screen.getByRole('button',{name:'Browser back'}));
    await screen.findByText('Earlier page');
    await waitFor(()=>expect(failedWrites).toBe(1));
    expect(JSON.parse(sessionStorage.getItem(key)!).draft.loop.epochs).toBe(27);
    state.config = {...state.config,dataset:{...state.config.dataset,sources:[{path:'/v2/new-source',repeats:4}]},output:{...state.config.output,dir:'/v2/new-output'}};
    failed = false;
    fireEvent.click(screen.getByRole('button',{name:'Browser forward'}));
    expect(await screen.findByRole('spinbutton',{name:'loop.epochs'})).toHaveValue(27);
    expect(screen.getByText('已恢复此版本上次未保存的草稿。')).toBeInTheDocument();
    expect(sessionStorage.getItem(key)).not.toBeNull();
    fireEvent.click(screen.getByRole('button',{name:'保存草稿'}));
    await screen.findByTestId('draft-saved');
    expect(saved.at(-1)).toMatchObject({loop:{epochs:27},dataset:{sources:[{path:'/v2/new-source',repeats:4}]},output:{dir:'/v2/new-output'}});
    expect(sessionStorage.getItem(key)).toBeNull();
    expect(sessionStorage.getItem('training-draft:p_nav:v1')).toBe(otherDraft);
  });

  it.each(['/api/projects/p_nav/config','/api/schema/train'])('does not use a session draft to bypass failed core loading: %s', async (endpoint) => {
    const state = fixtures();
    const cached = JSON.stringify({version:1,base:state.config,draft:{...state.config,loop:{...state.config.loop,epochs:99}}});
    sessionStorage.setItem('training-draft:p_nav:legacy',cached);
    const write = vi.fn();
    server.use(http.get(endpoint,()=>HttpResponse.json({error:{code:'unavailable',message:'Core configuration unavailable'}},{status:503})),http.put('/api/projects/p_nav/config',()=>{write();return HttpResponse.json({});}));
    show();
    expect(await screen.findByRole('alert')).toHaveTextContent('Core configuration unavailable');
    expect(screen.queryByRole('spinbutton',{name:'loop.epochs'})).not.toBeInTheDocument();
    expect(screen.getByRole('button',{name:'保存草稿'})).toBeDisabled();
    expect(screen.getByRole('button',{name:'开始训练'})).toBeDisabled();
    expect(sessionStorage.getItem('training-draft:p_nav:legacy')).toBe(cached);
    expect(write).not.toHaveBeenCalled();
  });
});
