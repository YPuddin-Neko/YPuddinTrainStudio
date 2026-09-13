import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Routes, Route, useLocation, useNavigate } from 'react-router-dom';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import ProjectDetail from '../src/pages/ProjectDetail/ProjectDetail';
import TrainConfig from '../src/pages/TrainConfig/TrainConfig';
import { handlers } from '../src/mocks/handlers';
import { schemaDefaults } from '../src/utils/config';
import trainSchema from '../src/schema/train-schema.json';
import i18n from '../src/i18n';

vi.mock('../src/events/useEventStream', () => ({ useEventStream: () => {} }));
const server = setupServer(...handlers);
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterAll(() => server.close());
afterEach(() => { server.resetHandlers(); sessionStorage.clear(); vi.restoreAllMocks(); });
let config: Record<string, any>;
let writes: Record<string, any>[];
beforeEach(async () => {
  await i18n.changeLanguage('zh-CN'); writes = [];
  config = schemaDefaults(trainSchema); config.model.dit_path = '/models/original.safetensors'; config.loop.epochs = 19;
  const version = { id: 'v2', project_id: 'p_model', name: 'Version two', status: 'ready', archived: false, paths: {}, stats: {} };
  server.use(
    http.get('/api/projects/p_model', () => HttpResponse.json({ id: 'p_model', name: 'Model workspace', active_version_id: 'v2' })),
    http.get('/api/projects/p_model/versions', () => HttpResponse.json([version])),
    http.get('/api/projects/p_model/config', () => HttpResponse.json(config)),
    http.put('/api/projects/p_model/config', async ({ request }) => {
      expect(new URL(request.url).searchParams.get('version_id')).toBe('v2');
      const body = await request.json() as Record<string, any>; writes.push(body); config = body; return HttpResponse.json(body);
    }),
    http.get('/api/projects/p_model/datasets', () => HttpResponse.json([])),
    http.get('/api/models', () => HttpResponse.json([])),
    http.get('/api/jobs', () => HttpResponse.json({ items: [], total: 0, page: 1, page_size: 5 })),
  );
});
function RouteProbe() {
  const location = useLocation(); const navigate = useNavigate();
  return <><output data-testid="route">{location.pathname}{location.search} {JSON.stringify(location.state)}</output><button onClick={() => navigate(-1)}>Browser back</button></>;
}
function show(path: string) {
  return render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><MemoryRouter initialEntries={[{pathname:path.split('?')[0],search:`?${path.split('?')[1] || ''}`,state:{origin:'kept'}}]}><Routes>
    <Route path="/projects/:id" element={<ProjectDetail/>}/><Route path="/projects/:id/v/:versionId" element={<ProjectDetail/>}/>
    <Route path="/projects/:id/train" element={<TrainConfig/>}/><Route path="/projects/:id/v/:versionId/train" element={<TrainConfig/>}/>
  </Routes><RouteProbe/></MemoryRouter></QueryClientProvider>);
}

describe('model choices share the training configuration workspace', () => {
  it.each(['/projects/p_model?step=models', '/projects/p_model/v/v2?step=models'])('redirects %s to the active/scoped model tab and migrates its unsaved model choices', async path => {
    sessionStorage.setItem('model-draft:p_model:v2', JSON.stringify({...config.model,dit_path:'/models/unsaved.safetensors'}));
    sessionStorage.setItem('model-draft:p_model:v1', JSON.stringify({...config.model,dit_path:'/models/other-version.safetensors'}));
    show(path);
    await screen.findByDisplayValue('/models/unsaved.safetensors');
    expect(screen.getByTestId('route')).toHaveTextContent('/projects/p_model/v/v2/train?tab=model');
    expect(screen.getByTestId('route')).toHaveTextContent('"origin":"kept"');
    expect(screen.getByRole('tab',{name:'底模与输出'})).toHaveAttribute('aria-selected','true');
    expect(within(screen.getByRole('navigation',{name:'项目训练步骤'})).getAllByRole('link').map(link => link.textContent)).toEqual(['项目概览','1训练数据','2训练参数','3训练结果']);
    fireEvent.click(screen.getByRole('button',{name:'保存草稿'})); await screen.findByTestId('draft-saved');
    expect(writes.at(-1)).toMatchObject({model:{dit_path:'/models/unsaved.safetensors'},loop:{epochs:19}});
    expect(sessionStorage.getItem('model-draft:p_model:v2')).toBeNull();
    expect(sessionStorage.getItem('model-draft:p_model:v1')).toContain('other-version');
  });

  it('keeps query/state and edits through tab history, and refreshes selectable local models without replacing the path', async () => {
    show('/projects/p_model/v/v2/train?tab=model&retained=1');
    const input = await screen.findByDisplayValue('/models/original.safetensors');
    fireEvent.change(input,{target:{value:'/models/chosen.safetensors'}});
    fireEvent.click(screen.getByRole('tab',{name:'训练参数'}));
    expect(screen.getByTestId('route')).toHaveTextContent('tab=train&retained=1');
    await act(async () => fireEvent.click(screen.getByRole('button',{name:'Browser back'})));
    expect(screen.getByRole('tab',{name:'底模与输出'})).toHaveAttribute('aria-selected','true');
    expect(screen.getByDisplayValue('/models/chosen.safetensors')).toBeInTheDocument();
    expect(screen.getByTestId('route')).toHaveTextContent('"origin":"kept"');
    server.use(http.get('/api/models',()=>HttpResponse.json([
      {id:'downloaded',family:'anima',kind:'dit',path:'/models/downloaded.safetensors',exists:true},
      {id:'missing',family:'anima',kind:'dit',path:'/models/missing.safetensors',exists:false},
    ])));
    fireEvent(window,new Event('studio-models-changed'));
    const registered = await screen.findByRole('combobox',{name:/主模型 \/ DiT.*从已注册模型选择/});
    fireEvent.click(registered);
    expect(screen.getByRole('option',{name:'[anima] /models/downloaded.safetensors'})).toBeInTheDocument();
    expect(screen.queryByRole('option',{name:/missing/})).not.toBeInTheDocument();
    expect(screen.getByDisplayValue('/models/chosen.safetensors')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('option',{name:'[anima] /models/downloaded.safetensors'}));
    await waitFor(()=>expect(screen.getByDisplayValue('/models/downloaded.safetensors')).toBeInTheDocument());
  });
});
