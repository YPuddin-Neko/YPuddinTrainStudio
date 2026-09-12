import {fireEvent,render,screen,waitFor} from '@testing-library/react';
import {afterAll,afterEach,beforeAll,beforeEach,describe,expect,it,vi} from 'vitest';
import {MemoryRouter,Routes,Route} from 'react-router-dom';
import {QueryClient,QueryClientProvider} from '@tanstack/react-query';
import {http,HttpResponse} from 'msw';
import {setupServer} from 'msw/node';
import {handlers} from '../src/mocks/handlers';
import ProjectDetail from '../src/pages/ProjectDetail/ProjectDetail';
import TrainConfig from '../src/pages/TrainConfig/TrainConfig';
import {schemaDefaults} from '../src/utils/config';
import trainSchema from '../src/schema/train-schema.json';
import i18n from '../src/i18n';
const server=setupServer(...handlers);
vi.mock('../src/events/useEventStream',()=>({useEventStream:()=>{}}));
beforeAll(()=>server.listen({onUnhandledRequest:'error'}));
afterAll(()=>server.close());
afterEach(()=>{server.resetHandlers();sessionStorage.clear();vi.restoreAllMocks();});
let config:Record<string,any>;
let saved:Record<string,any>[];
const fail=(message:string)=>HttpResponse.json({error:{code:'unavailable',message}},{status:503});
beforeEach(async()=>{
  await i18n.changeLanguage('zh-CN');config=schemaDefaults(trainSchema);config.model.dit_path='/models/explicit.safetensors';config.loop.epochs=17;config.dataset.sources=[{path:'/version/photos',repeats:3}];saved=[];
  server.use(
    http.get('/api/projects/p_isolation',()=>HttpResponse.json({id:'p_isolation',name:'Isolation',active_version_id:'v2'})),
    http.get('/api/projects/p_isolation/versions',()=>HttpResponse.json([{id:'v2',project_id:'p_isolation',name:'v2',status:'ready',archived:false,paths:{root:'/version/v2'},stats:{images:1,jobs:0,artifacts:0}}])),
    http.get('/api/projects/p_isolation/config',({request})=>{expect(new URL(request.url).searchParams.get('version_id')).toBe('v2');return HttpResponse.json(config);}),
    http.put('/api/projects/p_isolation/config',async({request})=>{expect(new URL(request.url).searchParams.get('version_id')).toBe('v2');const body=await request.json() as Record<string,any>;saved.push(body);config=body;return HttpResponse.json(body);}),
    http.get('/api/projects/p_isolation/datasets',()=>HttpResponse.json([])),
    http.get('/api/models',()=>HttpResponse.json([])),
    http.get('/api/jobs',()=>HttpResponse.json({items:[],page:1,page_size:5,total:0})),
  );
});
function show(train=false){return render(<QueryClientProvider client={new QueryClient({defaultOptions:{queries:{retry:false}}})}><MemoryRouter initialEntries={[`/projects/p_isolation/v/v2${train?'/train':'?step=models'}`]}><Routes><Route path="/projects/:id/v/:versionId" element={<ProjectDetail/>}/><Route path="/projects/:id/v/:versionId/train" element={<TrainConfig/>}/></Routes></MemoryRouter></QueryClientProvider>);}

describe('workspace request isolation',()=>{
  it('keeps model paths editable on the merged page when the registry fails and retries without replacing edits',async()=>{
    let failed=true;server.use(http.get('/api/models',()=>failed?fail('registry offline'):HttpResponse.json([])));
    show();const path=await screen.findByDisplayValue('/models/explicit.safetensors');await waitFor(()=>expect(path).toBeEnabled());
    expect(await screen.findByTestId('training-auxiliary-error')).toHaveTextContent('registry offline');
    fireEvent.change(path,{target:{value:'/models/user-choice.safetensors'}});failed=false;fireEvent.click(screen.getByRole('button',{name:'重试辅助信息'}));
    await waitFor(()=>expect(screen.queryByTestId('training-auxiliary-error')).not.toBeInTheDocument());expect(path).toHaveValue('/models/user-choice.safetensors');
    fireEvent.click(screen.getByRole('button',{name:'保存草稿'}));await screen.findByTestId('draft-saved');
    expect(saved.at(-1)).toMatchObject({model:{dit_path:'/models/user-choice.safetensors'},loop:{epochs:17},dataset:{sources:[{path:'/version/photos',repeats:3}]}});
  });
  it('never mounts mutable model/data forms when the version config fails',async()=>{
    server.use(http.get('/api/projects/p_isolation/config',()=>fail('config unavailable')));show();
    expect(await screen.findByText(/config unavailable/)).toBeInTheDocument();expect(screen.queryByTestId('project-model-setup')).not.toBeInTheDocument();expect(saved).toHaveLength(0);
    fireEvent.click(screen.getByRole('link',{name:/^1\s*训练数据$/}));expect(screen.queryByTestId('project-data-import')).not.toBeInTheDocument();expect(screen.queryByTestId('datasets-empty')).not.toBeInTheDocument();
  });
  it('loads saved training config despite failed presets/models and retries supporting data without discarding edits',async()=>{
    let failed=true;
    server.use(http.get('/api/presets',()=>failed?fail('presets offline'):HttpResponse.json([])),http.get('/api/models',()=>failed?fail('registry offline'):HttpResponse.json([{id:'default',family:'anima',kind:'dit',path:'/models/other-default.safetensors',exists:true,is_default:true}])));
    show(true);const epochs=await screen.findByRole('spinbutton',{name:'loop.epochs'});expect(epochs).toHaveValue(17);
    expect(await screen.findByTestId('training-auxiliary-error')).toHaveTextContent('presets offline');expect(screen.getByTestId('training-auxiliary-error')).toHaveTextContent('registry offline');
    fireEvent.change(epochs,{target:{value:'23'}});failed=false;fireEvent.click(screen.getByRole('button',{name:'重试辅助信息'}));
    await waitFor(()=>expect(screen.queryByTestId('training-auxiliary-error')).not.toBeInTheDocument());expect(epochs).toHaveValue(23);
    fireEvent.click(screen.getByRole('button',{name:'保存草稿'}));await screen.findByTestId('draft-saved');
    expect(saved.at(-1)).toMatchObject({model:{dit_path:'/models/explicit.safetensors'},loop:{epochs:23},dataset:{sources:[{path:'/version/photos',repeats:3}]}});
  });
});
