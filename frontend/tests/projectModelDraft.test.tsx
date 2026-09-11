import React from 'react';
import {fireEvent,render,screen,waitFor,within} from '@testing-library/react';
import {afterAll,afterEach,beforeAll,beforeEach,describe,expect,it,vi} from 'vitest';
import {QueryClient,QueryClientProvider} from '@tanstack/react-query';
import {MemoryRouter,Routes,Route,Link,useLocation,useNavigate,type Location} from 'react-router-dom';
import {http,HttpResponse} from 'msw';
import {setupServer} from 'msw/node';
import ProjectModelSetup from '../src/pages/ProjectDetail/ProjectModelSetup';
import i18n from '../src/i18n';

const server=setupServer();
beforeAll(()=>server.listen({onUnhandledRequest:'error'}));
afterAll(()=>server.close());
afterEach(()=>{server.resetHandlers();sessionStorage.clear();vi.restoreAllMocks();});
let config:Record<string,any>;
let writes:Record<string,any>[];
beforeEach(async()=>{
  await i18n.changeLanguage('zh-CN');
  config={model:{family:'anima',dtype:'bf16',dit_path:'/models/original.safetensors'},dataset:{sources:[{path:'/data/original'}]},adapter:{preset:'attn'},checkpoint:{output_dir:'/outputs/version'},loop:{epochs:2}};
  writes=[];
  server.use(
    http.get('/api/families',()=>HttpResponse.json([{name:'anima',label:'Anima',weights:[{field:'dit_path',label:'DiT',hint:''}],default_preset:'attn'}])),
    http.get('/api/models',()=>HttpResponse.json([])),
    http.get('/api/projects/p_model/config',({request})=>{expect(new URL(request.url).searchParams.get('version_id')).toBe('v2');return HttpResponse.json(config);}),
    http.put('/api/projects/p_model/config',async({request})=>{const body=await request.json() as Record<string,any>;writes.push(body);config=body;return HttpResponse.json(body);}),
  );
});
function show(element:React.ReactNode){return render(<QueryClientProvider client={new QueryClient({defaultOptions:{queries:{retry:false}}})}><MemoryRouter initialEntries={['/projects/p_model/v/v2?step=models']}>{element}</MemoryRouter></QueryClientProvider>);}
function ModelPage({registerSave}:{registerSave?:(save:(()=>Promise<void>)|null)=>void}){
  return <><ProjectModelSetup projectId="p_model" versionId="v2" config={config} onSaved={()=>{}} registerSave={registerSave}/><Link to="/next">Leave project</Link></>;
}

describe('model draft save boundaries',()=>{
  it('waits for the registered save, blocks edits including an already-open select, and merges only model choices into current server config',async()=>{
    let finish:(()=>void)|undefined;
    server.use(http.put('/api/projects/p_model/config',async({request})=>{
      expect(new URL(request.url).searchParams.get('version_id')).toBe('v2');
      const body=await request.json() as Record<string,any>;writes.push(body);await new Promise<void>(resolve=>{finish=resolve;});config=body;return HttpResponse.json(body);
    }));
    function Host(){const save=React.useRef<(()=>Promise<void>)|null>(null);const [opened,setOpened]=React.useState(false);const register=React.useCallback((next:(()=>Promise<void>)|null)=>{save.current=next;},[]);return <><ModelPage registerSave={register}/><button onClick={()=>void save.current?.().then(()=>setOpened(true))}>Create version after saving</button>{opened&&<p>Version dialog opened</p>}</>;}
    show(<Host/>);
    const input=await screen.findByRole('textbox',{name:'DiT'});await waitFor(()=>expect(input).toBeEnabled());
    fireEvent.change(input,{target:{value:'/models/chosen.safetensors'}});
    config={...config,dataset:{sources:[{path:'/data/imported-while-editing'}]},loop:{epochs:19},checkpoint:{output_dir:'/outputs/custom'}};
    const dtype=screen.getByRole('combobox',{name:'计算精度'});fireEvent.click(dtype);
    fireEvent.click(screen.getByRole('button',{name:'Create version after saving'}));
    await waitFor(()=>expect(writes).toHaveLength(1));
    expect(input).toBeDisabled();expect(dtype).toBeDisabled();
    fireEvent.click(screen.getByRole('option',{name:'FP32'}));
    expect(dtype).toHaveTextContent('BF16');expect(screen.queryByText('Version dialog opened')).not.toBeInTheDocument();
    expect(writes[0]).toMatchObject({model:{dit_path:'/models/chosen.safetensors',dtype:'bf16'},dataset:{sources:[{path:'/data/imported-while-editing'}]},loop:{epochs:19},checkpoint:{output_dir:'/outputs/custom'}});
    finish!();await screen.findByText('Version dialog opened');expect(input).toBeEnabled();expect(sessionStorage.getItem('model-draft:p_model:v2')).toBeNull();
  });

  it('keeps the model editor and unsaved input behind settings, refreshes its registry without replacing paths, and restores it without writing',async()=>{
    function Host(){const location=useLocation();const navigate=useNavigate();const background=(location.state as {backgroundLocation?:Location}|null)?.backgroundLocation;return <><Routes location={background||location}><Route path="/projects/:id/v/:versionId" element={<ModelPage/>}/></Routes>{background&&<div role="dialog" aria-label="Model settings"><button onClick={()=>navigate(`${background.pathname}${background.search}`,{replace:true})}>Close settings</button></div>}</>;}
    show(<Host/>);
    const editor=await screen.findByTestId('project-model-setup');const input=await screen.findByRole('textbox',{name:'DiT'});await waitFor(()=>expect(input).toBeEnabled());
    fireEvent.change(input,{target:{value:'/models/unsaved-custom.safetensors'}});
    fireEvent.click(screen.getByRole('link',{name:'下载 / 管理模型'}));await screen.findByRole('dialog',{name:'Model settings'});
    expect(screen.getByTestId('project-model-setup')).toBe(editor);expect(writes).toHaveLength(0);
    server.use(http.get('/api/models',()=>HttpResponse.json([{id:'new-default',family:'anima',kind:'dit',path:'/models/downloaded.safetensors',exists:true,is_default:true}])));
    fireEvent(window,new Event('studio-models-changed'));
    fireEvent.click(within(editor).getByRole('combobox',{name:'从模型库选择 DiT'}));
    expect(await screen.findByRole('option',{name:'downloaded.safetensors · 默认'})).toBeInTheDocument();
    fireEvent.keyDown(within(editor).getByRole('combobox',{name:'从模型库选择 DiT'}),{key:'Escape'});
    fireEvent.click(screen.getByRole('button',{name:'Close settings'}));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();expect(screen.getByRole('textbox',{name:'DiT'})).toHaveValue('/models/unsaved-custom.safetensors');expect(writes).toHaveLength(0);
    expect(JSON.parse(sessionStorage.getItem('model-draft:p_model:v2')!)).toMatchObject({dit_path:'/models/unsaved-custom.safetensors'});
  });
});
