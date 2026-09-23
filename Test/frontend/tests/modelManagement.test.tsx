import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import Models from '../../../frontend/src/pages/Models/Models';
import Preferences from '../../../frontend/src/pages/Settings/Preferences';
import { ModelAsset, ModelDownload, Settings } from '../../../frontend/src/api/types';
import i18n from '../../../frontend/src/i18n';
import { handlers } from '../mocks/handlers';

const server = setupServer(...handlers);
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => { cleanup(); server.resetHandlers(); vi.restoreAllMocks(); });
afterAll(() => server.close());
let models: ModelAsset[];
let downloads: ModelDownload[];
let settings: Settings;
let patch = vi.fn();
let download = vi.fn();
let catalogDownload = vi.fn();
let useRecommendation = vi.fn();
let cancel = vi.fn();
let register = vi.fn();
const entries = [
  {id:'anima-dit',family:'anima',kind:'dit',name:'Anima base',filename:'anima.safetensors'},
  {id:'anima-encoder',family:'anima',kind:'text_encoder',name:'Anima text encoder',filename:'encoder.safetensors'},
  {id:'anima-vae',family:'anima',kind:'vae',name:'Shared VAE',filename:'vae.safetensors'},
];
function task(overrides: Partial<ModelDownload> = {}): ModelDownload {
  return {id:'dl1',provider:'huggingface',mirror:'official',family:'anima',kind:'text_encoder',source_url:'https://huggingface.co/official/anima/resolve/main/encoder.safetensors',filename:'encoder.safetensors',target_path:'D:\\models\\encoder.safetensors',status:'downloading',bytes_per_second:0,eta_seconds:null,progress_at:null,downloaded_bytes:500,total_bytes:1000,error:null,model_id:null,dtype:'bf16',is_default:true,purpose:'training',created_at:1,finished_at:null,...overrides};
}
beforeEach(async () => {
  await i18n.changeLanguage('zh-CN');
  models = [
    {id:'a',family:'anima',kind:'dit',path:'C:\\models\\anima.safetensors',dtype:'bf16',exists:true,is_default:false,purpose:'training',size:1024,created_at:1},
    {id:'k',family:'krea2',kind:'dit',path:'C:\\models\\krea2.safetensors',dtype:'bf16',exists:true,is_default:true,purpose:'training',variant:'raw',size:1024,created_at:1},
  ];
  downloads=[];
  settings={paths:{bootstrap_env_dir:'',data_root:'C:\\studio',models_dir:'D:\\models',cache_dir:'D:\\cache',output_dir:'D:\\runs',output_mode:'project'},server:{host:'127.0.0.1',port:8765},ui:{language:'zh-CN',theme:'light'}};
  patch=vi.fn();download=vi.fn();cancel=vi.fn();catalogDownload=vi.fn();useRecommendation=vi.fn();register=vi.fn();
  server.use(
    http.get('/api/families',()=>HttpResponse.json(['anima','krea2'].map(name=>({name,label:name==='anima'?'Anima':'Krea 2',weights:[{field:'dit_path'},{field:'text_encoder_path'},{field:'vae_path'}]})))),
    http.get('/api/models/credentials',()=>HttpResponse.json({huggingface:{configured:false},modelscope:{configured:false}})),
    http.get('/api/models',()=>HttpResponse.json(models)),
    http.get('/api/settings',()=>HttpResponse.json(settings)),
    http.put('/api/settings',async({request})=>{settings=await request.json() as Settings;return HttpResponse.json(settings);}),
    http.get('/api/models/downloads',()=>HttpResponse.json(downloads)),
    http.get('/api/models/recommendations',()=>HttpResponse.json(entries.map(entry=>{
      const model=models.find(model=>model.family===entry.family && model.kind===entry.kind);
      return {...entry,dtype:'bf16',size:1024,recommended:true,model_id:model?.id||null,available_path:model?.exists?model.path:null,is_default:model?.is_default||false,
        sources:['huggingface','modelscope'].map(provider=>({provider,repo_id:`official/${entry.family}`,filename:`original/${entry.filename}`,revision:provider==='modelscope'?'master':'main',url:`https://${provider==='modelscope'?'modelscope.cn':'huggingface.co'}/official/${entry.family}/${entry.filename}`}))};
    }))),
    http.post('/api/models/recommendations/:id/use',({params})=>{
      useRecommendation(params.id);models=models.map(model=>model.id==='a'?{...model,is_default:true}:model);return HttpResponse.json(models[0]);
    }),
    http.post('/api/models/recommendations/:id/download',async({params,request})=>{
      const body=await request.json() as {provider:'huggingface'|'modelscope';is_default:boolean};catalogDownload(params.id,body);
      const entry=entries.find(entry=>entry.id===params.id)!;const started=task({kind:entry.kind,filename:entry.filename,provider:body.provider,recommendation_id:entry.id});downloads.unshift(started);return HttpResponse.json(started,{status:202});
    }),
    http.patch('/api/models/:id',async({params,request})=>{
      const body=await request.json() as {is_default:boolean};patch(params.id,body);const row=models.find(model=>model.id===params.id)!;
      models=models.map(model=>model.family===row.family && model.kind===row.kind?{...model,is_default:model.id===row.id && body.is_default}:model);return HttpResponse.json(models.find(model=>model.id===params.id));
    }),
    http.post('/api/models',async({request})=>{
      const body=await request.json() as ModelAsset;register(body);const row={...body,id:'registered',exists:true,created_at:1};models.push(row);return HttpResponse.json(row);
    }),
    http.post('/api/models/downloads',async({request})=>{
      const body=await request.json() as {url?:string;repo_id?:string;filename?:string;family:string;kind:string;provider:'huggingface'|'modelscope'};download(body);
      downloads=[task({family:body.family,kind:body.kind,provider:body.provider,source_url:body.url||`https://server-resolved/${body.repo_id}/${body.filename}`})];return HttpResponse.json(downloads[0],{status:202});
    }),
    http.post('/api/models/downloads/:id/cancel',({params})=>{cancel(params.id);downloads=downloads.map(task=>task.id===params.id?{...task,status:'cancelled'}:task);return HttpResponse.json(downloads.find(task=>task.id===params.id));}),
  );
});
function Location(){const location=useLocation();return <output data-testid="model-location">{location.pathname}{location.search}</output>;}
function mount(element:React.ReactNode,path='/models?family=anima'){
  return render(<QueryClientProvider client={new QueryClient({defaultOptions:{queries:{retry:false}}})}><MemoryRouter initialEntries={[path]}>{element}<Location/></MemoryRouter></QueryClientProvider>);
}
function choose(name:string,option:string){fireEvent.click(screen.getByRole('combobox',{name}));fireEvent.click(screen.getByRole('option',{name:option}));}
async function openCustom(){fireEvent.click(await screen.findByTestId('download-model-btn'));return screen.getByRole('dialog',{name:'自定义下载'});}
function enterRepo(){fireEvent.change(screen.getByRole('textbox',{name:'仓库 ID'}),{target:{value:'my/weights'}});fireEvent.change(screen.getByRole('textbox',{name:'仓库内文件路径'}),{target:{value:'weights/custom.safetensors'}});}

describe('real model management UI contracts',()=>{
  it('clears the DiT variant when switching a custom download to a shared VAE',async()=>{
    mount(<Models/>,'/models?family=krea2');
    const dialog=await openCustom();
    choose('Krea 2 版本 / 用途','Turbo · 仅采样');
    expect(within(dialog).getByRole('checkbox')).toBeDisabled();
    choose('组件','VAE');
    expect(within(dialog).getByRole('checkbox')).toBeEnabled();
    fireEvent.click(within(dialog).getByRole('checkbox'));
    enterRepo();
    fireEvent.click(screen.getByTestId('model-download-start'));
    await waitFor(()=>expect(download).toHaveBeenCalledWith(expect.objectContaining({family:'krea2',kind:'vae',is_default:true})));
    expect(download.mock.calls[0][0]).not.toHaveProperty('variant');
  });
  it('labels Turbo as sampling only and blocks setting it as a training default',async()=>{
    models.push({...models[1],id:'turbo',path:'C:\\models\\turbo.safetensors',is_default:false,purpose:'inference',variant:'turbo'} as ModelAsset);
    mount(<Models/>,'/models?family=krea2&view=library');
    expect(await screen.findByText('turbo.safetensors')).toBeInTheDocument();
    expect(screen.getByText('仅采样')).toBeInTheDocument();
    expect(screen.getByRole('button',{name:'设为默认 turbo.safetensors'})).toBeDisabled();
    expect(screen.getByRole('button',{name:'设为默认 krea2.safetensors'})).toBeEnabled();
    fireEvent.click(screen.getByRole('tab',{name:/准备模型/}));
    const card=await screen.findByTestId('model-component-dit');
    fireEvent.click(within(card).getByRole('combobox'));
    expect(screen.queryByRole('option',{name:'turbo.safetensors'})).not.toBeInTheDocument();
    expect(screen.getByRole('option',{name:'krea2.safetensors'})).toBeInTheDocument();
  });
  it('requires an explicit local Krea variant and registers Turbo without a training default',async()=>{
    server.use(http.post('/api/models/inspect',()=>HttpResponse.json({path:'D:\\unknown.safetensors',family:'krea2',family_candidates:['krea2'],kind:'dit',dtype:'bf16',dtypes:{BF16:100},confidence:'high',evidence:[],warnings:[],files_inspected:1,variant:null,purpose:null})));
    mount(<Models/>,'/models?family=krea2');
    fireEvent.click(await screen.findByTestId('add-model-btn'));
    const dialog=screen.getByRole('dialog',{name:'添加本地模型'});
    fireEvent.change(within(dialog).getByRole('textbox',{name:'文件路径'}),{target:{value:'D:\\unknown.safetensors'}});
    await screen.findByRole('combobox',{name:'Krea 2 版本 / 用途'});
    expect(screen.getByTestId('add-model-submit')).toBeDisabled();
    choose('Krea 2 版本 / 用途','Turbo · 仅采样');
    expect(within(dialog).getByRole('checkbox')).toBeDisabled();
    fireEvent.click(screen.getByTestId('add-model-submit'));
    await waitFor(()=>expect(register).toHaveBeenCalledWith(expect.objectContaining({family:'krea2',kind:'dit',variant:'turbo',purpose:'inference',is_default:false})));
  });
  it('keeps local files and downloads usable when the recommendation list fails, then clears the recovered error',async()=>{
    let failing=true;
    downloads=[task()];
    server.use(http.get('/api/models/recommendations',()=> failing ? HttpResponse.json({detail:'catalog unavailable'},{status:503}) : HttpResponse.json([])));
    mount(<Models/>,'/models?family=anima&view=library');
    expect(await screen.findByText('C:\\models\\anima.safetensors')).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveTextContent('推荐模型：Service Unavailable');
    expect(await screen.findByText('encoder.safetensors')).toBeInTheDocument();
    failing=false;
    fireEvent.click(screen.getByRole('button',{name:'重新读取'}));
    await waitFor(()=>expect(screen.queryByRole('alert')).not.toBeInTheDocument());
    expect(screen.getByText('encoder.safetensors')).toBeInTheDocument();
  });
  it('shows training assets without retired automatic-tagging entry points',async()=>{
    downloads=[task({id:'legacy',family:'tagger',kind:'tagger',filename:'model.onnx',status:'failed'})];
    const oldCatalog=vi.fn();server.use(http.get('/api/models/catalog',()=>{oldCatalog();return HttpResponse.json([]);}));
    mount(<Models/>,'/models?family=anima&view=library');
    expect(await screen.findByText('C:\\models\\anima.safetensors')).toBeInTheDocument();
    expect(screen.queryByText(/自动打标|ONNX Runtime|WD14/)).not.toBeInTheDocument();expect(screen.queryByText('model.onnx')).not.toBeInTheDocument();
    expect(oldCatalog).not.toHaveBeenCalled();expect(download).not.toHaveBeenCalled();
  });
  it('sets only the selected family component default through its local picker',async()=>{
    mount(<Models/>);const card=await screen.findByTestId('model-component-dit');
    fireEvent.click(within(card).getByRole('combobox'));expect(screen.queryByRole('option',{name:'krea2.safetensors'})).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('option',{name:'anima.safetensors'}));
    await waitFor(()=>expect(patch).toHaveBeenCalledWith('a',{is_default:true}));
    await waitFor(()=>expect(within(card).getByRole('combobox')).toHaveTextContent('anima.safetensors'));
    expect(models.find(model=>model.id==='k')?.is_default).toBe(true);
  });
  it('uses a catalog asset already on disk without starting a new download',async()=>{
    mount(<Models/>);const card=await screen.findByTestId('model-component-dit');
    fireEvent.click(within(card).getByRole('button',{name:'设为默认'}));
    await waitFor(()=>expect(useRecommendation).toHaveBeenCalledWith('anima-dit'));
    expect(catalogDownload).not.toHaveBeenCalled();expect(await screen.findByText('已设为默认组件。')).toBeInTheDocument();
  });
  it('submits the ModelScope catalog identity/provider without constructing a file URL',async()=>{
    mount(<Models/>);const card=await screen.findByTestId('model-component-text_encoder');
    choose('下载来源','魔搭 ModelScope');
    expect(within(card).getByRole('link',{name:'发布页'})).toHaveAttribute('href','https://modelscope.cn/official/anima/encoder.safetensors');
    fireEvent.click(within(card).getByRole('button',{name:'下载'}));
    await waitFor(()=>expect(catalogDownload).toHaveBeenCalledWith('anima-encoder',{provider:'modelscope',is_default:true}));
    expect(download).not.toHaveBeenCalled();
    const progress=await screen.findByRole('progressbar');
    expect(progress).toHaveAttribute('value','500');expect(progress).toHaveAttribute('max','1000');
    expect(screen.getByRole('button',{name:'下载中'})).toBeDisabled();expect(cancel).not.toHaveBeenCalled();expect(models.some(model=>model.kind==='text_encoder')).toBe(false);
  });
  it('keeps only the current catalog attempt through retry, transfer and completion with retained server history',async()=>{
    downloads=[
      task({id:'older-failed',recommendation_id:'anima-encoder',created_at:1,status:'failed',error:'Old network failure'}),
      task({id:'failed',recommendation_id:'anima-encoder',created_at:2,status:'failed',error:'<urlopen error timed out>'}),
    ];
    const retry=vi.fn();server.use(http.post('/api/models/downloads/failed/retry',()=>{
      retry();const started=task({id:'retry',recommendation_id:'anima-encoder',created_at:3,status:'queued',downloaded_bytes:0});
      downloads.push(started);return HttpResponse.json(started,{status:202});
    }));
    mount(<Models/>,'/models?family=anima&view=downloads');
    const card=await screen.findByTestId('model-component-text_encoder');
    expect(await within(card).findByRole('alert')).toHaveTextContent('下载失败：<urlopen error timed out>');
    expect(within(card).getByRole('alert')).toHaveClass('model-download-error');
    expect(screen.queryByText(/Old network failure/)).not.toBeInTheDocument();
    expect(screen.queryByRole('region',{name:'下载状态'})).not.toBeInTheDocument();
    expect(screen.queryByRole('tab',{name:/下载/})).not.toBeInTheDocument();
    expect(screen.queryByRole('button',{name:'更换来源'})).not.toBeInTheDocument();
    fireEvent.click(within(card).getByRole('button',{name:'重试'}));await waitFor(()=>expect(retry).toHaveBeenCalledOnce());
    expect(await within(card).findByRole('button',{name:'下载中'})).toBeDisabled();
    expect(within(card).getByRole('button',{name:'下载中'})).toHaveClass('model-button-downloading');
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.queryByRole('region',{name:'下载状态'})).not.toBeInTheDocument();
    downloads=downloads.map(row=>row.id==='retry'?{...row,status:'downloading',downloaded_bytes:750}:row);
    fireEvent.click(screen.getByRole('button',{name:'刷新模型'}));
    await waitFor(()=>expect(within(card).getByRole('progressbar')).toHaveAttribute('value','750'));
    expect(within(card).getByRole('button',{name:'下载中'})).toBeDisabled();
    downloads=downloads.map(row=>row.id==='retry'?{...row,status:'completed',downloaded_bytes:1000}:row);
    models.push({...models[0],id:'encoder-ready',kind:'text_encoder',path:'D:\\models\\encoder.safetensors',is_default:true});
    fireEvent.click(screen.getByRole('button',{name:'刷新模型'}));
    await within(card).findByText('当前默认');
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.queryByRole('button',{name:'重试'})).not.toBeInTheDocument();
    expect(screen.queryByRole('button',{name:'下载中'})).not.toBeInTheDocument();
    expect(screen.queryByRole('region',{name:'下载状态'})).not.toBeInTheDocument();
    expect(downloads).toHaveLength(3);
  });
  it('keeps independent same-name custom downloads separate without resurrecting their completed history',async()=>{
    const first={filename:'encoder.safetensors',target_path:'D:\\models\\custom-one\\encoder.safetensors',source_url:'https://huggingface.co/one/model/resolve/main/encoder.safetensors'};
    const second={filename:'encoder.safetensors',target_path:'D:\\models\\custom-two\\encoder.safetensors',source_url:'https://huggingface.co/two/model/resolve/main/encoder.safetensors'};
    downloads=[
      task({...first,id:'one-failed',created_at:1,status:'failed',error:'Completed history failure'}),
      task({...second,id:'two-failed',created_at:2,status:'failed',error:'Custom model timed out'}),
      task({...first,id:'one-complete',created_at:3,status:'completed'}),
    ];
    server.use(http.post('/api/models/downloads/two-failed/retry',()=>{
      const started=task({...second,id:'two-retry',created_at:4,status:'downloading'});downloads.unshift(started);
      return HttpResponse.json(started,{status:202});
    }));
    mount(<Models/>);
    const state=await screen.findByRole('region',{name:'下载状态'});
    expect(within(state).getAllByText('encoder.safetensors')).toHaveLength(1);
    expect(within(state).getByRole('alert')).toHaveTextContent('Custom model timed out');
    expect(screen.queryByText(/Completed history failure/)).not.toBeInTheDocument();
    const card=screen.getByTestId('model-component-text_encoder');
    expect(within(card).queryByRole('alert')).not.toBeInTheDocument();
    expect(within(card).getByRole('button',{name:'下载'})).toBeEnabled();
    fireEvent.click(within(state).getByRole('button',{name:'重试'}));
    expect(await within(state).findByRole('button',{name:'下载中'})).toBeDisabled();
    expect(within(state).getByRole('button',{name:'下载中'})).toHaveClass('model-button-downloading');
    expect(within(state).queryByRole('alert')).not.toBeInTheDocument();
    expect(within(state).getAllByText('encoder.safetensors')).toHaveLength(1);
    downloads=downloads.map(row=>row.id==='two-retry'?{...row,status:'completed'}:row);
    fireEvent.click(screen.getByRole('button',{name:'刷新模型'}));
    await waitFor(()=>expect(screen.queryByRole('region',{name:'下载状态'})).not.toBeInTheDocument());
    fireEvent.click(screen.getByRole('tab',{name:'本地模型 · 1'}));
    expect(await screen.findByText('anima.safetensors')).toBeInTheDocument();
  });
  it('preferences broadcasts saved appearance without duplicating model settings',async()=>{
    mount(<Preferences/>,'/settings/preferences?section=interface');const page=await screen.findByTestId('settings-page');
    expect(within(page).queryByLabelText(i18n.t('models.kind_dit'))).not.toBeInTheDocument();const changed=vi.fn();window.addEventListener('studio.settings.changed',changed);
    fireEvent.click(screen.getByTestId('settings-theme'));fireEvent.click(screen.getByRole('option',{name:i18n.t('settings.themeDark')}));fireEvent.click(screen.getByTestId('settings-save-btn'));
    await waitFor(()=>expect(changed).toHaveBeenCalled());expect(document.documentElement.classList.contains('dark')).toBe(true);expect(settings.ui.theme).toBe('dark');
    window.removeEventListener('studio.settings.changed',changed);document.documentElement.classList.remove('dark');
  });
  it('keeps server downloads running when settings closes and restores progress on reopen',async()=>{
    const view=mount(<Models embedded/>);await openCustom();enterRepo();fireEvent.click(screen.getByTestId('model-download-start'));
    await screen.findByRole('progressbar');view.unmount();expect(cancel).not.toHaveBeenCalled();
    mount(<Models embedded/>,'/settings/environment?tab=models&family=anima');
    expect(await screen.findByRole('progressbar')).toHaveAttribute('value','500');expect(cancel).not.toHaveBeenCalled();
  });
  it('registers local files through a dialog and keeps the selected family and actual path',async()=>{
    server.use(http.post('/api/models/inspect',()=>HttpResponse.json({path:'D:\\shared\\vae.safetensors',family:null,family_candidates:['anima','krea2'],kind:'vae',dtype:'fp32',dtypes:{F32:100},confidence:'partial',evidence:['Shared VAE'],warnings:[],files_inspected:1})));
    mount(<Models embedded/>);fireEvent.click(await screen.findByTestId('add-model-btn'));const dialog=screen.getByRole('dialog',{name:'添加本地模型'});
    fireEvent.change(within(dialog).getByRole('textbox',{name:'文件路径'}),{target:{value:'D:\\shared\\vae.safetensors'}});
    await waitFor(()=>expect(screen.getByTestId('add-model-submit')).toBeEnabled());
    expect(within(dialog).getByRole('combobox',{name:'组件'})).toHaveTextContent('VAE');
    fireEvent.click(screen.getByTestId('add-model-submit'));
    await waitFor(()=>expect(register).toHaveBeenCalledWith(expect.objectContaining({family:'anima',kind:'vae',path:'D:\\shared\\vae.safetensors'})));
    await waitFor(()=>expect(screen.queryByRole('dialog')).not.toBeInTheDocument());expect(screen.getByTestId('model-location')).toHaveTextContent('view=library');
  });
  it('preserves a typed custom source while changing component and platform',async()=>{
    mount(<Models/>);await openCustom();choose('来源格式','文件链接');const custom='https://huggingface.co/my/repo/blob/main/custom.safetensors';
    fireEvent.change(screen.getByTestId('model-download-url'),{target:{value:custom}});choose('组件','VAE');choose('下载平台','魔搭 ModelScope');
    expect(screen.getByTestId('model-download-url')).toHaveValue(custom);fireEvent.click(screen.getByTestId('model-download-start'));
    await waitFor(()=>expect(download).toHaveBeenCalledWith(expect.objectContaining({url:custom,provider:'modelscope',kind:'vae',mirror:'official'})));
  });
  it('retains custom inputs and actionable field errors when the service rejects a download',async()=>{
    server.use(http.post('/api/models/downloads',()=>HttpResponse.json({error:{code:'validation',message:'request validation failed',details:{errors:[{loc:['body','url'],msg:'Only a model file URL is accepted'}]}}},{status:422})));
    mount(<Models/>);await openCustom();enterRepo();choose('组件','VAE');fireEvent.click(screen.getByTestId('model-download-start'));
    expect(await screen.findByRole('alert')).toHaveTextContent('Only a model file URL is accepted');
    expect(screen.getByRole('textbox',{name:'仓库 ID'})).toHaveValue('my/weights');expect(screen.getByRole('textbox',{name:'仓库内文件路径'})).toHaveValue('weights/custom.safetensors');
    expect(screen.getByRole('combobox',{name:'组件'})).toHaveTextContent('VAE');expect(screen.getByTestId('model-download-start')).toBeEnabled();expect(screen.getByRole('dialog')).toBeInTheDocument();
  });
  it('retries a failed server task without replacing its source with a client URL',async()=>{
    downloads=[task({id:'failed',provider:'modelscope',status:'failed',error:'HTTP 403: save credentials'})];const retry=vi.fn();
    server.use(http.post('/api/models/downloads/failed/retry',()=>{retry();const started=task({id:'retry',provider:'modelscope',created_at:2,status:'queued'});downloads.unshift(started);return HttpResponse.json(started,{status:202});}));
    mount(<Models/>,'/models?family=anima&view=downloads');fireEvent.click(await screen.findByRole('button',{name:'重试'}));await waitFor(()=>expect(retry).toHaveBeenCalledOnce());expect(download).not.toHaveBeenCalled();
  });
  it('keeps completed downloads out of the model preparation page',async()=>{
    downloads=Array.from({length:26},(_,index)=>task({id:`history-${index}`,filename:`history-${index}.safetensors`,status:'completed'}));downloads.push(task({id:'krea-history',family:'krea2',filename:'krea-history.safetensors',status:'completed'}));
    mount(<Models/>,'/models?family=anima&view=downloads');await screen.findByRole('tab',{name:'准备模型'});
    expect(screen.queryByRole('tab',{name:/下载/})).not.toBeInTheDocument();expect(screen.queryByText('history-0.safetensors')).not.toBeInTheDocument();expect(screen.queryByText('history-25.safetensors')).not.toBeInTheDocument();
  });
});

it.each(['huggingface','modelscope'])('shows both Klein base sizes and submits the 9B single-file entry through %s', async(provider)=>{
  const catalog=[
    ['flux2-klein-base-4b','dit','FLUX.2 Klein Base 4B · BF16'],
    ['flux2-klein-base-9b','dit','FLUX.2 Klein Base 9B · BF16'],
    ['flux2-qwen3-4b','text_encoder','Qwen3 4B · Klein 4B 文本编码器'],
    ['flux2-qwen3-8b','text_encoder','Qwen3 8B · Klein 9B 文本编码器'],
    ['flux2-vae','vae','FLUX.2 VAE · Klein 4B / 9B 共用'],
  ].map(([id,kind,name])=>({id,kind,name,family:'flux2',dtype:'bf16',size:1024,recommended:true,purpose:'training',model_id:null,available_path:null,is_default:false,sources:['huggingface','modelscope'].map(source=>({provider:source,url:`https://${source==='modelscope'?'modelscope.cn/models':'huggingface.co'}/black-forest-labs/FLUX.2-klein-base-9B`,repo_id:'official/klein',filename:id+'.safetensors',revision:'pinned'}))}));
  const request=vi.fn();
  server.use(
    http.get('/api/families',()=>HttpResponse.json([{name:'flux2',label:'FLUX.2 Klein 4B / 9B',weights:['dit','text_encoder','vae'].map(kind=>({field:kind==='dit'?'dit_path':kind+'_path',kind,required:true,downloadable:true}))}])),
    http.get('/api/models/recommendations',()=>HttpResponse.json(catalog)),
    http.post('/api/models/recommendations/:id/download',async({params,request:incoming})=>{request(params.id,await incoming.json());return HttpResponse.json(task({family:'flux2',kind:'dit'}),{status:202});}),
  );
  mount(<Models/>,'/models?family=flux2&view=prepare');
  expect(await screen.findByText('FLUX.2 Klein Base 4B · BF16')).toBeVisible();
  const title=await screen.findByText('FLUX.2 Klein Base 9B · BF16');
  expect(await screen.findByText('Qwen3 8B · Klein 9B 文本编码器')).toBeVisible();
  expect(screen.getByText('FLUX.2 VAE · Klein 4B / 9B 共用')).toBeVisible();
  if(provider==='modelscope') choose('下载来源','魔搭 ModelScope');
  const row=title.closest('.model-catalog-row') as HTMLElement;
  expect(within(row).getByRole('link',{name:'发布页'})).toHaveAttribute('href',`https://${provider==='modelscope'?'modelscope.cn/models':'huggingface.co'}/black-forest-labs/FLUX.2-klein-base-9B`);
  expect(within(row).getByRole('button',{name:'下载'})).toBeEnabled();
  fireEvent.click(within(title.closest('.model-catalog-row') as HTMLElement).getByRole('button',{name:'下载'}));
  await waitFor(()=>expect(request).toHaveBeenCalledWith('flux2-klein-base-9b',{provider,is_default:true}));
});
