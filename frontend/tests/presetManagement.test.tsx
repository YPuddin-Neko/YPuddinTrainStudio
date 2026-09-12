import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { beforeAll, afterAll, beforeEach, afterEach, describe, expect, it } from 'vitest';
import { createMemoryRouter, Link, RouterProvider } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import Presets from '../src/pages/Presets/Presets';
import { handlers } from '../src/mocks/handlers';
import trainSchema from '../src/schema/train-schema.json';
import { schemaDefaults } from '../src/utils/config';
import type { Preset } from '../src/api/types';
import i18n from '../src/i18n';

const server = setupServer(...handlers);
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterAll(() => server.close());
afterEach(() => server.resetHandlers());
beforeEach(async () => { await i18n.changeLanguage('zh-CN'); });

function show(initial = '/presets') {
  const rows: Preset[] = [
    { name: 'builtin-anima', description: '内置人物训练参数', config: { model:{family:'anima'}, loop:{epochs:2}, optimizer:{lr:0.0002}, adapter:{rank:16,algo:'lora'} }, builtin:true, updated_at:null },
    { name: 'my-style', description: '自定义风格', config: { model:{family:'anima'}, loop:{epochs:4} }, builtin:false, updated_at:1 },
    { name: 'krea-base', description: 'Krea风格参数', config: { model:{family:'krea2'}, loop:{epochs:3} }, builtin:true, updated_at:null },
  ];
  const writes: {method:string;name?:string;body:any}[] = [];
  const requestedFamilies: string[] = [];
  server.use(
    http.get('/api/presets',()=>HttpResponse.json(rows)),
    http.get('/api/config/defaults',({request})=>{
      const family = new URL(request.url).searchParams.get('family') || 'anima'; requestedFamilies.push(family);
      const config = schemaDefaults(trainSchema); config.model.family=family;
      config.model.dit_path='/registered/dit'; config.model.tokenizer_path='/registered/tokenizer';
      config.dataset.sources=[{path:'/project/images'}]; config.dataset.cache_dir='/project/cache';
      config.sampling.output_dir='/project/samples'; config.sampling.prompts_file='/project/prompts'; config.adapter.resume_weights='/project/old-adapter';
      if(family==='krea2'){config.sampling.steps=28;config.sampling.cfg=5.5;config.dataset.text_encoding='cached';}
      return HttpResponse.json(config);
    }),
    http.post('/api/presets',async({request})=>{const body=await request.json() as any;writes.push({method:'POST',body});const row={...body,builtin:false,updated_at:2};rows.push(row);return HttpResponse.json(row);}),
    http.put('/api/presets/:name',async({request,params})=>{const body=await request.json() as any;writes.push({method:'PUT',name:String(params.name),body});return HttpResponse.json({...body,name:params.name,builtin:false,updated_at:3});}),
    http.delete('/api/presets/:name',({params})=>{writes.push({method:'DELETE',name:String(params.name),body:null});return HttpResponse.json({ok:true});}),
  );
  const router=createMemoryRouter([{path:'/presets',element:<><Presets/><Link to="/projects">离开预设</Link></>},{path:'/projects',element:<div>项目列表页</div>}],{initialEntries:[initial]});
  const client=new QueryClient({defaultOptions:{queries:{retry:false},mutations:{retry:false}}});
  render(<QueryClientProvider client={client}><RouterProvider router={router}/></QueryClientProvider>);
  return {rows,writes,requestedFamilies,router,client};
}
async function open(name: string) {
  fireEvent.click(await screen.findByRole('button',{name:`打开预设 ${name}`}));
  return screen.findByRole('spinbutton',{name:'loop.epochs'});
}

describe('independent preset management',()=>{
  it('groups/searches presets by family and shows descriptions and actual parameter summaries',async()=>{
    show();
    const anima=await screen.findByRole('button',{name:'打开预设 builtin-anima'});
    expect(anima).toHaveTextContent('内置人物训练参数');expect(anima).toHaveTextContent('Rank 16');expect(anima).toHaveTextContent('LR 0.0002');
    fireEvent.click(screen.getByRole('combobox',{name:'按模型筛选预设'}));fireEvent.click(screen.getByRole('option',{name:/Krea/}));
    expect(screen.queryByRole('button',{name:'打开预设 builtin-anima'})).not.toBeInTheDocument();
    expect(screen.getByRole('button',{name:'打开预设 krea-base'})).toBeInTheDocument();
    fireEvent.change(screen.getByRole('textbox',{name:'搜索预设'}),{target:{value:'没有此名称'}});
    expect(screen.getByText('暂无匹配预设。')).toBeInTheDocument();
  });

  it('keeps built-ins read only, then duplicates into editable schema fields without project files',async()=>{
    const state=show();const epochs=await open('builtin-anima');expect(epochs).toBeDisabled();
    const help=within(screen.getByTestId('field-loop.epochs')).getByRole('button',{name:/说明|help/});
    expect(help).toBeEnabled();fireEvent.click(help);expect(screen.getByRole('tooltip')).toBeInTheDocument();fireEvent.keyDown(document,{key:'Escape'});
    const group=screen.getByTestId('field-loop.epochs').closest('.config-group')!;
    const heading=within(group as HTMLElement).getByRole('button',{expanded:true});
    fireEvent.click(heading);expect(screen.queryByRole('spinbutton',{name:'loop.epochs'})).not.toBeInTheDocument();
    fireEvent.click(heading);expect(screen.getByRole('spinbutton',{name:'loop.epochs'})).toBeDisabled();
    expect(screen.queryByRole('button',{name:'保存预设'})).not.toBeInTheDocument();
    expect(screen.queryByRole('button',{name:'删除预设'})).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button',{name:'复制为新预设'}));
    await waitFor(()=>expect(screen.getByRole('spinbutton',{name:'loop.epochs'})).toBeEnabled());
    expect(screen.getByRole('textbox',{name:'预设名称'})).toHaveValue('builtin-anima-copy');
    fireEvent.change(screen.getByRole('spinbutton',{name:'loop.epochs'}),{target:{value:'7'}});
    fireEvent.click(screen.getByRole('button',{name:'保存预设'}));
    await screen.findByText('预设已保存，可在项目训练参数中加载。');
    expect(state.writes).toHaveLength(1);expect(state.writes[0].method).toBe('POST');
    const body=state.writes[0].body;expect(body.config.loop.epochs).toBe(7);expect(body.config.model.family).toBe('anima');
    for(const [group,key] of [['model','dit_path'],['model','tokenizer_path'],['dataset','sources'],['dataset','cache_dir'],['sampling','output_dir'],['sampling','prompts_file'],['adapter','resume_weights']])expect(body.config[group]).not.toHaveProperty(key);
    expect(state.rows[0].config.loop).toEqual({epochs:2});
    fireEvent.click(screen.getByRole('tab',{name:'精度与保存'}));
    expect(screen.queryByTestId('field-model.dit_path')).not.toBeInTheDocument();expect(screen.queryByTestId('field-checkpoint.resume')).not.toBeInTheDocument();
  });

  it('creates a Krea preset from actual family defaults while retaining its entered name',async()=>{
    const state=show();await screen.findByRole('button',{name:'打开预设 krea-base'});
    fireEvent.click(screen.getByRole('button',{name:'新建预设'}));await screen.findByRole('textbox',{name:'预设名称'});
    fireEvent.change(screen.getByRole('textbox',{name:'预设名称'}),{target:{value:'新的_Krea参数'}});
    fireEvent.click(screen.getByRole('combobox',{name:'适用模型'}));fireEvent.click(screen.getByRole('option',{name:/Krea/}));
    await waitFor(()=>expect(screen.getByRole('combobox',{name:'适用模型'})).toHaveTextContent('Krea'));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(screen.getByRole('textbox',{name:'预设名称'})).toHaveValue('新的_Krea参数');
    fireEvent.click(screen.getByRole('button',{name:'保存预设'}));await screen.findByText('预设已保存，可在项目训练参数中加载。');
    expect(state.requestedFamilies).toEqual(['anima','krea2']);expect(state.writes[0].body.config).toMatchObject({model:{family:'krea2'},sampling:{steps:28,cfg:5.5},dataset:{text_encoding:'cached'}});
  });

  it('retains failed edits and reports field errors, then explicitly updates the existing preset',async()=>{
    const state=show();const epochs=await open('my-style');
    let fail=true;server.use(http.put('/api/presets/my-style',async({request})=>{const body=await request.json() as any;if(fail)return HttpResponse.json({error:{code:'config.invalid',message:'invalid config',details:{errors:[{loc:'loop.epochs',msg:'must be positive'}]}}},{status:400});state.writes.push({method:'PUT',body,name:'my-style'});return HttpResponse.json({...body,builtin:false,updated_at:3});}));
    fireEvent.change(epochs,{target:{value:'8'}});fireEvent.change(screen.getByRole('textbox',{name:'用途与说明'}),{target:{value:'保留我的描述'}});
    fireEvent.click(screen.getByRole('button',{name:'保存预设'}));
    expect(await screen.findByRole('alert')).toHaveTextContent('loop.epochs: must be positive');expect(epochs).toHaveValue(8);expect(screen.getByRole('textbox',{name:'用途与说明'})).toHaveValue('保留我的描述');
    fail=false;fireEvent.click(screen.getByRole('button',{name:'保存预设'}));await screen.findByText('预设已保存，可在项目训练参数中加载。');
    expect(state.writes).toHaveLength(1);expect(state.writes[0]).toMatchObject({method:'PUT',name:'my-style',body:{description:'保留我的描述',config:{loop:{epochs:8}}}});
  });

  it('blocks real navigation until the draft is saved, preserves it on failure and permits retry',async()=>{
    const state=show();const epochs=await open('my-style');fireEvent.change(epochs,{target:{value:'9'}});
    let fail=true;server.use(http.put('/api/presets/my-style',async({request})=>fail?HttpResponse.json({error:{message:'Disk full',code:'storage.error'}},{status:500}):HttpResponse.json({...await request.json() as any,builtin:false,updated_at:3})));
    await act(async()=>{await state.router.navigate('/projects');});
    const dialog=await screen.findByRole('dialog',{name:'保存预设修改？'});fireEvent.click(within(dialog).getByRole('button',{name:'保存并继续'}));
    expect(await within(dialog).findByRole('alert')).toHaveTextContent('Disk full');expect(state.router.state.location.pathname).toBe('/presets');expect(epochs).toHaveValue(9);
    fail=false;fireEvent.click(within(dialog).getByRole('button',{name:'保存并继续'}));await screen.findByText('项目列表页');expect(state.router.state.location.pathname).toBe('/projects');
  });

  it('does not overwrite an edited draft when family defaults fail on a different preset',async()=>{
    show();const epochs=await open('my-style');fireEvent.change(epochs,{target:{value:'11'}});
    fireEvent.click(screen.getByRole('button',{name:'打开预设 krea-base'}));
    server.use(http.get('/api/config/defaults',()=>HttpResponse.json({error:{code:'read.failed',message:'defaults unavailable'}},{status:500})));
    fireEvent.click(screen.getByRole('button',{name:'放弃修改'}));
    expect(await screen.findByRole('alert')).toHaveTextContent('defaults unavailable');expect(epochs).toHaveValue(11);
    expect(screen.getByRole('textbox',{name:'预设名称'})).toHaveValue('my-style');
  });

  it('requires deletion confirmation and only removes the chosen custom preset after success',async()=>{
    const state=show();await open('my-style');fireEvent.click(screen.getByRole('button',{name:'删除预设'}));
    expect(state.writes).toHaveLength(0);fireEvent.click(screen.getByRole('button',{name:'取消'}));
    expect(screen.getByRole('textbox',{name:'预设名称'})).toHaveValue('my-style');
    fireEvent.click(screen.getByRole('button',{name:'删除预设'}));fireEvent.click(screen.getByRole('button',{name:'确认删除'}));
    await screen.findByText('预设已删除。');expect(state.writes).toEqual([{method:'DELETE',name:'my-style',body:null}]);
    expect(screen.queryByRole('button',{name:'打开预设 my-style'})).not.toBeInTheDocument();expect(screen.getByRole('button',{name:'打开预设 builtin-anima'})).toBeInTheDocument();
  });
});
