import { afterAll, afterEach, beforeAll, beforeEach, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import LocalModelRegistration, { type ModelInspection } from '../../../frontend/src/pages/Models/LocalModelRegistration';
import type { FamilyInfo } from '../../../frontend/src/api/types';
import i18n from '../../../frontend/src/i18n';
const server=setupServer();
beforeAll(()=>server.listen({onUnhandledRequest:'error'}));
afterEach(()=>{cleanup();server.resetHandlers();});
afterAll(()=>server.close());
beforeEach(async()=>{await i18n.changeLanguage('zh-CN');});
const result=(patch:Partial<ModelInspection>={}):ModelInspection=>({path:'/models/real.safetensors',family:'krea2',family_candidates:['krea2'],kind:'dit',dtype:'fp16',dtypes:{F16:12},confidence:'high',evidence:['Krea structure'],warnings:[],files_inspected:1,...patch});
const families = ['anima', 'krea2'].map(name => ({ name, label: name === 'anima' ? 'Anima' : 'Krea 2', weights: [{ field: 'dit_path' }, { field: 'text_encoder_path' }, { field: 'vae_path' }] })) as FamilyInfo[];
function mount(){const saved=vi.fn(async()=>{});render(<LocalModelRegistration initialFamily="anima" families={families} onClose={vi.fn()} onRegistered={saved} onBusyChange={vi.fn()}/>);return saved;}
function path(value:string){fireEvent.change(screen.getByRole('textbox',{name:'文件路径'}),{target:{value}});}
function choose(label:string,option:string){fireEvent.click(screen.getByRole('combobox',{name:label}));fireEvent.click(screen.getByRole('option',{name:option}));}
it('uses detected family/type/precision and locks identified fields before registering',async()=>{
  const register=vi.fn();server.use(http.post('/api/models/inspect',()=>HttpResponse.json(result({variant:'raw',purpose:'training'}))),http.post('/api/models',async({request})=>{register(await request.json());return HttpResponse.json({});}));
  const saved=mount();expect(screen.getByRole('combobox',{name:'权重精度'})).toHaveTextContent('未知');path('/models/input.safetensors');await screen.findByTestId('model-inspection-status');
  expect(screen.getByRole('combobox',{name:'登记模型系列'})).toHaveTextContent('Krea 2');expect(screen.getByRole('combobox',{name:'Krea 2 版本 / 用途'})).toHaveTextContent('Raw');expect(screen.getByRole('combobox',{name:'Krea 2 版本 / 用途'})).toBeDisabled();expect(screen.getByRole('combobox',{name:'组件'})).toBeDisabled();expect(screen.getByRole('combobox',{name:'权重精度'})).toHaveTextContent('FP16');expect(screen.getByRole('combobox',{name:'权重精度'})).toBeDisabled();
  fireEvent.click(screen.getByTestId('add-model-submit'));await waitFor(()=>expect(saved).toHaveBeenCalled());expect(register).toHaveBeenCalledWith(expect.objectContaining({family:'krea2',kind:'dit',dtype:'fp16',path:'/models/real.safetensors',variant:'raw',purpose:'training'}));
});
it('leaves unknown component and precision explicit, without a BF16 default',async()=>{
  const register=vi.fn();server.use(http.post('/api/models/inspect',()=>HttpResponse.json(result({family:null,family_candidates:[],kind:null,dtype:null,confidence:'unknown',dtypes:{}}))),http.post('/api/models',async({request})=>{register(await request.json());return HttpResponse.json({});}));
  mount();path('/models/unknown.safetensors');await screen.findByTestId('model-inspection-status');expect(screen.getByTestId('add-model-submit')).toBeDisabled();choose('组件','VAE');fireEvent.click(screen.getByTestId('add-model-submit'));await waitFor(()=>expect(register).toHaveBeenCalledWith(expect.objectContaining({family:'anima',kind:'vae',dtype:null})));
});
it('does not let an old path response replace the current detected asset',async()=>{
  let release:()=>void=()=>{};const first=vi.fn();server.use(http.post('/api/models/inspect',async({request})=>{const body=await request.json() as {path:string};if(body.path==='/old'){first();await new Promise<void>(resolve=>{release=resolve;});return HttpResponse.json(result({family:'anima',family_candidates:['anima'],dtype:'bf16'}));}return HttpResponse.json(result());}));
  mount();path('/old');await waitFor(()=>expect(first).toHaveBeenCalled());path('/new');await screen.findByTestId('model-inspection-status');await act(async()=>{release();await Promise.resolve();});expect(screen.getByRole('combobox',{name:'登记模型系列'})).toHaveTextContent('Krea 2');expect(screen.getByRole('combobox',{name:'权重精度'})).toHaveTextContent('FP16');
});
it('keeps registration disabled after inspection failure and offers a real retry',async()=>{
  let failed=true;server.use(http.post('/api/models/inspect',()=>failed?HttpResponse.json({error:{message:'bad header',code:'model.inspect'}},{status:422}):HttpResponse.json(result())));mount();path('/bad');expect(await screen.findByRole('alert')).toHaveTextContent('bad header');expect(screen.getByTestId('add-model-submit')).toBeDisabled();failed=false;fireEvent.click(screen.getByRole('button',{name:'重新检测'}));await screen.findByTestId('model-inspection-status');expect(screen.getByTestId('add-model-submit')).toBeDisabled();choose('Krea 2 版本 / 用途','Raw · 训练与采样');await waitFor(()=>expect(screen.getByTestId('add-model-submit')).toBeEnabled());
});
