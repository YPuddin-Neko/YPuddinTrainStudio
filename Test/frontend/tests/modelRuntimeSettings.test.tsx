import {cleanup, fireEvent, render, screen, waitFor} from '@testing-library/react';
import {afterAll, afterEach, beforeAll, expect, it, vi} from 'vitest';
import {http, HttpResponse} from 'msw';
import {setupServer} from 'msw/node';
import {handlers} from '../mocks/handlers';
import {SchemaForm} from '../../../frontend/src/schema/SchemaForm/SchemaForm';
import trainSchema from '../../../frontend/src/schema/train-schema.json';
import '../../../frontend/src/i18n';

const server=setupServer(...handlers);
beforeAll(()=>server.listen({onUnhandledRequest:'error'}));
afterEach(()=>{cleanup();server.resetHandlers();});
afterAll(()=>server.close());

it.each([
  ['CUDA',['auto','sdpa','xformers','flash_attn'],['xFormers','FlashAttention 2'],['Metal FlashAttention · Apple']],
  ['Apple',['auto','sdpa','metal_flash'],['Metal FlashAttention · Apple'],['xFormers','FlashAttention 2']],
  ['CPU',['auto','sdpa'],[],['xFormers','FlashAttention 2','Metal FlashAttention · Apple']],
])('shows only %s attention backends returned by the service',(_platform,backends,present,absent)=>{
  render(<SchemaForm schema={trainSchema} value={{model:{family:'anima',attention:'auto'}}} family={{name:'anima',attention_backends:backends} as any} onChange={()=>{}} compact showAdvanced groupFilter={['memory']}/>);
  fireEvent.click(screen.getByRole('combobox',{name:'注意力后端'}));
  for(const label of present) expect(screen.getByRole('option',{name:label})).toBeInTheDocument();
  for(const label of absent) expect(screen.queryByRole('option',{name:label})).not.toBeInTheDocument();
});

it('offers follow-model precision and explains FP8 compute separately',()=>{
  const change=vi.fn();
  render(<SchemaForm schema={trainSchema} value={{model:{family:'krea2',dtype:'auto'}}} family={{name:'krea2',runtime_backend:'cuda'} as any} onChange={change} compact showAdvanced groupFilter={['model']}/>);
  const precision=screen.getByRole('combobox',{name:'底模加载精度'});
  expect(precision).toHaveTextContent('跟随模型');
  expect(screen.getByText(/受支持的 FP8 权重使用 BF16 计算/)).toBeInTheDocument();
  expect(screen.getByTestId('field-model.dtype')).not.toHaveTextContent(/CPU|Apple/);
  fireEvent.click(precision);
  fireEvent.click(screen.getByRole('option',{name:'FP16'}));
  expect(change).toHaveBeenLastCalledWith({model:{family:'krea2',dtype:'fp16'}});
});

it('uses the registered Krea type when choosing a model and hides its redundant selector',async()=>{
  const assets=[{id:'raw',family:'krea2',kind:'dit',path:'/models/raw.safetensors',variant:'raw',dtype:'bf16',exists:true,purpose:'training'}];
  server.use(http.get('/api/models',()=>HttpResponse.json(assets)));
  const change=vi.fn();
  const view=render(<SchemaForm schema={trainSchema} value={{model:{family:'krea2',dit_path:null,krea2_variant:'auto',dtype:'auto'}}} onChange={change} compact showAdvanced groupFilter={['model']}/>);
  fireEvent.click(await screen.findByTestId('model-registry-select'));
  fireEvent.click(screen.getByRole('option',{name:'raw.safetensors'}));
  expect(change).toHaveBeenLastCalledWith({model:{family:'krea2',dit_path:'/models/raw.safetensors',krea2_variant:'raw',dtype:'auto'}});
  view.rerender(<SchemaForm schema={trainSchema} value={change.mock.calls.at(-1)![0]} onChange={change} compact showAdvanced groupFilter={['model']}/>);
  await waitFor(()=>expect(screen.queryByTestId('field-model.krea2_variant')).not.toBeInTheDocument());
  expect(screen.getByText('RAW · BF16')).toBeInTheDocument();
});


it('replaces a stale Krea variant using the selected registered model metadata',async()=>{
  server.use(http.get('/api/models',()=>HttpResponse.json([{id:'raw',family:'krea2',kind:'dit',path:'/models/raw.safetensors',variant:'raw',dtype:'bf16',exists:true,purpose:'training'}])));
  const change=vi.fn();
  const initial={model:{family:'krea2',dit_path:'/models/raw.safetensors',krea2_variant:'turbo',dtype:'auto'}};
  const view=render(<SchemaForm schema={trainSchema} value={initial} onChange={change} compact showAdvanced groupFilter={['model']}/>);
  await waitFor(()=>expect(change).toHaveBeenCalledWith({...initial,model:{...initial.model,krea2_variant:'raw'}}));
  view.rerender(<SchemaForm schema={trainSchema} value={change.mock.calls.at(-1)![0]} onChange={change} compact showAdvanced groupFilter={['model']}/>);
  expect(screen.queryByTestId('field-model.krea2_variant')).not.toBeInTheDocument();
});

it('keeps confirmation available for an unregistered Krea model',async()=>{
  server.use(http.get('/api/models',()=>HttpResponse.json([])));
  const change=vi.fn();
  render(<SchemaForm schema={trainSchema} value={{model:{family:'krea2',dit_path:'/models/external.safetensors',krea2_variant:'auto'}}} onChange={change} compact showAdvanced groupFilter={['model']}/>);
  expect(screen.getByRole('combobox',{name:'Krea 2 类型'})).toHaveTextContent('确认本地模型类型');
  await waitFor(()=>expect(change).not.toHaveBeenCalled());
  fireEvent.click(screen.getByRole('combobox',{name:'Krea 2 类型'}));
  expect(screen.queryByRole('option',{name:/auto/})).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('option',{name:'Raw · 训练模型'}));
  expect(change).toHaveBeenCalledWith({model:{family:'krea2',dit_path:'/models/external.safetensors',krea2_variant:'raw'}});
});


it.each([
  ['mps','当前 Apple GPU 使用 FP32 加载和计算。'],
  ['cpu','当前 CPU 环境使用 FP32 加载和计算。'],
  [undefined,'按模型与平台兼容的精度加载。'],
])('shows only the current %s runtime precision guidance', (backend,expected)=>{
  render(<SchemaForm schema={trainSchema} value={{model:{family:'krea2',dtype:'auto'}}} family={{name:'krea2',runtime_backend:backend} as any} onChange={()=>{}} compact showAdvanced groupFilter={['model']}/>);
  const field=screen.getByTestId('field-model.dtype');
  expect(field).toHaveTextContent(expected);
  expect(field).not.toHaveTextContent('FP8');
  expect(field).not.toHaveTextContent('BF16 计算');
  if(backend===undefined) expect(field).not.toHaveTextContent(/CPU|Apple/);
});
