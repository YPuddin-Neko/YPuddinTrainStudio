import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest';
import { setupServer } from 'msw/node';
import { http, HttpResponse } from 'msw';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { handlers } from '../src/mocks/handlers';
import { SchemaForm } from '../src/schema/SchemaForm/SchemaForm';
import { FamilyInfo } from '../src/api/types';
import trainSchema from '../src/schema/train-schema.json';
import { schemaDefaults } from '../src/utils/config';
import '../src/i18n';

const server = setupServer(...handlers);

beforeAll(() => server.listen());
afterEach(() => { server.resetHandlers(); sessionStorage.clear(); });
afterAll(() => server.close());

const krea2Family = {
  name: 'krea2',
  label: 'Krea 2 Raw 12.9B',
  text_modes: ['auto', 'cached'],
  presets: [
    { name: 'all-linear', description: '全部 264 个 Linear', include: ['*'], exclude: [], layers: 264 },
    { name: 'attn-mlp', description: '注意力 + SwiGLU', include: [], exclude: [], layers: 224 },
    { name: 'attn-only', description: '仅注意力投影', include: [], exclude: [], layers: 140 },
  ],
  default_preset: 'attn-mlp',
  sampling: { steps: 28, cfg: 5.5, shift: null, sampler: 'euler' },
  weights: [
    { field: 'dit_path', label: 'DiT', hint: 'krea2_raw_bf16.safetensors（约 26 GB）' },
    { field: 'text_encoder_path', label: 'Qwen3-VL-4B-Instruct', hint: 'HF 目录（推荐）' },
    { field: 'vae_path', label: 'Qwen-Image VAE', hint: 'qwen_image_vae.safetensors' },
  ],
} as unknown as FamilyInfo;

const animaFamily = {
  name: 'anima',
  label: 'Anima 2B',
  text_modes: ['auto', 'cached', 'online'],
  presets: [
    { name: 'attn-mlp', description: '注意力 + MLP', include: [], exclude: [], layers: 280 },
    { name: 'full-linear', description: '全部 Linear', include: [], exclude: [], layers: 448 },
  ],
  default_preset: 'attn-mlp',
  sampling: { steps: 25, cfg: 4.0, shift: 3.0, sampler: 'euler' },
  weights: [],
} as unknown as FamilyInfo;

const baseConfig = {
  model: { family: 'krea2', dtype: 'bf16' },
  adapter: { algo: 'lokr', rank: 32, alpha: 32, preset: 'all-linear' },
  dataset: { batch_size: 2, text_encoding: 'auto', sources: [] },
  sampling: { enabled: true, prompts: [] },
};

describe('FE-M7: family-driven SchemaForm', () => {
  it('1. adapter.preset 下拉选项随族切换', () => {
    let changed: any;
    const { rerender } = render(
      <SchemaForm schema={trainSchema as any} value={baseConfig} onChange={next => { changed = next; }} family={krea2Family} />
    );
    const select = screen.getByTestId('adapter-preset-select');
    expect(select).toHaveAccessibleName('训练范围');
    expect(select).toHaveTextContent('全部线性层');
    fireEvent.click(select);
    expect(screen.getAllByRole('option').map(option => option.textContent)).toEqual(['全部线性层', '常规范围（默认）', '精简范围']);
    fireEvent.click(screen.getByRole('option', {name: '精简范围'}));
    expect(changed.adapter.preset).toBe('attn-only');
    rerender(<SchemaForm schema={trainSchema as any} value={baseConfig} onChange={()=>{}} family={animaFamily}/>);
    fireEvent.click(screen.getByTestId('adapter-preset-select'));
    expect(screen.getAllByRole('option').map(option => option.textContent)).toEqual(['常规范围（默认）', '全部线性层']);
  });

  it('explains training scope separately and reserves technical identifiers for advanced help', () => {
    const {rerender} = render(<SchemaForm schema={trainSchema} value={baseConfig} onChange={() => {}} family={krea2Family} compact/>);
    fireEvent.click(screen.getByRole('button', {name: '训练范围 说明'}));
    expect(screen.getByRole('tooltip')).toHaveTextContent('决定本次训练可以调整模型的哪些部分');
    expect(screen.getByRole('tooltip')).toHaveTextContent('不保证效果更好');
    expect(screen.getByRole('tooltip')).not.toHaveTextContent('all-linear');
    fireEvent.keyDown(document, {key: 'Escape'});
    rerender(<SchemaForm schema={trainSchema} value={baseConfig} onChange={() => {}} family={krea2Family} compact showAdvanced/>);
    fireEvent.click(screen.getByRole('button', {name: '训练范围 说明'}));
    expect(screen.getByRole('tooltip')).toHaveTextContent('all-linear');
    expect(screen.getByRole('tooltip')).toHaveTextContent('264');
  });

  it('2. krea2 高级标签处理方式仅提供受支持的缓存选项', () => {
    render(
      <SchemaForm schema={trainSchema as any} value={baseConfig} onChange={() => {}} family={krea2Family} showAdvanced />
    );
    fireEvent.click(screen.getByTestId('text-encoding-select'));
    expect(screen.getAllByRole('option')).toHaveLength(2);
    expect(screen.getByRole('option',{name:'训练前缓存标签'})).toBeInTheDocument();
    expect(screen.queryByRole('option',{name:'每步处理标签'})).not.toBeInTheDocument();
  });

  it('3. anima 高级标签处理方式使用易懂中文且仍写入 online 枚举', () => {
    let changed: any;
    render(
      <SchemaForm schema={trainSchema as any} value={baseConfig} onChange={next => { changed = next; }} family={animaFamily} showAdvanced />
    );
    fireEvent.click(screen.getByTestId('text-encoding-select'));
    expect(screen.getAllByRole('option')).toHaveLength(3);
    expect(screen.queryByRole('option',{name:/在线|预缓存/})).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('option',{name:'每步处理标签'}));
    expect(changed.dataset.text_encoding).toBe('online');
  });

  it('4. krea2 的 sampling.shift 显示"自动（按分辨率）"占位', () => {
    render(
      <SchemaForm schema={trainSchema as any} value={baseConfig} onChange={() => {}} family={krea2Family} />
    );
    const field = screen.getByTestId('field-sampling.shift');
    const input = field.querySelector('input[type=number]') as HTMLInputElement;
    expect(input.placeholder).toContain('自动');
  });

  it('5. krea2 的 weights hint 显示在路径字段下方', () => {
    render(
      <SchemaForm schema={trainSchema as any} value={baseConfig} onChange={() => {}} family={krea2Family} />
    );
    expect(screen.getByTestId('weight-hint-dit_path').textContent).toContain('krea2_raw_bf16');
    expect(screen.getByTestId('weight-hint-vae_path').textContent).toContain('qwen_image_vae');
  });
});

describe('FE-M7: TrainConfig 预设联动（MSW）', () => {
  async function showTrainingPresetPage() {
    let config = schemaDefaults(trainSchema);
    config.model = { ...config.model, family:'anima', dit_path:'/current/dit.safetensors', text_encoder_path:'/current/text', vae_path:'/current/vae.safetensors', tokenizer_path:'/current/tokenizer' };
    config.adapter = {...config.adapter,algo:'lora',rank:16,alpha:16,resume_weights:'/current/adapter.safetensors'};
    config.dataset = {...config.dataset,sources:[{path:'/current/images',caption_ext:'.txt',repeats:2}],cache_dir:'/current/cache'};
    config.validation.sources=[{path:'/current/validation'}];
    config.checkpoint={...config.checkpoint,output_dir:'/current/output',resume:'/current/full-state'};
    config.sampling={...config.sampling,output_dir:'/current/samples',prompts_file:'/current/prompts.txt'};
    config.logging.events_path='/current/events.jsonl';
    config.optimizer.lr=0.0001;config.loop.epochs=2;
    const original=structuredClone(config);
    const preset={name:'anima-own',description:'同族风格参数，调整学习率和轮数',builtin:false,updated_at:1,config:{
      model:{family:'anima',dit_path:'/foreign/dit',text_encoder_path:'/foreign/text',vae_path:'/foreign/vae',tokenizer_path:'/foreign/tokenizer'},
      dataset:{sources:[{path:'/foreign/images'}],cache_dir:'/foreign/cache'},validation:{sources:[{path:'/foreign/validation'}]},
      checkpoint:{output_dir:'/foreign/output',resume:'/foreign/state'},adapter:{resume_weights:'/foreign/adapter'},
      sampling:{output_dir:'/foreign/samples',prompts_file:'/foreign/prompts'},logging:{events_path:'/foreign/events'},
      optimizer:{lr:0.0003},loop:{epochs:7},
    }};
    server.use(
      http.get('/api/projects/p_test/config',()=>HttpResponse.json(config)),
      http.put('/api/projects/p_test/config',async({request})=>{config=await request.json() as any;return HttpResponse.json(config);}),
      http.get('/api/models',()=>HttpResponse.json([])),
      http.get('/api/presets',()=>HttpResponse.json([preset,{name:'krea2-own',description:'Krea only',builtin:false,updated_at:2,config:{model:{family:'krea2'},adapter:{preset:'all-linear'}}},{name:'legacy-builtin',description:'Hidden legacy preset',builtin:true,updated_at:999,config:{model:{family:'anima'},loop:{epochs:99}}}])),
    );
    const { default: TrainConfig } = await import('../src/pages/TrainConfig/TrainConfig');
    const qc = new QueryClient({defaultOptions:{queries:{retry:false}}});
    render(
      <QueryClientProvider client={qc}>
        <MemoryRouter initialEntries={['/projects/p_test/train?tab=train']}>
          <Routes>
            <Route path="/projects/:id/train" element={<TrainConfig />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>
    );

    const picker = await screen.findByRole('combobox',{name:/加载预设/});
    await waitFor(()=>expect(picker).toBeEnabled());
    await screen.findByRole('spinbutton',{name:'学习率'});
    return {picker,original,config:()=>config};
  }

  it('明确禁用另一模型族的预设，点击不会打开预览或更换当前模型', async () => {
    const {picker,config,original}=await showTrainingPresetPage();
    fireEvent.click(picker);
    expect(screen.queryByRole('option',{name:/legacy-builtin/})).not.toBeInTheDocument();
    const other=await screen.findByRole('option',{name:'krea2-own · 适用于 krea2'});
    expect(other).toHaveAttribute('aria-disabled','true');
    fireEvent.click(other);
    expect(screen.queryByRole('dialog',{name:'加载预设前确认参数'})).not.toBeInTheDocument();
    expect(screen.getByRole('spinbutton',{name:'学习率'})).toHaveValue(0.0001);
    expect(config().model).toEqual(original.model);
  });

  it('同族预设先展示真实变化，取消不改草稿，应用并保存只更新参数且保留全部项目路径', async () => {
    const {picker,config,original}=await showTrainingPresetPage();
    fireEvent.click(picker);fireEvent.click(await screen.findByRole('option',{name:'anima-own'}));
    let dialog=await screen.findByRole('dialog',{name:'加载预设前确认参数'});
    expect(dialog).toHaveTextContent('同族风格参数，调整学习率和轮数');
    expect(dialog).toHaveTextContent('将修改 2 个参数');
    expect(within(dialog).getByRole('table')).toHaveTextContent('optimizer.lr');
    expect(within(dialog).getByRole('table')).toHaveTextContent('loop.epochs');
    expect(within(dialog).getByRole('table')).not.toHaveTextContent('/foreign');
    expect(screen.getByRole('spinbutton',{name:'学习率'})).toHaveValue(0.0001);
    fireEvent.click(within(dialog).getByRole('button',{name:'取消'}));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(screen.getByRole('spinbutton',{name:'loop.epochs'})).toHaveValue(2);
    expect(config()).toEqual(original);
    fireEvent.click(picker);fireEvent.click(await screen.findByRole('option',{name:'anima-own'}));
    dialog=await screen.findByRole('dialog',{name:'加载预设前确认参数'});
    fireEvent.click(within(dialog).getByRole('button',{name:'应用到当前版本'}));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(screen.getByRole('spinbutton',{name:'学习率'})).toHaveValue(0.0003);
    expect(screen.getByRole('spinbutton',{name:'loop.epochs'})).toHaveValue(7);
    fireEvent.click(screen.getByRole('button',{name:'保存草稿'}));
    await waitFor(()=>expect(config().optimizer.lr).toBe(0.0003));
    expect(config()).toEqual({...original,optimizer:{...original.optimizer,lr:0.0003},loop:{...original.loop,epochs:7}});
  });
});
