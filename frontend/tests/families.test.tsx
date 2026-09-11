import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest';
import { setupServer } from 'msw/node';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { handlers } from '../src/mocks/handlers';
import { SchemaForm } from '../src/schema/SchemaForm/SchemaForm';
import { FamilyInfo } from '../src/api/types';
import trainSchema from '../src/schema/train-schema.json';
import '../src/i18n';

const server = setupServer(...handlers);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
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
    const { rerender } = render(
      <SchemaForm schema={trainSchema as any} value={baseConfig} onChange={() => {}} family={krea2Family} />
    );
    const select = screen.getByTestId('adapter-preset-select');
    expect(select).toHaveTextContent('all-linear');
    fireEvent.click(select);
    expect(screen.getAllByRole('option').map(option=>option.textContent?.split(' — ')[0])).toEqual(['all-linear','attn-mlp','attn-only']);
    fireEvent.keyDown(select,{key:'Escape'});
    rerender(<SchemaForm schema={trainSchema as any} value={baseConfig} onChange={()=>{}} family={animaFamily}/>);
    fireEvent.click(screen.getByTestId('adapter-preset-select'));
    expect(screen.getAllByRole('option').map(option=>option.textContent?.split(' — ')[0])).toEqual(['attn-mlp','full-linear']);
  });

  it('2. krea2 下 text_encoding 无 online 选项并显示提示', () => {
    render(
      <SchemaForm schema={trainSchema as any} value={baseConfig} onChange={() => {}} family={krea2Family} />
    );
    fireEvent.click(screen.getByTestId('text-encoding-select'));
    expect(screen.getAllByRole('option')).toHaveLength(2);
    expect(screen.queryByRole('option',{name:/online|在线/i})).not.toBeInTheDocument();
  });

  it('3. anima 下 text_encoding 含 online', () => {
    render(
      <SchemaForm schema={trainSchema as any} value={baseConfig} onChange={() => {}} family={animaFamily} />
    );
    fireEvent.click(screen.getByTestId('text-encoding-select'));
    expect(screen.getAllByRole('option')).toHaveLength(3);
    expect(screen.getByRole('option',{name:/online|在线/i})).toBeInTheDocument();
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
  it('选择 krea2-lokr-default 预设后表单值正确', async () => {
    const { default: TrainConfig } = await import('../src/pages/TrainConfig/TrainConfig');
    const qc = new QueryClient();
    render(
      <QueryClientProvider client={qc}>
        <MemoryRouter initialEntries={['/projects/p_test/train']}>
          <Routes>
            <Route path="/projects/:id/train" element={<TrainConfig />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>
    );

    const picker = await screen.findByRole('combobox',{name:/加载预设/});
    await waitFor(()=>expect(picker).toBeEnabled());
    fireEvent.click(picker);
    fireEvent.click(await screen.findByRole('option',{name:'krea2-lokr-default'}));
    await waitFor(()=>expect(screen.getByTestId('adapter-preset-select')).toHaveTextContent('all-linear'));
  });
});
