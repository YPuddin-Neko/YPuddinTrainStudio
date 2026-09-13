import React from 'react';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createMemoryRouter, RouterProvider } from 'react-router-dom';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { afterAll, afterEach, beforeAll, beforeEach, expect, it, vi } from 'vitest';
import type { FamilyInfo, ModelAsset } from '../src/api/types';
import { handlers } from '../src/mocks/handlers';
import trainSchema from '../src/schema/train-schema.json';
import { schemaDefaults } from '../src/utils/config';
import { fillDefaultModels } from '../src/utils/workspaceConfig';
import { availableTrainingFamilies } from '../src/utils/trainingFamilies';
import { SchemaForm } from '../src/schema/SchemaForm/SchemaForm';
import ProjectEditor from '../src/pages/Projects/ProjectEditor';
import Presets from '../src/pages/Presets/Presets';
import Models from '../src/pages/Models/Models';
import LocalModelRegistration from '../src/pages/Models/LocalModelRegistration';
import TrainConfig from '../src/pages/TrainConfig/TrainConfig';
import i18n from '../src/i18n';

const family = (name: string, label: string): FamilyInfo => ({
  name, label, architecture: name, adapter_prefix: 'lora_transformer', capabilities: ['activation_checkpointing'],
  objective: 'rectified_flow', sampling_samplers: ['euler', 'heun'], sampling_schedulers: ['uniform'],
  objective_timestep_sampling: ['uniform', 'resolution_shift'], objective_weighting: ['none'],
  text_modes: ['auto', 'cached'], presets: [{ name: 'attn-mlp', description: 'Attention and MLP', layers: 10, include: ['*'], exclude: [] }],
  default_preset: 'attn-mlp', sampling: { steps: 50, cfg: 4, shift: null, sampler: 'euler' },
  latent: { channels: 128, stride: 16, patch: 1, align: 16 }, text_max_len: 512, linear_modules: 10,
  weights: [
    { field: 'dit_path', kind: 'dit', label: '主模型', hint: '', required: true, downloadable: true },
    { field: 'text_encoder_path', kind: 'text_encoder', label: '文本编码器', hint: '', required: false, downloadable: true },
    { field: 'text_encoder_2_path', kind: 'text_encoder_2', label: '第二文本编码器', hint: '', required: false, downloadable: true },
    { field: 'vae_path', kind: 'vae', label: 'VAE', hint: '', required: false, downloadable: true },
  ],
});
const oldFamilies = [family('anima', 'Anima'), family('flux', 'FLUX.1'), family('flux2', 'FLUX.2 dev / Klein base'), family('sdxl', 'SDXL')];
const server = setupServer(...handlers);
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterAll(() => server.close());
afterEach(() => { cleanup(); server.resetHandlers(); sessionStorage.clear(); });
beforeEach(async () => {
  await i18n.changeLanguage('zh-CN');
  server.use(
    http.get('/api/families', () => HttpResponse.json(oldFamilies)),
    http.get('/api/presets', () => HttpResponse.json([])),
    http.get('/api/models', () => HttpResponse.json([])),
    http.get('/api/models/downloads', () => HttpResponse.json([])),
    http.get('/api/models/recommendations', () => HttpResponse.json([])),
  );
});
function mount(element: React.ReactNode, path = '/', route = '*') {
  const router = createMemoryRouter([{ path: route, element }], { initialEntries: [path] });
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <RouterProvider router={router}/>
  </QueryClientProvider>);
}
function configured(name: string, variant = 'auto') {
  const config = schemaDefaults(trainSchema);
  config.model = { ...config.model, family: name, flux2_variant: variant, dit_path: 'J:/models/original.safetensors',
    text_encoder_path: 'J:/models/encoder', text_encoder_2_path: 'J:/models/old-second-encoder', training_guidance: 7 };
  config.sampling.guidance = 9;
  config.loop.epochs = 23;
  config.dataset.sources = [{ path: 'J:/project/original-images', repeats: 1 }];
  return config;
}

const retiredAsset: ModelAsset & { unsupported_reason: string } = {
  id: 'retired', family: 'flux2', kind: 'dit', path: 'J:/models/unrelated-filename.safetensors', exists: true,
  is_default: true, dtype: 'bf16', size: 1024, created_at: 1, unsupported_reason: 'FLUX.2 dev 已停用',
};
const kleinAsset: ModelAsset = { ...retiredAsset, id: 'klein', path: 'J:/models/klein.safetensors', is_default: false, unsupported_reason: null };

it('excludes server-inspected retired defaults from automatic filling and model selection', async () => {
  const assets = [retiredAsset, kleinAsset];
  server.use(http.get('/api/models', () => HttpResponse.json(assets)));
  const config = configured('flux2'); config.model.dit_path = null;
  expect(fillDefaultModels(config, assets).model.dit_path).toBeNull();
  const explicit = { ...config, model: { ...config.model, dit_path: retiredAsset.path } };
  expect(fillDefaultModels(explicit, assets).model.dit_path).toBe(retiredAsset.path);
  const families = availableTrainingFamilies(oldFamilies);
  mount(<SchemaForm compact schema={trainSchema} value={config} onChange={vi.fn()} family={families.find(item => item.name === 'flux2')} families={families}/>);
  const field = screen.getByTestId('field-model.dit_path');
  fireEvent.click(await within(field).findByTestId('model-registry-select'));
  expect(screen.queryByRole('option', { name: 'unrelated-filename.safetensors' })).not.toBeInTheDocument();
  expect(screen.getByRole('option', { name: 'klein.safetensors' })).toBeInTheDocument();
});

it('keeps retired assets visible with their reason and permits clearing the old default', async () => {
  const patch = vi.fn();
  server.use(
    http.get('/api/models', () => HttpResponse.json([retiredAsset, kleinAsset])),
    http.patch('/api/models/retired', async ({ request }) => { patch(await request.json()); return HttpResponse.json(retiredAsset); }),
  );
  mount(<Models/>, '/models?family=flux2&view=library');
  expect(await screen.findByText('FLUX.2 dev 已停用')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: '取消默认 unrelated-filename.safetensors' }));
  await waitFor(() => expect(patch).toHaveBeenCalledWith({ is_default: false }));
  expect(retiredAsset.path).toBe('J:/models/unrelated-filename.safetensors');
});

it('filters stale FLUX.1 responses and names Klein in the actual new-project selector', async () => {
  mount(<ProjectEditor categories={[]} onClose={vi.fn()} onSaved={vi.fn()} onPartial={vi.fn()}/>);
  const combo = await screen.findByRole('combobox', { name: '初始模型类型' });
  await waitFor(() => expect(combo).toHaveTextContent('Anima'));
  fireEvent.click(combo);
  expect(screen.queryByRole('option', { name: /FLUX.1/ })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('option', { name: 'FLUX.2 Klein' }));
  expect(combo).toHaveTextContent('FLUX.2 Klein');
});

it('offers Klein but no FLUX.1 when creating a standalone preset', async () => {
  const defaults = vi.fn();
  server.use(http.get('/api/config/defaults', ({ request }) => {
    const name = new URL(request.url).searchParams.get('family') || 'anima'; defaults(name);
    return HttpResponse.json(configured(name));
  }));
  mount(<Presets/>, '/presets');
  const combo = await screen.findByRole('combobox', { name: '适用模型' });
  fireEvent.click(combo);
  expect(screen.queryByRole('option', { name: /FLUX.1/ })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('option', { name: 'FLUX.2 Klein' }));
  await waitFor(() => expect(defaults).toHaveBeenCalledWith('flux2'));
  expect(defaults).not.toHaveBeenCalledWith('flux');
});

it.each([['flux', 'auto'], ['flux2', 'dev']])('keeps retired %s/%s presets viewable without requesting new defaults', async (name, variant) => {
  const config = configured(name, variant);
  const defaults = vi.fn();
  server.use(
    http.get('/api/presets', () => HttpResponse.json([{ name: 'old-config', description: '', config, builtin: false, updated_at: 1 }])),
    http.get('/api/config/defaults', () => { defaults(); return HttpResponse.json({}, { status: 404 }); }),
  );
  mount(<Presets/>, '/presets');
  expect(await screen.findByTestId('retired-preset')).toHaveTextContent('已停用');
  expect(screen.getByRole('textbox', { name: '预设名称' })).toHaveValue('old-config');
  expect(screen.getByRole('spinbutton', { name: 'loop.epochs' })).toHaveValue(23);
  expect(screen.getByRole('spinbutton', { name: 'loop.epochs' })).toBeDisabled();
  expect(screen.getByRole('button', { name: '复制为新预设' })).toBeDisabled();
  expect(screen.getByRole('button', { name: '保存预设' })).toBeDisabled();
  expect(defaults).not.toHaveBeenCalled();
  expect(config.model.flux2_variant).toBe(variant);
  expect(config.dataset.sources[0].path).toBe('J:/project/original-images');
});

it('removes FLUX.1 model registration and download entry points even with a stale service', async () => {
  mount(<Models/>, '/models?family=flux');
  expect(await screen.findByRole('alert')).toHaveTextContent('FLUX.1 已停用');
  await waitFor(() => expect(screen.getByRole('combobox', { name: '模型系列' })).toBeEnabled());
  expect(screen.queryByTestId('add-model-btn')).not.toBeInTheDocument();
  expect(screen.queryByTestId('download-model-btn')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('combobox', { name: '模型系列' }));
  expect(screen.queryByRole('option', { name: /FLUX.1/ })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('option', { name: 'FLUX.2 Klein' }));
  expect(await screen.findByTestId('add-model-btn')).toBeInTheDocument();
});

it('never reclassifies an inspected FLUX.1 file as the currently selected supported family', async () => {
  server.use(http.post('/api/models/inspect', () => HttpResponse.json({
    path: 'J:/models/flux.safetensors', family: 'flux', family_candidates: [], kind: 'dit', dtype: 'bf16',
    dtypes: { BF16: 2 }, confidence: 'high', evidence: [], warnings: [], files_inspected: 1,
  })));
  mount(<LocalModelRegistration initialFamily="anima" families={oldFamilies} onClose={vi.fn()} onRegistered={vi.fn()} onBusyChange={vi.fn()}/>);
  fireEvent.change(screen.getByRole('textbox', { name: '文件路径' }), { target: { value: 'J:/models/flux.safetensors' } });
  await screen.findByTestId('model-inspection-status');
  expect(screen.getByTestId('add-model-submit')).toBeDisabled();
  expect(screen.getByRole('alert')).toHaveTextContent('不支持检测到的模型系列');
});

it.each(['auto', 'dev'])('renders only Klein variants and preserves a saved %s value', (variant) => {
  // Emulate a pre-upgrade service schema; frontend restrictions must stand on their own.
  const schema = structuredClone(trainSchema);
  schema.$defs.ModelConfig.properties.flux2_variant.enum = ['auto', 'dev', 'klein-base-4b', 'klein-base-9b'];
  const changed = vi.fn();
  const config = configured('flux2', variant);
  const families = availableTrainingFamilies(oldFamilies);
  mount(<SchemaForm compact showAdvanced schema={schema} value={config} onChange={changed}
    family={families.find(item => item.name === 'flux2')} families={families}/>);
  expect(screen.queryByTestId('field-model.training_guidance')).not.toBeInTheDocument();
  expect(screen.queryByTestId('field-sampling.guidance')).not.toBeInTheDocument();
  expect(screen.queryByTestId('field-model.text_encoder_2_path')).not.toBeInTheDocument();
  const combo = screen.getByRole('combobox', { name: 'Klein 类型' });
  if (variant === 'dev') expect(combo).toHaveTextContent('FLUX.2 dev（已停用）');
  fireEvent.click(combo);
  expect(screen.getAllByRole('option').filter(option => option.getAttribute('aria-disabled') !== 'true').map(option => option.textContent)).toEqual(['自动读取模型配置', 'Klein 基础版 4B', 'Klein 基础版 9B']);
  expect(changed).not.toHaveBeenCalled();
  expect(config.model.flux2_variant).toBe(variant);
  expect(config.model.training_guidance).toBe(7);
  expect(config.sampling.guidance).toBe(9);
});

it.each([['flux', 'auto'], ['flux2', 'dev']])('shows retired %s/%s training settings without enqueue or automatic path changes', async (name, variant) => {
  const config = configured(name, variant);
  const writes = vi.fn(), enqueue = vi.fn(), planned = vi.fn();
  server.use(
    http.get('/api/projects/:id/config', () => HttpResponse.json(config)),
    http.put('/api/projects/:id/config', async ({ request }) => { writes(await request.json()); return HttpResponse.json({}); }),
    http.post('/api/jobs', () => { enqueue(); return HttpResponse.json({ id: 'unexpected' }); }),
    http.post('/api/plan', () => { planned(); return HttpResponse.json({ ok: true, errors: [], warnings: [], total_steps: 10 }); }),
  );
  mount(<TrainConfig/>, '/projects/p_legacy/train?tab=model', '/projects/:id/train');
  expect(await screen.findByTestId('retired-training-config')).toHaveTextContent('已停用');
  await waitFor(() => expect(planned).toHaveBeenCalled());
  const start = screen.getByRole('button', { name: '开始训练' });
  expect(start).toBeDisabled();
  fireEvent.click(start);
  expect(enqueue).not.toHaveBeenCalled();
  const path = within(screen.getByTestId('field-model.dit_path')).getByRole('textbox');
  expect(path).toHaveValue('J:/models/original.safetensors');
  expect(path).toBeDisabled();
  expect(writes).not.toHaveBeenCalled();
  expect(fillDefaultModels(config, [])).toEqual(config);
});
