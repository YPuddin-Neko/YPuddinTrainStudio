import React from 'react';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import type { FamilyInfo, ModelAsset } from '../src/api/types';
import Models from '../src/pages/Models/Models';
import LocalModelRegistration from '../src/pages/Models/LocalModelRegistration';
import { SchemaForm } from '../src/schema/SchemaForm/SchemaForm';
import trainSchema from '../src/schema/train-schema.json';
import { schemaDefaults } from '../src/utils/config';
import { changeModelFamily, fillDefaultModels } from '../src/utils/workspaceConfig';
import { applyTrainingPreset, reusableTrainingPreset } from '../src/utils/trainingPresets';
import { presetEditorSchema } from '../src/utils/presetEditor';
import { modelFamilyWeights, trainingFamilyOptions } from '../src/utils/trainingFamilies';
import { handlers } from '../src/mocks/handlers';
import i18n from '../src/i18n';

const sdxl: FamilyInfo = {
  attention_backends: ['auto', 'sdpa', 'xformers'], name: 'sdxl', label: 'SDXL', architecture: 'sdxl', adapter_prefix: 'lora', objective: 'ddpm',
  capabilities: [], text_modes: ['auto', 'cached', 'online'], presets: [], default_preset: 'all-linear',
  sampling: { steps: 28, cfg: 7, shift: 1, sampler: 'euler', guidance: null },
  sampling_samplers: ['euler', 'heun'], sampling_schedulers: ['uniform'],
  objective_timestep_sampling: ['uniform', 'logit_normal'], objective_weighting: ['none', 'min_snr'],
  latent: { channels: 4, stride: 8, patch: 1, align: 8 }, text_max_len: 77, linear_modules: 1,
  weights: [
    { field: 'dit_path', kind: 'dit', label: 'SDXL 完整模型', hint: '内含双文本编码器和 VAE。', required: true, downloadable: true },
    { field: 'text_encoder_path', kind: 'text_encoder', label: 'CLIP-L', hint: '留空使用完整模型内含的 CLIP-L。', required: false, downloadable: true },
    { field: 'text_encoder_2_path', kind: 'text_encoder_2', label: 'CLIP-G', hint: '留空使用完整模型内含的 CLIP-G。', required: false, downloadable: true },
    { field: 'vae_path', kind: 'vae', label: 'VAE', hint: '留空使用完整模型内含的 VAE。', required: false, downloadable: true },
  ],
};
const flow: FamilyInfo = {
  ...sdxl, name: 'anima', label: 'Anima', objective: 'rectified_flow',
  sampling_samplers: ['euler', 'heun', 'er_sde'], sampling_schedulers: ['uniform', 'simple', 'sgm_uniform', 'normal'],
  objective_timestep_sampling: ['uniform', 'logit_normal', 'shift', 'resolution_shift', 'mode', 'cosmap'],
  objective_weighting: ['none', 'sigma_sqrt', 'cosmap', 'snr_like', 'cosmos'],
  weights: sdxl.weights.filter(weight => weight.kind !== 'text_encoder_2').map(weight => ({ ...weight, required: true })),
};
const asset = (kind: string, patch: Partial<ModelAsset> = {}): ModelAsset => ({
  id: kind, family: 'sdxl', kind, path: `/models/${kind}.safetensors`, exists: true,
  is_default: true, purpose: 'training', dtype: 'bf16', size: 2048, created_at: 1, ...patch,
});
const server = setupServer(...handlers);
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterAll(() => server.close());
afterEach(() => { cleanup(); server.resetHandlers(); });
beforeEach(async () => {
  await i18n.changeLanguage('zh-CN');
  server.use(
    http.get('/api/families', () => HttpResponse.json([flow, sdxl])),
    http.get('/api/models', () => HttpResponse.json([asset('dit')])),
    http.get('/api/models/downloads', () => HttpResponse.json([])),
    http.get('/api/models/recommendations', () => HttpResponse.json([])),
  );
});
function choose(name: string, option: string) {
  fireEvent.click(screen.getByRole('combobox', { name }));
  fireEvent.click(screen.getByRole('option', { name: option }));
}
function mountModels() {
  return render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <MemoryRouter initialEntries={['/models?family=sdxl']}><Models /></MemoryRouter>
  </QueryClientProvider>);
}
function Editor({ advanced = false, groupFilter = ['model'], family = sdxl }: {
  advanced?: boolean; groupFilter?: string[]; family?: FamilyInfo;
}) {
  const [value, setValue] = React.useState(() => ({
    ...schemaDefaults(trainSchema), model: { family: family.name, prediction_type: 'v_prediction', zero_terminal_snr: true },
    sampling: { enabled: true, sampler: 'euler', scheduler: 'uniform' }, objective: { timestep_sampling: 'uniform', weighting: 'none' },
  }));
  return <><SchemaForm schema={trainSchema} value={value} onChange={setValue} compact showAdvanced={advanced}
    family={family} families={[flow, sdxl]} groupFilter={groupFilter} />
  <output data-testid="changed-config">{JSON.stringify(value)}</output></>;
}

describe('SDXL model preparation and registration', () => {
  it('uses the verified provider for the pinned SDXL model and never submits an unavailable mirror', async () => {
    const download = vi.fn();
    server.use(
      http.get('/api/models', () => HttpResponse.json([])),
      http.get('/api/models/recommendations', () => HttpResponse.json([{
        id: 'sdxl-illustrious-v01', family: 'sdxl', kind: 'dit', name: 'Illustrious-XL v0.1',
        dtype: 'fp16', size: 6938040760, recommended: true,
        sources: [{ provider: 'huggingface', repo_id: 'OnomaAIResearch/Illustrious-xl-early-release-v0',
          filename: 'Illustrious-XL-v0.1.safetensors', revision: 'pinned', url: 'https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0' }],
      }])),
      http.post('/api/models/recommendations/sdxl-illustrious-v01/download', async ({ request }) => {
        download(await request.json()); return HttpResponse.json({});
      }),
    );
    mountModels();
    await screen.findByText('Illustrious-XL v0.1');
    fireEvent.click(screen.getByRole('combobox', { name: '下载来源' }));
    expect(screen.getAllByRole('option').map(option => option.textContent)).toEqual(['Hugging Face']);
    fireEvent.click(screen.getByRole('option', { name: 'Hugging Face' }));
    fireEvent.click(screen.getByRole('button', { name: '准备缺失组件' }));
    await waitFor(() => expect(download).toHaveBeenCalledWith({ provider: 'huggingface', is_default: true }));
  });

  it('counts the complete checkpoint as the only required component and supports a custom CLIP-G download without catalog entries', async () => {
    const download = vi.fn();
    server.use(http.post('/api/models/downloads', async ({ request }) => { download(await request.json()); return HttpResponse.json({}); }));
    mountModels();
    expect(await screen.findByText('必需组件 1 / 1')).toBeInTheDocument();
    expect(screen.getByTestId('model-component-dit')).toHaveTextContent('SDXL 完整模型');
    expect(screen.getByTestId('model-component-text_encoder_2')).toHaveTextContent('可选');
    expect(screen.getByRole('button', { name: '准备缺失组件' })).toBeDisabled();
    fireEvent.click(screen.getByRole('combobox', { name: '模型系列' }));
    expect(screen.getAllByRole('option').map(option => option.textContent)).toEqual(['Anima', 'SDXL 2.6B']);
    fireEvent.keyDown(screen.getByRole('combobox', { name: '模型系列' }), { key: 'Escape' });
    fireEvent.click(screen.getByTestId('download-model-btn'));
    choose('组件', 'CLIP-G');
    fireEvent.change(screen.getByRole('textbox', { name: '仓库 ID' }), { target: { value: 'my/weights' } });
    fireEvent.change(screen.getByRole('textbox', { name: '仓库内文件路径' }), { target: { value: 'clip-g.safetensors' } });
    fireEvent.click(screen.getByTestId('model-download-start'));
    await waitFor(() => expect(download).toHaveBeenCalledWith(expect.objectContaining({ family: 'sdxl', kind: 'text_encoder_2', filename: 'clip-g.safetensors' })));
  });

  it('does not count optional defaults or a missing checkpoint as ready', async () => {
    server.use(http.get('/api/models', () => HttpResponse.json([asset('dit', { exists: false }), asset('text_encoder'), asset('text_encoder_2'), asset('vae')])));
    mountModels();
    expect(await screen.findByText('必需组件 0 / 1')).toBeInTheDocument();
    expect(screen.queryByText(/3 \/ 3|4 \/ 4/)).not.toBeInTheDocument();
  });

  it('registers detected SDXL CLIP-G using the API family and component metadata', async () => {
    const registered = vi.fn(async () => {});
    const request = vi.fn();
    server.use(
      http.post('/api/models/inspect', () => HttpResponse.json({ path: '/models/clip-g.safetensors', family: 'sdxl', family_candidates: ['sdxl'], kind: 'text_encoder_2', dtype: 'fp16', dtypes: { F16: 3 }, confidence: 'high', evidence: ['CLIP-G'], warnings: [], files_inspected: 1 })),
      http.post('/api/models', async ({ request: incoming }) => { request(await incoming.json()); return HttpResponse.json({}); }),
    );
    render(<LocalModelRegistration initialFamily="anima" families={[flow, sdxl]} onRegistered={registered} onBusyChange={() => {}} onClose={() => {}} />);
    fireEvent.change(screen.getByRole('textbox', { name: '文件路径' }), { target: { value: '/models/clip-g.safetensors' } });
    await screen.findByTestId('model-inspection-status');
    expect(screen.getByRole('combobox', { name: '登记模型系列' })).toHaveTextContent('SDXL');
    expect(screen.getByRole('combobox', { name: '组件' })).toHaveTextContent('CLIP-G');
    expect(screen.getByTestId('add-model-submit')).toBeEnabled();
    fireEvent.click(screen.getByTestId('add-model-submit'));
    await waitFor(() => expect(registered).toHaveBeenCalledWith('sdxl'));
    expect(request).toHaveBeenCalledWith(expect.objectContaining({ family: 'sdxl', kind: 'text_encoder_2', path: '/models/clip-g.safetensors', dtype: 'fp16' }));
  });
});

describe('family-driven training fields', () => {
  it('shows the complete model and prediction mode normally, with component overrides in advanced mode', async () => {
    server.use(http.get('/api/models', () => HttpResponse.json([asset('text_encoder_2', { path: '/models/my-clip-g.safetensors' })])));
    const { rerender } = render(<Editor />);
    expect(within(screen.getByTestId('field-model.dit_path')).getByLabelText('SDXL 完整模型')).toBeInTheDocument();
    expect(screen.queryByTestId('field-model.text_encoder_path')).not.toBeInTheDocument();
    expect(screen.queryByTestId('field-model.text_encoder_2_path')).not.toBeInTheDocument();
    expect(screen.queryByTestId('field-model.vae_path')).not.toBeInTheDocument();
    expect(screen.getByTestId('field-model.prediction_type')).toBeInTheDocument();
    rerender(<Editor advanced />);
    expect(screen.getByTestId('field-model.text_encoder_path')).toHaveTextContent('CLIP-L');
    expect(screen.getByTestId('field-model.text_encoder_2_path')).toHaveTextContent('CLIP-G');
    expect(screen.getByTestId('field-model.vae_path')).toHaveTextContent('可选');
    expect(screen.getByTestId('field-model.tokenizer_path')).toBeInTheDocument();
    const tokenizer = within(screen.getByTestId('field-model.tokenizer_path'));
    fireEvent.change(tokenizer.getByRole('textbox'), { target: { value: '/models/custom-tokenizers' } });
    expect(JSON.parse(screen.getByTestId('changed-config').textContent!).model.tokenizer_path).toBe('/models/custom-tokenizers');
    fireEvent.click(tokenizer.getByRole('button', { name: /说明$/ }));
    expect(screen.getByRole('tooltip')).toHaveTextContent('tokenizer/ 和 tokenizer_2/');
    fireEvent.keyDown(document, { key: 'Escape' });
    const registry = await within(screen.getByTestId('field-model.text_encoder_2_path')).findByTestId('model-registry-select');
    fireEvent.click(registry);
    fireEvent.click(screen.getByRole('option', { name: 'my-clip-g.safetensors' }));
    expect(JSON.parse(screen.getByTestId('changed-config').textContent!).model.text_encoder_2_path).toBe('/models/my-clip-g.safetensors');
    const prediction = within(screen.getByTestId('field-model.prediction_type')).getByRole('combobox');
    fireEvent.click(prediction);
    fireEvent.click(screen.getByRole('option', { name: 'ε 预测（常规模型）' }));
    expect(JSON.parse(screen.getByTestId('changed-config').textContent!).model.zero_terminal_snr).toBe(false);
    expect(screen.queryByTestId('field-model.zero_terminal_snr')).not.toBeInTheDocument();
  });

  it('keeps mode beside the model family, component switches together, and SDXL paths in distinct rows', () => {
    render(<Editor advanced />);
    const form = screen.getByTestId('schema-form');
    const identity = form.querySelector('.config-model-identity')!;
    expect(within(identity as HTMLElement).getByTestId('field-model.family')).toBeInTheDocument();
    expect(within(identity as HTMLElement).getByTestId('field-training.mode')).toBeInTheDocument();
    expect(form.querySelector('[data-group="training"]')).toBeNull();
    const components = form.querySelector('.config-model-components')!;
    expect(within(components as HTMLElement).getAllByRole('checkbox')).toHaveLength(2);
    const paths = form.querySelector('.config-model-assets-sdxl')!;
    expect(Array.from(paths.children).map(node => node.getAttribute('data-field-path'))).toEqual([
      'model.dit_path', 'model.text_encoder_path', 'model.text_encoder_2_path', 'model.vae_path',
    ]);
    const prediction = form.querySelector('.config-model-prediction')!;
    expect(within(prediction as HTMLElement).getByTestId('field-model.prediction_type')).toBeInTheDocument();
    expect(within(prediction as HTMLElement).getByTestId('field-model.zero_terminal_snr')).toBeInTheDocument();
    expect(screen.getByText('CLIP-L / CLIP-G 双分词器自动读取，缺省使用内置资源。')).toBeInTheDocument();
  });

  it('finds training mode in its new model group and retains the preset editor mode control', () => {
    const value = schemaDefaults(trainSchema);
    value.model.family = 'sdxl';
    const view = render(<SchemaForm schema={trainSchema} value={value} onChange={() => {}} compact search="training.mode" family={sdxl}/>);
    expect(screen.getByTestId('field-training.mode').closest('[data-group]')).toHaveAttribute('data-group', 'model');
    expect(screen.queryByTestId('field-model.family')).not.toBeInTheDocument();
    expect(screen.queryByText(/双分词器自动读取/)).not.toBeInTheDocument();
    view.rerender(<SchemaForm schema={presetEditorSchema(trainSchema)} value={value} onChange={() => {}} compact family={sdxl}/>);
    expect(screen.getByTestId('field-training.mode').closest('[data-group]')).toHaveAttribute('data-group', 'model');
    expect(screen.getByRole('checkbox', { name: '训练文本编码器' })).toBeDisabled();
    expect(screen.queryByTestId('field-model.family')).not.toBeInTheDocument();
    expect(screen.queryByTestId('field-model.dit_path')).not.toBeInTheDocument();
  });

  it.each([
    ['sampling.sampler', ['Euler', 'Heun']], ['sampling.scheduler', ['Uniform']],
    ['objective.timestep_sampling', ['uniform', 'logit_normal']], ['objective.weighting', ['none', 'Min-SNR']],
  ])('offers only SDXL-supported %s choices', async (path, expected) => {
    render(<Editor advanced groupFilter={['sampling', 'objective']} />);
    fireEvent.click(within(screen.getByTestId(`field-${path}`)).getByRole('combobox'));
    expect(screen.getAllByRole('option').map(option => option.textContent)).toEqual(expected);
    expect(screen.queryByTestId('field-sampling.shift')).not.toBeInTheDocument();
    expect(screen.queryByTestId('field-objective.shift')).not.toBeInTheDocument();
    expect(screen.queryByTestId('field-sampling.er_sde_order')).not.toBeInTheDocument();
    await act(async () => {});
  });

  it('exposes DDPM Min-SNR Gamma only when selected and keeps Flow choices separate', async () => {
    const { unmount } = render(<Editor advanced groupFilter={['objective']} />);
    expect(screen.queryByTestId('field-objective.snr_gamma')).not.toBeInTheDocument();
    choose('损失加权', 'Min-SNR');
    const gamma = screen.getByRole('spinbutton', { name: 'Min-SNR Gamma' });
    fireEvent.change(gamma, { target: { value: '7' } });
    expect(JSON.parse(screen.getByTestId('changed-config').textContent!).objective).toMatchObject({ weighting: 'min_snr', snr_gamma: 7 });
    choose('损失加权', 'none');
    expect(screen.queryByTestId('field-objective.snr_gamma')).not.toBeInTheDocument();
    unmount();
    render(<Editor advanced groupFilter={['objective']} family={flow} />);
    fireEvent.click(screen.getByRole('combobox', { name: '损失加权' }));
    expect(screen.queryByRole('option', { name: 'Min-SNR' })).not.toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'snr_like' })).toBeInTheDocument();
    await act(async () => {});
  });

  it('retains ER-SDE and all current scheduler choices for Flow families', () => {
    render(<Editor advanced groupFilter={['sampling']} family={flow} />);
    choose('采样器', 'ER-SDE');
    expect(screen.getByTestId('field-sampling.er_sde_order')).toBeInTheDocument();
    fireEvent.click(within(screen.getByTestId('field-sampling.scheduler')).getByRole('combobox'));
    expect(screen.getAllByRole('option').map(option => option.textContent)).toEqual(['Uniform', 'Simple', 'SGM Uniform', 'Normal']);
  });
});

describe('SDXL configuration ownership', () => {
  it('removes unsupported memory options when changing to SDXL without mutating the source recipe', () => {
    const memory = { base_precision: 'fp8_e4m3', blocks_to_swap: 20, activation_checkpointing: 'unsloth', compile: true, allow_tf32: false };
    const current = { model: { family: 'anima' }, memory, dataset: {}, sampling: {}, objective: {} };
    const next = changeModelFamily(current, { ...sdxl, capabilities: ['activation_checkpointing', 'masked_loss'] }, []);
    expect(next.memory).toEqual({ base_precision: 'auto', blocks_to_swap: 0, activation_checkpointing: 'block', compile: false, allow_tf32: false });
    expect(current.memory).toBe(memory);
    expect(current.memory).toEqual({ base_precision: 'fp8_e4m3', blocks_to_swap: 20, activation_checkpointing: 'unsloth', compile: true, allow_tf32: false });
  });

  it('uses target capabilities to preserve supported FP8 or swap and select compatible checkpointing', () => {
    const current = { model: { family: 'anima' }, memory: { base_precision: 'fp8_e4m3', blocks_to_swap: 20, activation_checkpointing: 'none', compile: false }, dataset: {}, sampling: {}, objective: {} };
    const flux = { ...flow, name: 'flux', capabilities: ['activation_checkpointing', 'fp8_base', 'masked_loss'] };
    expect(changeModelFamily(current, flux, []).memory).toEqual({ base_precision: 'fp8_e4m3', blocks_to_swap: 0, activation_checkpointing: 'none', compile: false });
    const flux2 = { ...flow, name: 'flux2', capabilities: ['activation_checkpointing', 'block_swap', 'masked_loss'] };
    expect(changeModelFamily(current, flux2, []).memory).toEqual({ base_precision: 'auto', blocks_to_swap: 20, activation_checkpointing: 'block', compile: false });
    expect(changeModelFamily(current, { ...flow, name: 'toy', capabilities: [] }, []).memory.activation_checkpointing).toBe('none');
  });

  it('keeps existing memory settings while filling same-family paths or applying same-family presets', () => {
    const current = { model: { family: 'anima' }, memory: { base_precision: 'fp8_e4m3', blocks_to_swap: 20, activation_checkpointing: 'unsloth', compile: false } };
    expect(fillDefaultModels(current, []).memory).toEqual(current.memory);
    expect(applyTrainingPreset(current, { model: { family: 'anima' }, optimizer: { lr: 0.0002 } }).memory).toEqual(current.memory);
  });

  it('fills both registered encoders, clears old-family paths and incompatible choices only on an explicit family change', () => {
    const assets = [asset('dit'), asset('text_encoder'), asset('text_encoder_2'), asset('text_encoder_2', { family: 'anima', path: '/wrong-family' })];
    const current = { model: { family: 'anima', dit_path: '/old/base', text_encoder_path: '/old/clip-l', text_encoder_2_path: '/old/clip-g', prediction_type: 'v_prediction', zero_terminal_snr: true },
      sampling: { sampler: 'er_sde', scheduler: 'normal', shift: 3 }, objective: { timestep_sampling: 'shift', weighting: 'cosmos' }, dataset: { sources: [{ path: '/keep/images' }] } };
    const next = changeModelFamily(current, sdxl, assets);
    expect(next.model).toMatchObject({ family: 'sdxl', dit_path: '/models/dit.safetensors', text_encoder_path: '/models/text_encoder.safetensors', text_encoder_2_path: '/models/text_encoder_2.safetensors', prediction_type: 'epsilon', zero_terminal_snr: false });
    expect(next.sampling).toMatchObject({ sampler: 'euler', scheduler: 'uniform', shift: 1 });
    expect(next.objective).toMatchObject({ timestep_sampling: 'uniform', weighting: 'none' });
    expect(next.dataset.sources).toEqual(current.dataset.sources);
    const explicit = { model: { family: 'sdxl', text_encoder_2_path: '/my/clip-g' } };
    expect(fillDefaultModels(explicit, assets).model.text_encoder_2_path).toBe('/my/clip-g');
    expect(changeModelFamily(next, flow, []).model.text_encoder_2_path).toBeNull();
    expect(current.model.text_encoder_2_path).toBe('/old/clip-g');
  });

  it('strips both encoder paths from presets and preserves the current project paths when applying one', () => {
    const current = { model: { family: 'sdxl', text_encoder_path: '/keep/clip-l', text_encoder_2_path: '/keep/clip-g' } };
    const preset = { model: { family: 'sdxl', text_encoder_path: '/other/clip-l', text_encoder_2_path: '/other/clip-g', prediction_type: 'v_prediction' } };
    expect(reusableTrainingPreset(preset)).toEqual({ model: { family: 'sdxl', prediction_type: 'v_prediction' } });
    expect(applyTrainingPreset(current, preset)).toEqual({ model: { ...current.model, prediction_type: 'v_prediction' } });
    expect(presetEditorSchema(trainSchema).$defs.ModelConfig.properties.text_encoder_2_path).toBeUndefined();
  });

  it('derives selectable families and weight kinds from the registry without built-in unsupported entries', () => {
    expect(trainingFamilyOptions([sdxl])).toEqual([{ value: 'sdxl', label: 'SDXL 2.6B' }]);
    expect(trainingFamilyOptions([{ ...flow, name: 'new-family', label: 'New family' }])).toEqual([{ value: 'new-family', label: 'New family' }]);
    expect(modelFamilyWeights({ ...flow, weights: [{ field: 'dit_path', label: 'Legacy base', hint: '' }] } as FamilyInfo)).toEqual([{ field: 'dit_path', kind: 'dit', label: 'Legacy base', hint: '', required: true, downloadable: true }]);
  });
});
