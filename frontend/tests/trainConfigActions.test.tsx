import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { handlers } from '../src/mocks/handlers';
import TrainConfig from '../src/pages/TrainConfig/TrainConfig';
import trainSchema from '../src/schema/train-schema.json';
import { schemaDefaults } from '../src/utils/config';
import '../src/i18n';

const server = setupServer(...handlers);
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => { server.resetHandlers(); vi.restoreAllMocks(); });
afterAll(() => server.close());
function showConfig() {
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <MemoryRouter initialEntries={['/projects/p_test/train?tab=train']}><Routes>
      <Route path="/projects/:id/train" element={<TrainConfig />} />
      <Route path="/jobs/:id" element={<div>Created job</div>} />
    </Routes></MemoryRouter>
  </QueryClientProvider>);
}
const enqueue = () => screen.getByRole('button', { name: '开始训练' });

describe('training configuration actions', () => {
  it('keeps one title with its model badge and groups search, advanced options and configuration actions', async () => {
    showConfig();
    const badge = await screen.findByTestId('training-family-badge');
    const heading = screen.getByRole('heading', { name: '训练参数', level: 1 });
    expect(heading.parentElement).toContainElement(badge);
    expect(screen.getAllByRole('heading', { name: '训练参数' })).toHaveLength(1);
    const toolbar = screen.getByRole('group', { name: '训练参数工具栏' });
    const search = within(toolbar).getByRole('textbox', { name: '搜索训练参数' });
    const advanced = within(toolbar).getByRole('checkbox', { name: '高级选项' });
    expect(search.closest('.training-toolbar-filters')).toContainElement(advanced);
    expect(within(toolbar).getByRole('button', { name: '保存草稿' })).toBeInTheDocument();
    expect(within(toolbar).getByRole('combobox', { name: /加载预设/ })).toBeInTheDocument();
    fireEvent.change(search, { target: { value: 'vae_path' } });
    expect(await screen.findByTestId('field-model.vae_path')).toBeInTheDocument();
    expect(screen.getByText('搜索所有分区，包含高级参数')).toBeInTheDocument();
    fireEvent.click(within(toolbar).getByRole('button', { name: '清空搜索' }));
    expect(screen.getByRole('tab', { name: '训练参数' })).toHaveAttribute('aria-selected', 'true');
    expect(search).toHaveValue('');
    expect(heading.parentElement).toContainElement(badge);
  });

  it('loads the live schema and submits job name, priority and local schedule', async () => {
    const schema: any = structuredClone(trainSchema);
    schema.$defs.LoopConfig.properties.backend_added = { type: 'string', title: 'Backend added field' };
    let submitted: any;
    server.use(
      http.get('/api/schema/train', () => HttpResponse.json(schema)),
      http.post('/api/jobs', async ({ request }) => { submitted = await request.json(); return HttpResponse.json({ id: 'new-job' }); }),
    );
    showConfig();
    expect(enqueue()).toBeDisabled();
    const dynamicField = await screen.findByTestId('field-loop.backend_added');
    fireEvent.change(within(dynamicField).getByRole('textbox'), { target: { value: 'from-current-backend' } });
    fireEvent.change(screen.getByRole('textbox', { name: '任务名称（可选）' }), { target: { value: 'Scheduled training' } });
    fireEvent.click(screen.getByText('排期', {selector:'summary'}));
    fireEvent.change(screen.getByRole('spinbutton', { name: '优先级' }), { target: { value: '7' } });
    fireEvent.change(screen.getByLabelText('计划开始时间（留空立即排队）'), { target: { value: '2026-10-12T08:30' } });
    await waitFor(() => expect(enqueue()).toBeEnabled());
    fireEvent.click(enqueue());
    await screen.findByText('Created job');
    expect(submitted).toMatchObject({ name: 'Scheduled training', priority: 7, project_id: 'p_test', config: { loop: { backend_added: 'from-current-backend' } } });
    expect(submitted.scheduled_at).toBe(new Date('2026-10-12T08:30').getTime() / 1000);
  });

  it('imports TOML through validation, saves a preset and exports the current config', async () => {
    let imported: any; let preset: any; let exported: any;
    const config = schemaDefaults(trainSchema);
    config.dataset.caption.trigger_word = 'from_toml';
    server.use(
      http.post('/api/config/import', async ({ request }) => { imported = await request.json(); return HttpResponse.json(config); }),
      http.post('/api/presets', async ({ request }) => { preset = await request.json(); return HttpResponse.json({ ...preset, builtin: false }); }),
      http.post('/api/config/export', async ({ request }) => { exported = await request.json(); return HttpResponse.json({ text: '[dataset.caption]\ntrigger_word = "from_toml"\n' }); }),
    );
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn(() => 'blob:test') });
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() });
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    showConfig();
    fireEvent.click(screen.getByText('配置工具', {selector:'summary'}));
    await waitFor(() => expect(screen.getByRole('button', { name: '导入 TOML' })).toBeEnabled());
    fireEvent.click(screen.getByRole('button', { name: '导入 TOML' }));
    fireEvent.change(screen.getByRole('textbox', { name: 'TOML 配置内容' }), { target: { value: '[dataset.caption]\ntrigger_word="from_toml"' } });
    fireEvent.click(screen.getByRole('button', { name: '校验并应用' }));
    await waitFor(() => expect(screen.queryByRole('button', { name: '校验并应用' })).not.toBeInTheDocument());
    fireEvent.click(screen.getByRole('tab', {name:'数据与分桶'}));
    fireEvent.click(screen.getByRole('checkbox',{name:'高级选项'}));
    await waitFor(() => expect(within(screen.getByTestId('field-dataset.caption.trigger_word')).getByRole('textbox')).toHaveValue('from_toml'));
    expect(imported.format).toBe('toml');
    fireEvent.change(screen.getByRole('textbox', { name: '新预设名称' }), { target: { value: 'my-preset' } });
    fireEvent.click(screen.getByRole('button', { name: '另存为预设' }));
    await waitFor(() => expect(preset?.config?.dataset?.caption?.trigger_word).toBe('from_toml'));
    expect(preset.name).toBe('my-preset');
    fireEvent.click(screen.getByRole('button', { name: '导出 TOML' }));
    await waitFor(() => expect(click).toHaveBeenCalled());
    expect(exported.config.dataset.caption.trigger_word).toBe('from_toml');
    expect(exported.format).toBe('toml');
  });

  it('blocks enqueue on plan errors and shows failed requests in the page', async () => {
    server.use(http.post('/api/plan', () => HttpResponse.json({ ok: false, errors: [{ loc: 'model.dit_path', msg: 'Weight file is missing' }], warnings: [] })));
    showConfig();
    await screen.findByText('1 项待配置');
    expect(screen.queryByText('Weight file is missing')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', {name:'1 项待配置'}));
    fireEvent.click(screen.getByRole('button', {name:'配置主模型 / DiT'}));
    await waitFor(() => expect(screen.getByRole('tab', {name:'底模与输出'})).toHaveAttribute('aria-selected','true'));
    expect(screen.getAllByText('找不到指定文件，请检查训练机上的路径').length).toBeGreaterThan(0);
    fireEvent.click(screen.getByText('配置工具', {selector:'summary'}));
    expect(enqueue()).toBeDisabled();
    server.use(http.post('/api/config/export', () => HttpResponse.json({ error: { code: 'config.invalid', message: 'Export validation failed' } }, { status: 400 })));
    fireEvent.click(screen.getByRole('button', { name: '导出 TOML' }));
    expect(await screen.findByText('Export validation failed')).toBeInTheDocument();
  });

  it.each([
    ['model.text_encoder_path', '文本编码器', 'textbox'],
    ['optimizer.type', '优化器', 'combobox'],
  ])('focuses the editable control for %s instead of its help button', async (path, label, role) => {
    server.use(http.post('/api/plan', () => HttpResponse.json({ ok: false, errors: [{ loc: path, msg: 'Field required' }], warnings: [] })));
    showConfig();
    fireEvent.click(await screen.findByRole('button', { name: '1 项待配置' }));
    fireEvent.click(screen.getByRole('button', { name: `配置${label}` }));
    const field = await screen.findByTestId(`field-${path}`);
    const input = within(field).getByRole(role);
    await waitFor(() => expect(input).toHaveFocus());
    expect(within(field).getByRole('button', { name: `${label} 说明` })).not.toHaveFocus();
  });

  it('shows valid buckets while empty sampling prompts keep validation visible and training disabled', async () => {
    const issue = {loc: '', msg: 'Value error, sampling.enabled requires sampling.prompts or sampling.prompts_file'};
    const config = schemaDefaults(trainSchema);
    config.model.family = 'toy';
    config.dataset.sources = [{path:'/data/eight-images', repeats:1}];
    config.sampling = {...config.sampling, enabled:true, prompts:[], prompts_file:null};
    const createJob = vi.fn();
    server.use(
      http.get('/api/projects/p_test/config', () => HttpResponse.json(config)),
      http.post('/api/config/validate', () => HttpResponse.json({ok:false, errors:[issue], warnings:[]})),
      http.post('/api/plan', () => HttpResponse.json({ok:false, errors:[issue], warnings:[], images:8,items:8,captioned:8,buckets:[{w:64,h:64,items:8,batches:4}],steps_per_epoch:4,total_steps:12})),
      http.post('/api/jobs', () => {createJob();return HttpResponse.json({id:'should-not-exist'});}),
    );
    showConfig();
    await screen.findByRole('button', {name:'64 × 64, 8 样本'});
    expect(screen.getByRole('button', {name:'1 项待配置'})).toBeInTheDocument();
    expect(enqueue()).toBeDisabled();
    fireEvent.click(enqueue());
    expect(createJob).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', {name:'分桶明细表'}));
    expect(screen.getByRole('table')).toHaveTextContent('64 × 6484');
    fireEvent.click(screen.getByRole('button', {name:'1 项待配置'}));
    expect(screen.getByRole('region', {name:'训练前检查'})).toHaveTextContent(/采样|sampling/);
  });

  it('shows the invalid TOML field and reason while retaining the import text and current draft', async () => {
    const badToml = '[checkpoint]\nsave_full_state = true';
    server.use(http.post('/api/config/import', () => HttpResponse.json({
      error: {
        code: 'config.invalid', message: 'invalid config',
        details: { errors: [{ loc: 'checkpoint.save_full_state', msg: 'Extra inputs are not permitted' }] },
      },
    }, { status: 400 })));
    showConfig();
    fireEvent.click(screen.getByText('配置工具', {selector:'summary'}));
    await waitFor(() => expect(screen.getByRole('button', { name: '导入 TOML' })).toBeEnabled());
    fireEvent.click(screen.getByRole('tab', {name:'数据与分桶'}));
    fireEvent.click(screen.getByRole('checkbox',{name:'高级选项'}));
    const trigger = within(screen.getByTestId('field-dataset.caption.trigger_word')).getByRole('textbox');
    fireEvent.change(trigger, { target: { value: 'keep_draft' } });
    fireEvent.click(screen.getByRole('button', { name: '导入 TOML' }));
    fireEvent.change(screen.getByRole('textbox', { name: 'TOML 配置内容' }), { target: { value: badToml } });
    fireEvent.click(screen.getByRole('button', { name: '校验并应用' }));
    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('invalid config');
    expect(alert).toHaveTextContent('checkpoint.save_full_state: Extra inputs are not permitted');
    expect(screen.getByRole('textbox', { name: 'TOML 配置内容' })).toHaveValue(badToml);
    expect(trigger).toHaveValue('keep_draft');
    expect(screen.getByRole('button', { name: '校验并应用' })).toBeEnabled();
  });
});
