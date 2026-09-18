import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
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
  it('only displays effective DTK precision for the current plan and restores original choices when disabled', async () => {
    const config = schemaDefaults(trainSchema);
    config.model.family = 'anima'; config.model.attention = 'xformers';
    config.training.mode = 'full'; config.training.train_backbone = true;
    config.loop.deterministic = true; config.loop.mixed_precision = 'bf16';
    config.memory.allow_tf32 = true;
    const replies: Array<(result: object) => void> = [];
    server.use(
      http.get('/api/projects/p_test/config', () => HttpResponse.json(config)),
      http.post('/api/plan', () => new Promise<Response>(resolve => {
        replies.push(result => resolve(HttpResponse.json(result)));
      })),
    );
    const confirmed = {ok:true,errors:[],warnings:[],params:{},compute_policy:{id:'dtk-full-fp32-math-v1',mixed_precision:'no',allow_tf32:false,attention:'sdpa',sdpa_backend:'math'}};
    showConfig();
    await screen.findByTestId('field-loop.epochs');
    fireEvent.click(screen.getByRole('button',{name:'高级'}));
    expect(screen.getByRole('combobox',{name:'混合精度'})).toHaveTextContent('BF16');
    await waitFor(() => expect(replies).toHaveLength(1));
    await act(async () => replies[0](confirmed));
    expect(await screen.findByRole('status',{name:'混合精度'})).toHaveTextContent('FP32 计算');
    expect(screen.getByRole('status',{name:'注意力后端'})).toHaveTextContent('数学实现');
    fireEvent.click(screen.getByRole('checkbox',{name:'可复现训练'}));
    expect(screen.getByRole('combobox',{name:'混合精度'})).toHaveTextContent('BF16');
    expect(screen.getByRole('combobox',{name:'注意力后端'})).toHaveTextContent('xFormers');
    expect(screen.getByRole('checkbox',{name:'允许 TF32'})).toBeChecked();
    await waitFor(() => expect(replies).toHaveLength(2));
    // Even a stale/malformed server policy cannot lock a draft whose switch is off.
    await act(async () => replies[1](confirmed));
    expect(screen.queryByRole('status',{name:'混合精度'})).not.toBeInTheDocument();
    expect(screen.getByRole('combobox',{name:'混合精度'})).toHaveTextContent('BF16');
  });

  it('renders one continuous form in workflow order even when opened through a legacy tab link', async () => {
    showConfig(); // Legacy ?tab=train locates the training step, without hiding the model.
    await screen.findByTestId('field-model.dit_path');
    fireEvent.click(screen.getByRole('button', {name:'高级'}));
    const form = screen.getByTestId('schema-form');
    const groups = Array.from(form.querySelectorAll(':scope > [data-group]')).map(node => node.getAttribute('data-group'));
    expect(groups).toEqual(['model','dataset','caption','loop','adapter','optimizer','scheduler','memory','objective','sampling','validation','checkpoint','logging']);
    expect(screen.getByTestId('field-checkpoint.save_dtype')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', {name:/保存与恢复$/}));
    expect(screen.getByTestId('field-model.dit_path')).toBeInTheDocument();
    expect(Array.from(form.querySelectorAll(':scope > [data-group]')).map(node => node.getAttribute('data-group'))).toEqual(groups);
  });

  it('shows translated field errors once and keeps additional unrecognized system details', async () => {
    const extra = 'Runtime backend: unexpected worker exit 17';
    server.use(http.post('/api/plan', () => HttpResponse.json({ok:false, errors:[
      {loc:'optimizer.d0',msg:'Input should be greater than 0'},
      {loc:'memory',msg:extra},
    ],warnings:[]})));
    showConfig();
    const launch = screen.getByRole('group', {name:'训练启动操作'});
    fireEvent.click(await within(launch).findByRole('button', {name:'2 项待配置'}));
    const panel = screen.getByRole('region', {name:'训练前检查'});
    expect(within(panel).getByRole('button', {name:'配置初始步长估计（D0）'})).toHaveTextContent('输入值应大于 0');
    expect(within(panel).queryByText('Input should be greater than 0')).not.toBeInTheDocument();
    expect(within(panel).getAllByText('技术详情')).toHaveLength(1);
    fireEvent.click(within(panel).getByText('技术详情'));
    expect(within(panel).getByText(extra)).toBeVisible();
  });

  it('marks a field whose reason has no Chinese wording without repeating the panel instruction on it', async () => {
    const raw = 'Unsupported by the selected optimizer';
    server.use(http.post('/api/plan', () => HttpResponse.json({ok:false, errors:[{loc:'optimizer.weight_decay',msg:raw}],warnings:[]})));
    showConfig();
    const field = await screen.findByTestId('field-optimizer.weight_decay');
    await waitFor(() => expect(field).toHaveClass('config-field-invalid'));
    expect(within(field).getByRole('spinbutton')).toHaveAttribute('aria-invalid','true');
    // The border carries the rejection; the untranslatable reason belongs to the preflight panel.
    expect(field.querySelector('.config-field-error')).toBeNull();
    expect(within(field).queryByText(/展开详情查看具体原因/)).not.toBeInTheDocument();
    expect(within(field).queryByText(raw)).not.toBeInTheDocument();
    const launch = screen.getByRole('group', {name:'训练启动操作'});
    fireEvent.click(await within(launch).findByRole('button', {name:'1 项待配置'}));
    const panel = screen.getByRole('region', {name:'训练前检查'});
    expect(within(panel).getByRole('button', {name:'配置权重衰减'})).toHaveTextContent('此配置未通过检查，展开详情查看具体原因');
    fireEvent.click(within(panel).getByText('技术详情'));
    expect(within(panel).getByText(raw)).toBeVisible();
  });
  it('invalidates completed workflow steps on edits and waits for the current server plan', async () => {
    const replies: Array<(result: object) => void> = [];
    server.use(http.post('/api/plan', () => new Promise<Response>(resolve => {
      replies.push(result => resolve(HttpResponse.json(result)));
    })));
    showConfig();
    await screen.findByTestId('field-loop.epochs');
    const model = screen.getByRole('button', {name:/模型选择$/});
    const optimizer = screen.getByRole('button', {name:/优化器$/});
    expect(model).toHaveAccessibleDescription('待检查');
    await waitFor(() => expect(replies).toHaveLength(1));
    await act(async () => replies[0]({ok:false,errors:[{loc:'model.dit_path',msg:'Weight file is missing'}],warnings:[],params:{}}));
    await waitFor(() => expect(model).toHaveAccessibleDescription('1 项待配置'));
    expect(optimizer).toHaveAccessibleDescription('检查通过');
    fireEvent.click(model);
    const path = within(await screen.findByTestId('field-model.dit_path')).getByRole('textbox');
    fireEvent.change(path,{target:{value:'/models/checked.safetensors'}});
    expect(optimizer).toHaveAccessibleDescription('待检查');
    await waitFor(() => expect(replies).toHaveLength(2));
    expect(model).not.toHaveAccessibleDescription('检查通过');
    await act(async () => replies[1]({ok:true,errors:[],warnings:[],params:{}}));
    await waitFor(() => expect(model).toHaveAccessibleDescription('检查通过'));
    fireEvent.change(path,{target:{value:'/models/not-yet-checked.safetensors'}});
    expect(model).toHaveAccessibleDescription('待检查');
    expect(enqueue()).toBeDisabled();
  });

  it('preserves advanced values when switching to Simple and back', async () => {
    showConfig();
    await screen.findByTestId('field-loop.epochs');
    fireEvent.click(screen.getByRole('button', {name:'高级'}));
    const field = await screen.findByTestId('field-adapter.rs_lora');
    const toggle = within(field).getByRole('checkbox');
    fireEvent.click(toggle);
    const edited = (toggle as HTMLInputElement).checked;
    fireEvent.click(screen.getByRole('button', {name:'简单'}));
    expect(screen.queryByTestId('field-adapter.rs_lora')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', {name:'高级'}));
    expect((within(screen.getByTestId('field-adapter.rs_lora')).getByRole('checkbox') as HTMLInputElement).checked).toBe(edited);
  });
  it('recovers from an empty search and exposes a plan disclosure without changing the draft', async () => {
    showConfig();
    const search = screen.getByRole('textbox', {name: '搜索训练参数'});
    await screen.findByTestId('field-loop.epochs');
    fireEvent.change(search, {target: {value: '不存在的参数123'}});
    expect(screen.getByText('没有匹配的参数。')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', {name: '返回参数分区'}));
    expect(search).toHaveValue(''); expect(search).toHaveFocus();
    const toggle = screen.getByRole('button', {name: /训练估算与分桶/});
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByTestId('field-loop.epochs')).toBeInTheDocument();
  });

  it('keeps one title with its model badge and groups search, advanced options and configuration actions', async () => {
    showConfig();
    const badge = await screen.findByTestId('training-family-badge');
    const heading = screen.getByRole('heading', { name: '训练参数', level: 1 });
    expect(heading.parentElement).toContainElement(badge);
    expect(screen.getAllByRole('heading', { name: '训练参数' })).toHaveLength(1);
    expect(screen.queryByRole('tablist', {name: '参数分区'})).not.toBeInTheDocument();
    const launch = screen.getByRole('group', {name: '训练启动操作'});
    expect(launch.closest('.parameter-workspace-header')).toContainElement(heading);
    expect(launch.closest('footer')).toBeNull();
    expect(launch.querySelector('.launch-estimate')).toBeNull();
    const toolbar = screen.getByRole('group', { name: '训练参数工具栏' });
    const search = within(toolbar).getByRole('textbox', { name: '搜索训练参数' });
    const advanced = within(toolbar).getByRole('button', { name: '高级' });
    expect(search.closest('.training-toolbar-filters')).toContainElement(advanced);
    expect(within(toolbar).getByRole('button', { name: '保存草稿' })).toBeInTheDocument();
    expect(within(toolbar).getByRole('combobox', { name: /加载预设/ })).toBeInTheDocument();
    fireEvent.change(search, { target: { value: 'vae_path' } });
    expect(await screen.findByTestId('field-model.vae_path')).toBeInTheDocument();
    expect(screen.getByText('搜索所有分区，包含高级参数')).toBeInTheDocument();
    fireEvent.click(within(toolbar).getByRole('button', { name: '清空搜索' }));
    expect(screen.getByTestId('field-loop.epochs')).toBeInTheDocument();
    expect(search).toHaveValue('');
    expect(heading.parentElement).toContainElement(badge);
  });

  it('loads the live schema and submits job name, explicit GPU, priority and local schedule', async () => {
    const schema: any = structuredClone(trainSchema);
    schema.$defs.LoopConfig.properties.backend_added = { type: 'string', title: 'Backend added field' };
    let submitted: any;
    server.use(
      http.get('/api/schema/train', () => HttpResponse.json(schema)),
      http.get('/api/queue/devices', () => HttpResponse.json({devices:[{device:'cuda:0',name:'BW',job_id:null},{device:'cuda:1',name:'BW',job_id:null}],max_concurrent:null})),
      http.post('/api/jobs', async ({ request }) => { submitted = await request.json(); return HttpResponse.json({ id: 'new-job' }); }),
    );
    showConfig();
    expect(enqueue()).toBeDisabled();
    const dynamicField = await screen.findByTestId('field-loop.backend_added');
    expect(within(screen.getByRole('group', {name:'训练启动操作'})).getByRole('button', {name:'训练显卡'})).toHaveTextContent('自动');
    fireEvent.click(screen.getByRole('button', {name:'训练显卡'}));
    fireEvent.click(await screen.findByRole('radio', {name:/GPU 1.*空闲/}));
    fireEvent.keyDown(document, {key:'Escape'});
    fireEvent.change(within(dynamicField).getByRole('textbox'), { target: { value: 'from-current-backend' } });
    fireEvent.change(screen.getByRole('textbox', { name: '任务名称（可选）' }), { target: { value: 'Scheduled training' } });
    fireEvent.click(screen.getByText('排期', {selector:'summary'}));
    fireEvent.change(screen.getByRole('spinbutton', { name: '优先级' }), { target: { value: '7' } });
    fireEvent.change(screen.getByLabelText('计划开始时间（留空立即排队）'), { target: { value: '2026-10-12T08:30' } });
    await waitFor(() => expect(enqueue()).toBeEnabled());
    fireEvent.click(enqueue());
    await screen.findByText('Created job');
    expect(submitted).toMatchObject({ name: 'Scheduled training', gpu_devices: ['cuda:1'], priority: 7, project_id: 'p_test', config: { loop: { gpu_count: 1, backend_added: 'from-current-backend' } } });
    expect(submitted.scheduled_at).toBe(new Date('2026-10-12T08:30').getTime() / 1000);
  });

  it.each([false, true])('submits matching two-GPU configuration and queue devices (automatic=%s)', async automatic => {
    let submitted: any;
    const plans: any[] = [];
    server.use(
      http.get('/api/queue/devices', () => HttpResponse.json({devices:[{device:'cuda:0',name:'BW 0',status:'running',job_id:'busy'}, {device:'cuda:1',name:'BW 1',status:'free'}],max_concurrent:null})),
      http.post('/api/plan', async ({request}) => { const body: any = await request.json(); plans.push(body); return HttpResponse.json({ok:true,errors:[],warnings:[],params:{}}); }),
      http.post('/api/jobs', async ({request}) => {submitted = await request.json(); return HttpResponse.json({id:'new-job'});}),
    );
    showConfig();
    fireEvent.change(await screen.findByRole('spinbutton', {name:'训练显卡数量'}), {target:{value:'2'}});
    fireEvent.click(screen.getByRole('button', {name:'训练显卡'}));
    await screen.findByRole('checkbox', {name:/GPU 1.*BW 1/});
    if (!automatic) {
      fireEvent.click(screen.getByRole('checkbox', {name:/GPU 0.*BW 0/}));
      expect(enqueue()).toBeDisabled();
      expect(screen.getByText('已选 1 张，还需选择 1 张；也可改为自动。')).toBeInTheDocument();
      fireEvent.click(screen.getByRole('checkbox', {name:/GPU 1.*BW 1/}));
      expect(screen.getByText('所选显卡正在使用，任务会等待它空闲后启动。')).toBeInTheDocument();
    }
    expect(enqueue()).toBeDisabled(); // The previous one-GPU plan cannot approve this edit.
    fireEvent.keyDown(document, {key:'Escape'});
    expect(screen.getByRole('combobox', {name:'多卡训练方式'})).toBeInTheDocument();
    await waitFor(() => expect(enqueue()).toBeEnabled());
    expect(plans[plans.length - 1].config.loop.gpu_count).toBe(2);
    fireEvent.click(enqueue());
    await screen.findByText('Created job');
    expect(submitted).toMatchObject({gpu_devices:automatic ? [] : ['cuda:0','cuda:1'],config:{loop:{gpu_count:2}}});
  });

  it('preserves explicit choices when increasing the count and announces removed cards when decreasing it', async () => {
    server.use(http.get('/api/queue/devices', () => HttpResponse.json({devices:[{device:'cuda:0',name:'BW 0',status:'free'}, {device:'cuda:1',name:'BW 1',status:'free'}],max_concurrent:null})));
    showConfig();
    const count = await screen.findByRole('spinbutton', {name:'训练显卡数量'});
    fireEvent.click(screen.getByRole('button', {name:'训练显卡'}));
    fireEvent.click(await screen.findByRole('radio', {name:/GPU 1.*BW 1/}));
    fireEvent.keyDown(document, {key:'Escape'});
    fireEvent.change(count, {target:{value:'2'}});
    expect(screen.getByRole('button', {name:'训练显卡'})).toHaveTextContent('GPU 1');
    expect(screen.getByText('已选 1 张，还需选择 1 张；也可改为自动。')).toBeInTheDocument();
    expect(enqueue()).toBeDisabled();
    fireEvent.click(screen.getByRole('button', {name:'训练显卡'}));
    fireEvent.click(screen.getByRole('checkbox', {name:/GPU 0.*BW 0/}));
    fireEvent.keyDown(document, {key:'Escape'});
    fireEvent.change(count, {target:{value:'1'}});
    expect(screen.getByRole('button', {name:'训练显卡'})).toHaveTextContent('GPU 1');
    expect(screen.getByRole('button', {name:'训练显卡'})).not.toHaveTextContent('GPU 0');
    expect(screen.getByText('显卡数量已减少为 1 张，保留前 1 张，已移除 GPU 0。')).toBeInTheDocument();
    expect(count).toHaveValue(1);
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
    fireEvent.click(screen.getByRole('button', { name: /数据与分桶$/ }));
    fireEvent.click(screen.getByRole('button',{name:'高级'}));
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
    await within(screen.getByRole('group', {name: '训练启动操作'})).findByText('1 项待配置');
    expect(screen.queryByText('Weight file is missing')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', {name:'1 项待配置'}));
    expect(within(screen.getByRole('region', {name: '训练前检查'})).queryByText('技术详情')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', {name:'配置主模型 / DiT'}));
    await waitFor(() => expect(screen.getByTestId('field-model.dit_path')).toBeInTheDocument());
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
    // Wait for the debounced plan without repeatedly traversing every field's
    // accessibility tree; then retain the actual button/focus contract below.
    await within(screen.getByRole('group', {name: '训练启动操作'})).findByText('1 项待配置');
    fireEvent.click(screen.getByRole('button', { name: '1 项待配置' }));
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
    await screen.findByTestId('plan-buckets');
    expect(screen.getByRole('button', {name:'64 × 64, 8 样本'})).toBeInTheDocument();
    expect(screen.getByRole('button', {name:'1 项待配置'})).toBeInTheDocument();
    expect(enqueue()).toBeDisabled();
    fireEvent.click(enqueue());
    expect(createJob).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', {name:'分桶明细表'}));
    expect(screen.getByRole('table')).toHaveTextContent('64 × 6484');
    fireEvent.click(screen.getByRole('button', {name:'1 项待配置'}));
    expect(screen.getByRole('region', {name:'训练前检查'})).toHaveTextContent(/采样|sampling/);
  });


  it('keeps an in-flight TOML import in its dialog so a late response cannot replace later edits', async () => {
    let finish = () => {};
    const pending = new Promise<void>(resolve => {finish = resolve;});
    server.use(http.post('/api/config/import', async () => {await pending; return HttpResponse.json(schemaDefaults(trainSchema));}));
    showConfig();
    fireEvent.click(screen.getByText('配置工具', {selector:'summary'}));
    await waitFor(() => expect(screen.getByRole('button', {name:'导入 TOML'})).toBeEnabled());
    fireEvent.click(screen.getByRole('button', {name:'导入 TOML'}));
    const dialog = screen.getByRole('dialog', {name:'导入 TOML'});
    fireEvent.change(within(dialog).getByRole('textbox', {name:'TOML 配置内容'}), {target:{value:'[loop]\nmax_steps=2'}});
    fireEvent.click(within(dialog).getByRole('button', {name:'校验并应用'}));
    expect(within(dialog).getByRole('textbox', {name:'TOML 配置内容'})).toBeDisabled();
    expect(within(dialog).getByRole('button', {name:'关闭'})).toBeDisabled();
    fireEvent.keyDown(dialog, {key:'Escape'});
    expect(dialog).toBeInTheDocument();
    finish();
    await waitFor(() => expect(screen.queryByRole('dialog', {name:'导入 TOML'})).not.toBeInTheDocument());
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
    fireEvent.click(screen.getByRole('button', { name: /数据与分桶$/ }));
    fireEvent.click(screen.getByRole('button',{name:'高级'}));
    const trigger = within(screen.getByTestId('field-dataset.caption.trigger_word')).getByRole('textbox');
    fireEvent.change(trigger, { target: { value: 'keep_draft' } });
    fireEvent.click(screen.getByRole('button', { name: '导入 TOML' }));
    fireEvent.change(screen.getByRole('textbox', { name: 'TOML 配置内容' }), { target: { value: badToml } });
    fireEvent.click(screen.getByRole('button', { name: '校验并应用' }));
    const dialog = screen.getByRole('dialog', {name:'导入 TOML'});
    const alert = await within(dialog).findByRole('alert');
    expect(alert).toHaveTextContent('invalid config');
    expect(alert).toHaveTextContent('checkpoint.save_full_state: Extra inputs are not permitted');
    expect(screen.getByRole('textbox', { name: 'TOML 配置内容' })).toHaveValue(badToml);
    expect(trigger).toHaveValue('keep_draft');
    expect(screen.getByRole('button', { name: '校验并应用' })).toBeEnabled();
  });
});
