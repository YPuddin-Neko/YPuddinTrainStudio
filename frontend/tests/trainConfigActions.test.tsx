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
    <MemoryRouter initialEntries={['/projects/p_test/train']}><Routes>
      <Route path="/projects/:id/train" element={<TrainConfig />} />
      <Route path="/jobs/:id" element={<div>Created job</div>} />
    </Routes></MemoryRouter>
  </QueryClientProvider>);
}
const enqueue = () => screen.getByRole('button', { name: '开始训练' });

describe('training configuration actions', () => {
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
    await waitFor(() => expect(screen.getByRole('button', { name: '导入 TOML' })).toBeEnabled());
    fireEvent.click(screen.getByRole('button', { name: '导入 TOML' }));
    fireEvent.change(screen.getByRole('textbox', { name: 'TOML 配置内容' }), { target: { value: '[dataset.caption]\ntrigger_word="from_toml"' } });
    fireEvent.click(screen.getByRole('button', { name: '校验并应用' }));
    await waitFor(() => expect(screen.getByRole('textbox', { name: 'dataset.caption.trigger_word' })).toHaveValue('from_toml'));
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
    await screen.findAllByText(/Weight file is missing/);
    expect(enqueue()).toBeDisabled();
    server.use(http.post('/api/config/export', () => HttpResponse.json({ error: { code: 'config.invalid', message: 'Export validation failed' } }, { status: 400 })));
    fireEvent.click(screen.getByRole('button', { name: '导出 TOML' }));
    expect(await screen.findByText('Export validation failed')).toBeInTheDocument();
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
    await waitFor(() => expect(screen.getByRole('button', { name: '导入 TOML' })).toBeEnabled());
    const trigger = screen.getByRole('textbox', { name: 'dataset.caption.trigger_word' });
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
