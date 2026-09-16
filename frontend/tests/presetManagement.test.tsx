import userEvent from '@testing-library/user-event';
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

const legacyBuiltin: Preset = {
  name: 'builtin-anima', description: '旧 API 内置人物参数',
  config: { model: { family: 'anima' }, loop: { epochs: 99 } }, builtin: true, updated_at: 999,
};
function presetRows(): Preset[] {
  return [
    legacyBuiltin,
    { name: 'krea-base', description: 'Krea 风格参数', config: { model: { family: 'krea2' }, loop: { epochs: 3 } }, builtin: false, updated_at: 10 },
    { name: 'my-style', description: '自定义风格', config: { model: { family: 'anima' }, loop: { epochs: 4 }, optimizer: { lr: 0.0002 }, adapter: { rank: 16, algo: 'lora' } }, builtin: false, updated_at: 20 },
  ];
}
function show(seed = presetRows()) {
  const rows = structuredClone(seed);
  const writes: { method: string; name?: string; body: any }[] = [];
  const requestedFamilies: string[] = [];
  let timestamp = 30;
  server.use(
    http.get('/api/presets', () => HttpResponse.json(rows)),
    http.get('/api/config/defaults', ({ request }) => {
      const family = new URL(request.url).searchParams.get('family') || 'anima';
      requestedFamilies.push(family);
      const config = schemaDefaults(trainSchema); config.model.family = family;
      config.model.dit_path = '/registered/dit'; config.model.tokenizer_path = '/registered/tokenizer';
      config.dataset.sources = [{ path: '/project/images' }]; config.dataset.cache_dir = '/project/cache';
      config.sampling.output_dir = '/project/samples'; config.sampling.prompts_file = '/project/prompts';
      config.adapter.resume_weights = '/project/old-adapter';
      if (family === 'krea2') { config.sampling.steps = 28; config.sampling.cfg = 5.5; config.dataset.text_encoding = 'cached'; }
      return HttpResponse.json(config);
    }),
    http.post('/api/presets', async ({ request }) => {
      const body = await request.json() as any; writes.push({ method: 'POST', body });
      const row = { ...body, builtin: false, updated_at: timestamp++ }; rows.push(row);
      return HttpResponse.json(row);
    }),
    http.put('/api/presets/:name', async ({ request, params }) => {
      const body = await request.json() as any; writes.push({ method: 'PUT', name: String(params.name), body });
      const row = { ...body, name: String(params.name), builtin: false, updated_at: timestamp++ };
      const index = rows.findIndex(item => item.name === params.name);
      if (index >= 0) rows[index] = row;
      return HttpResponse.json(row);
    }),
    http.delete('/api/presets/:name', ({ params }) => {
      writes.push({ method: 'DELETE', name: String(params.name), body: null });
      const index = rows.findIndex(item => item.name === params.name);
      if (index >= 0) rows.splice(index, 1);
      return HttpResponse.json({ ok: true });
    }),
  );
  const router = createMemoryRouter([
    { path: '/presets', element: <><Presets /><Link to="/projects">离开预设</Link></> },
    { path: '/projects', element: <div>项目列表页</div> },
  ], { initialEntries: ['/presets'] });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);
  return { rows, writes, requestedFamilies, router, client };
}
async function ready(name = 'my-style') {
  await waitFor(() => expect(screen.getByRole('textbox', { name: '预设名称' })).toHaveValue(name));
  return screen.findByRole('spinbutton', { name: 'loop.epochs' });
}
async function choose(name: string) {
  fireEvent.click(await screen.findByRole('combobox', { name: '选择预设' }));
  fireEvent.click(await screen.findByRole('option', { name: new RegExp(`^${name} · `) }));
}

describe('compact user preset management', () => {
  it('searches advanced parameters across sections and restores the selected section after clearing', async () => {
    show(); await ready();
    const search = screen.getByRole('textbox', {name: '搜索预设参数'});
    expect(screen.queryByTestId('field-adapter.rs_lora')).not.toBeInTheDocument();
    fireEvent.change(search, {target: {value: 'adapter.rs_lora'}});
    expect(screen.getByTestId('field-adapter.rs_lora')).toBeInTheDocument();
    expect(screen.getByRole('button', {name: '高级'})).toHaveAttribute('aria-pressed', 'false');
    fireEvent.change(search, {target: {value: '不存在的参数123'}});
    expect(screen.getByText('没有匹配的参数。')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', {name: '返回参数分区'}));
    expect(search).toHaveValue(''); expect(search).toHaveFocus();
    expect(screen.getByTestId('field-loop.epochs')).toBeInTheDocument();
    expect(screen.queryByTestId('field-adapter.rs_lora')).not.toBeInTheDocument();
    const dataStep = screen.getByRole('button', {name: /数据与分桶$/});
    dataStep.focus();
    await userEvent.keyboard('{Enter}');
    expect(await screen.findByTestId('field-dataset.resolutions')).toBeInTheDocument();
    expect(screen.getByRole('region', {name: '预设参数内容'})).toBeInTheDocument();
    expect(screen.queryByRole('tablist', {name: '预设参数分区'})).not.toBeInTheDocument();
  });

  it('locates a server validation error in another section without discarding draft edits', async () => {
    show(); const epochs = await ready();
    server.use(http.put('/api/presets/my-style', () => HttpResponse.json({error: {code: 'config.invalid', message: 'invalid config', details: {errors: [{loc: 'checkpoint.save_dtype', msg: 'unsupported precision'}]}}}, {status: 400})));
    fireEvent.change(epochs, {target: {value: '12'}});
    fireEvent.click(screen.getByRole('button', {name: '保存预设'}));
    fireEvent.click(await screen.findByRole('button', {name: '定位 权重保存精度'}));
    await waitFor(() => expect(screen.getByTestId('field-checkpoint.save_dtype')).toBeInTheDocument());
    await waitFor(() => expect(within(screen.getByTestId('field-checkpoint.save_dtype')).getByRole('combobox')).toHaveFocus());
    fireEvent.click(screen.getByRole('button', { name: /设备与时长$/ }));
    expect(screen.getByRole('spinbutton', {name: 'loop.epochs'})).toHaveValue(12);
  });

  it('opens the most recently updated user preset and keeps the library inside a searchable picker', async () => {
    const state = show(); const epochs = await ready();
    expect(epochs).toHaveValue(4);
    fireEvent.click(screen.getByRole('button', { name: '编辑用途与说明' }));
    expect(screen.getByRole('textbox', { name: '用途与说明' })).toHaveValue('自定义风格');
    expect(screen.getByRole('combobox', { name: '适用模型' })).toHaveTextContent('Anima');
    expect(screen.queryByRole('complementary', { name: '预设列表' })).not.toBeInTheDocument();
    expect(screen.queryByRole('combobox', { name: '按模型筛选预设' })).not.toBeInTheDocument();
    expect(screen.queryByRole('searchbox', { name: '选择预设 · 搜索' })).not.toBeInTheDocument();
    expect(screen.queryByRole('option')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('combobox', { name: '选择预设' }));
    expect(screen.queryByRole('option', { name: /builtin-anima/ })).not.toBeInTheDocument();
    const search = screen.getByRole('searchbox', { name: '选择预设 · 搜索' });
    fireEvent.change(search, { target: { value: '没有此名称' } });
    expect(screen.queryByRole('option')).not.toBeInTheDocument();
    expect(screen.getByText('没有匹配的选项')).toBeInTheDocument();
    fireEvent.change(search, { target: { value: 'Krea' } });
    expect(screen.getAllByRole('option')).toHaveLength(1);
    fireEvent.keyDown(search, { key: 'Enter' });
    await ready('krea-base');
    expect(screen.getByRole('spinbutton', { name: 'loop.epochs' })).toHaveValue(3);
    expect(screen.queryByRole('searchbox', { name: '选择预设 · 搜索' })).not.toBeInTheDocument();
    expect(state.writes).toEqual([]);
  });

  it.each([['empty', []], ['legacy built-ins only', [legacyBuiltin]]] as const)('starts %s libraries as a clean unnamed form without saving or blocking navigation', async (_label, seed) => {
    const state = show([...seed]); await ready('');
    fireEvent.click(screen.getByRole('button', { name: '编辑用途与说明' }));
    expect(screen.getByRole('textbox', { name: '用途与说明' })).toHaveValue('');
    expect(screen.getByRole('combobox', { name: '适用模型' })).toHaveTextContent('Anima');
    expect(screen.getByRole('spinbutton', { name: 'loop.epochs' })).toBeEnabled();
    expect(screen.getByRole('button', { name: '保存预设' })).toBeDisabled();
    expect(screen.queryByText('builtin-anima')).not.toBeInTheDocument();
    expect(state.writes).toEqual([]);
    await act(async () => { await state.router.navigate('/projects'); });
    await screen.findByText('项目列表页');
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(state.writes).toEqual([]);
  });

  it('keeps description edits when collapsed and saves them with the parameter draft', async () => {
    const state = show(); await ready();
    const toggle = screen.getByRole('button', { name: '编辑用途与说明' });
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByRole('textbox', { name: '用途与说明' })).not.toBeInTheDocument();
    fireEvent.click(toggle);
    const description = screen.getByRole('textbox', { name: '用途与说明' });
    await waitFor(() => expect(description).toHaveFocus());
    fireEvent.change(description, { target: { value: '更适合细节训练' } });
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByRole('textbox', { name: '用途与说明' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '保存预设' }));
    await screen.findByText('预设已保存，可在项目训练参数中加载。');
    expect(state.writes[0].body.description).toBe('更适合细节训练');
    fireEvent.click(toggle);
    expect(screen.getByRole('textbox', { name: '用途与说明' })).toHaveValue('更适合细节训练');
  });

  it('duplicates a user preset into editable fields without changing its source or retaining project files', async () => {
    const state = show(); await ready();
    fireEvent.click(screen.getByRole('button', { name: '复制为新预设' }));
    const epochs = await ready('my-style-copy'); expect(epochs).toBeEnabled();
    fireEvent.change(epochs, { target: { value: '7' } });
    fireEvent.click(screen.getByRole('button', { name: '保存预设' }));
    await screen.findByText('预设已保存，可在项目训练参数中加载。');
    expect(state.writes).toHaveLength(1); expect(state.writes[0].method).toBe('POST');
    const body = state.writes[0].body;
    expect(body.config.loop.epochs).toBe(7); expect(body.config.model.family).toBe('anima');
    for (const [group, key] of [['model', 'dit_path'], ['model', 'tokenizer_path'], ['dataset', 'sources'], ['dataset', 'cache_dir'], ['sampling', 'output_dir'], ['sampling', 'prompts_file'], ['adapter', 'resume_weights']]) expect(body.config[group]).not.toHaveProperty(key);
    expect(state.rows.find(row => row.name === 'my-style')?.config.loop).toEqual({ epochs: 4 });
    fireEvent.click(screen.getByRole('button', { name: /模型与训练方式$/ }));
    expect(screen.queryByTestId('field-model.dit_path')).not.toBeInTheDocument();
    expect(screen.queryByTestId('field-checkpoint.resume')).not.toBeInTheDocument();
  });

  it('creates a Krea preset from family defaults while retaining the entered name', async () => {
    const state = show(); await ready();
    fireEvent.click(screen.getByRole('button', { name: '新建预设' })); await ready('');
    fireEvent.change(screen.getByRole('textbox', { name: '预设名称' }), { target: { value: '新的_Krea参数' } });
    fireEvent.click(screen.getByRole('combobox', { name: '适用模型' }));
    fireEvent.click(screen.getByRole('option', { name: /Krea/ }));
    await waitFor(() => expect(screen.getByRole('combobox', { name: '适用模型' })).toHaveTextContent('Krea'));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: '预设名称' })).toHaveValue('新的_Krea参数');
    fireEvent.click(screen.getByRole('button', { name: '保存预设' }));
    await screen.findByText('预设已保存，可在项目训练参数中加载。');
    expect(state.requestedFamilies).toContain('krea2');
    expect(state.writes[0].body.config).toMatchObject({ model: { family: 'krea2' }, sampling: { steps: 28, cfg: 5.5 }, dataset: { text_encoding: 'cached' } });
  });

  it('retains failed edits and field errors, then explicitly updates the existing preset', async () => {
    const state = show(); const epochs = await ready();
    let fail = true;
    server.use(http.put('/api/presets/my-style', async ({ request }) => {
      const body = await request.json() as any;
      if (fail) return HttpResponse.json({ error: { code: 'config.invalid', message: 'invalid config', details: { errors: [{ loc: 'loop.epochs', msg: 'must be positive' }] } } }, { status: 400 });
      state.writes.push({ method: 'PUT', body, name: 'my-style' });
      return HttpResponse.json({ ...body, builtin: false, updated_at: 30 });
    }));
    fireEvent.change(epochs, { target: { value: '8' } });
    fireEvent.click(screen.getByRole('button', { name: '编辑用途与说明' }));
    fireEvent.change(screen.getByRole('textbox', { name: '用途与说明' }), { target: { value: '保留我的描述' } });
    fireEvent.click(screen.getByRole('button', { name: '保存预设' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('loop.epochs: must be positive');
    expect(epochs).toHaveValue(8); expect(screen.getByRole('textbox', { name: '用途与说明' })).toHaveValue('保留我的描述');
    fail = false; fireEvent.click(screen.getByRole('button', { name: '保存预设' }));
    await screen.findByText('预设已保存，可在项目训练参数中加载。');
    expect(state.writes).toEqual([{ method: 'PUT', name: 'my-style', body: expect.objectContaining({ description: '保留我的描述', config: expect.objectContaining({ loop: expect.objectContaining({ epochs: 8 }) }) }) }]);
  });

  it('waits for saving before switching presets and lets Keep editing cancel a switch', async () => {
    const state = show(); const epochs = await ready(); fireEvent.change(epochs, { target: { value: '9' } });
    await choose('krea-base');
    let dialog = await screen.findByRole('dialog', { name: '保存预设修改？' });
    fireEvent.click(within(dialog).getByRole('button', { name: '继续编辑' }));
    expect(epochs).toHaveValue(9); expect(state.writes).toEqual([]); await ready();
    let release!: () => void;
    const pending = new Promise<void>(resolve => { release = resolve; });
    server.use(http.put('/api/presets/my-style', async ({ request }) => {
      const body = await request.json() as any; state.writes.push({ method: 'PUT', name: 'my-style', body });
      await pending; return HttpResponse.json({ ...body, builtin: false, updated_at: 30 });
    }));
    await choose('krea-base'); dialog = await screen.findByRole('dialog', { name: '保存预设修改？' });
    fireEvent.click(within(dialog).getByRole('button', { name: '保存并继续' }));
    await waitFor(() => expect(state.writes).toHaveLength(1));
    expect(screen.getByRole('textbox', { name: '预设名称' })).toHaveValue('my-style'); expect(epochs).toHaveValue(9);
    await act(async () => { release(); });
    await ready('krea-base'); expect(screen.getByRole('spinbutton', { name: 'loop.epochs' })).toHaveValue(3);
    expect(state.writes[0].body.config.loop.epochs).toBe(9);
  });

  it('blocks navigation on a save failure, preserves the draft and permits retry', async () => {
    const state = show(); const epochs = await ready(); fireEvent.change(epochs, { target: { value: '9' } });
    let fail = true;
    server.use(http.put('/api/presets/my-style', async ({ request }) => fail
      ? HttpResponse.json({ error: { message: 'Disk full', code: 'storage.error' } }, { status: 500 })
      : HttpResponse.json({ ...await request.json() as any, builtin: false, updated_at: 30 })));
    await act(async () => { await state.router.navigate('/projects'); });
    const dialog = await screen.findByRole('dialog', { name: '保存预设修改？' });
    fireEvent.click(within(dialog).getByRole('button', { name: '保存并继续' }));
    expect(await within(dialog).findByRole('alert')).toHaveTextContent('Disk full');
    expect(state.router.state.location.pathname).toBe('/presets'); expect(epochs).toHaveValue(9);
    fail = false; fireEvent.click(within(dialog).getByRole('button', { name: '保存并继续' }));
    await screen.findByText('项目列表页'); expect(state.router.state.location.pathname).toBe('/projects');
  });

  it('does not overwrite an edited draft when loading defaults for a different preset fails', async () => {
    const state = show(); const epochs = await ready(); fireEvent.change(epochs, { target: { value: '11' } });
    await choose('krea-base');
    server.use(http.get('/api/config/defaults', () => HttpResponse.json({ error: { code: 'read.failed', message: 'defaults unavailable' } }, { status: 500 })));
    fireEvent.click(screen.getByRole('button', { name: '放弃修改' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('defaults unavailable');
    expect(epochs).toHaveValue(11); expect(screen.getByRole('textbox', { name: '预设名称' })).toHaveValue('my-style');
    expect(state.writes).toEqual([]);
  });

  it('keeps the current draft visible when refreshing the library fails', async () => {
    const state = show(); const epochs = await ready(); fireEvent.change(epochs, { target: { value: '12' } });
    server.use(http.get('/api/presets', () => HttpResponse.json({ error: { code: 'list.failed', message: 'preset list unavailable' } }, { status: 500 })));
    await act(async () => { await state.client.invalidateQueries(); });
    expect(await screen.findByRole('alert')).toHaveTextContent('preset list unavailable');
    expect(screen.getByRole('textbox', { name: '预设名称' })).toHaveValue('my-style'); expect(epochs).toHaveValue(12);
    expect(state.writes).toEqual([]);
  });

  it('confirms deletion and opens the most recent remaining user preset only after success', async () => {
    const state = show(); await ready(); fireEvent.click(screen.getByRole('button', { name: '删除预设' }));
    expect(state.writes).toHaveLength(0); fireEvent.click(screen.getByRole('button', { name: '取消' })); await ready();
    fireEvent.click(screen.getByRole('button', { name: '删除预设' }));
    fireEvent.click(screen.getByRole('button', { name: '确认删除' })); await ready('krea-base');
    expect(state.writes).toEqual([{ method: 'DELETE', name: 'my-style', body: null }]);
    fireEvent.click(screen.getByRole('combobox', { name: '选择预设' }));
    expect(screen.queryByRole('option', { name: /my-style|builtin-anima/ })).not.toBeInTheDocument();
    expect(screen.getByRole('option', { name: /^krea-base · / })).toBeInTheDocument();
  });

  it('returns to a clean new form after deleting the last user preset, without resurrecting built-ins', async () => {
    const state = show(presetRows().filter(row => row.name !== 'krea-base')); await ready();
    fireEvent.click(screen.getByRole('button', { name: '删除预设' }));
    fireEvent.click(screen.getByRole('button', { name: '确认删除' })); await ready('');
    expect(screen.getByRole('button', { name: '保存预设' })).toBeDisabled();
    expect(screen.queryByText('builtin-anima')).not.toBeInTheDocument();
    await act(async () => { await state.router.navigate('/projects'); });
    await screen.findByText('项目列表页'); expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(state.writes).toEqual([{ method: 'DELETE', name: 'my-style', body: null }]);
  });
});
