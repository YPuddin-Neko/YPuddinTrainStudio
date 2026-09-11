import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import Models from '../src/pages/Models/Models';
import SettingsPage from '../src/pages/Settings/Settings';
import { ModelAsset, ModelDownload, Settings } from '../src/api/types';
import i18n from '../src/i18n';
import { handlers } from '../src/mocks/handlers';

const server = setupServer(...handlers);
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => { cleanup(); server.resetHandlers(); vi.restoreAllMocks(); });
afterAll(() => server.close());

let models: ModelAsset[];
let downloads: ModelDownload[];
let settings: Settings;
let patch = vi.fn();
let download = vi.fn();
let cancel = vi.fn();

beforeEach(async () => {
  await i18n.changeLanguage('zh-CN');
  models = [
    { id: 'a', family: 'anima', kind: 'dit', path: 'C:\\models\\anima.safetensors', dtype: 'bf16', exists: true, is_default: false, size: 1024, created_at: 1 },
    { id: 'k', family: 'krea2', kind: 'dit', path: 'C:\\models\\krea2.safetensors', dtype: 'bf16', exists: true, is_default: true, size: 1024, created_at: 1 },
  ];
  downloads = [];
  settings = { paths: { data_root: 'C:\\studio', models_dir: 'D:\\models', cache_dir: 'D:\\cache', output_dir: 'D:\\runs' }, server: { host: '127.0.0.1', port: 8765 }, ui: { language: 'zh-CN', theme: 'light' } };
  patch = vi.fn(); download = vi.fn(); cancel = vi.fn();
  server.use(
    http.get('/api/families', () => HttpResponse.json(['anima', 'krea2'].map(name => ({ name, label: name === 'anima' ? 'Anima' : 'Krea 2', weights: [{ field: 'dit_path' }, { field: 'text_encoder_path' }, { field: 'vae_path' }] })))),
    http.get('/api/models', () => HttpResponse.json(models)),
    http.get('/api/settings', () => HttpResponse.json(settings)),
    http.put('/api/settings', async ({ request }) => { settings = await request.json() as Settings; return HttpResponse.json(settings); }),
    http.get('/api/models/downloads', () => HttpResponse.json(downloads)),
    http.patch('/api/models/:id', async ({ params, request }) => {
      const body = await request.json() as { is_default: boolean };
      patch(params.id, body);
      const row = models.find(m => m.id === params.id)!;
      models = models.map(m => m.family === row.family && m.kind === row.kind ? { ...m, is_default: m.id === row.id && body.is_default } : m);
      return HttpResponse.json(models.find(m => m.id === params.id));
    }),
    http.post('/api/models/downloads', async ({ request }) => {
      const body = await request.json() as { url: string; family: string; kind: string };
      download(body);
      downloads = [{ id: 'dl1', family: body.family, kind: body.kind, source_url: body.url, filename: 'encoder.safetensors', target_path: 'D:\\models\\encoder.safetensors', status: 'downloading', downloaded_bytes: 500, total_bytes: 1000, error: null, model_id: null, dtype: 'bf16', is_default: true, created_at: 1, finished_at: null }];
      return HttpResponse.json(downloads[0], { status: 202 });
    }),
    http.post('/api/models/downloads/:id/cancel', ({ params }) => { cancel(params.id); downloads = downloads.map(d => ({ ...d, status: 'cancelled' })); return HttpResponse.json(downloads[0]); }),
  );
});

function mount(element: React.ReactNode, path = '/models?family=anima') {
  return render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><MemoryRouter initialEntries={[path]}>{element}</MemoryRouter></QueryClientProvider>);
}

describe('real model management UI contracts', () => {
  it('sets an existing component default for the selected family', async () => {
    mount(<Models />);
    const card = await screen.findByTestId('model-component-dit');
    await waitFor(() => expect(within(card).getByRole('option', { name: 'anima.safetensors' })).toBeInTheDocument());
    expect(within(card).queryByRole('option', { name: 'krea2.safetensors' })).not.toBeInTheDocument();
    fireEvent.change(within(card).getByRole('combobox'), { target: { value: 'a' } });
    await waitFor(() => expect(patch).toHaveBeenCalledWith('a', { is_default: true }));
    await waitFor(() => expect(within(card).getByRole('combobox')).toHaveValue('a'));
    expect(models.find(m => m.id === 'k')?.is_default).toBe(true);
  });

  it('starts a download from a component source and cancels with visible progress', async () => {
    mount(<Models />);
    const card = await screen.findByTestId('model-component-text_encoder');
    fireEvent.click(within(card).getByRole('button', { name: i18n.t('models.getComponent') }));
    expect((screen.getByTestId('model-download-url') as HTMLInputElement).value).toContain('qwen_3_06b_base.safetensors');
    fireEvent.click(screen.getByTestId('model-download-start'));
    await waitFor(() => expect(download).toHaveBeenCalledWith(expect.objectContaining({ family: 'anima', kind: 'text_encoder', is_default: true })));
    const list = await screen.findByTestId('model-downloads');
    expect(within(list).getByRole('progressbar')).toHaveAttribute('value', '500');
    expect(within(list).getByRole('progressbar')).toHaveAttribute('max', '1000');
    fireEvent.click(within(list).getByRole('button', { name: i18n.t('common.cancel') }));
    await waitFor(() => expect(cancel).toHaveBeenCalledWith('dl1'));
    await waitFor(() => expect(within(list).queryByRole('progressbar')).not.toBeInTheDocument());
    expect(within(list).getByText(/已取消/)).toBeInTheDocument();
    expect(models.some(m => m.kind === 'text_encoder')).toBe(false);
  });

  it('settings changes model defaults immediately and broadcasts saved appearance', async () => {
    mount(<SettingsPage />, '/settings');
    const page = await screen.findByTestId('settings-page');
    fireEvent.change(within(page).getByLabelText(i18n.t('models.kind_dit')), { target: { value: 'a' } });
    await waitFor(() => expect(patch).toHaveBeenCalledWith('a', { is_default: true }));
    const changed = vi.fn();
    window.addEventListener('studio.settings.changed', changed);
    fireEvent.change(screen.getByTestId('settings-theme'), { target: { value: 'dark' } });
    fireEvent.click(screen.getByTestId('settings-save-btn'));
    await waitFor(() => expect(changed).toHaveBeenCalled());
    expect(document.documentElement.classList.contains('dark')).toBe(true);
    expect(settings.ui.theme).toBe('dark');
    window.removeEventListener('studio.settings.changed', changed);
    document.documentElement.classList.remove('dark');
  });

  it('switching component updates a standard URL and preserves an explicit custom source', async () => {
    mount(<Models />);
    fireEvent.click(await screen.findByTestId('download-model-btn'));
    const form = screen.getByTestId('download-model-form');
    const kind = within(form).getAllByRole('combobox')[0];
    fireEvent.change(kind, { target: { value: 'vae' } });
    expect((screen.getByTestId('model-download-url') as HTMLInputElement).value).toContain('qwen_image_vae.safetensors');
    const custom = 'https://huggingface.co/my/repo/blob/main/custom.safetensors';
    fireEvent.change(screen.getByTestId('model-download-url'), { target: { value: custom } });
    fireEvent.change(kind, { target: { value: 'text_encoder' } });
    expect(screen.getByTestId('model-download-url')).toHaveValue(custom);
    fireEvent.click(within(form).getByRole('button', { name: i18n.t('models.fillSuggestedSource', '填入当前组件的标准来源') }));
    expect((screen.getByTestId('model-download-url') as HTMLInputElement).value).toContain('qwen_3_06b_base.safetensors');
  });

  it('shows actionable server validation details when download creation fails', async () => {
    server.use(http.post('/api/models/downloads', () => HttpResponse.json({ error: { code: 'validation', message: 'request validation failed', details: { errors: [{ loc: ['body', 'url'], msg: 'Only a Hugging Face model file URL is accepted' }] } } }, { status: 422 })));
    mount(<Models />);
    fireEvent.click(await screen.findByTestId('download-model-btn'));
    fireEvent.click(screen.getByTestId('model-download-start'));
    expect(await screen.findByText(/Only a Hugging Face model file URL is accepted/)).toBeInTheDocument();
    expect(screen.getByTestId('download-model-form')).toBeInTheDocument();
  });
});
