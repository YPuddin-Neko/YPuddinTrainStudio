import { act, cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import Layout from '../src/components/Layout';
import type { Settings } from '../src/api/types';
import i18n from '../src/i18n';

vi.mock('../src/events/useEventStream', () => ({ useEventStream: () => {}, useEventStreamStatus: () => 'connected' }));
const server = setupServer();
const initialSettings = (): Settings => ({ paths: { data_root: 'D:/studio', cache_dir: 'D:/cache', models_dir: 'D:/models', output_dir: 'D:/runs', output_mode: 'project' }, server: { host: '127.0.0.1', port: 8765 }, ui: { theme: 'system', language: 'zh-CN' } });
let settings: Settings;
let writes: unknown[];
let mediaChanged: ((event: MediaQueryListEvent) => void) | undefined;
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterAll(() => server.close());
beforeEach(async () => {
  await i18n.changeLanguage('zh-CN');
  settings = initialSettings(); writes = []; mediaChanged = undefined;
  vi.stubGlobal('matchMedia', () => ({ matches: false, media: '(prefers-color-scheme: dark)', addEventListener: (_: string, callback: typeof mediaChanged) => { mediaChanged = callback; }, removeEventListener: () => {} }));
  server.use(
    http.get('/api/settings', () => HttpResponse.json(settings)),
    http.get('/api/system/stats', () => HttpResponse.json({ cpu_pct: 0, ram: { used_mb: 1000, total_mb: 8000 }, disks: [], gpus: [] })),
    http.get('/api/jobs', () => HttpResponse.json({ items: [] })),
    http.get('/api/system/info', () => HttpResponse.json({ ypuddin: '99.88.77-server' })),
    http.put('/api/settings', async ({ request }) => {
      const body = await request.json() as { ui: Partial<Settings['ui']> };
      writes.push(body); settings = { ...settings, ui: { ...settings.ui, ...body.ui } };
      return HttpResponse.json(settings);
    }),
  );
});
afterEach(() => { cleanup(); server.resetHandlers(); vi.restoreAllMocks(); vi.unstubAllGlobals(); document.documentElement.classList.remove('dark'); });
function Page() { const location = useLocation(); return <output data-testid="route">{location.pathname}{location.search}</output>; }
function show() {
  render(<MemoryRouter initialEntries={['/projects/p1/train?version=v1']}><Routes><Route element={<Layout />}><Route path="*" element={<Page />} /></Route></Routes></MemoryRouter>);
}
const sidebar = () => within(screen.getByRole('complementary'));

describe('sidebar appearance preferences', () => {
  it('offers system/light/dark plus the current language in the footer without a sidebar version or empty feedback slot', async () => {
    const user = userEvent.setup();
    show();
    await waitFor(() => expect(screen.getByLabelText('CPU 占用率')).toHaveTextContent('0%'));
    const footer = screen.getByRole('complementary').querySelector('.sidebar-preferences');
    const theme = sidebar().getByRole('combobox', { name: '主题' });
    expect(footer).toContainElement(theme);
    expect(footer).toContainElement(sidebar().getByRole('combobox', { name: '语言' }));
    expect(theme).toHaveTextContent('自动');
    expect(sidebar().getByRole('combobox', { name: '语言' })).toHaveTextContent('中文');
    await user.click(theme);
    expect(screen.getAllByRole('option').map(option => option.textContent)).toEqual(['自动', '浅色', '深色']);
    expect(sidebar().queryByText(/99\.88\.77|服务版本|Server version/)).not.toBeInTheDocument();
    expect(screen.getByTestId('app-topbar').querySelector('.topbar-feedback-slot')).toBeNull();
    expect(screen.getByTestId('app-topbar').querySelector('.topbar-job-slot')).toBeNull();
    expect(writes).toEqual([]);
  });

  it('keeps the previous selection during saving then applies the server response without navigation', async () => {
    const user = userEvent.setup();
    let finish = () => {};
    const waitForSave = new Promise<void>(resolve => { finish = resolve; });
    server.use(http.put('/api/settings', async ({ request }) => {
      const body = await request.json() as { ui: Partial<Settings['ui']> }; writes.push(body);
      await waitForSave;
      settings = { ...settings, ui: { ...settings.ui, ...body.ui } };
      return HttpResponse.json(settings);
    }));
    show();
    const theme = await sidebar().findByRole('combobox', { name: '主题' });
    await user.click(theme);
    await user.click(screen.getByRole('option', { name: '深色' }));
    await waitFor(() => expect(writes).toEqual([{ ui: { theme: 'dark' } }]));
    expect(theme).toHaveTextContent('自动');
    expect(theme).toBeDisabled();
    expect(sidebar().getByRole('combobox', { name: '语言' })).toBeDisabled();
    expect(document.documentElement).not.toHaveClass('dark');
    await act(async () => finish());
    await waitFor(() => expect(theme).toHaveTextContent('深色'));
    expect(theme).toBeEnabled();
    expect(document.documentElement).toHaveClass('dark');
    expect(screen.getByTestId('route')).toHaveTextContent('/projects/p1/train?version=v1');
  });

  it('keeps the saved theme and reports a failed save instead of a false success', async () => {
    settings.ui.theme = 'light';
    server.use(http.put('/api/settings', () => HttpResponse.json({ error: { code: 'settings.write', message: 'Unable to save preferences' } }, { status: 503 })));
    const user = userEvent.setup(); show();
    const theme = await sidebar().findByRole('combobox', { name: '主题' });
    await waitFor(() => expect(theme).toHaveTextContent('浅色'));
    await user.click(theme); await user.click(screen.getByRole('option', { name: '深色' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Unable to save preferences');
    await waitFor(() => expect(theme).toBeEnabled());
    expect(theme).toHaveTextContent('浅色');
    expect(document.documentElement).not.toHaveClass('dark');
    expect(screen.getByTestId('route')).toHaveTextContent('/projects/p1/train?version=v1');
  });

  it('persists the selected language, updates its label and keeps the same route', async () => {
    const user = userEvent.setup(); show();
    await user.click(await sidebar().findByRole('combobox', { name: '语言' }));
    await user.click(screen.getByRole('option', { name: 'EN' }));
    await waitFor(() => expect(i18n.resolvedLanguage).toBe('en'));
    expect(writes).toEqual([{ ui: { language: 'en' } }]);
    expect(sidebar().getByRole('combobox', { name: 'Language' })).toHaveTextContent('EN');
    expect(document.documentElement).toHaveAttribute('lang', 'en');
    expect(screen.getByTestId('route')).toHaveTextContent('/projects/p1/train?version=v1');
  });

  it('preserves the current language after a failed language save', async () => {
    server.use(http.put('/api/settings', () => HttpResponse.json({ error: { code: 'settings.write', message: 'Language save failed' } }, { status: 503 })));
    const user = userEvent.setup(); show();
    await user.click(await sidebar().findByRole('combobox', { name: '语言' }));
    await user.click(screen.getByRole('option', { name: 'EN' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Language save failed');
    expect(i18n.resolvedLanguage).toBe('zh-CN');
    expect(sidebar().getByRole('combobox', { name: '语言' })).toHaveTextContent('中文');
    expect(screen.getByTestId('route')).toHaveTextContent('/projects/p1/train?version=v1');
  });

  it('follows system theme changes in auto mode without saving or navigating', async () => {
    show();
    await waitFor(() => expect(mediaChanged).toBeTypeOf('function'));
    act(() => mediaChanged?.({ matches: true } as MediaQueryListEvent));
    expect(document.documentElement).toHaveClass('dark');
    expect(sidebar().getByRole('combobox', { name: '主题' })).toHaveTextContent('自动');
    act(() => mediaChanged?.({ matches: false } as MediaQueryListEvent));
    expect(document.documentElement).not.toHaveClass('dark');
    expect(writes).toEqual([]);
    expect(screen.getByTestId('route')).toHaveTextContent('/projects/p1/train?version=v1');
  });
});
